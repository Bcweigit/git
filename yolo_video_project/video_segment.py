"""使用 YOLO 分割权重进行视频逐帧识别并保存多边形结果。

示例：
    python video_segment.py --source "/path/to/video.mp4" --show
    python video_segment.py --source "/path/to/video.mp4" --conf 0.35 --device 0
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_PYTHON = Path("/home/bcw/miniconda3/envs/yolo/bin/python")
MISSING_RUNTIME_MODULES = [
    name for name in ("cv2", "ultralytics", "torch")
    if importlib.util.find_spec(name) is None
]
try:
    import cv2
except (ImportError, OSError) as exc:
    MISSING_RUNTIME_MODULES.append(f"cv2 ({exc})")
if MISSING_RUNTIME_MODULES:
    current_python = Path(sys.executable).resolve()
    if PROJECT_PYTHON.is_file() and current_python != PROJECT_PYTHON.resolve():
        print(
            f"当前 Python 环境缺少 {', '.join(MISSING_RUNTIME_MODULES)}，"
            f"自动切换到项目 yolo 环境：{PROJECT_PYTHON}",
            flush=True,
        )
        entry_script = Path(sys.argv[0]).expanduser().resolve()
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), str(entry_script), *sys.argv[1:]])
    raise SystemExit(
        "运行环境缺少依赖：" + ", ".join(MISSING_RUNTIME_MODULES)
        + "。请使用项目的 yolo Conda 环境运行。"
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "latest_model.txt"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_model(value: str) -> Path:
    """解析命令行权重路径；未传路径时读取最近一次训练结果。"""
    if value:
        model_path = Path(value).expanduser().resolve()
    elif DEFAULT_MODEL.is_file():
        model_path = Path(DEFAULT_MODEL.read_text(encoding="utf-8").strip())
        if not model_path.is_absolute():
            model_path = (ROOT / model_path).resolve()
        else:
            model_path = model_path.expanduser().resolve()
    else:
        model_path = (
            ROOT
            / "runs"
            / "segment"
            / "segment_20261027_205536_165577"
            / "weights"
            / "best.pt"
        )
    if not model_path.is_file():
        raise ValueError(f"找不到分割模型权重：{model_path}")
    return model_path


def resolve_device(value: str) -> str:
    if value != "auto":
        if value != "cpu" and value.isdigit():
            import torch

            if not torch.cuda.is_available() or int(value) >= torch.cuda.device_count():
                raise ValueError(
                    f"当前环境不可用 CUDA 设备 {value}。"
                    "请去掉 --device 参数使用自动选择，或明确使用 --device cpu。"
                )
        return value
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def output_dir_for(source: Path, requested: str) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    return ROOT / "output" / f"{source.stem}_segment_{timestamp()}"


def choose_video() -> str:
    """没有传入路径时弹出本地视频选择窗口。"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilename(
            title="选择要进行分割检测的视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.m4v *.webm"),
                ("所有文件", "*.*"),
            ],
        )
        root.destroy()
        return selected
    except Exception as exc:
        print(f"无法打开视频选择窗口：{exc}", file=sys.stderr)
        return input("请输入视频文件路径：").strip().strip("\"'")


def make_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    width, height = size
    # mp4v 在当前项目的 OpenCV 环境中兼容性较好；输出不包含原视频音频。
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{output_path}")
    return writer


def simplify_polygon(polygon, epsilon):
    """减少 mask 轮廓中的阶梯点，让边缘更接近直线。"""
    points = polygon.astype("float32").reshape((-1, 1, 2))
    if epsilon <= 0 or len(points) < 4:
        return polygon
    simplified = cv2.approxPolyDP(points, epsilon, closed=True)
    return simplified.reshape((-1, 2)) if len(simplified) >= 3 else polygon


def write_detection_rows(table, frame_index, source_time, result, simplify=0.0):
    """把分割结果写成 CSV，每个实例一行，polygon_points 保存为 JSON。"""
    if result.boxes is None or result.masks is None:
        return 0

    boxes = result.boxes.data.cpu().tolist()
    polygons = result.masks.xy
    count = min(len(boxes), len(polygons))
    for index in range(count):
        x1, y1, x2, y2, score, class_id = boxes[index][:6]
        class_id = int(class_id)
        polygon_points = simplify_polygon(polygons[index], simplify)
        polygon = [[round(float(x), 2), round(float(y), 2)] for x, y in polygon_points]
        class_name = result.names.get(class_id, str(class_id))
        table.writerow(
            [
                frame_index,
                round(source_time, 3),
                class_id,
                class_name,
                round(float(score), 5),
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2),
                json.dumps(polygon, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    return count


def draw_segmentation_contours(frame, result, line_width=3, fill=False, simplify=0.0):
    """只绘制实例 mask 的真实轮廓，不绘制水平框或旋转框。"""
    annotated = frame.copy()
    if result.masks is None or result.boxes is None:
        return annotated

    polygons = result.masks.xy
    boxes = result.boxes
    names = result.names
    for index, polygon in enumerate(polygons):
        polygon = simplify_polygon(polygon, simplify)
        if len(polygon) < 3 or index >= len(boxes):
            continue
        class_id = int(boxes.cls[index].item())
        score = float(boxes.conf[index].item())
        color = (0, 220, 0) if class_id == 0 else (255, 120, 0)
        contour = polygon.astype("int32").reshape((-1, 1, 2))
        if fill:
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [contour], color)
            annotated = cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)
        cv2.polylines(annotated, [contour], isClosed=True, color=color, thickness=line_width)

        x, y = contour[:, 0, :].min(axis=0).tolist()
        label = f"{names.get(class_id, class_id)} {score:.2f}"
        cv2.putText(
            annotated,
            label,
            (max(0, int(x)), max(20, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def polygon_center_area(polygon):
    contour = polygon.astype("float32").reshape((-1, 1, 2))
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    else:
        center = tuple(contour[:, 0, :].mean(axis=0).tolist())
    return center, abs(float(cv2.contourArea(contour)))


def polygon_bbox(polygon):
    x, y, width, height = cv2.boundingRect(polygon.astype("float32"))
    return (float(x), float(y), float(x + width), float(y + height))


def bbox_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(area_first + area_second - intersection, 1e-6)


def update_stable_tracks(tracks, result, simplify, max_misses=3, shrink_ratio=0.55, match_iou=0.15):
    """只在空间重叠且疑似遮挡时复用历史轮廓，避免跨目标串线。"""
    detections = []
    if result.masks is not None and result.boxes is not None:
        for index, polygon in enumerate(result.masks.xy):
            polygon = simplify_polygon(polygon, simplify)
            if len(polygon) < 3 or index >= len(result.boxes):
                continue
            class_id = int(result.boxes.cls[index].item())
            score = float(result.boxes.conf[index].item())
            center, area = polygon_center_area(polygon)
            detections.append({"class_id": class_id, "score": score, "polygon": polygon,
                               "center": center, "area": area,
                               "bbox": polygon_bbox(polygon),
                               "class_name": result.names.get(class_id, str(class_id))})

    used = set()
    for track in tracks:
        candidates = [
            (index, detection)
            for index, detection in enumerate(detections)
            if index not in used and detection["class_id"] == track["class_id"]
            and bbox_iou(track["bbox"], detection["bbox"]) >= match_iou
        ]
        if candidates:
            index, detection = min(
                candidates,
                key=lambda item: (item[1]["center"][0] - track["center"][0]) ** 2
                + (item[1]["center"][1] - track["center"][1]) ** 2,
            )
            used.add(index)
            # 面积没有突然塌缩时，当前 mask 优先，允许目标正常移动。
            if detection["area"] >= track["area"] * shrink_ratio:
                track.update(detection)
            else:
                # 可能被手遮挡：保留原位置和原轮廓，绝不平移到新目标。
                track["score"] = max(track["score"], detection["score"])
            track["misses"] = 0
            continue
        track["misses"] += 1

    tracks[:] = [track for track in tracks if track["misses"] <= max_misses]
    for index, detection in enumerate(detections):
        if index not in used:
            tracks.append({**detection, "misses": 0})
    return tracks


def draw_stable_tracks(frame, tracks, names, line_width=3, fill=False):
    annotated = frame.copy()
    for track in tracks:
        polygon = track["polygon"].astype("int32").reshape((-1, 1, 2))
        if len(polygon) < 3:
            continue
        class_id = track["class_id"]
        color = (0, 220, 0) if class_id == 0 else (255, 120, 0)
        if fill:
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], color)
            annotated = cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)
        cv2.polylines(annotated, [polygon], True, color, line_width)
        x, y = polygon[:, 0, :].min(axis=0).tolist()
        label = f"{track.get('class_name', names.get(class_id, class_id))} {track['score']:.2f}"
        cv2.putText(annotated, label, (max(0, int(x)), max(20, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return annotated


def run(args) -> int:
    from ultralytics import YOLO

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"输入视频不存在：{source}")
    if not 0 < args.conf <= 1:
        raise ValueError("--conf 必须大于 0 且不超过 1。")
    if args.max_frames < 0:
        raise ValueError("--max-frames 不能为负数。")
    if args.infer_every < 1:
        raise ValueError("--infer-every 必须是正整数。")

    model_path = resolve_model(args.model)
    device = resolve_device(args.device)
    output_dir = output_dir_for(source, args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"输出目录已存在且非空：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"模型：{model_path}", flush=True)
    print(f"设备：{device}", flush=True)
    print(f"视频：{source}", flush=True)

    model = YOLO(str(model_path), task="segment")
    # Ultralytics 新版本用 quantize=16 表示 GPU FP16，避免旧 half 参数的弃用警告。
    precision = 16 if device != "cpu" and str(device).isdigit() else None
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    frame_count = 0
    detection_count = 0
    interrupted = False
    started = time.monotonic()
    failure = None
    has_prediction = False
    stable_tracks = []
    result_video = output_dir / "result.mp4"
    result_csv = output_dir / "detections.csv"

    try:
        with result_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
            table = csv.writer(csv_file)
            table.writerow(
                [
                    "frame",
                    "source_time_s",
                    "class_id",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "polygon_points",
                ]
            )
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                should_infer = (frame_count % args.infer_every == 0) or not has_prediction
                if should_infer:
                    result = model.predict(
                        frame,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        max_det=args.max_det,
                        device=device,
                        quantize=precision,
                        verbose=False,
                    )[0]
                    stable_tracks = update_stable_tracks(
                        stable_tracks, result, args.simplify,
                        max_misses=args.hold_frames,
                        shrink_ratio=args.occlusion_ratio,
                    )
                    annotated = draw_stable_tracks(
                        frame, stable_tracks, result.names,
                        line_width=args.line_width, fill=args.fill,
                    )
                    has_prediction = True
                    detection_count += write_detection_rows(
                        table, frame_count, (frame_count - 1) / fps, result, args.simplify
                    )
                else:
                    annotated = draw_stable_tracks(
                        frame, stable_tracks, {},
                        line_width=args.line_width, fill=args.fill,
                    )

                frame_count += 1
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = make_writer(result_video, fps, (width, height))
                writer.write(annotated)
                if frame_count == 1 or frame_count % 30 == 0:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    progress = f"/{total_frames}" if total_frames > 0 else ""
                    print(
                        f"处理 {frame_count}{progress} 帧；"
                        f"{frame_count / elapsed:.1f} FPS；实例 {detection_count}",
                        flush=True,
                    )

                if args.show:
                    display = annotated
                    height, width = display.shape[:2]
                    scale = min(1.0, 1280 / width, 800 / height)
                    if scale < 1:
                        display = cv2.resize(
                            display, (int(width * scale), int(height * scale))
                        )
                    cv2.imshow("YOLO Segmentation - Q / ESC to stop", display)
                    # 按视频帧率播放，检测结果会随每一帧实时刷新。
                    key = cv2.waitKey(max(1, round(1000 / fps))) & 0xFF
                    if key in (ord("q"), 27):
                        interrupted = True
                        break

                if args.max_frames and frame_count >= args.max_frames:
                    interrupted = total_frames == 0 or frame_count < total_frames
                    break
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        failure = str(exc)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        summary = {
            "source": str(source),
            "model": str(model_path),
            "frames": frame_count,
            "instances": detection_count,
            "fps": fps,
            "elapsed_s": round(time.monotonic() - started, 3),
            "stopped_early": interrupted,
            "status": "failed" if failure else ("stopped" if interrupted else "completed"),
            "error": failure,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"完成：处理 {frame_count} 帧，识别 {detection_count} 个实例。", flush=True)
    print(f"输出视频：{result_video}", flush=True)
    print(f"多边形结果：{result_csv}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="", help="输入视频路径；不填写时弹出文件选择窗口")
    parser.add_argument("--model", default="", help="分割权重 .pt；留空使用最近训练的 best.pt")
    parser.add_argument("--output", default="", help="输出目录；留空自动创建到 output/")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值，默认 0.35")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值，默认 0.45")
    parser.add_argument("--max-det", type=int, default=300, help="每帧最多实例数")
    parser.add_argument("--imgsz", type=int, default=512, help="推理尺寸，默认 512；640 更精确但更慢")
    parser.add_argument("--device", default="auto", help="auto、0 或 cpu")
    parser.add_argument("--show", action="store_true", default=True, help="实时显示；按 Q 或 Esc 停止（默认开启）")
    parser.add_argument("--no-show", dest="show", action="store_false", help="关闭实时画面，只保存结果视频")
    parser.add_argument("--show-boxes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--line-width", type=int, default=3, help="轮廓线宽，默认 3")
    parser.add_argument(
        "--simplify",
        type=float,
        default=3.0,
        help="轮廓简化像素阈值，越大越直但细节越少，默认 3.0",
    )
    parser.add_argument("--fill", action="store_true", help="同时半透明填充分割区域")
    parser.add_argument("--hold-frames", type=int, default=3, help="遮挡时保留历史轮廓帧数，默认 3")
    parser.add_argument("--occlusion-ratio", type=float, default=0.55,
                        help="检测轮廓面积小于历史轮廓的此比例时视为遮挡，默认 0.55")
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="每隔多少帧推理一次，默认 1；实时加速可设为 2",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧；0 表示完整视频")
    args = parser.parse_args()
    if not args.source:
        args.source = choose_video()
        if not args.source:
            print("未选择视频，已取消。", flush=True)
            return 0
    try:
        return run(args)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
