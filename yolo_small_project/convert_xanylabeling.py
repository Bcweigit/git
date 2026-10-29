#!/usr/bin/env python3
"""Convert X-AnyLabeling/LabelMe rectangle JSON annotations to YOLO detect data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_classes(path: Path) -> dict[str, int]:
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if not names or names == ["object"]:
        raise ValueError(f"请先在 {path} 中填写真实类别名称（每行一个类别）。")
    if len(set(names)) != len(names):
        raise ValueError("classes.txt 中有重复类别名称。")
    return {name: i for i, name in enumerate(names)}


def locate_image(json_path: Path, data: dict[str, Any], source: Path) -> Path:
    candidates: list[Path] = []
    image_path = data.get("imagePath")
    if image_path:
        p = Path(str(image_path))
        candidates.extend([json_path.parent / p, source / p, p])
    candidates.extend(json_path.parent / (json_path.stem + ext) for ext in IMAGE_EXTS)
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
            return candidate.resolve()
    raise FileNotFoundError(f"{json_path.name} 对应的图片未找到。")


def convert_shape(shape: dict[str, Any], class_ids: dict[str, int], width: int, height: int) -> str | None:
    label = str(shape.get("label", "")).strip()
    if label not in class_ids:
        raise ValueError(f"标注类别 {label!r} 不在 classes.txt 中。")
    points = shape.get("points") or []
    if len(points) < 2:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    xc = (x1 + x2) / (2 * width)
    yc = (y1 + y2) / (2 * height)
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_ids[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path("images/manual50"),
                        help="包含图片和同名 X-AnyLabeling JSON 的目录")
    parser.add_argument("--classes", type=Path, default=Path("classes.txt"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="覆盖输出目录中已有的 train/val 内容")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"图片目录不存在：{source}")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio 必须在 [0, 1) 范围内")

    try:
        class_ids = read_classes(args.classes.resolve())
        json_files = sorted(source.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"{source} 中没有 .json 标注文件。")

        items: list[tuple[Path, list[str]]] = []
        for json_path in json_files:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
            image = locate_image(json_path, data, source)
            width, height = int(data.get("imageWidth") or 0), int(data.get("imageHeight") or 0)
            if width <= 0 or height <= 0:
                try:
                    from PIL import Image  # optional; fallback to cv2 below
                    with Image.open(image) as im:
                        width, height = im.size
                except Exception:
                    try:
                        import cv2
                        im = cv2.imread(str(image))
                        if im is None:
                            raise ValueError("OpenCV 无法读取图片")
                        height, width = im.shape[:2]
                    except Exception as exc:
                        raise ValueError(f"无法读取 {image.name} 的图片尺寸：{exc}") from exc
            labels = [line for shape in data.get("shapes", [])
                      if (line := convert_shape(shape, class_ids, width, height))]
            items.append((image, labels))

        output = args.output.resolve()
        target_dirs = [output / "images" / split for split in ("train", "val")] + [
            output / "labels" / split for split in ("train", "val")
        ]
        if not args.force and any(path.exists() and any(path.iterdir()) for path in target_dirs):
            raise FileExistsError(f"输出目录已有文件：{output}；确认覆盖后加 --force。")
        for path in target_dirs:
            path.mkdir(parents=True, exist_ok=True)

        order = list(range(len(items)))
        random.Random(args.seed).shuffle(order)
        val_count = round(len(items) * args.val_ratio) if len(items) >= 5 else 0
        val_indices = set(order[:val_count])
        for index, (image, labels) in enumerate(items):
            split = "val" if index in val_indices else "train"
            destination_image = output / "images" / split / image.name
            if destination_image.exists() and destination_image.resolve() != image:
                raise FileExistsError(f"输出图片重名：{destination_image.name}")
            shutil.copy2(image, destination_image)
            (output / "labels" / split / f"{image.stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
            )

        print(f"转换完成：{len(items)} 张图片；训练 {len(items)-val_count} 张，验证 {val_count} 张。")
        print(f"类别 ID：{class_ids}")
        print(f"数据集：{output}")
        print("现在可运行：bash train.sh")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"转换失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
