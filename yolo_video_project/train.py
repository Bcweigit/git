"""YOLO 检测、分割和 OBB 模型训练入口。"""

import argparse
import sys
from pathlib import Path

import yaml

import prepare_data


ROOT = Path(__file__).resolve().parent


def device_value(value):
    import torch
    if value == "auto":
        value = "0" if torch.cuda.is_available() else "cpu"
    if value != "cpu" and not torch.cuda.is_available():
        raise ValueError("当前环境无法使用 CUDA，请使用 --device cpu 或修复 NVIDIA 驱动。")
    print(f"设备：{value}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    return value


def train(args):
    from ultralytics import YOLO

    config_path = Path(args.data).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    names = prepare_data.names_list(raw.get("names"))
    root = Path(raw.get("path") or ".")
    root = (config_path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    data = {"path": str(root), "train": raw["train"], "val": raw["val"], "names": dict(enumerate(names))}

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise ValueError(f"找不到预训练权重：{model_path}")
    device = device_value(args.device)
    config = ROOT / "configs" / f"train_{prepare_data.timestamp()}.yaml"
    prepare_data.write_yaml(config, data)

    if args.task in ("segment", "obb") and args.task not in model_path.stem.lower():
        import ultralytics
        model_name = "yolo26-seg.yaml" if args.task == "segment" else "yolo26-obb.yaml"
        model_cfg = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "26" / model_name
        model = YOLO(str(model_cfg), task=args.task)
        model.load(str(model_path))
    else:
        model = YOLO(str(model_path), task=args.task)

    run_name = f"{args.task}_{prepare_data.timestamp()}"
    model.train(data=str(config), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, patience=40, workers=args.workers,
                project=str(ROOT / "runs" / args.task), name=run_name,
                exist_ok=False, seed=42, amp=False)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("训练结束但未生成 best.pt。")
    (ROOT / "models" / "latest_model.txt").write_text(str(best.resolve()) + "\n", encoding="utf-8")
    print(f"训练完成：{best}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data.yaml"))
    parser.add_argument("--model", default=str(ROOT / "models" / "yolo26n.pt"))
    parser.add_argument("--task", choices=["detect", "segment", "obb"], default="detect")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        train(args)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
