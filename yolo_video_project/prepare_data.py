"""训练数据集配置与基础检查工具。"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def names_list(raw):
    if isinstance(raw, dict):
        values = [raw[key] for key in sorted(raw, key=lambda item: int(item))]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError("names 必须是列表或数字字典。")
    values = [str(value).strip() for value in values]
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError("类别名称不能为空或重复。")
    return values


def write_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def images_in(folder):
    folder = Path(folder)
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def pairs_in(image_folder, label_folder):
    pairs = []
    for image in images_in(image_folder):
        label = Path(label_folder) / f"{image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"图片缺少对应标签：{image}")
        pairs.append((image, label))
    return pairs


def validate_pairs(pairs, names):
    for image, label in pairs:
        try:
            with Image.open(image) as opened:
                opened.verify()
        except Exception as exc:
            raise ValueError(f"图片无法读取：{image}") from exc
        for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"标签格式错误：{label}:{line_number}") from exc
            if class_id < 0 or class_id >= len(names):
                raise ValueError(f"类别 ID 超出范围：{label}:{line_number}")
            if len(values) not in (4, 8) and len(values) < 6:
                raise ValueError(f"标签坐标数量不正确：{label}:{line_number}")
            if any(value < 0 or value > 1 for value in values):
                raise ValueError(f"标签坐标必须在 0 到 1：{label}:{line_number}")


def read_dataset(config_path):
    config_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    names = names_list(raw.get("names"))
    root = Path(raw.get("path") or ".")
    root = (config_path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    splits = {}
    stats = {}
    for split in ("train", "val"):
        image_dir = root / raw[split]
        label_dir = root / "labels" / split
        pairs = pairs_in(image_dir, label_dir)
        validate_pairs(pairs, names)
        splits[split] = pairs
        stats[split] = {"images": len(pairs), "objects": sum(
            len([line for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()])
            for _, label in pairs
        )}
    data = {"path": str(root), "train": raw["train"], "val": raw["val"], "names": dict(enumerate(names))}
    return data, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(ROOT / "data.yaml"))
    args = parser.parse_args()
    try:
        data, stats = read_dataset(args.config)
        print(json.dumps({"data": data, "stats": stats}, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
