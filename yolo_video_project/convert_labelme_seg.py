"""LabelMe polygon/linestrip 标注转换为 YOLO 分割格式。"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_names(raw):
    names = [item.strip() for item in raw.replace("，", ",").split(",")]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("类别名称不能为空或重复。")
    return names


def convert_shape(shape, class_ids, width, height, json_path):
    label = str(shape.get("label", "")).strip()
    if label not in class_ids:
        raise ValueError(f"{json_path} 使用了未声明的类别：{label}")
    if shape.get("shape_type", "polygon") not in {"polygon", "linestrip"}:
        raise ValueError(f"{json_path} 含不支持的形状类型：{shape.get('shape_type')}")
    points = shape.get("points") or []
    if len(points) < 3:
        raise ValueError(f"{json_path} 标注点少于 3 个。")
    values = []
    for point in points:
        x = max(0.0, min(float(width), float(point[0]))) / width
        y = max(0.0, min(float(height), float(point[1]))) / height
        values.extend((x, y))
    return str(class_ids[label]) + " " + " ".join(f"{value:.8f}" for value in values)


def convert(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    names = parse_names(args.names)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录已存在且非空：{output}")
    images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if len(images) < 2:
        raise ValueError("至少需要两张图片。")
    items, class_ids = [], {name: index for index, name in enumerate(names)}
    for image in images:
        json_path = image.with_suffix(".json")
        with Image.open(image) as opened:
            width, height = opened.size
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        rows = [convert_shape(shape, class_ids, width, height, json_path) for shape in data.get("shapes", [])]
        if not rows:
            raise ValueError(f"没有有效标注：{json_path}")
        items.append((image, rows))
    order = list(range(len(items)))
    random.Random(args.seed).shuffle(order)
    val_indices = set(order[:max(1, round(len(items) * args.val_ratio))])
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for index, (image, rows) in enumerate(items):
        split = "val" if index in val_indices else "train"
        shutil.copy2(image, output / "images" / split / image.name)
        (output / "labels" / split / f"{image.stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (output / "data.yaml").write_text("path: " + str(output) + "\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(names)), encoding="utf-8")
    print(f"分割数据转换完成：{len(items)} 张图片。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--names", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        convert(args)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
