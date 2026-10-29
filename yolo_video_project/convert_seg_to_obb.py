"""把 YOLO 分割标签转换为 YOLO OBB 四点旋转框标签。"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def convert_split(source, output, split):
    image_dir, label_dir = source / "images" / split, source / "labels" / split
    out_image_dir, out_label_dir = output / "images" / split, output / "labels" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = next((image_dir / f"{label_path.stem}{ext}" for ext in IMAGE_EXTS if (image_dir / f"{label_path.stem}{ext}").is_file()), None)
        if image_path is None:
            raise ValueError(f"找不到图片：{label_path.stem}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"图片无法读取：{image_path}")
        height, width = image.shape[:2]
        rows = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if len(values) < 6 or len(values) % 2:
                raise ValueError(f"分割标签点数错误：{label_path}")
            points = np.array([(x * width, y * height) for x, y in zip(values[0::2], values[1::2])], dtype=np.float32)
            corners = cv2.boxPoints(cv2.minAreaRect(points))
            normalized = []
            for x, y in corners:
                normalized.extend((max(0, min(1, float(x) / width)), max(0, min(1, float(y) / height))))
            rows.append(str(class_id) + " " + " ".join(f"{value:.8f}" for value in normalized))
        shutil.copy2(image_path, out_image_dir / image_path.name)
        (out_label_dir / label_path.name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "dataset" / "seg_source"))
    parser.add_argument("--output", default=str(ROOT / "dataset" / "obb_source"))
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录已存在且非空：{output}")
    try:
        raw = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8-sig"))
        names = raw["names"]
        names = [names[key] for key in sorted(names, key=lambda item: int(item))] if isinstance(names, dict) else names
        train_count = convert_split(source, output, "train")
        val_count = convert_split(source, output, "val")
        (output / "data.yaml").write_text("path: " + str(output) + "\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(names)), encoding="utf-8")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(f"OBB 数据转换完成：训练 {train_count} 张，验证 {val_count} 张。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
