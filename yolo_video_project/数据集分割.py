"""将 LabelMe 图片+JSON 数据集划分为训练集/验证集，并转换为 YOLO 分割格式。

直接运行：
    python 数据集分割.py

也可以传入路径：
    python 数据集分割.py --source "/你的原始数据集" --names "red metal bracket,blue bracket"
"""

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
from pathlib import Path


PROJECT_PYTHON = Path("/home/bcw/miniconda3/envs/yolo/bin/python")
MISSING = [name for name in ("PIL",) if importlib.util.find_spec(name) is None]
if MISSING:
    current = Path(sys.executable).resolve()
    if PROJECT_PYTHON.is_file() and current != PROJECT_PYTHON.resolve():
        print(f"当前 Python 缺少依赖，自动切换到：{PROJECT_PYTHON}", flush=True)
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit("缺少 Pillow，请使用项目 yolo 环境运行。")

from PIL import Image  # noqa: E402


ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ask_path(prompt, default=""):
    value = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip().strip("\"'")
    return value or default


def parse_names(value):
    names = [item.strip() for item in value.replace("，", ",").split(",")]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("类别名称不能为空或重复，例如：red metal bracket,blue bracket")
    return names


def convert_shape(shape, class_ids, width, height, json_path):
    label = str(shape.get("label", "")).strip()
    if label not in class_ids:
        raise ValueError(f"{json_path} 使用了未声明类别：{label}")
    if shape.get("shape_type", "polygon") not in {"polygon", "linestrip"}:
        raise ValueError(f"{json_path} 含不支持的类型：{shape.get('shape_type')}")
    points = shape.get("points") or []
    if len(points) < 3:
        raise ValueError(f"{json_path} 的 {label} 点数少于 3")
    values = []
    for point in points:
        x = max(0.0, min(float(width), float(point[0]))) / width
        y = max(0.0, min(float(height), float(point[1]))) / height
        values.extend((x, y))
    return str(class_ids[label]) + " " + " ".join(f"{value:.8f}" for value in values)


def run(source, output, names, val_ratio, seed):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"数据集目录不存在：{source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录已存在且非空：{output}\n请换一个输出目录。")

    images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    if len(images) < 2:
        raise ValueError("至少需要两张图片才能划分训练集和验证集。")
    class_ids = {name: index for index, name in enumerate(names)}
    items = []
    for image in images:
        json_path = image.with_suffix(".json")
        if not json_path.is_file():
            raise ValueError(f"图片缺少同名 JSON：{image.name}")
        with Image.open(image) as opened:
            width, height = opened.size
            opened.verify()
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        rows = [convert_shape(shape, class_ids, width, height, json_path) for shape in data.get("shapes", [])]
        if not rows:
            raise ValueError(f"没有有效 polygon 标注：{json_path}")
        items.append((image, rows))

    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    val_count = max(1, min(len(items) - 1, round(len(items) * val_ratio)))
    val_indices = set(order[:val_count])
    try:
        for split in ("train", "val"):
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, (image, rows) in enumerate(items):
            split = "val" if index in val_indices else "train"
            shutil.copy2(image, output / "images" / split / image.name)
            (output / "labels" / split / f"{image.stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (output / "data.yaml").write_text(
            "path: " + str(output) + "\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(names)),
            encoding="utf-8",
        )
        (output / "split_manifest.json").write_text(
            json.dumps({"source": str(source), "names": names, "seed": seed,
                        "val_ratio": val_ratio, "images": len(items)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(f"完成：共 {len(items)} 张；训练集 {len(items) - val_count} 张；验证集 {val_count} 张。")
    print(f"YOLO 配置：{output / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="", help="包含图片和同名 JSON 的原始数据集目录")
    parser.add_argument("--output", default="", help="输出 YOLO 分割数据集目录")
    parser.add_argument("--names", default="", help="类别名称，逗号分隔，顺序对应类别 ID")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()
    try:
        source = args.source or ask_path("请输入原始数据集路径")
        names_value = args.names or ask_path("请输入类别名称（逗号分隔）", "red metal bracket,blue bracket")
        output = args.output or ask_path("请输入输出路径", str(ROOT / "dataset" / "seg_split"))
        if not source or not names_value or not output:
            raise ValueError("路径和类别名称不能为空。")
        if not 0 < args.val_ratio < 1:
            raise ValueError("--val-ratio 必须在 0 和 1 之间。")
        run(source, output, parse_names(names_value), args.val_ratio, args.seed)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
