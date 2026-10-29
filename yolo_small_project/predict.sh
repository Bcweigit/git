#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v yolo >/dev/null 2>&1; then
  echo "未找到 yolo 命令。请先运行：conda activate yolo" >&2
  exit 1
fi

WEIGHTS="${1:-$PROJECT_DIR/runs/detect/train/weights/best.pt}"
SOURCE="${SOURCE:-$PROJECT_DIR/images/unlabeled}"
CONF="${CONF:-0.25}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "找不到模型权重：$WEIGHTS" >&2
  echo "请先训练模型，或运行：bash predict.sh /path/to/best.pt" >&2
  exit 1
fi

if [[ ! -d "$SOURCE" ]]; then
  echo "找不到图片目录：$SOURCE" >&2
  echo "请把待预标注图片放入 images/unlabeled/，或设置 SOURCE=/图片目录" >&2
  exit 1
fi

yolo detect predict \
  model="$WEIGHTS" \
  source="$SOURCE" \
  conf="$CONF" \
  save=True \
  save_txt=True \
  save_conf=True \
  project="$PROJECT_DIR/prelabels" \
  name=predictions \
  exist_ok=True

echo "预标注结果在：$PROJECT_DIR/prelabels/predictions/（图片及 labels/ 子目录）"
