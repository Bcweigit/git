#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v yolo >/dev/null 2>&1; then
  echo "未找到 yolo 命令。请先运行：conda activate yolo" >&2
  exit 1
fi

if [[ ! -d dataset/images/train || ! -d dataset/labels/train ]]; then
  echo "请先把人工标注图片和同名 YOLO .txt 标签放进 dataset/images/train 与 dataset/labels/train。" >&2
  exit 1
fi

if [[ ! -f data.yaml ]]; then
  echo "找不到 data.yaml" >&2
  exit 1
fi

MODEL="${MODEL:-yolo26n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"

yolo detect train \
  model="$MODEL" \
  data="$PROJECT_DIR/data.yaml" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device="$DEVICE" \
  project="$PROJECT_DIR/runs/detect" \
  name=train \
  exist_ok=True
