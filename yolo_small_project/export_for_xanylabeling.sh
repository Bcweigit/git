#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="${1:-$PROJECT_DIR/runs/detect/train/weights/best.pt}"
cd "$PROJECT_DIR"

if ! command -v yolo >/dev/null 2>&1; then
  echo "请先运行：conda activate yolo" >&2
  exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "找不到训练权重：$WEIGHTS" >&2
  exit 1
fi

if ! python -c 'import onnx, onnxslim' >/dev/null 2>&1; then
  echo "缺少 ONNX 导出依赖，请在 yolo 环境运行：python -m pip install onnx onnxslim" >&2
  exit 1
fi

yolo export model="$WEIGHTS" format=onnx imgsz=640 opset=12 simplify=True nms=False

ONNX_PATH="${WEIGHTS%.pt}.onnx"
if [[ ! -f "$ONNX_PATH" ]]; then
  echo "导出结束但没找到 ONNX：$ONNX_PATH" >&2
  exit 1
fi
echo "ONNX 模型已导出：$ONNX_PATH"
echo "X-AnyLabeling 配置：$PROJECT_DIR/xanylabeling_model.yaml"
