#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
YOLO_PYTHON="/home/bcw/miniconda3/envs/yolo/bin/python"
if [ ! -x "$YOLO_PYTHON" ]; then
    echo "找不到已有 yolo 环境：$YOLO_PYTHON"
    exit 1
fi
cd "$PROJECT_DIR"
export YOLO_AUTOINSTALL=false
exec "$YOLO_PYTHON" "$PROJECT_DIR/视频检测.py"
