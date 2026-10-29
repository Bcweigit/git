"""运行此文件，选择视频后自动进行实时分割检测。"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from video_segment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
