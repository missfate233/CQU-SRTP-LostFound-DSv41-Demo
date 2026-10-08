"""CQU-SRTP 失物招领系统.

图像感知匹配（pHash + HSV 颜色直方图） + 文本标签匹配（TF-IDF + 标签重合 + 地点语义）
的混合召回引擎，后端 Flask + SQLite。
"""

from __future__ import annotations

import os
import sys

# 本项目自带 vendor/ 依赖目录（由 bootstrap.py 生成），保证任何入口导入
# lostfound 时都能找到 Flask，无需先配置 PYTHONPATH 或虚拟环境。
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDOR_DIR = os.path.join(_BASE_DIR, "vendor")
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

__version__ = "1.0.0"
__all__ = ["__version__"]
