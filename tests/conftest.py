"""pytest 全局配置：把项目根目录加入 sys.path，使 tests 能 import core/api。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
