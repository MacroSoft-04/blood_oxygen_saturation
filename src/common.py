"""Paths and small helpers shared by check.py, vs_base_line.py and vs_model_v1.py.

Kept separate so importing a constant never triggers check.py's data loading.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ROW = DATA_DIR / "raw"
PROCESS = DATA_DIR / "process"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_TARGET = ROW / "train_target.csv"
TRAIN_INPUT = ROW / "train_input.parquet"
TEST_INPUT = ROW / "test_input.parquet"
META = ROW / "case_metadata.csv"
SAMPLE_SUBMISSION = ROW / "sample_submission.csv"

SIGNALS = ["HR", "MBP", "SPO2"]

SUBMISSION_ID_PATTERN = re.compile(
    r"(?P<sample_id>t\d{5})_" r"(?P<variable>[^_]+)_" r"h(?P<horizon>\d+)"
)


def parse_submission_id(sample: pd.DataFrame) -> pd.DataFrame:
    """Split sample_submission's id into sample_id / variable / horizon, keeping row order."""
    parsed = sample["id"].str.extract(SUBMISSION_ID_PATTERN)
    parsed["horizon"] = parsed["horizon"].astype(int)
    return pd.concat([parsed, sample["prediction"]], axis=1)


def set_korean_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in [
        "Malgun Gothic",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
    ]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
