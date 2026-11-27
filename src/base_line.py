# %%
# 1. Imports and configuration
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

# %%
# 2. Load and prepare data
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR
DATA_DIR = PROJECT_ROOT / "data"
ROW = DATA_DIR / "raw"
PROCESS = DATA_DIR / "process"

TRAIN = ROW / "train_target.csv"
META = PROCESS / "meta_fillna.csv"
TEST = ROW / "test_input.parquet"
TRAIN_INPUT = ROW / "train_input.parquet"
SUB_ID = PROCESS / "sample_submission_id.csv"

train = pd.read_csv(TRAIN)
sub_id = pd.read_csv(SUB_ID)
meta = pd.read_csv(META)
test_parquet = pd.read_parquet(TEST)
train_parquet = pd.read_parquet(TRAIN_INPUT)

SIGNALS = ["HR", "MBP", "SPO2"]


# %%
def last_observed(inp: pd.DataFrame) -> pd.DataFrame:
    # groupby.last() skips NaN, so a window ending in a gap still gets its latest real reading.
    last = inp.sort_values(["sample_id", "minute"]).groupby("sample_id")[SIGNALS].last()
    return last.reset_index().melt(
        id_vars="sample_id", var_name="variable", value_name="last_value"
    )


# %%
def build_compare_value(standard: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    df = standard.merge(
        last_observed(target),
        on=["sample_id", "variable"],
        how="left",
        validate="many_to_one",
    )
    return df


# %%
def check_compare_value(standard: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    df = build_compare_value(standard, target)

    df["same"] = np.isclose(df["value"], df["last_value"], rtol=0, atol=1e-6)
    df["diff"] = df["value"] - df["last_value"]

    print("All values identical:", df["same"].all())
    same = df.loc[
        df["same"],
        ["sample_id", "variable", "horizon", "value", "last_value", "diff"],
    ]
    not_same = df.loc[
        ~df["same"],
        ["sample_id", "variable", "horizon", "value", "last_value", "diff"],
    ]

    print(same[same["variable"] == "SPO2"])

    print(len(not_same))
    return df


check_compare_value(train, train_parquet)

# %%
ID_PATTERN = r"(?P<case>c\d+)_o(?P<end>\d+)"


def check_overlap(standard: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Check whether a window's future target shows up in another window's input of the same case."""
    key = target["sample_id"].str.extract(ID_PATTERN)
    last_minute = target.groupby("sample_id")["minute"].transform("max")
    # Assumes o### is the case-level minute of the window's last row
    case_clock = (
        target.assign(
            case=key["case"],
            t=key["end"].astype(int) - (last_minute - target["minute"]),
        )
        .melt(
            id_vars=["case", "t"],
            value_vars=SIGNALS,
            var_name="variable",
            value_name="observed",
        )
        .dropna()
        .drop_duplicates(["case", "t", "variable"])
    )

    key = standard["sample_id"].str.extract(ID_PATTERN)
    future = standard.assign(
        case=key["case"], t=key["end"].astype(int) + standard["horizon"]
    )
    m = future.merge(case_clock, on=["case", "t", "variable"], how="left")
    m["found"] = m["observed"].notna()
    m["identical"] = np.isclose(m["value"], m["observed"])

    print("share of targets found in another window:")
    print(
        m.pivot_table(
            index="variable", columns="horizon", values="found", aggfunc="mean"
        ).round(3)
    )
    print("\nof those, share with an identical value:")
    print(
        m[m["found"]]
        .pivot_table(
            index="variable", columns="horizon", values="identical", aggfunc="mean"
        )
        .round(3)
    )
    return m


check_overlap(train, train_parquet)
# %%


def evaluate_persistence(
    train_input: pd.DataFrame, train_target: pd.DataFrame
) -> pd.DataFrame:
    df = train_target.merge(
        last_observed(train_input), on=["sample_id", "variable"], how="left"
    )
    print("Targets with no observed value in window:", df["last_value"].isna().sum())
    df["abs_err"] = (df["value"] - df["last_value"]).abs()
    print(df)
    return df.pivot_table(
        index="variable", columns="horizon", values="abs_err", aggfunc="mean"
    )


evaluate_persistence(train_parquet, train)


# %%
def make_persistence_submission(
    test_input: pd.DataFrame, sub_id: pd.DataFrame
) -> pd.DataFrame:
    pred = sub_id.merge(
        last_observed(test_input),
        on=["sample_id", "variable"],
        how="left",
        validate="many_to_one",
    )
    # Rebuild the id from its parts so the submission doesn't rely on row order
    pred["id"] = (
        pred["sample_id"]
        + "_"
        + pred["variable"]
        + "_h"
        + pred["horizon"].astype(int).astype(str)
    )
    print("Missing predictions:", pred["last_value"].isna().sum())
    return pred[["id", "last_value"]].rename(columns={"last_value": "prediction"})
