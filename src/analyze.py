import sys
import re
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR
DATA_DIR = PROJECT_ROOT / "data"
ROW_DATA = DATA_DIR / "raw"
TRAIN = ROW_DATA / "train_target.csv"
SAMPLE = ROW_DATA / "sample_submission.csv"

train = pd.read_csv(TRAIN)
sample = pd.read_csv(SAMPLE)


def common_check(target: pd.DataFrame):
    print(target.info())
    print(f"\n{target.head()}")

    print("\n---null---")
    print(target.isna().sum())


# train_target
def train_target_check(train: pd.DataFrame):
    """check train_target.csv structure. check whether all IDs have the same number of rows"""
    print("\n---col---")
    print(f"sample_id: {train["sample_id"].nunique()}")
    print(f"variable: {list(train["variable"].drop_duplicates())}")
    print(f"horizon: {list(train["horizon"].drop_duplicates())}")
    print(f"value: {list(train["value"].drop_duplicates().head(5))}")

    val = train["value"].agg(
        min="min",
        max="max",
        mean="mean",
        mode=lambda x: x.mode().iloc[0],
        std="std",
        missing=lambda x: x.isna().sum(),
    )
    print("\n---val---")
    print(val)

    print("\n---sample_id---")
    sample = train.groupby("sample_id")
    sample_counts = sample.size()
    print(f"Number of unique IDs: {len(sample_counts)}")
    print(f"Maximum rows per ID: {max(sample_counts)}")
    print(f"Minimum rows per ID: {min(sample_counts)}")
    if sample_counts.nunique() == 1:
        print("All IDs have the same number of rows")

    horizon = train.groupby("horizon")["variable"]

    print("\n---horizon_value---")
    for horizon_value, variable in horizon:
        print(f"\nHorizon value: {horizon_value}")
        print(variable.value_counts())

    expected = (
        train[["horizon", "variable"]]
        .drop_duplicates()
        .sort_values(["horizon", "variable"])
        .reset_index(drop=True)
    )
    structure_check = train.groupby("sample_id").apply(
        lambda x: (x[["horizon", "variable"]])
        .sort_values(["horizon", "variable"])
        .reset_index(drop=True)
        .equals(expected)
    )
    duplicate_count = train.duplicated(["sample_id", "horizon", "variable"]).sum()

    print("\nDuplicate combinations: ", duplicate_count)
    print("All IDs have the same structure: ", structure_check.all())
    print("IDs with different structure: ", (~structure_check).sum())

    first_id = train["sample_id"].iloc[0]
    print(f"\n{train[train["sample_id"] == first_id]}")


# sample_submission
def sample_submission_check(sample: pd.DataFrame):
    print("not 0 prediction row num: ", (sample["prediction"] != 0.0).sum())
    id_pattern = re.compile(r"(?P<id>t\d{5})_" r"(?P<name>[^_]+)_" r"h(?P<horizon>\d+)")
    first_id = sample["id"].iloc[0]

    print("Original ID:", repr(first_id))
    print("Regex match:", id_pattern.match(first_id))

    id_sort = sample["id"].str.extract(id_pattern)

    print(id_sort.head())

    expected = (
        id_sort[["name", "horizon"]]
        .drop_duplicates()
        .sort_values(["name", "horizon"])
        .reset_index(drop=True)
    )
    structure_check = id_sort.groupby("id").apply(
        lambda x: (x[["name", "horizon"]])
        .sort_values(["name", "horizon"])
        .reset_index(drop=True)
        .equals(expected)
    )
    duplicate_count = id_sort.duplicated(["id", "name", "horizon"]).sum()

    print("\nDuplicate combinations: ", duplicate_count)
    print("All IDs have the same structure: ", structure_check.all())
    print("IDs with different structure: ", (~structure_check).sum())

    first_id = id_sort["id"].iloc[0]
    print(f"\n{id_sort[id_sort["id"] == first_id]}")


# case_metadata

if __name__ == "__main__":
    common_check(sample)
    sample_submission_check(sample)
