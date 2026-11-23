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
META = ROW_DATA / "case_metadata.csv"

train = pd.read_csv(TRAIN)
sample = pd.read_csv(SAMPLE)
meta = pd.read_csv(META)


def print_header(title: str):
    print(f"\n{'=' * 20}\n{title}\n{'=' * 20}")


def check_common(target: pd.DataFrame):
    print(target.info())
    print(f"\n{target.head()}")

    print("\n---null---")
    isna = target.isna().sum()
    print(isna)
    if (isna != 0).any():
        print(f"\n{target[target.isna().any(axis=1)].head()}")


def check_structure(df: pd.DataFrame, title: str, value: str, list_standard: list):
    """Check whether IDs have consistent row counts."""

    print(f"<check {title} structure>")

    # Summary statistics
    val = df[value].agg(
        min="min",
        max="max",
        mean="mean",
        mode=lambda x: x.mode().iloc[0],
        std="std",
        missing=lambda x: x.isna().sum(),
    )

    print("\n---val---")
    print(val)

    for standard in list_standard:
        print(f"\n---{standard}---")

        # Exclude missing IDs and placeholder IDs (0.0)
        target = df.loc[df[standard].notna() & (df[standard] != 0.0)]

        sample_counts = target.groupby(standard).size()

        print(f"Number of unique IDs: {len(sample_counts)}")
        print(f"Maximum rows per ID: {sample_counts.max()}")
        print(f"Minimum rows per ID: {sample_counts.min()}")

        if sample_counts.nunique() == 1:
            print("All IDs have the same number of rows.")
        else:
            print("IDs have different numbers of rows.")


# train_target
def check_train_target(train: pd.DataFrame):
    """train_target.csv structure.
    check whether all IDs have the same number of rows
    check whether all IDs have the same structure"""
    print("---col---")
    print(f"sample_id: {train["sample_id"].nunique()}")
    print(f"variable: {list(train["variable"].drop_duplicates())}")
    print(f"horizon: {list(train["horizon"].drop_duplicates())}")
    print(f"value: {list(train["value"].drop_duplicates().head(5))}")

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
def check_sample_submission(sample: pd.DataFrame):
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
def fillna_meta(meta: pd.DataFrame) -> pd.DataFrame:
    print(f"caseid sample: {meta["caseid"].unique()[:5]}")
    print(f"subjectid sample: {meta["subjectid"].unique()[:5]}")

    meta["caseid"] = meta["caseid"].fillna(0.0)
    meta["subjectid"] = meta["subjectid"].fillna(0.0)
    fillna_check = meta[(meta["caseid"] == 0.0) & (meta["subjectid"] == 0.0)]
    print(f"\n{fillna_check.info()}")
    return meta


def check_meta(meta: pd.DataFrame):
    non_target_mask = (meta == 0.0).any(axis=1)
    target = meta.loc[~non_target_mask]

    counts_caseid = target.groupby("caseid").size()
    counts_subjectid = target.groupby("subjectid").size()

    max_caseid = counts_caseid.idxmax()
    max_subjectid = counts_subjectid.idxmax()
    caseid = meta[meta["caseid"] == max_caseid]
    subjectid = meta[meta["subjectid"] == max_subjectid]

    print("\n---max_ID---")
    print("max_caseid: ", max_caseid)
    print("max_subjectid: ", max_subjectid)
    print(
        caseid[["age", "sex", "department"]],
    )
    print(
        subjectid[["age", "sex", "department"]],
    )

    print(subjectid.groupby("caseid").size())
    print()

    caseid_num = target.groupby("subjectid")["caseid"].nunique()

    multiple_caseids = caseid_num[caseid_num > 1]
    no_caseids = caseid_num[caseid_num == 0]

    print("\n--Each of subjectid--")
    print("1) Multiple caseids :", multiple_caseids.count(), multiple_caseids.unique())
    sample_mask = target["subjectid"].isin(multiple_caseids.index)

    result = (
        target.loc[
            sample_mask,
            ["subjectid", "caseid"],
        ]
        .drop_duplicates()
        .sort_values(["subjectid", "caseid"])
    )

    print(result)
    print("2) No caseids :", no_caseids.count())

    unique_counts = target.groupby("caseid").nunique().drop(columns="sample_id")

    caseid_not_criteria = unique_counts[(unique_counts > 1).any(axis=1)]
    caseid_criteria = unique_counts[(unique_counts == 1).all(axis=1)]

    print("\n--Whether each caseid meets the criteria--")
    print("1) Inconsistent:", len(caseid_not_criteria))
    print("2) Consistent:", len(caseid_criteria))


if __name__ == "__main__":
    check_structure(train, "train", "value", ["sample_id"])
    print_header("meta")
    check_common(meta)
    fillna_meta(meta)
    check_structure(meta, "meta", "age", ["sample_id", "caseid", "subjectid"])
    check_meta(meta)
