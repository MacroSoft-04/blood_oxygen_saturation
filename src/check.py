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
TEST = ROW_DATA / "test_input.parquet"
TRAIN_INPUT = ROW_DATA / "train_input.parquet"

train = pd.read_csv(TRAIN)
sample = pd.read_csv(SAMPLE)
meta = pd.read_csv(META)
test_parquet = pd.read_parquet(TEST)
train_parquet = pd.read_parquet(TRAIN_INPUT)


def print_header(title: str):
    print(f"\n{'=' * 20}\n{title}\n{'=' * 20}")


def check_common(target: pd.DataFrame):
    print("[1] shape ", target.shape)
    print("\n[2] dtypes ", target.dtypes)
    print("\n[3] info ", target.info())
    print(f"\n[4] head {target.head()}")

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
    print(f"sample_id_cnt: {train["sample_id"].nunique()}")
    print(f"variable: {list(train["variable"].drop_duplicates())}")
    print(f"horizon: {list(train["horizon"].drop_duplicates())}")

    print("\n---horizon_value---")
    horizon_counts = (
        train.groupby(["sample_id", "horizon"]).size().unstack(fill_value=0)
    )
    count_check = horizon_counts.eq(horizon_counts.iloc[0]).all(axis=1)
    print("All IDs have the same horizon counts:", count_check.all())

    variable_counts = (
        train.groupby(["horizon", "variable"]).size().unstack(fill_value=0)
    )
    count_check = variable_counts.eq(variable_counts.iloc[0]).all(axis=1)
    print("Every horizon have the same variable counts:", count_check.all())

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
    """check which column contains other column between subjecid and caseid"""
    non_target_mask = (meta == 0.0).any(axis=1)
    target = meta.loc[~non_target_mask]

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


def check_sturucture_fillna_test_parquet(test: pd.DataFrame):
    """check strucure and fill the missing values"""
    print("\n---structure---")
    sample_id = test["sample_id"].nunique()
    print("sample_id_cnt: ", sample_id)
    sample_id_per_row = test.groupby("sample_id")["minute"].size().unique()
    print("sample_id_per_row: ", sample_id_per_row)

    expected = test["minute"].drop_duplicates().sort_values().reset_index(drop=True)
    structure_check = test.groupby("sample_id")["minute"].apply(
        lambda x: (
            x.drop_duplicates().sort_values().reset_index(drop=True).equals(expected)
        )
    )
    duplicate_count = test.duplicated(subset=["sample_id", "minute"]).sum()
    print("\nDuplicate combinations: ", duplicate_count)
    print("All IDs have the same structure: ", structure_check.all())
    print("IDs with different structure: ", (~structure_check).sum())
    print(test[test["sample_id"] == structure_check.index[0]].to_string(index=False))

    print("\n---missing_per_id---")
    SIGNALS = ["HR", "MBP", "SPO2"]
    missing_per_id = test.groupby("sample_id")[SIGNALS].apply(lambda g: g.isna().sum())
    print((missing_per_id > 0).sum())  # windows with any gap, per signal
    print((missing_per_id == 30).sum())  # windows where a signal is entirely missing

    print(missing_per_id.loc[(missing_per_id != 0).sum(axis=1).eq(1)])
    print(missing_per_id.loc[(missing_per_id != 0).sum(axis=1).ge(2)])


if __name__ == "__main__":
    # print_header("train")
    # check_common(train)
    # check_structure(train, "train", "value", ["sample_id"])
    # check_train_target(train)

    # print_header("sample")
    # check_common(sample)
    # print()
    # check_sample_submission(sample)

    # print_header("meta")
    # check_common(meta)
    # fillna_meta(meta)
    # check_structure(meta, "meta", "age", ["sample_id", "caseid", "subjectid"])
    # check_meta(meta)

    print_header("test_parquet")
    check_common(test_parquet)
    print()
    check_sturucture_fillna_test_parquet(test_parquet)

    # print_header("train_parquet")
    # check_common(train_parquet)
    # print()
    # check_sturucture_fillna_test_parquet(train_parquet)
