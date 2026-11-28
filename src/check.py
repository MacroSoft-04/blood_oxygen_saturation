# %%
# 1. Imports and configuration
import re

import numpy as np
import pandas as pd

from common import (
    META,
    PROCESS,
    SAMPLE_SUBMISSION,
    SIGNALS,
    TEST_INPUT,
    TRAIN_INPUT,
    TRAIN_TARGET,
)

SAVE = PROCESS / "sample_submission_id.csv"
SAVE_META = PROCESS / "meta_fillna.csv"

# %%
# 2. Load and prepare data
train = pd.read_csv(TRAIN_TARGET)
sample = pd.read_csv(SAMPLE_SUBMISSION)
meta = pd.read_csv(META)
test_parquet = pd.read_parquet(TEST_INPUT)
train_parquet = pd.read_parquet(TRAIN_INPUT)


def print_header(title: str):
    print(f"\n{'=' * 20}\n{title}\n{'=' * 20}")


# %%
def report_common(target: pd.DataFrame):
    print("[1] shape ", target.shape)
    print("\n[2] dtypes ", target.dtypes)
    # info() prints by itself and returns None, so it is not wrapped in print().
    print("\n[3] info ")
    target.info()
    print(f"\n[4] head \n{target.head()}")

    print("\n---null---")
    isna = target.isna().sum()
    print(isna)
    if (isna != 0).any():
        print(f"\n{target[target.isna().any(axis=1)].head()}")


# %%
# train_target
def report_train_target(train: pd.DataFrame):
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

    print("\n---value by variable---")
    print(train.groupby("variable")["value"].describe().round(1))


# %%
# split sample id and create a data frame
ID_PATTERN_SAMPLE = re.compile(
    r"(?P<sample_id>t\d{5})_" r"(?P<variable>[^_]+)_" r"h(?P<horizon>\d+)"
)


def build_sample_id(sample: pd.DataFrame) -> pd.DataFrame:
    id_sort = sample["id"].str.extract(ID_PATTERN_SAMPLE)
    parsed = pd.concat([id_sort, sample["prediction"]], axis=1)
    return parsed


# %%
ID_PATTERN = r"c(?P<caseid>\d+)_o(?P<end>\d+)"


def build_meta_sample_id(target: pd.DataFrame) -> pd.DataFrame:
    key = target["sample_id"].str.extract(ID_PATTERN)
    parsed = pd.concat([target, key], axis=1)
    return parsed


# %%
# sample_submission
def report_sample_id_structure(
    sample: pd.DataFrame, parsed: pd.DataFrame
) -> pd.DataFrame:
    print("not 0 prediction row num: ", (sample["prediction"] != 0.0).sum())
    first_id = sample["id"].iloc[0]
    print("Original ID:", repr(first_id))
    print("Regex match:", ID_PATTERN_SAMPLE.match(first_id))
    print("IDs not matching the pattern:", parsed["sample_id"].isna().sum())

    expected = (
        parsed[["variable", "horizon"]]
        .drop_duplicates()
        .sort_values(["variable", "horizon"])
        .reset_index(drop=True)
    )
    structure_check = parsed.groupby("sample_id").apply(
        lambda x: (x[["variable", "horizon"]])
        .sort_values(["variable", "horizon"])
        .reset_index(drop=True)
        .equals(expected)
    )
    duplicate_count = parsed.duplicated(["sample_id", "variable", "horizon"]).sum()

    print("\nDuplicate combinations: ", duplicate_count)
    print("All IDs have the same structure: ", structure_check.all())
    print("IDs with different structure: ", (~structure_check).sum())
    print("Variables to predict:", list(parsed["variable"].drop_duplicates()))

    first_id = parsed["sample_id"].iloc[0]
    print(f"\n{parsed[parsed["sample_id"] == first_id]}")
    return parsed


# %%
# case_metadata
def fillna_meta(meta: pd.DataFrame) -> pd.DataFrame:
    print(f"caseid sample: {meta["caseid"].unique()[:5]}")
    print(f"subjectid sample: {meta["subjectid"].unique()[:5]}")

    # Work on a copy so the global `meta` keeps its NaN for other checks.
    meta = meta.copy()
    meta["caseid"] = meta["caseid"].fillna(0.0)
    meta["subjectid"] = meta["subjectid"].fillna(0.0)
    fillna_check = meta[(meta["caseid"] == 0.0) & (meta["subjectid"] == 0.0)]
    print()
    fillna_check.info()
    return meta


# %%
def report_meta(meta: pd.DataFrame):
    """check which column contains other column between subjecid and caseid
    `meta` is expected to come from fillna_meta (0.0 = hidden test IDs)."""
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

    print(result.to_string(index=False))
    print("2) No caseids :", no_caseids.count())

    unique_counts = target.groupby("caseid").nunique().drop(columns="sample_id")

    caseid_not_criteria = unique_counts[(unique_counts > 1).any(axis=1)]
    caseid_criteria = unique_counts[(unique_counts == 1).all(axis=1)]

    print("\n--Whether each caseid meets the criteria--")
    print("1) Inconsistent:", len(caseid_not_criteria))
    print("2) Consistent:", len(caseid_criteria))

    print("\n---sample_id---")
    # Use the full meta here: test rows are exactly the ones the 0.0 mask removed above.
    caseid = meta.loc[meta["sample_id"].str.startswith("c", na=False)]

    print("\n1) caseid")
    print(caseid.head())
    print("Missing values:")
    print(caseid.isna().sum())

    subjectid = meta.loc[meta["sample_id"].str.startswith("t", na=False)]
    print("\n2) subjectid")
    print(subjectid.head())
    print("Missing values:")
    print(subjectid.isna().sum())

    print("caseid rows:", len(caseid))
    print("subjectid rows:", len(subjectid))
    print("Total rows:", len(meta))
    print("Unclassified rows:", len(meta) - len(caseid) - len(subjectid))

    print("\n---age---")
    # All values on a 5-year grid would mean age is binned, not exact.
    print("All ages multiples of 5:", (meta["age"] % 5 == 0).all())


# %%
def check_structure_parquet(test: pd.DataFrame):
    """check structure and missing values of train_input / test_input"""
    print("\n---structure---")
    sample_id = test["sample_id"].nunique()
    print("sample_id_cnt: ", sample_id)

    each_sample_id_row_cnt = test.groupby("sample_id")["minute"].size().unique()
    print("each_sample_id_row_cnt: ", each_sample_id_row_cnt.tolist())

    each_sample_id_row = test.groupby("sample_id")["minute"].unique()
    minute = sorted(each_sample_id_row.iloc[0].tolist(), reverse=True)
    print("minute in each sample id", min(minute), "~", max(minute))

    missing_minutes = sorted(set(range(min(minute), max(minute) + 1)) - set(minute))
    print("missing_minutes:", missing_minutes)

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
    missing_per_id = test.groupby("sample_id")[SIGNALS].apply(lambda g: g.isna().sum())
    window_len = test.groupby("sample_id").size().max()
    print(
        "missing_per_id > 0 :\n", (missing_per_id > 0).sum()
    )  # windows with any gap, per signal
    print(
        "missing_per_id == window length :\n", (missing_per_id == window_len).sum()
    )  # windows where a signal is entirely missing

    # The last minute is what persistence and most features lean on.
    last_minute = test.loc[test["minute"] == test["minute"].max(), SIGNALS]
    print("missing at the last minute :\n", last_minute.isna().sum())

    print(missing_per_id.loc[(missing_per_id != 0).sum(axis=1).eq(1)])
    print(missing_per_id.loc[(missing_per_id != 0).sum(axis=1).ge(2)])


# %%
def check_id_name(sample: pd.DataFrame, parquet: pd.DataFrame):
    sample_id = sample["sample_id"].unique()
    parquet_id = parquet["sample_id"].unique()

    sample_only = np.setdiff1d(sample_id, parquet_id)
    parquet_only = np.setdiff1d(parquet_id, sample_id)

    if len(sample_only) == 0 and len(parquet_only) == 0:
        print("Both files have the same IDs.")
    else:
        print("The IDs are different.")
        print("IDs only in sample:", len(sample_only), "/", len(sample_id))
        print(sample_only[:10])
        print("IDs only in parquet:", len(parquet_only), "/", len(parquet_id))
        print(parquet_only[:10])


# %%
OVERLAP_PATTERN = r"(?P<case>c\d+)_o(?P<end>\d+)"


def check_overlap(standard: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Check whether a window's future target shows up in another window's input of the same case."""
    key = target["sample_id"].str.extract(OVERLAP_PATTERN)
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

    ends = (
        target.assign(case=key["case"], end=key["end"].astype(int))
        .drop_duplicates("sample_id")
        .sort_values(["case", "end"])
    )
    print("gap between consecutive window ends (minutes):")
    print(ends.groupby("case")["end"].diff().value_counts().head())

    key = standard["sample_id"].str.extract(OVERLAP_PATTERN)
    future = standard.assign(
        case=key["case"], t=key["end"].astype(int) + standard["horizon"]
    )
    m = future.merge(case_clock, on=["case", "t", "variable"], how="left")
    m["found"] = m["observed"].notna()
    m["identical"] = np.isclose(m["value"], m["observed"])

    print("\nshare of targets found in another window:")
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


# %%
def report_train_test_shift(
    meta: pd.DataFrame, train_in: pd.DataFrame, test_in: pd.DataFrame
):
    split = meta["sample_id"].str[0].map({"c": "train", "t": "test"})
    print("---age---")
    print(meta.groupby(split)["age"].describe().round(1))
    for col in ["sex", "department"]:
        print(f"\n---{col}---")
        print(pd.crosstab(meta[col], split, normalize="columns").round(3))

    print("\n---missing row ratio---")
    print(
        pd.DataFrame(
            {
                "train": train_in[SIGNALS].isna().mean(),
                "test": test_in[SIGNALS].isna().mean(),
            }
        ).round(4)
    )


# %%
if __name__ == "__main__":
    print_header("train")
    report_common(train)
    report_train_target(train)

    print_header("sample")
    report_common(sample)
    parsed = report_sample_id_structure(sample, build_sample_id(sample))

    print_header("meta")
    report_common(meta)
    meta_filled = fillna_meta(meta)
    report_meta(meta_filled)

    print_header("train_parquet")
    report_common(train_parquet)
    check_structure_parquet(train_parquet)

    print_header("test_parquet")
    report_common(test_parquet)
    check_structure_parquet(test_parquet)

    print_header("ID check")
    test_meta = meta[meta["sample_id"].str.startswith("t")]
    train_meta = meta[meta["sample_id"].str.startswith("c")]
    print("[test_parquet vs sample]")
    check_id_name(parsed, test_parquet)
    print("[test_parquet vs meta(test)]")
    check_id_name(test_meta, test_parquet)
    print("[train_parquet vs train_target]")
    check_id_name(train, train_parquet)
    print("[train_parquet vs meta(train)]")
    check_id_name(train_meta, train_parquet)

    print_header("overlap")
    check_overlap(train, train_parquet)

    print_header("train vs test")
    report_train_test_shift(meta, train_parquet, test_parquet)

    PROCESS.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(SAVE, index=False)
    meta_filled.to_csv(SAVE_META, index=False)
