"""Data checks for every raw file, plus the cross-file checks the model relies on.

Run as a script to print the full report and write the small processed files that
other scripts use. Each `# %%` cell can also be run on its own in the interactive window.
"""

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESS_DIR = DATA_DIR / "process"

TRAIN_TARGET = RAW_DIR / "train_target.csv"
TRAIN_INPUT = RAW_DIR / "train_input.parquet"
TEST_INPUT = RAW_DIR / "test_input.parquet"
META = RAW_DIR / "case_metadata.csv"
SAMPLE_SUBMISSION = RAW_DIR / "sample_submission.csv"

SAVE_SUBMISSION_ID = PROCESS_DIR / "sample_submission_id.csv"
SAVE_META = PROCESS_DIR / "meta_split.csv"

SIGNALS = ["HR", "MBP", "SPO2"]
TRAIN_ID_PATTERN = re.compile(r"c(?P<caseid>\d+)_o(?P<end>\d+)")
SUBMISSION_ID_PATTERN = re.compile(
    r"(?P<sample_id>t\d{5})_(?P<variable>[^_]+)_h(?P<horizon>\d+)"
)


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "train_target": pd.read_csv(TRAIN_TARGET),
        "train_input": pd.read_parquet(TRAIN_INPUT),
        "test_input": pd.read_parquet(TEST_INPUT),
        "meta": pd.read_csv(META),
        "sample": pd.read_csv(SAMPLE_SUBMISSION),
    }


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


# %%
# Parsing helpers
def parse_submission_id(sample: pd.DataFrame) -> pd.DataFrame:
    parsed = sample["id"].str.extract(SUBMISSION_ID_PATTERN)
    parsed["horizon"] = parsed["horizon"].astype(int)
    return pd.concat([parsed, sample["prediction"]], axis=1)


def add_split(meta: pd.DataFrame) -> pd.DataFrame:
    # Test rows have caseid/subjectid hidden, so the ID prefix is the reliable split marker.
    meta = meta.copy()
    meta["split"] = meta["sample_id"].str[0].map({"c": "train", "t": "test"})
    return meta


def add_case_clock(inp: pd.DataFrame) -> pd.DataFrame:
    """Attach caseid and the case-level minute (`t`) to train input rows.

    Assumes o### in the sample_id is the case minute of the window's last row (minute 0).
    """
    key = inp["sample_id"].str.extract(TRAIN_ID_PATTERN).astype(int)
    return inp.assign(
        caseid=key["caseid"], end=key["end"], t=key["end"] + inp["minute"]
    )


# %%
# 1. Common report for any table
def report_common(df: pd.DataFrame) -> None:
    print("shape:", df.shape)
    print_section("dtypes")
    print(df.dtypes.to_string())
    print_section("head")
    print(df.head().to_string(index=False))

    print_section("nulls")
    nulls = df.isna().sum()
    print(nulls.to_string())
    if nulls.any():
        print("\nRows with any null (first 5):")
        print(df[df.isna().any(axis=1)].head().to_string(index=False))


# %%
# 2. Input windows (train_input / test_input)
def report_input(inp: pd.DataFrame) -> None:
    print_section("window structure")
    rows_per_id = inp.groupby("sample_id").size()
    print("windows:", len(rows_per_id))
    print("rows per window:", sorted(rows_per_id.unique().tolist()))

    minutes = np.sort(inp["minute"].unique())
    print(f"minute range: {minutes.min()} ~ {minutes.max()}")
    full_range = set(range(minutes.min(), minutes.max() + 1))
    print("minutes missing from the range:", sorted(full_range - set(minutes)))

    # Same set of minutes in every window <=> each window has exactly one row per minute.
    minute_grid = pd.crosstab(inp["sample_id"], inp["minute"])
    print(
        "duplicate (sample_id, minute):", inp.duplicated(["sample_id", "minute"]).sum()
    )
    print("all windows share the same minutes:", bool((minute_grid == 1).all().all()))

    print_section("signal values")
    print(inp[SIGNALS].describe().round(1).to_string())

    print_section("missing values per window")
    missing_per_id = inp[SIGNALS].isna().groupby(inp["sample_id"]).sum()
    window_len = rows_per_id.max()
    summary = pd.DataFrame(
        {
            "missing_row_ratio": inp[SIGNALS].isna().mean().round(4),
            "windows_with_any_gap": (missing_per_id > 0).sum(),
            "windows_fully_missing": (missing_per_id == window_len).sum(),
            # The last row is what persistence and most features lean on.
            "windows_missing_last_minute": inp.loc[
                inp["minute"] == minutes.max(), SIGNALS
            ]
            .isna()
            .sum(),
        }
    )
    print(summary.to_string())

    n_signals_with_gap = (missing_per_id > 0).sum(axis=1)
    print("\nwindows by number of signals with a gap:")
    print(n_signals_with_gap.value_counts().sort_index().to_string())


# %%
# 3. train_target
def report_target(target: pd.DataFrame) -> None:
    print_section("structure")
    print("windows:", target["sample_id"].nunique())
    print("variables:", sorted(target["variable"].unique().tolist()))
    print("horizons:", sorted(target["horizon"].unique().tolist()))

    combos = pd.crosstab(target["sample_id"], [target["variable"], target["horizon"]])
    print("rows per window:", sorted(combos.sum(axis=1).unique().tolist()))
    print(
        "duplicate (sample_id, variable, horizon):",
        target.duplicated(["sample_id", "variable", "horizon"]).sum(),
    )
    print(
        "every window has each (variable, horizon) exactly once:",
        bool((combos == 1).all().all()),
    )

    print_section("value by variable")
    print(target.groupby("variable")["value"].describe().round(1).to_string())

    print_section("value by variable and horizon (mean)")
    print(
        target.pivot_table(
            index="variable", columns="horizon", values="value", aggfunc="mean"
        )
        .round(2)
        .to_string()
    )


# %%
# 4. sample_submission
def report_submission(sample: pd.DataFrame) -> None:
    parsed = parse_submission_id(sample)

    print_section("id format")
    print("ids not matching the pattern:", parsed["sample_id"].isna().sum())
    print("non-zero predictions:", (sample["prediction"] != 0).sum())

    print_section("structure")
    print("windows:", parsed["sample_id"].nunique())
    combos = pd.crosstab(parsed["sample_id"], [parsed["variable"], parsed["horizon"]])
    print("variables:", sorted(parsed["variable"].unique().tolist()))
    print("horizons:", sorted(parsed["horizon"].unique().tolist()))
    print("duplicate ids:", sample["id"].duplicated().sum())
    print(
        "every window has each (variable, horizon) exactly once:",
        bool((combos == 1).all().all()),
    )


# %%
# 5. case_metadata
def report_meta(meta: pd.DataFrame) -> None:
    meta = add_split(meta)

    print_section("split")
    print(meta["split"].value_counts(dropna=False).to_string())
    print("unclassified rows:", meta["split"].isna().sum())
    print(
        "test rows with caseid/subjectid present:",
        meta.loc[meta["split"] == "test", ["caseid", "subjectid"]]
        .notna()
        .any(axis=1)
        .sum(),
    )
    print(
        "train rows with caseid/subjectid missing:",
        meta.loc[meta["split"] == "train", ["caseid", "subjectid"]]
        .isna()
        .any(axis=1)
        .sum(),
    )

    train_meta = meta[meta["split"] == "train"]

    print_section("caseid vs sample_id")
    id_case = (
        train_meta["sample_id"].str.extract(TRAIN_ID_PATTERN)["caseid"].astype(float)
    )
    print(
        "sample_id prefix matches caseid column:",
        bool((id_case == train_meta["caseid"]).all()),
    )

    print_section("subject / case relationship")
    print(
        "cases:",
        train_meta["caseid"].nunique(),
        " subjects:",
        train_meta["subjectid"].nunique(),
    )
    subjects_per_case = train_meta.groupby("caseid")["subjectid"].nunique()
    print("cases linked to more than one subject:", (subjects_per_case > 1).sum())
    cases_per_subject = train_meta.groupby("subjectid")["caseid"].nunique()
    print("cases per subject:")
    print(cases_per_subject.value_counts().sort_index().to_string())
    multi = cases_per_subject[cases_per_subject > 1].index
    print("\nsubjects with multiple cases:")
    print(
        train_meta.loc[train_meta["subjectid"].isin(multi), ["subjectid", "caseid"]]
        .drop_duplicates()
        .sort_values(["subjectid", "caseid"])
        .to_string(index=False)
    )

    print_section("windows per case / subject")
    print(train_meta.groupby("caseid").size().describe().round(1).to_string())
    print()
    print(train_meta.groupby("subjectid").size().describe().round(1).to_string())

    print_section("attributes constant within a case")
    varying = train_meta.groupby("caseid")[["age", "sex", "department"]].nunique().gt(1)
    print(varying.sum().to_string())

    print_section("age values")
    # All values on a 5-year grid would mean age is binned, not exact.
    print("all ages multiples of 5:", bool((meta["age"] % 5 == 0).all()))
    print(meta["age"].value_counts().sort_index().to_string())


# %%
# 6. IDs line up across files
def compare_ids(name_a: str, ids_a, name_b: str, ids_b) -> None:
    a, b = set(ids_a), set(ids_b)
    status = "same" if a == b else "DIFFERENT"
    print(f"{name_a} ({len(a)}) vs {name_b} ({len(b)}): {status}")
    if a != b:
        print(f"  only in {name_a}: {len(a - b)}  e.g. {sorted(a - b)[:5]}")
        print(f"  only in {name_b}: {len(b - a)}  e.g. {sorted(b - a)[:5]}")


def report_id_consistency(data: dict[str, pd.DataFrame]) -> None:
    meta = add_split(data["meta"])
    submission_ids = parse_submission_id(data["sample"])["sample_id"]

    print_section("train")
    compare_ids(
        "train_input",
        data["train_input"]["sample_id"],
        "train_target",
        data["train_target"]["sample_id"],
    )
    compare_ids(
        "train_input",
        data["train_input"]["sample_id"],
        "meta(train)",
        meta.loc[meta["split"] == "train", "sample_id"],
    )

    print_section("test")
    compare_ids(
        "test_input",
        data["test_input"]["sample_id"],
        "sample_submission",
        submission_ids,
    )
    compare_ids(
        "test_input",
        data["test_input"]["sample_id"],
        "meta(test)",
        meta.loc[meta["split"] == "test", "sample_id"],
    )

    print_section("train / test overlap")
    shared = set(data["train_input"]["sample_id"]) & set(
        data["test_input"]["sample_id"]
    )
    print("sample_ids in both train and test:", len(shared))


# %%
# 7. Overlapping windows within a case
def report_window_overlap(train_input: pd.DataFrame, target: pd.DataFrame) -> None:
    clock = add_case_clock(train_input)

    print_section("window spacing within a case")
    ends = clock.drop_duplicates("sample_id")[["caseid", "end"]].sort_values(
        ["caseid", "end"]
    )
    print("gap between consecutive window ends (minutes):")
    print(ends.groupby("caseid")["end"].diff().value_counts().head().to_string())

    long = clock.melt(
        id_vars=["caseid", "t"],
        value_vars=SIGNALS,
        var_name="variable",
        value_name="observed",
    ).dropna()
    n_values = long.groupby(["caseid", "t", "variable"])["observed"].nunique()
    print(
        "\nrepeat observations of the same case-minute:",
        int((long.duplicated(["caseid", "t", "variable"])).sum()),
    )
    print("of those, with conflicting values:", int((n_values > 1).sum()))
    case_minutes = long.drop_duplicates(["caseid", "t", "variable"])
    print("distinct case-minutes:", clock.drop_duplicates(["caseid", "t"]).shape[0])

    print_section("targets visible in another window's input")
    key = target["sample_id"].str.extract(TRAIN_ID_PATTERN).astype(int)
    future = target.assign(caseid=key["caseid"], t=key["end"] + target["horizon"])
    merged = future.merge(case_minutes, on=["caseid", "t", "variable"], how="left")
    merged["found"] = merged["observed"].notna()
    merged["identical"] = np.isclose(merged["value"], merged["observed"])
    print("share found:")
    print(
        merged.pivot_table(
            index="variable", columns="horizon", values="found", aggfunc="mean"
        )
        .round(3)
        .to_string()
    )
    print("\nshare identical, among found:")
    print(
        merged[merged["found"]]
        .pivot_table(
            index="variable", columns="horizon", values="identical", aggfunc="mean"
        )
        .round(3)
        .to_string()
    )


# %%
# 8. Train vs test distribution
def report_train_test_shift(data: dict[str, pd.DataFrame]) -> None:
    meta = add_split(data["meta"])

    print_section("meta")
    print(meta.groupby("split")["age"].describe().round(1).to_string())
    for col in ["sex", "department"]:
        print()
        print(
            pd.crosstab(meta[col], meta["split"], normalize="columns")
            .round(3)
            .to_string()
        )

    print_section("signals")
    both = pd.concat(
        [
            data["train_input"].assign(split="train"),
            data["test_input"].assign(split="test"),
        ]
    )
    print("missing row ratio:")
    print(
        both.groupby("split")[SIGNALS]
        .apply(lambda g: g.isna().mean())
        .round(4)
        .to_string()
    )
    print("\nmean:")
    print(both.groupby("split")[SIGNALS].mean().round(2).to_string())


# %%
# 9. Small processed files used by other scripts
def save_processed(data: dict[str, pd.DataFrame]) -> None:
    PROCESS_DIR.mkdir(parents=True, exist_ok=True)
    parse_submission_id(data["sample"]).to_csv(SAVE_SUBMISSION_ID, index=False)
    add_split(data["meta"]).to_csv(SAVE_META, index=False)
    print(f"Saved {SAVE_SUBMISSION_ID.relative_to(PROJECT_ROOT)}")
    print(f"Saved {SAVE_META.relative_to(PROJECT_ROOT)}")


# %%
def main() -> None:
    data = load_raw()

    for name in ["train_target", "train_input", "test_input", "meta", "sample"]:
        print_header(f"{name}: common")
        report_common(data[name])

    print_header("train_input: windows")
    report_input(data["train_input"])
    print_header("test_input: windows")
    report_input(data["test_input"])
    print_header("train_target")
    report_target(data["train_target"])
    print_header("sample_submission")
    report_submission(data["sample"])
    print_header("case_metadata")
    report_meta(data["meta"])
    print_header("ID consistency across files")
    report_id_consistency(data)
    print_header("Window overlap (train)")
    report_window_overlap(data["train_input"], data["train_target"])
    print_header("Train vs test")
    report_train_test_shift(data)
    print_header("Save")
    save_processed(data)


if __name__ == "__main__":
    main()
