"""First learned model: window features -> gradient boosting on the change from the last value.

Every (variable, horizon) pair gets its own model. Validation uses GroupKFold on subjectid,
and the score is reported next to the persistence baseline on exactly the same rows.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from check import parse_submission_id  # add to the imports at the top

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_INPUT = RAW_DIR / "train_input.parquet"
TRAIN_TARGET = RAW_DIR / "train_target.csv"
TEST_INPUT = RAW_DIR / "test_input.parquet"
META = RAW_DIR / "case_metadata.csv"
SAMPLE_SUBMISSION = RAW_DIR / "sample_submission.csv"

SIGNALS = ["HR", "MBP", "SPO2"]
LOOKBACKS = [5, 10, 30]
N_SPLITS = 5
SEED = 42


# %%
def to_wide(inp: pd.DataFrame, signal: str) -> pd.DataFrame:
    # One row per window, one column per minute, oldest first, so array slicing = "last k minutes".
    return inp.pivot(index="sample_id", columns="minute", values=signal).sort_index(
        axis=1
    )


# %%
def last_valid(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the last non-NaN value per row and how many minutes ago it was observed."""
    observed = ~np.isnan(arr)
    n_cols = arr.shape[1]
    # argmax on the reversed mask finds the most recent observed column.
    minutes_ago = np.argmax(observed[:, ::-1], axis=1).astype(float)
    has_any = observed.any(axis=1)
    rows = np.arange(arr.shape[0])
    values = arr[rows, n_cols - 1 - minutes_ago.astype(int)]
    values[~has_any] = np.nan
    minutes_ago[~has_any] = np.nan
    return values, minutes_ago


# %%
def nan_slope(arr: np.ndarray) -> np.ndarray:
    """Least-squares slope per row (units per minute), ignoring NaNs."""
    x = np.arange(arr.shape[1], dtype=float)
    mask = ~np.isnan(arr)
    n = mask.sum(axis=1)
    x_b = np.where(mask, x, 0.0)
    y_b = np.where(mask, arr, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = x_b.sum(axis=1) / n
        y_mean = y_b.sum(axis=1) / n
        cov = (
            np.where(mask, (x - x_mean[:, None]) * (arr - y_mean[:, None]), 0.0)
        ).sum(axis=1)
        var = (np.where(mask, (x - x_mean[:, None]) ** 2, 0.0)).sum(axis=1)
        slope = cov / var
    slope[n < 3] = np.nan
    return slope


# %%
def build_features(inp: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    index = None
    for signal in SIGNALS:
        wide = to_wide(inp, signal)
        index = wide.index if index is None else index
        arr = wide.reindex(index).to_numpy(dtype=float)

        last, minutes_ago = last_valid(arr)
        feats[f"{signal}_last"] = last
        feats[f"{signal}_minutes_since_obs"] = minutes_ago
        feats[f"{signal}_missing_ratio"] = np.isnan(arr).mean(axis=1)

        for k in LOOKBACKS:
            recent = arr[:, -k:]
            # All-NaN slices are expected (signal missing for the whole lookback); NaN is the right output.
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_k = np.nanmean(recent, axis=1)
                feats[f"{signal}_mean_{k}"] = mean_k
                feats[f"{signal}_std_{k}"] = np.nanstd(recent, axis=1)
                feats[f"{signal}_min_{k}"] = np.nanmin(recent, axis=1)
                feats[f"{signal}_max_{k}"] = np.nanmax(recent, axis=1)
            # Distance from the recent mean captures short-term deviation that tends to revert.
            feats[f"{signal}_last_minus_mean_{k}"] = last - mean_k
            if k >= 10:
                feats[f"{signal}_slope_{k}"] = nan_slope(recent)

    features = pd.DataFrame(feats, index=index)

    meta_cols = meta.set_index("sample_id")[["age", "sex", "department"]]
    features = features.join(meta_cols, how="left")
    # Categorical dtype lets HistGradientBoosting split on categories natively.
    for col in ["sex", "department"]:
        features[col] = features[col].astype("category")
    return features


# %%
def make_model() -> HistGradientBoostingRegressor:
    # absolute_error matches the MAE used to judge the baseline; swap if the competition metric differs.
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        categorical_features="from_dtype",
        random_state=SEED,
    )


# %%
def cross_validate(
    features: pd.DataFrame, target: pd.DataFrame, groups: pd.Series
) -> pd.DataFrame:
    rows = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for (variable, horizon), part in target.groupby(["variable", "horizon"]):
        y = part.set_index("sample_id")["value"].reindex(features.index)
        base = features[f"{variable}_last"]
        delta = y - base
        oof = pd.Series(np.nan, index=features.index)

        for train_idx, valid_idx in gkf.split(features, delta, groups=groups):
            model = make_model()
            model.fit(features.iloc[train_idx], delta.iloc[train_idx])
            oof.iloc[valid_idx] = base.iloc[valid_idx] + model.predict(
                features.iloc[valid_idx]
            )

        rows.append(
            {
                "variable": variable,
                "horizon": horizon,
                "persistence_mae": (y - base).abs().mean(),
                "model_mae": (y - oof).abs().mean(),
            }
        )
        print(
            f"{variable:>4} h{horizon:<2}  persistence={rows[-1]['persistence_mae']:.3f}  "
            f"model={rows[-1]['model_mae']:.3f}"
        )

    result = pd.DataFrame(rows)
    result["improvement_pct"] = 100 * (
        1 - result["model_mae"] / result["persistence_mae"]
    )
    return result


# %%
def fit_and_predict(
    train_features: pd.DataFrame,
    target: pd.DataFrame,
    test_features: pd.DataFrame,
    fallback: dict[str, float],
) -> pd.DataFrame:
    preds = []
    for (variable, horizon), part in target.groupby(["variable", "horizon"]):
        y = part.set_index("sample_id")["value"].reindex(train_features.index)
        model = make_model()
        model.fit(train_features, y - train_features[f"{variable}_last"])

        # A test window can lack a signal entirely; fall back to the train median for that variable.
        base = test_features[f"{variable}_last"].fillna(fallback[variable])
        pred = base + model.predict(test_features)
        preds.append(
            pd.DataFrame(
                {
                    "id": test_features.index + f"_{variable}_h{horizon}",
                    "prediction": pred.to_numpy(),
                }
            )
        )
    return pd.concat(preds, ignore_index=True)


# %%
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


# %%
def plot_cv_result(result: pd.DataFrame, path: Path) -> None:
    set_korean_font()
    variables = sorted(result["variable"].unique())
    fig, axes = plt.subplots(1, len(variables), figsize=(4.2 * len(variables), 3.8))
    for ax, variable in zip(np.atleast_1d(axes), variables):
        part = result[result["variable"] == variable].sort_values("horizon")
        ax.plot(
            part["horizon"],
            part["persistence_mae"],
            marker="o",
            color="#9aa0a6",
            label="Persistence",
        )
        ax.plot(
            part["horizon"],
            part["model_mae"],
            marker="o",
            color="#1a73e8",
            label="모델 (HGB)",
        )
        ax.set_xticks(part["horizon"])
        ax.set_title(variable)
        ax.set_xlabel("예측 시점 (분)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("MAE (GroupKFold OOF)")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle(
        "변수·예측 시점별 MAE: persistence vs HGB 모델 (train, subjectid 기준 5-fold)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# %%
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_input = pd.read_parquet(TRAIN_INPUT)
    target = pd.read_csv(TRAIN_TARGET)
    meta = pd.read_csv(META)

    train_features = build_features(train_input, meta)
    groups = meta.set_index("sample_id")["subjectid"].reindex(train_features.index)
    if groups.isna().any():
        raise ValueError("Some train windows have no subjectid; GroupKFold would leak.")
    print(
        f"Train windows: {len(train_features)}, features: {train_features.shape[1]}, "
        f"subjects: {groups.nunique()}"
    )

    result = cross_validate(train_features, target, groups)
    print("\n" + result.round(3).to_string(index=False))
    print(
        f"\nOverall MAE  persistence={result['persistence_mae'].mean():.3f}  "
        f"model={result['model_mae'].mean():.3f}"
    )
    result.to_csv(OUTPUT_DIR / "cv_result_v1.csv", index=False)
    plot_cv_result(result, OUTPUT_DIR / "cv_result_v1.png")

    if not TEST_INPUT.exists():
        print(f"\n{TEST_INPUT.name} not found; skipping submission.")
        return

    test_features = build_features(pd.read_parquet(TEST_INPUT), meta)
    fallback = target.groupby("variable")["value"].median().to_dict()
    submission = fit_and_predict(train_features, target, test_features, fallback)

    sample = pd.read_csv(SAMPLE_SUBMISSION)
    ours, official = set(submission["id"]), set(sample["id"])
    if ours != official:
        only_ours, only_official = sorted(ours - official), sorted(official - ours)
        raise ValueError(
            "Submission ids do not match sample_submission.\n"
            f"  file checked: {SAMPLE_SUBMISSION}\n"
            f"  ours: {len(ours)}, official: {len(official)}\n"
            f"  only in ours ({len(only_ours)}): {only_ours[:5]}\n"
            f"  only in official ({len(only_official)}): {only_official[:5]}"
        )
    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission.to_csv(OUTPUT_DIR / "submission_v1.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR / 'submission_v1.csv'}")


if __name__ == "__main__":
    main()
