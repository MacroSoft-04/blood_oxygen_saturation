"""v2: same features and model as v1, trained on windows cut from each case's full timeline.

Train windows of one case overlap by 20 minutes (30-minute windows every 10 minutes), so
stitching them gives a continuous per-minute timeline. Cutting a window at every minute
instead of every 10 multiplies the training rows. Validation still scores only the original
windows of held-out subjects, so the CV number stays comparable with v1 and with the test set.
"""

# %%
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import (
    META,
    OUTPUT_DIR,
    SAMPLE_SUBMISSION,
    SIGNALS,
    TEST_INPUT,
    TRAIN_INPUT,
    TRAIN_TARGET,
    parse_submission_id,
)
from vs_model_v1 import N_SPLITS, build_features, make_model

TRAIN_ID_PATTERN = r"c(?P<caseid>\d+)_o(?P<end>\d+)"


# %%
def build_case_timeline(
    train_input: pd.DataFrame, target: pd.DataFrame
) -> pd.DataFrame:
    """One row per (caseid, t) with HR/MBP/SPO2, t being the case-level minute."""
    key = train_input["sample_id"].str.extract(TRAIN_ID_PATTERN).astype(int)
    # o### is the case minute of the window's minute 0 (checked in check_overlap: 100% identical).
    from_input = train_input.assign(
        caseid=key["caseid"], t=key["end"] + train_input["minute"]
    )
    from_input = from_input.drop_duplicates(["caseid", "t"])[["caseid", "t", *SIGNALS]]

    # Targets after the last window of a case are real observations the inputs never show.
    tkey = target["sample_id"].str.extract(TRAIN_ID_PATTERN).astype(int)
    from_target = (
        target.assign(caseid=tkey["caseid"], t=tkey["end"] + target["horizon"])
        .drop_duplicates(["caseid", "t", "variable"])
        .pivot(index=["caseid", "t"], columns="variable", values="value")
        .reset_index()
    )

    timeline = (
        pd.concat([from_input, from_target])
        .groupby(["caseid", "t"], as_index=False)[SIGNALS]
        .first()
    )
    return timeline.sort_values(["caseid", "t"], ignore_index=True)


def cut_windows(
    timeline: pd.DataFrame, horizons: list[int], window: int = 30, stride: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut every possible window from each case's timeline.

    Returns (input rows shaped like train_input, MBP targets shaped like train_target).
    A window is kept only if all of its minutes exist on the timeline.
    """
    inputs, targets = [], []
    offsets = np.arange(-(window - 1), 1)

    for caseid, case in timeline.groupby("caseid", sort=False):
        case = case.set_index("t")
        full_t = np.arange(case.index.min(), case.index.max() + 1)
        case = case.reindex(full_t)
        present = case[SIGNALS].notna().any(axis=1).to_numpy()

        # A window needs every minute present on the timeline (values may still be NaN).
        run = np.convolve(present, np.ones(window, dtype=int), mode="valid") == window
        ends = full_t[window - 1 :][run][::stride]
        if len(ends) == 0:
            continue

        ids = np.array([f"c{caseid}_a{e}" for e in ends])
        idx = (ends[:, None] + offsets[None, :]) - full_t[0]
        block = case[SIGNALS].to_numpy()[idx.ravel()]
        inputs.append(
            pd.DataFrame(
                {
                    "sample_id": np.repeat(ids, window),
                    "minute": np.tile(offsets, len(ends)),
                    **{s: block[:, i] for i, s in enumerate(SIGNALS)},
                }
            )
        )

        mbp = case["MBP"]
        for h in horizons:
            value = mbp.reindex(ends + h).to_numpy()
            targets.append(
                pd.DataFrame(
                    {"sample_id": ids, "variable": "MBP", "horizon": h, "value": value}
                )
            )

    window_input = pd.concat(inputs, ignore_index=True)
    window_target = pd.concat(targets, ignore_index=True).dropna(subset=["value"])
    return window_input, window_target


def window_meta(window_ids: pd.Index, meta: pd.DataFrame) -> pd.DataFrame:
    """Give augmented windows the metadata of their case."""
    case_meta = (
        meta.dropna(subset=["caseid"])
        .drop_duplicates("caseid")
        .assign(caseid=lambda d: d["caseid"].astype(int))
        .set_index("caseid")[["subjectid", "age", "sex", "department"]]
    )
    caseid = window_ids.str.extract(r"c(\d+)_")[0].astype(int)
    out = case_meta.reindex(caseid.to_numpy()).reset_index(drop=True)
    out.insert(0, "sample_id", window_ids)
    return out


# %%
def cross_validate_v2(
    orig_features: pd.DataFrame,
    orig_target: pd.DataFrame,
    orig_groups: pd.Series,
    aug_features: pd.DataFrame,
    aug_target: pd.DataFrame,
    aug_groups: pd.Series,
    horizons: list[int],
) -> pd.DataFrame:
    rows = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(orig_features, groups=orig_groups))

    for h in horizons:
        y_orig = (
            orig_target.query("variable == 'MBP' and horizon == @h")
            .set_index("sample_id")["value"]
            .reindex(orig_features.index)
        )
        base_orig = orig_features["MBP_last"]
        y_aug = aug_target.query("horizon == @h").set_index("sample_id")["value"]
        aug_h = aug_features.loc[aug_features.index.intersection(y_aug.index)]
        y_aug = y_aug.reindex(aug_h.index)
        groups_h = aug_groups.reindex(aug_h.index)

        oof = pd.Series(np.nan, index=orig_features.index)
        for train_idx, valid_idx in folds:
            valid_subjects = set(orig_groups.iloc[valid_idx])
            # Drop every augmented window of a validation subject, or its future would leak in.
            train_mask = ~groups_h.isin(valid_subjects)
            model = make_model()
            model.fit(aug_h[train_mask], (y_aug - aug_h["MBP_last"])[train_mask])
            valid = orig_features.iloc[valid_idx]
            oof.iloc[valid_idx] = base_orig.iloc[valid_idx] + model.predict(valid)

        rows.append(
            {
                "horizon": h,
                "train_rows": len(aug_h),
                "persistence_mae": (y_orig - base_orig).abs().mean(),
                "model_mae": (y_orig - oof).abs().mean(),
            }
        )
        print(
            f"MBP h{h:<2}  train_rows={len(aug_h):>6}  "
            f"persistence={rows[-1]['persistence_mae']:.3f}  model={rows[-1]['model_mae']:.3f}"
        )
    return pd.DataFrame(rows)


# %%
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stride", type=int, default=1, help="minutes between cut windows"
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    started = time.time()

    train_input = pd.read_parquet(TRAIN_INPUT)
    target = pd.read_csv(TRAIN_TARGET)
    meta = pd.read_csv(META)
    sample = pd.read_csv(SAMPLE_SUBMISSION) if SAMPLE_SUBMISSION.exists() else None
    horizons = sorted(
        target.loc[target["variable"] == "MBP", "horizon"].unique().tolist()
    )

    timeline = build_case_timeline(train_input, target)
    aug_input, aug_target = cut_windows(timeline, horizons, stride=args.stride)
    aug_ids = pd.Index(aug_input["sample_id"].unique())
    aug_meta = window_meta(aug_ids, meta)
    print(
        f"Timeline rows: {len(timeline)}  augmented windows: {len(aug_ids)} "
        f"(original: {target['sample_id'].nunique()}, stride={args.stride})"
    )

    aug_features = build_features(aug_input, aug_meta)
    aug_groups = aug_meta.set_index("sample_id")["subjectid"]

    orig_features = build_features(train_input, meta)
    orig_groups = meta.set_index("sample_id")["subjectid"].reindex(orig_features.index)

    result = cross_validate_v2(
        orig_features,
        target,
        orig_groups,
        aug_features,
        aug_target,
        aug_groups,
        horizons,
    )
    print(f"\n{result.round(3).to_string(index=False)}")
    print(
        f"\nMBP MAE  persistence={result['persistence_mae'].mean():.3f}  "
        f"v2={result['model_mae'].mean():.3f}"
    )
    result.to_csv(OUTPUT_DIR / f"cv_result_v2_stride{args.stride}.csv", index=False)
    print(f"Elapsed: {time.time() - started:.0f}s")

    if not TEST_INPUT.exists() or sample is None:
        print("\nTest files not found; skipping submission.")
        return

    test_features = build_features(pd.read_parquet(TEST_INPUT), meta)
    fallback = target.loc[target["variable"] == "MBP", "value"].median()
    base = test_features["MBP_last"].fillna(fallback)
    preds = []
    for h in horizons:
        y = aug_target.query("horizon == @h").set_index("sample_id")["value"]
        x = aug_features.loc[aug_features.index.intersection(y.index)]
        model = make_model()
        model.fit(x, y.reindex(x.index) - x["MBP_last"])
        preds.append(
            pd.DataFrame(
                {
                    "id": test_features.index + f"_MBP_h{h}",
                    "prediction": base + model.predict(test_features),
                }
            )
        )
    submission = sample[["id"]].merge(pd.concat(preds), on="id", how="left")
    if submission["prediction"].isna().any() or len(submission) != len(sample):
        raise ValueError("Submission does not cover every id in sample_submission.")
    submission.to_csv(OUTPUT_DIR / "submission_v2.csv", index=False)
    print(f"\nSaved outputs/submission_v2.csv ({len(submission)} rows)")


if __name__ == "__main__":
    main()
