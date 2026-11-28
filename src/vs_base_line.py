"""Persistence baseline: predict that every signal stays at its last observed value.

Needs no training, so it is scored on the full train set. Every later model is judged
by how much it improves on these numbers.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    OUTPUT_DIR,
    SAMPLE_SUBMISSION,
    SIGNALS,
    TEST_INPUT,
    TRAIN_INPUT,
    TRAIN_TARGET,
    parse_submission_id,
    set_korean_font,
)


# %%
def last_observed(inp: pd.DataFrame) -> pd.DataFrame:
    """Long table of (sample_id, variable, last_value, minutes_since_obs)."""
    ordered = inp.sort_values(["sample_id", "minute"])
    # groupby.last() skips NaN, so a window ending in a gap still gets its latest real reading.
    last = ordered.groupby("sample_id")[SIGNALS].last()

    last_minute = ordered["minute"].max()
    observed_minute = (
        pd.DataFrame({s: ordered["minute"].where(ordered[s].notna()) for s in SIGNALS})
        .groupby(ordered["sample_id"])
        .max()
    )
    minutes_since = last_minute - observed_minute

    return (
        last.reset_index()
        .melt(id_vars="sample_id", var_name="variable", value_name="last_value")
        .merge(
            minutes_since.reset_index().melt(
                id_vars="sample_id", var_name="variable", value_name="minutes_since_obs"
            ),
            on=["sample_id", "variable"],
        )
    )


def attach_persistence(target: pd.DataFrame, inp: pd.DataFrame) -> pd.DataFrame:
    df = target.merge(
        last_observed(inp),
        on=["sample_id", "variable"],
        how="left",
        validate="many_to_one",
    )
    df["error"] = df["value"] - df["last_value"]
    df["abs_error"] = df["error"].abs()
    return df


# %%
def report_persistence(df: pd.DataFrame) -> pd.DataFrame:
    print("targets with no observed value in window:", df["last_value"].isna().sum())

    def table(values: str, aggfunc) -> pd.DataFrame:
        return df.pivot_table(
            index="variable", columns="horizon", values=values, aggfunc=aggfunc
        )

    mae = table("abs_error", "mean")
    print("\n--- MAE ---")
    print(mae.round(3).to_string())
    print(f"\noverall MAE (mean over variable x horizon): {mae.to_numpy().mean():.3f}")

    print("\n--- RMSE ---")
    print(table("error", lambda e: np.sqrt((e**2).mean())).round(3).to_string())

    # Large mean error at long horizons would mean signals drift in one direction, not just wander.
    print("\n--- mean error (target - last value) ---")
    print(table("error", "mean").round(3).to_string())

    print("\n--- share of targets exactly equal to the last value ---")
    print(table("abs_error", lambda e: (e < 1e-6).mean()).round(3).to_string())

    print("\n--- MAE by how stale the last value is ---")
    stale = df.assign(
        stale=df["minutes_since_obs"].gt(0).map({False: "fresh", True: "stale"})
    )
    print(
        stale.pivot_table(
            index="variable",
            columns="stale",
            values="abs_error",
            aggfunc=["mean", "size"],
        )
        .round(3)
        .to_string()
    )
    return mae


# %%
def plot_persistence_mae(mae: pd.DataFrame, path) -> None:
    set_korean_font()
    variables = mae.index.tolist()
    fig, axes = plt.subplots(1, len(variables), figsize=(4.2 * len(variables), 3.6))
    for ax, variable in zip(np.atleast_1d(axes), variables):
        row = mae.loc[variable]
        ax.plot(row.index, row.values, marker="o", color="#1a73e8")
        ax.set_xticks(row.index)
        ax.set_title(variable)
        ax.set_xlabel("예측 시점 (분)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("MAE")
    fig.suptitle("변수·예측 시점별 persistence baseline MAE (train 전체)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# %%
def make_persistence_submission(
    test_input: pd.DataFrame, sample: pd.DataFrame, fallback: dict[str, float]
) -> pd.DataFrame:
    parsed = parse_submission_id(sample)
    pred = parsed.merge(
        last_observed(test_input),
        on=["sample_id", "variable"],
        how="left",
        validate="many_to_one",
    )
    missing = pred["last_value"].isna()
    print(
        "test targets with no observed value (filled with train median):", missing.sum()
    )
    pred["prediction"] = pred["last_value"].fillna(pred["variable"].map(fallback))

    # parse_submission_id keeps row order, so ids can be taken straight from the sample file.
    submission = pd.DataFrame(
        {"id": sample["id"], "prediction": pred["prediction"].to_numpy()}
    )
    if submission["prediction"].isna().any():
        raise ValueError("Submission still has missing predictions.")
    return submission


# %%
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_input = pd.read_parquet(TRAIN_INPUT)
    target = pd.read_csv(TRAIN_TARGET)

    df = attach_persistence(target, train_input)
    mae = report_persistence(df)
    mae.to_csv(OUTPUT_DIR / "persistence_mae.csv")
    plot_persistence_mae(mae, OUTPUT_DIR / "persistence_mae.png")

    if not (TEST_INPUT.exists() and SAMPLE_SUBMISSION.exists()):
        print("\nTest files not found; skipping submission.")
        return

    fallback = target.groupby("variable")["value"].median().to_dict()
    submission = make_persistence_submission(
        pd.read_parquet(TEST_INPUT), pd.read_csv(SAMPLE_SUBMISSION), fallback
    )
    submission.to_csv(OUTPUT_DIR / "submission_persistence.csv", index=False)
    print(f"\nSaved outputs/submission_persistence.csv ({len(submission)} rows)")


if __name__ == "__main__":
    main()
