from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EXTERNAL_CLEANED_DATA_PATH,
    EXTERNAL_OUTPUT_DIR,
    TOP_K_LIST,
    ROUND_TRIP_COST,
)
from src.schema import (
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEAR_COL,
    TARGET_RETURN,
)
from src.metrics import summarize_strategy_metrics


EXTERNAL_BENCHMARK_DIR = EXTERNAL_OUTPUT_DIR / "benchmarks"
EXTERNAL_ALL_STOCK_RETURNS_PATH = EXTERNAL_BENCHMARK_DIR / "external_all_stock_returns.csv"
EXTERNAL_ALL_STOCK_METRICS_PATH = EXTERNAL_BENCHMARK_DIR / "external_all_stock_metrics.csv"
EXTERNAL_RANDOM_RUNS_PATH = EXTERNAL_BENCHMARK_DIR / "external_random_topk_runs.csv"
EXTERNAL_RANDOM_SUMMARY_PATH = EXTERNAL_BENCHMARK_DIR / "external_random_topk_summary.csv"


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def calculate_external_all_stock_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for year, group_df in df.groupby(YEAR_COL):
        gross_return = group_df[TARGET_RETURN].astype(float).mean() / 100.0
        net_return = gross_return - ROUND_TRIP_COST

        records.append(
            {
                "model_name": "external_all_stock_benchmark",
                YEAR_COL: int(year),
                "top_k": int(len(group_df)),
                "weight_method": "equal",
                "n_selected": int(len(group_df)),
                "gross_return": gross_return,
                "net_return": net_return,
                "transaction_cost": ROUND_TRIP_COST,
            }
        )

    return pd.DataFrame(records).sort_values(YEAR_COL).reset_index(drop=True)


def calculate_external_random_topk(
    df: pd.DataFrame,
    requested_top_k_list: list[int],
    n_runs: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    records = []

    min_stocks = df.groupby(YEAR_COL).size().min()
    valid_top_k_list = [int(k) for k in requested_top_k_list if int(k) <= min_stocks]

    if not valid_top_k_list:
        valid_top_k_list = [int(min_stocks)]

    for run_id in range(1, n_runs + 1):
        for year, group_df in df.groupby(YEAR_COL):
            group_df = group_df.copy().reset_index(drop=True)

            for top_k in valid_top_k_list:
                chosen_idx = rng.choice(
                    group_df.index.to_numpy(),
                    size=top_k,
                    replace=False,
                )

                selected = group_df.loc[chosen_idx].copy()

                gross_return = selected[TARGET_RETURN].astype(float).mean() / 100.0
                net_return = gross_return - ROUND_TRIP_COST

                records.append(
                    {
                        "random_run": run_id,
                        "model_name": "external_random_topk_benchmark",
                        YEAR_COL: int(year),
                        "top_k": int(top_k),
                        "weight_method": "equal",
                        "n_selected": int(top_k),
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "transaction_cost": ROUND_TRIP_COST,
                        "selected_tickers": ",".join(selected[STOCK_ID_COL].astype(str).tolist()),
                        "selected_names": ",".join(selected[STOCK_NAME_COL].astype(str).tolist()),
                    }
                )

    return pd.DataFrame(records)


def summarize_external_random_runs(random_runs_df: pd.DataFrame) -> pd.DataFrame:
    run_metrics = []

    for (random_run, top_k), group_df in random_runs_df.groupby(["random_run", "top_k"]):
        metrics_df = summarize_strategy_metrics(
            returns_df=group_df,
            group_cols=["model_name", "top_k", "weight_method"],
        )

        row = metrics_df.iloc[0].to_dict()
        row["random_run"] = int(random_run)
        run_metrics.append(row)

    run_metrics_df = pd.DataFrame(run_metrics)

    summary_records = []

    metric_cols = [
        "net_annualized_return",
        "net_cumulative_return",
        "net_maximum_drawdown",
        "net_volatility",
        "net_sharpe_ratio",
        "net_win_rate",
    ]

    for top_k, group_df in run_metrics_df.groupby("top_k"):
        record = {
            "model_name": "external_random_topk_benchmark",
            "top_k": int(top_k),
            "n_random_runs": int(group_df["random_run"].nunique()),
        }

        for col in metric_cols:
            record[f"{col}_mean"] = group_df[col].mean()
            record[f"{col}_median"] = group_df[col].median()
            record[f"{col}_p05"] = group_df[col].quantile(0.05)
            record[f"{col}_p95"] = group_df[col].quantile(0.95)

        summary_records.append(record)

    return pd.DataFrame(summary_records)


def main() -> None:
    print("========== Step 7: External Benchmarks ==========")

    df = pd.read_csv(EXTERNAL_CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df[TARGET_RETURN] = pd.to_numeric(df[TARGET_RETURN], errors="coerce")

    all_stock_returns_df = calculate_external_all_stock_benchmark(df)
    all_stock_metrics_df = summarize_strategy_metrics(
        returns_df=all_stock_returns_df,
        group_cols=["model_name", "weight_method"],
    )

    random_runs_df = calculate_external_random_topk(
        df=df,
        requested_top_k_list=TOP_K_LIST,
        n_runs=500,
        random_seed=42,
    )

    random_summary_df = summarize_external_random_runs(random_runs_df)

    save_dataframe(all_stock_returns_df, EXTERNAL_ALL_STOCK_RETURNS_PATH)
    save_dataframe(all_stock_metrics_df, EXTERNAL_ALL_STOCK_METRICS_PATH)
    save_dataframe(random_runs_df, EXTERNAL_RANDOM_RUNS_PATH)
    save_dataframe(random_summary_df, EXTERNAL_RANDOM_SUMMARY_PATH)

    print("[Saved]")
    print(f"- {EXTERNAL_ALL_STOCK_RETURNS_PATH}")
    print(f"- {EXTERNAL_ALL_STOCK_METRICS_PATH}")
    print(f"- {EXTERNAL_RANDOM_RUNS_PATH}")
    print(f"- {EXTERNAL_RANDOM_SUMMARY_PATH}")

    print("")
    print("External all-stock metrics:")
    print(all_stock_metrics_df.to_string(index=False))

    print("")
    print("External random benchmark summary:")
    print(random_summary_df.to_string(index=False))

    print("========== External Benchmarks Finished ==========")


if __name__ == "__main__":
    main()