from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    EXTERNAL_OUTPUT_DIR,
    EXTERNAL_RF_PORTFOLIO_RETURNS_PATH,
    EXTERNAL_RF_PORTFOLIO_METRICS_PATH,
)
from src.schema import YEAR_COL
from src.metrics import summarize_strategy_metrics


EXTERNAL_BENCHMARK_DIR = EXTERNAL_OUTPUT_DIR / "benchmarks"

EXTERNAL_ALL_STOCK_RETURNS_PATH = (
    EXTERNAL_BENCHMARK_DIR / "external_all_stock_returns.csv"
)

EXTERNAL_RANDOM_RUNS_PATH = (
    EXTERNAL_BENCHMARK_DIR / "external_random_topk_runs.csv"
)

EXTERNAL_ALL_STOCK_METRICS_ALIGNED_PATH = (
    EXTERNAL_BENCHMARK_DIR / "external_all_stock_metrics_aligned.csv"
)

EXTERNAL_RANDOM_SUMMARY_ALIGNED_PATH = (
    EXTERNAL_BENCHMARK_DIR / "external_random_topk_summary_aligned.csv"
)

EXTERNAL_RF_BENCHMARK_COMPARISON_PATH = (
    EXTERNAL_BENCHMARK_DIR / "external_rf_benchmark_comparison_aligned.csv"
)

EXTERNAL_RF_BENCHMARK_PROFILE_PATH = (
    EXTERNAL_BENCHMARK_DIR / "external_rf_benchmark_comparison_profile.txt"
)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def summarize_random_runs_aligned(random_runs_df: pd.DataFrame) -> pd.DataFrame:
    run_metrics = []

    for (random_run, top_k), group_df in random_runs_df.groupby(
        ["random_run", "top_k"]
    ):
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
            "model_name": "external_random_topk_benchmark_aligned",
            "top_k": int(top_k),
            "n_random_runs": int(group_df["random_run"].nunique()),
            "n_years": int(group_df["n_years"].iloc[0]),
        }

        for col in metric_cols:
            record[f"{col}_mean"] = group_df[col].mean()
            record[f"{col}_median"] = group_df[col].median()
            record[f"{col}_p05"] = group_df[col].quantile(0.05)
            record[f"{col}_p95"] = group_df[col].quantile(0.95)

        summary_records.append(record)

    return pd.DataFrame(summary_records)


def build_comparison_table(
    rf_metrics_df: pd.DataFrame,
    all_stock_metrics_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rf_df = rf_metrics_df.copy()

    all_stock_row = all_stock_metrics_df.iloc[0]

    rf_df["all_stock_net_annualized_return"] = float(
        all_stock_row["net_annualized_return"]
    )
    rf_df["all_stock_net_cumulative_return"] = float(
        all_stock_row["net_cumulative_return"]
    )
    rf_df["all_stock_net_sharpe_ratio"] = float(
        all_stock_row["net_sharpe_ratio"]
    )

    comparison_df = rf_df.merge(
        random_summary_df[
            [
                "top_k",
                "net_annualized_return_mean",
                "net_annualized_return_median",
                "net_annualized_return_p05",
                "net_annualized_return_p95",
                "net_sharpe_ratio_mean",
                "net_sharpe_ratio_p95",
            ]
        ],
        on="top_k",
        how="left",
    )

    comparison_df = comparison_df.rename(
        columns={
            "net_annualized_return_mean": "random_net_annualized_return_mean",
            "net_annualized_return_median": "random_net_annualized_return_median",
            "net_annualized_return_p05": "random_net_annualized_return_p05",
            "net_annualized_return_p95": "random_net_annualized_return_p95",
            "net_sharpe_ratio_mean": "random_net_sharpe_ratio_mean",
            "net_sharpe_ratio_p95": "random_net_sharpe_ratio_p95",
        }
    )

    comparison_df["excess_vs_all_stock"] = (
        comparison_df["net_annualized_return"]
        - comparison_df["all_stock_net_annualized_return"]
    )

    comparison_df["excess_vs_random_mean"] = (
        comparison_df["net_annualized_return"]
        - comparison_df["random_net_annualized_return_mean"]
    )

    comparison_df["above_random_p95"] = (
        comparison_df["net_annualized_return"]
        > comparison_df["random_net_annualized_return_p95"]
    )

    comparison_df["above_random_mean"] = (
        comparison_df["net_annualized_return"]
        > comparison_df["random_net_annualized_return_mean"]
    )

    return comparison_df


def generate_profile(
    test_years: list[int],
    all_stock_metrics_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> str:
    lines = []

    lines.append("========== External RF Benchmark Comparison Profile ==========")
    lines.append("")
    lines.append(f"Aligned testing years: {test_years}")
    lines.append("")
    lines.append("External all-stock benchmark aligned:")
    lines.append(all_stock_metrics_df.to_string(index=False))
    lines.append("")
    lines.append("External random Top-K benchmark aligned:")
    lines.append(random_summary_df.to_string(index=False))
    lines.append("")
    lines.append("External RF vs aligned benchmarks:")
    display_cols = [
        "model_name",
        "top_k",
        "weight_method",
        "n_years",
        "net_annualized_return",
        "all_stock_net_annualized_return",
        "random_net_annualized_return_mean",
        "random_net_annualized_return_p95",
        "excess_vs_all_stock",
        "excess_vs_random_mean",
        "above_random_mean",
        "above_random_p95",
        "net_sharpe_ratio",
        "random_net_sharpe_ratio_mean",
    ]
    lines.append(comparison_df[display_cols].to_string(index=False))
    lines.append("")
    lines.append("Interpretation guideline:")
    lines.append("- above_random_mean=True means the model beats average random selection.")
    lines.append("- above_random_p95=True means the model beats a high-percentile random strategy.")
    lines.append("- If the model only beats random mean for larger K, the result may reflect diversification rather than precise top-stock selection.")
    lines.append("")
    lines.append("========== End Profile ==========")

    return "\n".join(lines)


def main() -> None:
    print("========== Step 7: External RF Benchmark Comparison ==========")

    rf_returns_df = pd.read_csv(EXTERNAL_RF_PORTFOLIO_RETURNS_PATH)
    rf_metrics_df = pd.read_csv(EXTERNAL_RF_PORTFOLIO_METRICS_PATH)

    all_stock_returns_df = pd.read_csv(EXTERNAL_ALL_STOCK_RETURNS_PATH)
    random_runs_df = pd.read_csv(EXTERNAL_RANDOM_RUNS_PATH)

    test_years = sorted(rf_returns_df[YEAR_COL].astype(int).unique().tolist())

    print(f"[Info] RF testing years: {test_years}")

    all_stock_aligned_returns_df = all_stock_returns_df[
        all_stock_returns_df[YEAR_COL].astype(int).isin(test_years)
    ].copy()

    random_aligned_runs_df = random_runs_df[
        random_runs_df[YEAR_COL].astype(int).isin(test_years)
    ].copy()

    all_stock_metrics_aligned_df = summarize_strategy_metrics(
        returns_df=all_stock_aligned_returns_df,
        group_cols=["model_name", "weight_method"],
    )

    random_summary_aligned_df = summarize_random_runs_aligned(
        random_aligned_runs_df
    )

    comparison_df = build_comparison_table(
        rf_metrics_df=rf_metrics_df,
        all_stock_metrics_df=all_stock_metrics_aligned_df,
        random_summary_df=random_summary_aligned_df,
    )

    save_dataframe(
        all_stock_metrics_aligned_df,
        EXTERNAL_ALL_STOCK_METRICS_ALIGNED_PATH,
    )
    save_dataframe(
        random_summary_aligned_df,
        EXTERNAL_RANDOM_SUMMARY_ALIGNED_PATH,
    )
    save_dataframe(
        comparison_df,
        EXTERNAL_RF_BENCHMARK_COMPARISON_PATH,
    )

    profile_text = generate_profile(
        test_years=test_years,
        all_stock_metrics_df=all_stock_metrics_aligned_df,
        random_summary_df=random_summary_aligned_df,
        comparison_df=comparison_df,
    )

    save_text(profile_text, EXTERNAL_RF_BENCHMARK_PROFILE_PATH)

    print("")
    print(profile_text)
    print("")
    print("[Saved]")
    print(f"- {EXTERNAL_ALL_STOCK_METRICS_ALIGNED_PATH}")
    print(f"- {EXTERNAL_RANDOM_SUMMARY_ALIGNED_PATH}")
    print(f"- {EXTERNAL_RF_BENCHMARK_COMPARISON_PATH}")
    print(f"- {EXTERNAL_RF_BENCHMARK_PROFILE_PATH}")
    print("")
    print("========== External RF Benchmark Comparison Finished ==========")


if __name__ == "__main__":
    main()