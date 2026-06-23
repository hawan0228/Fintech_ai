from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    DT_PREDICTIONS_PATH,
    TOP_K_LIST,
    RANDOM_BENCHMARK_N_RUNS,
    RANDOM_SEED,
    DT_SELECTED_STOCKS_PATH,
    DT_PORTFOLIO_RETURNS_PATH,
    DT_PORTFOLIO_METRICS_PATH,
    ALL_STOCK_BENCHMARK_RETURNS_PATH,
    ALL_STOCK_BENCHMARK_METRICS_PATH,
    RANDOM_TOPK_BENCHMARK_RUNS_PATH,
    RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH,
    RANDOM_TOPK_BENCHMARK_SUMMARY_PATH,
    STEP4_PORTFOLIO_PROFILE_PATH,
)
from src.schema import STOCK_ID_COL, YEAR_COL
from src.portfolio import (
    select_top_k_stocks,
    calculate_portfolio_returns,
)
from src.benchmark import (
    calculate_all_stock_benchmark,
    calculate_random_topk_benchmark_runs,
    summarize_random_benchmark_annual_mean,
)
from src.metrics import summarize_strategy_metrics


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_text(text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def summarize_random_benchmark_distribution(
    random_runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    對每個 random_run、top_k 先計算整段回測績效，
    再彙總 random benchmark 的平均、分位數。
    """
    run_metrics = summarize_strategy_metrics(
        returns_df=random_runs_df,
        group_cols=["benchmark_name", "top_k", "random_run"],
    )

    summary_records = []

    for (benchmark_name, top_k), group_df in run_metrics.groupby(
        ["benchmark_name", "top_k"]
    ):
        record = {
            "benchmark_name": benchmark_name,
            "top_k": int(top_k),
            "n_runs": group_df["random_run"].nunique(),
        }

        metric_cols = [
            "gross_cumulative_return",
            "gross_annualized_return",
            "gross_maximum_drawdown",
            "gross_volatility",
            "gross_sharpe_ratio",
            "gross_win_rate",
            "net_cumulative_return",
            "net_annualized_return",
            "net_maximum_drawdown",
            "net_volatility",
            "net_sharpe_ratio",
            "net_win_rate",
        ]

        for col in metric_cols:
            record[f"{col}_mean"] = group_df[col].mean()
            record[f"{col}_median"] = group_df[col].median()
            record[f"{col}_p5"] = group_df[col].quantile(0.05)
            record[f"{col}_p95"] = group_df[col].quantile(0.95)

        summary_records.append(record)

    return pd.DataFrame(summary_records)


def generate_step4_profile(
    selected_df: pd.DataFrame,
    selected_with_weights_df: pd.DataFrame,
    portfolio_returns_df: pd.DataFrame,
    portfolio_metrics_df: pd.DataFrame,
    all_stock_returns_df: pd.DataFrame,
    all_stock_metrics_df: pd.DataFrame,
    random_runs_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
) -> str:
    lines = []

    lines.append("========== Step 4 Portfolio Construction Profile ==========")
    lines.append("")

    lines.append("Portfolio construction assumptions:")
    lines.append("- Stock selection rule: rank by score_label_1 = P(label=1)")
    lines.append(f"- Top-K values: {TOP_K_LIST}")
    lines.append("- Main strategy: Equal-weight portfolio")
    lines.append("- Extension: Score-weighted portfolio")
    lines.append("- Rebalancing rule: annual rebalancing")
    lines.append("- Position rule: long-only, no short-selling, no leverage")
    lines.append("- Return unit: actual_return is percentage, divided by 100 in portfolio calculation")
    lines.append("- Transaction cost: round-trip cost = buy fee + sell fee + sell tax")
    lines.append("")

    lines.append("Selected stocks summary:")
    lines.append(f"- Selected stock rows without weight expansion: {len(selected_df)}")
    lines.append(f"- Selected stock rows with weight methods: {len(selected_with_weights_df)}")
    lines.append("")

    lines.append("Portfolio annual returns summary:")
    lines.append(f"- Portfolio annual return rows: {len(portfolio_returns_df)}")
    lines.append(f"- Testing years: {portfolio_returns_df[YEAR_COL].min()} - {portfolio_returns_df[YEAR_COL].max()}")
    lines.append(f"- Number of testing years: {portfolio_returns_df[YEAR_COL].nunique()}")
    lines.append("")

    lines.append("Portfolio metrics:")
    display_cols = [
        "model_name",
        "top_k",
        "weight_method",
        "n_years",
        "gross_annualized_return",
        "gross_cumulative_return",
        "gross_maximum_drawdown",
        "gross_volatility",
        "gross_sharpe_ratio",
        "gross_win_rate",
        "net_annualized_return",
        "net_cumulative_return",
        "net_maximum_drawdown",
        "net_volatility",
        "net_sharpe_ratio",
        "net_win_rate",
    ]

    lines.append(portfolio_metrics_df[display_cols].to_string(index=False))
    lines.append("")

    lines.append("All-stock benchmark metrics:")
    lines.append(all_stock_metrics_df.to_string(index=False))
    lines.append("")

    lines.append("Random benchmark summary:")
    lines.append(
        random_summary_df[
            [
                "benchmark_name",
                "top_k",
                "n_runs",
                "gross_annualized_return_mean",
                "gross_annualized_return_p5",
                "gross_annualized_return_p95",
                "net_annualized_return_mean",
                "net_annualized_return_p5",
                "net_annualized_return_p95",
            ]
        ].to_string(index=False)
    )
    lines.append("")

    lines.append("Generated files:")
    lines.append(f"- Selected stocks: {DT_SELECTED_STOCKS_PATH}")
    lines.append(f"- Portfolio annual returns: {DT_PORTFOLIO_RETURNS_PATH}")
    lines.append(f"- Portfolio metrics: {DT_PORTFOLIO_METRICS_PATH}")
    lines.append(f"- All-stock benchmark annual returns: {ALL_STOCK_BENCHMARK_RETURNS_PATH}")
    lines.append(f"- All-stock benchmark metrics: {ALL_STOCK_BENCHMARK_METRICS_PATH}")
    lines.append(f"- Random benchmark runs: {RANDOM_TOPK_BENCHMARK_RUNS_PATH}")
    lines.append(f"- Random benchmark annual mean: {RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH}")
    lines.append(f"- Random benchmark summary: {RANDOM_TOPK_BENCHMARK_SUMMARY_PATH}")
    lines.append("")

    lines.append("========== End of Step 4 Profile ==========")

    return "\n".join(lines)


def main() -> None:
    print("========== Step 4: Top-K Portfolio Construction ==========")

    print(f"[Info] Loading predictions from: {DT_PREDICTIONS_PATH}")
    predictions_df = pd.read_csv(DT_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] Predictions shape: {predictions_df.shape}")

    print(f"[Info] Loading cleaned data from: {CLEANED_DATA_PATH}")
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] Cleaned data shape: {cleaned_df.shape}")

    testing_years = sorted(predictions_df[YEAR_COL].astype(int).unique().tolist())
    print(f"[Info] Testing years: {testing_years}")

    print("")
    print("[Info] Selecting Top-K stocks...")
    selected_df = select_top_k_stocks(
        predictions_df=predictions_df,
        top_k_list=TOP_K_LIST,
        score_col="score_label_1",
    )

    print("[Info] Calculating portfolio returns...")
    selected_with_weights_df, portfolio_returns_df = calculate_portfolio_returns(
        selected_df=selected_df,
        weight_methods=["equal", "score"],
        score_col="score_label_1",
    )

    print("[Info] Calculating portfolio metrics...")
    portfolio_metrics_df = summarize_strategy_metrics(
        returns_df=portfolio_returns_df,
        group_cols=["model_name", "top_k", "weight_method"],
    )

    print("")
    print("[Info] Building all-stock benchmark...")
    all_stock_returns_df = calculate_all_stock_benchmark(
        cleaned_df=cleaned_df,
        testing_years=testing_years,
    )

    all_stock_metrics_df = summarize_strategy_metrics(
        returns_df=all_stock_returns_df,
        group_cols=["benchmark_name"],
    )

    print("[Info] Building random Top-K benchmark...")
    random_runs_df = calculate_random_topk_benchmark_runs(
        cleaned_df=cleaned_df,
        testing_years=testing_years,
        top_k_list=TOP_K_LIST,
        n_runs=RANDOM_BENCHMARK_N_RUNS,
        random_seed=RANDOM_SEED,
    )

    random_annual_mean_df = summarize_random_benchmark_annual_mean(random_runs_df)

    random_summary_df = summarize_random_benchmark_distribution(random_runs_df)

    print("")
    print("[Info] Saving outputs...")
    save_dataframe(selected_with_weights_df, DT_SELECTED_STOCKS_PATH)
    save_dataframe(portfolio_returns_df, DT_PORTFOLIO_RETURNS_PATH)
    save_dataframe(portfolio_metrics_df, DT_PORTFOLIO_METRICS_PATH)

    save_dataframe(all_stock_returns_df, ALL_STOCK_BENCHMARK_RETURNS_PATH)
    save_dataframe(all_stock_metrics_df, ALL_STOCK_BENCHMARK_METRICS_PATH)

    save_dataframe(random_runs_df, RANDOM_TOPK_BENCHMARK_RUNS_PATH)
    save_dataframe(random_annual_mean_df, RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH)
    save_dataframe(random_summary_df, RANDOM_TOPK_BENCHMARK_SUMMARY_PATH)

    profile_text = generate_step4_profile(
        selected_df=selected_df,
        selected_with_weights_df=selected_with_weights_df,
        portfolio_returns_df=portfolio_returns_df,
        portfolio_metrics_df=portfolio_metrics_df,
        all_stock_returns_df=all_stock_returns_df,
        all_stock_metrics_df=all_stock_metrics_df,
        random_runs_df=random_runs_df,
        random_summary_df=random_summary_df,
    )

    save_text(profile_text, STEP4_PORTFOLIO_PROFILE_PATH)

    print("")
    print(profile_text)
    print("")
    print("========== Step 4 Finished Successfully ==========")


if __name__ == "__main__":
    main()