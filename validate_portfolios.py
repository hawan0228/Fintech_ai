import pandas as pd

from src.config import (
    TOP_K_LIST,
    RANDOM_BENCHMARK_N_RUNS,
    DT_SELECTED_STOCKS_PATH,
    DT_PORTFOLIO_RETURNS_PATH,
    DT_PORTFOLIO_METRICS_PATH,
    ALL_STOCK_BENCHMARK_RETURNS_PATH,
    ALL_STOCK_BENCHMARK_METRICS_PATH,
    RANDOM_TOPK_BENCHMARK_RUNS_PATH,
    RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH,
    RANDOM_TOPK_BENCHMARK_SUMMARY_PATH,
)
from src.schema import STOCK_ID_COL, YEAR_COL


def main() -> None:
    print("========== Step 4 Final Validation ==========")

    selected_df = pd.read_csv(DT_SELECTED_STOCKS_PATH, dtype={STOCK_ID_COL: str})
    portfolio_returns_df = pd.read_csv(DT_PORTFOLIO_RETURNS_PATH)
    portfolio_metrics_df = pd.read_csv(DT_PORTFOLIO_METRICS_PATH)

    all_stock_returns_df = pd.read_csv(ALL_STOCK_BENCHMARK_RETURNS_PATH)
    all_stock_metrics_df = pd.read_csv(ALL_STOCK_BENCHMARK_METRICS_PATH)

    random_runs_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_RUNS_PATH)
    random_annual_mean_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH)
    random_summary_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_SUMMARY_PATH)

    print(f"[Info] selected_df shape: {selected_df.shape}")
    print(f"[Info] portfolio_returns_df shape: {portfolio_returns_df.shape}")
    print(f"[Info] portfolio_metrics_df shape: {portfolio_metrics_df.shape}")
    print(f"[Info] all_stock_returns_df shape: {all_stock_returns_df.shape}")
    print(f"[Info] random_runs_df shape: {random_runs_df.shape}")

    testing_years = list(range(1998, 2009))
    n_years = len(testing_years)
    n_topk_sum = sum(TOP_K_LIST)

    # selected_df 包含 equal 與 score 兩種 weight_method，所以會乘以 2
    expected_selected_rows = n_years * n_topk_sum * 2
    assert len(selected_df) == expected_selected_rows, (
        f"selected rows 應為 {expected_selected_rows}，但目前是 {len(selected_df)}"
    )

    # 每年、每 K、每 weight_method 的 selected 數量必須等於 K
    group_cols = ["model_name", YEAR_COL, "top_k", "weight_method"]
    selected_counts = selected_df.groupby(group_cols).size().reset_index(name="count")

    bad_counts = selected_counts[selected_counts["count"] != selected_counts["top_k"]]
    assert bad_counts.empty, f"以下 portfolio selected count 不等於 top_k：\n{bad_counts}"

    # 權重加總必須接近 1
    weight_sum = (
        selected_df.groupby(group_cols)["weight"]
        .sum()
        .reset_index(name="weight_sum")
    )

    bad_weights = weight_sum[(weight_sum["weight_sum"] - 1.0).abs() > 1e-8]
    assert bad_weights.empty, f"以下 portfolio 權重加總不等於 1：\n{bad_weights}"

    # portfolio annual returns rows
    expected_portfolio_return_rows = n_years * len(TOP_K_LIST) * 2
    assert len(portfolio_returns_df) == expected_portfolio_return_rows, (
        f"portfolio returns rows 應為 {expected_portfolio_return_rows}，"
        f"但目前是 {len(portfolio_returns_df)}"
    )

    assert sorted(portfolio_returns_df[YEAR_COL].unique().tolist()) == testing_years, (
        "portfolio returns testing years 不正確"
    )

    assert portfolio_returns_df["gross_return"].notna().all(), "gross_return 有缺失"
    assert portfolio_returns_df["net_return"].notna().all(), "net_return 有缺失"

    # metrics
    expected_metrics_rows = len(TOP_K_LIST) * 2
    assert len(portfolio_metrics_df) == expected_metrics_rows, (
        f"portfolio metrics rows 應為 {expected_metrics_rows}，"
        f"但目前是 {len(portfolio_metrics_df)}"
    )

    # all-stock benchmark
    assert len(all_stock_returns_df) == n_years, (
        f"all-stock benchmark annual rows 應為 {n_years}，"
        f"但目前是 {len(all_stock_returns_df)}"
    )

    assert len(all_stock_metrics_df) == 1, "all-stock benchmark metrics 應只有 1 筆"

    # random benchmark
    expected_random_rows = RANDOM_BENCHMARK_N_RUNS * n_years * len(TOP_K_LIST)
    assert len(random_runs_df) == expected_random_rows, (
        f"random benchmark runs rows 應為 {expected_random_rows}，"
        f"但目前是 {len(random_runs_df)}"
    )

    expected_random_annual_mean_rows = n_years * len(TOP_K_LIST)
    assert len(random_annual_mean_df) == expected_random_annual_mean_rows, (
        f"random annual mean rows 應為 {expected_random_annual_mean_rows}，"
        f"但目前是 {len(random_annual_mean_df)}"
    )

    assert len(random_summary_df) == len(TOP_K_LIST), (
        f"random summary rows 應為 {len(TOP_K_LIST)}，"
        f"但目前是 {len(random_summary_df)}"
    )

    print("")
    print("[Pass] Step 4 portfolio, benchmark, and metrics outputs are valid.")
    print("========== Step 4 Validation Finished ==========")


if __name__ == "__main__":
    main()