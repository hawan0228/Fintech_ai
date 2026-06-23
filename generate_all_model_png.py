from pathlib import Path

import pandas as pd

from src.config import (
    # Inputs
    DT_PORTFOLIO_RETURNS_PATH,
    TASK2_PORTFOLIO_RETURNS_PATH,
    SVR_GA_PORTFOLIO_RETURNS_PATH,
    ALL_MODELS_PORTFOLIO_METRICS_PATH,
    ALL_STOCK_BENCHMARK_RETURNS_PATH,
    ALL_STOCK_BENCHMARK_METRICS_PATH,
    RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH,
    RANDOM_TOPK_BENCHMARK_SUMMARY_PATH,
    SVR_GA_PREDICTIONS_PATH,

    # Outputs
    STEP6_ALL_MODEL_TOP10_CUMULATIVE_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOP10_ANNUAL_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH,
    STEP6_ALL_MODEL_TOPK_HEATMAP_FIG_PATH,
    STEP6_ALL_MODEL_RETURN_RISK_SCATTER_FIG_PATH,
    STEP6_ALL_MODEL_SHARPE_FIG_PATH,
    STEP6_ALL_MODEL_DRAWDOWN_FIG_PATH,
    STEP6_SVR_PREDICTED_VS_ACTUAL_FIG_PATH,
)

from src.visualization import (
    plot_step6_all_model_top10_cumulative_return,
    plot_step6_all_model_top10_annual_return,
    plot_step6_all_model_top10_net_annualized_return,
    plot_step6_all_model_topk_heatmap,
    plot_step6_all_model_return_risk_scatter,
    plot_step6_all_model_sharpe,
    plot_step6_all_model_drawdown,
    plot_step6_svr_predicted_vs_actual,
)


def require_file(path: str | Path) -> None:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到必要檔案：{path}\n"
            "請確認你已完成 Step 4、Step 5、Step 6，並已重建 all_models_portfolio_metrics.csv。"
        )


def main() -> None:
    print("========== Step 6: Generate All-model Comparison PNGs ==========")

    required_files = [
        DT_PORTFOLIO_RETURNS_PATH,
        TASK2_PORTFOLIO_RETURNS_PATH,
        SVR_GA_PORTFOLIO_RETURNS_PATH,
        ALL_MODELS_PORTFOLIO_METRICS_PATH,
        ALL_STOCK_BENCHMARK_RETURNS_PATH,
        ALL_STOCK_BENCHMARK_METRICS_PATH,
        RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH,
        RANDOM_TOPK_BENCHMARK_SUMMARY_PATH,
        SVR_GA_PREDICTIONS_PATH,
    ]

    for file_path in required_files:
        require_file(file_path)

    print("[Info] Loading portfolio returns...")
    dt_returns_df = pd.read_csv(DT_PORTFOLIO_RETURNS_PATH)
    task2_returns_df = pd.read_csv(TASK2_PORTFOLIO_RETURNS_PATH)
    svr_returns_df = pd.read_csv(SVR_GA_PORTFOLIO_RETURNS_PATH)

    all_portfolio_returns_df = pd.concat(
        [dt_returns_df, task2_returns_df, svr_returns_df],
        ignore_index=True,
    )

    print(f"[Info] all_portfolio_returns_df shape: {all_portfolio_returns_df.shape}")

    print("[Info] Loading portfolio metrics and benchmarks...")
    all_models_metrics_df = pd.read_csv(ALL_MODELS_PORTFOLIO_METRICS_PATH)
    all_stock_returns_df = pd.read_csv(ALL_STOCK_BENCHMARK_RETURNS_PATH)
    all_stock_metrics_df = pd.read_csv(ALL_STOCK_BENCHMARK_METRICS_PATH)
    random_annual_mean_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH)
    random_summary_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_SUMMARY_PATH)
    svr_predictions_df = pd.read_csv(SVR_GA_PREDICTIONS_PATH)

    print(f"[Info] all_models_metrics_df shape: {all_models_metrics_df.shape}")
    print(f"[Info] models: {sorted(all_models_metrics_df['model_name'].unique().tolist())}")

    print("")
    print("========== Generating Step 6 All-model PNGs ==========")

    plot_step6_all_model_top10_cumulative_return(
        all_portfolio_returns_df=all_portfolio_returns_df,
        all_stock_returns_df=all_stock_returns_df,
        random_annual_mean_df=random_annual_mean_df,
        output_path=STEP6_ALL_MODEL_TOP10_CUMULATIVE_RETURN_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP6_ALL_MODEL_TOP10_CUMULATIVE_RETURN_FIG_PATH}")

    plot_step6_all_model_top10_annual_return(
        all_portfolio_returns_df=all_portfolio_returns_df,
        all_stock_returns_df=all_stock_returns_df,
        random_annual_mean_df=random_annual_mean_df,
        output_path=STEP6_ALL_MODEL_TOP10_ANNUAL_RETURN_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP6_ALL_MODEL_TOP10_ANNUAL_RETURN_FIG_PATH}")

    plot_step6_all_model_top10_net_annualized_return(
        all_models_metrics_df=all_models_metrics_df,
        all_stock_metrics_df=all_stock_metrics_df,
        random_summary_df=random_summary_df,
        output_path=STEP6_ALL_MODEL_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP6_ALL_MODEL_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH}")

    plot_step6_all_model_topk_heatmap(
        all_models_metrics_df=all_models_metrics_df,
        output_path=STEP6_ALL_MODEL_TOPK_HEATMAP_FIG_PATH,
        weight_method="equal",
    )
    print(f"[Saved] {STEP6_ALL_MODEL_TOPK_HEATMAP_FIG_PATH}")

    plot_step6_all_model_return_risk_scatter(
        all_models_metrics_df=all_models_metrics_df,
        all_stock_metrics_df=all_stock_metrics_df,
        random_summary_df=random_summary_df,
        output_path=STEP6_ALL_MODEL_RETURN_RISK_SCATTER_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP6_ALL_MODEL_RETURN_RISK_SCATTER_FIG_PATH}")

    plot_step6_all_model_sharpe(
        all_models_metrics_df=all_models_metrics_df,
        output_path=STEP6_ALL_MODEL_SHARPE_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP6_ALL_MODEL_SHARPE_FIG_PATH}")

    plot_step6_all_model_drawdown(
        all_portfolio_returns_df=all_portfolio_returns_df,
        output_path=STEP6_ALL_MODEL_DRAWDOWN_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP6_ALL_MODEL_DRAWDOWN_FIG_PATH}")

    plot_step6_svr_predicted_vs_actual(
        svr_predictions_df=svr_predictions_df,
        output_path=STEP6_SVR_PREDICTED_VS_ACTUAL_FIG_PATH,
    )
    print(f"[Saved] {STEP6_SVR_PREDICTED_VS_ACTUAL_FIG_PATH}")

    print("")
    print("========== Step 6 All-model PNG Generation Finished Successfully ==========")


if __name__ == "__main__":
    main()  