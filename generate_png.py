from pathlib import Path

import pandas as pd

from src.config import *
from src.schema import STOCK_ID_COL
from src.visualization import *

def require_file(path: str | Path) -> None:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到必要檔案：{path}\n"
            f"請確認你已先執行 Step 3 與 Step 4。"
        )


def main() -> None:
    print("========== Generate Step 3 + Step 4 PNG Figures ==========")

    required_files = [
        DT_PREDICTIONS_PATH,
        DT_CLASSIFICATION_METRICS_PATH,
        DT_FEATURE_IMPORTANCE_PATH,
        DT_PORTFOLIO_RETURNS_PATH,
        DT_PORTFOLIO_METRICS_PATH,
        ALL_STOCK_BENCHMARK_RETURNS_PATH,
        ALL_STOCK_BENCHMARK_METRICS_PATH,
        RANDOM_TOPK_BENCHMARK_RUNS_PATH,
        RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH,
        RANDOM_TOPK_BENCHMARK_SUMMARY_PATH,
    ]

    for path in required_files:
        require_file(path)

    print("[Info] Loading Step 3 outputs...")
    predictions_df = pd.read_csv(DT_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str})
    metrics_df = pd.read_csv(DT_CLASSIFICATION_METRICS_PATH)
    feature_importance_df = pd.read_csv(DT_FEATURE_IMPORTANCE_PATH)

    print(f"[Info] predictions_df shape: {predictions_df.shape}")
    print(f"[Info] metrics_df shape: {metrics_df.shape}")
    print(f"[Info] feature_importance_df shape: {feature_importance_df.shape}")

    print("")
    print("[Info] Loading Step 4 outputs...")
    portfolio_returns_df = pd.read_csv(DT_PORTFOLIO_RETURNS_PATH)
    portfolio_metrics_df = pd.read_csv(DT_PORTFOLIO_METRICS_PATH)
    all_stock_returns_df = pd.read_csv(ALL_STOCK_BENCHMARK_RETURNS_PATH)
    all_stock_metrics_df = pd.read_csv(ALL_STOCK_BENCHMARK_METRICS_PATH)
    random_runs_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_RUNS_PATH)
    random_annual_mean_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH)
    random_summary_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_SUMMARY_PATH)

    print(f"[Info] portfolio_returns_df shape: {portfolio_returns_df.shape}")
    print(f"[Info] portfolio_metrics_df shape: {portfolio_metrics_df.shape}")
    print(f"[Info] all_stock_returns_df shape: {all_stock_returns_df.shape}")
    print(f"[Info] random_runs_df shape: {random_runs_df.shape}")

    print("")
    print("========== Generating Step 3 PNGs ==========")

    plot_metrics_by_year(
        metrics_df=metrics_df,
        output_path=DT_METRICS_BY_YEAR_FIG_PATH,
    )
    print(f"[Saved] {DT_METRICS_BY_YEAR_FIG_PATH}")

    plot_confusion_matrix_overall(
        predictions_df=predictions_df,
        output_path=DT_CONFUSION_MATRIX_FIG_PATH,
    )
    print(f"[Saved] {DT_CONFUSION_MATRIX_FIG_PATH}")

    plot_average_feature_importance(
        feature_importance_df=feature_importance_df,
        output_path=DT_FEATURE_IMPORTANCE_AVG_FIG_PATH,
    )
    print(f"[Saved] {DT_FEATURE_IMPORTANCE_AVG_FIG_PATH}")

    plot_feature_importance_heatmap(
        feature_importance_df=feature_importance_df,
        output_path=DT_FEATURE_IMPORTANCE_HEATMAP_FIG_PATH,
        top_n=10,
    )
    print(f"[Saved] {DT_FEATURE_IMPORTANCE_HEATMAP_FIG_PATH}")

    plot_score_boxplot_by_actual_label(
        predictions_df=predictions_df,
        output_path=DT_SCORE_BOXPLOT_FIG_PATH,
    )
    print(f"[Saved] {DT_SCORE_BOXPLOT_FIG_PATH}")

    model_path = (
        Path(SAVED_MODEL_DIR)
        / f"decision_tree_split_{DT_TREE_PLOT_SPLIT_ID:02d}.joblib"
    )

    plot_decision_tree_png(
        model_path=model_path,
        output_path=DT_TREE_SPLIT_11_FIG_PATH,
        max_depth=DT_TREE_PLOT_MAX_DEPTH,
    )
    print(f"[Saved] {DT_TREE_SPLIT_11_FIG_PATH}")

    print("")
    print("========== Generating Step 4 PNGs ==========")

    plot_top10_cumulative_return_comparison(
        portfolio_returns_df=portfolio_returns_df,
        all_stock_returns_df=all_stock_returns_df,
        random_annual_mean_df=random_annual_mean_df,
        output_path=DT_TOP10_CUMULATIVE_RETURN_FIG_PATH,
        top_k=10,
    )
    print(f"[Saved] {DT_TOP10_CUMULATIVE_RETURN_FIG_PATH}")

    plot_top10_annual_return_comparison(
        portfolio_returns_df=portfolio_returns_df,
        all_stock_returns_df=all_stock_returns_df,
        random_annual_mean_df=random_annual_mean_df,
        output_path=DT_TOP10_ANNUAL_RETURN_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {DT_TOP10_ANNUAL_RETURN_FIG_PATH}")

    plot_topk_net_annualized_return(
        portfolio_metrics_df=portfolio_metrics_df,
        random_summary_df=random_summary_df,
        all_stock_metrics_df=all_stock_metrics_df,
        output_path=DT_TOPK_NET_ANNUALIZED_RETURN_FIG_PATH,
    )
    print(f"[Saved] {DT_TOPK_NET_ANNUALIZED_RETURN_FIG_PATH}")

    plot_top10_drawdown_comparison(
        portfolio_returns_df=portfolio_returns_df,
        all_stock_returns_df=all_stock_returns_df,
        random_annual_mean_df=random_annual_mean_df,
        output_path=DT_TOP10_DRAWDOWN_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {DT_TOP10_DRAWDOWN_FIG_PATH}")

    plot_random_benchmark_distribution_topk(
        portfolio_metrics_df=portfolio_metrics_df,
        random_runs_df=random_runs_df,
        output_path=DT_RANDOM_BENCHMARK_DISTRIBUTION_TOP10_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {DT_RANDOM_BENCHMARK_DISTRIBUTION_TOP10_FIG_PATH}")

    plot_return_risk_scatter(
        portfolio_metrics_df=portfolio_metrics_df,
        random_summary_df=random_summary_df,
        all_stock_metrics_df=all_stock_metrics_df,
        output_path=DT_RETURN_RISK_SCATTER_FIG_PATH,
    )
    print(f"[Saved] {DT_RETURN_RISK_SCATTER_FIG_PATH}")

    print("[Info] Generating random cumulative band figure...")
    plot_top10_random_cumulative_band(
        portfolio_returns_df=portfolio_returns_df,
        all_stock_returns_df=all_stock_returns_df,
        random_runs_df=random_runs_df,
        output_path=DT_RANDOM_CUMULATIVE_BAND_TOP10_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {DT_RANDOM_CUMULATIVE_BAND_TOP10_FIG_PATH}")

    print("[Info] Generating benchmark zoom figure...")
    plot_top10_benchmark_zoom(
        all_stock_returns_df=all_stock_returns_df,
        random_runs_df=random_runs_df,
        output_path=DT_BENCHMARK_ZOOM_TOP10_FIG_PATH,
        top_k=10,
    )
    print(f"[Saved] {DT_BENCHMARK_ZOOM_TOP10_FIG_PATH}")

    print("")
    print("========== Step 3 + Step 4 PNG Generation Finished Successfully ==========")


def require_file(path: str | Path) -> None:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到必要檔案：{path}\n"
            "請確認你已依序完成 Step 3、Step 4、Step 5。"
        )


def main() -> None:
    print("========== Step 5: Generate Multi-model PNG Figures ==========")

    required_files = [
        ALL_CLASSIFICATION_METRICS_PATH,
        TASK2_FEATURE_IMPORTANCE_PATH,
        DT_PORTFOLIO_RETURNS_PATH,
        TASK2_PORTFOLIO_RETURNS_PATH,
        ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH,
        ALL_STOCK_BENCHMARK_RETURNS_PATH,
        ALL_STOCK_BENCHMARK_METRICS_PATH,
        RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH,
        RANDOM_TOPK_BENCHMARK_SUMMARY_PATH,
    ]

    for file_path in required_files:
        require_file(file_path)

    print("[Info] Loading classification metrics...")
    all_classification_metrics_df = pd.read_csv(ALL_CLASSIFICATION_METRICS_PATH)
    print(f"[Info] all_classification_metrics_df shape: {all_classification_metrics_df.shape}")

    print("[Info] Loading Task 2 feature importance...")
    task2_feature_importance_df = pd.read_csv(TASK2_FEATURE_IMPORTANCE_PATH)
    print(f"[Info] task2_feature_importance_df shape: {task2_feature_importance_df.shape}")

    print("[Info] Loading portfolio annual returns...")
    dt_portfolio_returns_df = pd.read_csv(DT_PORTFOLIO_RETURNS_PATH)
    task2_portfolio_returns_df = pd.read_csv(TASK2_PORTFOLIO_RETURNS_PATH)

    all_portfolio_returns_df = pd.concat(
        [dt_portfolio_returns_df, task2_portfolio_returns_df],
        ignore_index=True,
    )

    print(f"[Info] all_portfolio_returns_df shape: {all_portfolio_returns_df.shape}")

    print("[Info] Loading portfolio metrics and benchmarks...")
    all_portfolio_metrics_df = pd.read_csv(ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH)
    all_stock_returns_df = pd.read_csv(ALL_STOCK_BENCHMARK_RETURNS_PATH)
    all_stock_metrics_df = pd.read_csv(ALL_STOCK_BENCHMARK_METRICS_PATH)
    random_annual_mean_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH)
    random_summary_df = pd.read_csv(RANDOM_TOPK_BENCHMARK_SUMMARY_PATH)

    print(f"[Info] all_portfolio_metrics_df shape: {all_portfolio_metrics_df.shape}")
    print(f"[Info] all_stock_returns_df shape: {all_stock_returns_df.shape}")
    print(f"[Info] random_annual_mean_df shape: {random_annual_mean_df.shape}")

    print("")
    print("========== Generating Step 5 PNGs ==========")

    plot_step5_classification_overall_metrics(
        all_classification_metrics_df=all_classification_metrics_df,
        output_path=STEP5_CLASSIFICATION_OVERALL_FIG_PATH,
    )
    print(f"[Saved] {STEP5_CLASSIFICATION_OVERALL_FIG_PATH}")

    plot_step5_f1_by_testing_year(
        all_classification_metrics_df=all_classification_metrics_df,
        output_path=STEP5_F1_BY_YEAR_FIG_PATH,
    )
    print(f"[Saved] {STEP5_F1_BY_YEAR_FIG_PATH}")

    plot_step5_top10_cumulative_return_comparison(
        all_portfolio_returns_df=all_portfolio_returns_df,
        all_stock_returns_df=all_stock_returns_df,
        random_annual_mean_df=random_annual_mean_df,
        output_path=STEP5_TOP10_CUMULATIVE_RETURN_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP5_TOP10_CUMULATIVE_RETURN_FIG_PATH}")

    plot_step5_top10_net_annualized_return(
        all_portfolio_metrics_df=all_portfolio_metrics_df,
        all_stock_metrics_df=all_stock_metrics_df,
        random_summary_df=random_summary_df,
        output_path=STEP5_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP5_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH}")

    plot_step5_topk_net_annualized_heatmap(
        all_portfolio_metrics_df=all_portfolio_metrics_df,
        output_path=STEP5_TOPK_NET_ANNUALIZED_HEATMAP_FIG_PATH,
        weight_method="equal",
    )
    print(f"[Saved] {STEP5_TOPK_NET_ANNUALIZED_HEATMAP_FIG_PATH}")

    plot_step5_return_risk_scatter(
        all_portfolio_metrics_df=all_portfolio_metrics_df,
        all_stock_metrics_df=all_stock_metrics_df,
        random_summary_df=random_summary_df,
        output_path=STEP5_RETURN_RISK_SCATTER_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP5_RETURN_RISK_SCATTER_FIG_PATH}")

    plot_step5_sharpe_by_model(
        all_portfolio_metrics_df=all_portfolio_metrics_df,
        output_path=STEP5_SHARPE_BY_MODEL_FIG_PATH,
        top_k=10,
        weight_method="equal",
    )
    print(f"[Saved] {STEP5_SHARPE_BY_MODEL_FIG_PATH}")

    plot_step5_rf_gb_feature_importance(
        task2_feature_importance_df=task2_feature_importance_df,
        output_path=STEP5_FEATURE_IMPORTANCE_RF_GB_FIG_PATH,
        top_n=10,
    )
    print(f"[Saved] {STEP5_FEATURE_IMPORTANCE_RF_GB_FIG_PATH}")

    print("")
    print("========== Step 5 PNG Generation Finished Successfully ==========")


if __name__ == "__main__":
    main()