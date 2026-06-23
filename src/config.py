# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"

RAW_DATA_PATH = RAW_DATA_DIR / "top200.xlsx"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_top200.csv"
DATA_PROFILE_PATH = LOG_DIR / "step1_data_profile.txt"

RANDOM_SEED = 42

# 刪除 年月 = 200912
DROP_YEARMONTH = 200912

TEMPORAL_SPLITS_NEXT_YEAR_PATH = PROCESSED_DATA_DIR / "temporal_splits_next_year.csv"
TEMPORAL_SPLITS_REMAINING_YEARS_PATH = PROCESSED_DATA_DIR / "temporal_splits_remaining_years.csv"

SPLIT_PROFILE_NEXT_YEAR_PATH = LOG_DIR / "step2_split_profile_next_year.txt"
SPLIT_PROFILE_REMAINING_YEARS_PATH = LOG_DIR / "step2_split_profile_remaining_years.txt"

# Main validation mode for portfolio backtesting
MAIN_VALIDATION_MODE = "next_year"


PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_REPORT_DIR = OUTPUT_DIR / "model_reports"

SAVED_MODEL_DIR = PROJECT_ROOT / "saved_models"

DT_PREDICTIONS_PATH = PREDICTION_DIR / "decision_tree_predictions.csv"
DT_CLASSIFICATION_METRICS_PATH = METRICS_DIR / "decision_tree_classification_metrics.csv"
DT_FEATURE_IMPORTANCE_PATH = METRICS_DIR / "decision_tree_feature_importance.csv"

DT_RULES_DIR = MODEL_REPORT_DIR / "decision_tree_rules"
DT_STEP3_PROFILE_PATH = LOG_DIR / "step3_decision_tree_profile.txt"

# Decision Tree main setting
DT_MODEL_NAME = "decision_tree_entropy"

DT_MAX_DEPTH = 5
DT_MIN_SAMPLES_LEAF = 5
DT_CRITERION = "entropy"

FIGURE_DIR = OUTPUT_DIR / "figures"

DT_FIGURE_DIR = FIGURE_DIR / "decision_tree"

DT_METRICS_BY_YEAR_FIG_PATH = DT_FIGURE_DIR / "dt_metrics_by_year.png"
DT_CONFUSION_MATRIX_FIG_PATH = DT_FIGURE_DIR / "dt_confusion_matrix_overall.png"
DT_FEATURE_IMPORTANCE_AVG_FIG_PATH = DT_FIGURE_DIR / "dt_feature_importance_avg.png"
DT_FEATURE_IMPORTANCE_HEATMAP_FIG_PATH = DT_FIGURE_DIR / "dt_feature_importance_heatmap.png"
DT_SCORE_BOXPLOT_FIG_PATH = DT_FIGURE_DIR / "dt_score_by_actual_label_boxplot.png"
DT_TREE_SPLIT_11_FIG_PATH = DT_FIGURE_DIR / "dt_tree_split_11_depth3.png"

DT_TREE_PLOT_SPLIT_ID = 11
DT_TREE_PLOT_MAX_DEPTH = 3


# Step 4 portfolio figures
DT_PORTFOLIO_FIGURE_DIR = FIGURE_DIR / "decision_tree_portfolio"

DT_TOP10_CUMULATIVE_RETURN_FIG_PATH = (
    DT_PORTFOLIO_FIGURE_DIR / "dt_top10_cumulative_return_comparison.png"
)

DT_TOP10_ANNUAL_RETURN_FIG_PATH = (
    DT_PORTFOLIO_FIGURE_DIR / "dt_top10_annual_return_comparison.png"
)

DT_TOPK_NET_ANNUALIZED_RETURN_FIG_PATH = (
    DT_PORTFOLIO_FIGURE_DIR / "dt_topk_net_annualized_return.png"
)

DT_TOP10_DRAWDOWN_FIG_PATH = (
    DT_PORTFOLIO_FIGURE_DIR / "dt_top10_drawdown_comparison.png"
)

DT_RANDOM_BENCHMARK_DISTRIBUTION_TOP10_FIG_PATH = (
    DT_PORTFOLIO_FIGURE_DIR / "dt_random_benchmark_distribution_top10.png"
)

DT_RETURN_RISK_SCATTER_FIG_PATH = (
    DT_PORTFOLIO_FIGURE_DIR / "dt_return_risk_scatter.png"
)

# 用最後一個 split 畫決策樹，因為它使用最多 training years：1997-2007 → 2008
DT_TREE_PLOT_SPLIT_ID = 11
DT_TREE_PLOT_MAX_DEPTH = 3


TOP_K_LIST = [5, 10, 20, 30]
MAIN_TOP_K = 10

# Transaction cost assumptions
# 國泰券商折扣手續費假設：
# Buy fee  = 0.0399%
# Sell fee = 0.0399%
# Sell tax = 0.3%
BUY_FEE = 0.000399
SELL_FEE = 0.000399
SELL_TAX = 0.003
ROUND_TRIP_COST = BUY_FEE + SELL_FEE + SELL_TAX

RANDOM_BENCHMARK_N_RUNS = 500


SELECTION_DIR = OUTPUT_DIR / "selections"
PORTFOLIO_DIR = OUTPUT_DIR / "portfolio"
BENCHMARK_DIR = OUTPUT_DIR / "benchmarks"

DT_SELECTED_STOCKS_PATH = SELECTION_DIR / "decision_tree_selected_stocks.csv"
DT_PORTFOLIO_RETURNS_PATH = PORTFOLIO_DIR / "decision_tree_portfolio_returns.csv"
DT_PORTFOLIO_METRICS_PATH = PORTFOLIO_DIR / "decision_tree_portfolio_metrics.csv"

ALL_STOCK_BENCHMARK_RETURNS_PATH = BENCHMARK_DIR / "all_stock_benchmark_annual_returns.csv"
ALL_STOCK_BENCHMARK_METRICS_PATH = BENCHMARK_DIR / "all_stock_benchmark_metrics.csv"

RANDOM_TOPK_BENCHMARK_RUNS_PATH = BENCHMARK_DIR / "random_topk_benchmark_runs.csv"
RANDOM_TOPK_BENCHMARK_ANNUAL_MEAN_PATH = BENCHMARK_DIR / "random_topk_benchmark_annual_mean.csv"
RANDOM_TOPK_BENCHMARK_SUMMARY_PATH = BENCHMARK_DIR / "random_topk_benchmark_summary.csv"

STEP4_PORTFOLIO_PROFILE_PATH = LOG_DIR / "step4_portfolio_profile.txt"

DT_RANDOM_CUMULATIVE_BAND_TOP10_FIG_PATH = (
    DT_PORTFOLIO_FIGURE_DIR / "dt_top10_random_cumulative_band.png"
)

DT_BENCHMARK_ZOOM_TOP10_FIG_PATH = (
    DT_PORTFOLIO_FIGURE_DIR / "dt_top10_benchmark_zoom.png"
)

TASK2_CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]

TASK2_PREDICTIONS_PATH = PREDICTION_DIR / "task2_classification_predictions.csv"
TASK2_CLASSIFICATION_METRICS_PATH = METRICS_DIR / "task2_classification_metrics.csv"
TASK2_FEATURE_IMPORTANCE_PATH = METRICS_DIR / "task2_feature_importance.csv"
TASK2_STEP5_PROFILE_PATH = LOG_DIR / "step5_task2_models_profile.txt"

TASK2_SELECTED_STOCKS_PATH = SELECTION_DIR / "task2_selected_stocks.csv"
TASK2_PORTFOLIO_RETURNS_PATH = PORTFOLIO_DIR / "task2_portfolio_returns.csv"
TASK2_PORTFOLIO_METRICS_PATH = PORTFOLIO_DIR / "task2_portfolio_metrics.csv"

ALL_CLASSIFICATION_METRICS_PATH = METRICS_DIR / "all_classification_metrics.csv"
ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH = (
    PORTFOLIO_DIR / "all_classification_portfolio_metrics.csv"
)

# Logistic Regression settings
LR_MAX_ITER = 2000
LR_CLASS_WEIGHT = "balanced"

# Random Forest settings
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 3
RF_MAX_FEATURES = "sqrt"
RF_CLASS_WEIGHT = "balanced"

# Gradient Boosting settings
GB_N_ESTIMATORS = 200
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH = 3
GB_MIN_SAMPLES_LEAF = 3

STEP5_FIGURE_DIR = FIGURE_DIR / "step5_model_comparison"

STEP5_CLASSIFICATION_OVERALL_FIG_PATH = (
    STEP5_FIGURE_DIR / "step5_classification_overall_metrics.png"
)

STEP5_F1_BY_YEAR_FIG_PATH = (
    STEP5_FIGURE_DIR / "step5_f1_by_testing_year.png"
)

STEP5_TOP10_CUMULATIVE_RETURN_FIG_PATH = (
    STEP5_FIGURE_DIR / "step5_top10_cumulative_return_comparison.png"
)

STEP5_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH = (
    STEP5_FIGURE_DIR / "step5_top10_net_annualized_return.png"
)

STEP5_TOPK_NET_ANNUALIZED_HEATMAP_FIG_PATH = (
    STEP5_FIGURE_DIR / "step5_topk_net_annualized_return_heatmap.png"
)

STEP5_RETURN_RISK_SCATTER_FIG_PATH = (
    STEP5_FIGURE_DIR / "step5_return_risk_scatter.png"
)

STEP5_SHARPE_BY_MODEL_FIG_PATH = (
    STEP5_FIGURE_DIR / "step5_top10_sharpe_ratio_by_model.png"
)

STEP5_FEATURE_IMPORTANCE_RF_GB_FIG_PATH = (
    STEP5_FIGURE_DIR / "step5_rf_gb_feature_importance.png"
)


SVR_GA_MODEL_NAME = "svr_ga_regression"

SVR_GA_PREDICTIONS_PATH = PREDICTION_DIR / "svr_ga_regression_predictions.csv"
SVR_GA_REGRESSION_METRICS_PATH = METRICS_DIR / "svr_ga_regression_metrics.csv"
SVR_GA_SEARCH_LOG_PATH = METRICS_DIR / "svr_ga_search_log.csv"

SVR_GA_SELECTED_STOCKS_PATH = SELECTION_DIR / "svr_ga_selected_stocks.csv"
SVR_GA_PORTFOLIO_RETURNS_PATH = PORTFOLIO_DIR / "svr_ga_portfolio_returns.csv"
SVR_GA_PORTFOLIO_METRICS_PATH = PORTFOLIO_DIR / "svr_ga_portfolio_metrics.csv"

SVR_GA_STEP6_PROFILE_PATH = LOG_DIR / "step6_svr_ga_profile.txt"

ALL_MODELS_PORTFOLIO_METRICS_PATH = PORTFOLIO_DIR / "all_models_portfolio_metrics.csv"

# GA settings
SVR_GA_POPULATION_SIZE = 14
SVR_GA_N_GENERATIONS = 8
SVR_GA_MUTATION_RATE = 0.25
SVR_GA_CROSSOVER_RATE = 0.80
SVR_GA_TOURNAMENT_SIZE = 3

# SVR hyperparameter search space.
# 使用 log10 空間搜尋，避免 C / gamma / epsilon 尺度差太大。
SVR_GA_PARAM_BOUNDS = {
    "log10_C": (-1.0, 4.0),        # C: 0.1 ~ 10000
    "log10_gamma": (-4.0, 0.0),    # gamma: 0.0001 ~ 1
    "log10_epsilon": (-2.0, 1.5),  # epsilon: 0.01 ~ 31.62
}

STEP6_ALL_MODEL_FIGURE_DIR = FIGURE_DIR / "step6_all_model_comparison"

STEP6_ALL_MODEL_TOP10_CUMULATIVE_RETURN_FIG_PATH = (
    STEP6_ALL_MODEL_FIGURE_DIR / "step6_all_model_top10_cumulative_return.png"
)

STEP6_ALL_MODEL_TOP10_ANNUAL_RETURN_FIG_PATH = (
    STEP6_ALL_MODEL_FIGURE_DIR / "step6_all_model_top10_annual_return.png"
)

STEP6_ALL_MODEL_TOP10_NET_ANNUALIZED_RETURN_FIG_PATH = (
    STEP6_ALL_MODEL_FIGURE_DIR / "step6_all_model_top10_net_annualized_return.png"
)

STEP6_ALL_MODEL_TOPK_HEATMAP_FIG_PATH = (
    STEP6_ALL_MODEL_FIGURE_DIR / "step6_all_model_topk_net_annualized_heatmap.png"
)

STEP6_ALL_MODEL_RETURN_RISK_SCATTER_FIG_PATH = (
    STEP6_ALL_MODEL_FIGURE_DIR / "step6_all_model_return_risk_scatter.png"
)

STEP6_ALL_MODEL_SHARPE_FIG_PATH = (
    STEP6_ALL_MODEL_FIGURE_DIR / "step6_all_model_top10_sharpe_ratio.png"
)

STEP6_ALL_MODEL_DRAWDOWN_FIG_PATH = (
    STEP6_ALL_MODEL_FIGURE_DIR / "step6_all_model_top10_drawdown.png"
)

STEP6_SVR_PREDICTED_VS_ACTUAL_FIG_PATH = (
    STEP6_ALL_MODEL_FIGURE_DIR / "step6_svr_ga_predicted_vs_actual_return.png"
)


EXTERNAL_TOP_K_LIST = [5, 10, 20, 30]

EXTERNAL_FEATURE_MIN_NON_MISSING_COUNT = 1

EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_RAW_DIR = EXTERNAL_DATA_DIR / "raw"
EXTERNAL_PROCESSED_DIR = EXTERNAL_DATA_DIR / "processed"
EXTERNAL_TICKERS_PATH = EXTERNAL_DATA_DIR / "tickers.csv"

EXTERNAL_CLEANED_DATA_PATH = EXTERNAL_PROCESSED_DIR / "external_cleaned_dataset.csv"

EXTERNAL_OUTPUT_DIR = OUTPUT_DIR / "external"
EXTERNAL_PREDICTION_DIR = EXTERNAL_OUTPUT_DIR / "predictions"
EXTERNAL_SELECTION_DIR = EXTERNAL_OUTPUT_DIR / "selections"
EXTERNAL_PORTFOLIO_DIR = EXTERNAL_OUTPUT_DIR / "portfolio"
EXTERNAL_METRICS_DIR = EXTERNAL_OUTPUT_DIR / "metrics"
EXTERNAL_FIGURE_DIR = EXTERNAL_OUTPUT_DIR / "figures"
EXTERNAL_LOG_DIR = EXTERNAL_OUTPUT_DIR / "logs"

EXTERNAL_RF_MODEL_NAME = "external_random_forest"

EXTERNAL_RF_PREDICTIONS_PATH = EXTERNAL_PREDICTION_DIR / "external_rf_predictions.csv"
EXTERNAL_RF_SELECTED_STOCKS_PATH = EXTERNAL_SELECTION_DIR / "external_rf_selected_stocks.csv"
EXTERNAL_RF_PORTFOLIO_RETURNS_PATH = EXTERNAL_PORTFOLIO_DIR / "external_rf_portfolio_returns.csv"
EXTERNAL_RF_PORTFOLIO_METRICS_PATH = EXTERNAL_PORTFOLIO_DIR / "external_rf_portfolio_metrics.csv"
EXTERNAL_RF_CLASSIFICATION_METRICS_PATH = EXTERNAL_METRICS_DIR / "external_rf_classification_metrics.csv"
EXTERNAL_PROFILE_PATH = EXTERNAL_LOG_DIR / "external_data_profile.txt"

EXTERNAL_CUMULATIVE_RETURN_FIG_PATH = EXTERNAL_FIGURE_DIR / "external_rf_cumulative_return.png"
EXTERNAL_TOPK_ANNUALIZED_RETURN_FIG_PATH = EXTERNAL_FIGURE_DIR / "external_rf_topk_annualized_return.png"


DEMO_OUTPUT_DIR = OUTPUT_DIR / "demo"
DEMO_PREDICTIONS_PATH = DEMO_OUTPUT_DIR / "demo_predictions.csv"
DEMO_SELECTED_STOCKS_PATH = DEMO_OUTPUT_DIR / "demo_selected_stocks.csv"
DEMO_PORTFOLIO_RETURNS_PATH = DEMO_OUTPUT_DIR / "demo_portfolio_returns.csv"
DEMO_METRICS_PATH = DEMO_OUTPUT_DIR / "demo_metrics.csv"
DEMO_PROFILE_PATH = DEMO_OUTPUT_DIR / "demo_profile.txt"
DEMO_FIGURE_PATH = DEMO_OUTPUT_DIR / "demo_cumulative_return.png"

EXTERNAL_METADATA_DIR = EXTERNAL_DATA_DIR / "metadata"

EXTERNAL_FEATURE_MAPPING_PATH = (
    EXTERNAL_METADATA_DIR / "external_feature_mapping.csv"
)

EXTERNAL_UNIVERSE_PATH = (
    EXTERNAL_METADATA_DIR / "external_universe.csv"
)

EXTERNAL_CRAWLER_LOG_PATH = (
    EXTERNAL_LOG_DIR / "external_crawler_log.csv"
)

EXTERNAL_DATA_QUALITY_REPORT_PATH = (
    EXTERNAL_LOG_DIR / "external_data_quality_report.csv"
)

EXTERNAL_DATA_QUALITY_SUMMARY_PATH = (
    EXTERNAL_LOG_DIR / "external_data_quality_summary.txt"
)

EXTERNAL_FEATURE_MISSING_RATIO_FIG_PATH = (
    EXTERNAL_FIGURE_DIR / "external_feature_missing_ratio.png"
)

EXTERNAL_RF_VS_BENCHMARK_FIG_PATH = (
    EXTERNAL_FIGURE_DIR / "external_rf_vs_benchmark_annualized_return.png"
)

EXTERNAL_TOPK_SENSITIVITY_FIG_PATH = (
    EXTERNAL_FIGURE_DIR / "external_topk_sensitivity.png"
)


EXTERNAL_FINANCIAL_VALUE_UNIT = "thousand_ntd"

# 計算市值時，若使用股本推估股數，台灣普通股面額通常以 10 元估算。
EXTERNAL_PAR_VALUE_PER_SHARE = 10.0