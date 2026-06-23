# step5_validate_task2.py
from pathlib import Path

import pandas as pd

from src.config import (
    TASK2_CLASSIFICATION_MODELS,
    TOP_K_LIST,
    TASK2_PREDICTIONS_PATH,
    TASK2_CLASSIFICATION_METRICS_PATH,
    TASK2_FEATURE_IMPORTANCE_PATH,
    TASK2_SELECTED_STOCKS_PATH,
    TASK2_PORTFOLIO_RETURNS_PATH,
    TASK2_PORTFOLIO_METRICS_PATH,
    SAVED_MODEL_DIR,
)
from src.schema import STOCK_ID_COL, YEAR_COL, FEATURE_COLUMNS


def main() -> None:
    print("========== Step 5 Final Validation ==========")

    pred_df = pd.read_csv(TASK2_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str})
    metrics_df = pd.read_csv(TASK2_CLASSIFICATION_METRICS_PATH)
    fi_df = pd.read_csv(TASK2_FEATURE_IMPORTANCE_PATH)

    selected_df = pd.read_csv(TASK2_SELECTED_STOCKS_PATH, dtype={STOCK_ID_COL: str})
    portfolio_returns_df = pd.read_csv(TASK2_PORTFOLIO_RETURNS_PATH)
    portfolio_metrics_df = pd.read_csv(TASK2_PORTFOLIO_METRICS_PATH)

    print(f"[Info] pred_df shape: {pred_df.shape}")
    print(f"[Info] metrics_df shape: {metrics_df.shape}")
    print(f"[Info] feature_importance_df shape: {fi_df.shape}")
    print(f"[Info] selected_df shape: {selected_df.shape}")
    print(f"[Info] portfolio_returns_df shape: {portfolio_returns_df.shape}")
    print(f"[Info] portfolio_metrics_df shape: {portfolio_metrics_df.shape}")

    n_models = len(TASK2_CLASSIFICATION_MODELS)
    n_splits = 11
    n_years = 11
    n_stocks_per_year = 200

    expected_pred_rows = n_models * n_years * n_stocks_per_year
    assert len(pred_df) == expected_pred_rows, (
        f"Task2 predictions 應有 {expected_pred_rows} rows，"
        f"但目前是 {len(pred_df)}"
    )

    assert set(pred_df["model_name"].unique().tolist()) == set(TASK2_CLASSIFICATION_MODELS), (
        "prediction model_name 與 TASK2_CLASSIFICATION_MODELS 不一致"
    )

    for model_name in TASK2_CLASSIFICATION_MODELS:
        model_df = pred_df[pred_df["model_name"] == model_name]

        assert model_df["split_id"].nunique() == 11, (
            f"{model_name} 應有 11 個 split"
        )

        year_counts = model_df.groupby(YEAR_COL).size()
        bad_counts = year_counts[year_counts != 200]

        assert bad_counts.empty, (
            f"{model_name} 以下年份不是 200 筆：\n{bad_counts}"
        )

        assert model_df["score_label_1"].between(0, 1).all(), (
            f"{model_name} score_label_1 應介於 0 到 1"
        )

        assert set(model_df["predicted_label"].unique().tolist()).issubset({-1, 1}), (
            f"{model_name} predicted_label 應只包含 -1 / 1"
        )

    # classification metrics
    expected_metrics_rows = n_models * 12  # 11 splits + 1 overall per model
    assert len(metrics_df) == expected_metrics_rows, (
        f"classification metrics 應有 {expected_metrics_rows} rows，"
        f"但目前是 {len(metrics_df)}"
    )

    metric_cols = [
        "accuracy",
        "precision_label_1",
        "recall_label_1",
        "f1_label_1",
    ]

    for col in metric_cols:
        assert metrics_df[col].notna().all(), f"{col} 存在缺失"
        assert metrics_df[col].between(0, 1).all(), f"{col} 應介於 0 到 1"

    # feature importance
    expected_fi_rows = n_models * n_splits * len(FEATURE_COLUMNS)
    assert len(fi_df) == expected_fi_rows, (
        f"feature importance 應有 {expected_fi_rows} rows，"
        f"但目前是 {len(fi_df)}"
    )

    assert set(fi_df["feature"].unique().tolist()) == set(FEATURE_COLUMNS), (
        "feature importance 的 feature 欄位不正確"
    )

    # saved models
    saved_model_dir = Path(SAVED_MODEL_DIR)

    for model_name in TASK2_CLASSIFICATION_MODELS:
        model_files = sorted(saved_model_dir.glob(f"{model_name}_split_*.joblib"))
        assert len(model_files) == 11, (
            f"{model_name} 應有 11 個 saved models，但目前是 {len(model_files)}"
        )

    # portfolio validation
    expected_selected_rows = n_models * n_years * sum(TOP_K_LIST) * 2
    assert len(selected_df) == expected_selected_rows, (
        f"selected stocks 應有 {expected_selected_rows} rows，"
        f"但目前是 {len(selected_df)}"
    )

    group_cols = ["model_name", YEAR_COL, "top_k", "weight_method"]
    selected_counts = selected_df.groupby(group_cols).size().reset_index(name="count")
    bad_selected_counts = selected_counts[
        selected_counts["count"] != selected_counts["top_k"]
    ]

    assert bad_selected_counts.empty, (
        f"以下 selected count 不等於 top_k：\n{bad_selected_counts}"
    )

    weight_sum = (
        selected_df.groupby(group_cols)["weight"]
        .sum()
        .reset_index(name="weight_sum")
    )

    bad_weight_sum = weight_sum[
        (weight_sum["weight_sum"] - 1.0).abs() > 1e-8
    ]

    assert bad_weight_sum.empty, (
        f"以下 portfolio 權重加總不等於 1：\n{bad_weight_sum}"
    )

    expected_portfolio_return_rows = n_models * n_years * len(TOP_K_LIST) * 2
    assert len(portfolio_returns_df) == expected_portfolio_return_rows, (
        f"portfolio returns 應有 {expected_portfolio_return_rows} rows，"
        f"但目前是 {len(portfolio_returns_df)}"
    )

    expected_portfolio_metrics_rows = n_models * len(TOP_K_LIST) * 2
    assert len(portfolio_metrics_df) == expected_portfolio_metrics_rows, (
        f"portfolio metrics 應有 {expected_portfolio_metrics_rows} rows，"
        f"但目前是 {len(portfolio_metrics_df)}"
    )

    print("")
    print("[Pass] Step 5 Task 2 models, predictions, metrics, portfolios, and saved models are valid.")
    print("========== Step 5 Validation Finished ==========")


if __name__ == "__main__":
    main()