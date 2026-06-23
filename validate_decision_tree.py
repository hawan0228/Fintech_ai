from pathlib import Path

import pandas as pd

from src.config import (
    DT_PREDICTIONS_PATH,
    DT_CLASSIFICATION_METRICS_PATH,
    DT_FEATURE_IMPORTANCE_PATH,
    DT_RULES_DIR,
    SAVED_MODEL_DIR,
)
from src.schema import FEATURE_COLUMNS, STOCK_ID_COL


def main() -> None:
    print("========== Step 3 Final Validation ==========")

    print(f"[Info] Loading predictions: {DT_PREDICTIONS_PATH}")
    pred_df = pd.read_csv(DT_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] Predictions shape: {pred_df.shape}")

    print(f"[Info] Loading metrics: {DT_CLASSIFICATION_METRICS_PATH}")
    metrics_df = pd.read_csv(DT_CLASSIFICATION_METRICS_PATH)
    print(f"[Info] Metrics shape: {metrics_df.shape}")

    print(f"[Info] Loading feature importance: {DT_FEATURE_IMPORTANCE_PATH}")
    fi_df = pd.read_csv(DT_FEATURE_IMPORTANCE_PATH)
    print(f"[Info] Feature importance shape: {fi_df.shape}")

    # 1. predictions rows
    assert len(pred_df) == 2200, (
        f"Prediction rows 應為 2200，因為 1998-2008 共 11 年 × 每年 200 檔，"
        f"但目前是 {len(pred_df)}"
    )

    # 2. split count
    assert pred_df["split_id"].nunique() == 11, (
        f"應有 11 個 split，但目前是 {pred_df['split_id'].nunique()}"
    )

    # 3. test years
    expected_years = list(range(1998, 2009))
    actual_years = sorted(pred_df["year"].unique().tolist())
    assert actual_years == expected_years, (
        f"Testing years 應為 {expected_years}，但目前是 {actual_years}"
    )

    # 4. 每年 200 rows
    year_counts = pred_df.groupby("year").size()
    bad_year_counts = year_counts[year_counts != 200]
    assert bad_year_counts.empty, f"以下 testing years 不是 200 筆：\n{bad_year_counts}"

    # 5. labels
    actual_labels = set(pred_df["actual_label"].unique().tolist())
    pred_labels = set(pred_df["predicted_label"].unique().tolist())

    assert actual_labels.issubset({-1, 1}), f"actual_label 應只包含 -1/1，但目前是 {actual_labels}"
    assert pred_labels.issubset({-1, 1}), f"predicted_label 應只包含 -1/1，但目前是 {pred_labels}"

    # 6. score
    assert pred_df["score_label_1"].notna().all(), "score_label_1 存在缺失值"
    assert pred_df["score_label_1"].between(0, 1).all(), "score_label_1 應介於 0 到 1"

    # 7. actual_return
    assert pred_df["actual_return"].notna().all(), "actual_return 存在缺失值"

    # 8. metrics
    split_metrics = metrics_df[metrics_df["scope"] == "split"]
    overall_metrics = metrics_df[metrics_df["scope"] == "overall"]

    assert len(split_metrics) == 11, f"split metrics 應有 11 筆，但目前是 {len(split_metrics)}"
    assert len(overall_metrics) == 1, f"overall metrics 應有 1 筆，但目前是 {len(overall_metrics)}"

    metric_cols = [
        "accuracy",
        "precision_label_1",
        "recall_label_1",
        "f1_label_1",
    ]

    for col in metric_cols:
        assert metrics_df[col].notna().all(), f"{col} 存在缺失值"
        assert metrics_df[col].between(0, 1).all(), f"{col} 應介於 0 到 1"

    # 9. feature importance
    expected_fi_rows = 11 * len(FEATURE_COLUMNS)
    assert len(fi_df) == expected_fi_rows, (
        f"feature importance 應有 {expected_fi_rows} 筆，"
        f"但目前是 {len(fi_df)}"
    )

    assert set(fi_df["feature"].unique().tolist()) == set(FEATURE_COLUMNS), (
        "feature importance 的 feature 欄位與 schema.FEATURE_COLUMNS 不一致"
    )

    assert fi_df["importance"].notna().all(), "feature importance 存在缺失值"
    assert (fi_df["importance"] >= 0).all(), "feature importance 不應為負數"

    # 10. saved models and rules
    saved_model_dir = Path(SAVED_MODEL_DIR)
    rules_dir = Path(DT_RULES_DIR)

    model_files = sorted(saved_model_dir.glob("decision_tree_split_*.joblib"))
    rule_files = sorted(rules_dir.glob("decision_tree_rules_split_*.txt"))

    assert len(model_files) == 11, f"應有 11 個 saved model，但目前是 {len(model_files)}"
    assert len(rule_files) == 11, f"應有 11 個 tree rule 檔案，但目前是 {len(rule_files)}"

    print("")
    print("[Pass] Prediction rows, years, labels, scores, metrics, feature importance, models, and rules are valid.")
    print("========== Step 3 Validation Finished ==========")


if __name__ == "__main__":
    main()