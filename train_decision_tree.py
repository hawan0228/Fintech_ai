from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    TEMPORAL_SPLITS_NEXT_YEAR_PATH,
    DT_PREDICTIONS_PATH,
    DT_CLASSIFICATION_METRICS_PATH,
    DT_FEATURE_IMPORTANCE_PATH,
    DT_RULES_DIR,
    DT_STEP3_PROFILE_PATH,
    SAVED_MODEL_DIR,
)
from src.schema import STOCK_ID_COL
from src.prediction import train_decision_tree_all_splits
from src.metrics import calculate_classification_metrics


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def generate_step3_profile(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
) -> str:
    """
    產生 Step 3 log profile。
    """
    lines = []

    lines.append("========== Step 3 Decision Tree Profile ==========")
    lines.append("")
    lines.append("Model:")
    lines.append("- Model name: decision_tree_entropy")
    lines.append("- Classifier: sklearn DecisionTreeClassifier")
    lines.append("- Criterion: entropy")
    lines.append("- Purpose: Task 1 ID3-like stock-selection classification model")
    lines.append("")

    lines.append("Prediction output summary:")
    lines.append(f"- Total prediction rows: {len(predictions_df)}")
    lines.append(f"- Number of splits: {predictions_df['split_id'].nunique()}")
    lines.append(f"- Test year range: {predictions_df['year'].min()} - {predictions_df['year'].max()}")
    lines.append("")

    lines.append("Rows by testing year:")
    year_counts = predictions_df.groupby("year").size()
    for year, count in year_counts.items():
        lines.append(f"- {year}: {count} rows")
    lines.append("")

    lines.append("Overall classification metrics:")
    overall = metrics_df[metrics_df["scope"] == "overall"].copy()
    if not overall.empty:
        row = overall.iloc[0]
        lines.append(f"- Accuracy: {row['accuracy']:.6f}")
        lines.append(f"- Precision label=1: {row['precision_label_1']:.6f}")
        lines.append(f"- Recall label=1: {row['recall_label_1']:.6f}")
        lines.append(f"- F1 label=1: {row['f1_label_1']:.6f}")
    lines.append("")

    lines.append("Classification metrics by split:")
    display_cols = [
        "split_id",
        "test_years",
        "n_samples",
        "accuracy",
        "precision_label_1",
        "recall_label_1",
        "f1_label_1",
    ]
    split_metrics = metrics_df[metrics_df["scope"] == "split"][display_cols]
    lines.append(split_metrics.to_string(index=False))
    lines.append("")

    lines.append("Average feature importance across splits:")
    avg_fi = (
        feature_importance_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    lines.append(avg_fi.to_string(index=False))
    lines.append("")

    lines.append("Generated files:")
    lines.append(f"- Predictions: {DT_PREDICTIONS_PATH}")
    lines.append(f"- Classification metrics: {DT_CLASSIFICATION_METRICS_PATH}")
    lines.append(f"- Feature importance: {DT_FEATURE_IMPORTANCE_PATH}")
    lines.append(f"- Tree rules directory: {DT_RULES_DIR}")
    lines.append(f"- Saved models directory: {SAVED_MODEL_DIR}")
    lines.append("")
    lines.append("========== End of Step 3 Profile ==========")

    return "\n".join(lines)


def save_text(text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    print("========== Step 3: Decision Tree Entropy Model ==========")

    print(f"[Info] Loading cleaned data from: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] Cleaned data shape: {df.shape}")

    print(f"[Info] Loading temporal splits from: {TEMPORAL_SPLITS_NEXT_YEAR_PATH}")
    splits_df = pd.read_csv(TEMPORAL_SPLITS_NEXT_YEAR_PATH)
    print(f"[Info] Splits shape: {splits_df.shape}")

    print("")
    print("[Info] Training Decision Tree on all next_year splits...")
    predictions_df, feature_importance_df = train_decision_tree_all_splits(
        df=df,
        splits_df=splits_df,
        saved_model_dir=SAVED_MODEL_DIR,
        rules_dir=DT_RULES_DIR,
    )

    print("")
    print("[Info] Calculating classification metrics...")
    metrics_df = calculate_classification_metrics(predictions_df)

    print(f"[Info] Saving predictions to: {DT_PREDICTIONS_PATH}")
    save_dataframe(predictions_df, DT_PREDICTIONS_PATH)

    print(f"[Info] Saving classification metrics to: {DT_CLASSIFICATION_METRICS_PATH}")
    save_dataframe(metrics_df, DT_CLASSIFICATION_METRICS_PATH)

    print(f"[Info] Saving feature importance to: {DT_FEATURE_IMPORTANCE_PATH}")
    save_dataframe(feature_importance_df, DT_FEATURE_IMPORTANCE_PATH)

    print(f"[Info] Saving Step 3 profile to: {DT_STEP3_PROFILE_PATH}")
    profile_text = generate_step3_profile(
        predictions_df=predictions_df,
        metrics_df=metrics_df,
        feature_importance_df=feature_importance_df,
    )
    save_text(profile_text, DT_STEP3_PROFILE_PATH)

    print("")
    print(profile_text)
    print("")
    print("========== Step 3 Finished Successfully ==========")


if __name__ == "__main__":
    main()