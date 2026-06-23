from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    TEMPORAL_SPLITS_NEXT_YEAR_PATH,
    SAVED_MODEL_DIR,
    TASK2_CLASSIFICATION_MODELS,
    TASK2_PREDICTIONS_PATH,
    TASK2_CLASSIFICATION_METRICS_PATH,
    TASK2_FEATURE_IMPORTANCE_PATH,
    TASK2_STEP5_PROFILE_PATH,
    DT_CLASSIFICATION_METRICS_PATH,
    ALL_CLASSIFICATION_METRICS_PATH,
)
from src.schema import STOCK_ID_COL
from src.prediction import train_classification_models_all_splits
from src.metrics import calculate_classification_metrics_by_model


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_text(text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def maybe_save_all_classification_metrics(
    task2_metrics_df: pd.DataFrame,
) -> None:
    """
    若 Step 3 Decision Tree metrics 已存在，則合併成 all_classification_metrics.csv。
    """
    dt_metrics_path = Path(DT_CLASSIFICATION_METRICS_PATH)

    if not dt_metrics_path.exists():
        print("[Warning] 找不到 Decision Tree metrics，暫不輸出 all_classification_metrics.csv")
        return

    dt_metrics_df = pd.read_csv(dt_metrics_path)

    if "model_name" not in dt_metrics_df.columns:
        dt_metrics_df.insert(0, "model_name", "decision_tree_entropy")

    all_metrics_df = pd.concat(
        [dt_metrics_df, task2_metrics_df],
        ignore_index=True,
    )

    save_dataframe(all_metrics_df, ALL_CLASSIFICATION_METRICS_PATH)
    print(f"[Saved] {ALL_CLASSIFICATION_METRICS_PATH}")


def generate_step5_profile(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
) -> str:
    lines = []

    lines.append("========== Step 5 Task 2 Classification Models Profile ==========")
    lines.append("")

    lines.append("Models:")
    for model_name in TASK2_CLASSIFICATION_MODELS:
        lines.append(f"- {model_name}")
    lines.append("")

    lines.append("Validation:")
    lines.append("- Mode: next_year expanding-window temporal validation")
    lines.append("- Testing years: 1998-2008")
    lines.append("- Each split uses only earlier years as training data")
    lines.append("")

    lines.append("Prediction summary:")
    lines.append(f"- Total prediction rows: {len(predictions_df)}")
    lines.append(f"- Number of models: {predictions_df['model_name'].nunique()}")
    lines.append(f"- Number of splits per model: {predictions_df['split_id'].nunique()}")
    lines.append("")

    lines.append("Rows by model:")
    model_counts = predictions_df.groupby("model_name").size()
    for model_name, count in model_counts.items():
        lines.append(f"- {model_name}: {count} rows")
    lines.append("")

    lines.append("Overall classification metrics:")
    overall = metrics_df[metrics_df["scope"] == "overall"].copy()
    display_cols = [
        "model_name",
        "n_samples",
        "accuracy",
        "precision_label_1",
        "recall_label_1",
        "f1_label_1",
    ]
    lines.append(overall[display_cols].to_string(index=False))
    lines.append("")

    lines.append("Average feature importance / coefficient by model:")
    avg_fi = (
        feature_importance_df.groupby(["model_name", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["model_name", "importance"], ascending=[True, False])
    )

    for model_name in avg_fi["model_name"].unique():
        lines.append("")
        lines.append(f"[{model_name}]")
        temp = avg_fi[avg_fi["model_name"] == model_name].head(10)
        lines.append(temp.to_string(index=False))

    lines.append("")
    lines.append("Generated files:")
    lines.append(f"- Predictions: {TASK2_PREDICTIONS_PATH}")
    lines.append(f"- Classification metrics: {TASK2_CLASSIFICATION_METRICS_PATH}")
    lines.append(f"- Feature importance: {TASK2_FEATURE_IMPORTANCE_PATH}")
    lines.append("")
    lines.append("========== End of Step 5 Profile ==========")

    return "\n".join(lines)


def main() -> None:
    print("========== Step 5: Task 2 Classification Models ==========")

    print(f"[Info] Loading cleaned data from: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] Cleaned data shape: {df.shape}")

    print(f"[Info] Loading temporal splits from: {TEMPORAL_SPLITS_NEXT_YEAR_PATH}")
    splits_df = pd.read_csv(TEMPORAL_SPLITS_NEXT_YEAR_PATH)
    print(f"[Info] Splits shape: {splits_df.shape}")

    print("")
    print("[Info] Training Task 2 classification models...")
    predictions_df, feature_importance_df = train_classification_models_all_splits(
        df=df,
        splits_df=splits_df,
        model_names=TASK2_CLASSIFICATION_MODELS,
        saved_model_dir=SAVED_MODEL_DIR,
    )

    print("")
    print("[Info] Calculating classification metrics...")
    metrics_df = calculate_classification_metrics_by_model(predictions_df)

    print("")
    print("[Info] Saving outputs...")
    save_dataframe(predictions_df, TASK2_PREDICTIONS_PATH)
    save_dataframe(metrics_df, TASK2_CLASSIFICATION_METRICS_PATH)
    save_dataframe(feature_importance_df, TASK2_FEATURE_IMPORTANCE_PATH)

    maybe_save_all_classification_metrics(metrics_df)

    profile_text = generate_step5_profile(
        predictions_df=predictions_df,
        metrics_df=metrics_df,
        feature_importance_df=feature_importance_df,
    )

    save_text(profile_text, TASK2_STEP5_PROFILE_PATH)

    print("")
    print(profile_text)
    print("")
    print("========== Step 5 Training Finished Successfully ==========")


if __name__ == "__main__":
    main()