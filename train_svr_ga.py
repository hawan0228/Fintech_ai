# step6_train_svr_ga.py
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEANED_DATA_PATH,
    TEMPORAL_SPLITS_NEXT_YEAR_PATH,
    SAVED_MODEL_DIR,
    SVR_GA_MODEL_NAME,
    SVR_GA_PREDICTIONS_PATH,
    SVR_GA_REGRESSION_METRICS_PATH,
    SVR_GA_SEARCH_LOG_PATH,
    SVR_GA_STEP6_PROFILE_PATH,
    RANDOM_SEED,
)
from src.schema import (
    FEATURE_COLUMNS,
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEAR_COL,
    TARGET_CLASS,
    TARGET_RETURN,
)
from src.svr_ga import run_svr_ga_search, build_svr_pipeline
from src.metrics import calculate_regression_metrics_by_model


def parse_years(years_text: str) -> list[int]:
    return [int(x) for x in str(years_text).split(",") if str(x).strip()]


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_text(text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def make_inner_train_valid(
    df: pd.DataFrame,
    train_years: list[int],
    split_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    在 outer training years 內部建立 GA 用的 inner train / validation。

    標準做法：
    - 若 training years >= 2：用最後一個 training year 作 validation。
    - 若 training years 只有 1 年：在該年內部做 train_test_split。
      這不會使用 outer testing year，因此不造成未來資料洩漏。
    """
    if len(train_years) >= 2:
        inner_train_years = train_years[:-1]
        inner_valid_years = [train_years[-1]]

        inner_train_df = df[df[YEAR_COL].isin(inner_train_years)].copy()
        inner_valid_df = df[df[YEAR_COL].isin(inner_valid_years)].copy()

        method = (
            f"last_training_year_validation: "
            f"inner_train={inner_train_years}, inner_valid={inner_valid_years}"
        )

        return inner_train_df, inner_valid_df, method

    one_year_df = df[df[YEAR_COL].isin(train_years)].copy()

    inner_train_df, inner_valid_df = train_test_split(
        one_year_df,
        test_size=0.25,
        random_state=RANDOM_SEED + split_id,
        shuffle=True,
    )

    method = (
        f"single_training_year_random_holdout: "
        f"year={train_years[0]}, test_size=0.25"
    )

    return inner_train_df.copy(), inner_valid_df.copy(), method


def train_svr_ga_one_split(
    df: pd.DataFrame,
    split_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    split_id = int(split_row["split_id"])
    train_years = parse_years(split_row["train_years"])
    test_years = parse_years(split_row["test_years"])

    print("")
    print("============================================================")
    print(f"[Info] SVR-GA split {split_id}")
    print(f"[Info] outer train years: {train_years}")
    print(f"[Info] outer test years: {test_years}")
    print("============================================================")

    outer_train_df = df[df[YEAR_COL].isin(train_years)].copy()
    test_df = df[df[YEAR_COL].isin(test_years)].copy()

    inner_train_df, inner_valid_df, inner_method = make_inner_train_valid(
        df=df,
        train_years=train_years,
        split_id=split_id,
    )

    X_inner_train = inner_train_df[FEATURE_COLUMNS].copy()
    y_inner_train = inner_train_df[TARGET_RETURN].astype(float).copy()

    X_inner_valid = inner_valid_df[FEATURE_COLUMNS].copy()
    y_inner_valid = inner_valid_df[TARGET_RETURN].astype(float).copy()

    best_result, search_log_df = run_svr_ga_search(
        X_train=X_inner_train,
        y_train=y_inner_train,
        X_valid=X_inner_valid,
        y_valid=y_inner_valid,
        split_id=split_id,
        random_seed=RANDOM_SEED + split_id,
    )

    print(
        f"[Info] Best SVR-GA split {split_id}: "
        f"C={best_result.C:.6f}, "
        f"gamma={best_result.gamma:.6f}, "
        f"epsilon={best_result.epsilon:.6f}, "
        f"validation_rmse={best_result.validation_rmse:.6f}"
    )

    X_outer_train = outer_train_df[FEATURE_COLUMNS].copy()
    y_outer_train = outer_train_df[TARGET_RETURN].astype(float).copy()

    X_test = test_df[FEATURE_COLUMNS].copy()

    final_pipeline = build_svr_pipeline(
        C=best_result.C,
        gamma=best_result.gamma,
        epsilon=best_result.epsilon,
    )

    # 重要：final model 只用 outer training years fit。
    final_pipeline.fit(X_outer_train, y_outer_train)

    predicted_return = final_pipeline.predict(X_test)

    prediction_df = test_df[
        [
            YEAR_COL,
            STOCK_ID_COL,
            STOCK_NAME_COL,
            TARGET_CLASS,
            TARGET_RETURN,
        ]
    ].copy()

    prediction_df.insert(0, "split_id", split_id)
    prediction_df.insert(1, "model_name", SVR_GA_MODEL_NAME)
    prediction_df.insert(2, "train_years", ",".join(map(str, train_years)))
    prediction_df.insert(3, "test_years", ",".join(map(str, test_years)))

    prediction_df = prediction_df.rename(
        columns={
            TARGET_CLASS: "actual_label",
            TARGET_RETURN: "actual_return",
        }
    )

    prediction_df["predicted_return"] = predicted_return

    prediction_df = prediction_df[
        [
            "split_id",
            "model_name",
            "train_years",
            "test_years",
            YEAR_COL,
            STOCK_ID_COL,
            STOCK_NAME_COL,
            "actual_label",
            "actual_return",
            "predicted_return",
        ]
    ].reset_index(drop=True)

    saved_model_dir = Path(SAVED_MODEL_DIR)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model_path = saved_model_dir / f"{SVR_GA_MODEL_NAME}_split_{split_id:02d}.joblib"
    joblib.dump(final_pipeline, model_path)

    summary_record = {
        "split_id": split_id,
        "train_years": ",".join(map(str, train_years)),
        "test_years": ",".join(map(str, test_years)),
        "inner_validation_method": inner_method,
        "best_C": best_result.C,
        "best_gamma": best_result.gamma,
        "best_epsilon": best_result.epsilon,
        "best_validation_rmse": best_result.validation_rmse,
        "best_validation_mae": best_result.validation_mae,
        "best_validation_r2": best_result.validation_r2,
        "n_outer_train": len(outer_train_df),
        "n_test": len(test_df),
        "saved_model_path": str(model_path),
    }

    return prediction_df, search_log_df, summary_record


def generate_step6_profile(
    summary_df: pd.DataFrame,
    regression_metrics_df: pd.DataFrame,
) -> str:
    lines = []

    lines.append("========== Step 6 SVR-GA Regression Profile ==========")
    lines.append("")
    lines.append("Model:")
    lines.append(f"- {SVR_GA_MODEL_NAME}")
    lines.append("")
    lines.append("Target:")
    lines.append("- Regression target: Return")
    lines.append("- Portfolio ranking score: predicted_return")
    lines.append("")
    lines.append("Validation design:")
    lines.append("- Outer validation: next-year expanding-window temporal validation")
    lines.append("- GA hyperparameter search uses only outer training years")
    lines.append("- Testing year is not used in GA search or model fitting")
    lines.append("")
    lines.append("Split summary:")
    lines.append(summary_df.to_string(index=False))
    lines.append("")
    lines.append("Regression metrics:")
    lines.append(regression_metrics_df.to_string(index=False))
    lines.append("")
    lines.append("Generated files:")
    lines.append(f"- Predictions: {SVR_GA_PREDICTIONS_PATH}")
    lines.append(f"- Regression metrics: {SVR_GA_REGRESSION_METRICS_PATH}")
    lines.append(f"- GA search log: {SVR_GA_SEARCH_LOG_PATH}")
    lines.append("")
    lines.append("========== End of Step 6 Profile ==========")

    return "\n".join(lines)


def main() -> None:
    print("========== Step 6: SVR-GA Regression ==========")

    print(f"[Info] Loading cleaned data: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] cleaned data shape: {df.shape}")

    print(f"[Info] Loading temporal splits: {TEMPORAL_SPLITS_NEXT_YEAR_PATH}")
    splits_df = pd.read_csv(TEMPORAL_SPLITS_NEXT_YEAR_PATH)
    print(f"[Info] splits shape: {splits_df.shape}")

    all_predictions = []
    all_search_logs = []
    summary_records = []

    for _, split_row in splits_df.iterrows():
        pred_df, search_log_df, summary_record = train_svr_ga_one_split(
            df=df,
            split_row=split_row,
        )

        all_predictions.append(pred_df)
        all_search_logs.append(search_log_df)
        summary_records.append(summary_record)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    search_log_all_df = pd.concat(all_search_logs, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)

    print("")
    print("[Info] Calculating regression metrics...")
    regression_metrics_df = calculate_regression_metrics_by_model(predictions_df)

    print("[Info] Saving outputs...")
    save_dataframe(predictions_df, SVR_GA_PREDICTIONS_PATH)
    save_dataframe(regression_metrics_df, SVR_GA_REGRESSION_METRICS_PATH)
    save_dataframe(search_log_all_df, SVR_GA_SEARCH_LOG_PATH)

    profile_text = generate_step6_profile(
        summary_df=summary_df,
        regression_metrics_df=regression_metrics_df,
    )

    save_text(profile_text, SVR_GA_STEP6_PROFILE_PATH)

    print("")
    print(profile_text)
    print("")
    print("========== Step 6 SVR-GA Training Finished Successfully ==========")


if __name__ == "__main__":
    main()