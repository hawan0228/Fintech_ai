# step6_validate_svr_ga.py
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import *
from src.schema import *


def validate_predictions(pred_df: pd.DataFrame) -> None:
    print("\n========== 1. Prediction validation ==========")

    expected_rows = 11 * 200

    assert len(pred_df) == expected_rows, (
        f"SVR-GA predictions 應有 {expected_rows} rows，目前是 {len(pred_df)}"
    )

    assert pred_df["model_name"].nunique() == 1
    assert pred_df["model_name"].iloc[0] == SVR_GA_MODEL_NAME

    assert pred_df[YEAR_COL].min() == 1998
    assert pred_df[YEAR_COL].max() == 2008
    assert pred_df[YEAR_COL].nunique() == 11

    year_counts = pred_df.groupby(YEAR_COL).size()
    bad_years = year_counts[year_counts != 200]
    assert bad_years.empty, f"每個 testing year 應有 200 筆：\n{bad_years}"

    assert pred_df["actual_return"].notna().all()
    assert pred_df["predicted_return"].notna().all()

    print("[Pass] SVR-GA predictions valid.")
    print(f"[Info] predictions shape: {pred_df.shape}")


def validate_saved_models() -> None:
    print("\n========== 2. Saved model validation ==========")

    saved_model_dir = Path(SAVED_MODEL_DIR)
    model_files = sorted(saved_model_dir.glob(f"{SVR_GA_MODEL_NAME}_split_*.joblib"))

    assert len(model_files) == 11, (
        f"SVR-GA 應有 11 個 saved models，目前是 {len(model_files)}"
    )

    for model_path in model_files:
        pipeline = joblib.load(model_path)
        assert hasattr(pipeline, "named_steps"), f"{model_path} 不是 sklearn Pipeline"
        assert "imputer" in pipeline.named_steps, f"{model_path} 缺少 imputer"
        assert "scaler" in pipeline.named_steps, f"{model_path} 缺少 scaler"
        assert "model" in pipeline.named_steps, f"{model_path} 缺少 model"

    print("[Pass] SVR-GA saved models valid.")
    print(f"[Info] model files: {len(model_files)}")


def validate_ga_search_log(search_log_df: pd.DataFrame) -> None:
    print("\n========== 3. GA search log validation ==========")

    required_cols = [
        "split_id",
        "generation",
        "individual_id",
        "C",
        "gamma",
        "epsilon",
        "validation_rmse",
        "validation_mae",
        "validation_r2",
        "is_valid",
    ]

    missing = [col for col in required_cols if col not in search_log_df.columns]
    assert not missing, f"GA search log 缺少欄位：{missing}"

    expected_min_rows = 11 * SVR_GA_POPULATION_SIZE * SVR_GA_N_GENERATIONS

    assert len(search_log_df) == expected_min_rows, (
        f"GA search log rows 應為 {expected_min_rows}，目前是 {len(search_log_df)}"
    )

    assert search_log_df["validation_rmse"].notna().all()
    assert (search_log_df["validation_rmse"] >= 0).all()

    print("[Pass] GA search log valid.")
    print(f"[Info] search log shape: {search_log_df.shape}")


def reconstruct_selected_from_predictions(
    pred_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for (model_name, year), group_df in pred_df.groupby(["model_name", YEAR_COL]):
        group_df = group_df.copy()

        group_df = group_df.sort_values(
            by=["predicted_return", STOCK_ID_COL],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        group_df["rank"] = group_df.index + 1

        for top_k in TOP_K_LIST:
            selected = group_df.head(top_k).copy()
            selected["top_k"] = int(top_k)
            records.append(selected)

    reconstructed = pd.concat(records, ignore_index=True)

    return reconstructed


def validate_selection_no_leakage(
    pred_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> None:
    print("\n========== 4. Top-K selection leakage validation ==========")

    unique_selected = selected_df.drop_duplicates(
        subset=["model_name", YEAR_COL, "top_k", "rank", STOCK_ID_COL]
    ).copy()

    reconstructed = reconstruct_selected_from_predictions(pred_df)

    key_cols = ["model_name", YEAR_COL, "top_k", "rank", STOCK_ID_COL]

    left = unique_selected[key_cols].sort_values(key_cols).reset_index(drop=True)
    right = reconstructed[key_cols].sort_values(key_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right, check_dtype=False)

    print("[Pass] SVR-GA selected stocks 可由 predicted_return + 股票代碼完整重建。")
    print("[Pass] 未偵測到 actual_return selection leakage。")


def validate_portfolio_recomputation(
    selected_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> None:
    print("\n========== 5. Portfolio return recomputation validation ==========")

    group_cols = [
        "model_name",
        YEAR_COL,
        "top_k",
        "weight_method",
    ]

    selected_df = selected_df.copy()
    selected_df["actual_return_decimal"] = selected_df["actual_return"] / 100.0
    selected_df["recomputed_contribution"] = (
        selected_df["weight"] * selected_df["actual_return_decimal"]
    )

    recomputed = (
        selected_df.groupby(group_cols, as_index=False)
        .agg(
            recomputed_gross_return=("recomputed_contribution", "sum"),
            recomputed_n_selected=(STOCK_ID_COL, "count"),
            recomputed_weight_sum=("weight", "sum"),
        )
    )

    recomputed["recomputed_net_return"] = (
        recomputed["recomputed_gross_return"] - ROUND_TRIP_COST
    )

    merged = returns_df.merge(
        recomputed,
        on=group_cols,
        how="left",
        validate="one_to_one",
    )

    assert np.allclose(
        merged["gross_return"],
        merged["recomputed_gross_return"],
        atol=1e-10,
    )

    assert np.allclose(
        merged["net_return"],
        merged["recomputed_net_return"],
        atol=1e-10,
    )

    bad_weight_sum = merged[(merged["recomputed_weight_sum"] - 1.0).abs() > 1e-8]
    assert bad_weight_sum.empty, f"權重加總錯誤：\n{bad_weight_sum}"

    bad_count = merged[
        merged["recomputed_n_selected"] != merged["top_k"].astype(int)
    ]
    assert bad_count.empty, f"選股數量不等於 top_k：\n{bad_count}"

    print("[Pass] SVR-GA portfolio returns 可由 selected stocks 完整重算。")


def validate_metrics(
    regression_metrics_df: pd.DataFrame,
    portfolio_metrics_df: pd.DataFrame,
) -> None:
    print("\n========== 6. Metrics validation ==========")

    assert len(regression_metrics_df) == 12, (
        "Regression metrics 應包含 11 splits + 1 overall。"
    )

    regression_required_cols = [
        "mae",
        "rmse",
        "r2",
        "pearson_corr",
        "spearman_corr",
    ]

    for col in regression_required_cols:
        assert col in regression_metrics_df.columns, f"缺少 regression metric 欄位：{col}"

    expected_portfolio_rows = len(TOP_K_LIST) * 2

    assert len(portfolio_metrics_df) == expected_portfolio_rows, (
        f"SVR-GA portfolio metrics 應有 {expected_portfolio_rows} rows，"
        f"目前是 {len(portfolio_metrics_df)}"
    )

    print("[Pass] SVR-GA regression metrics and portfolio metrics valid.")


def main() -> None:
    print("============================================================")
    print("Step 6 SVR-GA Validation")
    print("============================================================")

    pred_df = pd.read_csv(SVR_GA_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str})
    regression_metrics_df = pd.read_csv(SVR_GA_REGRESSION_METRICS_PATH)
    search_log_df = pd.read_csv(SVR_GA_SEARCH_LOG_PATH)
    selected_df = pd.read_csv(SVR_GA_SELECTED_STOCKS_PATH, dtype={STOCK_ID_COL: str})
    returns_df = pd.read_csv(SVR_GA_PORTFOLIO_RETURNS_PATH)
    portfolio_metrics_df = pd.read_csv(SVR_GA_PORTFOLIO_METRICS_PATH)

    validate_predictions(pred_df)
    validate_saved_models()
    validate_ga_search_log(search_log_df)
    validate_selection_no_leakage(pred_df, selected_df)
    validate_portfolio_recomputation(selected_df, returns_df)
    validate_metrics(regression_metrics_df, portfolio_metrics_df)

    print("")
    print("============================================================")
    print("[Pass] Step 6 SVR-GA validation completed successfully.")
    print("============================================================")


if __name__ == "__main__":
    main()