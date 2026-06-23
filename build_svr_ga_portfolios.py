# step6_build_svr_ga_portfolios.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    TOP_K_LIST,
    ROUND_TRIP_COST,
    SVR_GA_PREDICTIONS_PATH,
    SVR_GA_SELECTED_STOCKS_PATH,
    SVR_GA_PORTFOLIO_RETURNS_PATH,
    SVR_GA_PORTFOLIO_METRICS_PATH,
    ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH,
    ALL_MODELS_PORTFOLIO_METRICS_PATH,
)
from src.schema import (
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEAR_COL,
)
from src.metrics import summarize_strategy_metrics


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def validate_svr_predictions(predictions_df: pd.DataFrame) -> None:
    required_cols = [
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

    missing = [col for col in required_cols if col not in predictions_df.columns]
    if missing:
        raise ValueError(f"SVR-GA predictions 缺少必要欄位：{missing}")

    if predictions_df["predicted_return"].isna().any():
        raise ValueError("predicted_return 存在缺失值。")


def select_top_k_svr_ga_stocks(
    predictions_df: pd.DataFrame,
    top_k_list: list[int],
    score_col: str = "predicted_return",
) -> pd.DataFrame:
    """
    使用 predicted_return 排序選股。

    重要：
    - 不可使用 actual_return 排序。
    - actual_return 只能在選股完成後用於績效計算。
    """
    validate_svr_predictions(predictions_df)

    selected_records = []

    for (model_name, year), group_df in predictions_df.groupby(["model_name", YEAR_COL]):
        group_df = group_df.copy()

        group_df = group_df.sort_values(
            by=[score_col, STOCK_ID_COL],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        group_df["rank"] = group_df.index + 1

        for top_k in top_k_list:
            if len(group_df) < top_k:
                raise ValueError(
                    f"{model_name}, year={year} 只有 {len(group_df)} 檔股票，"
                    f"不足以選 Top-{top_k}。"
                )

            selected = group_df.head(top_k).copy()
            selected["top_k"] = int(top_k)
            selected_records.append(selected)

    selected_df = pd.concat(selected_records, ignore_index=True)

    selected_df = selected_df[
        [
            "split_id",
            "model_name",
            "train_years",
            "test_years",
            YEAR_COL,
            "top_k",
            "rank",
            STOCK_ID_COL,
            STOCK_NAME_COL,
            "predicted_return",
            "actual_label",
            "actual_return",
        ]
    ].copy()

    return selected_df


def calculate_equal_weights(n: int) -> np.ndarray:
    return np.repeat(1.0 / n, n)


def calculate_score_weights(
    scores: pd.Series,
) -> np.ndarray:
    """
    Regression score-weighted portfolio。

    predicted_return 可能為負，因此不能直接除以總和。
    這裡使用 shift-to-positive：
        adjusted_score = score - min(score)
    若全部分數相同，則退回 equal weight。
    """
    values = scores.astype(float).to_numpy()

    adjusted = values - np.min(values)

    if np.allclose(adjusted.sum(), 0.0):
        return calculate_equal_weights(len(values))

    adjusted = adjusted + 1e-12
    weights = adjusted / adjusted.sum()

    return weights


def calculate_svr_ga_portfolio_returns(
    selected_df: pd.DataFrame,
    weight_methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if weight_methods is None:
        weight_methods = ["equal", "score"]

    selected_with_weights = []
    annual_records = []

    group_cols = [
        "model_name",
        YEAR_COL,
        "top_k",
    ]

    for (model_name, year, top_k), group_df in selected_df.groupby(group_cols):
        group_df = group_df.copy()
        group_df = group_df.sort_values("rank").reset_index(drop=True)

        for weight_method in weight_methods:
            temp = group_df.copy()

            if weight_method == "equal":
                weights = calculate_equal_weights(len(temp))
            elif weight_method == "score":
                weights = calculate_score_weights(temp["predicted_return"])
            else:
                raise ValueError(f"Unknown weight_method: {weight_method}")

            temp["weight_method"] = weight_method
            temp["weight"] = weights

            temp["actual_return_decimal"] = temp["actual_return"] / 100.0
            temp["return_contribution"] = (
                temp["weight"] * temp["actual_return_decimal"]
            )

            gross_return = float(temp["return_contribution"].sum())
            net_return = gross_return - ROUND_TRIP_COST

            temp["gross_return"] = gross_return
            temp["net_return"] = net_return

            selected_with_weights.append(temp)

            annual_records.append(
                {
                    "model_name": model_name,
                    YEAR_COL: int(year),
                    "top_k": int(top_k),
                    "weight_method": weight_method,
                    "n_selected": len(temp),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "transaction_cost": ROUND_TRIP_COST,
                    "avg_predicted_return": float(temp["predicted_return"].mean()),
                    "avg_actual_return": float(temp["actual_return"].mean()),
                    "max_actual_return": float(temp["actual_return"].max()),
                    "min_actual_return": float(temp["actual_return"].min()),
                }
            )

    selected_with_weights_df = pd.concat(
        selected_with_weights,
        ignore_index=True,
    )

    portfolio_returns_df = pd.DataFrame(annual_records).sort_values(
        ["model_name", "top_k", "weight_method", YEAR_COL]
    ).reset_index(drop=True)

    return selected_with_weights_df, portfolio_returns_df


def maybe_save_all_models_portfolio_metrics(
    svr_metrics_df: pd.DataFrame,
) -> None:
    """
    若 all_classification_portfolio_metrics.csv 已存在，
    則合併 classification models + SVR-GA。
    """
    classification_path = Path(ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH)

    if not classification_path.exists():
        print("[Warning] 找不到 all_classification_portfolio_metrics.csv，暫不輸出 all_models_portfolio_metrics.csv。")
        return

    classification_df = pd.read_csv(classification_path)

    all_models_df = pd.concat(
        [classification_df, svr_metrics_df],
        ignore_index=True,
    )

    save_dataframe(all_models_df, ALL_MODELS_PORTFOLIO_METRICS_PATH)
    print(f"[Saved] {ALL_MODELS_PORTFOLIO_METRICS_PATH}")


def main() -> None:
    print("========== Step 6: Build SVR-GA Portfolios ==========")

    print(f"[Info] Loading SVR-GA predictions: {SVR_GA_PREDICTIONS_PATH}")
    predictions_df = pd.read_csv(SVR_GA_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] predictions shape: {predictions_df.shape}")

    print("[Info] Selecting Top-K stocks by predicted_return...")
    selected_df = select_top_k_svr_ga_stocks(
        predictions_df=predictions_df,
        top_k_list=TOP_K_LIST,
        score_col="predicted_return",
    )

    print("[Info] Calculating SVR-GA portfolio returns...")
    selected_with_weights_df, portfolio_returns_df = calculate_svr_ga_portfolio_returns(
        selected_df=selected_df,
        weight_methods=["equal", "score"],
    )

    print("[Info] Calculating SVR-GA portfolio metrics...")
    portfolio_metrics_df = summarize_strategy_metrics(
        returns_df=portfolio_returns_df,
        group_cols=["model_name", "top_k", "weight_method"],
    )

    print("[Info] Saving outputs...")
    save_dataframe(selected_with_weights_df, SVR_GA_SELECTED_STOCKS_PATH)
    save_dataframe(portfolio_returns_df, SVR_GA_PORTFOLIO_RETURNS_PATH)
    save_dataframe(portfolio_metrics_df, SVR_GA_PORTFOLIO_METRICS_PATH)

    maybe_save_all_models_portfolio_metrics(portfolio_metrics_df)

    print("")
    print("SVR-GA portfolio metrics:")
    display_cols = [
        "model_name",
        "top_k",
        "weight_method",
        "n_years",
        "net_annualized_return",
        "net_cumulative_return",
        "net_maximum_drawdown",
        "net_volatility",
        "net_sharpe_ratio",
        "net_win_rate",
    ]
    print(portfolio_metrics_df[display_cols].to_string(index=False))

    print("")
    print("========== Step 6 Portfolio Construction Finished Successfully ==========")


if __name__ == "__main__":
    main()