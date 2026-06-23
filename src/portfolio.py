from __future__ import annotations

import pandas as pd

from src.config import ROUND_TRIP_COST
from src.schema import STOCK_ID_COL, STOCK_NAME_COL, YEAR_COL


def validate_prediction_input(predictions_df: pd.DataFrame) -> None:
    """
    檢查 Step 3 prediction file 是否包含 Step 4 必要欄位。
    """
    required_columns = [
        "split_id",
        "model_name",
        "train_years",
        "test_years",
        YEAR_COL,
        STOCK_ID_COL,
        STOCK_NAME_COL,
        "actual_label",
        "predicted_label",
        "score_label_1",
        "actual_return",
    ]

    missing = [col for col in required_columns if col not in predictions_df.columns]

    if missing:
        raise ValueError(f"prediction file 缺少必要欄位：{missing}")

    if predictions_df["score_label_1"].isna().any():
        raise ValueError("score_label_1 存在缺失值，無法進行 Top-K 選股。")

    if not predictions_df["score_label_1"].between(0, 1).all():
        raise ValueError("score_label_1 應介於 0 到 1。")

    if predictions_df["actual_return"].isna().any():
        raise ValueError("actual_return 存在缺失值，無法計算 portfolio return。")


def select_top_k_stocks(
    predictions_df: pd.DataFrame,
    top_k_list: list[int],
    score_col: str = "score_label_1",
) -> pd.DataFrame:
    """
    根據每個 testing year 的模型分數由高到低排序，選出 Top-K 股票。

    重要：
    - 不可使用 actual_return 作為排序或 tie-break 條件。
    - actual_return 是未來 realized return，只能在選股完成後用於績效計算。
    """
    validate_prediction_input(predictions_df)

    selected_records = []
    group_cols = ["model_name", YEAR_COL]

    for (model_name, year), group_df in predictions_df.groupby(group_cols):
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
                    f"Year {year}, model {model_name} 只有 {len(group_df)} 檔股票，"
                    f"不足以選 Top-{top_k}。"
                )

            selected = group_df.head(top_k).copy()
            selected["top_k"] = top_k
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
            "score_label_1",
            "actual_label",
            "predicted_label",
            "actual_return",
        ]
    ].copy()

    return selected_df

def _calculate_equal_weights(group_df: pd.DataFrame) -> pd.Series:
    """
    Equal-weight 權重。
    """
    n = len(group_df)
    return pd.Series([1.0 / n] * n, index=group_df.index)


def _calculate_score_weights(
    group_df: pd.DataFrame,
    score_col: str = "score_label_1",
) -> pd.Series:
    """
    Score-weighted 權重。

    分類模型使用 P(label=1) 作為分數。
    若所有分數加總為 0，則退回 equal-weight。
    """
    scores = group_df[score_col].clip(lower=0).astype(float)
    score_sum = scores.sum()

    if score_sum <= 0:
        return _calculate_equal_weights(group_df)

    return scores / score_sum


def calculate_portfolio_returns(
    selected_df: pd.DataFrame,
    weight_methods: list[str] | None = None,
    score_col: str = "score_label_1",
    transaction_cost: float = ROUND_TRIP_COST,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    將 Top-K 選股結果轉換為年度 portfolio returns。

    Return 單位注意：
    - 原始 actual_return 是百分比，例如 10 代表 10%。
    - 投資組合計算時必須除以 100，轉成 0.10。

    Returns
    -------
    selected_with_weights_df:
        每檔 selected stock 的權重與貢獻。
    portfolio_returns_df:
        每年 portfolio gross/net return。
    """
    if weight_methods is None:
        weight_methods = ["equal", "score"]

    valid_methods = {"equal", "score"}
    invalid = set(weight_methods) - valid_methods
    if invalid:
        raise ValueError(f"不支援的 weight_methods：{invalid}")

    selected_with_weights_records = []
    portfolio_return_records = []

    group_cols = ["model_name", YEAR_COL, "top_k"]

    for (model_name, year, top_k), group_df in selected_df.groupby(group_cols):
        group_df = group_df.copy().sort_values("rank")

        for weight_method in weight_methods:
            if weight_method == "equal":
                weights = _calculate_equal_weights(group_df)
            elif weight_method == "score":
                weights = _calculate_score_weights(group_df, score_col=score_col)
            else:
                raise ValueError(f"Unknown weight_method: {weight_method}")

            temp_df = group_df.copy()
            temp_df["weight_method"] = weight_method
            temp_df["weight"] = weights

            # actual_return 是百分比，必須 /100
            temp_df["actual_return_decimal"] = temp_df["actual_return"] / 100.0
            temp_df["return_contribution"] = (
                temp_df["weight"] * temp_df["actual_return_decimal"]
            )

            gross_return = temp_df["return_contribution"].sum()
            net_return = gross_return - transaction_cost

            temp_df["portfolio_gross_return"] = gross_return
            temp_df["portfolio_net_return"] = net_return
            temp_df["transaction_cost"] = transaction_cost

            selected_with_weights_records.append(temp_df)

            portfolio_return_records.append(
                {
                    "model_name": model_name,
                    YEAR_COL: int(year),
                    "top_k": int(top_k),
                    "weight_method": weight_method,
                    "n_selected": len(temp_df),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "transaction_cost": transaction_cost,
                    "avg_selected_return_percent": temp_df["actual_return"].mean(),
                    "min_selected_return_percent": temp_df["actual_return"].min(),
                    "max_selected_return_percent": temp_df["actual_return"].max(),
                    "sum_weight": temp_df["weight"].sum(),
                }
            )

    selected_with_weights_df = pd.concat(
        selected_with_weights_records,
        ignore_index=True,
    )

    portfolio_returns_df = pd.DataFrame(portfolio_return_records)
    portfolio_returns_df = portfolio_returns_df.sort_values(
        ["model_name", "top_k", "weight_method", YEAR_COL]
    ).reset_index(drop=True)

    return selected_with_weights_df, portfolio_returns_df