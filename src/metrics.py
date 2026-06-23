from __future__ import annotations

import pandas as pd
from sklearn.metrics import *

import numpy as np
import pandas as pd


def calculate_one_classification_metric_block(
    df: pd.DataFrame,
    scope: str,
    split_id: int | str,
    test_years: str,
) -> dict:
    """
    計算一組 classification metrics。
    """
    y_true = df["actual_label"].astype(int)
    y_pred = df["predicted_label"].astype(int)

    record = {
        "scope": scope,
        "split_id": split_id,
        "test_years": test_years,
        "n_samples": len(df),
        "actual_label_neg1_count": int((y_true == -1).sum()),
        "actual_label_pos1_count": int((y_true == 1).sum()),
        "pred_label_neg1_count": int((y_pred == -1).sum()),
        "pred_label_pos1_count": int((y_pred == 1).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_label_1": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_label_1": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_label_1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
    }

    return record


def calculate_classification_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    計算每個 split 與整體的分類指標。
    """
    records = []

    for split_id, group_df in predictions_df.groupby("split_id"):
        test_years = str(group_df["test_years"].iloc[0])

        record = calculate_one_classification_metric_block(
            df=group_df,
            scope="split",
            split_id=int(split_id),
            test_years=test_years,
        )

        records.append(record)

    overall_record = calculate_one_classification_metric_block(
        df=predictions_df,
        scope="overall",
        split_id="overall",
        test_years="1998-2008",
    )

    records.append(overall_record)

    metrics_df = pd.DataFrame(records)

    return metrics_df


def calculate_cumulative_return(returns: pd.Series) -> float:
    """
    Cumulative Return = product(1 + R_t) - 1
    """
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) == 0:
        return np.nan

    return float((1.0 + returns).prod() - 1.0)


def calculate_annualized_return(returns: pd.Series) -> float:
    """
    Annualized Return = product(1 + R_t)^(1/T) - 1
    """
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) == 0:
        return np.nan

    cumulative_growth = (1.0 + returns).prod()

    if cumulative_growth <= 0:
        return np.nan

    n_years = len(returns)

    return float(cumulative_growth ** (1.0 / n_years) - 1.0)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown based on annual equity curve.
    """
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) == 0:
        return np.nan

    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0

    return float(drawdown.min())


def calculate_volatility(returns: pd.Series) -> float:
    """
    Annual return volatility.
    """
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) <= 1:
        return 0.0

    return float(returns.std(ddof=1))


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Sharpe Ratio using annual returns.

    本研究採簡化設定 risk_free_rate = 0。
    """
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) <= 1:
        return np.nan

    excess_returns = returns - risk_free_rate
    volatility = excess_returns.std(ddof=1)

    if volatility == 0:
        return np.nan

    return float(excess_returns.mean() / volatility)


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Win Rate = proportion of years with positive returns.
    """
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) == 0:
        return np.nan

    return float((returns > 0).mean())


def summarize_return_series(
    returns: pd.Series,
    return_type: str,
) -> dict:
    """
    對一條 annual return series 計算完整績效指標。
    """
    returns = pd.Series(returns).dropna().astype(float)

    return {
        f"{return_type}_mean_annual_return": float(returns.mean()) if len(returns) else np.nan,
        f"{return_type}_cumulative_return": calculate_cumulative_return(returns),
        f"{return_type}_annualized_return": calculate_annualized_return(returns),
        f"{return_type}_maximum_drawdown": calculate_max_drawdown(returns),
        f"{return_type}_volatility": calculate_volatility(returns),
        f"{return_type}_sharpe_ratio": calculate_sharpe_ratio(returns),
        f"{return_type}_win_rate": calculate_win_rate(returns),
    }


def summarize_strategy_metrics(
    returns_df: pd.DataFrame,
    group_cols: list[str],
    strategy_name_col: str | None = None,
) -> pd.DataFrame:
    """
    對不同策略分組計算 gross 與 net performance metrics。

    returns_df 必須包含：
    - gross_return
    - net_return
    """
    required_cols = [*group_cols, "gross_return", "net_return"]

    missing = [col for col in required_cols if col not in returns_df.columns]
    if missing:
        raise ValueError(f"returns_df 缺少必要欄位：{missing}")

    records = []

    for group_key, group_df in returns_df.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        record = {
            col: value for col, value in zip(group_cols, group_key)
        }

        record["n_years"] = len(group_df)

        gross_metrics = summarize_return_series(
            group_df.sort_values("year")["gross_return"],
            return_type="gross",
        )

        net_metrics = summarize_return_series(
            group_df.sort_values("year")["net_return"],
            return_type="net",
        )

        record.update(gross_metrics)
        record.update(net_metrics)

        records.append(record)

    metrics_df = pd.DataFrame(records)

    return metrics_df


def calculate_classification_metrics_by_model(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    計算多模型分類指標。

    輸出包含：
    1. 每個 model / split 的 metrics
    2. 每個 model 的 overall metrics
    """
    required_columns = [
        "model_name",
        "split_id",
        "test_years",
        "year",
        "actual_label",
        "predicted_label",
    ]

    missing = [col for col in required_columns if col not in predictions_df.columns]
    if missing:
        raise ValueError(f"predictions_df 缺少必要欄位：{missing}")

    records = []

    # =========================
    # 1. Metrics by model and split
    # =========================
    for (model_name, split_id), group_df in predictions_df.groupby(
        ["model_name", "split_id"]
    ):
        test_years = str(group_df["test_years"].iloc[0])

        metric_record = calculate_one_classification_metric_block(
            df=group_df,
            scope="split",
            split_id=int(split_id),
            test_years=test_years,
        )

        # dict 不能用 insert，所以用新的 dict 合併
        record = {
            "model_name": model_name,
            **metric_record,
        }

        records.append(record)

    # =========================
    # 2. Overall metrics by model
    # =========================
    for model_name, group_df in predictions_df.groupby("model_name"):
        min_year = int(group_df["year"].min())
        max_year = int(group_df["year"].max())

        metric_record = calculate_one_classification_metric_block(
            df=group_df,
            scope="overall",
            split_id="overall",
            test_years=f"{min_year}-{max_year}",
        )

        record = {
            "model_name": model_name,
            **metric_record,
        }

        records.append(record)

    metrics_df = pd.DataFrame(records)

    # =========================
    # 3. Sort output clearly
    # =========================
    scope_order_map = {
        "split": 0,
        "overall": 1,
    }

    metrics_df["_scope_order"] = metrics_df["scope"].map(scope_order_map)

    def _split_sort_value(value):
        if str(value) == "overall":
            return 999
        return int(value)

    metrics_df["_split_sort"] = metrics_df["split_id"].apply(_split_sort_value)

    metrics_df = metrics_df.sort_values(
        ["model_name", "_scope_order", "_split_sort"]
    ).reset_index(drop=True)

    metrics_df = metrics_df.drop(columns=["_scope_order", "_split_sort"])

    return metrics_df


def calculate_one_regression_metric_block(
    df: pd.DataFrame,
    scope: str,
    split_id: int | str,
    test_years: str,
) -> dict:
    y_true = df["actual_return"].astype(float).to_numpy()
    y_pred = df["predicted_return"].astype(float).to_numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan

    if len(np.unique(y_true)) <= 1 or len(np.unique(y_pred)) <= 1:
        pearson_corr = np.nan
        spearman_corr = np.nan
    else:
        pearson_corr = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))
        spearman_corr = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))

    return {
        "scope": scope,
        "split_id": split_id,
        "test_years": test_years,
        "n_samples": len(df),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
    }


def calculate_regression_metrics_by_model(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "model_name",
        "split_id",
        "test_years",
        "year",
        "actual_return",
        "predicted_return",
    ]

    missing = [col for col in required_columns if col not in predictions_df.columns]
    if missing:
        raise ValueError(f"regression predictions 缺少必要欄位：{missing}")

    records = []

    for (model_name, split_id), group_df in predictions_df.groupby(
        ["model_name", "split_id"]
    ):
        test_years = str(group_df["test_years"].iloc[0])

        metric_record = calculate_one_regression_metric_block(
            df=group_df,
            scope="split",
            split_id=int(split_id),
            test_years=test_years,
        )

        records.append(
            {
                "model_name": model_name,
                **metric_record,
            }
        )

    for model_name, group_df in predictions_df.groupby("model_name"):
        min_year = int(group_df["year"].min())
        max_year = int(group_df["year"].max())

        metric_record = calculate_one_regression_metric_block(
            df=group_df,
            scope="overall",
            split_id="overall",
            test_years=f"{min_year}-{max_year}",
        )

        records.append(
            {
                "model_name": model_name,
                **metric_record,
            }
        )

    metrics_df = pd.DataFrame(records)

    scope_order = {
        "split": 0,
        "overall": 1,
    }

    metrics_df["_scope_order"] = metrics_df["scope"].map(scope_order)
    metrics_df["_split_sort"] = metrics_df["split_id"].apply(
        lambda x: 999 if str(x) == "overall" else int(x)
    )

    metrics_df = metrics_df.sort_values(
        ["model_name", "_scope_order", "_split_sort"]
    ).reset_index(drop=True)

    metrics_df = metrics_df.drop(columns=["_scope_order", "_split_sort"])

    return metrics_df