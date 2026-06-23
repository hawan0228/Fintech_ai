from pathlib import Path

import pandas as pd

from src.config import (
    TOP_K_LIST,
    TASK2_PREDICTIONS_PATH,
    TASK2_SELECTED_STOCKS_PATH,
    TASK2_PORTFOLIO_RETURNS_PATH,
    TASK2_PORTFOLIO_METRICS_PATH,
    DT_PORTFOLIO_METRICS_PATH,
    ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH,
)
from src.schema import STOCK_ID_COL
from src.portfolio import select_top_k_stocks, calculate_portfolio_returns
from src.metrics import summarize_strategy_metrics


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def maybe_save_all_portfolio_metrics(
    task2_portfolio_metrics_df: pd.DataFrame,
) -> None:
    """
    若 Step 4 Decision Tree portfolio metrics 已存在，
    則合併成 all_classification_portfolio_metrics.csv。
    """
    dt_path = Path(DT_PORTFOLIO_METRICS_PATH)

    if not dt_path.exists():
        print("[Warning] 找不到 Decision Tree portfolio metrics，暫不輸出 all classification portfolio metrics。")
        return

    dt_metrics_df = pd.read_csv(dt_path)

    all_metrics_df = pd.concat(
        [dt_metrics_df, task2_portfolio_metrics_df],
        ignore_index=True,
    )

    save_dataframe(all_metrics_df, ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH)
    print(f"[Saved] {ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH}")


def main() -> None:
    print("========== Step 5: Build Task 2 Portfolios ==========")

    print(f"[Info] Loading Task 2 predictions from: {TASK2_PREDICTIONS_PATH}")
    predictions_df = pd.read_csv(TASK2_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] Predictions shape: {predictions_df.shape}")

    print("[Info] Selecting Top-K stocks...")
    selected_df = select_top_k_stocks(
        predictions_df=predictions_df,
        top_k_list=TOP_K_LIST,
        score_col="score_label_1",
    )

    print("[Info] Calculating portfolio returns...")
    selected_with_weights_df, portfolio_returns_df = calculate_portfolio_returns(
        selected_df=selected_df,
        weight_methods=["equal", "score"],
        score_col="score_label_1",
    )

    print("[Info] Calculating portfolio metrics...")
    portfolio_metrics_df = summarize_strategy_metrics(
        returns_df=portfolio_returns_df,
        group_cols=["model_name", "top_k", "weight_method"],
    )

    print("[Info] Saving outputs...")
    save_dataframe(selected_with_weights_df, TASK2_SELECTED_STOCKS_PATH)
    save_dataframe(portfolio_returns_df, TASK2_PORTFOLIO_RETURNS_PATH)
    save_dataframe(portfolio_metrics_df, TASK2_PORTFOLIO_METRICS_PATH)

    maybe_save_all_portfolio_metrics(portfolio_metrics_df)

    print("")
    print("Task 2 portfolio metrics:")
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
    print("========== Step 5 Portfolio Construction Finished Successfully ==========")


if __name__ == "__main__":
    main()