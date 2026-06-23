from pathlib import Path

import pandas as pd

from src.config import (
    ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH,
    SVR_GA_PORTFOLIO_METRICS_PATH,
    ALL_MODELS_PORTFOLIO_METRICS_PATH,
)


def main() -> None:
    print("========== Rebuild All Models Portfolio Metrics ==========")

    classification_path = Path(ALL_CLASSIFICATION_PORTFOLIO_METRICS_PATH)
    svr_path = Path(SVR_GA_PORTFOLIO_METRICS_PATH)
    output_path = Path(ALL_MODELS_PORTFOLIO_METRICS_PATH)

    if not classification_path.exists():
        raise FileNotFoundError(
            f"找不到 classification portfolio metrics: {classification_path}"
        )

    if not svr_path.exists():
        raise FileNotFoundError(
            f"找不到 SVR-GA portfolio metrics: {svr_path}"
        )

    classification_df = pd.read_csv(classification_path)
    svr_df = pd.read_csv(svr_path)

    all_models_df = pd.concat(
        [classification_df, svr_df],
        ignore_index=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_models_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[Info] classification metrics shape: {classification_df.shape}")
    print(f"[Info] svr-ga metrics shape: {svr_df.shape}")
    print(f"[Info] all models metrics shape: {all_models_df.shape}")
    print(f"[Saved] {output_path}")

    print("")
    print("Models in all_models_portfolio_metrics.csv:")
    print(all_models_df["model_name"].value_counts().to_string())

    print("")
    print("Top-10 Equal-weight summary:")
    check = all_models_df[
        (all_models_df["top_k"].astype(int) == 10)
        & (all_models_df["weight_method"] == "equal")
    ].copy()

    display_cols = [
        "model_name",
        "top_k",
        "weight_method",
        "net_annualized_return",
        "net_cumulative_return",
        "net_maximum_drawdown",
        "net_volatility",
        "net_sharpe_ratio",
        "net_win_rate",
    ]

    print(check[display_cols].to_string(index=False))

    print("")
    print("========== Rebuild Finished ==========")


if __name__ == "__main__":
    main()