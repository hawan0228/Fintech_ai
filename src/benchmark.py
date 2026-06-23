from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, ROUND_TRIP_COST, RANDOM_BENCHMARK_N_RUNS
from src.schema import YEAR_COL, TARGET_RETURN, STOCK_ID_COL, STOCK_NAME_COL


def calculate_all_stock_benchmark(
    cleaned_df: pd.DataFrame,
    testing_years: list[int],
    transaction_cost: float = ROUND_TRIP_COST,
) -> pd.DataFrame:
    """
    All-stock equal-weight benchmark。

    每年等權買入該年度所有 200 檔股票。
    Return 單位：
    - 原始 Return 是百分比
    - 計算時除以 100
    """
    records = []

    for year in sorted(testing_years):
        year_df = cleaned_df[cleaned_df[YEAR_COL] == year].copy()

        if year_df.empty:
            raise ValueError(f"cleaned data 找不到 year={year} 的資料。")

        gross_return = (year_df[TARGET_RETURN] / 100.0).mean()
        net_return = gross_return - transaction_cost

        records.append(
            {
                "benchmark_name": "all_stock_equal_weight",
                YEAR_COL: int(year),
                "n_stocks": len(year_df),
                "gross_return": gross_return,
                "net_return": net_return,
                "transaction_cost": transaction_cost,
                "avg_stock_return_percent": year_df[TARGET_RETURN].mean(),
            }
        )

    return pd.DataFrame(records)


def calculate_random_topk_benchmark_runs(
    cleaned_df: pd.DataFrame,
    testing_years: list[int],
    top_k_list: list[int],
    n_runs: int = RANDOM_BENCHMARK_N_RUNS,
    random_seed: int = RANDOM_SEED,
    transaction_cost: float = ROUND_TRIP_COST,
) -> pd.DataFrame:
    """
    Random Top-K benchmark。

    每個 run、每個 year 隨機選 K 檔股票，計算 equal-weight return。
    """
    rng = np.random.default_rng(random_seed)

    records = []

    for run_id in range(1, n_runs + 1):
        for year in sorted(testing_years):
            year_df = cleaned_df[cleaned_df[YEAR_COL] == year].copy()

            if year_df.empty:
                raise ValueError(f"cleaned data 找不到 year={year} 的資料。")

            for top_k in top_k_list:
                if len(year_df) < top_k:
                    raise ValueError(
                        f"Year {year} 只有 {len(year_df)} 檔股票，無法隨機選 Top-{top_k}。"
                    )

                sampled_idx = rng.choice(year_df.index.to_numpy(), size=top_k, replace=False)
                sampled_df = year_df.loc[sampled_idx].copy()

                gross_return = (sampled_df[TARGET_RETURN] / 100.0).mean()
                net_return = gross_return - transaction_cost

                records.append(
                    {
                        "benchmark_name": "random_topk_equal_weight",
                        "random_run": run_id,
                        YEAR_COL: int(year),
                        "top_k": int(top_k),
                        "n_selected": len(sampled_df),
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "transaction_cost": transaction_cost,
                        "avg_selected_return_percent": sampled_df[TARGET_RETURN].mean(),
                        "selected_tickers": ",".join(
                            sampled_df[STOCK_ID_COL].astype(str).tolist()
                        ),
                        "selected_names": ",".join(
                            sampled_df[STOCK_NAME_COL].astype(str).tolist()
                        ),
                    }
                )

    return pd.DataFrame(records)


def summarize_random_benchmark_annual_mean(
    random_runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    對 Random Top-K benchmark 取每年、每個 K 的平均 annual return。
    這個表可用來畫圖或與模型年度報酬比較。
    """
    annual_mean_df = (
        random_runs_df.groupby([YEAR_COL, "top_k"], as_index=False)
        .agg(
            gross_return=("gross_return", "mean"),
            net_return=("net_return", "mean"),
            n_runs=("random_run", "nunique"),
            n_selected=("n_selected", "mean"),
        )
        .sort_values(["top_k", YEAR_COL])
        .reset_index(drop=True)
    )

    annual_mean_df["benchmark_name"] = "random_topk_mean"
    annual_mean_df["weight_method"] = "equal"

    return annual_mean_df