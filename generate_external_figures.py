# generate_external_figures.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.config import (
    EXTERNAL_CLEANED_DATA_PATH,
    EXTERNAL_FEATURE_MISSING_RATIO_FIG_PATH,
    EXTERNAL_RF_VS_BENCHMARK_FIG_PATH,
    EXTERNAL_TOPK_SENSITIVITY_FIG_PATH,
    EXTERNAL_OUTPUT_DIR,
)
from src.schema import (
    FEATURE_COLUMNS,
    STOCK_ID_COL,
)


# =========================
# Paths
# =========================

EXTERNAL_BENCHMARK_DIR = EXTERNAL_OUTPUT_DIR / "benchmarks"

EXTERNAL_RF_BENCHMARK_COMPARISON_PATH = (
    EXTERNAL_BENCHMARK_DIR / "external_rf_benchmark_comparison_aligned.csv"
)


# =========================
# Font setup
# =========================

def setup_chinese_font() -> None:
    """
    Fix matplotlib CJK font warning on Windows/macOS/Linux.

    Windows 常見可用：
    - Microsoft JhengHei
    - Microsoft YaHei
    - MingLiU

    macOS 常見：
    - PingFang TC
    - Heiti TC

    Linux 若有安裝：
    - Noto Sans CJK TC
    - Noto Sans CJK JP
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "MingLiU",
        "PingFang TC",
        "Heiti TC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# =========================
# Utilities
# =========================

def require_file(path: str | Path, hint: str = "") -> Path:
    path = Path(path)

    if not path.exists():
        message = f"找不到必要檔案：{path}"

        if hint:
            message += f"\n建議先執行：{hint}"

        raise FileNotFoundError(message)

    return path


def require_columns(df: pd.DataFrame, required_cols: list[str], file_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"{file_name} 缺少必要欄位：{missing}\n"
            f"目前欄位：{df.columns.tolist()}"
        )


def save_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Plot 1: Feature missing ratio
# =========================

def plot_feature_missing_ratio(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    available_feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]

    if not available_feature_cols:
        raise ValueError("external_cleaned_dataset.csv 中找不到 FEATURE_COLUMNS。")

    missing_ratio = (
        df[available_feature_cols]
        .isna()
        .mean()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        missing_ratio.index,
        missing_ratio.values * 100,
    )

    plt.title("External Dataset Feature Missing Ratio")
    plt.xlabel("Missing ratio")
    plt.ylabel("Feature")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=100))
    plt.grid(True, axis="x", alpha=0.3)

    save_figure(output_path)


# =========================
# Plot 2: RF vs benchmark annualized return
# =========================

def plot_external_rf_vs_benchmark(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    required_cols = [
        "top_k",
        "weight_method",
        "net_annualized_return",
        "all_stock_net_annualized_return",
        "random_net_annualized_return_mean",
        "random_net_annualized_return_p95",
    ]

    require_columns(
        comparison_df,
        required_cols,
        "external_rf_benchmark_comparison_aligned.csv",
    )

    df = comparison_df[
        comparison_df["weight_method"] == "equal"
    ].copy()

    if df.empty:
        raise ValueError(
            "comparison_df 中沒有 weight_method == 'equal' 的資料，無法畫圖。"
        )

    df["top_k"] = df["top_k"].astype(int)
    df = df.sort_values("top_k")

    x = list(range(len(df)))
    width = 0.22

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    ax.bar(
        [i - width for i in x],
        df["net_annualized_return"] * 100,
        width=width,
        label="External RF",
    )

    ax.bar(
        x,
        df["random_net_annualized_return_mean"] * 100,
        width=width,
        label="Random mean",
    )

    ax.bar(
        [i + width for i in x],
        df["random_net_annualized_return_p95"] * 100,
        width=width,
        label="Random p95",
    )

    all_stock_value = float(df["all_stock_net_annualized_return"].iloc[0]) * 100

    ax.axhline(
        all_stock_value,
        linestyle="--",
        linewidth=1.5,
        label="All-stock benchmark",
    )

    ax.set_title("External RF vs Benchmarks: Net Annualized Return")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Net annualized return")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in df["top_k"]])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_figure(output_path)


# =========================
# Plot 3: Top-K sensitivity
# =========================

def plot_external_topk_sensitivity(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    required_cols = [
        "top_k",
        "weight_method",
        "net_annualized_return",
        "all_stock_net_annualized_return",
        "random_net_annualized_return_mean",
        "random_net_annualized_return_p95",
    ]

    require_columns(
        comparison_df,
        required_cols,
        "external_rf_benchmark_comparison_aligned.csv",
    )

    df = comparison_df[
        comparison_df["weight_method"] == "equal"
    ].copy()

    if df.empty:
        raise ValueError(
            "comparison_df 中沒有 weight_method == 'equal' 的資料，無法畫圖。"
        )

    df["top_k"] = df["top_k"].astype(int)
    df = df.sort_values("top_k")

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["top_k"],
        df["net_annualized_return"] * 100,
        marker="o",
        label="External RF",
    )

    plt.plot(
        df["top_k"],
        df["random_net_annualized_return_mean"] * 100,
        marker="o",
        linestyle="--",
        label="Random mean",
    )

    plt.plot(
        df["top_k"],
        df["random_net_annualized_return_p95"] * 100,
        marker="o",
        linestyle=":",
        label="Random p95",
    )

    plt.axhline(
        float(df["all_stock_net_annualized_return"].iloc[0]) * 100,
        linestyle="-.",
        linewidth=1.5,
        label="All-stock benchmark",
    )

    plt.title("External Top-K Sensitivity")
    plt.xlabel("Top-K")
    plt.ylabel("Net annualized return")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_figure(output_path)


# =========================
# Main
# =========================

def main() -> None:
    setup_chinese_font()

    print("========== Step 7: Generate External Figures ==========")

    cleaned_path = require_file(
        EXTERNAL_CLEANED_DATA_PATH,
        hint="python step7_run_external_crawler.py --start-year 2015 --end-year 2024 --source finmind --use-cache",
    )

    comparison_path = require_file(
        EXTERNAL_RF_BENCHMARK_COMPARISON_PATH,
        hint="python step7_external_benchmark.py && python step7_compare_external_benchmarks.py",
    )

    cleaned_df = pd.read_csv(cleaned_path, dtype={STOCK_ID_COL: str})
    comparison_df = pd.read_csv(comparison_path)

    plot_feature_missing_ratio(
        df=cleaned_df,
        output_path=EXTERNAL_FEATURE_MISSING_RATIO_FIG_PATH,
    )
    print(f"[Saved] {EXTERNAL_FEATURE_MISSING_RATIO_FIG_PATH}")

    plot_external_rf_vs_benchmark(
        comparison_df=comparison_df,
        output_path=EXTERNAL_RF_VS_BENCHMARK_FIG_PATH,
    )
    print(f"[Saved] {EXTERNAL_RF_VS_BENCHMARK_FIG_PATH}")

    plot_external_topk_sensitivity(
        comparison_df=comparison_df,
        output_path=EXTERNAL_TOPK_SENSITIVITY_FIG_PATH,
    )
    print(f"[Saved] {EXTERNAL_TOPK_SENSITIVITY_FIG_PATH}")

    print("========== External Figures Finished ==========")


if __name__ == "__main__":
    main()