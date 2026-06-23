from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree

from src.schema import FEATURE_COLUMNS, YEAR_COL


# ============================================================
# General utilities
# ============================================================

def setup_matplotlib() -> None:
    """
    設定 matplotlib 字型。
    Windows 優先使用 Microsoft JhengHei，避免中文特徵名稱變成方塊。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_current_figure(output_path: str | Path) -> None:
    """
    儲存目前 figure 為高解析 PNG。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _parse_test_year(test_years_value: object) -> int:
    """
    將 test_years 欄位轉成單一年份。
    next_year mode 下 test_years 通常是 '1998'。
    """
    text = str(test_years_value)
    return int(text.split(",")[0])


def _calculate_cumulative_return_series(returns: pd.Series) -> pd.Series:
    """
    annual returns → cumulative return series.
    returns 應為 decimal，例如 0.1 代表 10%。
    """
    returns = pd.Series(returns).astype(float).fillna(0.0)
    return (1.0 + returns).cumprod() - 1.0


def _calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    annual returns → drawdown series.
    """
    returns = pd.Series(returns).astype(float).fillna(0.0)
    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return drawdown


def _calculate_annualized_return(returns: pd.Series) -> float:
    """
    Annualized return = product(1 + r_t)^(1/T) - 1
    """
    returns = pd.Series(returns).dropna().astype(float)

    if len(returns) == 0:
        return np.nan

    growth = (1.0 + returns).prod()

    if growth <= 0:
        return np.nan

    return float(growth ** (1.0 / len(returns)) - 1.0)


# ============================================================
# Step 3 figures: Decision Tree classification
# ============================================================

def plot_metrics_by_year(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Step 3 Figure 1:
    每年 Accuracy / Precision / Recall / F1-score。
    """
    setup_matplotlib()

    split_metrics = metrics_df[metrics_df["scope"] == "split"].copy()
    split_metrics["test_year"] = split_metrics["test_years"].apply(_parse_test_year)
    split_metrics = split_metrics.sort_values("test_year")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        split_metrics["test_year"],
        split_metrics["accuracy"],
        marker="o",
        label="Accuracy",
    )
    ax.plot(
        split_metrics["test_year"],
        split_metrics["precision_label_1"],
        marker="o",
        label="Precision label=1",
    )
    ax.plot(
        split_metrics["test_year"],
        split_metrics["recall_label_1"],
        marker="o",
        label="Recall label=1",
    )
    ax.plot(
        split_metrics["test_year"],
        split_metrics["f1_label_1"],
        marker="o",
        label="F1 label=1",
    )

    ax.set_title("Decision Tree Classification Metrics by Testing Year")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Metric score")
    ax.set_ylim(0, 1)
    ax.set_xticks(split_metrics["test_year"].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_confusion_matrix_overall(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Step 3 Figure 2:
    整體混淆矩陣。
    """
    setup_matplotlib()

    y_true = predictions_df["actual_label"].astype(int)
    y_pred = predictions_df["predicted_label"].astype(int)

    labels = [-1, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set_title("Decision Tree Overall Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    save_current_figure(output_path)


def plot_average_feature_importance(
    feature_importance_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Step 3 Figure 3:
    跨 temporal split 的平均 feature importance。
    """
    setup_matplotlib()

    avg_fi = (
        feature_importance_df.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(avg_fi["feature"], avg_fi["importance"])

    ax.set_title("Average Feature Importance across Temporal Splits")
    ax.set_xlabel("Average feature importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)

    save_current_figure(output_path)


def plot_feature_importance_heatmap(
    feature_importance_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 10,
) -> None:
    """
    Step 3 Figure 4:
    Feature importance heatmap。
    顯示平均重要性最高的 top_n 個特徵在各 split 的變化。
    """
    setup_matplotlib()

    avg_fi = (
        feature_importance_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
    )

    top_features = avg_fi.head(top_n).index.tolist()

    pivot = (
        feature_importance_df[feature_importance_df["feature"].isin(top_features)]
        .pivot_table(
            index="split_id",
            columns="feature",
            values="importance",
            aggfunc="mean",
        )
        .fillna(0)
    )

    pivot = pivot[top_features]

    fig, ax = plt.subplots(figsize=(12, 6))

    image = ax.imshow(pivot.values, aspect="auto")
    fig.colorbar(image, ax=ax, label="Feature importance")

    ax.set_title(f"Feature Importance Heatmap by Split - Top {top_n} Features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Split ID")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))

    save_current_figure(output_path)


def plot_score_boxplot_by_actual_label(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Step 3 Figure 5:
    比較 actual label=-1 與 actual label=1 的 score_label_1 分布。
    """
    setup_matplotlib()

    score_neg = predictions_df.loc[
        predictions_df["actual_label"].astype(int) == -1,
        "score_label_1",
    ].dropna()

    score_pos = predictions_df.loc[
        predictions_df["actual_label"].astype(int) == 1,
        "score_label_1",
    ].dropna()

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.boxplot(
        [score_neg, score_pos],
        tick_labels=["Actual label = -1", "Actual label = 1"],
        showmeans=True,
    )

    ax.set_title("Predicted P(label=1) by Actual Label")
    ax.set_ylabel("Predicted P(label=1)")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    save_current_figure(output_path)


def plot_decision_tree_png(
    model_path: str | Path,
    output_path: str | Path,
    max_depth: int = 3,
) -> None:
    """
    Step 3 Figure 6:
    將指定 split 的 Decision Tree 畫成 PNG。
    建議報告使用 max_depth=3，完整 rules 用 txt。
    """
    setup_matplotlib()

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"找不到 saved model：{model_path}")

    pipeline = joblib.load(model_path)
    tree_model = pipeline.named_steps["model"]

    class_names = [str(c) for c in tree_model.classes_]

    fig, ax = plt.subplots(figsize=(24, 12))

    plot_tree(
        tree_model,
        feature_names=FEATURE_COLUMNS,
        class_names=class_names,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )

    ax.set_title(
        f"Decision Tree Visualization - Displayed Max Depth = {max_depth}"
    )

    save_current_figure(output_path)


# ============================================================
# Step 4 figures: Portfolio and benchmark
# ============================================================

def plot_top10_cumulative_return_comparison(
    portfolio_returns_df: pd.DataFrame,
    all_stock_returns_df: pd.DataFrame,
    random_annual_mean_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
) -> None:
    """
    Step 4 Figure 1:
    Top-10 net cumulative return comparison。

    比較：
    - Decision Tree Top-10 Equal-weight net
    - Decision Tree Top-10 Score-weighted net
    - All-stock benchmark net
    - Random Top-10 benchmark mean net
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 6))

    dt_top10 = portfolio_returns_df[
        portfolio_returns_df["top_k"].astype(int) == top_k
    ].copy()

    for weight_method in ["equal", "score"]:
        temp = dt_top10[dt_top10["weight_method"] == weight_method].copy()
        temp = temp.sort_values(YEAR_COL)

        cumulative_return = _calculate_cumulative_return_series(temp["net_return"])

        ax.plot(
            temp[YEAR_COL],
            cumulative_return * 100,
            marker="o",
            label=f"Decision Tree Top-{top_k} {weight_method} net",
        )

    all_stock = all_stock_returns_df.sort_values(YEAR_COL).copy()
    all_stock_cum = _calculate_cumulative_return_series(all_stock["net_return"])

    ax.plot(
        all_stock[YEAR_COL],
        all_stock_cum * 100,
        marker="o",
        label="All-stock benchmark net",
    )

    random_mean = random_annual_mean_df[
        random_annual_mean_df["top_k"].astype(int) == top_k
    ].copy()
    random_mean = random_mean.sort_values(YEAR_COL)
    random_cum = _calculate_cumulative_return_series(random_mean["net_return"])

    ax.plot(
        random_mean[YEAR_COL],
        random_cum * 100,
        marker="o",
        label=f"Random Top-{top_k} benchmark mean net",
    )

    ax.set_title(f"Top-{top_k} Net Cumulative Return Comparison")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(sorted(portfolio_returns_df[YEAR_COL].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_top10_annual_return_comparison(
    portfolio_returns_df: pd.DataFrame,
    all_stock_returns_df: pd.DataFrame,
    random_annual_mean_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    Step 4 Figure 2:
    Top-10 equal-weight annual net return comparison。
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 6))

    dt = portfolio_returns_df[
        (portfolio_returns_df["top_k"].astype(int) == top_k)
        & (portfolio_returns_df["weight_method"] == weight_method)
    ].copy()
    dt = dt.sort_values(YEAR_COL)

    all_stock = all_stock_returns_df.sort_values(YEAR_COL).copy()

    random_mean = random_annual_mean_df[
        random_annual_mean_df["top_k"].astype(int) == top_k
    ].copy()
    random_mean = random_mean.sort_values(YEAR_COL)

    ax.plot(
        dt[YEAR_COL],
        dt["net_return"] * 100,
        marker="o",
        label=f"Decision Tree Top-{top_k} {weight_method} net",
    )

    ax.plot(
        all_stock[YEAR_COL],
        all_stock["net_return"] * 100,
        marker="o",
        label="All-stock benchmark net",
    )

    ax.plot(
        random_mean[YEAR_COL],
        random_mean["net_return"] * 100,
        marker="o",
        label=f"Random Top-{top_k} benchmark mean net",
    )

    ax.axhline(0, linewidth=1)

    ax.set_title(f"Top-{top_k} Annual Net Return Comparison")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Annual return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(dt[YEAR_COL].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_topk_net_annualized_return(
    portfolio_metrics_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    all_stock_metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Step 4 Figure 3:
    比較 Top-K net annualized return。
    包含：
    - Decision Tree equal
    - Decision Tree score
    - Random benchmark mean
    - All-stock benchmark reference line
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = portfolio_metrics_df.copy()
    metrics["top_k"] = metrics["top_k"].astype(int)

    top_k_values = sorted(metrics["top_k"].unique().tolist())
    x = np.arange(len(top_k_values))
    width = 0.25

    equal_values = []
    score_values = []
    random_values = []

    for top_k in top_k_values:
        equal_row = metrics[
            (metrics["top_k"] == top_k)
            & (metrics["weight_method"] == "equal")
        ]
        score_row = metrics[
            (metrics["top_k"] == top_k)
            & (metrics["weight_method"] == "score")
        ]
        random_row = random_summary_df[
            random_summary_df["top_k"].astype(int) == top_k
        ]

        equal_values.append(float(equal_row["net_annualized_return"].iloc[0]) * 100)
        score_values.append(float(score_row["net_annualized_return"].iloc[0]) * 100)
        random_values.append(
            float(random_row["net_annualized_return_mean"].iloc[0]) * 100
        )

    ax.bar(x - width, equal_values, width, label="Decision Tree equal net")
    ax.bar(x, score_values, width, label="Decision Tree score net")
    ax.bar(x + width, random_values, width, label="Random benchmark mean net")

    all_stock_net_annualized = (
        float(all_stock_metrics_df["net_annualized_return"].iloc[0]) * 100
    )

    ax.axhline(
        all_stock_net_annualized,
        linestyle="--",
        linewidth=1.5,
        label="All-stock benchmark net",
    )

    ax.set_title("Net Annualized Return by Top-K")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Net annualized return")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in top_k_values])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_top10_drawdown_comparison(
    portfolio_returns_df: pd.DataFrame,
    all_stock_returns_df: pd.DataFrame,
    random_annual_mean_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    Step 4 Figure 4:
    Top-10 net drawdown comparison。
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 6))

    dt = portfolio_returns_df[
        (portfolio_returns_df["top_k"].astype(int) == top_k)
        & (portfolio_returns_df["weight_method"] == weight_method)
    ].copy()
    dt = dt.sort_values(YEAR_COL)

    dt_drawdown = _calculate_drawdown_series(dt["net_return"])

    ax.plot(
        dt[YEAR_COL],
        dt_drawdown * 100,
        marker="o",
        label=f"Decision Tree Top-{top_k} {weight_method} net",
    )

    all_stock = all_stock_returns_df.sort_values(YEAR_COL).copy()
    all_stock_drawdown = _calculate_drawdown_series(all_stock["net_return"])

    ax.plot(
        all_stock[YEAR_COL],
        all_stock_drawdown * 100,
        marker="o",
        label="All-stock benchmark net",
    )

    random_mean = random_annual_mean_df[
        random_annual_mean_df["top_k"].astype(int) == top_k
    ].copy()
    random_mean = random_mean.sort_values(YEAR_COL)
    random_drawdown = _calculate_drawdown_series(random_mean["net_return"])

    ax.plot(
        random_mean[YEAR_COL],
        random_drawdown * 100,
        marker="o",
        label=f"Random Top-{top_k} benchmark mean net",
    )

    ax.axhline(0, linewidth=1)

    ax.set_title(f"Top-{top_k} Net Drawdown Comparison")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(dt[YEAR_COL].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_random_benchmark_distribution_topk(
    portfolio_metrics_df: pd.DataFrame,
    random_runs_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    Step 4 Figure 5:
    Random Top-K benchmark distribution。
    顯示 random benchmark 各 run 的 net annualized return 分布，
    並用垂直線標示 Decision Tree Top-K net annualized return。
    """
    setup_matplotlib()

    random_topk = random_runs_df[
        random_runs_df["top_k"].astype(int) == top_k
    ].copy()

    run_records = []

    for random_run, group_df in random_topk.groupby("random_run"):
        group_df = group_df.sort_values(YEAR_COL)
        net_ann = _calculate_annualized_return(group_df["net_return"])

        run_records.append(
            {
                "random_run": int(random_run),
                "net_annualized_return": net_ann,
            }
        )

    run_df = pd.DataFrame(run_records)

    dt_row = portfolio_metrics_df[
        (portfolio_metrics_df["top_k"].astype(int) == top_k)
        & (portfolio_metrics_df["weight_method"] == weight_method)
    ]

    if dt_row.empty:
        raise ValueError(
            f"找不到 Decision Tree top_k={top_k}, weight_method={weight_method} 的 metrics。"
        )

    dt_net_ann = float(dt_row["net_annualized_return"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(run_df["net_annualized_return"] * 100, bins=30, alpha=0.8)

    ax.axvline(
        dt_net_ann * 100,
        linestyle="--",
        linewidth=2,
        label=f"Decision Tree Top-{top_k} {weight_method} net",
    )

    ax.set_title(f"Random Top-{top_k} Benchmark Distribution")
    ax.set_xlabel("Net annualized return")
    ax.set_ylabel("Number of random runs")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_return_risk_scatter(
    portfolio_metrics_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    all_stock_metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Step 4 Figure 6:
    Return-risk scatter。
    X 軸：net volatility
    Y 軸：net annualized return

    用來直觀比較：
    - Decision Tree 不同 Top-K / weighting
    - Random benchmark mean
    - All-stock benchmark
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(9, 6))

    metrics = portfolio_metrics_df.copy()

    for _, row in metrics.iterrows():
        label = f"DT K={int(row['top_k'])}, {row['weight_method']}"
        ax.scatter(
            row["net_volatility"] * 100,
            row["net_annualized_return"] * 100,
        )
        ax.text(
            row["net_volatility"] * 100,
            row["net_annualized_return"] * 100,
            label,
            fontsize=8,
            ha="left",
            va="bottom",
        )

    for _, row in random_summary_df.iterrows():
        label = f"Random K={int(row['top_k'])}"
        ax.scatter(
            row["net_volatility_mean"] * 100,
            row["net_annualized_return_mean"] * 100,
        )
        ax.text(
            row["net_volatility_mean"] * 100,
            row["net_annualized_return_mean"] * 100,
            label,
            fontsize=8,
            ha="left",
            va="bottom",
        )

    all_stock = all_stock_metrics_df.iloc[0]
    ax.scatter(
        all_stock["net_volatility"] * 100,
        all_stock["net_annualized_return"] * 100,
    )
    ax.text(
        all_stock["net_volatility"] * 100,
        all_stock["net_annualized_return"] * 100,
        "All-stock",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    ax.set_title("Return-Risk Comparison")
    ax.set_xlabel("Net volatility")
    ax.set_ylabel("Net annualized return")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, alpha=0.3)

    save_current_figure(output_path)

def plot_top10_random_cumulative_band(
    portfolio_returns_df: pd.DataFrame,
    all_stock_returns_df: pd.DataFrame,
    random_runs_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    顯示 Random Top-K cumulative return distribution band。
    這張圖比 random mean 更適合判斷模型是否優於隨機選股。
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 6))

    # Decision Tree strategy
    dt = portfolio_returns_df[
        (portfolio_returns_df["top_k"].astype(int) == top_k)
        & (portfolio_returns_df["weight_method"] == weight_method)
    ].copy()
    dt = dt.sort_values(YEAR_COL)

    dt_cum = _calculate_cumulative_return_series(dt["net_return"])

    ax.plot(
        dt[YEAR_COL],
        dt_cum * 100,
        marker="o",
        linewidth=2,
        label=f"Decision Tree Top-{top_k} {weight_method} net",
    )

    # All-stock benchmark
    all_stock = all_stock_returns_df.sort_values(YEAR_COL).copy()
    all_stock_cum = _calculate_cumulative_return_series(all_stock["net_return"])

    ax.plot(
        all_stock[YEAR_COL],
        all_stock_cum * 100,
        marker="o",
        linewidth=2,
        label="All-stock benchmark net",
    )

    # Random cumulative paths
    random_topk = random_runs_df[
        random_runs_df["top_k"].astype(int) == top_k
    ].copy()

    cumulative_records = []

    for random_run, group_df in random_topk.groupby("random_run"):
        group_df = group_df.sort_values(YEAR_COL).copy()
        group_df["cumulative_return"] = _calculate_cumulative_return_series(
            group_df["net_return"]
        ).values
        group_df["random_run"] = random_run
        cumulative_records.append(group_df[[YEAR_COL, "random_run", "cumulative_return"]])

    random_cum_df = pd.concat(cumulative_records, ignore_index=True)

    random_band = (
        random_cum_df.groupby(YEAR_COL)["cumulative_return"]
        .agg(
            random_p5=lambda x: x.quantile(0.05),
            random_median="median",
            random_mean="mean",
            random_p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
        .sort_values(YEAR_COL)
    )

    ax.fill_between(
        random_band[YEAR_COL],
        random_band["random_p5"] * 100,
        random_band["random_p95"] * 100,
        alpha=0.2,
        label=f"Random Top-{top_k} 5%-95% band",
    )

    ax.plot(
        random_band[YEAR_COL],
        random_band["random_median"] * 100,
        linestyle="--",
        linewidth=2,
        label=f"Random Top-{top_k} median net",
    )

    ax.set_title(f"Decision Tree Top-{top_k} vs Random Benchmark Distribution")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(dt[YEAR_COL].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)

def plot_top10_benchmark_zoom(
    all_stock_returns_df: pd.DataFrame,
    random_runs_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
) -> None:
    """
    只放 All-stock 與 Random Top-K benchmark。
    用來檢查 random mean 是否合理接近 all-stock，以及 random dispersion。
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 6))

    all_stock = all_stock_returns_df.sort_values(YEAR_COL).copy()
    all_stock_cum = _calculate_cumulative_return_series(all_stock["net_return"])

    ax.plot(
        all_stock[YEAR_COL],
        all_stock_cum * 100,
        marker="o",
        linewidth=2,
        label="All-stock benchmark net",
    )

    random_topk = random_runs_df[
        random_runs_df["top_k"].astype(int) == top_k
    ].copy()

    cumulative_records = []

    for random_run, group_df in random_topk.groupby("random_run"):
        group_df = group_df.sort_values(YEAR_COL).copy()
        group_df["cumulative_return"] = _calculate_cumulative_return_series(
            group_df["net_return"]
        ).values
        group_df["random_run"] = random_run
        cumulative_records.append(group_df[[YEAR_COL, "random_run", "cumulative_return"]])

    random_cum_df = pd.concat(cumulative_records, ignore_index=True)

    random_band = (
        random_cum_df.groupby(YEAR_COL)["cumulative_return"]
        .agg(
            random_p5=lambda x: x.quantile(0.05),
            random_median="median",
            random_mean="mean",
            random_p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
        .sort_values(YEAR_COL)
    )

    ax.fill_between(
        random_band[YEAR_COL],
        random_band["random_p5"] * 100,
        random_band["random_p95"] * 100,
        alpha=0.2,
        label=f"Random Top-{top_k} 5%-95% band",
    )

    ax.plot(
        random_band[YEAR_COL],
        random_band["random_mean"] * 100,
        marker="o",
        linestyle="--",
        linewidth=2,
        label=f"Random Top-{top_k} mean net",
    )

    ax.plot(
        random_band[YEAR_COL],
        random_band["random_median"] * 100,
        linestyle=":",
        linewidth=2,
        label=f"Random Top-{top_k} median net",
    )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"Benchmark Zoom: All-stock vs Random Top-{top_k}")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(all_stock[YEAR_COL].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def _model_display_name(model_name: str) -> str:
    """
    將內部 model_name 轉成圖表中較好讀的名稱。
    """
    name_map = {
        "decision_tree_entropy": "Decision Tree",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }
    return name_map.get(model_name, model_name)


def plot_step5_classification_overall_metrics(
    all_classification_metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Step 5 Figure 1:
    比較各分類模型 overall Accuracy / Precision / Recall / F1。
    """
    setup_matplotlib()

    overall = all_classification_metrics_df[
        all_classification_metrics_df["scope"] == "overall"
    ].copy()

    overall["model_display"] = overall["model_name"].apply(_model_display_name)

    metric_cols = [
        "accuracy",
        "precision_label_1",
        "recall_label_1",
        "f1_label_1",
    ]

    x = np.arange(len(overall))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, metric in enumerate(metric_cols):
        ax.bar(
            x + (idx - 1.5) * width,
            overall[metric],
            width,
            label=metric.replace("_", " "),
        )

    ax.set_title("Overall Classification Metrics by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(overall["model_display"], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_step5_f1_by_testing_year(
    all_classification_metrics_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Step 5 Figure 2:
    各模型每個 testing year 的 F1-score(label=1)。
    """
    setup_matplotlib()

    split_df = all_classification_metrics_df[
        all_classification_metrics_df["scope"] == "split"
    ].copy()

    split_df["test_year"] = split_df["test_years"].apply(_parse_test_year)
    split_df["model_display"] = split_df["model_name"].apply(_model_display_name)

    fig, ax = plt.subplots(figsize=(11, 6))

    for model_display, group_df in split_df.groupby("model_display"):
        group_df = group_df.sort_values("test_year")

        ax.plot(
            group_df["test_year"],
            group_df["f1_label_1"],
            marker="o",
            label=model_display,
        )

    ax.set_title("F1-score for Label=1 by Testing Year")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("F1-score label=1")
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(split_df["test_year"].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_step5_top10_cumulative_return_comparison(
    all_portfolio_returns_df: pd.DataFrame,
    all_stock_returns_df: pd.DataFrame,
    random_annual_mean_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    Step 5 Figure 3:
    比較各模型 Top-10 Equal-weight net cumulative return。
    同時加入 All-stock benchmark 與 Random Top-10 mean。
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))

    strategy_df = all_portfolio_returns_df[
        (all_portfolio_returns_df["top_k"].astype(int) == top_k)
        & (all_portfolio_returns_df["weight_method"] == weight_method)
    ].copy()

    for model_name, group_df in strategy_df.groupby("model_name"):
        group_df = group_df.sort_values(YEAR_COL)
        cumulative_return = _calculate_cumulative_return_series(group_df["net_return"])

        ax.plot(
            group_df[YEAR_COL],
            cumulative_return * 100,
            marker="o",
            label=f"{_model_display_name(model_name)} Top-{top_k}",
        )

    all_stock = all_stock_returns_df.sort_values(YEAR_COL).copy()
    all_stock_cum = _calculate_cumulative_return_series(all_stock["net_return"])

    ax.plot(
        all_stock[YEAR_COL],
        all_stock_cum * 100,
        marker="o",
        linestyle="--",
        label="All-stock benchmark",
    )

    random_mean = random_annual_mean_df[
        random_annual_mean_df["top_k"].astype(int) == top_k
    ].copy()
    random_mean = random_mean.sort_values(YEAR_COL)

    random_cum = _calculate_cumulative_return_series(random_mean["net_return"])

    ax.plot(
        random_mean[YEAR_COL],
        random_cum * 100,
        marker="o",
        linestyle=":",
        label=f"Random Top-{top_k} mean",
    )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"Top-{top_k} Net Cumulative Return by Model")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(sorted(strategy_df[YEAR_COL].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_step5_top10_net_annualized_return(
    all_portfolio_metrics_df: pd.DataFrame,
    all_stock_metrics_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    Step 5 Figure 4:
    比較各模型 Top-10 Equal-weight net annualized return。
    """
    setup_matplotlib()

    strategy_df = all_portfolio_metrics_df[
        (all_portfolio_metrics_df["top_k"].astype(int) == top_k)
        & (all_portfolio_metrics_df["weight_method"] == weight_method)
    ].copy()

    strategy_df["model_display"] = strategy_df["model_name"].apply(_model_display_name)
    strategy_df = strategy_df.sort_values("net_annualized_return", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        strategy_df["model_display"],
        strategy_df["net_annualized_return"] * 100,
    )

    all_stock_value = float(all_stock_metrics_df["net_annualized_return"].iloc[0]) * 100

    random_value = float(
        random_summary_df[
            random_summary_df["top_k"].astype(int) == top_k
        ]["net_annualized_return_mean"].iloc[0]
    ) * 100

    ax.axvline(
        all_stock_value,
        linestyle="--",
        linewidth=1.5,
        label="All-stock benchmark",
    )

    ax.axvline(
        random_value,
        linestyle=":",
        linewidth=1.5,
        label=f"Random Top-{top_k} mean",
    )

    ax.set_title(f"Top-{top_k} Net Annualized Return by Model")
    ax.set_xlabel("Net annualized return")
    ax.set_ylabel("Model")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_step5_topk_net_annualized_heatmap(
    all_portfolio_metrics_df: pd.DataFrame,
    output_path: str | Path,
    weight_method: str = "equal",
) -> None:
    """
    Step 5 Figure 5:
    熱圖顯示不同模型在 Top-5/10/20/30 下的 net annualized return。
    """
    setup_matplotlib()

    df = all_portfolio_metrics_df[
        all_portfolio_metrics_df["weight_method"] == weight_method
    ].copy()

    df["model_display"] = df["model_name"].apply(_model_display_name)
    df["top_k"] = df["top_k"].astype(int)

    pivot = df.pivot_table(
        index="model_display",
        columns="top_k",
        values="net_annualized_return",
        aggfunc="mean",
    )

    pivot = pivot.sort_index()
    pivot = pivot[sorted(pivot.columns.tolist())]

    fig, ax = plt.subplots(figsize=(9, 5))

    image = ax.imshow(pivot.values * 100, aspect="auto")
    fig.colorbar(image, ax=ax, label="Net annualized return (%)")

    ax.set_title(f"Net Annualized Return Heatmap ({weight_method})")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Model")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j] * 100
            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

    save_current_figure(output_path)


def plot_step5_return_risk_scatter(
    all_portfolio_metrics_df: pd.DataFrame,
    all_stock_metrics_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    Step 5 Figure 6:
    Return-risk scatter。
    X 軸為 net volatility，Y 軸為 net annualized return。
    """
    setup_matplotlib()

    strategy_df = all_portfolio_metrics_df[
        (all_portfolio_metrics_df["top_k"].astype(int) == top_k)
        & (all_portfolio_metrics_df["weight_method"] == weight_method)
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in strategy_df.iterrows():
        label = _model_display_name(row["model_name"])

        ax.scatter(
            row["net_volatility"] * 100,
            row["net_annualized_return"] * 100,
        )

        ax.text(
            row["net_volatility"] * 100,
            row["net_annualized_return"] * 100,
            label,
            fontsize=8,
            ha="left",
            va="bottom",
        )

    all_stock = all_stock_metrics_df.iloc[0]

    ax.scatter(
        all_stock["net_volatility"] * 100,
        all_stock["net_annualized_return"] * 100,
    )

    ax.text(
        all_stock["net_volatility"] * 100,
        all_stock["net_annualized_return"] * 100,
        "All-stock",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    random_row = random_summary_df[
        random_summary_df["top_k"].astype(int) == top_k
    ].iloc[0]

    ax.scatter(
        random_row["net_volatility_mean"] * 100,
        random_row["net_annualized_return_mean"] * 100,
    )

    ax.text(
        random_row["net_volatility_mean"] * 100,
        random_row["net_annualized_return_mean"] * 100,
        f"Random Top-{top_k}",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    ax.set_title(f"Return-Risk Comparison: Top-{top_k} {weight_method}")
    ax.set_xlabel("Net volatility")
    ax.set_ylabel("Net annualized return")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, alpha=0.3)

    save_current_figure(output_path)


def plot_step5_sharpe_by_model(
    all_portfolio_metrics_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    Step 5 Figure 7:
    比較各模型 Top-10 net Sharpe ratio。
    """
    setup_matplotlib()

    df = all_portfolio_metrics_df[
        (all_portfolio_metrics_df["top_k"].astype(int) == top_k)
        & (all_portfolio_metrics_df["weight_method"] == weight_method)
    ].copy()

    df["model_display"] = df["model_name"].apply(_model_display_name)
    df = df.sort_values("net_sharpe_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        df["model_display"],
        df["net_sharpe_ratio"],
    )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"Top-{top_k} Net Sharpe Ratio by Model")
    ax.set_xlabel("Net Sharpe ratio")
    ax.set_ylabel("Model")
    ax.grid(True, axis="x", alpha=0.3)

    save_current_figure(output_path)


def plot_step5_rf_gb_feature_importance(
    task2_feature_importance_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 10,
) -> None:
    """
    Step 5 Figure 8:
    顯示 Random Forest 與 Gradient Boosting 的平均 feature importance。
    Logistic Regression 係數尺度與 tree importance 不同，建議另外討論，不放同圖。
    """
    setup_matplotlib()

    df = task2_feature_importance_df[
        task2_feature_importance_df["model_name"].isin(
            ["random_forest", "gradient_boosting"]
        )
    ].copy()

    avg_df = (
        df.groupby(["model_name", "feature"], as_index=False)["importance"]
        .mean()
    )

    top_features = (
        avg_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    avg_df = avg_df[avg_df["feature"].isin(top_features)].copy()
    avg_df["model_display"] = avg_df["model_name"].apply(_model_display_name)

    pivot = avg_df.pivot_table(
        index="feature",
        columns="model_display",
        values="importance",
        aggfunc="mean",
    ).fillna(0)

    pivot["mean_importance"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_importance", ascending=True)
    pivot = pivot.drop(columns=["mean_importance"])

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(len(pivot.index))
    width = 0.35

    columns = pivot.columns.tolist()

    for idx, col in enumerate(columns):
        offset = (idx - (len(columns) - 1) / 2) * width

        ax.barh(
            y + offset,
            pivot[col],
            height=width,
            label=col,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)

    ax.set_title(f"Random Forest and Gradient Boosting Feature Importance - Top {top_n}")
    ax.set_xlabel("Average feature importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def _all_model_display_name(model_name: str) -> str:
    name_map = {
        "decision_tree_entropy": "Decision Tree",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "svr_ga_regression": "SVR-GA",
    }
    return name_map.get(model_name, model_name)


def _ordered_model_display_names() -> list[str]:
    return [
        "Random Forest",
        "Gradient Boosting",
        "Decision Tree",
        "SVR-GA",
        "Logistic Regression",
    ]


def plot_step6_all_model_top10_cumulative_return(
    all_portfolio_returns_df: pd.DataFrame,
    all_stock_returns_df: pd.DataFrame,
    random_annual_mean_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    比較 DT / LR / RF / GB / SVR-GA 的 Top-10 cumulative return。
    """
    setup_matplotlib()

    df = all_portfolio_returns_df[
        (all_portfolio_returns_df["top_k"].astype(int) == top_k)
        & (all_portfolio_returns_df["weight_method"] == weight_method)
    ].copy()

    df["model_display"] = df["model_name"].apply(_all_model_display_name)

    fig, ax = plt.subplots(figsize=(13, 7))

    model_order = _ordered_model_display_names()

    for model_display in model_order:
        group_df = df[df["model_display"] == model_display].copy()

        if group_df.empty:
            continue

        group_df = group_df.sort_values(YEAR_COL)
        cumulative_return = _calculate_cumulative_return_series(group_df["net_return"])

        ax.plot(
            group_df[YEAR_COL],
            cumulative_return * 100,
            marker="o",
            label=model_display,
        )

    all_stock = all_stock_returns_df.sort_values(YEAR_COL).copy()
    all_stock_cum = _calculate_cumulative_return_series(all_stock["net_return"])

    ax.plot(
        all_stock[YEAR_COL],
        all_stock_cum * 100,
        linestyle="--",
        marker="o",
        label="All-stock benchmark",
    )

    random_mean = random_annual_mean_df[
        random_annual_mean_df["top_k"].astype(int) == top_k
    ].copy()

    if not random_mean.empty:
        random_mean = random_mean.sort_values(YEAR_COL)
        random_cum = _calculate_cumulative_return_series(random_mean["net_return"])

        ax.plot(
            random_mean[YEAR_COL],
            random_cum * 100,
            linestyle=":",
            marker="o",
            label=f"Random Top-{top_k} mean",
        )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"All Models Top-{top_k} Net Cumulative Return ({weight_method})")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(sorted(df[YEAR_COL].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_step6_all_model_top10_annual_return(
    all_portfolio_returns_df: pd.DataFrame,
    all_stock_returns_df: pd.DataFrame,
    random_annual_mean_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    比較各模型每年 Top-10 net return。
    """
    setup_matplotlib()

    df = all_portfolio_returns_df[
        (all_portfolio_returns_df["top_k"].astype(int) == top_k)
        & (all_portfolio_returns_df["weight_method"] == weight_method)
    ].copy()

    df["model_display"] = df["model_name"].apply(_all_model_display_name)

    fig, ax = plt.subplots(figsize=(13, 7))

    for model_display in _ordered_model_display_names():
        group_df = df[df["model_display"] == model_display].copy()

        if group_df.empty:
            continue

        group_df = group_df.sort_values(YEAR_COL)

        ax.plot(
            group_df[YEAR_COL],
            group_df["net_return"] * 100,
            marker="o",
            label=model_display,
        )

    all_stock = all_stock_returns_df.sort_values(YEAR_COL).copy()

    ax.plot(
        all_stock[YEAR_COL],
        all_stock["net_return"] * 100,
        linestyle="--",
        marker="o",
        label="All-stock benchmark",
    )

    random_mean = random_annual_mean_df[
        random_annual_mean_df["top_k"].astype(int) == top_k
    ].copy()

    if not random_mean.empty:
        random_mean = random_mean.sort_values(YEAR_COL)

        ax.plot(
            random_mean[YEAR_COL],
            random_mean["net_return"] * 100,
            linestyle=":",
            marker="o",
            label=f"Random Top-{top_k} mean",
        )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"All Models Top-{top_k} Annual Net Return ({weight_method})")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Annual net return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(sorted(df[YEAR_COL].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_step6_all_model_top10_net_annualized_return(
    all_models_metrics_df: pd.DataFrame,
    all_stock_metrics_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    比較所有模型 Top-10 net annualized return。
    """
    setup_matplotlib()

    df = all_models_metrics_df[
        (all_models_metrics_df["top_k"].astype(int) == top_k)
        & (all_models_metrics_df["weight_method"] == weight_method)
    ].copy()

    df["model_display"] = df["model_name"].apply(_all_model_display_name)

    order = _ordered_model_display_names()
    df["model_display"] = pd.Categorical(
        df["model_display"],
        categories=order,
        ordered=True,
    )

    df = df.sort_values("net_annualized_return", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(
        df["model_display"].astype(str),
        df["net_annualized_return"] * 100,
    )

    all_stock_value = float(all_stock_metrics_df["net_annualized_return"].iloc[0]) * 100

    ax.axvline(
        all_stock_value,
        linestyle="--",
        linewidth=1.5,
        label="All-stock benchmark",
    )

    random_topk = random_summary_df[
        random_summary_df["top_k"].astype(int) == top_k
    ].copy()

    if not random_topk.empty:
        random_value = float(random_topk["net_annualized_return_mean"].iloc[0]) * 100

        ax.axvline(
            random_value,
            linestyle=":",
            linewidth=1.5,
            label=f"Random Top-{top_k} mean",
        )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"All Models Top-{top_k} Net Annualized Return ({weight_method})")
    ax.set_xlabel("Net annualized return")
    ax.set_ylabel("Model")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_step6_all_model_topk_heatmap(
    all_models_metrics_df: pd.DataFrame,
    output_path: str | Path,
    weight_method: str = "equal",
) -> None:
    """
    比較所有模型 Top-K sensitivity。
    """
    setup_matplotlib()

    df = all_models_metrics_df[
        all_models_metrics_df["weight_method"] == weight_method
    ].copy()

    df["model_display"] = df["model_name"].apply(_all_model_display_name)
    df["top_k"] = df["top_k"].astype(int)

    pivot = df.pivot_table(
        index="model_display",
        columns="top_k",
        values="net_annualized_return",
        aggfunc="mean",
    )

    order = _ordered_model_display_names()
    pivot = pivot.reindex([m for m in order if m in pivot.index])
    pivot = pivot[sorted(pivot.columns.tolist())]

    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(pivot.values * 100, aspect="auto")
    fig.colorbar(image, ax=ax, label="Net annualized return (%)")

    ax.set_title(f"All Models Net Annualized Return Heatmap ({weight_method})")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Model")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j] * 100

            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

    save_current_figure(output_path)


def plot_step6_all_model_return_risk_scatter(
    all_models_metrics_df: pd.DataFrame,
    all_stock_metrics_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    所有模型 return-risk scatter。
    X = net volatility, Y = net annualized return。
    """
    setup_matplotlib()

    df = all_models_metrics_df[
        (all_models_metrics_df["top_k"].astype(int) == top_k)
        & (all_models_metrics_df["weight_method"] == weight_method)
    ].copy()

    fig, ax = plt.subplots(figsize=(11, 7))

    for _, row in df.iterrows():
        label = _all_model_display_name(row["model_name"])

        x = row["net_volatility"] * 100
        y = row["net_annualized_return"] * 100

        ax.scatter(x, y)
        ax.text(
            x,
            y,
            label,
            fontsize=8,
            ha="left",
            va="bottom",
        )

    all_stock = all_stock_metrics_df.iloc[0]

    ax.scatter(
        all_stock["net_volatility"] * 100,
        all_stock["net_annualized_return"] * 100,
    )

    ax.text(
        all_stock["net_volatility"] * 100,
        all_stock["net_annualized_return"] * 100,
        "All-stock",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    random_topk = random_summary_df[
        random_summary_df["top_k"].astype(int) == top_k
    ].copy()

    if not random_topk.empty:
        random_row = random_topk.iloc[0]

        ax.scatter(
            random_row["net_volatility_mean"] * 100,
            random_row["net_annualized_return_mean"] * 100,
        )

        ax.text(
            random_row["net_volatility_mean"] * 100,
            random_row["net_annualized_return_mean"] * 100,
            f"Random Top-{top_k}",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"All Models Return-Risk Comparison: Top-{top_k} {weight_method}")
    ax.set_xlabel("Net volatility")
    ax.set_ylabel("Net annualized return")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(True, alpha=0.3)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.10)
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.10)

    save_current_figure(output_path)


def plot_step6_all_model_sharpe(
    all_models_metrics_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    setup_matplotlib()

    df = all_models_metrics_df[
        (all_models_metrics_df["top_k"].astype(int) == top_k)
        & (all_models_metrics_df["weight_method"] == weight_method)
    ].copy()

    df["model_display"] = df["model_name"].apply(_all_model_display_name)
    df = df.sort_values("net_sharpe_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        df["model_display"],
        df["net_sharpe_ratio"],
    )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"All Models Top-{top_k} Net Sharpe Ratio ({weight_method})")
    ax.set_xlabel("Net Sharpe ratio")
    ax.set_ylabel("Model")
    ax.grid(True, axis="x", alpha=0.3)

    save_current_figure(output_path)


def plot_step6_all_model_drawdown(
    all_portfolio_returns_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
    weight_method: str = "equal",
) -> None:
    """
    畫各模型 cumulative drawdown。
    """
    setup_matplotlib()

    df = all_portfolio_returns_df[
        (all_portfolio_returns_df["top_k"].astype(int) == top_k)
        & (all_portfolio_returns_df["weight_method"] == weight_method)
    ].copy()

    df["model_display"] = df["model_name"].apply(_all_model_display_name)

    fig, ax = plt.subplots(figsize=(13, 7))

    for model_display in _ordered_model_display_names():
        group_df = df[df["model_display"] == model_display].copy()

        if group_df.empty:
            continue

        group_df = group_df.sort_values(YEAR_COL)

        cumulative = _calculate_cumulative_return_series(group_df["net_return"])
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        ax.plot(
            group_df[YEAR_COL],
            drawdown * 100,
            marker="o",
            label=model_display,
        )

    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(f"All Models Top-{top_k} Net Drawdown ({weight_method})")
    ax.set_xlabel("Testing year")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xticks(sorted(df[YEAR_COL].unique().tolist()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_current_figure(output_path)


def plot_step6_svr_predicted_vs_actual(
    svr_predictions_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    SVR-GA regression diagnostic:
    predicted_return vs actual_return。
    """
    setup_matplotlib()

    df = svr_predictions_df.copy()

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        df["actual_return"],
        df["predicted_return"],
        alpha=0.45,
    )

    x_min = min(df["actual_return"].min(), df["predicted_return"].min())
    x_max = max(df["actual_return"].max(), df["predicted_return"].max())

    ax.plot(
        [x_min, x_max],
        [x_min, x_max],
        linestyle="--",
        linewidth=1,
        label="Ideal prediction",
    )

    corr = df["actual_return"].corr(df["predicted_return"], method="spearman")

    ax.set_title("SVR-GA Predicted Return vs Actual Return")
    ax.set_xlabel("Actual return (%)")
    ax.set_ylabel("Predicted return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.text(
        0.02,
        0.98,
        f"Spearman corr = {corr:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    save_current_figure(output_path)