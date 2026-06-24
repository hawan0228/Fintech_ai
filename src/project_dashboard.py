from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ALL_CLASSIFICATION_METRICS_PATH,
    ALL_MODELS_PORTFOLIO_METRICS_PATH,
    CLEANED_DATA_PATH,
    DEMO_METRICS_PATH,
    DEMO_PORTFOLIO_RETURNS_PATH,
    DEMO_PREDICTIONS_PATH,
    DEMO_PROFILE_PATH,
    DEMO_SELECTED_STOCKS_PATH,
    DT_PREDICTIONS_PATH,
    EXTERNAL_RF_PORTFOLIO_METRICS_PATH,
    PROJECT_ROOT,
    SAVED_MODEL_DIR,
    SVR_GA_PREDICTIONS_PATH,
    TASK2_PREDICTIONS_PATH,
    TEMPORAL_SPLITS_NEXT_YEAR_PATH,
    TEMPORAL_SPLITS_REMAINING_YEARS_PATH,
)
from src.demo_runner import MODEL_METADATA
from src.metrics import calculate_classification_metrics_by_model, calculate_cumulative_return
from src.schema import STOCK_ID_COL, STOCK_NAME_COL, YEAR_COL


EXPECTED_COUNTS = {
    "next_year_splits": 11,
    "remaining_years_splits": 11,
    "dt_predictions": 2200,
    "task2_predictions": 6600,
    "svr_predictions": 2200,
    "dt_saved_models": 11,
    "task2_saved_models": 33,
    "svr_saved_models": 11,
}

MODEL_LABELS = {
    **{name: meta["label"] for name, meta in MODEL_METADATA.items()},
    "svr_ga_regression": "SVR-GA Regression",
    "external_random_forest": "外部 Random Forest",
}

MODEL_TASKS = {
    "decision_tree_entropy": "任務 1",
    "logistic_regression": "任務 2",
    "random_forest": "任務 2",
    "gradient_boosting": "任務 2",
    "svr_ga_regression": "任務 2 延伸",
    "external_random_forest": "任務 3",
}

UI_COLUMN_RENAMES = {
    STOCK_ID_COL: "stock_id",
    STOCK_NAME_COL: "stock_name",
}

CURATED_FIGURES = [
    {
        "section": "任務 1",
        "title": "Decision Tree 各測試年度指標",
        "caption": "呈現任務 1 在 1998-2008 各年度次年度切分下的分類表現。",
        "path": Path("outputs/figures/decision_tree/dt_metrics_by_year.png"),
    },
    {
        "section": "任務 1",
        "title": "Decision Tree Top-10 累積報酬",
        "caption": "比較 Decision Tree 投組、基準組與隨機基線的累積報酬。",
        "path": Path("outputs/figures/decision_tree_portfolio/dt_top10_cumulative_return_comparison.png"),
    },
    {
        "section": "任務 1",
        "title": "Decision Tree Top-K 年化報酬",
        "caption": "呈現任務 1 選股規則下，不同 Top-K 的敏感度變化。",
        "path": Path("outputs/figures/decision_tree_portfolio/dt_topk_net_annualized_return.png"),
    },
    {
        "section": "任務 2",
        "title": "分類模型比較",
        "caption": "比較 Decision Tree、LR、RF 與 GB 的整體分類指標。",
        "path": Path("outputs/figures/step5_model_comparison/step5_classification_overall_metrics.png"),
    },
    {
        "section": "任務 2",
        "title": "任務 2 各模型 Top-10 累積報酬",
        "caption": "將不同分類器依分數重新排序後，比較年度再平衡投組表現。",
        "path": Path("outputs/figures/step5_model_comparison/step5_top10_cumulative_return_comparison.png"),
    },
    {
        "section": "任務 2",
        "title": "RF 與 GB 特徵重要度",
        "caption": "用於解讀任務 2 中表現最強的樹模型。",
        "path": Path("outputs/figures/step5_model_comparison/step5_rf_gb_feature_importance.png"),
    },
    {
        "section": "全模型",
        "title": "淨年化報酬排行榜",
        "caption": "統一比較 DT、LR、RF、GB 與 SVR-GA 的績效表現。",
        "path": Path("outputs/figures/step6_all_model_comparison/step6_all_model_top10_net_annualized_return.png"),
    },
    {
        "section": "全模型",
        "title": "報酬-風險散佈圖",
        "caption": "以年化報酬與波動度描繪各模型的風險報酬分布。",
        "path": Path("outputs/figures/step6_all_model_comparison/step6_all_model_return_risk_scatter.png"),
    },
    {
        "section": "全模型",
        "title": "SVR-GA 預測報酬與實際報酬",
        "caption": "展示任務 2 延伸實驗中迴歸訊號的品質。",
        "path": Path("outputs/figures/step6_all_model_comparison/step6_svr_ga_predicted_vs_actual_return.png"),
    },
    {
        "section": "任務 3",
        "title": "外部 RF 與 benchmark 比較",
        "caption": "比較外部資料重跑的績效與 benchmark 基線。",
        "path": Path("outputs/external/figures/external_rf_vs_benchmark_annualized_return.png"),
    },
    {
        "section": "任務 3",
        "title": "外部特徵缺值比例",
        "caption": "檢視重建外部因子的資料品質。",
        "path": Path("outputs/external/figures/external_feature_missing_ratio.png"),
    },
    {
        "section": "任務 3",
        "title": "外部 Top-K 敏感度",
        "caption": "觀察外部 RF 在不同投組寬度下的績效變化。",
        "path": Path("outputs/external/figures/external_topk_sensitivity.png"),
    },
]

KEY_DOWNLOADS = [
    ("清理後資料集", Path("data/processed/cleaned_top200.csv")),
    ("next-year 時序切分", Path("data/processed/temporal_splits_next_year.csv")),
    ("remaining-years 時序切分", Path("data/processed/temporal_splits_remaining_years.csv")),
    ("任務 1 預測結果", Path("outputs/predictions/decision_tree_predictions.csv")),
    ("任務 2 預測結果", Path("outputs/predictions/task2_classification_predictions.csv")),
    ("SVR-GA 預測結果", Path("outputs/predictions/svr_ga_regression_predictions.csv")),
    ("全模型投組指標", Path("outputs/portfolio/all_models_portfolio_metrics.csv")),
    ("外部投組指標", Path("outputs/external/portfolio/external_rf_portfolio_metrics.csv")),
    ("最新 Demo 預測結果", Path("outputs/demo/demo_predictions.csv")),
]


def _artifact_url(path: str | Path) -> str:
    target = Path(path)
    rel = target.relative_to(PROJECT_ROOT) if target.is_absolute() else target
    return f"/artifacts/{rel.as_posix()}"


def _safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return round(numeric, 6)

    if value is pd.NA or pd.isna(value):
        return None

    return value


def dataframe_to_ui_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    converted = df.rename(columns=UI_COLUMN_RENAMES).copy()
    return [
        {col: _safe_value(value) for col, value in row.items()}
        for _, row in converted.iterrows()
    ]


def _read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _build_dataset_summary() -> dict[str, Any]:
    df = _read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    year_counts = (
        df.groupby(YEAR_COL)
        .size()
        .reset_index(name="n_rows")
        .sort_values(YEAR_COL)
    )

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "year_start": int(df[YEAR_COL].min()),
        "year_end": int(df[YEAR_COL].max()),
        "n_features": 16,
        "n_unique_stock_ids": int(df[STOCK_ID_COL].nunique()),
        "rows_by_year": dataframe_to_ui_records(year_counts),
    }


def _build_split_summary() -> dict[str, Any]:
    next_df = _read_csv(TEMPORAL_SPLITS_NEXT_YEAR_PATH)
    remaining_df = _read_csv(TEMPORAL_SPLITS_REMAINING_YEARS_PATH)
    testing_years = pd.to_numeric(next_df["test_years"], errors="coerce").dropna().astype(int)

    return {
        "next_year_split_count": int(len(next_df)),
        "remaining_years_split_count": int(len(remaining_df)),
        "test_year_start": int(testing_years.min()),
        "test_year_end": int(testing_years.max()),
    }


def _best_strategy_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    order_cols = ["model_name", "net_annualized_return", "net_sharpe_ratio", "top_k"]
    ordered = metrics_df.sort_values(
        order_cols,
        ascending=[True, False, False, True],
    )
    best_rows = ordered.groupby("model_name", as_index=False).head(1).reset_index(drop=True)
    return best_rows.sort_values(
        ["net_annualized_return", "net_sharpe_ratio", "top_k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _decorate_model_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []

    for record in records:
        model_name = str(record["model_name"])
        enriched.append(
            {
                **record,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "task_group": MODEL_TASKS.get(model_name, "專題"),
            }
        )

    return enriched


def _build_leaderboards() -> dict[str, Any]:
    all_models_df = _read_csv(ALL_MODELS_PORTFOLIO_METRICS_PATH)
    best_models_df = _best_strategy_rows(all_models_df)

    external_df = _read_csv(EXTERNAL_RF_PORTFOLIO_METRICS_PATH)
    best_external_df = external_df.sort_values(
        ["net_annualized_return", "net_sharpe_ratio", "top_k"],
        ascending=[False, False, True],
    ).head(3)

    classification_df = _read_csv(ALL_CLASSIFICATION_METRICS_PATH)
    overall_classification_df = classification_df[classification_df["scope"] == "overall"].copy()
    overall_classification_df = overall_classification_df.sort_values(
        ["accuracy", "f1_label_1", "precision_label_1", "recall_label_1"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    overall_classification_df["model_label"] = overall_classification_df["model_name"].map(
        lambda name: MODEL_LABELS.get(str(name), str(name))
    )

    return {
        "best_models": _decorate_model_metrics(dataframe_to_ui_records(best_models_df)),
        "best_external": _decorate_model_metrics(dataframe_to_ui_records(best_external_df)),
        "classification_overall": dataframe_to_ui_records(
            overall_classification_df[
                [
                    "model_name",
                    "model_label",
                    "accuracy",
                    "precision_label_1",
                    "recall_label_1",
                    "f1_label_1",
                ]
            ]
        ),
    }


def _build_validation_checks() -> list[dict[str, Any]]:
    next_rows = len(_read_csv(TEMPORAL_SPLITS_NEXT_YEAR_PATH))
    remaining_rows = len(_read_csv(TEMPORAL_SPLITS_REMAINING_YEARS_PATH))
    dt_pred_rows = len(_read_csv(DT_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str}))
    task2_pred_rows = len(_read_csv(TASK2_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str}))
    svr_pred_rows = len(_read_csv(SVR_GA_PREDICTIONS_PATH, dtype={STOCK_ID_COL: str}))

    dt_model_count = len(list(Path(SAVED_MODEL_DIR).glob("decision_tree_split_*.joblib")))
    task2_model_count = sum(
        len(list(Path(SAVED_MODEL_DIR).glob(f"{model_name}_split_*.joblib")))
        for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]
    )
    svr_model_count = len(list(Path(SAVED_MODEL_DIR).glob("svr_ga_regression_split_*.joblib")))
    gallery_count = sum((PROJECT_ROOT / figure["path"]).exists() for figure in CURATED_FIGURES)

    checks = [
        {
            "name": "時序切分",
            "expected": f"{EXPECTED_COUNTS['next_year_splits']} 組次年度 / {EXPECTED_COUNTS['remaining_years_splits']} 組剩餘年度",
            "actual": f"{next_rows} / {remaining_rows}",
            "status": "pass"
            if next_rows == EXPECTED_COUNTS["next_year_splits"]
            and remaining_rows == EXPECTED_COUNTS["remaining_years_splits"]
            else "fail",
        },
        {
            "name": "任務 1 預測結果",
            "expected": str(EXPECTED_COUNTS["dt_predictions"]),
            "actual": str(dt_pred_rows),
            "status": "pass" if dt_pred_rows == EXPECTED_COUNTS["dt_predictions"] else "fail",
        },
        {
            "name": "任務 2 預測結果",
            "expected": str(EXPECTED_COUNTS["task2_predictions"]),
            "actual": str(task2_pred_rows),
            "status": "pass"
            if task2_pred_rows == EXPECTED_COUNTS["task2_predictions"]
            else "fail",
        },
        {
            "name": "SVR-GA 預測結果",
            "expected": str(EXPECTED_COUNTS["svr_predictions"]),
            "actual": str(svr_pred_rows),
            "status": "pass" if svr_pred_rows == EXPECTED_COUNTS["svr_predictions"] else "fail",
        },
        {
            "name": "已保存模型輸出",
            "expected": (
                f"DT {EXPECTED_COUNTS['dt_saved_models']} / "
                f"Task2 {EXPECTED_COUNTS['task2_saved_models']} / "
                f"SVR {EXPECTED_COUNTS['svr_saved_models']}"
            ),
            "actual": f"DT {dt_model_count} / Task2 {task2_model_count} / SVR {svr_model_count}",
            "status": "pass"
            if dt_model_count == EXPECTED_COUNTS["dt_saved_models"]
            and task2_model_count == EXPECTED_COUNTS["task2_saved_models"]
            and svr_model_count == EXPECTED_COUNTS["svr_saved_models"]
            else "fail",
        },
        {
            "name": "圖表資產",
            "expected": str(len(CURATED_FIGURES)),
            "actual": str(gallery_count),
            "status": "pass" if gallery_count == len(CURATED_FIGURES) else "warn",
        },
    ]

    return checks


def _build_requirement_cards(
    leaderboard: list[dict[str, Any]],
    best_external: list[dict[str, Any]],
    latest_demo: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    by_model = {row["model_name"]: row for row in leaderboard}
    external_best = best_external[0] if best_external else None

    cards = [
        {
            "title": "任務 1：Decision Tree",
            "subtitle": "以 entropy-based 選股方法搭配時序驗證。",
            "status": "完成",
            "metric_label": "最佳淨年化報酬",
            "metric_value": by_model["decision_tree_entropy"]["net_annualized_return"],
            "evidence": [
                "已完成 11 組次年度切分與 11 個 tree artifacts",
                "Top-K 排名僅使用 score_label_1",
                "投組與 benchmark 輸出已重建並驗證",
            ],
        },
        {
            "title": "任務 2：替代分類模型",
            "subtitle": "Logistic Regression、Random Forest 與 Gradient Boosting 在相同時序設計下重跑。",
            "status": "完成",
            "metric_label": "最強分類模型",
            "metric_value": by_model["random_forest"]["net_annualized_return"],
            "evidence": [
                "Random Forest 為內部分類模型中表現最佳者",
                "整體分類指標與投組指標皆已齊備",
                "任務 2 的已保存模型登錄表已完整重建",
            ],
        },
        {
            "title": "任務 2 延伸：SVR-GA",
            "subtitle": "以 GA 最適化 regression signal，並依 predicted_return 排名。",
            "status": "完成",
            "metric_label": "最佳淨年化報酬",
            "metric_value": by_model["svr_ga_regression"]["net_annualized_return"],
            "evidence": [
                "已完成 11 組 regression 切分與 GA 搜尋紀錄",
                "已檢查 predicted_return 排名不存在 selection leakage",
                "加權選股後的 portfolio 重算結果已驗證",
            ],
        },
        {
            "title": "任務 3：外部資料重跑",
            "subtitle": "重建外部財務資料、檢查資料品質，並比較外部 RF 與 benchmark。",
            "status": "完成",
            "metric_label": "最佳外部淨年化報酬",
            "metric_value": external_best["net_annualized_return"] if external_best else None,
            "evidence": [
                "外部 portfolio metrics 與對齊後 benchmark 已可取得",
                "特徵缺值與 benchmark 圖表皆已產生",
                "爬取資料重跑結果已保留為可重現輸出",
            ],
        },
        {
            "title": "展示：CLI + Web",
            "subtitle": "採先上傳後執行的流程，並共用 preprocessing、排序與輸出匯出。",
            "status": "就緒",
            "metric_label": "最新展示淨年化報酬",
            "metric_value": (
                latest_demo["evaluation_summary"]["portfolio"]["net_annualized_return"]
                if latest_demo
                and latest_demo.get("evaluation_summary", {}).get("portfolio")
                else None
            ),
            "evidence": [
                "Web 上傳與執行端點已完成端到端驗證",
                "展示輸出會持久化到 outputs/demo",
                "若存在已保存模型則直接載入，否則改用歷史資料 refit",
            ],
        },
    ]

    return cards


def _build_project_timeline() -> list[dict[str, str]]:
    return [
        {
            "step": "1",
            "title": "資料清理",
            "detail": "標準化欄位 schema、清理 yearmonth 與 stock id、移除 200912，並轉換 16 個 canonical features 與 targets。",
        },
        {
            "step": "2",
            "title": "時序驗證",
            "detail": "建立次年度與剩餘年度切分，避免 leakage 並保留時間順序。",
        },
        {
            "step": "3",
            "title": "任務 1 建模",
            "detail": "訓練 entropy-based Decision Tree，並依 score_label_1 排名股票。",
        },
        {
            "step": "4",
            "title": "任務 2 建模",
            "detail": "在相同 temporal protocol 下訓練 LR、RF、GB 與 SVR-GA，確保可公平比較。",
        },
        {
            "step": "5",
            "title": "投組與 benchmark",
            "detail": "建立含交易成本的等權與分數加權投組，並與 all-stock / random benchmark 比較。",
        },
        {
            "step": "6",
            "title": "外部資料重跑與展示",
            "detail": "重建 external factors、比較外部模型與 benchmark，並提供可重現的 Web 展示。",
        },
    ]


def _build_gallery() -> list[dict[str, Any]]:
    gallery = []

    for figure in CURATED_FIGURES:
        target = PROJECT_ROOT / figure["path"]
        gallery.append(
            {
                "section": figure["section"],
                "title": figure["title"],
                "caption": figure["caption"],
                "exists": target.exists(),
                "url": _artifact_url(target),
            }
        )

    return gallery


def _build_downloads() -> list[dict[str, str]]:
    return [
        {
            "label": label,
            "url": _artifact_url(PROJECT_ROOT / path),
        }
        for label, path in KEY_DOWNLOADS
        if (PROJECT_ROOT / path).exists()
    ]


def _build_portfolio_curve(portfolio_returns_df: pd.DataFrame) -> list[dict[str, Any]]:
    if portfolio_returns_df.empty:
        return []

    curve_df = portfolio_returns_df.sort_values(YEAR_COL).copy()
    curve_df["cumulative_net_return"] = (
        (1.0 + curve_df["net_return"].astype(float)).cumprod() - 1.0
    )

    return [
        {
            "year": int(row[YEAR_COL]),
            "gross_return": _safe_value(row["gross_return"]),
            "net_return": _safe_value(row["net_return"]),
            "cumulative_net_return": _safe_value(row["cumulative_net_return"]),
        }
        for _, row in curve_df.iterrows()
    ]


def _extract_demo_input_path() -> str | None:
    path = Path(DEMO_PROFILE_PATH)
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Input file: "):
            return line.replace("Input file: ", "", 1).strip()
        if line.startswith("輸入檔案："):
            return line.replace("輸入檔案：", "", 1).strip()

    return None


def build_saved_demo_payload() -> dict[str, Any] | None:
    demo_predictions_path = Path(DEMO_PREDICTIONS_PATH)
    demo_selected_path = Path(DEMO_SELECTED_STOCKS_PATH)
    demo_returns_path = Path(DEMO_PORTFOLIO_RETURNS_PATH)
    demo_metrics_path = Path(DEMO_METRICS_PATH)

    if not demo_predictions_path.exists() or not demo_selected_path.exists():
        return None

    predictions_df = _read_csv(demo_predictions_path, dtype={STOCK_ID_COL: str})
    selected_df = _read_csv(demo_selected_path, dtype={STOCK_ID_COL: str})
    portfolio_returns_df = (
        _read_csv(demo_returns_path) if demo_returns_path.exists() else pd.DataFrame()
    )
    portfolio_metrics_df = (
        _read_csv(demo_metrics_path) if demo_metrics_path.exists() else pd.DataFrame()
    )

    if "actual_label" in predictions_df.columns and predictions_df["actual_label"].notna().any():
        classification_metrics_df = calculate_classification_metrics_by_model(
            predictions_df.dropna(subset=["actual_label"]).copy()
        )
    else:
        classification_metrics_df = pd.DataFrame()

    evaluation_summary: dict[str, Any] = {
        "prediction_rows": int(len(predictions_df)),
        "selected_rows": int(len(selected_df)),
        "predicted_positive_rate": _safe_value(
            (predictions_df["predicted_label"] == 1).mean()
        ),
        "mean_score_label_1": _safe_value(predictions_df["score_label_1"].mean()),
        "portfolio_curve": _build_portfolio_curve(portfolio_returns_df),
    }

    if not portfolio_metrics_df.empty:
        row = portfolio_metrics_df.iloc[0]
        net_returns = (
            portfolio_returns_df.sort_values(YEAR_COL)["net_return"].astype(float)
            if not portfolio_returns_df.empty
            else pd.Series(dtype=float)
        )
        evaluation_summary["portfolio"] = {
            "net_annualized_return": _safe_value(row["net_annualized_return"]),
            "net_cumulative_return": _safe_value(row["net_cumulative_return"]),
            "net_maximum_drawdown": _safe_value(row["net_maximum_drawdown"]),
            "net_sharpe_ratio": _safe_value(row["net_sharpe_ratio"]),
            "net_win_rate": _safe_value(row["net_win_rate"]),
            "recomputed_cumulative_net_return": _safe_value(calculate_cumulative_return(net_returns)),
        }

    if not classification_metrics_df.empty:
        overall = classification_metrics_df[classification_metrics_df["scope"] == "overall"].copy()
        if not overall.empty:
            row = overall.iloc[0]
            evaluation_summary["classification"] = {
                "accuracy": _safe_value(row["accuracy"]),
                "precision_label_1": _safe_value(row["precision_label_1"]),
                "recall_label_1": _safe_value(row["recall_label_1"]),
                "f1_label_1": _safe_value(row["f1_label_1"]),
            }

    model_name = str(predictions_df["model_name"].iloc[0]) if not predictions_df.empty else "demo"
    top_k = int(selected_df["top_k"].iloc[0]) if "top_k" in selected_df.columns and not selected_df.empty else 0

    return {
        "run_id": "latest-demo",
        "is_saved_demo": True,
        "request": {
            "input_path": _extract_demo_input_path() or "outputs/demo",
            "model_name": model_name,
            "top_k": top_k,
        },
        "evaluation_summary": evaluation_summary,
        "artifacts": {
            "predictions": str(demo_predictions_path),
            "selected_stocks": str(demo_selected_path),
            "portfolio_returns": str(demo_returns_path) if demo_returns_path.exists() else None,
            "metrics": str(demo_metrics_path) if demo_metrics_path.exists() else None,
            "profile": str(DEMO_PROFILE_PATH) if Path(DEMO_PROFILE_PATH).exists() else None,
        },
        "tables": {
            "predictions": dataframe_to_ui_records(
                predictions_df.sort_values([YEAR_COL, "score_label_1"], ascending=[True, False])
            ),
            "selected_stocks": dataframe_to_ui_records(
                selected_df.sort_values([YEAR_COL, "rank"], ascending=[True, True])
            ),
            "portfolio_returns": dataframe_to_ui_records(portfolio_returns_df.sort_values(YEAR_COL)),
            "portfolio_metrics": dataframe_to_ui_records(portfolio_metrics_df),
            "classification_metrics": dataframe_to_ui_records(classification_metrics_df),
        },
    }


def build_project_overview_payload() -> dict[str, Any]:
    dataset = _build_dataset_summary()
    splits = _build_split_summary()
    leaderboards = _build_leaderboards()
    latest_demo = build_saved_demo_payload()

    return {
        "project_title": "AIFT Final Project 專題儀表板",
        "dataset": dataset,
        "splits": splits,
        "validation_checks": _build_validation_checks(),
        "requirement_cards": _build_requirement_cards(
            leaderboard=leaderboards["best_models"],
            best_external=leaderboards["best_external"],
            latest_demo=latest_demo,
        ),
        "timeline": _build_project_timeline(),
        "leaderboards": leaderboards,
        "gallery": _build_gallery(),
        "downloads": _build_downloads(),
        "latest_demo": latest_demo,
    }
