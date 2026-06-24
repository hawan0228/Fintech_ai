# webapp/service.py
"""
Web demo inference service.

設計原則：
- 直接重用 src/ 既有分析模組（preprocessing / models / prediction / portfolio / metrics）。
- 採 README 15.8 策略 B：依 demo 年份，只用該年以前的歷史 cleaned data 重新訓練後推論，
  與目前 CLI demo (src/demo_runner.py) 完全一致，避免 demo 與正式流程行為不一致。
- 不在伺服器端管理 job 狀態：一個 request 直接回傳完整 JSON 結果（stateless）。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    SAVED_MODEL_DIR,
    TEMPORAL_SPLITS_NEXT_YEAR_PATH,
)
from src.schema import (
    FEATURE_COLUMNS,
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEARMONTH_COL,
    YEAR_COL,
    TARGET_CLASS,
    TARGET_RETURN,
)
from src.preprocessing import standardize_column_names
from src.models import get_classification_pipeline
from src.prediction import get_probability_for_label_one
from src.portfolio import calculate_portfolio_returns
from src.metrics import summarize_strategy_metrics
from src.demo_runner import load_table, prepare_demo_data, select_top_k_demo


# =========================
# Model registry
# =========================

AVAILABLE_MODELS = [
    {"value": "decision_tree_entropy", "label": "Decision Tree (Task 1)"},
    {"value": "logistic_regression", "label": "Logistic Regression"},
    {"value": "random_forest", "label": "Random Forest"},
    {"value": "gradient_boosting", "label": "Gradient Boosting"},
]

_MODEL_VALUES = {m["value"] for m in AVAILABLE_MODELS}
_MODEL_LABELS = {m["value"]: m["label"] for m in AVAILABLE_MODELS}

# saved_models/ 內的檔名前綴。注意 decision_tree_entropy 存檔時前綴為 decision_tree。
_MODEL_FILE_PREFIX = {
    "decision_tree_entropy": "decision_tree",
    "logistic_regression": "logistic_regression",
    "random_forest": "random_forest",
    "gradient_boosting": "gradient_boosting",
}

# 推論策略
STRATEGY_RETRAIN = "retrain"        # 策略 B：依年份用歷史資料即時重訓
STRATEGY_PRETRAINED = "pretrained"  # 策略 A：載入 saved_models/ 預訓練模型
VALID_STRATEGIES = {STRATEGY_RETRAIN, STRATEGY_PRETRAINED}


def list_models() -> list[dict[str, str]]:
    return AVAILABLE_MODELS


def feature_schema() -> dict[str, Any]:
    """提供前端顯示「必要欄位 / 特徵欄位」的說明。"""
    return {
        "id_columns": [STOCK_ID_COL, STOCK_NAME_COL, YEARMONTH_COL],
        "feature_columns": list(FEATURE_COLUMNS),
        "optional_target_columns": [TARGET_RETURN, TARGET_CLASS],
        "note": (
            "至少需要 證券代碼 與 年月。若含 Return / ReturnMean_year_Label，"
            "會額外計算投組績效。缺少的特徵欄位會以中位數補值。"
        ),
    }


# =========================
# Pretrained models (策略 A)
# =========================

_YEAR_SPLIT_MAP: dict[int, int] | None = None
_PRETRAINED_CACHE: dict[tuple[str, int], Any] = {}


def _load_year_split_map() -> dict[int, int]:
    """
    從 next_year temporal splits 建立 {test_year: split_id}。
    next_year 模式每個 split 只測一個年份，因此 test_year 唯一對應一個 split。
    """
    global _YEAR_SPLIT_MAP
    if _YEAR_SPLIT_MAP is None:
        if not Path(TEMPORAL_SPLITS_NEXT_YEAR_PATH).exists():
            _YEAR_SPLIT_MAP = {}
            return _YEAR_SPLIT_MAP
        splits = pd.read_csv(TEMPORAL_SPLITS_NEXT_YEAR_PATH)
        mapping: dict[int, int] = {}
        for _, row in splits.iterrows():
            for y in str(row["test_years"]).split(","):
                y = y.strip()
                if y:
                    mapping[int(y)] = int(row["split_id"])
        _YEAR_SPLIT_MAP = mapping
    return _YEAR_SPLIT_MAP


def _pretrained_path(model_value: str, split_id: int) -> Path:
    prefix = _MODEL_FILE_PREFIX[model_value]
    return Path(SAVED_MODEL_DIR) / f"{prefix}_split_{split_id:02d}.joblib"


def _load_pretrained_model(model_value: str, split_id: int):
    """載入並快取單一預訓練模型 pipeline。找不到回傳 None。"""
    key = (model_value, split_id)
    if key not in _PRETRAINED_CACHE:
        path = _pretrained_path(model_value, split_id)
        _PRETRAINED_CACHE[key] = joblib.load(path) if path.exists() else None
    return _PRETRAINED_CACHE[key]


def pretrained_status() -> dict[str, Any]:
    """回報 saved_models/ 預訓練模型可用情形，供前端決定是否開放策略 A。"""
    year_split = _load_year_split_map()
    covered_years = sorted(year_split.keys())

    per_model: dict[str, list[int]] = {}
    total = 0
    for model_value in _MODEL_FILE_PREFIX:
        years = [y for y in covered_years if _pretrained_path(model_value, year_split[y]).exists()]
        per_model[model_value] = years
        total += len(years)

    # 只有「每個分類模型在每個年份都有檔」才算完整可用
    available = bool(covered_years) and all(
        len(per_model[m]) == len(covered_years) for m in _MODEL_FILE_PREFIX
    )

    return {
        "available": available,
        "n_model_files": total,
        "covered_years": covered_years,
        "per_model_years": per_model,
        "saved_model_dir": str(SAVED_MODEL_DIR),
    }


# =========================
# Training data (cached)
# =========================

_TRAIN_DF: pd.DataFrame | None = None


def _load_training_data() -> pd.DataFrame:
    global _TRAIN_DF
    if _TRAIN_DF is None:
        if not Path(CLEANED_DATA_PATH).exists():
            raise FileNotFoundError(
                f"找不到 cleaned training data：{CLEANED_DATA_PATH}。"
                "請先執行 python prepare_data.py。"
            )
        _TRAIN_DF = pd.read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    return _TRAIN_DF


# =========================
# JSON sanitization
# =========================

def _to_jsonable(value: Any) -> Any:
    """
    遞迴清理：把 numpy 型別轉 python，NaN/Inf 轉 None，
    讓結果可被嚴格 JSON parser（JS JSON.parse）解析。
    """
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return _to_jsonable(df.replace({np.nan: None}).to_dict(orient="records"))


# =========================
# Validation
# =========================

class DemoInputError(ValueError):
    """前端可讀的輸入錯誤。"""


def _validate_raw_table(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """
    載入並做基本欄位驗證，回傳 (raw_df_after_alias, warnings)。
    在進入正式 preprocessing 前先給出人類可讀的錯誤。
    """
    raw_df = load_table(path)
    if raw_df.empty:
        raise DemoInputError("上傳的檔案沒有任何資料列。")

    aliased = standardize_column_names(raw_df)

    warnings: list[str] = []

    if STOCK_ID_COL not in aliased.columns:
        raise DemoInputError(
            f"缺少必要欄位「{STOCK_ID_COL}」。請確認檔案含證券代碼欄位。"
        )

    if YEARMONTH_COL not in aliased.columns and YEAR_COL not in aliased.columns:
        raise DemoInputError(
            f"缺少必要欄位「{YEARMONTH_COL}」，無法解析年份。"
        )

    # 依規格會移除 年月=200912（特殊無效列），先告知使用者
    if YEARMONTH_COL in aliased.columns:
        n_200912 = int((pd.to_numeric(aliased[YEARMONTH_COL], errors="coerce") == 200912).sum())
        if n_200912:
            warnings.append(
                f"已依規格移除 年月=200912 的 {n_200912} 筆資料（與訓練前處理一致，"
                "該批為特殊無效列）。"
            )

    missing_features = [c for c in FEATURE_COLUMNS if c not in aliased.columns]
    if missing_features:
        warnings.append(
            f"有 {len(missing_features)} 個特徵欄位不存在，將以缺失值（中位數補值）處理："
            + "、".join(missing_features[:6])
            + ("…" if len(missing_features) > 6 else "")
        )

    return aliased, warnings


# =========================
# Main inference
# =========================

def run_inference(
    input_path: str | Path,
    model_name: str = "random_forest",
    top_k: int = 10,
    strategy: str = STRATEGY_RETRAIN,
) -> dict[str, Any]:
    """
    對上傳的測試資料執行 demo 推論，回傳 JSON-serializable 結果。

    strategy:
    - "retrain"（策略 B）：依 demo 年份用該年以前歷史資料即時重訓。
    - "pretrained"（策略 A）：載入 saved_models/ 對應 split 的預訓練模型，免訓練、秒回。
      找不到對應模型時，該年自動 fallback 為即時重訓。
    """
    if model_name not in _MODEL_VALUES:
        raise DemoInputError(
            f"不支援的模型：{model_name}。可用模型："
            + "、".join(sorted(_MODEL_VALUES))
        )

    if not isinstance(top_k, int) or top_k < 1:
        raise DemoInputError("Top-K 必須是 >= 1 的整數。")

    if strategy not in VALID_STRATEGIES:
        raise DemoInputError(
            f"不支援的推論策略：{strategy}。可用：retrain / pretrained。"
        )

    # 策略 A 但完全沒有預訓練模型 → 退回即時重訓（安全網）
    if strategy == STRATEGY_PRETRAINED and not pretrained_status()["available"]:
        warnings_pretrained_missing = True
        strategy = STRATEGY_RETRAIN
    else:
        warnings_pretrained_missing = False

    # 1. 基本驗證（人類可讀錯誤）
    _aliased, warnings = _validate_raw_table(input_path)

    if warnings_pretrained_missing:
        warnings.append(
            "找不到完整的預訓練模型（saved_models/），已自動改用即時重訓。"
            "可執行各 train_*.py 產生模型檔後再使用策略 A。"
        )

    # 2. 正式前處理（與 CLI demo 共用同一套邏輯）
    try:
        demo_df = prepare_demo_data(input_path)
    except Exception as exc:  # noqa: BLE001 - 轉成前端可讀錯誤
        raise DemoInputError(f"資料前處理失敗：{exc}") from exc

    demo_years = sorted(demo_df[YEAR_COL].dropna().astype(int).unique().tolist())
    if not demo_years:
        raise DemoInputError("無法從「年月」解析出任何有效年份。")

    if top_k > len(demo_df):
        warnings.append(
            f"Top-K ({top_k}) 大於資料筆數 ({len(demo_df)})，將回傳全部股票。"
        )

    has_return = TARGET_RETURN in demo_df.columns and demo_df[TARGET_RETURN].notna().any()
    has_label = TARGET_CLASS in demo_df.columns and demo_df[TARGET_CLASS].notna().any()

    train_df = _load_training_data()
    year_split_map = _load_year_split_map() if strategy == STRATEGY_PRETRAINED else {}

    def _retrain_predict(year: int, X_test: pd.DataFrame):
        """策略 B：依年份用歷史資料即時重訓後預測。"""
        year_train_df = train_df[train_df[YEAR_COL].astype(int) < int(year)].copy()
        if year_train_df.empty:
            warnings.append(
                f"沒有 year < {year} 的歷史訓練資料，改用全部 cleaned training data 訓練。"
            )
            year_train_df = train_df.copy()
        pipeline = get_classification_pipeline(model_name)
        pipeline.fit(
            year_train_df[FEATURE_COLUMNS].copy(),
            year_train_df[TARGET_CLASS].astype(int).copy(),
        )
        return pipeline

    # 3. 逐年推論
    all_predictions = []
    model_sources: list[str] = []
    for year in demo_years:
        year_demo_df = demo_df[demo_df[YEAR_COL].astype(int) == int(year)].copy()
        X_test = year_demo_df[FEATURE_COLUMNS].copy()

        if strategy == STRATEGY_PRETRAINED:
            split_id = year_split_map.get(int(year))
            pipeline = (
                _load_pretrained_model(model_name, split_id)
                if split_id is not None
                else None
            )
            if pipeline is not None:
                source = f"pretrained(split_{split_id:02d})"
            else:
                warnings.append(
                    f"{year} 年無對應預訓練模型，該年改為即時重訓。"
                )
                pipeline = _retrain_predict(year, X_test)
                source = "retrain(fallback)"
        else:
            pipeline = _retrain_predict(year, X_test)
            source = "retrain"

        model_sources.append(source)

        predicted_label = pipeline.predict(X_test).astype(int)
        score_label_1 = get_probability_for_label_one(pipeline, X_test)

        pred_df = year_demo_df[[YEAR_COL, STOCK_ID_COL, STOCK_NAME_COL]].copy()
        pred_df.insert(0, "split_id", 0)
        pred_df.insert(1, "model_name", model_name)
        pred_df.insert(2, "train_years", f"historical_before_{year}")
        pred_df.insert(3, "test_years", str(year))
        pred_df["predicted_label"] = predicted_label
        pred_df["score_label_1"] = score_label_1
        pred_df["actual_label"] = (
            year_demo_df[TARGET_CLASS].astype("Int64").values if has_label else pd.NA
        )
        pred_df["actual_return"] = (
            year_demo_df[TARGET_RETURN].astype(float).values if has_return else np.nan
        )

        all_predictions.append(pred_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # 4. Top-K 選股
    selected_df = select_top_k_demo(
        predictions_df=predictions_df,
        top_k_list=[top_k],
        score_col="score_label_1",
    )
    selected_df = selected_df.sort_values([YEAR_COL, "rank"]).reset_index(drop=True)

    # 5. 投組績效（僅在有真實 Return 時）
    portfolio_block: dict[str, Any] = {"available": False, "reason": None}
    if has_return:
        sel_with_weights, portfolio_returns_df = calculate_portfolio_returns(
            selected_df=selected_df,
            weight_methods=["equal"],
            score_col="score_label_1",
        )
        metrics_df = summarize_strategy_metrics(
            returns_df=portfolio_returns_df,
            group_cols=["model_name", "top_k", "weight_method"],
        )
        metric_row = metrics_df.iloc[0].to_dict()
        portfolio_block = {
            "available": True,
            "reason": None,
            "weight_method": "equal",
            "annual_returns": _df_to_records(
                portfolio_returns_df[
                    [YEAR_COL, "top_k", "weight_method", "gross_return", "net_return"]
                ]
            ),
            "summary": _to_jsonable(
                {
                    "mean_annual_net_return": metric_row.get("net_mean_annual_return"),
                    "annualized_net_return": metric_row.get("net_annualized_return"),
                    "cumulative_net_return": metric_row.get("net_cumulative_return"),
                    "maximum_drawdown": metric_row.get("net_maximum_drawdown"),
                    "volatility": metric_row.get("net_volatility"),
                    "sharpe_ratio": metric_row.get("net_sharpe_ratio"),
                    "win_rate": metric_row.get("net_win_rate"),
                    "n_years": metric_row.get("n_years"),
                }
            ),
        }
    else:
        portfolio_block["reason"] = (
            "此資料不含真實報酬欄位（Return），因此無法計算 realized performance。"
        )

    # 6. 分類品質（僅在有真實 label 時）
    classification_block: dict[str, Any] | None = None
    if has_label:
        eval_df = predictions_df.dropna(subset=["actual_label"]).copy()
        if not eval_df.empty:
            y_true = eval_df["actual_label"].astype(int)
            y_pred = eval_df["predicted_label"].astype(int)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == -1) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == -1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            classification_block = _to_jsonable(
                {
                    "accuracy": float((y_true == y_pred).mean()),
                    "precision_label_1": precision,
                    "recall_label_1": recall,
                    "f1_label_1": f1,
                    "n_samples": int(len(eval_df)),
                }
            )

    # 7. 特徵缺失統計
    feature_missing = {
        col: int(demo_df[col].isna().sum())
        for col in FEATURE_COLUMNS
        if demo_df[col].isna().any()
    }

    strategy_label = (
        "載入預訓練模型 (策略A)"
        if strategy == STRATEGY_PRETRAINED
        else "即時重訓 (策略B)"
    )
    summary = {
        "model_name": model_name,
        "model_label": _MODEL_LABELS[model_name],
        "top_k": int(top_k),
        "n_rows": int(len(demo_df)),
        "demo_years": demo_years,
        "n_predicted_stocks": int(len(predictions_df)),
        "has_return": bool(has_return),
        "has_label": bool(has_label),
        "feature_missing_count": int(sum(feature_missing.values())),
        "strategy": strategy,
        "strategy_label": strategy_label,
        "model_sources": sorted(set(model_sources)),
    }

    return {
        "ok": True,
        "summary": _to_jsonable(summary),
        "selected_stocks": _df_to_records(
            selected_df[
                [
                    YEAR_COL,
                    "rank",
                    STOCK_ID_COL,
                    STOCK_NAME_COL,
                    "score_label_1",
                    "predicted_label",
                    "actual_label",
                    "actual_return",
                ]
            ]
        ),
        "predictions": _df_to_records(
            predictions_df[
                [
                    YEAR_COL,
                    STOCK_ID_COL,
                    STOCK_NAME_COL,
                    "predicted_label",
                    "score_label_1",
                    "actual_label",
                    "actual_return",
                ]
            ]
        ),
        "portfolio": portfolio_block,
        "classification": classification_block,
        "feature_missing": _to_jsonable(feature_missing),
        "warnings": warnings,
    }
