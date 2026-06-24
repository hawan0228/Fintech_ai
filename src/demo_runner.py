from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    DEMO_METRICS_PATH,
    DEMO_PORTFOLIO_RETURNS_PATH,
    DEMO_PREDICTIONS_PATH,
    DEMO_PROFILE_PATH,
    DEMO_SELECTED_STOCKS_PATH,
    SAVED_MODEL_DIR,
    TEMPORAL_SPLITS_NEXT_YEAR_PATH,
)
from src.metrics import (
    calculate_classification_metrics_by_model,
    calculate_cumulative_return,
    summarize_strategy_metrics,
)
from src.models import get_classification_pipeline
from src.portfolio import calculate_portfolio_returns
from src.prediction import get_probability_for_label_one
from src.preprocessing import (
    clean_stock_id,
    clean_stock_name,
    clean_yearmonth,
    convert_percent_or_numeric_series,
    create_year_column,
    sort_cleaned_data,
    standardize_column_names,
)
from src.schema import (
    FEATURE_COLUMNS,
    STOCK_ID_COL,
    STOCK_NAME_COL,
    TARGET_CLASS,
    TARGET_RETURN,
    YEARMONTH_COL,
    YEAR_COL,
)


SUPPORTED_DEMO_MODELS = [
    "decision_tree_entropy",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]


MODEL_METADATA: dict[str, dict[str, str]] = {
    "decision_tree_entropy": {
        "label": "Decision Tree (Entropy)",
        "task": "Task 1 基線模型",
        "strategy": "以 ID3-like entropy classifier 建模，並依 P(label = 1) 排名。",
    },
    "logistic_regression": {
        "label": "Logistic Regression",
        "task": "Task 2 基線模型",
        "strategy": "以含 scaling 的 linear classifier 建模，並依 P(label = 1) 排名。",
    },
    "random_forest": {
        "label": "Random Forest",
        "task": "Task 2 主力模型",
        "strategy": "以 tree ensemble 建模，並依 P(label = 1) 排名。",
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "task": "Task 2 延伸模型",
        "strategy": "以 boosted trees 建模，並依 P(label = 1) 排名。",
    },
}


@dataclass
class DemoExecutionResult:
    input_path: Path
    model_name: str
    top_k: int
    prepared_df: pd.DataFrame
    input_summary: dict[str, Any]
    predictions_df: pd.DataFrame
    selected_df: pd.DataFrame
    selected_with_weights_df: pd.DataFrame
    portfolio_returns_df: pd.DataFrame
    portfolio_metrics_df: pd.DataFrame
    classification_metrics_df: pd.DataFrame
    evaluation_summary: dict[str, Any]
    model_sources: list[dict[str, Any]]
    output_paths: dict[str, Path]
    profile_text: str


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"輸入檔案不存在：{path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"不支援的輸入格式：{suffix}")


def _series_to_int_years(series: pd.Series) -> list[int]:
    cleaned = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    return sorted(cleaned.unique().tolist())


def _build_feature_coverage(prepared_df: pd.DataFrame) -> list[dict[str, Any]]:
    total_rows = len(prepared_df)
    coverage = []

    for feature in FEATURE_COLUMNS:
        non_null_count = int(prepared_df[feature].notna().sum())
        non_null_ratio = 0.0 if total_rows == 0 else non_null_count / total_rows

        coverage.append(
            {
                "feature": feature,
                "non_null_count": non_null_count,
                "non_null_ratio": round(non_null_ratio, 4),
            }
        )

    coverage.sort(key=lambda item: (item["non_null_ratio"], item["feature"]))
    return coverage


def _build_input_summary(
    raw_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    original_columns: set[str],
) -> dict[str, Any]:
    years = _series_to_int_years(prepared_df[YEAR_COL])
    rows_by_year = (
        prepared_df.groupby(YEAR_COL, dropna=True)
        .size()
        .reset_index(name="n_rows")
        .sort_values(YEAR_COL)
    )

    missing_feature_columns = [
        feature for feature in FEATURE_COLUMNS if feature not in original_columns
    ]

    has_return = (
        TARGET_RETURN in prepared_df.columns
        and prepared_df[TARGET_RETURN].notna().any()
    )
    has_label = (
        TARGET_CLASS in prepared_df.columns
        and prepared_df[TARGET_CLASS].notna().any()
    )

    return {
        "rows_raw": int(len(raw_df)),
        "rows_prepared": int(len(prepared_df)),
        "columns_raw": int(raw_df.shape[1]),
        "years": years,
        "rows_by_year": [
            {
                "year": int(row[YEAR_COL]),
                "n_rows": int(row["n_rows"]),
            }
            for _, row in rows_by_year.iterrows()
        ],
        "missing_feature_columns": missing_feature_columns,
        "available_feature_columns": [
            feature for feature in FEATURE_COLUMNS if feature in original_columns
        ],
        "feature_coverage": _build_feature_coverage(prepared_df),
        "has_return": bool(has_return),
        "has_label": bool(has_label),
    }


def prepare_demo_data(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_df = load_table(path)
    df = standardize_column_names(raw_df)
    original_columns = set(df.columns)

    required_columns = [STOCK_ID_COL, YEARMONTH_COL]
    missing_required = [col for col in required_columns if col not in df.columns]

    if missing_required:
        missing_text = ", ".join(missing_required)
        raise ValueError(
            "Demo 輸入缺少必要欄位："
            f"{missing_text}。檔案必須包含 stock id 與 yearmonth 欄位。"
        )

    if STOCK_NAME_COL not in df.columns:
        df[STOCK_NAME_COL] = df[STOCK_ID_COL]

    df = clean_yearmonth(df)
    df = create_year_column(df)
    df = clean_stock_id(df)
    df = clean_stock_name(df)

    df[STOCK_ID_COL] = (
        df[STOCK_ID_COL]
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""})
        .str.zfill(4)
    )

    df = df[df[YEAR_COL].notna()].copy()
    df = df[df[STOCK_ID_COL] != ""].copy()

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = convert_percent_or_numeric_series(df[col])

    if TARGET_RETURN in df.columns:
        df[TARGET_RETURN] = convert_percent_or_numeric_series(df[TARGET_RETURN])

    if TARGET_CLASS in df.columns:
        df[TARGET_CLASS] = convert_percent_or_numeric_series(df[TARGET_CLASS]).astype("Int64")

    df = sort_cleaned_data(df)

    if df.empty:
        raise ValueError("Demo 輸入在 preprocessing 後為空。")

    summary = _build_input_summary(raw_df=raw_df, prepared_df=df, original_columns=original_columns)
    return df, summary


def _saved_model_filename(model_name: str, split_id: int) -> str:
    if model_name == "decision_tree_entropy":
        return f"decision_tree_split_{split_id:02d}.joblib"

    return f"{model_name}_split_{split_id:02d}.joblib"


def _load_saved_model_for_year(
    model_name: str,
    year: int,
) -> tuple[Any, int, Path] | None:
    split_path = Path(TEMPORAL_SPLITS_NEXT_YEAR_PATH)
    saved_model_dir = Path(SAVED_MODEL_DIR)

    if not split_path.exists() or not saved_model_dir.exists():
        return None

    splits_df = pd.read_csv(split_path)
    matched = splits_df[splits_df["test_years"].astype(str) == str(year)]

    if matched.empty:
        return None

    split_id = int(matched.iloc[0]["split_id"])
    model_path = saved_model_dir / _saved_model_filename(model_name, split_id)

    if not model_path.exists():
        return None

    return joblib.load(model_path), split_id, model_path


def _fit_historical_model(
    train_df: pd.DataFrame,
    model_name: str,
    year: int,
) -> tuple[Any, int, str]:
    year_train_df = train_df[train_df[YEAR_COL].astype(int) < int(year)].copy()

    if year_train_df.empty:
        raise ValueError(
            f"無法為 {year} 年建立 Demo 模型：該年度之前沒有可用的歷史訓練資料。"
        )

    pipeline = get_classification_pipeline(model_name)
    pipeline.fit(
        year_train_df[FEATURE_COLUMNS].copy(),
        year_train_df[TARGET_CLASS].astype(int).copy(),
    )

    return pipeline, 0, f"historical_refit_before_{year}"


def _resolve_demo_pipeline(
    train_df: pd.DataFrame,
    model_name: str,
    year: int,
) -> tuple[Any, int, str]:
    saved = _load_saved_model_for_year(model_name=model_name, year=year)

    if saved is not None:
        pipeline, split_id, model_path = saved
        return pipeline, split_id, f"saved_model:{model_path.name}"

    return _fit_historical_model(train_df=train_df, model_name=model_name, year=year)


def select_top_k_demo(
    predictions_df: pd.DataFrame,
    top_k_list: list[int],
    score_col: str = "score_label_1",
) -> pd.DataFrame:
    records = []

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
                    f"{year} 年只有 {len(group_df)} 筆資料，不足以選出 Top-{top_k}。"
                )

            selected = group_df.head(top_k).copy()
            selected["top_k"] = int(top_k)
            records.append(selected)

    return pd.concat(records, ignore_index=True)


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
            "gross_return": round(float(row["gross_return"]), 6),
            "net_return": round(float(row["net_return"]), 6),
            "cumulative_net_return": round(float(row["cumulative_net_return"]), 6),
        }
        for _, row in curve_df.iterrows()
    ]


def _build_evaluation_summary(
    predictions_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    portfolio_returns_df: pd.DataFrame,
    portfolio_metrics_df: pd.DataFrame,
    classification_metrics_df: pd.DataFrame,
) -> dict[str, Any]:
    predicted_positive_rate = float((predictions_df["predicted_label"] == 1).mean())
    mean_score = float(predictions_df["score_label_1"].mean())

    summary: dict[str, Any] = {
        "prediction_rows": int(len(predictions_df)),
        "selected_rows": int(len(selected_df)),
        "predicted_positive_rate": round(predicted_positive_rate, 6),
        "mean_score_label_1": round(mean_score, 6),
        "portfolio_curve": _build_portfolio_curve(portfolio_returns_df),
    }

    if not classification_metrics_df.empty:
        overall = classification_metrics_df[
            classification_metrics_df["scope"] == "overall"
        ].copy()

        if not overall.empty:
            row = overall.iloc[0]
            summary["classification"] = {
                "accuracy": round(float(row["accuracy"]), 6),
                "precision_label_1": round(float(row["precision_label_1"]), 6),
                "recall_label_1": round(float(row["recall_label_1"]), 6),
                "f1_label_1": round(float(row["f1_label_1"]), 6),
            }

    if not portfolio_metrics_df.empty:
        row = portfolio_metrics_df.iloc[0]
        net_returns = portfolio_returns_df.sort_values(YEAR_COL)["net_return"].astype(float)
        summary["portfolio"] = {
            "gross_annualized_return": round(float(row["gross_annualized_return"]), 6),
            "net_annualized_return": round(float(row["net_annualized_return"]), 6),
            "net_cumulative_return": round(float(row["net_cumulative_return"]), 6),
            "net_maximum_drawdown": round(float(row["net_maximum_drawdown"]), 6),
            "net_sharpe_ratio": round(float(row["net_sharpe_ratio"]), 6)
            if pd.notna(row["net_sharpe_ratio"])
            else None,
            "net_win_rate": round(float(row["net_win_rate"]), 6)
            if pd.notna(row["net_win_rate"])
            else None,
            "mean_net_return": round(float(net_returns.mean()), 6),
            "best_year_net_return": round(float(net_returns.max()), 6),
            "worst_year_net_return": round(float(net_returns.min()), 6),
            "recomputed_cumulative_net_return": round(
                float(calculate_cumulative_return(net_returns)),
                6,
            ),
        }

    return summary


def _output_paths() -> dict[str, Path]:
    return {
        "predictions": Path(DEMO_PREDICTIONS_PATH),
        "selected_stocks": Path(DEMO_SELECTED_STOCKS_PATH),
        "portfolio_returns": Path(DEMO_PORTFOLIO_RETURNS_PATH),
        "metrics": Path(DEMO_METRICS_PATH),
        "profile": Path(DEMO_PROFILE_PATH),
    }


def _generate_profile_text(result: DemoExecutionResult) -> str:
    lines = [
        "========== Demo 設定檔 ==========",
        f"輸入檔案：{result.input_path}",
        f"模型：{result.model_name}",
        f"Top-K: {result.top_k}",
        f"Demo 年度：{result.input_summary['years']}",
        f"原始/可用筆數：{result.input_summary['rows_raw']} / {result.input_summary['rows_prepared']}",
        f"含 Return：{result.input_summary['has_return']}",
        f"含 Label：{result.input_summary['has_label']}",
        f"預測筆數：{len(result.predictions_df)}",
        f"入選筆數：{len(result.selected_df)}",
        "",
        "各年度模型來源：",
    ]

    for source in result.model_sources:
        lines.append(
            f"- {source['year']}: split_id={source['split_id']}, source={source['source']}"
        )

    lines.extend(
        [
            "",
            "輸出檔案：",
        ]
    )

    for key, path in result.output_paths.items():
        lines.append(f"- {key}: {path}")

    lines.append("========== Demo 設定檔結束 ==========")
    return "\n".join(lines)


def _save_demo_outputs(result: DemoExecutionResult) -> None:
    for path in result.output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    result.predictions_df.to_csv(
        result.output_paths["predictions"],
        index=False,
        encoding="utf-8-sig",
    )

    result.selected_with_weights_df.to_csv(
        result.output_paths["selected_stocks"],
        index=False,
        encoding="utf-8-sig",
    )

    result.portfolio_returns_df.to_csv(
        result.output_paths["portfolio_returns"],
        index=False,
        encoding="utf-8-sig",
    )

    result.portfolio_metrics_df.to_csv(
        result.output_paths["metrics"],
        index=False,
        encoding="utf-8-sig",
    )

    result.output_paths["profile"].write_text(result.profile_text, encoding="utf-8")


def run_demo_analysis(
    input_path: str | Path,
    model_name: str = "random_forest",
    top_k: int = 10,
    persist_outputs: bool = True,
) -> DemoExecutionResult:
    if model_name not in SUPPORTED_DEMO_MODELS:
        raise ValueError(
            f"不支援的 Demo 模型：{model_name}。可用模型為 {SUPPORTED_DEMO_MODELS}。"
        )

    if top_k <= 0:
        raise ValueError("Top-K 必須為正整數。")

    train_df = pd.read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    demo_df, input_summary = prepare_demo_data(input_path)

    demo_years = input_summary["years"]
    if not demo_years:
        raise ValueError("Demo 輸入不包含任何有效年度。")

    min_rows_in_year = min(item["n_rows"] for item in input_summary["rows_by_year"])
    if top_k > min_rows_in_year:
        raise ValueError(
            f"Top-K={top_k} 對這份輸入資料過大。最小年度切片只有 "
            f"{min_rows_in_year} 筆資料。"
        )

    has_return = bool(input_summary["has_return"])
    has_label = bool(input_summary["has_label"])

    all_predictions = []
    model_sources = []

    for year in demo_years:
        year_demo_df = demo_df[demo_df[YEAR_COL].astype(int) == int(year)].copy()
        pipeline, split_id, source = _resolve_demo_pipeline(
            train_df=train_df,
            model_name=model_name,
            year=year,
        )

        predicted_label = pipeline.predict(year_demo_df[FEATURE_COLUMNS].copy()).astype(int)
        score_label_1 = get_probability_for_label_one(
            pipeline,
            year_demo_df[FEATURE_COLUMNS].copy(),
        )

        pred_df = year_demo_df[
            [
                YEAR_COL,
                STOCK_ID_COL,
                STOCK_NAME_COL,
            ]
        ].copy()

        pred_df.insert(0, "split_id", split_id)
        pred_df.insert(1, "model_name", model_name)
        pred_df.insert(2, "train_years", source)
        pred_df.insert(3, "test_years", str(year))
        pred_df["predicted_label"] = predicted_label
        pred_df["score_label_1"] = score_label_1
        pred_df["actual_label"] = (
            year_demo_df[TARGET_CLASS].astype("Int64").values
            if has_label
            else np.nan
        )
        pred_df["actual_return"] = (
            year_demo_df[TARGET_RETURN].astype(float).values
            if has_return
            else np.nan
        )

        pred_df = pred_df[
            [
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
        ].reset_index(drop=True)

        all_predictions.append(pred_df)
        model_sources.append(
            {
                "year": int(year),
                "split_id": int(split_id),
                "source": source,
            }
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    selected_df = select_top_k_demo(
        predictions_df=predictions_df,
        top_k_list=[top_k],
        score_col="score_label_1",
    )

    if has_return:
        selected_with_weights_df, portfolio_returns_df = calculate_portfolio_returns(
            selected_df=selected_df,
            weight_methods=["equal"],
            score_col="score_label_1",
        )

        portfolio_metrics_df = summarize_strategy_metrics(
            returns_df=portfolio_returns_df,
            group_cols=["model_name", "top_k", "weight_method"],
        )
    else:
        selected_with_weights_df = selected_df.copy()
        portfolio_returns_df = pd.DataFrame()
        portfolio_metrics_df = pd.DataFrame()

    if has_label:
        label_eval_df = predictions_df.dropna(subset=["actual_label"]).copy()
        classification_metrics_df = (
            calculate_classification_metrics_by_model(label_eval_df)
            if not label_eval_df.empty
            else pd.DataFrame()
        )
    else:
        classification_metrics_df = pd.DataFrame()

    evaluation_summary = _build_evaluation_summary(
        predictions_df=predictions_df,
        selected_df=selected_df,
        portfolio_returns_df=portfolio_returns_df,
        portfolio_metrics_df=portfolio_metrics_df,
        classification_metrics_df=classification_metrics_df,
    )

    provisional_result = DemoExecutionResult(
        input_path=Path(input_path),
        model_name=model_name,
        top_k=int(top_k),
        prepared_df=demo_df,
        input_summary=input_summary,
        predictions_df=predictions_df,
        selected_df=selected_df,
        selected_with_weights_df=selected_with_weights_df,
        portfolio_returns_df=portfolio_returns_df,
        portfolio_metrics_df=portfolio_metrics_df,
        classification_metrics_df=classification_metrics_df,
        evaluation_summary=evaluation_summary,
        model_sources=model_sources,
        output_paths=_output_paths(),
        profile_text="",
    )

    provisional_result.profile_text = _generate_profile_text(provisional_result)

    if persist_outputs:
        _save_demo_outputs(provisional_result)

    return provisional_result


def run_demo(
    input_path: str | Path,
    model_name: str = "random_forest",
    top_k: int = 10,
) -> dict[str, Path]:
    print("========== Demo Runner ==========")
    print(f"[Info] Input: {input_path}")
    print(f"[Info] Model: {model_name}")
    print(f"[Info] Top-K: {top_k}")

    result = run_demo_analysis(
        input_path=input_path,
        model_name=model_name,
        top_k=top_k,
        persist_outputs=True,
    )

    print(result.profile_text)
    print("========== Demo Finished ==========")

    return result.output_paths
