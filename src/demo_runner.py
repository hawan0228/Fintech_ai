# src/demo_runner.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    TOP_K_LIST,
    DROP_YEARMONTH,
    DEMO_PREDICTIONS_PATH,
    DEMO_SELECTED_STOCKS_PATH,
    DEMO_PORTFOLIO_RETURNS_PATH,
    DEMO_METRICS_PATH,
    DEMO_PROFILE_PATH,
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
from src.preprocessing import standardize_column_names, create_year_column
from src.models import get_classification_pipeline
from src.prediction import get_probability_for_label_one
from src.portfolio import calculate_portfolio_returns
from src.metrics import summarize_strategy_metrics


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到輸入檔案：{path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype={STOCK_ID_COL: str})

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={STOCK_ID_COL: str})

    raise ValueError(f"Unsupported input format: {path.suffix}")


def prepare_demo_data(path: str | Path) -> pd.DataFrame:
    df = load_table(path)
    df = standardize_column_names(df)
    df = create_year_column(df)

    # 依規格移除 年月 = 200912（與正式訓練前處理 remove_200912 一致）。
    # 這批為特殊無效列（Return 多為 -100），若保留會破壞投組績效計算。
    if YEARMONTH_COL in df.columns:
        ym = pd.to_numeric(df[YEARMONTH_COL], errors="coerce")
        n_dropped = int((ym == DROP_YEARMONTH).sum())
        if n_dropped:
            df = df[ym != DROP_YEARMONTH].copy()
            print(f"[Info] Demo: 已移除 年月={DROP_YEARMONTH} 的資料 {n_dropped} 筆")

    if STOCK_ID_COL not in df.columns:
        raise ValueError(f"Demo data 缺少欄位：{STOCK_ID_COL}")

    if STOCK_NAME_COL not in df.columns:
        df[STOCK_NAME_COL] = df[STOCK_ID_COL].astype(str)

    df[STOCK_ID_COL] = (
        df[STOCK_ID_COL]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(4)
    )

    df[STOCK_NAME_COL] = df[STOCK_NAME_COL].astype(str)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_RETURN in df.columns:
        df[TARGET_RETURN] = pd.to_numeric(df[TARGET_RETURN], errors="coerce")

    if TARGET_CLASS in df.columns:
        df[TARGET_CLASS] = pd.to_numeric(df[TARGET_CLASS], errors="coerce")

    return df


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
            selected = group_df.head(top_k).copy()
            selected["top_k"] = int(top_k)
            records.append(selected)

    return pd.concat(records, ignore_index=True)


def run_demo(
    input_path: str | Path,
    model_name: str = "random_forest",
    top_k: int = 10,
) -> dict[str, Path]:
    print("========== Demo Runner ==========")
    print(f"[Info] Input: {input_path}")
    print(f"[Info] Model: {model_name}")
    print(f"[Info] Top-K: {top_k}")

    train_df = pd.read_csv(CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    demo_df = prepare_demo_data(input_path)

    demo_years = sorted(demo_df[YEAR_COL].dropna().astype(int).unique().tolist())

    if not demo_years:
        raise ValueError("Demo data 無法解析 year。請確認有 年月 欄位。")

    has_return = TARGET_RETURN in demo_df.columns and demo_df[TARGET_RETURN].notna().any()
    has_label = TARGET_CLASS in demo_df.columns and demo_df[TARGET_CLASS].notna().any()

    all_predictions = []

    for year in demo_years:
        year_demo_df = demo_df[demo_df[YEAR_COL].astype(int) == int(year)].copy()

        # Demo 標準邏輯：只使用 demo year 以前的歷史訓練資料。
        year_train_df = train_df[train_df[YEAR_COL].astype(int) < int(year)].copy()

        if year_train_df.empty:
            print(
                f"[Warning] 沒有 year < {year} 的 training data，改用全部 cleaned training data。"
            )
            year_train_df = train_df.copy()

        X_train = year_train_df[FEATURE_COLUMNS].copy()
        y_train = year_train_df[TARGET_CLASS].astype(int).copy()

        X_test = year_demo_df[FEATURE_COLUMNS].copy()

        pipeline = get_classification_pipeline(model_name)
        pipeline.fit(X_train, y_train)

        predicted_label = pipeline.predict(X_test).astype(int)
        score_label_1 = get_probability_for_label_one(pipeline, X_test)

        pred_df = year_demo_df[
            [
                YEAR_COL,
                STOCK_ID_COL,
                STOCK_NAME_COL,
            ]
        ].copy()

        pred_df.insert(0, "split_id", 0)
        pred_df.insert(1, "model_name", model_name)
        pred_df.insert(2, "train_years", f"historical_before_{year}")
        pred_df.insert(3, "test_years", str(year))

        pred_df["predicted_label"] = predicted_label
        pred_df["score_label_1"] = score_label_1

        if has_label:
            pred_df["actual_label"] = year_demo_df[TARGET_CLASS].astype(int).values
        else:
            pred_df["actual_label"] = np.nan

        if has_return:
            pred_df["actual_return"] = year_demo_df[TARGET_RETURN].astype(float).values
        else:
            pred_df["actual_return"] = np.nan

        all_predictions.append(pred_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    selected_df = select_top_k_demo(
        predictions_df=predictions_df,
        top_k_list=[top_k],
        score_col="score_label_1",
    )

    output_paths = {
        "predictions": Path(DEMO_PREDICTIONS_PATH),
        "selected_stocks": Path(DEMO_SELECTED_STOCKS_PATH),
        "portfolio_returns": Path(DEMO_PORTFOLIO_RETURNS_PATH),
        "metrics": Path(DEMO_METRICS_PATH),
        "profile": Path(DEMO_PROFILE_PATH),
    }

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(DEMO_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

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

        selected_with_weights_df.to_csv(DEMO_SELECTED_STOCKS_PATH, index=False, encoding="utf-8-sig")
        portfolio_returns_df.to_csv(DEMO_PORTFOLIO_RETURNS_PATH, index=False, encoding="utf-8-sig")
        portfolio_metrics_df.to_csv(DEMO_METRICS_PATH, index=False, encoding="utf-8-sig")

    else:
        selected_df.to_csv(DEMO_SELECTED_STOCKS_PATH, index=False, encoding="utf-8-sig")

        pd.DataFrame().to_csv(DEMO_PORTFOLIO_RETURNS_PATH, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(DEMO_METRICS_PATH, index=False, encoding="utf-8-sig")

    profile_lines = []
    profile_lines.append("========== Demo Profile ==========")
    profile_lines.append(f"Input file: {input_path}")
    profile_lines.append(f"Model: {model_name}")
    profile_lines.append(f"Top-K: {top_k}")
    profile_lines.append(f"Demo years: {demo_years}")
    profile_lines.append(f"Has Return: {has_return}")
    profile_lines.append(f"Has Label: {has_label}")
    profile_lines.append(f"Prediction rows: {len(predictions_df)}")
    profile_lines.append(f"Selected rows: {len(selected_df)}")
    profile_lines.append("")
    profile_lines.append("Output files:")
    for key, path in output_paths.items():
        profile_lines.append(f"- {key}: {path}")
    profile_lines.append("========== End Demo Profile ==========")

    profile_text = "\n".join(profile_lines)

    with open(DEMO_PROFILE_PATH, "w", encoding="utf-8") as f:
        f.write(profile_text)

    print(profile_text)
    print("========== Demo Finished ==========")

    return output_paths