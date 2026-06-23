# rerun_external_pipeline.py
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    EXTERNAL_CLEANED_DATA_PATH,
    EXTERNAL_RF_MODEL_NAME,
    EXTERNAL_RF_PREDICTIONS_PATH,
    EXTERNAL_RF_SELECTED_STOCKS_PATH,
    EXTERNAL_RF_PORTFOLIO_RETURNS_PATH,
    EXTERNAL_RF_PORTFOLIO_METRICS_PATH,
    EXTERNAL_RF_CLASSIFICATION_METRICS_PATH,
    EXTERNAL_PROFILE_PATH,
    TOP_K_LIST,
    SAVED_MODEL_DIR,
)

try:
    from src.config import EXTERNAL_TOP_K_LIST
except ImportError:
    EXTERNAL_TOP_K_LIST = TOP_K_LIST

from src.schema import (
    FEATURE_COLUMNS,
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEAR_COL,
    TARGET_CLASS,
    TARGET_RETURN,
)
from src.models import build_random_forest_pipeline
from src.prediction import get_probability_for_label_one
from src.portfolio import select_top_k_stocks, calculate_portfolio_returns
from src.metrics import (
    calculate_classification_metrics_by_model,
    summarize_strategy_metrics,
)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_years(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def make_external_splits(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df[YEAR_COL].dropna().astype(int).unique().tolist())

    if len(years) < 3:
        raise ValueError(
            "外部資料年份太少，至少需要 3 年以上才能做 expanding-window validation。"
        )

    records = []

    for idx in range(1, len(years)):
        train_years = years[:idx]
        test_years = [years[idx]]

        records.append(
            {
                "split_id": idx,
                "train_years": ",".join(map(str, train_years)),
                "test_years": ",".join(map(str, test_years)),
                "n_train_years": len(train_years),
                "n_test_years": 1,
            }
        )

    return pd.DataFrame(records)


def get_usable_feature_columns(train_df: pd.DataFrame) -> list[str]:
    """
    外部資料不一定能完整取得原始 16 個欄位。

    嚴謹做法：
    - 每個 split 只根據 training data 判斷哪些 feature 有至少一筆非缺失值。
    - 不用 testing data 決定 feature availability，避免隱性資料洩漏。
    """
    usable_features = []

    for col in FEATURE_COLUMNS:
        if col not in train_df.columns:
            continue

        non_missing_count = train_df[col].notna().sum()

        if non_missing_count >= 1:
            usable_features.append(col)

    if not usable_features:
        raise ValueError(
            "此 split 的 training data 中沒有任何可用 feature。"
            "請檢查 external_cleaned_dataset.csv。"
        )

    return usable_features


def resolve_valid_top_k_list(
    predictions_df: pd.DataFrame,
    requested_top_k_list: list[int],
) -> list[int]:
    """
    外部資料股票數可能少於主實驗 200 檔，因此不能硬跑 Top-20 / Top-30。

    這裡根據每個 testing year 的股票數，自動保留可執行的 Top-K。
    """
    counts = (
        predictions_df.groupby(["model_name", YEAR_COL])
        .size()
        .reset_index(name="n_stocks")
    )

    min_n_stocks = int(counts["n_stocks"].min())

    valid_top_k_list = [
        int(k) for k in requested_top_k_list
        if int(k) <= min_n_stocks
    ]

    skipped_top_k_list = [
        int(k) for k in requested_top_k_list
        if int(k) > min_n_stocks
    ]

    if not valid_top_k_list:
        valid_top_k_list = [min_n_stocks]

    print("")
    print("========== External Top-K Resolution ==========")
    print(f"[Info] Minimum stocks per testing year: {min_n_stocks}")
    print(f"[Info] Requested Top-K list: {requested_top_k_list}")
    print(f"[Info] Valid Top-K list: {valid_top_k_list}")

    if skipped_top_k_list:
        print(
            f"[Warning] Skipped Top-K values because external data only has "
            f"{min_n_stocks} stocks per year: {skipped_top_k_list}"
        )

    return valid_top_k_list


def generate_external_rerun_profile(
    df: pd.DataFrame,
    splits_df: pd.DataFrame,
    feature_records_df: pd.DataFrame,
    valid_top_k_list: list[int],
    portfolio_metrics_df: pd.DataFrame,
) -> str:
    lines = []

    lines.append("========== External Re-run Profile ==========")
    lines.append("")
    lines.append("Dataset summary:")
    lines.append(f"- Shape: {df.shape}")
    lines.append(f"- Years: {df[YEAR_COL].min()} - {df[YEAR_COL].max()}")
    lines.append("")
    lines.append("Rows by year:")
    lines.append(df.groupby(YEAR_COL).size().to_string())
    lines.append("")
    lines.append("Label distribution:")
    lines.append(df[TARGET_CLASS].value_counts().sort_index().to_string())
    lines.append("")
    lines.append("Feature missing ratio:")
    missing_ratio = df[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False)
    lines.append(missing_ratio.to_string())
    lines.append("")
    lines.append("Temporal splits:")
    lines.append(splits_df.to_string(index=False))
    lines.append("")
    lines.append("Usable features by split:")
    lines.append(feature_records_df.to_string(index=False))
    lines.append("")
    lines.append("Valid Top-K list:")
    lines.append(str(valid_top_k_list))
    lines.append("")
    lines.append("Portfolio metrics:")
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
    lines.append(portfolio_metrics_df[display_cols].to_string(index=False))
    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- External data may not fully match the 16 original financial attributes. "
        "Features with no observed values in a training split are excluded from that split."
    )
    lines.append(
        "- Top-K values larger than the number of available stocks per year are skipped."
    )
    lines.append(
        "- Realized Return is used only after stock selection to compute portfolio performance."
    )
    lines.append("")
    lines.append("========== End External Re-run Profile ==========")

    return "\n".join(lines)


def train_external_rf_one_split(
    df: pd.DataFrame,
    split_row: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    split_id = int(split_row["split_id"])
    train_years = parse_years(split_row["train_years"])
    test_years = parse_years(split_row["test_years"])

    train_df = df[df[YEAR_COL].isin(train_years)].copy()
    test_df = df[df[YEAR_COL].isin(test_years)].copy()

    if train_df.empty:
        raise ValueError(f"split {split_id} training data is empty.")

    if test_df.empty:
        raise ValueError(f"split {split_id} testing data is empty.")

    if train_df[TARGET_CLASS].nunique() < 2:
        raise ValueError(
            f"split {split_id} training labels 只有一種類別，無法訓練分類模型。"
        )

    usable_features = get_usable_feature_columns(train_df)

    print(
        f"[Info] External RF split {split_id}: "
        f"train={split_row['train_years']}, "
        f"test={split_row['test_years']}, "
        f"usable_features={len(usable_features)}"
    )

    if len(usable_features) < len(FEATURE_COLUMNS):
        missing_features = [
            col for col in FEATURE_COLUMNS
            if col not in usable_features
        ]
        print(f"[Warning] Split {split_id} skipped all-missing features: {missing_features}")

    X_train = train_df[usable_features].copy()
    y_train = train_df[TARGET_CLASS].astype(int).copy()

    X_test = test_df[usable_features].copy()

    pipeline = build_random_forest_pipeline()
    pipeline.fit(X_train, y_train)

    predicted_label = pipeline.predict(X_test).astype(int)
    score_label_1 = get_probability_for_label_one(pipeline, X_test)

    pred_df = test_df[
        [
            YEAR_COL,
            STOCK_ID_COL,
            STOCK_NAME_COL,
            TARGET_CLASS,
            TARGET_RETURN,
        ]
    ].copy()

    pred_df.insert(0, "split_id", split_id)
    pred_df.insert(1, "model_name", EXTERNAL_RF_MODEL_NAME)
    pred_df.insert(2, "train_years", ",".join(map(str, train_years)))
    pred_df.insert(3, "test_years", ",".join(map(str, test_years)))

    pred_df = pred_df.rename(
        columns={
            TARGET_CLASS: "actual_label",
            TARGET_RETURN: "actual_return",
        }
    )

    pred_df["predicted_label"] = predicted_label
    pred_df["score_label_1"] = score_label_1

    model_path = (
        Path(SAVED_MODEL_DIR)
        / f"{EXTERNAL_RF_MODEL_NAME}_split_{split_id:02d}.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    feature_record = {
        "split_id": split_id,
        "train_years": ",".join(map(str, train_years)),
        "test_years": ",".join(map(str, test_years)),
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_usable_features": len(usable_features),
        "usable_features": "|".join(usable_features),
        "saved_model_path": str(model_path),
    }

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

    return pred_df, feature_record


def main() -> None:
    print("========== Step 7: External Data Re-run ==========")

    df = pd.read_csv(EXTERNAL_CLEANED_DATA_PATH, dtype={STOCK_ID_COL: str})
    print(f"[Info] external cleaned data shape: {df.shape}")

    if df.empty:
        raise ValueError("external_cleaned_dataset.csv 是空的，請先檢查 crawler。")

    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df[TARGET_CLASS] = df[TARGET_CLASS].astype(int)
    df[TARGET_RETURN] = pd.to_numeric(df[TARGET_RETURN], errors="coerce")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    splits_df = make_external_splits(df)

    print("[Info] external splits:")
    print(splits_df.to_string(index=False))

    predictions = []
    feature_records = []

    for _, split_row in splits_df.iterrows():
        pred_df, feature_record = train_external_rf_one_split(
            df=df,
            split_row=split_row,
        )

        predictions.append(pred_df)
        feature_records.append(feature_record)

    predictions_df = pd.concat(predictions, ignore_index=True)
    feature_records_df = pd.DataFrame(feature_records)

    classification_metrics_df = calculate_classification_metrics_by_model(
        predictions_df
    )

    valid_top_k_list = resolve_valid_top_k_list(
        predictions_df=predictions_df,
        requested_top_k_list=EXTERNAL_TOP_K_LIST,
    )

    selected_df = select_top_k_stocks(
        predictions_df=predictions_df,
        top_k_list=valid_top_k_list,
        score_col="score_label_1",
    )

    selected_with_weights_df, portfolio_returns_df = calculate_portfolio_returns(
        selected_df=selected_df,
        weight_methods=["equal", "score"],
        score_col="score_label_1",
    )

    portfolio_metrics_df = summarize_strategy_metrics(
        returns_df=portfolio_returns_df,
        group_cols=["model_name", "top_k", "weight_method"],
    )

    save_dataframe(predictions_df, EXTERNAL_RF_PREDICTIONS_PATH)
    save_dataframe(classification_metrics_df, EXTERNAL_RF_CLASSIFICATION_METRICS_PATH)
    save_dataframe(selected_with_weights_df, EXTERNAL_RF_SELECTED_STOCKS_PATH)
    save_dataframe(portfolio_returns_df, EXTERNAL_RF_PORTFOLIO_RETURNS_PATH)
    save_dataframe(portfolio_metrics_df, EXTERNAL_RF_PORTFOLIO_METRICS_PATH)

    profile_text = generate_external_rerun_profile(
        df=df,
        splits_df=splits_df,
        feature_records_df=feature_records_df,
        valid_top_k_list=valid_top_k_list,
        portfolio_metrics_df=portfolio_metrics_df,
    )

    save_text(profile_text, EXTERNAL_PROFILE_PATH)

    print("")
    print(profile_text)
    print("")
    print("[Saved outputs]")
    print(f"- {EXTERNAL_RF_PREDICTIONS_PATH}")
    print(f"- {EXTERNAL_RF_CLASSIFICATION_METRICS_PATH}")
    print(f"- {EXTERNAL_RF_SELECTED_STOCKS_PATH}")
    print(f"- {EXTERNAL_RF_PORTFOLIO_RETURNS_PATH}")
    print(f"- {EXTERNAL_RF_PORTFOLIO_METRICS_PATH}")
    print(f"- {EXTERNAL_PROFILE_PATH}")
    print("")
    print("========== External Data Re-run Finished ==========")


if __name__ == "__main__":
    main()