from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.tree import export_text

from src.config import DT_MODEL_NAME
from src.schema import (
    FEATURE_COLUMNS,
    STOCK_ID_COL,
    STOCK_NAME_COL,
    YEAR_COL,
    TARGET_CLASS,
    TARGET_RETURN,
)
from src.models import (
    build_decision_tree_pipeline,
    get_classification_pipeline,
)


def parse_years(years_text: str) -> list[int]:
    """
    將 split 檔案中的 '1997,1998,1999' 轉成 [1997, 1998, 1999]。
    """
    return [int(x) for x in str(years_text).split(",") if str(x).strip()]


def prepare_xy(
    df: pd.DataFrame,
    years: list[int],
):
    """
    依指定年份建立 X 與 y。
    """
    subset = df[df[YEAR_COL].isin(years)].copy()

    X = subset[FEATURE_COLUMNS].copy()
    y = subset[TARGET_CLASS].astype(int).copy()

    return subset, X, y


def get_probability_for_label_one(pipeline, X_test: pd.DataFrame):
    """
    取得 P(label = 1)。

    注意：
    sklearn 的 classes_ 順序不應直接假設，因此必須用 classes_ 找出 label=1 的位置。
    """
    model = pipeline.named_steps["model"]

    if not hasattr(model, "classes_"):
        raise ValueError("模型尚未 fit，找不到 classes_。")

    classes = list(model.classes_)

    if 1 not in classes:
        # 理論上本資料不會發生，因為每個 training split 都有 label=1。
        # 保守處理：若 training data 沒有 label=1，則所有機率設為 0。
        return [0.0] * len(X_test)

    label_one_index = classes.index(1)
    probabilities = pipeline.predict_proba(X_test)[:, label_one_index]

    return probabilities


def train_decision_tree_one_split(
    df: pd.DataFrame,
    split_row: pd.Series,
    saved_model_dir: str | Path,
    rules_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對單一 temporal split 訓練 Decision Tree 並預測 testing data。

    Returns
    -------
    prediction_df : pd.DataFrame
        該 split 的 testing prediction results。
    feature_importance_df : pd.DataFrame
        該 split 的 feature importance。
    """
    split_id = int(split_row["split_id"])
    train_years = parse_years(split_row["train_years"])
    test_years = parse_years(split_row["test_years"])

    train_df, X_train, y_train = prepare_xy(df, train_years)
    test_df, X_test, y_test = prepare_xy(df, test_years)

    pipeline = build_decision_tree_pipeline()

    # 重要：pipeline.fit 只使用 training data。
    # SimpleImputer 也只會在 X_train 上 fit，避免 data leakage。
    pipeline.fit(X_train, y_train)

    predicted_label = pipeline.predict(X_test).astype(int)
    score_label_1 = get_probability_for_label_one(pipeline, X_test)

    prediction_df = test_df[
        [
            YEAR_COL,
            STOCK_ID_COL,
            STOCK_NAME_COL,
            TARGET_CLASS,
            TARGET_RETURN,
        ]
    ].copy()

    prediction_df.insert(0, "split_id", split_id)
    prediction_df.insert(1, "model_name", DT_MODEL_NAME)
    prediction_df.insert(2, "train_years", ",".join(map(str, train_years)))
    prediction_df.insert(3, "test_years", ",".join(map(str, test_years)))

    prediction_df = prediction_df.rename(
        columns={
            TARGET_CLASS: "actual_label",
            TARGET_RETURN: "actual_return",
        }
    )

    prediction_df["predicted_label"] = predicted_label
    prediction_df["score_label_1"] = score_label_1

    # 排序欄位，方便後續 Top-K 選股與檢查
    prediction_df = prediction_df[
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

    # Feature importance
    tree_model = pipeline.named_steps["model"]

    feature_importance_df = pd.DataFrame(
        {
            "split_id": split_id,
            "model_name": DT_MODEL_NAME,
            "feature": FEATURE_COLUMNS,
            "importance": tree_model.feature_importances_,
        }
    )

    feature_importance_df = feature_importance_df.sort_values(
        ["split_id", "importance"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # 儲存模型
    saved_model_dir = Path(saved_model_dir)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model_path = saved_model_dir / f"decision_tree_split_{split_id:02d}.joblib"
    joblib.dump(pipeline, model_path)

    # 儲存規則文字
    rules_dir = Path(rules_dir)
    rules_dir.mkdir(parents=True, exist_ok=True)

    rules_text = export_text(
        tree_model,
        feature_names=FEATURE_COLUMNS,
        decimals=4,
    )

    rules_path = rules_dir / f"decision_tree_rules_split_{split_id:02d}.txt"

    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(f"Decision Tree Rules - Split {split_id}\n")
        f.write(f"Train years: {train_years}\n")
        f.write(f"Test years: {test_years}\n")
        f.write("\n")
        f.write(rules_text)

    return prediction_df, feature_importance_df


def train_decision_tree_all_splits(
    df: pd.DataFrame,
    splits_df: pd.DataFrame,
    saved_model_dir: str | Path,
    rules_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對所有 temporal splits 訓練 Decision Tree 並合併結果。
    """
    all_predictions = []
    all_feature_importance = []

    for _, split_row in splits_df.iterrows():
        split_id = int(split_row["split_id"])
        train_years = split_row["train_years"]
        test_years = split_row["test_years"]

        print(
            f"[Info] Training Decision Tree split {split_id}: "
            f"train={train_years}, test={test_years}"
        )

        pred_df, fi_df = train_decision_tree_one_split(
            df=df,
            split_row=split_row,
            saved_model_dir=saved_model_dir,
            rules_dir=rules_dir,
        )

        all_predictions.append(pred_df)
        all_feature_importance.append(fi_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    feature_importance_df = pd.concat(all_feature_importance, ignore_index=True)

    return predictions_df, feature_importance_df

def extract_feature_importance_from_pipeline(
    pipeline,
    model_name: str,
    split_id: int,
) -> pd.DataFrame:
    """
    統一抽取不同分類模型的 feature importance。

    Random Forest / Gradient Boosting:
        使用 feature_importances_

    Logistic Regression:
        使用 coefficient 與 abs(coefficient)
    """
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "split_id": split_id,
                "model_name": model_name,
                "feature": FEATURE_COLUMNS,
                "importance": model.feature_importances_,
                "signed_coefficient": np.nan,
                "importance_type": "tree_feature_importance",
            }
        )

    elif hasattr(model, "coef_"):
        coef = model.coef_[0]

        importance_df = pd.DataFrame(
            {
                "split_id": split_id,
                "model_name": model_name,
                "feature": FEATURE_COLUMNS,
                "importance": np.abs(coef),
                "signed_coefficient": coef,
                "importance_type": "abs_logistic_coefficient",
            }
        )

    else:
        importance_df = pd.DataFrame(
            {
                "split_id": split_id,
                "model_name": model_name,
                "feature": FEATURE_COLUMNS,
                "importance": np.nan,
                "signed_coefficient": np.nan,
                "importance_type": "not_available",
            }
        )

    importance_df = importance_df.sort_values(
        ["split_id", "model_name", "importance"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return importance_df


def train_classification_model_one_split(
    df: pd.DataFrame,
    split_row: pd.Series,
    model_name: str,
    saved_model_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對單一 temporal split 訓練指定分類模型並預測 testing data。
    """
    split_id = int(split_row["split_id"])
    train_years = parse_years(split_row["train_years"])
    test_years = parse_years(split_row["test_years"])

    train_df, X_train, y_train = prepare_xy(df, train_years)
    test_df, X_test, y_test = prepare_xy(df, test_years)

    pipeline = get_classification_pipeline(model_name)

    # 重要：fit 只能使用 training data。
    # pipeline 內部的 imputer / scaler 也只會在 X_train 上 fit。
    pipeline.fit(X_train, y_train)

    predicted_label = pipeline.predict(X_test).astype(int)
    score_label_1 = get_probability_for_label_one(pipeline, X_test)

    prediction_df = test_df[
        [
            YEAR_COL,
            STOCK_ID_COL,
            STOCK_NAME_COL,
            TARGET_CLASS,
            TARGET_RETURN,
        ]
    ].copy()

    prediction_df.insert(0, "split_id", split_id)
    prediction_df.insert(1, "model_name", model_name)
    prediction_df.insert(2, "train_years", ",".join(map(str, train_years)))
    prediction_df.insert(3, "test_years", ",".join(map(str, test_years)))

    prediction_df = prediction_df.rename(
        columns={
            TARGET_CLASS: "actual_label",
            TARGET_RETURN: "actual_return",
        }
    )

    prediction_df["predicted_label"] = predicted_label
    prediction_df["score_label_1"] = score_label_1

    prediction_df = prediction_df[
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

    feature_importance_df = extract_feature_importance_from_pipeline(
        pipeline=pipeline,
        model_name=model_name,
        split_id=split_id,
    )

    saved_model_dir = Path(saved_model_dir)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model_path = saved_model_dir / f"{model_name}_split_{split_id:02d}.joblib"
    joblib.dump(pipeline, model_path)

    return prediction_df, feature_importance_df


def train_classification_models_all_splits(
    df: pd.DataFrame,
    splits_df: pd.DataFrame,
    model_names: list[str],
    saved_model_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對多個分類模型與所有 temporal splits 進行訓練與預測。
    """
    all_predictions = []
    all_feature_importance = []

    for model_name in model_names:
        print("")
        print(f"========== Training model: {model_name} ==========")

        for _, split_row in splits_df.iterrows():
            split_id = int(split_row["split_id"])
            train_years = split_row["train_years"]
            test_years = split_row["test_years"]

            print(
                f"[Info] {model_name} split {split_id}: "
                f"train={train_years}, test={test_years}"
            )

            pred_df, fi_df = train_classification_model_one_split(
                df=df,
                split_row=split_row,
                model_name=model_name,
                saved_model_dir=saved_model_dir,
            )

            all_predictions.append(pred_df)
            all_feature_importance.append(fi_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    feature_importance_df = pd.concat(all_feature_importance, ignore_index=True)

    return predictions_df, feature_importance_df