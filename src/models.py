from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.config import (
    RANDOM_SEED,

    DT_CRITERION,
    DT_MAX_DEPTH,
    DT_MIN_SAMPLES_LEAF,

    LR_MAX_ITER,
    LR_CLASS_WEIGHT,

    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_MAX_FEATURES,
    RF_CLASS_WEIGHT,

    GB_N_ESTIMATORS,
    GB_LEARNING_RATE,
    GB_MAX_DEPTH,
    GB_MIN_SAMPLES_LEAF,
)


def build_decision_tree_pipeline() -> Pipeline:
    """
    Task 1: ID3-like entropy-based Decision Tree.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    criterion=DT_CRITERION,
                    max_depth=DT_MAX_DEPTH,
                    min_samples_leaf=DT_MIN_SAMPLES_LEAF,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Task 2 baseline: Logistic Regression.

    Logistic Regression 是線性模型，對特徵尺度敏感，
    因此需要 StandardScaler。
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=LR_MAX_ITER,
                    class_weight=LR_CLASS_WEIGHT,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """
    Task 2 main model: Random Forest.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    max_depth=RF_MAX_DEPTH,
                    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                    max_features=RF_MAX_FEATURES,
                    class_weight=RF_CLASS_WEIGHT,
                    random_state=RANDOM_SEED,
                    # Keep the project runnable on restricted Windows environments.
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_gradient_boosting_pipeline() -> Pipeline:
    """
    Task 2 extension: Gradient Boosting.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=GB_N_ESTIMATORS,
                    learning_rate=GB_LEARNING_RATE,
                    max_depth=GB_MAX_DEPTH,
                    min_samples_leaf=GB_MIN_SAMPLES_LEAF,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def get_classification_pipeline(model_name: str) -> Pipeline:
    """
    根據 model_name 回傳對應分類模型 pipeline。
    """
    if model_name == "decision_tree_entropy":
        return build_decision_tree_pipeline()

    if model_name == "logistic_regression":
        return build_logistic_regression_pipeline()

    if model_name == "random_forest":
        return build_random_forest_pipeline()

    if model_name == "gradient_boosting":
        return build_gradient_boosting_pipeline()

    raise ValueError(f"Unknown classification model_name: {model_name}")
