# src/svr_ga.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    RANDOM_SEED,
    SVR_GA_POPULATION_SIZE,
    SVR_GA_N_GENERATIONS,
    SVR_GA_MUTATION_RATE,
    SVR_GA_CROSSOVER_RATE,
    SVR_GA_TOURNAMENT_SIZE,
    SVR_GA_PARAM_BOUNDS,
)


@dataclass
class SVRGABestResult:
    C: float
    gamma: float
    epsilon: float
    validation_rmse: float
    validation_mae: float
    validation_r2: float


def build_svr_pipeline(
    C: float,
    gamma: float,
    epsilon: float,
) -> Pipeline:
    """
    建立 SVR regression pipeline。

    注意：
    - imputer 只會在 pipeline.fit(X_train, y_train) 時 fit training data。
    - scaler 也只會在 training data fit。
    - testing data 僅 transform，不會參與 fit。
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVR(
                    kernel="rbf",
                    C=float(C),
                    gamma=float(gamma),
                    epsilon=float(epsilon),
                ),
            ),
        ]
    )


def decode_chromosome(chromosome: np.ndarray) -> dict[str, float]:
    """
    chromosome = [log10_C, log10_gamma, log10_epsilon]
    """
    log10_C = float(chromosome[0])
    log10_gamma = float(chromosome[1])
    log10_epsilon = float(chromosome[2])

    return {
        "C": 10.0 ** log10_C,
        "gamma": 10.0 ** log10_gamma,
        "epsilon": 10.0 ** log10_epsilon,
    }


def calculate_regression_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    回歸指標。
    Return 的單位是百分比，因此 MAE / RMSE 也是百分點。
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def evaluate_chromosome(
    chromosome: np.ndarray,
    X_train,
    y_train,
    X_valid,
    y_valid,
) -> dict[str, float]:
    """
    評估單一 chromosome 的 validation performance。
    fitness 目標是最小化 RMSE。
    """
    params = decode_chromosome(chromosome)

    try:
        pipeline = build_svr_pipeline(
            C=params["C"],
            gamma=params["gamma"],
            epsilon=params["epsilon"],
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_valid)

        scores = calculate_regression_scores(y_valid, y_pred)

        return {
            **params,
            "validation_mae": scores["mae"],
            "validation_rmse": scores["rmse"],
            "validation_r2": scores["r2"],
            "is_valid": 1,
        }

    except Exception:
        return {
            **params,
            "validation_mae": np.inf,
            "validation_rmse": np.inf,
            "validation_r2": np.nan,
            "is_valid": 0,
        }


def initialize_population(
    rng: np.random.Generator,
    population_size: int,
) -> np.ndarray:
    bounds = SVR_GA_PARAM_BOUNDS

    low = np.array(
        [
            bounds["log10_C"][0],
            bounds["log10_gamma"][0],
            bounds["log10_epsilon"][0],
        ],
        dtype=float,
    )

    high = np.array(
        [
            bounds["log10_C"][1],
            bounds["log10_gamma"][1],
            bounds["log10_epsilon"][1],
        ],
        dtype=float,
    )

    population = rng.uniform(
        low=low,
        high=high,
        size=(population_size, 3),
    )

    return population


def clip_population(population: np.ndarray) -> np.ndarray:
    bounds = SVR_GA_PARAM_BOUNDS

    low = np.array(
        [
            bounds["log10_C"][0],
            bounds["log10_gamma"][0],
            bounds["log10_epsilon"][0],
        ],
        dtype=float,
    )

    high = np.array(
        [
            bounds["log10_C"][1],
            bounds["log10_gamma"][1],
            bounds["log10_epsilon"][1],
        ],
        dtype=float,
    )

    return np.clip(population, low, high)


def tournament_select(
    population: np.ndarray,
    fitness_values: np.ndarray,
    rng: np.random.Generator,
    tournament_size: int,
) -> np.ndarray:
    """
    fitness_values 越小越好，這裡使用 RMSE。
    """
    candidate_indices = rng.choice(
        len(population),
        size=tournament_size,
        replace=False,
    )

    best_idx = candidate_indices[np.argmin(fitness_values[candidate_indices])]

    return population[best_idx].copy()


def crossover(
    parent_1: np.ndarray,
    parent_2: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Arithmetic crossover。
    """
    alpha = rng.uniform(0.0, 1.0)
    child = alpha * parent_1 + (1.0 - alpha) * parent_2

    return child


def mutate(
    chromosome: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float,
) -> np.ndarray:
    child = chromosome.copy()

    for gene_idx in range(child.shape[0]):
        if rng.uniform(0.0, 1.0) < mutation_rate:
            child[gene_idx] += rng.normal(loc=0.0, scale=0.35)

    child = clip_population(child.reshape(1, -1))[0]

    return child


def run_svr_ga_search(
    X_train,
    y_train,
    X_valid,
    y_valid,
    split_id: int,
    random_seed: int | None = None,
) -> tuple[SVRGABestResult, pd.DataFrame]:
    """
    在 outer training data 內部執行 GA search。

    注意：
    - X_valid / y_valid 必須來自 training years 內部切出的 validation set。
    - outer testing year 不可參與此 search。
    """
    seed = RANDOM_SEED + split_id if random_seed is None else random_seed
    rng = np.random.default_rng(seed)

    population = initialize_population(
        rng=rng,
        population_size=SVR_GA_POPULATION_SIZE,
    )

    search_records = []

    best_record = None
    best_chromosome = None

    for generation in range(1, SVR_GA_N_GENERATIONS + 1):
        generation_records = []
        fitness_values = []

        for individual_id, chromosome in enumerate(population, start=1):
            eval_record = evaluate_chromosome(
                chromosome=chromosome,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            params = decode_chromosome(chromosome)

            record = {
                "split_id": split_id,
                "generation": generation,
                "individual_id": individual_id,
                "log10_C": float(chromosome[0]),
                "log10_gamma": float(chromosome[1]),
                "log10_epsilon": float(chromosome[2]),
                "C": params["C"],
                "gamma": params["gamma"],
                "epsilon": params["epsilon"],
                "validation_mae": eval_record["validation_mae"],
                "validation_rmse": eval_record["validation_rmse"],
                "validation_r2": eval_record["validation_r2"],
                "is_valid": eval_record["is_valid"],
            }

            generation_records.append(record)
            fitness_values.append(eval_record["validation_rmse"])

            if best_record is None or eval_record["validation_rmse"] < best_record["validation_rmse"]:
                best_record = record.copy()
                best_chromosome = chromosome.copy()

        search_records.extend(generation_records)

        fitness_values = np.array(fitness_values, dtype=float)

        # Elitism: 保留當代最佳個體
        elite_idx = int(np.argmin(fitness_values))
        new_population = [population[elite_idx].copy()]

        while len(new_population) < SVR_GA_POPULATION_SIZE:
            parent_1 = tournament_select(
                population=population,
                fitness_values=fitness_values,
                rng=rng,
                tournament_size=SVR_GA_TOURNAMENT_SIZE,
            )

            parent_2 = tournament_select(
                population=population,
                fitness_values=fitness_values,
                rng=rng,
                tournament_size=SVR_GA_TOURNAMENT_SIZE,
            )

            if rng.uniform(0.0, 1.0) < SVR_GA_CROSSOVER_RATE:
                child = crossover(parent_1, parent_2, rng)
            else:
                child = parent_1.copy()

            child = mutate(
                chromosome=child,
                rng=rng,
                mutation_rate=SVR_GA_MUTATION_RATE,
            )

            new_population.append(child)

        population = np.vstack(new_population)
        population = clip_population(population)

        print(
            f"[Info] SVR-GA split {split_id}, generation {generation}: "
            f"best_rmse={best_record['validation_rmse']:.6f}, "
            f"C={best_record['C']:.6f}, "
            f"gamma={best_record['gamma']:.6f}, "
            f"epsilon={best_record['epsilon']:.6f}"
        )

    if best_record is None or best_chromosome is None:
        raise RuntimeError(f"SVR-GA split {split_id} failed to find valid parameters.")

    best_result = SVRGABestResult(
        C=float(best_record["C"]),
        gamma=float(best_record["gamma"]),
        epsilon=float(best_record["epsilon"]),
        validation_rmse=float(best_record["validation_rmse"]),
        validation_mae=float(best_record["validation_mae"]),
        validation_r2=float(best_record["validation_r2"]),
    )

    search_log_df = pd.DataFrame(search_records)

    return best_result, search_log_df