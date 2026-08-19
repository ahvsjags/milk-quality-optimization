"""Bounded, real metaheuristic hyperparameter optimization for the web UI.

The service evaluates candidate parameters on a fixed validation split.  It is
an experimentation path only: it never writes to or replaces the frozen
production joblib artifact.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class OptimizationInputError(ValueError):
    pass


@dataclass(frozen=True)
class Budget:
    iterations: int
    population: int


BUDGETS = {
    "quick": Budget(iterations=3, population=6),
    "standard": Budget(iterations=6, population=8),
    "thorough": Budget(iterations=10, population=12),
}

ALGORITHMS = {
    "PSO": {"name": "粒子群优化", "english": "Particle Swarm Optimization"},
    "SA": {"name": "模拟退火", "english": "Simulated Annealing"},
    "AFSA": {"name": "人工鱼群算法", "english": "Artificial Fish Swarm Algorithm"},
    "GA": {"name": "遗传算法", "english": "Genetic Algorithm"},
    "DE": {"name": "差分进化", "english": "Differential Evolution"},
    "GWO": {"name": "灰狼优化", "english": "Grey Wolf Optimizer"},
    "WOA": {"name": "鲸鱼优化", "english": "Whale Optimization Algorithm"},
    "ACO": {"name": "蚁群优化", "english": "Continuous Ant Colony Optimization"},
}

MODEL_SPACES = {
    "SVR_RBF": {
        "name": "SVR（RBF核）",
        "parameters": [
            {"name": "C", "minimum": 0.1, "maximum": 100.0, "scale": "log"},
            {"name": "gamma", "minimum": 0.0001, "maximum": 1.0, "scale": "log"},
            {"name": "epsilon", "minimum": 0.001, "maximum": 0.5, "scale": "linear"},
        ],
    },
    "ExtraTrees": {
        "name": "Extra Trees",
        "parameters": [
            {"name": "n_estimators", "minimum": 40, "maximum": 160, "scale": "int"},
            {"name": "max_depth", "minimum": 3, "maximum": 24, "scale": "int"},
            {"name": "min_samples_split", "minimum": 2, "maximum": 12, "scale": "int"},
            {"name": "min_samples_leaf", "minimum": 1, "maximum": 6, "scale": "int"},
        ],
    },
    "RandomForest": {
        "name": "Random Forest",
        "parameters": [
            {"name": "n_estimators", "minimum": 40, "maximum": 160, "scale": "int"},
            {"name": "max_depth", "minimum": 3, "maximum": 24, "scale": "int"},
            {"name": "min_samples_split", "minimum": 2, "maximum": 12, "scale": "int"},
            {"name": "min_samples_leaf", "minimum": 1, "maximum": 6, "scale": "int"},
        ],
    },
    "GradientBoosting": {
        "name": "Gradient Boosting",
        "parameters": [
            {"name": "n_estimators", "minimum": 40, "maximum": 160, "scale": "int"},
            {"name": "learning_rate", "minimum": 0.01, "maximum": 0.3, "scale": "log"},
            {"name": "max_depth", "minimum": 2, "maximum": 8, "scale": "int"},
            {"name": "min_samples_split", "minimum": 2, "maximum": 12, "scale": "int"},
        ],
    },
}

TARGET_LABELS = {
    "Milky": "奶香味",
    "Fatty": "脂香味",
    "Cooked": "蒸煮味",
    "Oxidized": "氧化味",
    "Sweet": "甜味",
    "Fresh": "青草味",
    "Preference": "喜好度",
}


def _evaluate_population(objective: Callable[[np.ndarray], float], population: np.ndarray) -> np.ndarray:
    return np.asarray([objective(np.clip(candidate, 0.0, 1.0)) for candidate in population], dtype=float)


def _history_append(history: list[float], scores: np.ndarray) -> None:
    best = float(np.max(scores))
    history.append(max(best, history[-1]) if history else best)


def optimize_continuous(
    algorithm: str,
    objective: Callable[[np.ndarray], float],
    dimensions: int,
    iterations: int,
    population_size: int,
    seed: int,
) -> tuple[np.ndarray, float, list[float]]:
    """Maximize an objective over a unit hypercube using the selected algorithm."""
    if algorithm not in ALGORITHMS:
        raise OptimizationInputError(f"未知启发式算法：{algorithm}")
    rng = np.random.default_rng(seed)
    raw_objective = objective
    global_best_vector: np.ndarray | None = None
    global_best_score = -math.inf

    def tracked_objective(candidate: np.ndarray) -> float:
        nonlocal global_best_vector, global_best_score
        clipped = np.clip(candidate, 0.0, 1.0)
        score = float(raw_objective(clipped))
        if score > global_best_score:
            global_best_score = score
            global_best_vector = clipped.copy()
        return score

    objective = tracked_objective
    n = max(population_size, 6)
    population = rng.random((n, dimensions))
    scores = _evaluate_population(objective, population)
    history: list[float] = []
    _history_append(history, scores)

    if algorithm == "PSO":
        velocity = np.zeros_like(population)
        personal = population.copy()
        personal_scores = scores.copy()
        for _ in range(iterations):
            leader = personal[np.argmax(personal_scores)]
            velocity = 0.65 * velocity + 1.6 * rng.random(population.shape) * (personal - population) + 1.6 * rng.random(population.shape) * (leader - population)
            population = np.clip(population + velocity, 0, 1)
            scores = _evaluate_population(objective, population)
            improved = scores > personal_scores
            personal[improved], personal_scores[improved] = population[improved], scores[improved]
            _history_append(history, personal_scores)
        population, scores = personal, personal_scores

    elif algorithm == "SA":
        current = population[np.argmax(scores)].copy()
        current_score = float(np.max(scores))
        best, best_score = current.copy(), current_score
        for iteration in range(iterations):
            temperature = max(0.03, 1.0 - iteration / max(iterations, 1))
            candidates = np.clip(current + rng.normal(0, 0.25 * temperature, size=(n, dimensions)), 0, 1)
            candidate_scores = _evaluate_population(objective, candidates)
            for candidate, score in zip(candidates, candidate_scores):
                if score > current_score or rng.random() < math.exp(min(0.0, (score - current_score) / temperature)):
                    current, current_score = candidate.copy(), float(score)
                if current_score > best_score:
                    best, best_score = current.copy(), current_score
            history.append(max(history[-1], best_score))
        assert global_best_vector is not None
        return global_best_vector, global_best_score, history

    elif algorithm == "AFSA":
        for iteration in range(iterations):
            leader = population[np.argmax(scores)]
            visual = 0.35 * (1 - 0.6 * iteration / max(iterations, 1))
            candidates = np.clip(population + rng.random((n, 1)) * (leader - population) + rng.normal(0, visual, population.shape), 0, 1)
            candidate_scores = _evaluate_population(objective, candidates)
            improved = candidate_scores > scores
            population[improved], scores[improved] = candidates[improved], candidate_scores[improved]
            _history_append(history, scores)

    elif algorithm == "GA":
        for _ in range(iterations):
            elite = population[np.argsort(scores)[-2:]].copy()
            children = []
            while len(children) < n - 2:
                contenders = rng.integers(0, n, size=4)
                parent1 = population[contenders[:2][np.argmax(scores[contenders[:2]])]]
                parent2 = population[contenders[2:][np.argmax(scores[contenders[2:]])]]
                mask = rng.random(dimensions) < 0.5
                child = np.where(mask, parent1, parent2) + rng.normal(0, 0.08, dimensions)
                children.append(np.clip(child, 0, 1))
            population = np.vstack([elite, children])
            scores = _evaluate_population(objective, population)
            _history_append(history, scores)

    elif algorithm == "DE":
        for _ in range(iterations):
            next_population = population.copy()
            for index in range(n):
                pool = np.delete(np.arange(n), index)
                a, b, c = population[rng.choice(pool, 3, replace=False)]
                mutant = np.clip(a + 0.75 * (b - c), 0, 1)
                mask = rng.random(dimensions) < 0.7
                mask[rng.integers(dimensions)] = True
                trial = np.where(mask, mutant, population[index])
                score = objective(trial)
                if score > scores[index]:
                    next_population[index], scores[index] = trial, score
            population = next_population
            _history_append(history, scores)

    elif algorithm == "GWO":
        for iteration in range(iterations):
            leaders = population[np.argsort(scores)[-3:]]
            a = 2.0 * (1 - iteration / max(iterations, 1))
            proposals = []
            for wolf in population:
                positions = []
                for leader in leaders:
                    A = 2 * a * rng.random(dimensions) - a
                    C = 2 * rng.random(dimensions)
                    positions.append(leader - A * np.abs(C * leader - wolf))
                proposals.append(np.mean(positions, axis=0))
            population = np.clip(np.asarray(proposals), 0, 1)
            scores = _evaluate_population(objective, population)
            _history_append(history, scores)

    elif algorithm == "WOA":
        for iteration in range(iterations):
            leader = population[np.argmax(scores)].copy()
            a = 2.0 * (1 - iteration / max(iterations, 1))
            proposals = []
            for whale in population:
                if rng.random() < 0.5:
                    A = 2 * a * rng.random(dimensions) - a
                    C = 2 * rng.random(dimensions)
                    reference = leader if np.max(np.abs(A)) < 1 else population[rng.integers(n)]
                    proposal = reference - A * np.abs(C * reference - whale)
                else:
                    l = rng.uniform(-1, 1)
                    proposal = np.abs(leader - whale) * math.exp(l) * math.cos(2 * math.pi * l) + leader
                proposals.append(proposal)
            population = np.clip(np.asarray(proposals), 0, 1)
            scores = _evaluate_population(objective, population)
            _history_append(history, scores)

    elif algorithm == "ACO":
        for iteration in range(iterations):
            elite_count = max(2, n // 3)
            elite = population[np.argsort(scores)[-elite_count:]]
            weights = np.linspace(1.0, 2.0, elite_count)
            mean = np.average(elite, axis=0, weights=weights)
            sigma = np.maximum(np.std(elite, axis=0), 0.03) * (1 - 0.5 * iteration / max(iterations, 1))
            candidates = np.clip(rng.normal(mean, sigma, size=(n - elite_count, dimensions)), 0, 1)
            population = np.vstack([elite, candidates])
            scores = _evaluate_population(objective, population)
            _history_append(history, scores)

    assert global_best_vector is not None
    return global_best_vector, global_best_score, history


class MetaheuristicOptimizationService:
    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path)
        data = pd.read_csv(self.data_path, encoding="utf-8")
        self.targets = tuple(TARGET_LABELS)
        self.feature_names = tuple(column for column in data.columns if column not in self.targets)
        self.X = data[list(self.feature_names)].astype(float)
        self.y = {target: data[target].astype(float) for target in self.targets}

    def options(self) -> dict[str, Any]:
        return {
            "models": MODEL_SPACES,
            "algorithms": ALGORITHMS,
            "targets": {key: {"label": label, "scale": [0, 10] if key == "Preference" else [0, 5]} for key, label in TARGET_LABELS.items()},
            "budgets": {key: {"iterations": value.iterations, "population": value.population} for key, value in BUDGETS.items()},
            "production_policy": "实验结果不会自动替换冻结生产模型",
        }

    @staticmethod
    def _decode(model_key: str, vector: np.ndarray) -> dict[str, Any]:
        decoded = {}
        for value, spec in zip(vector, MODEL_SPACES[model_key]["parameters"]):
            low, high = float(spec["minimum"]), float(spec["maximum"])
            if spec["scale"] == "log":
                result = math.exp(math.log(low) + float(value) * (math.log(high) - math.log(low)))
            else:
                result = low + float(value) * (high - low)
            decoded[spec["name"]] = int(round(result)) if spec["scale"] == "int" else float(result)
        return decoded

    @staticmethod
    def _build_model(model_key: str, params: dict[str, Any]) -> Any:
        if model_key == "SVR_RBF":
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(f_regression, k=30)),
                ("model", SVR(kernel="rbf", cache_size=500, **params)),
            ])
        else:
            constructors = {
                "ExtraTrees": ExtraTreesRegressor,
                "RandomForest": RandomForestRegressor,
                "GradientBoosting": GradientBoostingRegressor,
            }
            estimator = Pipeline([
                ("selector", SelectKBest(f_regression, k=30)),
                ("model", constructors[model_key](random_state=42, n_jobs=1, **params) if model_key in {"ExtraTrees", "RandomForest"} else constructors[model_key](random_state=42, **params)),
            ])
        return estimator

    def optimize(self, model_key: str, algorithm: str, target: str, budget_key: str, seed: int = 42) -> dict[str, Any]:
        if model_key not in MODEL_SPACES:
            raise OptimizationInputError(f"未知模型：{model_key}")
        if algorithm not in ALGORITHMS:
            raise OptimizationInputError(f"未知启发式算法：{algorithm}")
        if target not in self.targets:
            raise OptimizationInputError(f"未知目标：{target}")
        if budget_key not in BUDGETS:
            raise OptimizationInputError(f"未知搜索预算：{budget_key}")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
            raise OptimizationInputError("seed 必须是 0 到 4294967295 的整数")

        X_train, X_valid, y_train, y_valid = train_test_split(
            self.X, self.y[target], test_size=0.30, random_state=42, shuffle=True
        )
        cache: dict[tuple[float, ...], tuple[float, dict[str, float], dict[str, Any]]] = {}

        def objective(vector: np.ndarray) -> float:
            key = tuple(np.round(vector, 8))
            if key not in cache:
                params = self._decode(model_key, vector)
                model = self._build_model(model_key, params)
                model.fit(X_train, y_train)
                prediction = model.predict(X_valid)
                metrics = {
                    "r2": float(r2_score(y_valid, prediction)),
                    "rmse": float(mean_squared_error(y_valid, prediction) ** 0.5),
                    "mae": float(mean_absolute_error(y_valid, prediction)),
                }
                cache[key] = (metrics["r2"], metrics, params)
            return cache[key][0]

        dimensions = len(MODEL_SPACES[model_key]["parameters"])
        midpoint = np.full(dimensions, 0.5)
        baseline_score = objective(midpoint)
        baseline_metrics = cache[tuple(np.round(midpoint, 8))][1]
        budget = BUDGETS[budget_key]
        started = time.perf_counter()
        best_vector, best_score, history = optimize_continuous(
            algorithm, objective, dimensions, budget.iterations, budget.population, seed
        )
        # The midpoint is the explicitly measured baseline candidate.  Never
        # report a worse candidate as "optimized" merely because a small search
        # budget did not rediscover it.
        if best_score < baseline_score:
            best_vector, best_score = midpoint.copy(), baseline_score
        history = [max(value, baseline_score) for value in history]
        # Ensure the returned candidate is present even if an implementation's
        # final bookkeeping selected a previously evaluated vector.
        objective(best_vector)
        best_key = tuple(np.round(best_vector, 8))
        _, best_metrics, best_params = cache[best_key]
        elapsed = time.perf_counter() - started

        return {
            "model": {"key": model_key, "name": MODEL_SPACES[model_key]["name"]},
            "algorithm": {"key": algorithm, **ALGORITHMS[algorithm]},
            "target": {"key": target, "label": TARGET_LABELS[target]},
            "budget": {"key": budget_key, "iterations": budget.iterations, "population": budget.population},
            "best_params": best_params,
            "best_metrics": {key: round(value, 6) for key, value in best_metrics.items()},
            "baseline_metrics": {key: round(value, 6) for key, value in baseline_metrics.items()},
            "r2_improvement": round(best_score - baseline_score, 6),
            "history": [round(value, 6) for value in history],
            "evaluations": len(cache),
            "elapsed_seconds": round(elapsed, 3),
            "validation": "fixed 70/30 holdout, random_state=42",
            "production_model_changed": False,
        }


_BROWSER_SERVICE = None

def run_optimization(request_json: str) -> str:
    global _BROWSER_SERVICE
    request = json.loads(request_json)
    if _BROWSER_SERVICE is None:
        _BROWSER_SERVICE = MetaheuristicOptimizationService('/home/pyodide/optimization-training-data.csv')
    result = _BROWSER_SERVICE.optimize(
        model_key=str(request['model']),
        algorithm=str(request['algorithm']),
        target=str(request['target']),
        budget_key=str(request.get('budget', 'quick')),
        seed=int(request.get('seed', 42)),
    )
    return json.dumps(result, ensure_ascii=False)
