import hashlib

import numpy as np
import pytest

from app import app, prediction_service
from metaheuristic_service import ALGORITHMS, MODEL_SPACES, optimize_continuous


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_eight_real_algorithm_implementations_run_on_objective():
    assert set(ALGORITHMS) == {"PSO", "SA", "AFSA", "GA", "DE", "GWO", "WOA", "ACO"}

    def objective(vector):
        return -float(np.sum((vector - 0.25) ** 2))

    for algorithm in ALGORITHMS:
        vector, score, history = optimize_continuous(
            algorithm, objective, dimensions=3, iterations=2, population_size=6, seed=42
        )
        assert vector.shape == (3,)
        assert np.isfinite(score)
        assert np.all((0 <= vector) & (vector <= 1))
        assert len(history) == 3
        assert history == sorted(history)


def test_options_api_lists_models_algorithms_targets_and_budgets(client):
    response = client.get("/api/metaheuristic/options")
    payload = response.get_json()
    assert response.status_code == 200
    assert len(payload["algorithms"]) == 8
    assert "PSO" in payload["algorithms"]
    assert set(payload["models"]) == set(MODEL_SPACES)
    assert "Preference" in payload["targets"]
    assert set(payload["budgets"]) == {"quick", "standard", "thorough"}


def test_real_quick_optimization_returns_measured_result_without_mutating_production(client):
    before = hashlib.sha256(prediction_service.model_path.read_bytes()).hexdigest()
    response = client.post(
        "/api/metaheuristic/optimize",
        json={
            "target": "Milky",
            "model": "SVR_RBF",
            "algorithm": "PSO",
            "budget": "quick",
            "seed": 42,
        },
    )
    payload = response.get_json()
    after = hashlib.sha256(prediction_service.model_path.read_bytes()).hexdigest()

    assert response.status_code == 200, payload
    assert payload["model"]["key"] == "SVR_RBF"
    assert payload["algorithm"]["key"] == "PSO"
    assert payload["evaluations"] >= 7
    assert np.isfinite(payload["best_metrics"]["r2"])
    assert payload["r2_improvement"] >= 0
    assert payload["production_model_changed"] is False
    assert before == after == prediction_service.manifest["artifact_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [("model", "bad"), ("algorithm", "bad"), ("target", "bad"), ("budget", "bad")],
)
def test_invalid_optimization_selection_returns_400(client, field, value):
    payload = {"target": "Milky", "model": "SVR_RBF", "algorithm": "PSO", "budget": "quick"}
    payload[field] = value
    response = client.post("/api/metaheuristic/optimize", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_algorithms_page_has_cascading_selectors_and_all_eight_options(client):
    response = client.get("/algorithms")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'id="optimizationModel"' in html
    assert 'id="optimizationAlgorithm"' in html
    assert 'id="optimizationTarget"' in html
    for algorithm in ALGORITHMS:
        assert f'value="{algorithm}"' in html
