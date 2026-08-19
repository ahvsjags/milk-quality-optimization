import hashlib
import json
from pathlib import Path

import pytest

from app import app, prediction_service
from model_service import OOD_WARNING


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_frozen_artifact_matches_manifest():
    model_path = Path(prediction_service.model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert prediction_service.manifest["status"] == "FROZEN"
    assert digest == prediction_service.manifest["artifact_sha256"]
    assert prediction_service.model_version == "svr-rbf-selectkbest30-v1"


def test_all_frozen_pipelines_and_scales_are_correct():
    sensory = {"Milky", "Fatty", "Cooked", "Oxidized", "Sweet", "Fresh"}
    bundle = prediction_service.bundle
    for target, pipeline in bundle["models"].items():
        assert list(pipeline.named_steps) == ["scaler", "selector", "svr"]
        assert pipeline.named_steps["selector"].k == 30
        assert pipeline.named_steps["svr"].kernel == "rbf"
        expected_scale = [0.0, 5.0] if target in sensory else [0.0, 10.0]
        assert bundle["target_config"][target]["range"] == expected_scale


def test_normal_median_input_predicts_without_warning(client):
    response = client.post(
        "/api/predict", json={"features": prediction_service.default_features()}
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["out_of_domain"] is False
    assert payload["warning"] is None
    assert len(payload["predictions"]) == 7
    for prediction in payload["predictions"]:
        low, high = prediction["scale"]
        assert low <= prediction["score"] <= high


def test_hexanal_1000x_is_stable_and_warns(client):
    features = prediction_service.default_features()
    hexanal = next(item for item in prediction_service.feature_schema() if item["name"] == "Hexanal")
    features["Hexanal"] = hexanal["maximum"] * 1000
    response = client.post("/api/predict", json={"features": features})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["out_of_domain"] is True
    assert payload["warning"] == OOD_WARNING
    assert any(item["feature"] == "Hexanal" for item in payload["domain"]["violations"])
    for prediction in payload["predictions"]:
        low, high = prediction["scale"]
        assert low <= prediction["score"] <= high


def test_legacy_flat_payload_cannot_bypass_ood(client):
    features = prediction_service.default_features()
    hexanal = next(item for item in prediction_service.feature_schema() if item["name"] == "Hexanal")
    features["Hexanal"] = hexanal["maximum"] + 1
    response = client.post("/api/predict", json=features)
    assert response.status_code == 200
    assert response.get_json()["out_of_domain"] is True


def test_non_milk_zero_profile_is_flagged(client):
    features = {name: 0 for name in prediction_service.feature_names}
    response = client.post("/api/predict", json={"features": features})
    assert response.status_code == 200
    assert response.get_json()["out_of_domain"] is True


def test_near_float_limit_does_not_overflow_or_crash(client):
    features = prediction_service.default_features()
    features["Hexanal"] = 1e308
    response = client.post("/api/predict", json={"features": features})
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["out_of_domain"] is True
    assert payload["warning"] == OOD_WARNING
    assert len(payload["predictions"]) == 7


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"features": {}},
        {"features": {**prediction_service.default_features(), "Hexanal": "not-a-number"}},
        {"features": {**prediction_service.default_features(), "Hexanal": -1}},
        {"features": {**prediction_service.default_features(), "unknown": 1}},
    ],
)
def test_invalid_input_returns_400(client, payload):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_prediction_page_displays_correct_scales_and_warning_copy(client):
    response = client.get("/predict")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "六大感官属性：0–5" in html
    assert "喜好度：0–10" in html
    assert OOD_WARNING in html
    assert "ExtraTrees" not in html
    assert 'id="oodDialog"' in html
    assert "form.requestSubmit()" in html
    assert response.headers["Cache-Control"].startswith("no-store")


def test_model_info_records_optimizer_assignments(client):
    response = client.get("/api/model_info")
    targets = response.get_json()["targets"]
    assert targets["Milky"]["optimizer"] == "PSO"
    assert targets["Oxidized"]["optimizer"] == "PSO"
    assert targets["Fresh"]["optimizer"] == "PSO"
    assert targets["Cooked"]["optimizer"] == "SA"
    assert targets["Fatty"]["optimizer"] == "AFSA"
    assert targets["Sweet"]["optimizer"] == "AFSA"


def test_visualization_only_uses_deployed_sensory_and_preference_targets(client):
    response = client.get("/api/convergence_data")
    data = response.get_json()
    assert response.status_code == 200
    assert {entry["target"] for entry in data} == {
        "Milky", "Fatty", "Cooked", "Oxidized", "Sweet", "Fresh", "Preference"
    }
    for entry in data:
        assert 0.0 <= entry["scores"][-1] <= 1.0
        assert entry["execution_time"] > 0


def test_visualization_page_excludes_legacy_nutritional_target_controls(client):
    response = client.get("/visualization")
    html = response.get_data(as_text=True)
    assert 'value="Protein"' not in html
    assert 'value="Fat"' not in html
    assert 'value="Carbohydrate"' not in html
    assert 'value="Milky"' in html
    assert 'value="Preference"' in html
