"""Export the frozen sklearn bundle for the GitHub Pages browser demo.

The exported JSON contains only inference parameters: StandardScaler,
SelectKBest indices, SVR-RBF support vectors, output bounds, and the frozen
applicability-domain guard.  It never retrains or alters the production model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "models" / "svr_rbf_selectkbest30_v1.joblib"
OUTPUT_PATH = ROOT / "docs" / "assets" / "frozen-model-v1.json"

TARGET_I18N = {
    "Milky": {"zh": "奶香味", "en": "Milky", "optimizer": "PSO", "range": [0, 5]},
    "Fatty": {"zh": "脂香味", "en": "Fatty", "optimizer": "AFSA", "range": [0, 5]},
    "Cooked": {"zh": "蒸煮味", "en": "Cooked", "optimizer": "SA", "range": [0, 5]},
    "Oxidized": {"zh": "氧化味", "en": "Oxidized", "optimizer": "PSO", "range": [0, 5]},
    "Sweet": {"zh": "甜味", "en": "Sweet", "optimizer": "AFSA", "range": [0, 5]},
    "Fresh": {"zh": "青草味", "en": "Fresh", "optimizer": "PSO", "range": [0, 5]},
    "Preference": {"zh": "喜好度", "en": "Preference", "optimizer": "GridSearchCV", "range": [0, 10]},
}


def plain(value):
    """Turn NumPy scalars/arrays into JSON-compatible standard objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def export() -> Path:
    bundle = joblib.load(BUNDLE_PATH)
    models = bundle["models"]
    first_pipeline = next(iter(models.values()))
    scaler = first_pipeline.named_steps["scaler"]

    browser_models = {}
    for target, pipeline in models.items():
        selector = pipeline.named_steps["selector"]
        svr = pipeline.named_steps["svr"]
        browser_models[target] = {
            "selected_indices": selector.get_support(indices=True),
            "gamma": float(svr._gamma),
            "support_vectors": svr.support_vectors_,
            "dual_coef": svr.dual_coef_[0],
            "intercept": float(svr.intercept_[0]),
            "target": TARGET_I18N[target],
        }

    payload = {
        "schema_version": "browser-inference-v1",
        "model_version": bundle["model_version"],
        "artifact_sha256": hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest(),
        "feature_names": bundle["feature_names"],
        "scaler": {"mean": scaler.mean_, "scale": scaler.scale_},
        "domain_guard": bundle["domain_guard"],
        "models": browser_models,
        "output_policy": bundle["output_policy"],
        "frozen_notice": "Exported from the immutable SVR-RBF + SelectKBest_30 deployment bundle.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(plain(payload), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return OUTPUT_PATH


if __name__ == "__main__":
    print(export())
