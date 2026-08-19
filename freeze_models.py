"""Train once and freeze the production SVR-RBF prediction bundle.

This script is deliberately *not* called by the web application.  Run it only
when a new, explicitly versioned model release is approved.  The server loads
the generated joblib file read-only and verifies its SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "yc.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "svr_rbf_selectkbest30_v1.joblib"
MANIFEST_PATH = MODEL_DIR / "svr_rbf_selectkbest30_v1.manifest.json"

TARGET_CONFIG = {
    "Milky": {
        "label": "奶香味",
        "optimizer": "PSO",
        "range": [0.0, 5.0],
        "svr": {"C": 35.2, "epsilon": 0.041, "gamma": 0.095},
    },
    "Fatty": {
        "label": "脂香味",
        "optimizer": "AFSA",
        "range": [0.0, 5.0],
        "svr": {"C": 29.8, "epsilon": 0.047, "gamma": 0.083},
    },
    "Cooked": {
        "label": "蒸煮味",
        "optimizer": "SA",
        "range": [0.0, 5.0],
        "svr": {"C": 31.5, "epsilon": 0.043, "gamma": 0.088},
    },
    "Oxidized": {
        "label": "氧化味",
        "optimizer": "PSO",
        "range": [0.0, 5.0],
        "svr": {"C": 27.9, "epsilon": 0.049, "gamma": 0.081},
    },
    "Sweet": {
        "label": "甜味",
        "optimizer": "AFSA",
        "range": [0.0, 5.0],
        "svr": {"C": 33.6, "epsilon": 0.039, "gamma": 0.091},
    },
    "Fresh": {
        "label": "青草味",
        "optimizer": "PSO",
        "range": [0.0, 5.0],
        "svr": {"C": 38.4, "epsilon": 0.035, "gamma": 0.098},
    },
    # Preference uses the independently tuned best SVR-RBF parameters recorded
    # in 可视化/Hyperparameter_Tuning_Results.csv.
    "Preference": {
        "label": "喜好度",
        "optimizer": "GridSearchCV",
        "range": [0.0, 10.0],
        "svr": {"C": 5.0, "epsilon": 0.2, "gamma": 0.01},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline(params: dict[str, float]) -> Pipeline:
    """Return the exact frozen StandardScaler -> SelectKBest_30 -> SVR pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_regression, k=30)),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=params["C"],
                    epsilon=params["epsilon"],
                    gamma=params["gamma"],
                    degree=3,
                    coef0=0.0,
                    shrinking=True,
                    tol=0.001,
                    cache_size=1000,
                    verbose=False,
                    max_iter=-1,
                ),
            ),
        ]
    )


def build_domain_guard(X: pd.DataFrame) -> dict:
    """Create deterministic univariate and multivariate applicability limits."""
    values = X.to_numpy(dtype=float)
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    covariance = LedoitWolf().fit(scaled)
    delta = scaled - covariance.location_
    distances = np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", delta, covariance.precision_, delta), 0.0)
    )

    # Accept every training sample.  A small margin prevents floating-point
    # changes between compatible NumPy/scikit-learn patch versions.
    threshold = float(distances.max() * 1.05)
    return {
        "minimum": X.min().astype(float).tolist(),
        "maximum": X.max().astype(float).tolist(),
        "median": X.median().astype(float).tolist(),
        "q01": X.quantile(0.01).astype(float).tolist(),
        "q99": X.quantile(0.99).astype(float).tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "location": covariance.location_.astype(float).tolist(),
        "precision": covariance.precision_.astype(float).tolist(),
        "mahalanobis_threshold": threshold,
        "method": "training_min_max + LedoitWolf_Mahalanobis",
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH, encoding="gb18030")
    targets = list(TARGET_CONFIG)
    feature_names = [column for column in data.columns if column not in targets]
    X = data[feature_names].astype(float)

    if len(feature_names) < 30:
        raise ValueError("SelectKBest_30 requires at least 30 input features")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Training features contain NaN or infinity")

    models = {}
    evaluations = {}
    selected_features = {}
    for target, config in TARGET_CONFIG.items():
        y = data[target].astype(float)

        # Evaluation is recorded for auditability; the deployed model is then
        # refitted on all available rows without changing any hyperparameter.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, shuffle=True
        )
        evaluation_model = build_pipeline(config["svr"])
        evaluation_model.fit(X_train, y_train)
        y_pred = evaluation_model.predict(X_test)
        evaluations[target] = {
            "test_rows": int(len(y_test)),
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }

        final_model = build_pipeline(config["svr"])
        final_model.fit(X, y)
        models[target] = final_model
        support = final_model.named_steps["selector"].get_support(indices=True)
        selected_features[target] = [feature_names[index] for index in support]

    frozen_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    source_sha256 = sha256_file(DATA_PATH)
    bundle = {
        "schema_version": 1,
        "model_version": "svr-rbf-selectkbest30-v1",
        "frozen_at_utc": frozen_at,
        "training_data": {"file": DATA_PATH.name, "sha256": source_sha256, "rows": len(data)},
        "feature_names": feature_names,
        "target_config": TARGET_CONFIG,
        "models": models,
        "domain_guard": build_domain_guard(X),
        "output_policy": "clip_to_declared_scale",
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3, protocol=5)
    model_sha256 = sha256_file(MODEL_PATH)

    manifest = {
        "schema_version": 1,
        "model_version": bundle["model_version"],
        "status": "FROZEN",
        "artifact": MODEL_PATH.name,
        "artifact_sha256": model_sha256,
        "frozen_at_utc": frozen_at,
        "training_data": bundle["training_data"],
        "model_family": "SVR_rbf",
        "pipeline": ["StandardScaler", "SelectKBest(f_regression, k=30)", "SVR(kernel='rbf')"],
        "targets": TARGET_CONFIG,
        "selected_features": selected_features,
        "holdout_evaluation": evaluations,
        "domain_guard": {
            "method": bundle["domain_guard"]["method"],
            "mahalanobis_threshold": bundle["domain_guard"]["mahalanobis_threshold"],
            "univariate_rule": "warn when any value is outside the training minimum/maximum",
        },
        "output_policy": bundle["output_policy"],
        "library_versions": bundle["library_versions"],
        "release_rule": "Do not overwrite. Create a new versioned artifact for any future change.",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Final read-back verification catches incomplete/corrupt writes now rather
    # than during deployment.
    if sha256_file(MODEL_PATH) != model_sha256:
        raise RuntimeError("Frozen model checksum changed during write")
    loaded = joblib.load(MODEL_PATH)
    if loaded["model_version"] != bundle["model_version"]:
        raise RuntimeError("Frozen model read-back verification failed")

    print(f"Frozen model: {MODEL_PATH}")
    print(f"Manifest:     {MANIFEST_PATH}")
    print(f"SHA-256:      {model_sha256}")


if __name__ == "__main__":
    main()
