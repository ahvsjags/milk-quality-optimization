"""Read-only inference service for the frozen milk sensory SVR bundle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


OOD_WARNING = "该数据已超出本模型的适用范围（Out of Domain），预测结果仅供参考"


class InputValidationError(ValueError):
    """Raised when a prediction request cannot be converted to a feature row."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenModelService:
    """Load, verify and serve one immutable, versioned joblib artifact."""

    def __init__(self, model_path: Path, manifest_path: Path) -> None:
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        actual_digest = _sha256_file(self.model_path)
        expected_digest = self.manifest["artifact_sha256"]
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"Frozen model checksum mismatch: expected {expected_digest}, got {actual_digest}"
            )

        self.bundle = joblib.load(self.model_path)
        if self.bundle["model_version"] != self.manifest["model_version"]:
            raise RuntimeError("Model and manifest versions do not match")
        self.feature_names = tuple(self.bundle["feature_names"])
        self.feature_set = frozenset(self.feature_names)

    @property
    def model_version(self) -> str:
        return self.bundle["model_version"]

    def default_features(self) -> dict[str, float]:
        medians = self.bundle["domain_guard"]["median"]
        return dict(zip(self.feature_names, medians))

    def feature_schema(self) -> list[dict[str, Any]]:
        guard = self.bundle["domain_guard"]
        return [
            {
                "name": name,
                "minimum": float(guard["minimum"][index]),
                "maximum": float(guard["maximum"][index]),
                "median": float(guard["median"][index]),
                "q01": float(guard["q01"][index]),
                "q99": float(guard["q99"][index]),
            }
            for index, name in enumerate(self.feature_names)
        ]

    def _validate(self, raw_features: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(raw_features, Mapping):
            raise InputValidationError("features 必须是以化合物名称为键的对象")

        missing = [name for name in self.feature_names if name not in raw_features]
        if missing:
            preview = "、".join(missing[:5])
            suffix = "…" if len(missing) > 5 else ""
            raise InputValidationError(f"缺少 {len(missing)} 个输入特征：{preview}{suffix}")

        unknown = [str(name) for name in raw_features if name not in self.feature_set]
        if unknown:
            preview = "、".join(unknown[:5])
            suffix = "…" if len(unknown) > 5 else ""
            raise InputValidationError(f"包含未知输入特征：{preview}{suffix}")

        values = []
        for name in self.feature_names:
            raw_value = raw_features[name]
            if isinstance(raw_value, bool):
                raise InputValidationError(f"{name} 必须是有限数值")
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InputValidationError(f"{name} 必须是有限数值") from exc
            if not math.isfinite(value):
                raise InputValidationError(f"{name} 必须是有限数值")
            if value < 0:
                raise InputValidationError(f"{name} 的浓度不能为负数")
            values.append(value)
        return np.asarray(values, dtype=float)

    def _check_domain(self, values: np.ndarray) -> dict[str, Any]:
        guard = self.bundle["domain_guard"]
        minimum = np.asarray(guard["minimum"], dtype=float)
        maximum = np.asarray(guard["maximum"], dtype=float)
        violations = []
        for index in np.flatnonzero((values < minimum) | (values > maximum)):
            violations.append(
                {
                    "feature": self.feature_names[index],
                    "value": float(values[index]),
                    "training_min": float(minimum[index]),
                    "training_max": float(maximum[index]),
                }
            )

        # Distances are evaluated on a boundary-clipped copy.  The original
        # values and their violations remain in the report, while this avoids
        # overflow for inputs such as 1e308 and keeps the API fail-safe.
        distance_values = np.clip(values, minimum, maximum)
        mean = np.asarray(guard["scaler_mean"], dtype=float)
        scale = np.asarray(guard["scaler_scale"], dtype=float)
        location = np.asarray(guard["location"], dtype=float)
        precision = np.asarray(guard["precision"], dtype=float)
        scaled = (distance_values - mean) / scale
        delta = scaled - location
        mahalanobis = float(math.sqrt(max(float(delta @ precision @ delta), 0.0)))
        threshold = float(guard["mahalanobis_threshold"])
        multivariate_outlier = mahalanobis > threshold
        out_of_domain = bool(violations or multivariate_outlier)

        reasons = []
        if violations:
            reasons.append(f"{len(violations)} 个化合物浓度超出训练数据最小/最大范围")
        if multivariate_outlier:
            reasons.append("整体化合物谱与训练牛奶样本差异过大")
        return {
            "out_of_domain": out_of_domain,
            "warning": OOD_WARNING if out_of_domain else None,
            "reasons": reasons,
            "violations": violations,
            "mahalanobis_distance": round(mahalanobis, 4),
            "mahalanobis_threshold": round(threshold, 4),
        }

    def predict(self, raw_features: Mapping[str, Any]) -> dict[str, Any]:
        values = self._validate(raw_features)
        domain = self._check_domain(values)
        guard = self.bundle["domain_guard"]
        inference_values = np.clip(
            values,
            np.asarray(guard["minimum"], dtype=float),
            np.asarray(guard["maximum"], dtype=float),
        )
        row = pd.DataFrame([inference_values], columns=self.feature_names)

        predictions = []
        for target, config in self.bundle["target_config"].items():
            raw_score = float(self.bundle["models"][target].predict(row)[0])
            lower, upper = (float(value) for value in config["range"])
            clipped_score = float(np.clip(raw_score, lower, upper))
            predictions.append(
                {
                    "target": target,
                    "label": config["label"],
                    "score": round(clipped_score, 3),
                    "raw_score": round(raw_score, 6),
                    "scale": [lower, upper],
                    "scale_label": f"{lower:g}–{upper:g}",
                    "optimizer": config["optimizer"],
                    "clipped": not math.isclose(raw_score, clipped_score, abs_tol=1e-12),
                }
            )

        return {
            "model_version": self.model_version,
            "model_family": "SVR_rbf + SelectKBest_30",
            "predictions": predictions,
            "out_of_domain": domain["out_of_domain"],
            "warning": domain["warning"],
            "domain": domain,
            "inference_input_policy": "out-of-range features clipped to training min/max",
        }
