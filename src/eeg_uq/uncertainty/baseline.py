"""Post-hoc confidence and entropy for SEED's binary temporal output."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def binary_uncertainty(probabilities: np.ndarray, epsilon: float = 1e-8) -> dict[str, np.ndarray]:
    """Return timestep-level binary confidence, uncertainty and predictive entropy."""
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1 - epsilon)
    confidence = np.maximum(probability, 1 - probability)
    entropy = -(probability * np.log(probability) + (1 - probability) * np.log(1 - probability))
    return {"confidence": confidence, "uncertainty": 1 - confidence, "entropy": entropy}


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    return {"count": int(len(values)), "mean": float(np.mean(values)) if len(values) else None, "median": float(np.median(values)) if len(values) else None}


def analyse_prediction_file(predictions_path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Analyse dense output without changing the detector or event-matching protocol."""
    with np.load(predictions_path, allow_pickle=False) as archive:
        probabilities = archive["probabilities"]
        target = archive["y_true"].astype(np.int8)
        prediction = archive["y_pred"].astype(np.int8)
        copied = {key: archive[key] for key in archive.files}
    scores = binary_uncertainty(probabilities)
    incorrect = prediction != target
    entropy = scores["entropy"]
    uncertainty = scores["uncertainty"]
    report = {
        "unit": "timestep",
        "correct_entropy": _summary(entropy[~incorrect]),
        "incorrect_entropy": _summary(entropy[incorrect]),
        "correct_uncertainty": _summary(uncertainty[~incorrect]),
        "incorrect_uncertainty": _summary(uncertainty[incorrect]),
    }
    labels = incorrect.reshape(-1).astype(np.int8)
    values = uncertainty.reshape(-1)
    if labels.min() != labels.max():
        try:
            from sklearn.metrics import roc_auc_score
            report["error_detection_auroc_uncertainty"] = float(roc_auc_score(labels, values))
        except ImportError:
            report["error_detection_auroc_uncertainty"] = None
    else:
        report["error_detection_auroc_uncertainty"] = None
    copied.update({key: value.astype(np.float32) for key, value in scores.items()})
    copied["temporal_error"] = incorrect.astype(np.int8)
    return report, copied


def event_mean_entropy(probabilities: np.ndarray, start: int, end: int) -> float | None:
    """Exploratory event aggregation; the event boundaries remain legacy boundaries."""
    if start < 0 or end <= start:
        return None
    return float(binary_uncertainty(probabilities[start:end])["entropy"].mean())
