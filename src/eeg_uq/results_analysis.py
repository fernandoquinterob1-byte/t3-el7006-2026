"""Shared, post-hoc metrics for the saved SEED/UQ result artifacts.

This module deliberately contains no model, data-loader, training, or inference
code.  It only reads arrays that have already been exported and applies the
same timestep-level definitions to SEED, Laplace, and Conformal reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from eeg_uq.metrics import calculate_metrics_simple


EPS = 1e-7
ECE_BINS = 15


def _as_bool_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"Mask shape {result.shape} does not match {shape}")
    return result


def _valid_vectors(
    probability: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    if probability.shape != target.shape:
        raise ValueError(f"Probability shape {probability.shape} does not match target {target.shape}")
    valid = _as_bool_mask(mask, target.shape)
    finite = np.isfinite(probability) & np.isfinite(target)
    valid &= finite
    return np.clip(probability[valid], EPS, 1.0 - EPS), target[valid].astype(np.int8)


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Binary entropy in nats, element-wise."""
    # Keep large posterior sample tensors in float32.  Scalar aggregation and
    # the flattened metric vectors below use float64 where it is useful, but
    # converting all (30, 150, 23000) samples would need excessive memory.
    probability = np.clip(np.asarray(probability, dtype=np.float32), EPS, 1.0 - EPS)
    return -(probability * np.log(probability) + (1.0 - probability) * np.log1p(-probability))


def expected_calibration_error(
    probability: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    bins: int = ECE_BINS,
) -> float:
    """Standard binary ECE using confidence and correctness.

    Confidence is ``max(p, 1-p)`` and correctness is the thresholded
    prediction against the binary target.  The returned value is always in
    [0, 1], never a percentage.
    """
    p, y = _valid_vectors(probability, target, mask)
    if p.size == 0:
        return float("nan")
    confidence = np.maximum(p, 1.0 - p)
    predicted = (p >= 0.5).astype(np.int8)
    correct = (predicted == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            in_bin = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            in_bin = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(in_bin):
            ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
    return float(ece)


def _safe_auc(y: np.ndarray, score: np.ndarray, average_precision: bool = False) -> float:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if np.unique(y).size < 2:
        return float("nan")
    if average_precision:
        return float(average_precision_score(y, score))
    return float(roc_auc_score(y, score))


def roc_pr_curves(
    probability: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Return ROC/PR arrays and scalar AUC/AP from valid timesteps."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score

    p, y = _valid_vectors(probability, target, mask)
    if np.unique(y).size < 2:
        return {
            "fpr": np.array([], dtype=np.float64),
            "tpr": np.array([], dtype=np.float64),
            "precision": np.array([], dtype=np.float64),
            "recall": np.array([], dtype=np.float64),
            "thresholds": np.array([], dtype=np.float64),
            "roc_auc": float("nan"),
            "auprc": float("nan"),
        }
    fpr, tpr, roc_thresholds = roc_curve(y, p)
    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "thresholds": roc_thresholds,
        "pr_thresholds": pr_thresholds,
        "roc_auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def timestep_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Common timestep-level metrics used in all three result notebooks."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    p, y = _valid_vectors(probability, target, mask)
    if p.size == 0:
        return {key: float("nan") for key in (
            "roc_auc", "auprc", "timestep_f1", "timestep_precision",
            "timestep_recall", "nll", "ece", "predictive_entropy",
        )} | {"valid_timesteps": 0.0}
    predicted = (p >= 0.5).astype(np.int8)
    curves = roc_pr_curves(probability, target, mask)
    return {
        "roc_auc": float(curves["roc_auc"]),
        "auprc": float(curves["auprc"]),
        "timestep_f1": float(f1_score(y, predicted, zero_division=0)),
        "timestep_precision": float(precision_score(y, predicted, zero_division=0)),
        "timestep_recall": float(recall_score(y, predicted, zero_division=0)),
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        "ece": expected_calibration_error(probability, target, mask, bins=ECE_BINS),
        "predictive_entropy": float(np.mean(binary_entropy(p))),
        "valid_timesteps": float(p.size),
    }


def event_metrics(event_iou_values: np.ndarray | None) -> dict[str, float]:
    """Return legacy event-level metrics without changing the project pairing."""
    if event_iou_values is None:
        return {"event_f1": float("nan"), "event_precision": float("nan"), "event_recall": float("nan")}
    values = np.asarray(event_iou_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"event_f1": float("nan"), "event_precision": float("nan"), "event_recall": float("nan")}
    metrics = calculate_metrics_simple(values)
    index = int(metrics["selected_index"])
    return {
        "event_f1": float(metrics["f1"][index]),
        "event_precision": float(metrics["precision"][index]),
        "event_recall": float(metrics["recall"][index]),
    }


def posterior_metrics(
    predictive_probabilities: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    sample_axis: int = 0,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Metrics for posterior predictive samples and their predictive mean."""
    samples = np.asarray(predictive_probabilities, dtype=np.float32)
    samples = np.moveaxis(samples, sample_axis, 0)
    mean_probability = np.mean(samples, axis=0)
    predictive_entropy = binary_entropy(mean_probability)
    # Accumulate sample entropies one posterior draw at a time instead of
    # materialising another (samples, records, time) tensor.
    expected_entropy = np.zeros_like(mean_probability, dtype=np.float32)
    for sample in samples:
        expected_entropy += binary_entropy(sample)
    expected_entropy /= float(samples.shape[0])
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    valid = _as_bool_mask(mask, mean_probability.shape)
    finite = np.isfinite(mean_probability) & np.isfinite(target)
    valid &= finite
    base = timestep_metrics(mean_probability, target, valid)
    base.update({
        "predictive_entropy": float(np.mean(predictive_entropy[valid])),
        "expected_entropy": float(np.mean(expected_entropy[valid])),
        "mutual_information": float(np.mean(mutual_information[valid])),
        "predictive_samples": float(samples.shape[0]),
    })
    arrays = {
        "mean_probability": mean_probability,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
    }
    return base, arrays


def posterior_dispersion_diagnostics(
    predictive_probabilities: np.ndarray,
    mask: np.ndarray | None = None,
    sample_axis: int = 0,
) -> dict[str, float]:
    """Summarize posterior probability dispersion to explain small MI values."""
    samples = np.moveaxis(np.asarray(predictive_probabilities, dtype=np.float32), sample_axis, 0)
    point_mask = _as_bool_mask(mask, samples.shape[1:])
    std = np.std(samples, axis=0)
    span = np.max(samples, axis=0) - np.min(samples, axis=0)
    valid_std = std[point_mask]
    valid_span = span[point_mask]
    return {
        "sample_probability_std_mean": float(np.mean(valid_std)),
        "sample_probability_std_median": float(np.median(valid_std)),
        "sample_probability_std_p95": float(np.quantile(valid_std, 0.95)),
        "sample_probability_std_max": float(np.max(valid_std)),
        "sample_probability_range_mean": float(np.mean(valid_span)),
        "sample_probability_range_p95": float(np.quantile(valid_span, 0.95)),
        "sample_probability_range_max": float(np.max(valid_span)),
    }


def scalar_value(value: np.ndarray) -> float:
    """Convert a zero-dimensional NPZ value to a Python float."""
    return float(np.asarray(value).reshape(()))


def fold_number_from_path(path: Path) -> int:
    """Return a zero-based fold number from either project folder convention."""
    name = path.name
    if "fold_" not in name:
        raise ValueError(f"Cannot infer fold from {path}")
    return int(name.rsplit("fold_", 1)[1].split("_", 1)[0])


def sorted_fold_dirs(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern), key=fold_number_from_path)
