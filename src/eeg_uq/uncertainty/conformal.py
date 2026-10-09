"""Post-hoc split conformal prediction for binary temporal SEED outputs.

This module operates only on already-computed probabilities.  It does not
import the SEED model, run inference, or alter the legacy event pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil

import numpy as np

from eeg_uq.metrics import calculate_metrics_simple, legacy_event_records


EVENT_FIELDS = (
    "event_sample_id",
    "event_subject_id",
    "event_true_start",
    "event_true_end",
    "event_pred_start",
    "event_pred_end",
    "event_status",
    "event_iou",
)


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")
    return alpha


def _validate_temporal_inputs(
    probabilities: np.ndarray,
    y_true: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.size == 0:
        raise ValueError("probabilities must not be empty.")
    if valid_mask is None:
        mask = np.ones(probabilities.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != probabilities.shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match probabilities shape "
                f"{probabilities.shape}."
            )
    if not mask.any():
        raise ValueError("valid_mask does not select any timestep.")
    valid_probabilities = probabilities[mask]
    if not np.isfinite(valid_probabilities).all():
        raise ValueError("Valid probabilities must all be finite.")
    if np.any((valid_probabilities < 0.0) | (valid_probabilities > 1.0)):
        raise ValueError("Valid probabilities must lie in [0, 1].")

    truth = None
    if y_true is not None:
        truth = np.asarray(y_true)
        if truth.shape != probabilities.shape:
            raise ValueError(
                f"y_true shape {truth.shape} does not match probabilities shape "
                f"{probabilities.shape}."
            )
        if not np.isin(truth[mask], (0, 1)).all():
            raise ValueError("Valid y_true values must be binary (0 or 1).")
    return probabilities, truth, mask


def compute_nonconformity_scores(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``1 - P(y_true | x)`` for valid binary timesteps.

    The returned array is flat and contains only entries selected by
    ``valid_mask``.  For y=1 the score is ``1-p``; for y=0 it is ``p``.
    """

    probabilities, truth, mask = _validate_temporal_inputs(probabilities, y_true, valid_mask)
    scores = np.where(truth == 1, 1.0 - probabilities, probabilities)
    return np.asarray(scores[mask], dtype=float)


def conformal_rank(n: int, alpha: float) -> int:
    """Return the one-based finite-sample split-conformal order statistic."""

    if n < 1:
        raise ValueError("At least one calibration score is required.")
    alpha = _validate_alpha(alpha)
    return int(ceil((n + 1) * (1.0 - alpha)))


def fit_conformal_threshold(scores: np.ndarray, alpha: float = 0.05) -> float:
    """Fit the corrected split-conformal threshold.

    For ``n`` calibration scores, this selects order statistic
    ``k = ceil((n + 1) * (1 - alpha))``.  If ``k=n+1``, the formal augmented
    quantile is infinity.  Since this score is bounded by one, returning 1.0
    is prediction-set equivalent and keeps artifacts finite/portable.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    if not values.size:
        raise ValueError("At least one calibration score is required.")
    if not np.isfinite(values).all():
        raise ValueError("Calibration scores must all be finite.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Nonconformity scores must lie in [0, 1].")
    rank = conformal_rank(len(values), alpha)
    if rank > len(values):
        return 1.0
    return float(np.partition(values, rank - 1)[rank - 1])


def build_prediction_sets(
    probabilities: np.ndarray,
    q_hat: float,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build encoded binary prediction sets without object arrays.

    ``contains_background`` encodes membership of class 0 and
    ``contains_event`` membership of class 1.  Invalid timesteps are cleared
    to zero and must be interpreted together with the returned ``valid_mask``.
    """

    probabilities, _, mask = _validate_temporal_inputs(probabilities, valid_mask=valid_mask)
    q_hat = float(q_hat)
    if np.isnan(q_hat) or q_hat < 0.0:
        raise ValueError("q_hat must be non-negative and not NaN.")
    contains_background = np.logical_and(probabilities <= q_hat, mask)
    contains_event = np.logical_and((1.0 - probabilities) <= q_hat, mask)
    set_size = contains_background.astype(np.uint8) + contains_event.astype(np.uint8)
    return {
        "contains_background": contains_background.astype(np.uint8),
        "contains_event": contains_event.astype(np.uint8),
        "set_size": set_size,
        "valid_mask": mask.astype(np.uint8),
    }


def compute_conformal_metrics(
    y_true: np.ndarray,
    contains_background: np.ndarray,
    contains_event: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    alpha: float | None = None,
) -> dict[str, float | int]:
    """Compute coverage and prediction-set efficiency on valid timesteps."""

    background = np.asarray(contains_background, dtype=bool)
    event = np.asarray(contains_event, dtype=bool)
    if background.shape != event.shape:
        raise ValueError("Prediction-set membership arrays must have the same shape.")
    _, truth, mask = _validate_temporal_inputs(
        np.full(background.shape, 0.5, dtype=float), y_true, valid_mask
    )
    sizes = background.astype(np.uint8) + event.astype(np.uint8)
    covered = np.where(truth == 1, event, background)
    valid_sizes = sizes[mask]
    metrics: dict[str, float | int] = {
        "n_valid_timesteps": int(mask.sum()),
        "empirical_coverage": float(covered[mask].mean()),
        "mean_set_size": float(valid_sizes.mean()),
        "singleton_rate": float((valid_sizes == 1).mean()),
        "ambiguous_rate": float((valid_sizes == 2).mean()),
        "empty_rate": float((valid_sizes == 0).mean()),
    }
    if alpha is not None:
        alpha = _validate_alpha(alpha)
        metrics["alpha"] = alpha
        metrics["target_coverage"] = 1.0 - alpha
        metrics["coverage_gap"] = metrics["empirical_coverage"] - (1.0 - alpha)
    return metrics


def extract_event_records(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Read saved legacy event arrays or derive them with the legacy pairing.

    Derivation is used only when ``y_true`` and ``processed_pred`` are present;
    it calls the existing legacy event-record helper unchanged.
    """

    saved_required = {
        "event_sample_id",
        "event_pred_start",
        "event_pred_end",
        "event_status",
        "event_iou_values",
    }
    if saved_required.issubset(arrays):
        event_sample_id = np.asarray(arrays["event_sample_id"])
        count = len(event_sample_id)

        def saved(name: str, default: int = -1, dtype=None) -> np.ndarray:
            if name not in arrays:
                return np.full(count, default, dtype=np.int64 if dtype is None else dtype)
            value = np.asarray(arrays[name], dtype=dtype)
            if len(value) != count:
                raise ValueError(f"{name} length does not match event_sample_id length.")
            return value

        result = {
            "event_sample_id": event_sample_id,
            "event_subject_id": saved("event_subject_id", default="", dtype="U1"),
            "event_true_start": saved("event_true_start"),
            "event_true_end": saved("event_true_end"),
            "event_pred_start": saved("event_pred_start"),
            "event_pred_end": saved("event_pred_end"),
            "event_status": saved("event_status", default="", dtype="U16"),
            "event_iou": np.asarray(arrays["event_iou_values"], dtype=float),
        }
        if len(result["event_iou"]) != count:
            raise ValueError("event_iou_values length does not match event_sample_id length.")
        return result

    if "y_true" not in arrays or "processed_pred" not in arrays:
        return _empty_event_records()
    y_true = np.asarray(arrays["y_true"])
    processed = np.asarray(arrays["processed_pred"])
    if y_true.shape != processed.shape or y_true.ndim != 2:
        raise ValueError("Event derivation requires matching two-dimensional y_true/processed_pred.")
    sample_ids = np.asarray(arrays.get("sample_id", np.arange(len(y_true))))
    subject_ids = np.asarray(arrays.get("subject_id", np.full(len(y_true), "")), dtype="U")
    values: dict[str, list] = {name: [] for name in EVENT_FIELDS}
    for row, (truth, prediction) in enumerate(zip(y_true, processed)):
        for record in legacy_event_records(truth.astype(int), prediction.astype(int)):
            values["event_sample_id"].append(sample_ids[row])
            values["event_subject_id"].append(subject_ids[row])
            values["event_true_start"].append(record["true_start"])
            values["event_true_end"].append(record["true_end"])
            values["event_pred_start"].append(record["pred_start"])
            values["event_pred_end"].append(record["pred_end"])
            values["event_status"].append(record["status"])
            values["event_iou"].append(record["iou"])
    return {
        "event_sample_id": np.asarray(values["event_sample_id"], dtype=sample_ids.dtype),
        "event_subject_id": np.asarray(values["event_subject_id"], dtype="U"),
        "event_true_start": np.asarray(values["event_true_start"], dtype=np.int64),
        "event_true_end": np.asarray(values["event_true_end"], dtype=np.int64),
        "event_pred_start": np.asarray(values["event_pred_start"], dtype=np.int64),
        "event_pred_end": np.asarray(values["event_pred_end"], dtype=np.int64),
        "event_status": np.asarray(values["event_status"], dtype="U16"),
        "event_iou": np.asarray(values["event_iou"], dtype=float),
    }


def _empty_event_records() -> dict[str, np.ndarray]:
    return {
        "event_sample_id": np.asarray([], dtype=np.int64),
        "event_subject_id": np.asarray([], dtype="U1"),
        "event_true_start": np.asarray([], dtype=np.int64),
        "event_true_end": np.asarray([], dtype=np.int64),
        "event_pred_start": np.asarray([], dtype=np.int64),
        "event_pred_end": np.asarray([], dtype=np.int64),
        "event_status": np.asarray([], dtype="U16"),
        "event_iou": np.asarray([], dtype=float),
    }


def summarize_event_conformal(
    probabilities: np.ndarray,
    set_size: np.ndarray,
    contains_event: np.ndarray,
    event_records: Mapping[str, np.ndarray],
    *,
    sample_ids: np.ndarray | None = None,
    subject_ids: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    event_probabilities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Attach exploratory timestep-conformal summaries to legacy events.

    The summaries do not create or claim an event-level conformal guarantee.
    Event spans and IoU/status are consumed as supplied by the legacy output.
    ``event_probability_seed`` is the 75th percentile within the predicted
    event span.  Callers may provide an adjusted probability array; otherwise
    the saved SEED probability array is used.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Event summaries require two-dimensional (sample, time) probabilities.")
    set_size = np.asarray(set_size, dtype=np.uint8)
    contains_event = np.asarray(contains_event, dtype=bool)
    if set_size.shape != probabilities.shape or contains_event.shape != probabilities.shape:
        raise ValueError("Prediction-set arrays must match probabilities for event summaries.")
    if valid_mask is None:
        mask = np.ones(probabilities.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != probabilities.shape:
            raise ValueError("valid_mask must match probabilities for event summaries.")
    if event_probabilities is None:
        event_probabilities = probabilities
    else:
        event_probabilities = np.asarray(event_probabilities, dtype=float)
        if event_probabilities.shape != probabilities.shape:
            raise ValueError("event_probabilities must match probabilities.")

    count = len(np.asarray(event_records["event_sample_id"]))
    for key in EVENT_FIELDS:
        if key not in event_records or len(np.asarray(event_records[key])) != count:
            raise ValueError(f"Event field {key} is missing or has inconsistent length.")
    if sample_ids is None:
        sample_ids = np.arange(probabilities.shape[0])
    else:
        sample_ids = np.asarray(sample_ids)
    if len(sample_ids) != probabilities.shape[0] or len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must contain one unique ID per probability row.")
    if subject_ids is None:
        subject_ids = np.full(probabilities.shape[0], "", dtype="U1")
    else:
        subject_ids = np.asarray(subject_ids, dtype="U")
        if len(subject_ids) != probabilities.shape[0]:
            raise ValueError("subject_ids must contain one ID per probability row.")
    row_by_sample = {value.item() if hasattr(value, "item") else value: row for row, value in enumerate(sample_ids)}

    probability_values = np.full(count, np.nan, dtype=float)
    ambiguous_values = np.full(count, np.nan, dtype=float)
    mean_size_values = np.full(count, np.nan, dtype=float)
    event_singleton_values = np.full(count, np.nan, dtype=float)
    valid_counts = np.zeros(count, dtype=np.int64)
    source_event_subject_ids = np.asarray(event_records["event_subject_id"], dtype="U")
    subject_width = max(
        1,
        max((len(str(value)) for value in subject_ids), default=1),
        max((len(str(value)) for value in source_event_subject_ids), default=1),
    )
    event_subject_ids = source_event_subject_ids.astype(f"U{subject_width}", copy=True)
    event_sample_ids = np.asarray(event_records["event_sample_id"])
    starts = np.asarray(event_records["event_pred_start"], dtype=np.int64)
    ends = np.asarray(event_records["event_pred_end"], dtype=np.int64)

    for index, (sample_id, start, end) in enumerate(zip(event_sample_ids, starts, ends)):
        if start < 0 or end <= start:
            continue
        key = sample_id.item() if hasattr(sample_id, "item") else sample_id
        if key not in row_by_sample:
            raise ValueError(f"Event references unknown sample_id {key!r}.")
        row = row_by_sample[key]
        start = max(0, int(start))
        end = min(probabilities.shape[1], int(end))
        inside = mask[row, start:end]
        if not inside.any():
            continue
        valid_counts[index] = int(inside.sum())
        event_probability_values = event_probabilities[row, start:end][inside]
        sizes = set_size[row, start:end][inside]
        event_membership = contains_event[row, start:end][inside]
        probability_values[index] = float(np.percentile(event_probability_values, 75))
        ambiguous_values[index] = float((sizes == 2).mean())
        mean_size_values[index] = float(sizes.mean())
        event_singleton_values[index] = float(np.logical_and(sizes == 1, event_membership).mean())
        if not event_subject_ids[index]:
            event_subject_ids[index] = subject_ids[row]

    status = np.asarray(event_records["event_status"], dtype="U16")
    return {
        "event_sample_id": event_sample_ids,
        "event_subject_id": event_subject_ids,
        "event_true_start": np.asarray(event_records["event_true_start"], dtype=np.int64),
        "event_true_end": np.asarray(event_records["event_true_end"], dtype=np.int64),
        "event_pred_start": starts,
        "event_pred_end": ends,
        "event_status": status,
        "event_iou": np.asarray(event_records["event_iou"], dtype=float),
        "event_probability_seed": probability_values,
        "fraction_ambiguous_timesteps": ambiguous_values,
        "mean_set_size_inside_event": mean_size_values,
        "fraction_event_singleton": event_singleton_values,
        "event_valid_timesteps": valid_counts,
        "event_is_tp": (status == "TP").astype(np.uint8),
        "event_is_fp": (status == "FP").astype(np.uint8),
        "event_has_prediction": np.logical_and(starts >= 0, ends > starts).astype(np.uint8),
    }


def compute_rejection_curve(
    event_summary: Mapping[str, np.ndarray],
    thresholds: Sequence[float] = (0.1, 0.25, 0.5),
) -> dict[str, np.ndarray]:
    """Evaluate exploratory event rejection using legacy IoU/F1 accounting.

    A predicted event is accepted when its ambiguous fraction is at most the
    threshold.  Rejected pure FPs disappear; rejected predictions paired with
    a true event become FNs.  Existing FNs remain.  Metric computation is
    delegated to ``calculate_metrics_simple`` and therefore retains the legacy
    effective IoU point.
    """

    threshold_values = np.asarray(thresholds, dtype=float).reshape(-1)
    if not len(threshold_values) or not np.isfinite(threshold_values).all():
        raise ValueError("Rejection thresholds must be a non-empty finite sequence.")
    if np.any((threshold_values < 0.0) | (threshold_values > 1.0)):
        raise ValueError("Rejection thresholds must lie in [0, 1].")

    ambiguity = np.asarray(event_summary["fraction_ambiguous_timesteps"], dtype=float)
    has_prediction = np.asarray(event_summary["event_has_prediction"], dtype=bool)
    true_start = np.asarray(event_summary["event_true_start"], dtype=np.int64)
    iou = np.asarray(event_summary["event_iou"], dtype=float)
    if not (len(ambiguity) == len(has_prediction) == len(true_start) == len(iou)):
        raise ValueError("Event arrays have inconsistent lengths.")
    total_predictions = int(has_prediction.sum())

    acceptance, f1, precision, recall, mean_iou = [], [], [], [], []
    tp_values, fp_values, fn_values, accepted_counts = [], [], [], []
    effective_iou_threshold = np.nan
    for threshold in threshold_values:
        accepted = np.logical_and.reduce((has_prediction, np.isfinite(ambiguity), ambiguity <= threshold))
        accepted_count = int(accepted.sum())
        accepted_counts.append(accepted_count)
        acceptance.append(accepted_count / total_predictions if total_predictions else 0.0)
        filtered_results: list[float] = []
        for index in range(len(iou)):
            if not has_prediction[index]:
                if true_start[index] >= 0:
                    filtered_results.append(0.0)
            elif accepted[index]:
                filtered_results.append(float(iou[index]))
            elif true_start[index] >= 0:
                filtered_results.append(0.0)
        curve = calculate_metrics_simple(filtered_results)
        selected = curve["selected_index"]
        effective_iou_threshold = curve["selected_iou_threshold"]
        f1.append(float(curve["f1"][selected]))
        precision.append(float(curve["precision"][selected]))
        recall.append(float(curve["recall"][selected]))
        tp_values.append(int(curve["true_positives"][selected]))
        fp_values.append(int(curve["false_positives"][selected]))
        fn_values.append(int(curve["false_negatives"][selected]))
        matched = np.logical_and(accepted, iou >= 0.0)
        mean_iou.append(float(iou[matched].mean()) if matched.any() else np.nan)
    return {
        "rejection_threshold": threshold_values,
        "acceptance_rate": np.asarray(acceptance, dtype=float),
        "accepted_event_count": np.asarray(accepted_counts, dtype=np.int64),
        "total_predicted_events": np.full(len(threshold_values), total_predictions, dtype=np.int64),
        "accepted_f1": np.asarray(f1, dtype=float),
        "accepted_precision": np.asarray(precision, dtype=float),
        "accepted_recall": np.asarray(recall, dtype=float),
        "accepted_mean_iou": np.asarray(mean_iou, dtype=float),
        "accepted_tp": np.asarray(tp_values, dtype=np.int64),
        "accepted_fp": np.asarray(fp_values, dtype=np.int64),
        "accepted_fn": np.asarray(fn_values, dtype=np.int64),
        "effective_iou_threshold": np.full(
            len(threshold_values), effective_iou_threshold, dtype=float
        ),
    }
