"""Checkpoint evaluation and reusable prediction export for the SEED baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .folds import get_fold, load_folds, validate_folds
from .metrics import (
    DETECTION_THRESHOLD,
    DURATION_THRESHOLD,
    SAMPLE_RATE_HZ,
    calculate_metrics_simple,
    get_results_simple,
    legacy_event_records,
    process_spindle_simple,
)
from .model import SEEDModel


@dataclass
class FoldPredictions:
    evaluation: int
    fold: int
    sample_ids: np.ndarray
    subject_ids: np.ndarray
    y_true: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    y_pred: np.ndarray
    processed_pred: np.ndarray
    event_iou_values: np.ndarray
    event_iou_offsets: np.ndarray
    event_sample_id: np.ndarray
    event_subject_id: np.ndarray
    event_true_start: np.ndarray
    event_true_end: np.ndarray
    event_pred_start: np.ndarray
    event_pred_end: np.ndarray
    event_status: np.ndarray
    event_duration_samples: np.ndarray


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_legacy_checkpoint(weights_dir: str | Path, evaluation: int, fold: int, device: torch.device) -> SEEDModel:
    weights_dir = Path(weights_dir) / f"eval_{evaluation}" / f"fold_{fold}"
    expected = {
        "embedding": weights_dir / "embedding_best.pth",
        "transformer": weights_dir / "lstm_best.pth",
        "classifier": weights_dir / "classifier_best.pth",
    }
    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files: {missing}")
    model = SEEDModel().to(device)
    model.embedding.load_state_dict(torch.load(expected["embedding"], map_location=device, weights_only=True), strict=False)
    model.transformer.load_state_dict(torch.load(expected["transformer"], map_location=device, weights_only=True))
    model.classifier.load_state_dict(torch.load(expected["classifier"], map_location=device, weights_only=True))
    return model.eval()


@torch.no_grad()
def infer_full_signal_logits(model: SEEDModel, signal: np.ndarray, device: torch.device) -> np.ndarray:
    """Legacy 20 s / 10 s inference stitching, returning logits instead of only probabilities."""
    signal_window_seconds = 20
    stride_seconds = 10
    samples_per_window = SAMPLE_RATE_HZ * signal_window_seconds
    segments = int(len(signal) / SAMPLE_RATE_HZ - signal_window_seconds) // stride_seconds + 1
    if segments < 1:
        raise ValueError("Signal is shorter than the required 20-second inference window.")
    stitched = []
    central_start = SAMPLE_RATE_HZ * (signal_window_seconds - stride_seconds) // 2
    central_end = central_start + SAMPLE_RATE_HZ * stride_seconds
    for index in range(segments):
        start = stride_seconds * SAMPLE_RATE_HZ * index
        end = start + samples_per_window
        segment = torch.from_numpy(np.asarray(signal[start:end], dtype=np.float32)).unsqueeze(0).to(device)
        logits = model(segment).squeeze(0).detach().cpu().numpy()
        if index == 0:
            stitched.append(logits[:central_end])
        elif index + 1 == segments:
            stitched.append(logits[central_start:])
        else:
            stitched.append(logits[central_start:central_end])
    rest = (len(signal) / SAMPLE_RATE_HZ - signal_window_seconds) % stride_seconds
    if rest != 0:
        segment = torch.from_numpy(np.asarray(signal[-samples_per_window:], dtype=np.float32)).unsqueeze(0).to(device)
        logits = model(segment).squeeze(0).detach().cpu().numpy()
        stitched.append(logits[-int(rest * SAMPLE_RATE_HZ) :])
    logits = np.concatenate(stitched)
    if len(logits) != len(signal):
        raise RuntimeError(f"Stitched output length {len(logits)} differs from signal length {len(signal)}.")
    return logits.astype(np.float32, copy=False)


def evaluate_fold(
    data_path: str | Path,
    weights_dir: str | Path,
    folds_path: str | Path,
    evaluation: int,
    fold: int,
    device: str = "auto",
) -> FoldPredictions:
    device_obj = select_device(device)
    folds = load_folds(folds_path)
    _, test_subjects = get_fold(folds, evaluation, fold)
    with h5py.File(data_path, "r") as source:
        decoded_subject_ids = np.asarray([value.decode("utf-8") for value in source["subject_ids"][:]])
        validate_folds(folds, set(decoded_subject_ids))
        test_indices = np.flatnonzero(np.isin(decoded_subject_ids, test_subjects))
        if not len(test_indices):
            raise ValueError(f"No records found for eval {evaluation}, fold {fold} test subjects.")
        model = load_legacy_checkpoint(weights_dir, evaluation, fold, device_obj)
        logits_values, true_values, predicted_values, processed_values, iou_values = [], [], [], [], []
        event_sample_ids, event_subject_ids, event_truth_start, event_truth_end = [], [], [], []
        event_prediction_start, event_prediction_end, event_status, event_duration_samples = [], [], [], []
        offsets = [0]
        for sample_id in test_indices:
            signal = source["signals"][sample_id].astype(np.float32)
            labels = source["labels"][sample_id].astype(np.int8)
            logits = infer_full_signal_logits(model, signal, device_obj)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            prediction = (probabilities >= DETECTION_THRESHOLD).astype(np.int8)
            duration_prediction = (probabilities >= DURATION_THRESHOLD).astype(np.int8)
            processed = process_spindle_simple(prediction, duration_prediction).astype(np.int8)
            event_ious = get_results_simple(labels.astype(int), processed)
            event_records = legacy_event_records(labels.astype(int), processed)
            logits_values.append(logits)
            true_values.append(labels)
            predicted_values.append(prediction)
            processed_values.append(processed)
            iou_values.extend(event_ious)
            event_sample_ids.extend([int(sample_id)] * len(event_records))
            event_subject_ids.extend([decoded_subject_ids[sample_id]] * len(event_records))
            event_truth_start.extend(record["true_start"] for record in event_records)
            event_truth_end.extend(record["true_end"] for record in event_records)
            event_prediction_start.extend(record["pred_start"] for record in event_records)
            event_prediction_end.extend(record["pred_end"] for record in event_records)
            event_status.extend(record["status"] for record in event_records)
            event_duration_samples.extend(
                max(0, record["pred_end"] - record["pred_start"]) for record in event_records
            )
            offsets.append(len(iou_values))
    logits_array = np.stack(logits_values)
    return FoldPredictions(
        evaluation=evaluation,
        fold=fold,
        sample_ids=test_indices.astype(np.int64),
        subject_ids=decoded_subject_ids[test_indices].astype("U"),
        y_true=np.stack(true_values).astype(np.int8),
        logits=logits_array,
        probabilities=(1.0 / (1.0 + np.exp(-logits_array))).astype(np.float32),
        y_pred=np.stack(predicted_values).astype(np.int8),
        processed_pred=np.stack(processed_values).astype(np.int8),
        event_iou_values=np.asarray(iou_values, dtype=np.float32),
        event_iou_offsets=np.asarray(offsets, dtype=np.int64),
        event_sample_id=np.asarray(event_sample_ids, dtype=np.int64),
        event_subject_id=np.asarray(event_subject_ids, dtype="U"),
        event_true_start=np.asarray(event_truth_start, dtype=np.int64),
        event_true_end=np.asarray(event_truth_end, dtype=np.int64),
        event_pred_start=np.asarray(event_prediction_start, dtype=np.int64),
        event_pred_end=np.asarray(event_prediction_end, dtype=np.int64),
        event_status=np.asarray(event_status, dtype="U16"),
        event_duration_samples=np.asarray(event_duration_samples, dtype=np.int64),
    )


def save_fold_predictions(predictions: FoldPredictions, output_dir: str | Path) -> Path:
    """Save raw temporal outputs plus ragged event IoU values in a reusable NPZ file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The baseline internals use JSON/checkpoint IDs (one-based).  Put that
    # convention in the filename because the public CLIs intentionally use
    # zero-based indices.
    output_path = output_dir / f"predictions_legacy_eval_{predictions.evaluation}_fold_{predictions.fold}.npz"
    np.savez_compressed(
        output_path,
        sample_id=predictions.sample_ids,
        subject_id=predictions.subject_ids,
        y_true=predictions.y_true,
        logits=predictions.logits,
        probabilities=predictions.probabilities,
        y_pred=predictions.y_pred,
        processed_pred=predictions.processed_pred,
        event_iou_values=predictions.event_iou_values,
        event_iou_offsets=predictions.event_iou_offsets,
        event_sample_id=predictions.event_sample_id,
        event_subject_id=predictions.event_subject_id,
        event_true_start=predictions.event_true_start,
        event_true_end=predictions.event_true_end,
        event_pred_start=predictions.event_pred_start,
        event_pred_end=predictions.event_pred_end,
        event_status=predictions.event_status,
        event_duration_samples=predictions.event_duration_samples,
        legacy_evaluation=np.asarray(predictions.evaluation),
        legacy_fold=np.asarray(predictions.fold),
        cli_evaluation_zero_based=np.asarray(predictions.evaluation - 1),
        cli_fold_zero_based=np.asarray(predictions.fold - 1),
        id_convention=np.asarray("legacy_evaluation/fold are one-based; cli_* are zero-based"),
        detection_threshold=np.asarray(DETECTION_THRESHOLD),
        duration_threshold=np.asarray(DURATION_THRESHOLD),
    )
    return output_path


def evaluate_baseline(
    data_path: str | Path,
    weights_dir: str | Path,
    folds_path: str | Path,
    output_dir: str | Path | None = None,
    device: str = "auto",
    evaluation: int | None = None,
    fold: int | None = None,
) -> dict:
    """Evaluate one fold or all legacy folds with an unchanged event-F1 definition."""
    if (evaluation is None) != (fold is None):
        raise ValueError("Provide both evaluation and fold, or neither for the complete baseline.")
    folds = load_folds(folds_path)
    pairs = (
        [(evaluation, fold)]
        if evaluation is not None
        else [
            (int(eval_name.split("_")[1]), int(fold_name.split("_")[1]))
            for eval_name in sorted(folds)
            for fold_name in sorted(folds[eval_name])
        ]
    )
    all_iou_values = []
    fold_reports = []
    for eval_number, fold_number in pairs:
        predictions = evaluate_fold(
            data_path=data_path,
            weights_dir=weights_dir,
            folds_path=folds_path,
            evaluation=eval_number,
            fold=fold_number,
            device=device,
        )
        if output_dir is not None:
            prediction_path = save_fold_predictions(predictions, output_dir)
        else:
            prediction_path = None
        all_iou_values.extend(predictions.event_iou_values.tolist())
        fold_reports.append(
            {
                "evaluation": eval_number,
                "fold": fold_number,
                "samples": int(len(predictions.sample_ids)),
                "events": int(len(predictions.event_iou_values)),
                "prediction_file": str(prediction_path) if prediction_path else None,
            }
        )
    metric_curve = calculate_metrics_simple(all_iou_values)
    selected = metric_curve["selected_index"]
    report = {
        "f1_score": float(metric_curve["f1"][selected] * 100),
        "precision": float(metric_curve["precision"][selected] * 100),
        "recall": float(metric_curve["recall"][selected] * 100),
        "requested_iou_threshold": 0.2,
        "effective_iou_threshold": metric_curve["selected_iou_threshold"],
        "total_results": len(all_iou_values),
        "device": str(select_device(device)),
        "fold_reports": fold_reports,
    }
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "baseline_metrics.json").write_text(json.dumps(report, indent=2))
    return report
