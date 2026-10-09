"""Raw validation/test inference for Neural Process checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .metrics import (
    DETECTION_THRESHOLD,
    DURATION_THRESHOLD,
    calculate_metrics_simple,
    get_results_simple,
    legacy_event_records,
    process_spindle_simple,
)
from .neural_process import REDUCED_FACTOR, SeedNeuralProcess
from .np_training import load_neural_process_checkpoint
from .trained_evaluation import checkpoint_sha256
from .windowing import legacy_window_plan, split_windows, stitch_windows

INFERENCE_PROTOCOL = "legacy 20 s windows / 10 s stride / central stitching"


@dataclass
class NeuralProcessSplitPredictions:
    evaluation: int
    fold: int
    split: str
    seed: int
    best_epoch: int
    checkpoint_path: str
    checkpoint_sha256: str
    source_seed_checkpoint: str
    source_seed_sha256: str
    sample_ids: np.ndarray
    subject_ids: np.ndarray
    y_true: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictive_samples: np.ndarray
    predictive_std: np.ndarray
    seed_probabilities: np.ndarray
    y_pred: np.ndarray
    processed_pred: np.ndarray
    valid_mask: np.ndarray
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


def _append_event_records(
    *,
    sample_id: int,
    subject_id: str,
    labels: np.ndarray,
    processed: np.ndarray,
    iou_values: list[float],
    event_sample_ids: list[int],
    event_subject_ids: list[str],
    event_truth_start: list[int],
    event_truth_end: list[int],
    event_prediction_start: list[int],
    event_prediction_end: list[int],
    event_status: list[str],
    event_duration_samples: list[int],
) -> None:
    iou_values.extend(get_results_simple(labels.astype(int), processed.astype(int)))
    records = legacy_event_records(labels.astype(int), processed.astype(int))
    event_sample_ids.extend([sample_id] * len(records))
    event_subject_ids.extend([subject_id] * len(records))
    event_truth_start.extend(record["true_start"] for record in records)
    event_truth_end.extend(record["true_end"] for record in records)
    event_prediction_start.extend(record["pred_start"] for record in records)
    event_prediction_end.extend(record["pred_end"] for record in records)
    event_status.extend(record["status"] for record in records)
    event_duration_samples.extend(
        max(0, record["pred_end"] - record["pred_start"]) for record in records
    )


@torch.no_grad()
def predict_stitched_record(
    model: SeedNeuralProcess, signal: np.ndarray, *, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """NP latent samples and SEED logits for one raw record, at the reduced rate.

    The record is cut into the 20 s windows the NP was trained on and the
    central halves are stitched exactly like SEED's legacy inference.  Returns
    ``(n_z_samples, time / 8)`` probabilities and ``(time / 8,)`` SEED logits.
    """
    plan = legacy_window_plan(len(signal), resolution=REDUCED_FACTOR)
    windows = torch.from_numpy(split_windows(signal, plan)).to(device)
    samples, _, seed_logits = model.predict(windows)  # (n_z, windows, 500), (windows, 500)
    stitched_samples = stitch_windows(samples.permute(1, 2, 0), plan).T
    stitched_logits = stitch_windows(seed_logits, plan)
    return (
        stitched_samples.cpu().numpy().astype(np.float32),
        stitched_logits.cpu().numpy().astype(np.float32),
    )


def evaluate_neural_process_split(
    *,
    data_path: str | Path,
    checkpoint_path: str | Path,
    seed_checkpoint: str | Path,
    evaluation: int,
    fold: int,
    split: str,
    subjects: list[str],
    device: str = "auto",
    save_predictive_samples: bool = True,
) -> NeuralProcessSplitPredictions:
    """Run NP inference on a raw validation or test subject split."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    device_obj = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    model, payload = load_neural_process_checkpoint(
        checkpoint_path,
        seed_checkpoint=seed_checkpoint,
        evaluation=evaluation,
        fold=fold,
        device=device_obj,
    )
    model_checkpoint_sha = checkpoint_sha256(checkpoint_path)
    source_seed_checkpoint = str(Path(seed_checkpoint).resolve())
    source_seed_sha256 = checkpoint_sha256(seed_checkpoint)

    sample_ids: list[int] = []
    subject_ids: list[str] = []
    y_true: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    predictive_samples: list[np.ndarray] = []
    predictive_std: list[np.ndarray] = []
    seed_probabilities: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    processed_pred: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    iou_values: list[float] = []
    event_sample_ids: list[int] = []
    event_subject_ids: list[str] = []
    event_truth_start: list[int] = []
    event_truth_end: list[int] = []
    event_prediction_start: list[int] = []
    event_prediction_end: list[int] = []
    event_status: list[str] = []
    event_duration_samples: list[int] = []
    offsets = [0]

    with h5py.File(data_path, "r") as source:
        decoded = np.asarray(
            [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in source["subject_ids"][:]]
        )
        selected = np.flatnonzero(np.isin(decoded, subjects))
        if not len(selected):
            raise ValueError("Subject split did not select any HDF5 records.")
        mask_key = next((key for key in ("valid_mask", "annotation_mask") if key in source), None)
        for sample_id in selected.tolist():
            signal = np.asarray(source["signals"][sample_id], dtype=np.float32)
            target = np.asarray(source["labels"][sample_id]).astype(np.int8)
            samples_reduced, seed_logits_reduced = predict_stitched_record(model, signal, device=device_obj)
            samples_full = np.repeat(samples_reduced, REDUCED_FACTOR, axis=-1)
            mean_full = np.clip(samples_full.mean(axis=0), 1e-6, 1.0 - 1e-6)
            seed_full = 1.0 / (1.0 + np.exp(-np.repeat(seed_logits_reduced, REDUCED_FACTOR)))
            prediction_full = (mean_full >= DETECTION_THRESHOLD).astype(np.int8)
            duration_full = (mean_full >= DURATION_THRESHOLD).astype(np.int8)
            processed = process_spindle_simple(prediction_full, duration_full).astype(np.int8)
            valid_mask = (
                np.asarray(source[mask_key][sample_id], dtype=np.uint8)
                if mask_key is not None
                else np.ones(target.shape, dtype=np.uint8)
            )
            subject_id = str(decoded[sample_id])
            _append_event_records(
                sample_id=sample_id,
                subject_id=subject_id,
                labels=target,
                processed=processed,
                iou_values=iou_values,
                event_sample_ids=event_sample_ids,
                event_subject_ids=event_subject_ids,
                event_truth_start=event_truth_start,
                event_truth_end=event_truth_end,
                event_prediction_start=event_prediction_start,
                event_prediction_end=event_prediction_end,
                event_status=event_status,
                event_duration_samples=event_duration_samples,
            )
            offsets.append(len(iou_values))
            sample_ids.append(sample_id)
            subject_ids.append(subject_id)
            y_true.append(target)
            logits.append(np.log(mean_full / (1.0 - mean_full)))
            probabilities.append(mean_full)
            predictive_samples.append(samples_full)
            predictive_std.append(samples_full.std(axis=0).astype(np.float32))
            seed_probabilities.append(seed_full)
            y_pred.append(prediction_full)
            processed_pred.append(processed)
            valid_masks.append(valid_mask)

    sample_array = np.stack(predictive_samples).astype(np.float32)
    if not save_predictive_samples:
        sample_array = np.empty((0,), dtype=np.float32)
    return NeuralProcessSplitPredictions(
        evaluation=evaluation,
        fold=fold,
        split=split,
        seed=int(payload.get("config", {}).get("seed", -1)),
        best_epoch=int(payload.get("epoch", -1)),
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        checkpoint_sha256=model_checkpoint_sha,
        source_seed_checkpoint=source_seed_checkpoint,
        source_seed_sha256=source_seed_sha256,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        subject_ids=np.asarray(subject_ids, dtype="U"),
        y_true=np.stack(y_true).astype(np.int8),
        logits=np.stack(logits).astype(np.float32),
        probabilities=np.stack(probabilities).astype(np.float32),
        predictive_samples=sample_array,
        predictive_std=np.stack(predictive_std).astype(np.float32),
        seed_probabilities=np.stack(seed_probabilities).astype(np.float32),
        y_pred=np.stack(y_pred).astype(np.int8),
        processed_pred=np.stack(processed_pred).astype(np.int8),
        valid_mask=np.stack(valid_masks).astype(np.uint8),
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


def save_neural_process_predictions(
    predictions: NeuralProcessSplitPredictions,
    output_path: str | Path,
) -> tuple[Path, Path]:
    """Save conformal-ready predictions and an event-level report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_id=predictions.sample_ids,
        subject_id=predictions.subject_ids,
        y_true=predictions.y_true,
        logits=predictions.logits,
        probabilities=predictions.probabilities,
        predictive_samples=predictions.predictive_samples,
        predictive_std=predictions.predictive_std,
        seed_probabilities=predictions.seed_probabilities,
        y_pred=predictions.y_pred,
        processed_pred=predictions.processed_pred,
        valid_mask=predictions.valid_mask,
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
        evaluation=np.asarray(predictions.evaluation, dtype=np.int64),
        fold=np.asarray(predictions.fold, dtype=np.int64),
        seed=np.asarray(predictions.seed, dtype=np.int64),
        split=np.asarray(predictions.split),
        model_id=np.asarray(f"np-sha256:{predictions.checkpoint_sha256}"),
        checkpoint_sha256=np.asarray(predictions.checkpoint_sha256),
        checkpoint_path=np.asarray(predictions.checkpoint_path),
        source_seed_checkpoint=np.asarray(predictions.source_seed_checkpoint),
        source_seed_sha256=np.asarray(predictions.source_seed_sha256),
        best_epoch=np.asarray(predictions.best_epoch, dtype=np.int64),
        detection_threshold=np.asarray(DETECTION_THRESHOLD),
        duration_threshold=np.asarray(DURATION_THRESHOLD),
        inference_protocol=np.asarray(INFERENCE_PROTOCOL),
    )
    curve = calculate_metrics_simple(predictions.event_iou_values.tolist())
    selected = curve["selected_index"]
    report = {
        "evaluation": predictions.evaluation,
        "fold": predictions.fold,
        "split": predictions.split,
        "seed": predictions.seed,
        "best_epoch": predictions.best_epoch,
        "checkpoint_path": predictions.checkpoint_path,
        "checkpoint_sha256": predictions.checkpoint_sha256,
        "source_seed_checkpoint": predictions.source_seed_checkpoint,
        "source_seed_sha256": predictions.source_seed_sha256,
        "samples": int(len(predictions.sample_ids)),
        "subjects": int(len(np.unique(predictions.subject_ids))),
        "valid_timesteps": int(predictions.valid_mask.sum()),
        "events": int(len(predictions.event_iou_values)),
        "f1_score": float(curve["f1"][selected] * 100),
        "precision": float(curve["precision"][selected] * 100),
        "recall": float(curve["recall"][selected] * 100),
        "requested_iou_threshold": 0.2,
        "effective_iou_threshold": curve["selected_iou_threshold"],
        "predictive_samples": int(predictions.predictive_samples.shape[1])
        if predictions.predictive_samples.ndim == 3
        else 0,
        "training_performed": True,
    }
    metrics_path = output_path.with_name(f"{predictions.split}_metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path, metrics_path
