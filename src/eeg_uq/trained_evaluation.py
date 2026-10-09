"""Prediction export for a checkpoint produced by ``train.py``.

It performs inference only.  The window stitching, legacy thresholds and event
post-processing live in :mod:`eeg_uq.evaluation`, so training exports and the
frozen baseline share one implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch

from .evaluation import predict_raw_records, select_device
from .folds import get_fold, load_folds, validate_folds
from .metrics import DETECTION_THRESHOLD, DURATION_THRESHOLD, calculate_metrics_simple
from .model import SEEDModel


SplitName = Literal["validation", "test"]


@dataclass
class TrainedSplitPredictions:
    evaluation: int  # zero-based public CLI index
    fold: int  # zero-based public CLI index
    split: SplitName
    seed: int
    best_epoch: int
    checkpoint_path: str
    checkpoint_sha256: str
    source_run_dir: str
    mask_source: str
    sample_ids: np.ndarray
    subject_ids: np.ndarray
    y_true: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
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


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trained_checkpoint(
    checkpoint_path: str | Path,
    *,
    evaluation: int,
    fold: int,
    device: torch.device,
) -> tuple[SEEDModel, dict]:
    """Load the best checkpoint and prove it belongs to the requested fold."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{checkpoint_path} is not the best_full.pt payload produced by train.py.")
    config = payload.get("config", {})
    if int(config.get("evaluation", -1)) != evaluation or int(config.get("fold", -1)) != fold:
        raise ValueError("Checkpoint evaluation/fold metadata does not match the requested split.")
    model = SEEDModel().to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), payload


def evaluate_trained_split(
    *,
    data_path: str | Path,
    folds_path: str | Path,
    checkpoint_path: str | Path,
    run_dir: str | Path,
    evaluation: int,
    fold: int,
    split: SplitName,
    device: str = "auto",
    progress: Callable[[int, int], None] | None = None,
) -> TrainedSplitPredictions:
    """Export raw validation or test predictions from a single trained SEED model."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    device_obj = select_device(device)
    checkpoint_path = Path(checkpoint_path).resolve()
    model, payload = load_trained_checkpoint(
        checkpoint_path, evaluation=evaluation, fold=fold, device=device_obj
    )
    folds = load_folds(folds_path)
    validation_subjects, test_subjects = get_fold(folds, evaluation + 1, fold + 1)
    selected_subjects = validation_subjects if split == "validation" else test_subjects
    with h5py.File(data_path, "r") as source:
        raw_subject_ids = np.asarray(source["subject_ids"][:])
        subject_ids = np.asarray(
            [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_subject_ids]
        )
    validate_folds(folds, set(subject_ids))
    sample_ids = np.flatnonzero(np.isin(subject_ids, selected_subjects))
    if not len(sample_ids):
        raise ValueError(f"No records found for eval {evaluation}, fold {fold} {split} subjects.")
    exported = predict_raw_records(
        data_path=data_path, model=model, sample_ids=sample_ids, device=device_obj, progress=progress
    )
    config = payload.get("config", {})
    return TrainedSplitPredictions(
        evaluation=evaluation,
        fold=fold,
        split=split,
        seed=int(config.get("seed", -1)),
        best_epoch=int(payload.get("epoch", -1)),
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256(checkpoint_path),
        source_run_dir=str(Path(run_dir).resolve()),
        mask_source=str(exported["mask_source"]),
        sample_ids=exported["sample_ids"], subject_ids=exported["subject_ids"],
        y_true=exported["y_true"], logits=exported["logits"], probabilities=exported["probabilities"],
        y_pred=exported["y_pred"], processed_pred=exported["processed_pred"],
        valid_mask=exported["valid_mask"], event_iou_values=exported["event_iou_values"],
        event_iou_offsets=exported["event_iou_offsets"], event_sample_id=exported["event_sample_id"],
        event_subject_id=exported["event_subject_id"], event_true_start=exported["event_true_start"],
        event_true_end=exported["event_true_end"], event_pred_start=exported["event_pred_start"],
        event_pred_end=exported["event_pred_end"], event_status=exported["event_status"],
        event_duration_samples=exported["event_duration_samples"],
    )


def save_trained_split_predictions(
    predictions: TrainedSplitPredictions, output_path: str | Path
) -> tuple[Path, Path]:
    """Save a conformal-ready NPZ and a small human-readable event report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_id=predictions.sample_ids, subject_id=predictions.subject_ids,
        y_true=predictions.y_true, logits=predictions.logits,
        probabilities=predictions.probabilities, y_pred=predictions.y_pred,
        processed_pred=predictions.processed_pred, valid_mask=predictions.valid_mask,
        event_iou_values=predictions.event_iou_values,
        event_iou_offsets=predictions.event_iou_offsets,
        event_sample_id=predictions.event_sample_id, event_subject_id=predictions.event_subject_id,
        event_true_start=predictions.event_true_start, event_true_end=predictions.event_true_end,
        event_pred_start=predictions.event_pred_start, event_pred_end=predictions.event_pred_end,
        event_status=predictions.event_status, event_duration_samples=predictions.event_duration_samples,
        evaluation=np.asarray(predictions.evaluation, dtype=np.int64),
        fold=np.asarray(predictions.fold, dtype=np.int64), seed=np.asarray(predictions.seed, dtype=np.int64),
        split=np.asarray(predictions.split), model_id=np.asarray(f"sha256:{predictions.checkpoint_sha256}"),
        checkpoint_sha256=np.asarray(predictions.checkpoint_sha256),
        checkpoint_path=np.asarray(predictions.checkpoint_path),
        source_run_dir=np.asarray(predictions.source_run_dir),
        best_epoch=np.asarray(predictions.best_epoch, dtype=np.int64), mask_source=np.asarray(predictions.mask_source),
        detection_threshold=np.asarray(DETECTION_THRESHOLD), duration_threshold=np.asarray(DURATION_THRESHOLD),
        inference_protocol=np.asarray("legacy 20 s windows / 10 s stride / central stitching"),
    )
    curve = calculate_metrics_simple(predictions.event_iou_values.tolist())
    selected = curve["selected_index"]
    report = {
        "evaluation": predictions.evaluation, "fold": predictions.fold, "split": predictions.split,
        "seed": predictions.seed, "best_epoch": predictions.best_epoch,
        "checkpoint_path": predictions.checkpoint_path, "checkpoint_sha256": predictions.checkpoint_sha256,
        "samples": int(len(predictions.sample_ids)), "subjects": int(len(np.unique(predictions.subject_ids))),
        "valid_timesteps": int(predictions.valid_mask.sum()), "mask_source": predictions.mask_source,
        "events": int(len(predictions.event_iou_values)),
        "f1_score": float(curve["f1"][selected] * 100),
        "precision": float(curve["precision"][selected] * 100),
        "recall": float(curve["recall"][selected] * 100),
        "requested_iou_threshold": 0.2, "effective_iou_threshold": curve["selected_iou_threshold"],
        "training_performed": False,
    }
    metrics_path = output_path.with_name(f"{predictions.split}_metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path, metrics_path


# Backward-compatible aliases for the original inference-only CLI.
TrainedTestPredictions = TrainedSplitPredictions


def evaluate_trained_test(**kwargs) -> TrainedSplitPredictions:
    return evaluate_trained_split(split="test", **kwargs)


def save_trained_test_predictions(predictions: TrainedSplitPredictions, output_path: str | Path) -> tuple[Path, Path]:
    return save_trained_split_predictions(predictions, output_path)
