"""Post-hoc last-layer Laplace Redux for the already-trained SEED checkpoint.

The legacy SEED model is never modified on disk.  A frozen in-memory copy
supplies its penultimate features (the 128-d input of ``classifier.logits``),
computed with the same 20 s / 10 s central stitching used at inference.
:class:`SeedLastLayerHead` is SEED's real last layer rewritten as a two-logit
linear layer that is exact at the MAP: ``softmax([0, z])[1] == sigmoid(z)``.
Laplace is fitted only to that layer.

The amount of evidence the posterior sees is kept honest:

- it is fitted on the raw training records, not on the augmented copies;
- one example per SEED output (the reduced rate), not per repeated sample;
- the likelihood is tempered by the integrated autocorrelation time ``tau`` of
  the SEED logit, so ``N`` correlated timesteps count as ``N / tau``
  independent observations.

Validation is used only to choose the prior precision (validation NLL); test
is inference-only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import entr
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation import select_device
from ..folds import get_fold, load_folds, validate_folds
from ..metrics import (
    DETECTION_THRESHOLD,
    DURATION_THRESHOLD,
    SAMPLE_RATE_HZ,
    legacy_event_records,
    process_spindle_simple,
)
from ..model import RESOL_FACTOR, SEEDModel
from ..trained_evaluation import load_trained_checkpoint
from ..windowing import legacy_window_plan, split_windows, stitch_windows
SplitName = Literal["validation", "test"]
FIT_BATCH_POINTS = 65536


class SeedLastLayerHead(nn.Module):
    """SEED's ``classifier.logits`` (128 -> 1) as a two-class linear layer.

    Weights ``[0; w]`` and bias ``[0; b]`` give logits ``[0, z]`` with ``z`` the
    original SEED logit, so the categorical likelihood in ``laplace-torch``
    equals SEED's BCE model.  It takes precomputed penultimate features.
    """

    def __init__(self, logits_layer: nn.Linear) -> None:
        super().__init__()
        self.last_layer = nn.Linear(logits_layer.in_features, 2, bias=True)
        with torch.no_grad():
            self.last_layer.weight.zero_()
            self.last_layer.bias.zero_()
            self.last_layer.weight[1].copy_(logits_layer.weight[0])
            self.last_layer.bias[1].copy_(logits_layer.bias[0])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.last_layer(features)


def seed_penultimate_features(model: SEEDModel, signal: torch.Tensor) -> torch.Tensor:
    """Return SEED's input to ``classifier.logits``: ``(batch, time / 8, 128)``."""
    classifier = model.classifier
    hidden = model.transformer(model.embedding(signal))
    return torch.relu(classifier.hidden(classifier.dropout(hidden)))


@torch.no_grad()
def stitched_penultimate_features(
    model: SEEDModel,
    signal: np.ndarray,
    *,
    device: torch.device,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: int = 20,
    stride_seconds: int = 10,
) -> torch.Tensor:
    """Legacy 20 s / 10 s central stitching, applied at SEED's reduced rate.

    Window borders are multiples of ``RESOL_FACTOR``, so stitching features and
    repeating the resulting logits x8 equals stitching SEED's full-rate output.
    Returns ``(len(signal) / RESOL_FACTOR, features)`` on ``device``.
    """
    plan = legacy_window_plan(
        len(signal),
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        resolution=RESOL_FACTOR,
    )
    windows = torch.from_numpy(split_windows(signal, plan)).to(device)
    return stitch_windows(seed_penultimate_features(model, windows), plan)


def integrated_autocorrelation_time(
    sequences: Iterable[np.ndarray], *, max_lag: int = 400, cutoff: float = 0.05
) -> float:
    """``tau = 1 + 2 * sum(rho_k)`` up to the first lag where ``rho_k < cutoff``.

    ``N`` timesteps with this autocorrelation carry about as much information
    as ``N / tau`` independent ones.
    """
    centred = [np.asarray(values, dtype=np.float64) - np.mean(values) for values in sequences]
    variance = sum(float(np.sum(values**2)) for values in centred) / sum(len(values) for values in centred)
    tau = 1.0
    for lag in range(1, max_lag + 1):
        pairs = [(values[:-lag], values[lag:]) for values in centred if len(values) > lag]
        if not pairs:
            break
        rho = sum(float(np.sum(a * b)) for a, b in pairs) / sum(len(a) for a, _ in pairs) / variance
        if rho < cutoff:
            break
        tau += 2.0 * rho
    return tau


@dataclass
class FeatureSet:
    features: torch.Tensor  # (points, 128), CPU float32
    labels: torch.Tensor  # (points,), majority label of each reduced timestep
    seed_logits: list[np.ndarray]  # one SEED logit sequence per record
    records: int


@torch.no_grad()
def build_feature_set(
    model: SEEDModel,
    head: SeedLastLayerHead,
    data_path: str | Path,
    subjects: Iterable[str],
    *,
    device: torch.device,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: int = 20,
    stride_seconds: int = 10,
) -> FeatureSet:
    """Stitched SEED features and labels at the reduced rate for raw records."""
    subject_ids = _decoded_subject_ids(data_path)
    sample_ids = np.flatnonzero(np.isin(subject_ids, sorted(set(subjects))))
    if not len(sample_ids):
        raise ValueError("Subject split did not select any HDF5 records.")
    weight, bias = head.last_layer.weight[1], head.last_layer.bias[1]
    features, labels, logits = [], [], []
    with h5py.File(data_path, "r") as source:
        mask_key = next((key for key in ("valid_mask", "annotation_mask") if key in source), None)
        for sample_id in sample_ids:
            record_features = stitched_penultimate_features(
                model,
                source["signals"][sample_id],
                device=device,
                sample_rate_hz=sample_rate_hz,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
            )
            truth = np.asarray(source["labels"][sample_id]).reshape(-1, RESOL_FACTOR)
            reduced_label = truth.mean(axis=1) >= 0.5
            keep = (
                np.asarray(source[mask_key][sample_id], dtype=bool).reshape(-1, RESOL_FACTOR).all(axis=1)
                if mask_key is not None
                else np.ones(len(reduced_label), dtype=bool)
            )
            logits.append((record_features @ weight + bias).cpu().numpy())
            keep_tensor = torch.from_numpy(keep).to(record_features.device)
            features.append(record_features[keep_tensor].cpu())
            labels.append(torch.from_numpy(reduced_label[keep].astype(np.int64)))
    return FeatureSet(
        features=torch.cat(features),
        labels=torch.cat(labels),
        seed_logits=logits,
        records=int(len(sample_ids)),
    )


@dataclass
class LaplaceFit:
    laplace: Any
    base_model: SEEDModel
    checkpoint_path: Path
    checkpoint_sha256: str
    payload: dict[str, Any]
    train_records: int
    train_points: int
    temperature: float
    autocorrelation_time: float


def _require_laplace() -> Any:
    try:
        from laplace import Laplace
    except ImportError as error:  # pragma: no cover - exercised in clean envs
        raise RuntimeError(
            "laplace-torch is required. Install requirements.txt in the project venv."
        ) from error
    return Laplace


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decoded_subject_ids(data_path: str | Path) -> np.ndarray:
    with h5py.File(data_path, "r") as source:
        raw = np.asarray(source["subject_ids"][:])
    return np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw])


def _split_subjects(
    data_path: str | Path,
    folds_path: str | Path,
    evaluation: int,
    fold: int,
) -> tuple[list[str], list[str], list[str]]:
    subjects = _decoded_subject_ids(data_path)
    folds = load_folds(folds_path)
    validation_subjects, test_subjects = get_fold(folds, evaluation + 1, fold + 1)
    validate_folds(folds, set(subjects))
    train_subjects = sorted(set(subjects) - set(validation_subjects) - set(test_subjects))
    if not train_subjects or not validation_subjects or not test_subjects:
        raise ValueError("Train, validation and test subject splits must all be non-empty.")
    return train_subjects, validation_subjects, test_subjects


def _validation_nll(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # A plain callable, so laplace-torch recomputes it from scratch for every
    # grid point (its default running metric is never reset between points).
    return F.nll_loss(probabilities.clamp_min(1e-12).log(), targets)


def fit_laplace_posterior(
    *,
    checkpoint_path: str | Path,
    raw_h5: str | Path,
    folds_path: str | Path,
    evaluation: int,
    fold: int,
    device: str,
    prior_precision: float,
    likelihood: str = "classification",
    subset_of_weights: str = "last_layer",
    hessian_structure: str = "diag",
    temperature: float | str = "auto",
    optimize_prior: bool = False,
    prior_tuning_method: str = "gridsearch",
    prior_tuning_steps: int = 25,
    prior_tuning_lr: float = 0.1,
    prior_grid_size: int = 15,
    prior_log_min: float = -4.0,
    prior_log_max: float = 4.0,
    mc_samples: int = 30,
    progress_bar: bool = False,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: int = 20,
    stride_seconds: int = 10,
) -> LaplaceFit:
    """Fit a last-layer Laplace posterior around the frozen SEED checkpoint."""

    if likelihood != "classification":
        raise ValueError("This binary SEED head currently requires likelihood='classification'.")
    if subset_of_weights != "last_layer":
        raise ValueError("This implementation currently supports subset_of_weights='last_layer' only.")
    Laplace = _require_laplace()
    checkpoint_path = Path(checkpoint_path).resolve()
    raw_h5 = Path(raw_h5).resolve()
    device_obj = select_device(device)
    base_model, payload = load_trained_checkpoint(
        checkpoint_path, evaluation=evaluation, fold=fold, device=device_obj
    )
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    head = SeedLastLayerHead(base_model.classifier.logits).to(device_obj).eval()
    train_subjects, validation_subjects, _ = _split_subjects(
        raw_h5, Path(folds_path).resolve(), evaluation, fold
    )
    geometry = dict(
        device=device_obj,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )
    train = build_feature_set(base_model, head, raw_h5, train_subjects, **geometry)
    tau = integrated_autocorrelation_time(train.seed_logits)
    likelihood_temperature = tau if temperature == "auto" else float(temperature)

    laplace = Laplace(
        head,
        likelihood,
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
        last_layer_name="last_layer",
        prior_precision=float(prior_precision),
        temperature=float(likelihood_temperature),
    )
    laplace.fit(
        DataLoader(TensorDataset(train.features, train.labels), batch_size=FIT_BATCH_POINTS),
        progress_bar=progress_bar,
    )

    if optimize_prior:
        kwargs: dict[str, Any] = {
            "pred_type": "nn",
            "method": prior_tuning_method,
            "init_prior_prec": float(prior_precision),
            "n_steps": int(prior_tuning_steps),
            "lr": float(prior_tuning_lr),
            "grid_size": int(prior_grid_size),
            "log_prior_prec_min": float(prior_log_min),
            "log_prior_prec_max": float(prior_log_max),
            "link_approx": "mc",
            "n_samples": int(mc_samples),
            "progress_bar": progress_bar,
        }
        if prior_tuning_method == "gridsearch":
            validation = build_feature_set(base_model, head, raw_h5, validation_subjects, **geometry)
            kwargs["val_loader"] = DataLoader(
                TensorDataset(validation.features, validation.labels), batch_size=FIT_BATCH_POINTS
            )
            kwargs["loss"] = _validation_nll
        laplace.optimize_prior_precision(**kwargs)

    return LaplaceFit(
        laplace=laplace,
        base_model=base_model,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256(checkpoint_path),
        payload=payload,
        train_records=train.records,
        train_points=int(len(train.labels)),
        temperature=float(likelihood_temperature),
        autocorrelation_time=float(tau),
    )


def save_laplace_posterior(fit: LaplaceFit, output_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Save only the posterior artifact and provenance, never the SEED checkpoint."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "laplace_state": fit.laplace.state_dict(),
        "metadata": {
            **metadata,
            "checkpoint_path": str(fit.checkpoint_path),
            "checkpoint_sha256": fit.checkpoint_sha256,
            "fit_data": "raw records (moda_moda.h5), reduced SEED rate",
            "train_records": fit.train_records,
            "train_points": fit.train_points,
            "likelihood_temperature": fit.temperature,
            "logit_autocorrelation_time": fit.autocorrelation_time,
            "effective_train_points": fit.train_points / fit.temperature,
            "seed_checkpoint_epoch": int(fit.payload.get("epoch", -1)),
            "training_performed": False,
        },
    }
    torch.save(payload, output_path)
    return output_path


@torch.no_grad()
def infer_full_signal_predictive_samples(
    laplace: Any,
    base_model: SEEDModel,
    signal: np.ndarray,
    *,
    device: torch.device,
    n_samples: int,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: int = 20,
    stride_seconds: int = 10,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Positive-class posterior samples ``(n_samples, len(signal))``.

    Features are stitched once per record, so every posterior draw uses the
    same last-layer weights across the whole record.
    """

    features = stitched_penultimate_features(
        base_model,
        signal,
        device=device,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )
    samples = laplace.predictive_samples(
        features, pred_type="nn", n_samples=int(n_samples), generator=generator
    )
    positive = samples[..., 1].repeat_interleave(RESOL_FACTOR, dim=-1)
    return positive.detach().cpu().numpy().astype(np.float32, copy=False)


def _binary_entropy(probability: np.ndarray, epsilon: float) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return entr(probability) + entr(1.0 - probability)  # nats


def _ece(probabilities: np.ndarray, truth: np.ndarray, mask: np.ndarray, bins: int) -> float:
    p = np.asarray(probabilities, dtype=float)[mask]
    y = np.asarray(truth, dtype=np.int8)[mask]
    if not len(p):
        return float("nan")
    confidence = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(np.int8) == y).astype(float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    value = 0.0
    for index in range(int(bins)):
        selected = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index + 1 < int(bins) else confidence <= edges[index + 1]
        )
        if selected.any():
            value += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(value)


def compute_laplace_uncertainty(
    predictive_samples: np.ndarray,
    y_true: np.ndarray,
    valid_mask: np.ndarray,
    *,
    epsilon: float = 1e-6,
    ece_bins: int = 15,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    """Compute predictive entropy, BALD/MI, NLL and ECE on valid timesteps."""

    samples = np.asarray(predictive_samples)
    truth = np.asarray(y_true, dtype=np.int8)
    mask = np.asarray(valid_mask, dtype=bool)
    if samples.ndim != 3 or samples.shape[1:] != truth.shape or truth.shape != mask.shape:
        raise ValueError("predictive_samples must have shape (samples, records, time) matching y_true.")
    if not samples.shape[0] or not mask.any():
        raise ValueError("At least one posterior sample and one valid timestep are required.")
    mean_probability = samples.mean(axis=0, dtype=np.float64)
    predictive_entropy = _binary_entropy(mean_probability, epsilon)
    # Chunk over records: a float64 copy of every sample of a whole split
    # (samples x records x time) does not fit in memory on raw MODA splits.
    expected_entropy = np.empty(truth.shape, dtype=np.float64)
    for start in range(0, truth.shape[0], 16):
        stop = start + 16
        expected_entropy[start:stop] = _binary_entropy(samples[:, start:stop], epsilon).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    clipped = np.clip(mean_probability, epsilon, 1.0 - epsilon)
    nll_values = -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))
    metrics: dict[str, float | int] = {
        "n_valid_timesteps": int(mask.sum()),
        "posterior_samples": int(samples.shape[0]),
        "mean_predictive_entropy": float(predictive_entropy[mask].mean()),
        "mean_mutual_information": float(mutual_information[mask].mean()),
        "nll": float(nll_values[mask].mean()),
        "ece": _ece(mean_probability, truth, mask, ece_bins),
        "ece_bins": int(ece_bins),
    }
    arrays = {
        "mean_probability": mean_probability.astype(np.float32),
        "predictive_entropy": predictive_entropy.astype(np.float32),
        "expected_entropy": expected_entropy.astype(np.float32),
        "mutual_information": mutual_information.astype(np.float32),
        "nll_values": nll_values.astype(np.float32),
    }
    return metrics, arrays


def _legacy_outputs(
    mean_probability: np.ndarray,
    *,
    detection_threshold: float,
    duration_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    y_pred = (mean_probability >= detection_threshold).astype(np.int8)
    duration_pred = (mean_probability >= duration_threshold).astype(np.int8)
    processed = process_spindle_simple(y_pred, duration_pred).astype(np.int8)
    return y_pred, processed, {
        "duration_pred": duration_pred,
    }


def summarize_records(
    records: Iterable[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]],
    *,
    mask_source: str,
    detection_threshold: float = DETECTION_THRESHOLD,
    duration_threshold: float = DURATION_THRESHOLD,
) -> dict[str, np.ndarray | str]:
    """Legacy detection and events from ``(sample_id, subject, truth, valid, samples)``.

    ``samples`` is ``(samples, time)``; detection uses their mean with the
    legacy thresholds and post-processing.
    """

    sample_ids, samples, truths, masks, subjects = [], [], [], [], []
    y_preds, processed_preds = [], []
    event_values: list[float] = []
    event_sample_ids: list[int] = []
    event_subject_ids: list[str] = []
    event_true_start: list[int] = []
    event_true_end: list[int] = []
    event_pred_start: list[int] = []
    event_pred_end: list[int] = []
    event_status: list[str] = []
    for sample_id, subject, truth, valid, predictive in records:
        if predictive.shape[1:] != truth.shape or valid.shape != truth.shape:
            raise ValueError("Predictive samples, labels and valid mask have incompatible shapes.")
        y_pred, processed, _ = _legacy_outputs(
            predictive.mean(axis=0),
            detection_threshold=detection_threshold,
            duration_threshold=duration_threshold,
        )
        sample_ids.append(int(sample_id))
        samples.append(predictive)
        truths.append(truth)
        masks.append(valid)
        subjects.append(str(subject))
        y_preds.append(y_pred)
        processed_preds.append(processed)
        events = legacy_event_records(truth.astype(int), processed.astype(int))
        event_values.extend(float(record["iou"]) for record in events)
        event_sample_ids.extend([int(sample_id)] * len(events))
        event_subject_ids.extend([str(subject)] * len(events))
        event_true_start.extend(int(record["true_start"]) for record in events)
        event_true_end.extend(int(record["true_end"]) for record in events)
        event_pred_start.extend(int(record["pred_start"]) for record in events)
        event_pred_end.extend(int(record["pred_end"]) for record in events)
        event_status.extend(str(record["status"]) for record in events)
    # Each record contributes (samples, time); expose the public artifact as
    # (samples, records, time) for uncertainty metrics.
    return {
        "sample_id": np.asarray(sample_ids, dtype=np.int64),
        "subject_id": np.asarray(subjects, dtype="U"),
        "y_true": np.stack(truths).astype(np.int8),
        "valid_mask": np.stack(masks).astype(np.uint8),
        "predictive_probabilities": np.stack(samples, axis=1).astype(np.float32),
        "y_pred": np.stack(y_preds).astype(np.int8),
        "processed_pred": np.stack(processed_preds).astype(np.int8),
        "event_iou_values": np.asarray(event_values, dtype=np.float32),
        "event_sample_id": np.asarray(event_sample_ids, dtype=np.int64),
        "event_subject_id": np.asarray(event_subject_ids, dtype="U"),
        "event_true_start": np.asarray(event_true_start, dtype=np.int64),
        "event_true_end": np.asarray(event_true_end, dtype=np.int64),
        "event_pred_start": np.asarray(event_pred_start, dtype=np.int64),
        "event_pred_end": np.asarray(event_pred_end, dtype=np.int64),
        "event_status": np.asarray(event_status, dtype="U16"),
        "mask_source": mask_source,
    }


def evaluate_raw_records(
    predict_samples: Callable[[np.ndarray], np.ndarray],
    *,
    data_path: str | Path,
    sample_ids: np.ndarray,
    detection_threshold: float = DETECTION_THRESHOLD,
    duration_threshold: float = DURATION_THRESHOLD,
) -> dict[str, np.ndarray | str]:
    """Run any sample-based predictor over raw records for one split.

    ``predict_samples(signal)`` returns ``(samples, time)`` positive-class
    probabilities for one record (posterior draws, ensemble members, ...).
    """

    with h5py.File(data_path, "r") as source:
        raw_subject_ids = np.asarray(source["subject_ids"][:])
        decoded = np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_subject_ids])
        mask_key = next((key for key in ("valid_mask", "annotation_mask") if key in source), None)

        def records():
            for sample_id in np.asarray(sample_ids, dtype=np.int64):
                signal = np.asarray(source["signals"][sample_id], dtype=np.float32)
                truth = np.asarray(source["labels"][sample_id], dtype=np.int8)
                valid = (
                    np.asarray(source[mask_key][sample_id], dtype=np.uint8)
                    if mask_key is not None
                    else np.ones(truth.shape, dtype=np.uint8)
                )
                yield sample_id, decoded[sample_id], truth, valid, predict_samples(signal)

        return summarize_records(
            records(),
            mask_source=mask_key or "all_timesteps (no HDF5 mask)",
            detection_threshold=detection_threshold,
            duration_threshold=duration_threshold,
        )


def evaluate_raw_split(
    *,
    laplace: Any,
    base_model: SEEDModel,
    data_path: str | Path,
    sample_ids: np.ndarray,
    device: torch.device,
    n_samples: int,
    seed: int,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: int = 20,
    stride_seconds: int = 10,
    detection_threshold: float = DETECTION_THRESHOLD,
    duration_threshold: float = DURATION_THRESHOLD,
) -> dict[str, np.ndarray | str]:
    """Run Laplace posterior-sample inference over raw records for one split."""

    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))

    def predict_samples(signal: np.ndarray) -> np.ndarray:
        return infer_full_signal_predictive_samples(
            laplace,
            base_model,
            signal,
            device=device,
            n_samples=n_samples,
            sample_rate_hz=sample_rate_hz,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            generator=rng,
        )

    return evaluate_raw_records(
        predict_samples,
        data_path=data_path,
        sample_ids=sample_ids,
        detection_threshold=detection_threshold,
        duration_threshold=duration_threshold,
    )


def save_split_predictions(
    predictions: dict[str, np.ndarray | str],
    output_path: str | Path,
    *,
    save_predictive_samples: bool,
    epsilon: float,
    ece_bins: int,
) -> tuple[Path, dict[str, float | int]]:
    """Save posterior samples and derived uncertainty arrays for one split."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(predictions["predictive_probabilities"], dtype=np.float32)
    truth = np.asarray(predictions["y_true"])
    mask = np.asarray(predictions["valid_mask"], dtype=bool)
    metrics, arrays = compute_laplace_uncertainty(
        samples, truth, mask, epsilon=epsilon, ece_bins=ece_bins
    )
    saved: dict[str, np.ndarray] = {
        key: np.asarray(value)
        for key, value in predictions.items()
        if key != "predictive_probabilities" and not isinstance(value, str)
    }
    saved.update(arrays)
    if save_predictive_samples:
        saved["predictive_probabilities"] = samples.astype(np.float32)
    saved["mask_source"] = np.asarray(str(predictions["mask_source"]))
    np.savez_compressed(output_path, **saved)
    return output_path, metrics


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
