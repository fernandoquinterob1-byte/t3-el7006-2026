"""Shared fold runner for sample-based methods built on a saved SEED checkpoint.

A method supplies ``make_predictor(model, device) -> predict_samples`` where
``predict_samples(signal)`` returns ``(samples, time)`` probabilities for one
raw record.  Validation and test are evaluated with the legacy thresholds and
saved in the Laplace NPZ layout, so the same analysis code applies.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..evaluation import select_device
from ..model import SEEDModel
from ..trained_evaluation import checkpoint_sha256, load_trained_checkpoint
from .laplace_redux import _decoded_subject_ids, _split_subjects, evaluate_raw_records, json_safe, save_split_predictions

PredictorFactory = Callable[[SEEDModel, torch.device], Callable[[np.ndarray], np.ndarray]]


def resolve_repo_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def run_sampling_fold(
    *,
    repo_root: Path,
    config: dict[str, Any],
    section: str,
    paths: Any,
    evaluation: int,
    fold: int,
    make_predictor: PredictorFactory,
    report_extra: dict[str, Any],
    output_root: Path | None = None,
    no_save_samples: bool = False,
) -> dict[str, Any]:
    method_config = config[section]
    inference = method_config["inference"]
    uncertainty_config = config["uncertainty"]
    checkpoint = Path(
        str(method_config["checkpoint_template"]).format(
            seed_run_root=resolve_repo_path(repo_root, method_config["seed_run_root"]),
            evaluation=evaluation,
            fold=fold,
            seed=int(config["seed"]),
        )
    ).resolve()
    device = select_device(str(method_config["device"]))
    model, payload = load_trained_checkpoint(checkpoint, evaluation=evaluation, fold=fold, device=device)
    predict_samples = make_predictor(model, device)
    root = output_root or resolve_repo_path(repo_root, method_config["output_root"])
    output_dir = root / f"eval_{evaluation + 1}_fold_{fold + 1}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _, validation_subjects, test_subjects = _split_subjects(paths.raw_h5, paths.folds, evaluation, fold)
    subject_ids = _decoded_subject_ids(paths.raw_h5)

    split_metrics: dict[str, dict[str, float | int]] = {}
    for split, subjects in (("validation", validation_subjects), ("test", test_subjects)):
        print(f"{section} eval {evaluation} fold {fold} {split}: {len(subjects)} subjects...", flush=True)
        predictions = evaluate_raw_records(
            predict_samples,
            data_path=paths.raw_h5,
            sample_ids=np.flatnonzero(np.isin(subject_ids, subjects)),
            detection_threshold=float(inference["detection_threshold"]),
            duration_threshold=float(inference["duration_threshold"]),
        )
        _, split_metrics[split] = save_split_predictions(
            predictions,
            output_dir / f"{split}_predictions.npz",
            save_predictive_samples=bool(uncertainty_config["save_predictive_samples"]) and not no_save_samples,
            epsilon=float(uncertainty_config["epsilon"]),
            ece_bins=int(uncertainty_config["ece_bins"]),
        )

    report = {
        "evaluation": evaluation,
        "fold": fold,
        **report_extra,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "seed_checkpoint_epoch": int(payload.get("epoch", -1)),
        "training_performed": False,
        "validation_uncertainty": split_metrics["validation"],
        "test_uncertainty": split_metrics["test"],
        "artifacts": {
            "validation_predictions": str(output_dir / "validation_predictions.npz"),
            "test_predictions": str(output_dir / "test_predictions.npz"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
