#!/usr/bin/env python3
"""Fit last-layer Laplace Redux on saved SEED checkpoints and evaluate it.

This command is inference/posterior fitting only.  It never trains SEED and
never overwrites the source run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.config import DEFAULT_CONFIG_PATH, configured, load_project_config  # noqa: E402
from eeg_uq.paths import resolve_data_paths, validate_data_paths  # noqa: E402
from eeg_uq.uncertainty.laplace_redux import (  # noqa: E402
    _decoded_subject_ids,
    _split_subjects,
    evaluate_raw_split,
    fit_laplace_posterior,
    json_safe,
    save_laplace_posterior,
    save_split_predictions,
)


def _resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--evaluation", type=int, choices=range(3), help="Optional zero-based override.")
    parser.add_argument("--fold", type=int, choices=range(5), help="Optional zero-based override.")
    parser.add_argument("--data-root", type=Path, help="Optional override; YAML/env is used otherwise.")
    parser.add_argument("--folds-path", type=Path, help="Optional override; YAML default is used otherwise.")
    parser.add_argument("--checkpoint", type=Path, help="Optional override for one saved best_full.pt.")
    parser.add_argument("--no-save-samples", action="store_true", help="Do not store the full MC sample tensor.")
    return parser


def _checkpoint_path(
    laplace_config: dict[str, Any],
    *,
    seed: int,
    evaluation: int,
    fold: int,
    override: Path | None,
) -> Path:
    if override is not None:
        return override.resolve()
    template = str(laplace_config["checkpoint_template"])
    seed_run_root = str(_resolve_repo_path(laplace_config["seed_run_root"]))
    rendered = template.format(
        seed=seed,
        evaluation=evaluation,
        fold=fold,
        seed_run_root=seed_run_root,
    )
    path = Path(rendered).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _sample_ids(data_path: Path, subjects: list[str]) -> np.ndarray:
    subject_ids = _decoded_subject_ids(data_path)
    return np.flatnonzero(np.isin(subject_ids, subjects)).astype(np.int64)


def _run_one(
    *,
    config: dict[str, Any],
    paths: Any,
    evaluation: int,
    fold: int,
    checkpoint_override: Path | None,
    no_save_samples: bool,
) -> dict[str, Any]:
    laplace_config = config["laplace"]
    uncertainty_config = config["uncertainty"]
    seed = int(config["seed"])
    checkpoint = _checkpoint_path(
        laplace_config,
        seed=seed,
        evaluation=evaluation,
        fold=fold,
        override=checkpoint_override,
    )
    output_root = _resolve_repo_path(laplace_config["output_root"])
    output_dir = output_root / f"eval_{evaluation + 1}_fold_{fold + 1}"
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_config = laplace_config["inference"]
    train_subjects, validation_subjects, test_subjects = _split_subjects(
        paths.raw_h5, paths.folds, evaluation, fold
    )
    fit = fit_laplace_posterior(
        checkpoint_path=checkpoint,
        raw_h5=paths.raw_h5,
        folds_path=paths.folds,
        evaluation=evaluation,
        fold=fold,
        device=str(laplace_config["device"]),
        prior_precision=float(laplace_config["prior_precision"]),
        likelihood=str(laplace_config["likelihood"]),
        subset_of_weights=str(laplace_config["subset_of_weights"]),
        hessian_structure=str(laplace_config["hessian_structure"]),
        temperature=laplace_config["temperature"],
        optimize_prior=bool(laplace_config["prior_tuning"]["enabled"]),
        prior_tuning_method=str(laplace_config["prior_tuning"]["method"]),
        prior_tuning_steps=int(laplace_config["prior_tuning"]["steps"]),
        prior_tuning_lr=float(laplace_config["prior_tuning"]["learning_rate"]),
        prior_grid_size=int(laplace_config["prior_tuning"]["grid_size"]),
        prior_log_min=float(laplace_config["prior_tuning"]["log_min"]),
        prior_log_max=float(laplace_config["prior_tuning"]["log_max"]),
        mc_samples=int(laplace_config["mc_samples"]),
        progress_bar=bool(laplace_config["progress_bar"]),
        sample_rate_hz=int(inference_config["sample_rate_hz"]),
        window_seconds=int(inference_config["window_seconds"]),
        stride_seconds=int(inference_config["stride_seconds"]),
    )
    print(
        f"Laplace fit: {fit.train_points} puntos de train, temperatura {fit.temperature:.1f} "
        f"(~{fit.train_points / fit.temperature:.0f} efectivos), "
        f"prior {float(fit.laplace.prior_precision.reshape(-1)[0]):.3g}",
        flush=True,
    )
    posterior_path = save_laplace_posterior(
        fit,
        output_dir / "laplace_posterior.pt",
        metadata={
            "evaluation": evaluation,
            "fold": fold,
            "subset_of_weights": laplace_config["subset_of_weights"],
            "hessian_structure": laplace_config["hessian_structure"],
            "likelihood": laplace_config["likelihood"],
            "prior_precision": float(fit.laplace.prior_precision.reshape(-1)[0].item()),
        },
    )
    raw_data_path = paths.raw_h5
    device = fit.laplace._device
    split_predictions: dict[str, dict[str, np.ndarray | str]] = {}
    split_metrics: dict[str, dict[str, float | int]] = {}
    for split, subjects in (("validation", validation_subjects), ("test", test_subjects)):
        print(f"Laplace {split}: loading {len(subjects)} subjects...", flush=True)
        predictions = evaluate_raw_split(
            laplace=fit.laplace,
            base_model=fit.base_model,
            data_path=raw_data_path,
            sample_ids=_sample_ids(raw_data_path, subjects),
            device=device,
            n_samples=int(laplace_config["mc_samples"]),
            seed=int(laplace_config["random_seed"]) + evaluation * 100 + fold,
            sample_rate_hz=int(inference_config["sample_rate_hz"]),
            window_seconds=int(inference_config["window_seconds"]),
            stride_seconds=int(inference_config["stride_seconds"]),
            detection_threshold=float(inference_config["detection_threshold"]),
            duration_threshold=float(inference_config["duration_threshold"]),
        )
        split_predictions[split] = predictions
        _, split_metrics[split] = save_split_predictions(
            predictions,
            output_dir / f"{split}_predictions.npz",
            save_predictive_samples=bool(uncertainty_config["save_predictive_samples"])
            and not no_save_samples,
            epsilon=float(uncertainty_config["epsilon"]),
            ece_bins=int(uncertainty_config["ece_bins"]),
        )

    metrics_report = {
        "evaluation": evaluation,
        "fold": fold,
        "checkpoint": str(checkpoint),
        "posterior": str(posterior_path),
        "source_checkpoint_unchanged": True,
        "train_subjects": len(train_subjects),
        "validation_subjects": len(validation_subjects),
        "test_subjects": len(test_subjects),
        "split_policy": "raw train records fit Laplace; validation is used only for optional prior tuning; test is held out",
        "fit": {
            "train_points": fit.train_points,
            "likelihood_temperature": fit.temperature,
            "logit_autocorrelation_time": fit.autocorrelation_time,
            "effective_train_points": fit.train_points / fit.temperature,
            "prior_precision": float(fit.laplace.prior_precision.reshape(-1)[0].item()),
        },
        "validation_uncertainty": split_metrics["validation"],
        "test_uncertainty": split_metrics["test"],
        "artifacts": {
            "validation_predictions": str(output_dir / "validation_predictions.npz"),
            "test_predictions": str(output_dir / "test_predictions.npz"),
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(json_safe(metrics_report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics_report


def main() -> None:
    args = _parser().parse_args()
    config = load_project_config(args.config)
    laplace_config = config["laplace"]
    paths_config = config["paths"]
    data_root = args.data_root or _resolve_repo_path(paths_config.get("data_root"))
    folds_path = args.folds_path or _resolve_repo_path(paths_config.get("folds_path"))
    paths = resolve_data_paths(data_root, folds_path)
    validate_data_paths(paths, require_raw=True)
    evaluations = [args.evaluation] if args.evaluation is not None else [int(value) for value in laplace_config["evaluations"]]
    folds = [args.fold] if args.fold is not None else [int(value) for value in laplace_config["folds"]]
    checkpoint_override = args.checkpoint.resolve() if args.checkpoint else None
    reports = []
    for evaluation in evaluations:
        for fold in folds:
            reports.append(
                _run_one(
                    config=config,
                    paths=paths,
                    evaluation=evaluation,
                    fold=fold,
                    checkpoint_override=checkpoint_override,
                    no_save_samples=args.no_save_samples,
                )
            )
    print(json.dumps(json_safe({"runs": reports}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
