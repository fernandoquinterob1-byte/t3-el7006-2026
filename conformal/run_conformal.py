#!/usr/bin/env python3
"""Fit split conformal on validation predictions and evaluate test predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from conformal.plots import generate_plots  # noqa: E402
from eeg_uq.config import DEFAULT_CONFIG_PATH, configured, load_project_config  # noqa: E402
from eeg_uq.uncertainty.conformal import (  # noqa: E402
    build_prediction_sets,
    compute_conformal_metrics,
    compute_nonconformity_scores,
    compute_rejection_curve,
    conformal_rank,
    extract_event_records,
    fit_conformal_threshold,
    summarize_event_conformal,
)


MASK_KEYS = ("valid_mask", "annotation_mask")
MODEL_ID_KEYS = ("model_id", "checkpoint_id", "checkpoint_path")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _scalar(arrays: dict[str, np.ndarray], key: str) -> Any | None:
    if key not in arrays:
        return None
    value = np.asarray(arrays[key])
    if value.size != 1:
        return None
    return value.reshape(-1)[0].item()


def _probabilities(arrays: dict[str, np.ndarray], path: Path) -> np.ndarray:
    if "probabilities" in arrays:
        return np.asarray(arrays["probabilities"], dtype=float)
    if "logits" not in arrays:
        raise ValueError(f"{path} has neither probabilities nor logits.")
    logits = np.asarray(arrays["logits"], dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _mask(
    arrays: dict[str, np.ndarray],
    shape: tuple[int, ...],
    requested_key: str | None,
) -> tuple[np.ndarray, str]:
    if requested_key is not None:
        if requested_key not in arrays:
            raise ValueError(f"Requested mask key {requested_key!r} is absent from the NPZ.")
        key = requested_key
    else:
        key = next((candidate for candidate in MASK_KEYS if candidate in arrays), None)
    if key is None:
        return np.ones(shape, dtype=bool), "all_timesteps (no saved mask)"
    mask = np.asarray(arrays[key], dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"Mask {key!r} shape {mask.shape} does not match temporal shape {shape}.")
    return mask, key


def _validate_role(arrays: dict[str, np.ndarray], role: str, path: Path) -> str | None:
    split = _scalar(arrays, "split")
    if split is None:
        return None
    split_text = str(split).strip().lower()
    allowed = {"calibration", "validation", "val"} if role == "calibration" else {"test"}
    if split_text not in allowed:
        raise ValueError(
            f"{path} declares split={split!r}, which cannot be used as the {role} artifact."
        )
    return split_text


def _zero_based_pair(arrays: dict[str, np.ndarray]) -> tuple[int, int] | None:
    legacy_evaluation = _scalar(arrays, "legacy_evaluation")
    legacy_fold = _scalar(arrays, "legacy_fold")
    if legacy_evaluation is not None and legacy_fold is not None:
        return int(legacy_evaluation) - 1, int(legacy_fold) - 1
    cli_evaluation = _scalar(arrays, "cli_evaluation_zero_based")
    cli_fold = _scalar(arrays, "cli_fold_zero_based")
    if cli_evaluation is not None and cli_fold is not None:
        return int(cli_evaluation), int(cli_fold)
    evaluation = _scalar(arrays, "evaluation")
    fold = _scalar(arrays, "fold")
    if evaluation is not None and fold is not None:
        return int(evaluation), int(fold)
    return None


def _check_no_subject_leakage(
    calibration: dict[str, np.ndarray], test: dict[str, np.ndarray]
) -> tuple[int, int]:
    if "subject_id" not in calibration or "subject_id" not in test:
        raise ValueError(
            "Both artifacts must contain subject_id so validation/test leakage can be checked."
        )
    calibration_subjects = set(np.asarray(calibration["subject_id"], dtype="U").reshape(-1))
    test_subjects = set(np.asarray(test["subject_id"], dtype="U").reshape(-1))
    overlap = calibration_subjects.intersection(test_subjects)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"Calibration/test subject leakage detected: {preview}")
    return len(calibration_subjects), len(test_subjects)


def _check_model_metadata(
    calibration: dict[str, np.ndarray], test: dict[str, np.ndarray], warnings: list[str]
) -> bool:
    for key in MODEL_ID_KEYS:
        calibration_value, test_value = _scalar(calibration, key), _scalar(test, key)
        if calibration_value is not None and test_value is not None:
            if str(calibration_value) != str(test_value):
                raise ValueError(f"Calibration/test {key} values do not match.")
            return True
    calibration_seed, test_seed = _scalar(calibration, "seed"), _scalar(test, "seed")
    if calibration_seed is not None and test_seed is not None and calibration_seed != test_seed:
        raise ValueError("Calibration/test seed metadata does not match.")
    warnings.append(
        "The NPZ files do not share a model/checkpoint identifier; the CLI cannot prove that "
        "both probability arrays came from the same trained SEED checkpoint."
    )
    return False


def _event_json_summary(event: dict[str, np.ndarray]) -> dict[str, Any]:
    status = np.asarray(event["event_status"], dtype="U")
    ambiguity = np.asarray(event["fraction_ambiguous_timesteps"], dtype=float)

    def finite_mean(mask: np.ndarray) -> float | None:
        selected = ambiguity[np.logical_and(mask, np.isfinite(ambiguity))]
        return float(selected.mean()) if len(selected) else None

    return {
        "n_event_records": int(len(status)),
        "n_predicted_events": int(np.asarray(event["event_has_prediction"], dtype=bool).sum()),
        "status_counts": {value: int((status == value).sum()) for value in np.unique(status)},
        "mean_ambiguous_fraction_tp": finite_mean(status == "TP"),
        "mean_ambiguous_fraction_fp": finite_mean(status == "FP"),
        "guarantee": "exploratory summaries only; timestep marginal coverage is not an event-level guarantee",
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        help="Optional alpha sweep for comparison plots; the detailed arrays still use --alpha.",
    )
    parser.add_argument(
        "--rejection-thresholds", type=float, nargs="+"
    )
    parser.add_argument("--calibration-mask-key")
    parser.add_argument("--test-mask-key")
    parser.add_argument(
        "--event-probability-key",
        help="Optional test NPZ array used for the event p75; defaults to adjusted_probabilities when present, otherwise probabilities.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "outputs" / "conformal"
    )
    parser.add_argument("--output-dir", type=Path, help="Override the inferred eval_N_fold_N directory.")
    parser.add_argument(
        "--output-evaluation", type=int, help="One-based output evaluation ID if NPZ metadata is absent."
    )
    parser.add_argument(
        "--output-fold", type=int, help="One-based output fold ID if NPZ metadata is absent."
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_config = load_project_config(args.config)
    conformal_defaults = project_config["conformal"]
    args.alpha = float(configured(args.alpha, conformal_defaults, "alpha"))
    args.alphas = args.alphas or conformal_defaults["alpha_sweep"]
    args.rejection_thresholds = args.rejection_thresholds or conformal_defaults[
        "rejection_thresholds"
    ]
    calibration_path = args.calibration_predictions.resolve()
    test_path = args.test_predictions.resolve()
    if calibration_path == test_path:
        raise ValueError("Calibration and test predictions must be different files.")
    calibration = _load_npz(calibration_path)
    test = _load_npz(test_path)
    warnings: list[str] = []
    calibration_split = _validate_role(calibration, "calibration", calibration_path)
    test_split = _validate_role(test, "test", test_path)
    if calibration_split is None:
        warnings.append("Calibration NPZ has no split metadata; its validation role cannot be verified.")
    if test_split is None:
        warnings.append("Test NPZ has no split metadata; its test role cannot be verified from the file.")
    calibration_subject_count, test_subject_count = _check_no_subject_leakage(calibration, test)
    model_identity_verified = _check_model_metadata(calibration, test, warnings)

    calibration_pair, test_pair = _zero_based_pair(calibration), _zero_based_pair(test)
    if calibration_pair is not None and test_pair is not None and calibration_pair != test_pair:
        raise ValueError(
            f"Calibration metadata identifies eval/fold {calibration_pair}, but test identifies {test_pair}."
        )
    pair = calibration_pair or test_pair
    if pair is None:
        if args.output_evaluation is None or args.output_fold is None:
            raise ValueError(
                "NPZ metadata has no evaluation/fold; provide --output-evaluation and --output-fold."
            )
        evaluation, fold = args.output_evaluation, args.output_fold
    else:
        evaluation, fold = pair[0] + 1, pair[1] + 1
        if args.output_evaluation is not None and args.output_evaluation != evaluation:
            raise ValueError("--output-evaluation conflicts with NPZ metadata.")
        if args.output_fold is not None and args.output_fold != fold:
            raise ValueError("--output-fold conflicts with NPZ metadata.")

    calibration_probabilities = _probabilities(calibration, calibration_path)
    test_probabilities = _probabilities(test, test_path)
    if "y_true" not in calibration or "y_true" not in test:
        raise ValueError("Both prediction artifacts must contain y_true.")
    calibration_truth = np.asarray(calibration["y_true"])
    test_truth = np.asarray(test["y_true"])
    calibration_mask, calibration_mask_source = _mask(
        calibration, calibration_probabilities.shape, args.calibration_mask_key
    )
    test_mask, test_mask_source = _mask(test, test_probabilities.shape, args.test_mask_key)
    if "no saved mask" in calibration_mask_source:
        warnings.append(
            "Calibration NPZ has no valid/annotation mask; all stored timesteps are treated as valid."
        )
    if "no saved mask" in test_mask_source:
        warnings.append("Test NPZ has no valid/annotation mask; all stored timesteps are treated as valid.")

    calibration_scores = compute_nonconformity_scores(
        calibration_probabilities, calibration_truth, calibration_mask
    )
    q_hat = fit_conformal_threshold(calibration_scores, args.alpha)
    prediction_sets = build_prediction_sets(test_probabilities, q_hat, test_mask)
    metrics = compute_conformal_metrics(
        test_truth,
        prediction_sets["contains_background"],
        prediction_sets["contains_event"],
        test_mask,
        alpha=args.alpha,
    )

    sweep_alphas = sorted(set([float(args.alpha), *(args.alphas or [])]))
    sweep_q_hat, sweep_metrics = [], []
    for alpha in sweep_alphas:
        threshold = fit_conformal_threshold(calibration_scores, alpha)
        sets = build_prediction_sets(test_probabilities, threshold, test_mask)
        sweep_q_hat.append(threshold)
        sweep_metrics.append(
            compute_conformal_metrics(
                test_truth,
                sets["contains_background"],
                sets["contains_event"],
                test_mask,
                alpha=alpha,
            )
        )

    event_records = extract_event_records(test)
    if args.event_probability_key:
        if args.event_probability_key not in test:
            raise ValueError(f"Event probability key {args.event_probability_key!r} is absent.")
        event_probability_array = np.asarray(test[args.event_probability_key], dtype=float)
        event_probability_source = args.event_probability_key
    elif "adjusted_probabilities" in test:
        event_probability_array = np.asarray(test["adjusted_probabilities"], dtype=float)
        event_probability_source = "adjusted_probabilities"
    else:
        event_probability_array = test_probabilities
        event_probability_source = "probabilities"
        warnings.append(
            "No adjusted_probabilities field was found; event_probability_seed uses the p75 "
            "of the saved probabilities inside each legacy predicted event."
        )
    if len(event_records["event_sample_id"]):
        event_summary = summarize_event_conformal(
            test_probabilities,
            prediction_sets["set_size"],
            prediction_sets["contains_event"],
            event_records,
            sample_ids=np.asarray(test.get("sample_id", np.arange(test_probabilities.shape[0]))),
            subject_ids=np.asarray(
                test.get("subject_id", np.full(test_probabilities.shape[0], "")), dtype="U"
            ),
            valid_mask=test_mask,
            event_probabilities=event_probability_array,
        )
        rejection = compute_rejection_curve(event_summary, args.rejection_thresholds)
    else:
        event_summary = None
        rejection = None
        warnings.append(
            "Test NPZ has neither saved event records nor processed_pred; event summaries and rejection are unavailable."
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (args.output_root / f"eval_{evaluation}_fold_{fold}").resolve()
    )
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    result_path = output_dir / "conformal_results.npz"
    metrics_path = output_dir / "metrics.json"

    sample_ids = np.asarray(test.get("sample_id", np.arange(test_probabilities.shape[0])))
    subject_ids = np.asarray(
        test.get("subject_id", np.full(test_probabilities.shape[0], "")), dtype="U"
    )
    result_arrays: dict[str, np.ndarray] = {
        "evaluation": np.asarray(evaluation, dtype=np.int64),
        "fold": np.asarray(fold, dtype=np.int64),
        "alpha": np.asarray(args.alpha, dtype=float),
        "target_coverage": np.asarray(1.0 - args.alpha, dtype=float),
        "q_hat": np.asarray(q_hat, dtype=float),
        "empirical_coverage": np.asarray(metrics["empirical_coverage"], dtype=float),
        "mean_set_size": np.asarray(metrics["mean_set_size"], dtype=float),
        "singleton_rate": np.asarray(metrics["singleton_rate"], dtype=float),
        "ambiguous_rate": np.asarray(metrics["ambiguous_rate"], dtype=float),
        "empty_rate": np.asarray(metrics["empty_rate"], dtype=float),
        "sample_id": sample_ids,
        "subject_id": subject_ids,
        "y_true": test_truth.astype(np.int8),
        "probabilities": test_probabilities.astype(np.float32),
        "valid_mask": prediction_sets["valid_mask"],
        "contains_background": prediction_sets["contains_background"],
        "contains_event": prediction_sets["contains_event"],
        "set_size": prediction_sets["set_size"],
        "coverage_indicator": np.logical_and(
            np.where(
                test_truth == 1,
                prediction_sets["contains_event"].astype(bool),
                prediction_sets["contains_background"].astype(bool),
            ),
            test_mask,
        ).astype(np.uint8),
        "sweep_alpha": np.asarray(sweep_alphas, dtype=float),
        "sweep_target_coverage": 1.0 - np.asarray(sweep_alphas, dtype=float),
        "sweep_q_hat": np.asarray(sweep_q_hat, dtype=float),
        "sweep_empirical_coverage": np.asarray(
            [item["empirical_coverage"] for item in sweep_metrics], dtype=float
        ),
        "sweep_mean_set_size": np.asarray(
            [item["mean_set_size"] for item in sweep_metrics], dtype=float
        ),
        "calibration_mask_source": np.asarray(calibration_mask_source),
        "test_mask_source": np.asarray(test_mask_source),
        "event_probability_source": np.asarray(event_probability_source),
    }
    if event_summary is not None:
        result_arrays.update(event_summary)
    if rejection is not None:
        result_arrays.update(rejection)
    for key, value in result_arrays.items():
        if np.asarray(value).dtype.kind == "O":
            raise TypeError(f"Refusing to save object array {key!r}.")
    np.savez_compressed(result_path, **result_arrays)

    rank = conformal_rank(len(calibration_scores), args.alpha)
    report: dict[str, Any] = {
        "evaluation": evaluation,
        "fold": fold,
        "alpha": float(args.alpha),
        "target_coverage": 1.0 - float(args.alpha),
        "q_hat": q_hat,
        "empirical_coverage": metrics["empirical_coverage"],
        "coverage_gap": metrics["coverage_gap"],
        "mean_set_size": metrics["mean_set_size"],
        "singleton_rate": metrics["singleton_rate"],
        "ambiguous_rate": metrics["ambiguous_rate"],
        "empty_rate": metrics["empty_rate"],
        "n_calibration_timesteps": int(len(calibration_scores)),
        "n_test_timesteps": metrics["n_valid_timesteps"],
        "conformal_rank_one_based": rank,
        "finite_sample_quantile_level": rank / len(calibration_scores),
        "calibration_predictions": str(calibration_path),
        "test_predictions": str(test_path),
        "calibration_split": calibration_split,
        "test_split": test_split,
        "split_policy": "validation predictions calibrate; test predictions are evaluated only",
        "calibration_mask_source": calibration_mask_source,
        "test_mask_source": test_mask_source,
        "calibration_subject_count": calibration_subject_count,
        "test_subject_count": test_subject_count,
        "validation_test_subject_overlap": 0,
        "model_identity_verified": model_identity_verified,
        "event_probability_source": event_probability_source,
        "alpha_sweep": [
            {"q_hat": threshold, **item}
            for threshold, item in zip(sweep_q_hat, sweep_metrics)
        ],
        "event_summary": _event_json_summary(event_summary) if event_summary is not None else None,
        "rejection_curve": (
            [
                {key: np.asarray(value)[index] for key, value in rejection.items()}
                for index in range(len(rejection["rejection_threshold"]))
            ]
            if rejection is not None
            else None
        ),
        "warnings": warnings,
        "methodological_note": (
            "The requested initial protocol reuses the validation fold as calibration. In the "
            "current training pipeline that fold also selects the checkpoint, so it is not a fully "
            "independent calibration split. Test data never enters q_hat estimation."
        ),
    }
    metrics_path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not args.no_plots:
        generate_plots(result_path, metrics_path, plots_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "results": str(result_path),
                "metrics": str(metrics_path),
                "plots": None if args.no_plots else str(plots_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
