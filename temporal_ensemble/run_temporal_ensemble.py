#!/usr/bin/env python3
"""Evaluate the phase-shift Temporal Ensemble on saved SEED checkpoints.

Inference only: SEED is never trained or modified.  Outputs use the same NPZ
layout as Laplace, so the same analysis notebook code applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.config import DEFAULT_CONFIG_PATH, load_project_config  # noqa: E402
from eeg_uq.paths import resolve_data_paths, validate_data_paths  # noqa: E402
from eeg_uq.uncertainty.laplace_redux import json_safe  # noqa: E402
from eeg_uq.uncertainty.sampling_runner import resolve_repo_path, run_sampling_fold  # noqa: E402
from eeg_uq.uncertainty.temporal_ensemble import phase_shifted_probabilities  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--evaluation", type=int, choices=range(3), help="Optional zero-based override.")
    parser.add_argument("--fold", type=int, choices=range(5), help="Optional zero-based override.")
    parser.add_argument("--data-root", type=Path, help="Optional override; YAML/env is used otherwise.")
    parser.add_argument("--folds-path", type=Path)
    parser.add_argument("--no-save-samples", action="store_true", help="Do not store the member probabilities.")
    args = parser.parse_args()
    config = load_project_config(args.config)
    section = config["temporal_ensemble"]
    inference = section["inference"]
    shifts = [int(shift) for shift in section["phase_shifts"]]
    paths = resolve_data_paths(
        args.data_root or resolve_repo_path(REPO_ROOT, config["paths"].get("data_root")),
        args.folds_path or resolve_repo_path(REPO_ROOT, config["paths"].get("folds_path")),
    )
    validate_data_paths(paths, require_raw=True)

    def make_predictor(model, device):
        return lambda signal: phase_shifted_probabilities(
            model,
            signal,
            device=device,
            shifts=shifts,
            sample_rate_hz=int(inference["sample_rate_hz"]),
            window_seconds=int(inference["window_seconds"]),
            stride_seconds=int(inference["stride_seconds"]),
        )

    evaluations = [args.evaluation] if args.evaluation is not None else [int(v) for v in section["evaluations"]]
    folds = [args.fold] if args.fold is not None else [int(v) for v in section["folds"]]
    reports = [
        run_sampling_fold(
            repo_root=REPO_ROOT,
            config=config,
            section="temporal_ensemble",
            paths=paths,
            evaluation=evaluation,
            fold=fold,
            make_predictor=make_predictor,
            report_extra={
                "method": "temporal_ensemble_phase_shift",
                "phase_shifts": shifts,
                "inference_protocol": "legacy 20 s windows / 10 s stride / central stitching, per phase shift",
            },
            no_save_samples=args.no_save_samples,
        )
        for evaluation in evaluations
        for fold in folds
    ]
    print(json.dumps(json_safe({"runs": reports}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
