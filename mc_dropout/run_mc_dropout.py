#!/usr/bin/env python3
"""Evaluate MC Dropout on saved SEED checkpoints.

Inference only: SEED is never trained or modified.  ``--groups`` selects which
dropout layers stay active, to see where the uncertainty comes from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.config import DEFAULT_CONFIG_PATH, load_project_config  # noqa: E402
from eeg_uq.paths import resolve_data_paths, validate_data_paths  # noqa: E402
from eeg_uq.uncertainty.laplace_redux import json_safe  # noqa: E402
from eeg_uq.uncertainty.mc_dropout import DROPOUT_GROUPS, mc_dropout_probabilities  # noqa: E402
from eeg_uq.uncertainty.sampling_runner import resolve_repo_path, run_sampling_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--evaluation", type=int, choices=range(3), help="Optional zero-based override.")
    parser.add_argument("--fold", type=int, choices=range(5), help="Optional zero-based override.")
    parser.add_argument("--data-root", type=Path, help="Optional override; YAML/env is used otherwise.")
    parser.add_argument("--folds-path", type=Path)
    parser.add_argument("--groups", nargs="+", choices=sorted(DROPOUT_GROUPS), help="Override the active dropout layers.")
    parser.add_argument("--output-root", type=Path, help="Override the output folder (e.g. per dropout group).")
    parser.add_argument("--no-save-samples", action="store_true", help="Do not store the pass probabilities.")
    args = parser.parse_args()
    config = load_project_config(args.config)
    section = config["mc_dropout"]
    inference = section["inference"]
    groups = list(args.groups or section["groups"])
    n_samples = int(section["n_samples"])
    paths = resolve_data_paths(
        args.data_root or resolve_repo_path(REPO_ROOT, config["paths"].get("data_root")),
        args.folds_path or resolve_repo_path(REPO_ROOT, config["paths"].get("folds_path")),
    )
    validate_data_paths(paths, require_raw=True)

    def make_predictor(model, device):
        return lambda signal: mc_dropout_probabilities(
            model,
            signal,
            device=device,
            n_samples=n_samples,
            groups=groups,
            sample_rate_hz=int(inference["sample_rate_hz"]),
            window_seconds=int(inference["window_seconds"]),
            stride_seconds=int(inference["stride_seconds"]),
        )

    evaluations = [args.evaluation] if args.evaluation is not None else [int(v) for v in section["evaluations"]]
    folds = [args.fold] if args.fold is not None else [int(v) for v in section["folds"]]
    reports = []
    for evaluation in evaluations:
        for fold in folds:
            torch.manual_seed(int(section["random_seed"]) + 100 * evaluation + fold)
            reports.append(
                run_sampling_fold(
                    repo_root=REPO_ROOT,
                    config=config,
                    section="mc_dropout",
                    paths=paths,
                    evaluation=evaluation,
                    fold=fold,
                    make_predictor=make_predictor,
                    report_extra={
                        "method": "mc_dropout",
                        "dropout_groups": groups,
                        "n_samples": n_samples,
                        "inference_protocol": "legacy 20 s windows / 10 s stride / central stitching, per pass",
                    },
                    output_root=resolve_repo_path(REPO_ROOT, args.output_root) if args.output_root else None,
                    no_save_samples=args.no_save_samples,
                )
            )
    print(json.dumps(json_safe({"runs": reports}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
