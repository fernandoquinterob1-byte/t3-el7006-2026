#!/usr/bin/env python3
"""Combine trained SEED members into a Deep Ensemble and save its predictions.

No inference: it stacks the validation/test probabilities that ``train.py``
exported for every member.  Outputs use the Laplace NPZ layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.config import DEFAULT_CONFIG_PATH, load_project_config  # noqa: E402
from eeg_uq.uncertainty.deep_ensemble import combine_member_predictions  # noqa: E402
from eeg_uq.uncertainty.laplace_redux import json_safe, save_split_predictions  # noqa: E402
from eeg_uq.uncertainty.sampling_runner import resolve_repo_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--no-save-samples", action="store_true", help="Do not store the member probabilities.")
    args = parser.parse_args()
    config = load_project_config(args.config)
    section = config["deep_ensemble"]
    uncertainty = config["uncertainty"]
    seed_run_root = resolve_repo_path(REPO_ROOT, section["seed_run_root"])
    output_root = resolve_repo_path(REPO_ROOT, section["output_root"])
    seeds = [int(seed) for seed in section["seeds"]]
    reports = []
    for evaluation in section["evaluations"]:
        for fold in section["folds"]:
            run_dirs = [seed_run_root / f"train_eval_{evaluation}_fold_{fold}_seed_{seed}" for seed in seeds]
            missing = [str(run_dir) for run_dir in run_dirs if not (run_dir / "test_predictions.npz").is_file()]
            if missing:
                raise FileNotFoundError("Members not trained yet:\n" + "\n".join(missing))
            output_dir = output_root / f"eval_{evaluation + 1}_fold_{fold + 1}"
            output_dir.mkdir(parents=True, exist_ok=True)
            split_metrics = {}
            for split in ("validation", "test"):
                predictions = combine_member_predictions(
                    [run_dir / f"{split}_predictions.npz" for run_dir in run_dirs],
                    detection_threshold=float(section["detection_threshold"]),
                    duration_threshold=float(section["duration_threshold"]),
                )
                _, split_metrics[split] = save_split_predictions(
                    predictions,
                    output_dir / f"{split}_predictions.npz",
                    save_predictive_samples=bool(uncertainty["save_predictive_samples"]) and not args.no_save_samples,
                    epsilon=float(uncertainty["epsilon"]),
                    ece_bins=int(uncertainty["ece_bins"]),
                )
            report = {
                "evaluation": evaluation,
                "fold": fold,
                "method": "deep_ensemble",
                "seeds": seeds,
                "members": [str(run_dir) for run_dir in run_dirs],
                "validation_uncertainty": split_metrics["validation"],
                "test_uncertainty": split_metrics["test"],
            }
            (output_dir / "metrics.json").write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"deep ensemble eval {evaluation} fold {fold}: {len(seeds)} members", flush=True)
            reports.append(report)
    print(json.dumps(json_safe({"runs": reports}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
