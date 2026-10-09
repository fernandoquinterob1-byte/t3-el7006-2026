#!/usr/bin/env python3
"""Train the missing Deep Ensemble members with ``train.py``, one after another.

Resumable: a member whose run directory already has ``test_predictions.npz``
is skipped, so the command can be relaunched after an interruption.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.config import DEFAULT_CONFIG_PATH, load_project_config  # noqa: E402
from eeg_uq.uncertainty.sampling_runner import resolve_repo_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_project_config(args.config)
    section = config["deep_ensemble"]
    seed_run_root = resolve_repo_path(REPO_ROOT, section["seed_run_root"])
    for evaluation in section["evaluations"]:
        for fold in section["folds"]:
            for seed in section["seeds"]:
                run_dir = seed_run_root / f"train_eval_{evaluation}_fold_{fold}_seed_{seed}"
                if (run_dir / "test_predictions.npz").is_file():
                    print(f"skip eval {evaluation} fold {fold} seed {seed}: already trained", flush=True)
                    continue
                command = [
                    sys.executable, str(REPO_ROOT / "train.py"),
                    "--evaluation", str(evaluation), "--fold", str(fold), "--seed", str(seed),
                    "--output-root", str(seed_run_root), "--data-root", str(args.data_root),
                    "--config", str(args.config),
                ]
                print(f"train eval {evaluation} fold {fold} seed {seed}", flush=True)
                returncode = subprocess.run(command).returncode
                if returncode and not (run_dir / "test_predictions.npz").is_file():
                    raise RuntimeError(f"train.py failed (exit {returncode}) for {run_dir}.")
                if returncode:
                    # Windows can crash while tearing down CUDA/DataLoader workers after
                    # every artifact is written; the member itself is complete.
                    print(f"warning: train.py exited with {returncode} after writing its predictions", flush=True)
                print(f"done eval {evaluation} fold {fold} seed {seed}", flush=True)
    print("all members trained", flush=True)


if __name__ == "__main__":
    main()
