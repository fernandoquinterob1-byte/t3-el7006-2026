#!/usr/bin/env python3
"""Sequentially train selected Neural Process folds from saved SEED models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", type=int, default=[0], choices=range(3))
    parser.add_argument("--folds", nargs="+", type=int, default=[0], choices=range(5))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed-run-root", type=Path)
    parser.add_argument("--extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for evaluation in args.evaluations:
        for fold in args.folds:
            command = [
                sys.executable,
                str(REPO_ROOT / "train_neural_process.py"),
                "--evaluation",
                str(evaluation),
                "--fold",
                str(fold),
                "--data-root",
                args.data_root,
                "--config",
                str(args.config),
            ]
            if args.output_root:
                command.extend(["--output-root", str(args.output_root)])
            if args.seed_run_root:
                command.extend(["--seed-run-root", str(args.seed_run_root)])
            if args.extra:
                command.extend(args.extra)
            print("Launching:", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

