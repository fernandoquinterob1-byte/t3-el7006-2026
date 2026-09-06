#!/usr/bin/env python3
"""Reproduce the legacy SEED baseline and save reusable prediction artifacts."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.evaluation import evaluate_baseline  # noqa: E402
from eeg_uq.paths import resolve_data_paths, validate_data_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Private directory containing moda_moda.h5 and model_weights_best_original.",
    )
    parser.add_argument("--data-path", type=Path, help="Override path to moda_moda.h5.")
    parser.add_argument("--weights-dir", type=Path, help="Override checkpoint directory.")
    parser.add_argument(
        "--folds-path", type=Path, default=REPO_ROOT / "eeg_code" / "data" / "moda_folds.json"
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "baseline")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--evaluation", type=int, choices=range(3), help="Zero-based evaluation (0-2); requires --fold.")
    parser.add_argument("--fold", type=int, choices=range(5), help="Zero-based fold (0-4); requires --evaluation.")
    parser.add_argument("--no-save", action="store_true", help="Compute metrics without saving NPZ outputs.")
    args = parser.parse_args()
    try:
        paths = resolve_data_paths(args.data_root, args.folds_path)
        args.data_path = args.data_path or paths.raw_h5
        args.weights_dir = args.weights_dir or paths.legacy_weights
        validate_data_paths(
            paths,
            require_raw=args.data_path == paths.raw_h5,
            require_legacy_weights=args.weights_dir == paths.legacy_weights,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    for required_path in (args.data_path, args.weights_dir, paths.folds):
        if not required_path.exists():
            parser.error(f"Required path does not exist: {required_path}")
    evaluation = args.evaluation + 1 if args.evaluation is not None else None
    fold = args.fold + 1 if args.fold is not None else None
    result = evaluate_baseline(
        data_path=args.data_path,
        weights_dir=args.weights_dir,
        folds_path=paths.folds,
        output_dir=None if args.no_save else args.output_dir,
        device=args.device,
        evaluation=evaluation,
        fold=fold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
