#!/usr/bin/env python3
"""Run inference-only raw validation/test export for an already-trained SEED fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.paths import resolve_data_paths, validate_data_paths  # noqa: E402
from eeg_uq.trained_evaluation import (  # noqa: E402
    evaluate_trained_split,
    save_trained_split_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=int, required=True, choices=range(3), help="Zero-based: 0, 1 or 2.")
    parser.add_argument("--fold", type=int, required=True, choices=range(5), help="Zero-based: 0 through 4.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training output containing checkpoints/best_full.pt.")
    parser.add_argument("--checkpoint", type=Path, help="Override the default best_full.pt path.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--data-path", type=Path, help="Override moda_moda.h5.")
    parser.add_argument("--folds-path", type=Path)
    parser.add_argument(
        "--splits", nargs="+", choices=("validation", "test"), default=("test",),
        help="Predictions to export. Default: test; use 'validation test' to backfill both.",
    )
    parser.add_argument("--output-path", type=Path, help="Custom NPZ path; valid only when exporting one split.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.output_path is not None and len(args.splits) != 1:
        parser.error("--output-path can be used only with one requested split.")

    try:
        paths = resolve_data_paths(args.data_root, args.folds_path)
        data_path = args.data_path.resolve() if args.data_path else paths.raw_h5
        validate_data_paths(paths, require_raw=args.data_path is None)
        run_dir = args.run_dir.resolve()
        checkpoint = args.checkpoint.resolve() if args.checkpoint else run_dir / "checkpoints" / "best_full.pt"
        for required in (data_path, paths.folds, checkpoint):
            if not required.is_file():
                raise FileNotFoundError(f"Required file does not exist: {required}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    last_reported = 0

    def report_progress(position: int, total: int) -> None:
        nonlocal last_reported
        step = max(1, total // 10)
        if position == 1 or position == total or position - last_reported >= step:
            print(f"Inference: {position}/{total}", flush=True)
            last_reported = position

    exports: dict[str, str] = {}
    for split in args.splits:
        last_reported = 0
        print(f"Exporting {split} predictions...", flush=True)
        predictions = evaluate_trained_split(
            data_path=data_path,
            folds_path=paths.folds,
            checkpoint_path=checkpoint,
            run_dir=run_dir,
            evaluation=args.evaluation,
            fold=args.fold,
            split=split,
            device=args.device,
            progress=report_progress,
        )
        output_path = args.output_path.resolve() if args.output_path else run_dir / f"{split}_predictions.npz"
        prediction_path, metrics_path = save_trained_split_predictions(predictions, output_path)
        exports[split] = str(prediction_path)
    print(
        json.dumps(
            {
                "training_performed": False,
                "prediction_exports": exports,
                "last_metrics": str(metrics_path),
                "samples_last_split": int(len(predictions.sample_ids)),
                "subjects_last_split": int(len(set(predictions.subject_ids))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
