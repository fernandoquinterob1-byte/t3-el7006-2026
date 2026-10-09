#!/usr/bin/env python3
"""Train a new SEED model for one original evaluation/fold pair.

CLI indices are zero-based: --evaluation 0 --fold 0 maps to eval_1/fold_1
in moda_folds.json. The legacy baseline checkpoint flow remains separate in
evaluate_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.config import DEFAULT_CONFIG_PATH, configured, load_project_config  # noqa: E402
from eeg_uq.folds import get_fold, load_folds, validate_folds  # noqa: E402
from eeg_uq.paths import resolve_data_paths, validate_data_paths  # noqa: E402
from eeg_uq.training import TrainConfig, train_one_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=int, required=True, choices=range(3), help="0, 1 or 2")
    parser.add_argument("--fold", type=int, required=True, choices=range(5), help="0 through 4")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--folds-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--logger", choices=("none", "wandb"))
    parser.add_argument(
        "--no-export-predictions",
        action="store_true",
        help="Do not run the automatic raw validation/test inference after training.",
    )
    parser.add_argument("--debug", action="store_true", help="Use 2 epochs, 0 workers and small data subsets.")
    parser.add_argument("--overfit-small-subset", type=int, metavar="N", help="Train/validate on N records to test pipeline wiring.")
    args = parser.parse_args()
    try:
        project_config = load_project_config(args.config)
        training_defaults = project_config["training"]
        args.seed = int(configured(args.seed, project_config, "seed"))
        args.output_root = Path(configured(args.output_root, project_config, "output_root"))
        args.batch_size = int(configured(args.batch_size, training_defaults, "batch_size"))
        args.learning_rate = float(configured(args.learning_rate, training_defaults, "learning_rate"))
        args.weight_decay = float(configured(args.weight_decay, training_defaults, "weight_decay"))
        args.epochs = int(configured(args.epochs, training_defaults, "epochs"))
        args.num_workers = int(configured(args.num_workers, training_defaults, "num_workers"))
        args.early_stopping_patience = int(configured(args.early_stopping_patience, training_defaults, "early_stopping_patience"))
        args.device = str(configured(args.device, training_defaults, "device"))
        args.logger = str(configured(args.logger, training_defaults, "logger"))
        export_predictions = bool(training_defaults["export_validation_and_test"])
        if args.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Configured training device must be auto, cpu or cuda.")
        if args.logger not in {"none", "wandb"}:
            raise ValueError("Configured logger must be none or wandb.")
        paths = resolve_data_paths(args.data_root, args.folds_path)
        export_predictions = export_predictions and not args.no_export_predictions
        # Debug/overfit runs are diagnostics; their validation split is not a
        # calibration artifact and should never be exported for Conformal.
        export_predictions = export_predictions and not args.debug and args.overfit_small_subset is None
        validate_data_paths(
            paths, require_augmented=True, require_raw=export_predictions
        )
        folds = load_folds(paths.folds)
        import h5py
        import numpy as np
        with h5py.File(paths.augmented_h5, "r") as source:
            subjects = {value.decode("utf-8") for value in source["subject_ids"][:]}
        validate_folds(folds, subjects)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    legacy_evaluation, legacy_fold = args.evaluation + 1, args.fold + 1
    validation_subjects, test_subjects = get_fold(folds, legacy_evaluation, legacy_fold)
    train_subjects = sorted(subjects - set(validation_subjects) - set(test_subjects))
    config = TrainConfig(
        evaluation=args.evaluation, fold=args.fold, seed=args.seed, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        epochs=2 if args.debug else args.epochs,
        num_workers=0 if args.debug else args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        debug=args.debug,
        overfit_small_subset=args.overfit_small_subset or (128 if args.debug else None),
        logger=args.logger,
    )
    output_dir = args.output_root / f"train_eval_{args.evaluation}_fold_{args.fold}_seed_{args.seed}"
    result = train_one_fold(config, augmented_h5=paths.augmented_h5, train_subjects=train_subjects, validation_subjects=validation_subjects, output_dir=output_dir)
    exported_paths: dict[str, str] = {}
    if export_predictions:
        # This is inference after training, never a second training pass.  The
        # raw validation artifact calibrates Conformal; raw test is held out.
        from eeg_uq.trained_evaluation import (  # noqa: PLC0415
            evaluate_trained_split,
            save_trained_split_predictions,
        )

        checkpoint = output_dir / "checkpoints" / "best_full.pt"
        for split in ("validation", "test"):
            print(f"Exporting {split} predictions from the best checkpoint...", flush=True)
            predictions = evaluate_trained_split(
                data_path=paths.raw_h5,
                folds_path=paths.folds,
                checkpoint_path=checkpoint,
                run_dir=output_dir,
                evaluation=args.evaluation,
                fold=args.fold,
                split=split,
                device=args.device,
            )
            prediction_path, _ = save_trained_split_predictions(
                predictions, output_dir / f"{split}_predictions.npz"
            )
            exported_paths[split] = str(prediction_path)
        metrics_path = output_dir / "metrics.json"
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["prediction_exports"] = exported_paths
        result["prediction_export_policy"] = (
            "raw validation/test inference from best_full.pt; validation calibrates Conformal and test is held out"
        )
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "best_val_loss": result["best_val_loss"], "prediction_exports": exported_paths}, indent=2))


if __name__ == "__main__":
    main()
