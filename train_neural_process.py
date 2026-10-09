#!/usr/bin/env python3
"""Train and export a SEED-backed Neural Process for one fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from eeg_uq.config import DEFAULT_CONFIG_PATH, configured, load_project_config  # noqa: E402
from eeg_uq.folds import get_fold, load_folds, validate_folds  # noqa: E402
from eeg_uq.np_evaluation import evaluate_neural_process_split, save_neural_process_predictions  # noqa: E402
from eeg_uq.np_training import NeuralProcessConfig, train_neural_process_one_fold  # noqa: E402
from eeg_uq.paths import resolve_data_paths, validate_data_paths  # noqa: E402


def _subjects_from_h5(path: Path) -> set[str]:
    with h5py.File(path, "r") as source:
        return {
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in source["subject_ids"][:]
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=int, required=True, choices=range(3))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--folds-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed-run-root", type=Path)
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--model-kind", choices=("attn_cnp", "attn_lnp"))
    parser.add_argument("--context-fraction", type=float)
    parser.add_argument("--r-dim", type=int)
    parser.add_argument("--z-dim", type=int)
    parser.add_argument("--latent-samples-train", type=int)
    parser.add_argument("--latent-samples-test", type=int)
    parser.add_argument("--no-freeze-seed-backbone", action="store_true")
    parser.add_argument("--no-export-predictions", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overfit-small-subset", type=int, metavar="N")
    args = parser.parse_args()

    try:
        project_config = load_project_config(args.config)
        training_defaults = project_config["training"]
        np_defaults = project_config.get("neural_process", {})
        args.seed = int(configured(args.seed, project_config, "seed"))
        args.output_root = Path(
            configured(args.output_root, np_defaults, "output_root")
            if args.output_root is None
            else args.output_root
        )
        args.seed_run_root = Path(
            configured(args.seed_run_root, np_defaults, "seed_run_root")
            if args.seed_run_root is None
            else args.seed_run_root
        )
        args.batch_size = int(configured(args.batch_size, np_defaults, "batch_size"))
        args.learning_rate = float(configured(args.learning_rate, np_defaults, "learning_rate"))
        args.weight_decay = float(configured(args.weight_decay, training_defaults, "weight_decay"))
        args.epochs = int(configured(args.epochs, np_defaults, "epochs"))
        args.num_workers = int(configured(args.num_workers, training_defaults, "num_workers"))
        args.early_stopping_patience = int(
            configured(args.early_stopping_patience, training_defaults, "early_stopping_patience")
        )
        args.device = str(configured(args.device, training_defaults, "device"))
        args.model_kind = str(configured(args.model_kind, np_defaults, "model_kind"))
        args.context_fraction = float(configured(args.context_fraction, np_defaults, "context_fraction"))
        args.r_dim = int(configured(args.r_dim, np_defaults, "r_dim"))
        args.z_dim = int(configured(args.z_dim, np_defaults, "z_dim"))
        args.latent_samples_train = int(
            configured(args.latent_samples_train, np_defaults, "latent_samples_train")
        )
        args.latent_samples_test = int(
            configured(args.latent_samples_test, np_defaults, "latent_samples_test")
        )
        export_predictions = bool(np_defaults.get("export_validation_and_test", True)) and not args.no_export_predictions
        paths = resolve_data_paths(args.data_root, args.folds_path)
        validate_data_paths(paths, require_augmented=True, require_raw=export_predictions)
        folds = load_folds(paths.folds)
        subjects = _subjects_from_h5(paths.augmented_h5)
        validate_folds(folds, subjects)
        validation_subjects, test_subjects = get_fold(folds, args.evaluation + 1, args.fold + 1)
        train_subjects = sorted(subjects - set(validation_subjects) - set(test_subjects))
        seed_checkpoint = (
            args.seed_checkpoint
            if args.seed_checkpoint is not None
            else args.seed_run_root
            / f"train_eval_{args.evaluation}_fold_{args.fold}_seed_{args.seed}"
            / "checkpoints"
            / "best_full.pt"
        )
        if not seed_checkpoint.is_file():
            raise FileNotFoundError(
                f"SEED checkpoint not found: {seed_checkpoint}. Pass --seed-checkpoint explicitly."
            )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    config = NeuralProcessConfig(
        evaluation=args.evaluation,
        fold=args.fold,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=2 if args.debug else args.epochs,
        num_workers=0 if args.debug else args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        model_kind=args.model_kind,
        context_fraction=args.context_fraction,
        r_dim=args.r_dim,
        z_dim=args.z_dim,
        latent_samples_train=args.latent_samples_train,
        latent_samples_test=args.latent_samples_test,
        freeze_seed_backbone=not args.no_freeze_seed_backbone,
        debug=args.debug,
        overfit_small_subset=args.overfit_small_subset or (128 if args.debug else None),
    )
    output_dir = args.output_root / f"train_eval_{args.evaluation}_fold_{args.fold}_seed_{args.seed}"
    result = train_neural_process_one_fold(
        config,
        augmented_h5=paths.augmented_h5,
        train_subjects=train_subjects,
        validation_subjects=validation_subjects,
        seed_checkpoint=seed_checkpoint,
        output_dir=output_dir,
    )

    exported_paths: dict[str, str] = {}
    if export_predictions:
        checkpoint = output_dir / "checkpoints" / "best_np.pt"
        for split, split_subjects in (("validation", validation_subjects), ("test", test_subjects)):
            print(f"Exporting Neural Process {split} predictions...", flush=True)
            predictions = evaluate_neural_process_split(
                data_path=paths.raw_h5,
                checkpoint_path=checkpoint,
                seed_checkpoint=seed_checkpoint,
                evaluation=args.evaluation,
                fold=args.fold,
                split=split,
                subjects=split_subjects,
                device=args.device,
                save_predictive_samples=True,
            )
            prediction_path, _ = save_neural_process_predictions(
                predictions, output_dir / f"{split}_predictions.npz"
            )
            exported_paths[split] = str(prediction_path)
        metrics_path = output_dir / "metrics.json"
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["prediction_exports"] = exported_paths
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "best_val_loss": result["best_val_loss"],
                "source_seed_checkpoint": str(seed_checkpoint),
                "prediction_exports": exported_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

