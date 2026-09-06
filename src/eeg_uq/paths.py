"""Portable resolution and validation of the private SEED data directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS_PATH = REPO_ROOT / "eeg_code" / "data" / "moda_folds.json"


@dataclass(frozen=True)
class DataPaths:
    root: Path
    augmented_h5: Path
    raw_h5: Path
    trained_weights: Path
    legacy_weights: Path
    folds: Path


def resolve_data_paths(
    data_root: str | Path | None = None,
    folds_path: str | Path | None = None,
) -> DataPaths:
    """Resolve paths without assuming that private data lives inside the repository."""
    configured_root = data_root or os.environ.get("EEG_DATA_ROOT")
    if not configured_root:
        raise ValueError(
            "Data root is not configured. Pass --data-root /path/to/eeg_private_data "
            "or set EEG_DATA_ROOT. See README.md#dataset-location."
        )
    root = Path(configured_root).expanduser().resolve()
    folds = Path(folds_path).expanduser().resolve() if folds_path else DEFAULT_FOLDS_PATH
    return DataPaths(
        root=root,
        augmented_h5=root / "moda_augmented_s5.h5",
        raw_h5=root / "moda_moda.h5",
        trained_weights=root / "model_weights",
        legacy_weights=root / "model_weights_best_original",
        folds=folds,
    )


def validate_data_paths(
    paths: DataPaths,
    *,
    require_augmented: bool = False,
    require_raw: bool = False,
    require_legacy_weights: bool = False,
    require_folds: bool = True,
) -> None:
    """Fail early with the exact expected portable layout."""
    required: list[Path] = [paths.root]
    if require_augmented:
        required.append(paths.augmented_h5)
    if require_raw:
        required.append(paths.raw_h5)
    if require_legacy_weights:
        required.append(paths.legacy_weights)
    if require_folds:
        required.append(paths.folds)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        expected = (
            f"{paths.root}/\n"
            "  moda_augmented_s5.h5\n"
            "  moda_moda.h5\n"
            "  model_weights/\n"
            "  model_weights_best_original/\n"
            f"Fold metadata: {paths.folds}"
        )
        raise FileNotFoundError(
            "Missing required SEED data path(s):\n- "
            + "\n- ".join(missing)
            + "\nExpected layout:\n"
            + expected
        )
