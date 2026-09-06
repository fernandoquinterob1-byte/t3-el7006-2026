#!/usr/bin/env python3
"""Reproducible, opt-in EDA and subject-leakage audit for the private MODA HDF5 files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from eeg_uq.folds import load_folds, validate_folds  # noqa: E402
from eeg_uq.paths import resolve_data_paths, validate_data_paths  # noqa: E402


def decode(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def inspect_h5(path: Path) -> dict:
    with h5py.File(path, "r") as source:
        subject_ids = decode(source["subject_ids"][:])
        counts = Counter(subject_ids)
        labels = source["labels"]
        summary = {
            "path": str(path),
            "datasets": {name: {"shape": list(source[name].shape), "dtype": str(source[name].dtype)} for name in source.keys()},
            "records": len(subject_ids),
            "subjects": len(counts),
            "records_per_subject": {"min": min(counts.values()), "max": max(counts.values()), "mean": float(np.mean(list(counts.values())))},
            "positive_temporal_fraction": float(np.asarray(labels[:]).mean()),
            "has_nan_signals": bool(np.isnan(source["signals"][:]).any()),
            "attributes": {str(key): str(value) for key, value in source.attrs.items()},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--folds-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "eda")
    parser.add_argument("--include-raw", action="store_true", help="Also inspect moda_moda.h5.")
    parser.add_argument("--plot-examples", type=int, default=0, metavar="N", help="Save up to N EEG examples per label class.")
    args = parser.parse_args()
    try:
        paths = resolve_data_paths(args.data_root, args.folds_path)
        validate_data_paths(paths, require_augmented=True, require_raw=args.include_raw)
        with h5py.File(paths.augmented_h5, "r") as source:
            subjects = set(decode(source["subject_ids"][:]))
        folds = load_folds(paths.folds)
        leakage = validate_folds(folds, subjects)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    report = {"augmented": inspect_h5(paths.augmented_h5), "fold_audit": leakage}
    if args.include_raw:
        report["raw"] = inspect_h5(paths.raw_h5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "eda_summary.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    if args.plot_examples:
        import matplotlib.pyplot as plt
        examples = {0: [], 1: []}
        with h5py.File(paths.augmented_h5, "r") as source:
            for index in range(len(source["signals"])):
                label = int(np.asarray(source["labels"][index]).any())
                if len(examples[label]) < args.plot_examples:
                    examples[label].append(index)
                if all(len(values) >= args.plot_examples for values in examples.values()):
                    break
            for label, indices in examples.items():
                for index in indices:
                    signal = np.asarray(source["signals"][index])
                    target = np.asarray(source["labels"][index])
                    time = np.arange(len(signal)) / 200.0  # legacy/paper sampling rate
                    figure, axis = plt.subplots(figsize=(12, 3))
                    axis.plot(time, signal, linewidth=0.7, label="EEG")
                    axis.fill_between(time, signal.min(), signal.max(), where=target.astype(bool), alpha=0.2, label="label=1")
                    axis.set(title=f"HDF5 row {index}, class {label}", xlabel="seconds", ylabel="amplitude")
                    axis.legend(loc="upper right")
                    figure.tight_layout()
                    figure.savefig(args.output_dir / f"example_class_{label}_row_{index}.png", dpi=150)
                    plt.close(figure)
    print(json.dumps(report, indent=2, default=str))
    print(f"Saved EDA summary to {path}")


if __name__ == "__main__":
    main()
