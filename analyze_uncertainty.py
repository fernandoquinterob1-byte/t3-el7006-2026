#!/usr/bin/env python3
"""Add binary temporal entropy/confidence to an exported SEED NPZ file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from eeg_uq.uncertainty.baseline import analyse_prediction_file, event_mean_entropy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report, arrays = analyse_prediction_file(args.predictions)
    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Event means are deliberately an exploratory first aggregation only.
    if {"event_sample_id", "event_pred_start", "event_pred_end"}.issubset(arrays):
        lookup = {int(sample): index for index, sample in enumerate(arrays["sample_id"])}
        means = []
        for sample, start, end in zip(arrays["event_sample_id"], arrays["event_pred_start"], arrays["event_pred_end"]):
            value = event_mean_entropy(arrays["probabilities"][lookup[int(sample)]], int(start), int(end))
            means.append(np.nan if value is None else value)
        arrays["event_mean_entropy"] = np.asarray(means, dtype=np.float32)
        report["event_uncertainty_aggregation"] = "mean entropy over each legacy predicted event; exploratory"
    np.savez_compressed(output_dir / "predictions_with_uncertainty.npz", **arrays)
    (output_dir / "uncertainty_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
