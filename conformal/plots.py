#!/usr/bin/env python3
"""Generate the standard conformal meeting plots from saved result artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib_cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_plots(results_path: Path, metrics_path: Path, plots_dir: Path) -> list[Path]:
    """Create coverage, set-size, event-ambiguity and rejection figures."""

    results_path, metrics_path, plots_dir = Path(results_path), Path(metrics_path), Path(plots_dir)
    with np.load(results_path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    alpha = arrays["sweep_alpha"]
    order = np.argsort(alpha)
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(alpha[order], arrays["sweep_target_coverage"][order], "o--", label="Target 1-alpha")
    axis.plot(alpha[order], arrays["sweep_empirical_coverage"][order], "o-", label="Empirical")
    axis.set(xlabel="alpha", ylabel="Coverage", ylim=(0.0, 1.02), title="Target vs empirical coverage")
    axis.grid(alpha=0.25)
    axis.legend()
    path = plots_dir / "coverage.png"
    _finish(fig, path)
    created.append(path)

    valid_sizes = arrays["set_size"][arrays["valid_mask"].astype(bool)]
    counts = np.asarray([(valid_sizes == value).mean() for value in (0, 1, 2)])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].plot(alpha[order], arrays["sweep_mean_set_size"][order], "o-")
    axes[0].set(xlabel="alpha", ylabel="Mean prediction set size", title="Efficiency by alpha")
    axes[0].grid(alpha=0.25)
    axes[1].bar(["0 (empty)", "1 (singleton)", "2 (ambiguous)"], counts, color=["#d95f02", "#1b9e77", "#7570b3"])
    axes[1].set(ylabel="Fraction of valid timesteps", ylim=(0.0, 1.0), title=f"Set-size distribution (alpha={metrics['alpha']:g})")
    axes[1].tick_params(axis="x", rotation=15)
    path = plots_dir / "set_size.png"
    _finish(fig, path)
    created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    if "fraction_ambiguous_timesteps" in arrays:
        ambiguity = arrays["fraction_ambiguous_timesteps"].astype(float)
        status = arrays["event_status"].astype("U")
        iou = arrays["event_iou"].astype(float)
        rng = np.random.default_rng(42)
        for position, (label, color) in enumerate((("TP", "#1b9e77"), ("FP", "#d95f02"))):
            selected = np.logical_and(status == label, np.isfinite(ambiguity))
            x = position + rng.normal(0.0, 0.035, selected.sum())
            axes[0].scatter(x, ambiguity[selected], s=13, alpha=0.55, color=color, label=label)
        axes[0].set_xticks([0, 1], ["TP", "FP"])
        axes[0].set(ylabel="Ambiguous fraction", ylim=(-0.02, 1.02), title="Event ambiguity: TP vs FP")
        axes[0].grid(axis="y", alpha=0.25)
        selected = np.logical_and.reduce((arrays["event_has_prediction"].astype(bool), np.isfinite(ambiguity), iou >= 0.0))
        axes[1].scatter(ambiguity[selected], iou[selected], s=14, alpha=0.55, color="#377eb8")
        axes[1].set(xlabel="Ambiguous fraction", ylabel="Legacy IoU", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title="Event ambiguity vs IoU")
        axes[1].grid(alpha=0.25)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No event records in test NPZ", ha="center", va="center")
            axis.set_axis_off()
    path = plots_dir / "ambiguity_vs_iou.png"
    _finish(fig, path)
    created.append(path)

    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    if "rejection_threshold" in arrays:
        acceptance = arrays["acceptance_rate"]
        axis.plot(acceptance, arrays["accepted_f1"], "o-", label="F1")
        axis.plot(acceptance, arrays["accepted_precision"], "o-", label="Precision")
        axis.plot(acceptance, arrays["accepted_mean_iou"], "o-", label="Mean IoU")
        for index, (x, y, threshold) in enumerate(
            zip(acceptance, arrays["accepted_f1"], arrays["rejection_threshold"])
        ):
            axis.annotate(
                f"X={threshold:g}",
                (x, y),
                xytext=(4, 5 + 11 * (index % 3)),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set(xlabel="Accepted predicted-event fraction", ylabel="Metric", xlim=(-0.02, 1.08), ylim=(-0.02, 1.08), title="Exploratory conformal rejection curve")
        axis.grid(alpha=0.25)
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No event records in test NPZ", ha="center", va="center")
        axis.set_axis_off()
    path = plots_dir / "rejection_curve.png"
    _finish(fig, path)
    created.append(path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    generated = generate_plots(
        args.results_dir / "conformal_results.npz",
        args.results_dir / "metrics.json",
        args.results_dir / "plots",
    )
    print("\n".join(str(path) for path in generated))


if __name__ == "__main__":
    main()
