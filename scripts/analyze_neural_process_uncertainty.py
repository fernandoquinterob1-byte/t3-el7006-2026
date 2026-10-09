"""Post-hoc Neural Process uncertainty metrics and temporal figures.

This script only reads the exported Neural Process NPZ files.  It does not
load a model, train, or run inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_uq.results_analysis import posterior_metrics  # noqa: E402


LOG2 = float(np.log(2.0))
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "results_reports" / "neural_process"


def _fold_number(path: Path) -> int:
    return int(path.name.split("fold_", 1)[1].split("_", 1)[0])


def _fold_dirs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.glob("train_eval_0_fold_*_seed_*") if path.is_dir()),
        key=_fold_number,
    )


def _valid_error_auc(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool) & np.isfinite(score)
    labels = (y_pred != y_true).astype(np.int8)
    labels = labels[valid].reshape(-1)
    values = np.asarray(score)[valid].reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def _metric_rows(root: Path) -> tuple[list[dict], dict[int, np.ndarray]]:
    rows: list[dict] = []
    mi_by_fold: dict[int, np.ndarray] = {}
    for folder in _fold_dirs(root):
        fold = _fold_number(folder)
        path = folder / "test_predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            y_true = data["y_true"].astype(np.int8)
            mask = data["valid_mask"].astype(bool)
            mean_probability = data["probabilities"].astype(np.float32)
            y_pred = data["y_pred"].astype(np.int8)
            samples = data["predictive_samples"].astype(np.float32)
            posterior, derived = posterior_metrics(
                samples,
                y_true,
                mask,
                sample_axis=1,
            )
            mi_bits = derived["mutual_information"].astype(np.float32) / LOG2
            entropy_bits = derived["predictive_entropy"].astype(np.float32) / LOG2
            expected_entropy_bits = derived["expected_entropy"].astype(np.float32) / LOG2
            mi_valid = mi_bits[mask & np.isfinite(mi_bits)]
            mi_by_fold[fold] = mi_valid
            row = {
                "method": "Neural Process",
                "fold": fold,
                "split": "test",
                "predictive_samples": int(samples.shape[1]),
                "valid_timesteps": int(mask.sum()),
                "roc_auc": posterior["roc_auc"],
                "auprc": posterior["auprc"],
                "timestep_f1": posterior["timestep_f1"],
                "timestep_precision": posterior["timestep_precision"],
                "timestep_recall": posterior["timestep_recall"],
                "nll": posterior["nll"],
                "ece": posterior["ece"],
                "predictive_entropy_nats": posterior["predictive_entropy"],
                "expected_entropy_nats": posterior["expected_entropy"],
                "mutual_information_nats": posterior["mutual_information"],
                "predictive_entropy_bits": float(np.mean(entropy_bits[mask])),
                "expected_entropy_bits": float(np.mean(expected_entropy_bits[mask])),
                "mutual_information_bits": float(np.mean(mi_bits[mask])),
                "uncertainty_auroc": _valid_error_auc(y_true, y_pred, mi_bits, mask),
                "source_file": str(path.resolve()),
            }
            rows.append(row)
    return rows, mi_by_fold


def _write_metrics(rows: list[dict], output: Path) -> None:
    import pandas as pd

    frame = pd.DataFrame(rows).sort_values("fold")
    frame.to_csv(output / "neural_process_test_metrics_by_fold.csv", index=False, float_format="%.10g")
    summary_rows = []
    for metric in (
        "roc_auc",
        "uncertainty_auroc",
        "auprc",
        "timestep_f1",
        "nll",
        "ece",
        "predictive_entropy_bits",
        "expected_entropy_bits",
        "mutual_information_bits",
    ):
        values = frame[metric].dropna().to_numpy(dtype=float)
        if values.size:
            summary_rows.append({
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n_folds": int(values.size),
            })
    pd.DataFrame(summary_rows).to_csv(
        output / "neural_process_test_metrics_summary.csv", index=False, float_format="%.10g"
    )
    (output / "neural_process_test_metrics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )


def _plot_mi_boxplot(mi_by_fold: dict[int, np.ndarray], output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([values for values in mi_by_fold.values() if values.size])
    values = rng.choice(pooled, size=min(200_000, pooled.size), replace=False)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = values[(values < lower_fence) | (values > upper_fence)]
    ax.boxplot(
        [values],
        tick_labels=["Promedio de folds"],
        showfliers=False,
        showmeans=True,
        meanline=True,
        meanprops={"color": "#d62728", "linewidth": 1.8},
    )
    # Draw every sampled outlier explicitly: default boxplot fliers can become
    # effectively invisible at this scale. Horizontal jitter reveals density.
    ax.scatter(
        1.0 + rng.uniform(-0.12, 0.12, size=outliers.size),
        outliers,
        s=3,
        color="#3569a8",
        alpha=0.10,
        linewidths=0,
        zorder=3,
        rasterized=True,
    )
    ax.set_xlim(0.5, 1.5)
    ax.set_ylabel("Mutual information / BALD (bits)")
    ax.set_title("Neural Process — MI/BALD en test (incluye atípicos; línea roja = media)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "neural_process_test_mi_boxplot_bits.png", dpi=180)
    plt.close(fig)


def _plot_roc(rows: list[dict], root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    grid = np.linspace(0.0, 1.0, 501)
    detection_curves, uncertainty_curves = [], []
    for folder in _fold_dirs(root):
        fold = _fold_number(folder)
        path = folder / "test_predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            y = data["y_true"].astype(np.int8)
            mask = data["valid_mask"].astype(bool)
            p = data["probabilities"].astype(np.float32)
            samples = data["predictive_samples"].astype(np.float32)
            _, derived = posterior_metrics(samples, y, mask, sample_axis=1)
            mi = derived["mutual_information"].astype(np.float32) / LOG2
            pred = data["y_pred"].astype(np.int8)
            valid = mask.reshape(-1)
            fpr, tpr, _ = roc_curve(y.reshape(-1)[valid], p.reshape(-1)[valid])
            error = (pred != y).astype(np.int8).reshape(-1)[valid]
            mi_flat = mi.reshape(-1)[valid]
            error_fpr, error_tpr, _ = roc_curve(error, mi_flat)
            detection_curves.append(np.interp(grid, fpr, tpr))
            uncertainty_curves.append(np.interp(grid, error_fpr, error_tpr))
    for ax, curves, metric in (
        (axes[0], detection_curves, "roc_auc"),
        (axes[1], uncertainty_curves, "uncertainty_auroc"),
    ):
        mean_curve = np.mean(curves, axis=0)
        mean_curve[0], mean_curve[-1] = 0.0, 1.0
        mean_auc = float(np.mean([row[metric] for row in rows]))
        ax.plot(grid, mean_curve, color="#0072b2", lw=2, label=f"Media entre folds (AUC={mean_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="AUROC de detección")
    axes[1].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="AUROC de incertidumbre (error vs MI)",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Neural Process — curvas ROC en test")
    fig.savefig(output / "neural_process_test_roc_uncertainty.png", dpi=180)
    plt.close(fig)


def _binary_axis(ax, time: np.ndarray, label: np.ndarray, ylabel: str) -> None:
    ax.step(time, np.nan_to_num(label, nan=np.nan), where="post", color="#0072b2", lw=1.2)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 1], ["Fondo", "Evento"])
    ax.set_ylabel(ylabel)


def _plot_temporal(
    *,
    root: Path,
    output: Path,
    data_root: Path,
    fold: int | None,
    sample_id: int,
    segment_start: int,
    window_seconds: float,
    sample_rate_hz: int,
    channel_index: int,
) -> Path:
    fold_data = []
    for folder in _fold_dirs(root):
        path = folder / "test_predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            matches = np.flatnonzero(data["sample_id"] == sample_id)
            if len(matches) != 1:
                continue
            index = int(matches[0])
            y = data["y_true"][index].astype(float)
            valid = data["valid_mask"][index].astype(bool)
            _, derived = posterior_metrics(data["predictive_samples"][index][None, ...], y[None, ...], valid[None, ...], sample_axis=1)
            fold_data.append((y, valid, derived["mutual_information"][0] / LOG2,
                              data["seed_probabilities"][index].astype(float),
                              data["processed_pred"][index].astype(float)))
    if not fold_data:
        raise ValueError(f"Sample {sample_id} was not found in any test fold.")
    y_true_full, valid_full = fold_data[0][0], np.logical_and.reduce([item[1] for item in fold_data])
    mi_bits_full = np.nanmean(np.stack([item[2] for item in fold_data]), axis=0)
    seed_pred_full = (np.mean(np.stack([item[3] for item in fold_data]), axis=0) >= 0.5).astype(float)
    np_pred_full = (np.mean(np.stack([item[4] for item in fold_data]), axis=0) >= 0.5).astype(float)
    np_pred_full[~valid_full] = np.nan
    seed_pred_full[~valid_full] = np.nan
    mi_bits_full[~valid_full] = np.nan

    raw_h5 = data_root / "moda_moda.h5"
    with h5py.File(raw_h5, "r") as source:
        sample_signal = np.asarray(source["signals"][sample_id], dtype=float)
    signal_full = sample_signal[channel_index] if sample_signal.ndim > 1 else sample_signal
    end = segment_start + int(window_seconds * sample_rate_hz)
    if end > len(signal_full):
        raise ValueError(f"Temporal segment ends at {end}, signal length is {len(signal_full)}.")
    sl = slice(segment_start, end)
    y_true = y_true_full[sl]
    valid = valid_full[sl]
    signal = signal_full[sl]
    seed_pred = seed_pred_full[sl]
    np_pred = np_pred_full[sl]
    mi_bits = mi_bits_full[sl]
    time = np.arange(len(signal), dtype=float) / sample_rate_hz

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 10.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 0.7, 0.7, 0.7, 0.9]},
    )
    event_mask = np.nan_to_num(y_true, nan=0.0) > 0.5
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.axvspan(time[0], time[-1], color="0.95", zorder=-3)
        if np.any(event_mask):
            ax.fill_between(
                time,
                0,
                1,
                where=event_mask,
                transform=ax.get_xaxis_transform(),
                color="#b8e3cf",
                alpha=0.35,
                step="post",
                zorder=-2,
            )
    axes[0].plot(time, signal, color="#263238", lw=0.7)
    axes[0].set_ylabel("EEG (microV)")
    axes[0].set_title(f"Neural Process - promedio entre folds, sample {sample_id}, segmento de {window_seconds:g} s")
    _binary_axis(axes[1], time, y_true, "y_true")
    _binary_axis(axes[2], time, seed_pred, "SEED\ny predict")
    _binary_axis(axes[3], time, np_pred, "Neural Process\ny predict")
    axes[4].plot(time, mi_bits, color="#d81b60", lw=1.2)
    axes[4].set_ylabel("NP\nMI (bits)")
    axes[4].yaxis.set_major_locator(MaxNLocator(nbins=4))
    for ax in axes:
        ax.set_xlim(time[0], time[-1])
    axes[4].set_xlabel("Tiempo dentro del segmento (s)")
    fig.suptitle("Neural Process: detección binaria e incertidumbre - sin probabilidades", fontsize=14)
    output_path = output / f"neural_process_temporal_detection_uncertainty_mean_sample_{sample_id}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--np-root", type=Path, default=REPO_ROOT / "outputs" / "neural_process")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("EEG_DATA_ROOT", Path.home() / "Downloads" / "data" / "data")))
    parser.add_argument("--temporal-fold", type=int, default=None, help="Deprecated; temporal plot now averages all folds.")
    parser.add_argument("--sample-id", type=int, default=261)
    parser.add_argument("--segment-start", type=int, default=12000)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--sample-rate-hz", type=int, default=200)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows, mi_by_fold = _metric_rows(args.np_root)
    if not rows:
        raise FileNotFoundError(f"No test_predictions.npz found under {args.np_root}")
    _write_metrics(rows, args.output)
    _plot_mi_boxplot(mi_by_fold, args.output, args.seed)
    _plot_roc(rows, args.np_root, args.output)
    temporal_path = _plot_temporal(
        root=args.np_root,
        output=args.output,
        data_root=args.data_root,
        fold=args.temporal_fold,
        sample_id=args.sample_id,
        segment_start=args.segment_start,
        window_seconds=args.window_seconds,
        sample_rate_hz=args.sample_rate_hz,
        channel_index=args.channel_index,
    )
    manifest = {
        "method": "Neural Process",
        "source_root": str(args.np_root.resolve()),
        "split": "test",
        "mutual_information": "predictive_entropy(mean samples) - mean sample entropy, converted from nats to bits",
        "uncertainty_auroc": "AUROC for detecting timestep errors using MI/BALD bits",
        "temporal_plot": str(temporal_path.resolve()),
        "no_training_or_inference": True,
        "predictive_samples": 16,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "folds": [row["fold"] for row in rows], "temporal_plot": str(temporal_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
