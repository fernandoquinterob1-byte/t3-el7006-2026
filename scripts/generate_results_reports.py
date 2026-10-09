"""Generate presentation-ready reports from saved SEED/UQ artifacts.

The script is post-hoc only: it never imports a model, opens an HDF5 dataset,
loads a checkpoint, trains, or performs inference.  It reads the existing
NPZ/JSON exports and writes CSV summaries and PNG figures under
``outputs/results_reports``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_uq.results_analysis import (  # noqa: E402
    ECE_BINS,
    binary_entropy,
    event_metrics,
    fold_number_from_path,
    posterior_dispersion_diagnostics,
    posterior_metrics,
    roc_pr_curves,
    timestep_metrics,
)


SEED_ROOT = REPO_ROOT / "outputs" / "seed_batch8_earlystop"
LAPLACE_ROOT = REPO_ROOT / "outputs" / "laplace_redux"
CONFORMAL_ROOT = REPO_ROOT / "outputs" / "conformal"
REPORT_ROOT = REPO_ROOT / "outputs" / "results_reports"


def _seed_dirs() -> list[Path]:
    return sorted(
        SEED_ROOT.glob("train_eval_0_fold_*_seed_*"),
        key=fold_number_from_path,
    )


def _one_based_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("eval_1_fold_*"), key=fold_number_from_path)


def _write_csv(rows: list[dict], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")
    return frame


def _summary(rows: list[dict], group_key: str = "split") -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    excluded = {"fold", group_key, "alpha", "target_coverage", "q_hat", "valid_timesteps"}
    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    summary_rows = []
    for group, group_frame in frame.groupby(group_key, sort=False):
        for metric in numeric:
            values = group_frame[metric].dropna().to_numpy(dtype=float)
            if values.size:
                summary_rows.append({
                    group_key: group,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "n": int(values.size),
                })
    return pd.DataFrame(summary_rows)


def _save_curves(curve_rows: list[dict], path: Path) -> None:
    _write_csv(curve_rows, path)


def _plot_roc_pr(curve_rows: list[dict], path: Path, title: str) -> None:
    curves = pd.DataFrame(curve_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if curves.empty:
        return
    for fold in sorted(curves["fold"].unique()):
        subset = curves[curves["fold"] == fold]
        roc = subset[subset["curve"] == "roc"]
        pr = subset[subset["curve"] == "pr"]
        if not roc.empty:
            auc = float(roc["score"].iloc[0])
            axes[0].plot(roc["x"], roc["y"], alpha=0.65, label=f"Fold {int(fold)} (AUC={auc:.3f})")
        if not pr.empty:
            ap = float(pr["score"].iloc[0])
            axes[1].plot(pr["x"], pr["y"], alpha=0.65, label=f"Fold {int(fold)} (AP={ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatorio")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC timestep-level")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR timestep-level")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_summary(frame: pd.DataFrame, path: Path, metrics: list[tuple[str, str]], split: str) -> None:
    subset = frame[frame["split"] == split].copy()
    if subset.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        values = subset[metric].to_numpy(dtype=float)
        axis.bar(subset["fold"].astype(str), values, color="#4472c4")
        axis.axhline(np.nanmean(values), color="#d62728", linestyle="--", linewidth=1.5, label="Media")
        axis.set_title(label)
        axis.set_xlabel("Fold")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"Métricas {split} por fold — líneas rojas: media")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_entropy_distribution(rows: list[dict], path: Path, title: str) -> None:
    rng = np.random.default_rng(42)
    correct_values: list[np.ndarray] = []
    incorrect_values: list[np.ndarray] = []
    for row in rows:
        data = np.load(row["path"], allow_pickle=False)
        probability = data[row["probability_key"]]
        target = data["y_true"]
        mask = data["valid_mask"].astype(bool)
        entropy = binary_entropy(probability)
        correct = (probability >= 0.5) == target
        correct_values.append(entropy[mask & correct])
        incorrect_values.append(entropy[mask & ~correct])
    correct = np.concatenate(correct_values)
    incorrect = np.concatenate(incorrect_values)
    if correct.size > 250_000:
        correct = rng.choice(correct, 250_000, replace=False)
    if incorrect.size > 250_000:
        incorrect = rng.choice(incorrect, 250_000, replace=False)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(correct, bins=60, density=True, alpha=0.65, label="Correcto")
    axis.hist(incorrect, bins=60, density=True, alpha=0.65, label="Incorrecto")
    axis.set(xlabel="Predictive entropy (nats)", ylabel="Densidad", title=title)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _add_curves(rows: list[dict], curves: list[dict], fold: int, probability: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
    result = roc_pr_curves(probability, target, mask)
    for x, y, score, curve in [
        (result["fpr"], result["tpr"], result["roc_auc"], "roc"),
        (result["recall"], result["precision"], result["auprc"], "pr"),
    ]:
        # CSVs are for plotting, not for preserving every threshold.  Keep a
        # deterministic visual representation with at most 1,000 points per
        # curve so five folds remain easy to version and inspect.
        if len(x) > 1000:
            indices = np.linspace(0, len(x) - 1, 1000, dtype=int)
            x, y = x[indices], y[indices]
        for index, (x_value, y_value) in enumerate(zip(x, y)):
            rows.append({"fold": fold, "curve": curve, "point": index, "x": float(x_value), "y": float(y_value), "score": float(score)})


def run_seed_report(report_root: Path = REPORT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = report_root / "seed"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    curve_rows: list[dict] = []
    entropy_sources: list[dict] = []
    for folder in _seed_dirs():
        fold = fold_number_from_path(folder)
        for split in ("validation", "test"):
            path = folder / f"{split}_predictions.npz"
            with np.load(path, allow_pickle=False) as data:
                metrics = timestep_metrics(data["probabilities"], data["y_true"], data["valid_mask"])
                metrics.update(event_metrics(data["event_iou_values"]))
                rows.append({"method": "SEED", "fold": fold, "split": split, **metrics})
                if split == "test":
                    _add_curves(curve_rows, curve_rows, fold, data["probabilities"], data["y_true"], data["valid_mask"].astype(bool))
                    entropy_sources.append({"path": path, "probability_key": "probabilities"})
    frame = _write_csv(rows, output / "seed_metrics_by_fold.csv")
    summary = _summary(rows)
    summary.to_csv(output / "seed_metrics_summary.csv", index=False, float_format="%.10g")
    _save_curves(curve_rows, output / "seed_test_roc_pr_curves.csv")
    _plot_roc_pr(curve_rows, output / "seed_test_roc_pr.png", "SEED — curvas de test por fold")
    _plot_metric_summary(
        frame,
        output / "seed_test_metric_summary.png",
        [("roc_auc", "ROC/AUC"), ("auprc", "AUPRC"), ("timestep_f1", "F1 timestep"), ("nll", "NLL"), ("ece", f"ECE ({ECE_BINS} bins, 0–1)"), ("predictive_entropy", "Predictive entropy")],
        "test",
    )
    _plot_entropy_distribution(entropy_sources, output / "seed_entropy_correct_vs_incorrect.png", "SEED — predictive entropy: correcto vs incorrecto")
    (output / "manifest.json").write_text(json.dumps({"method": "SEED", "source_root": str(SEED_ROOT), "no_training_or_inference": True, "ece_bins": ECE_BINS}, indent=2), encoding="utf-8")
    return frame, summary


def run_laplace_report(report_root: Path = REPORT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = report_root / "laplace"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    curve_rows: list[dict] = []
    diagnostics: list[dict] = []
    entropy_sources: list[dict] = []
    for folder in _one_based_dirs(LAPLACE_ROOT):
        fold = fold_number_from_path(folder) - 1
        for split in ("validation", "test"):
            path = folder / f"{split}_predictions.npz"
            with np.load(path, allow_pickle=False) as data:
                metrics, derived = posterior_metrics(data["predictive_probabilities"], data["y_true"], data["valid_mask"].astype(bool), sample_axis=0)
                metrics.update(event_metrics(data["event_iou_values"]))
                rows.append({"method": "Laplace Redux", "fold": fold, "split": split, **metrics})
                diagnostics.append({"fold": fold, "split": split, **posterior_dispersion_diagnostics(data["predictive_probabilities"], data["valid_mask"].astype(bool), sample_axis=0)})
                if split == "test":
                    _add_curves(curve_rows, curve_rows, fold, derived["mean_probability"], data["y_true"], data["valid_mask"].astype(bool))
                    entropy_sources.append({"path": path, "probability_key": "mean_probability"})
    frame = _write_csv(rows, output / "laplace_metrics_by_fold.csv")
    summary = _summary(rows)
    summary.to_csv(output / "laplace_metrics_summary.csv", index=False, float_format="%.10g")
    diagnostics_frame = _write_csv(diagnostics, output / "laplace_posterior_dispersion_diagnostics.csv")
    _save_curves(curve_rows, output / "laplace_test_roc_pr_curves.csv")
    _plot_roc_pr(curve_rows, output / "laplace_test_roc_pr.png", "Laplace Redux — curvas de test por fold")
    _plot_metric_summary(
        frame,
        output / "laplace_test_metric_summary.png",
        [("roc_auc", "ROC/AUC"), ("auprc", "AUPRC"), ("timestep_f1", "F1 timestep"), ("nll", "NLL"), ("ece", f"ECE ({ECE_BINS} bins, 0–1)"), ("mutual_information", "MI/BALD")],
        "test",
    )
    # Use the saved per-timestep arrays for distributions; no inference is run.
    rng = np.random.default_rng(42)
    entropy_values: list[np.ndarray] = []
    expected_values: list[np.ndarray] = []
    mi_values: list[np.ndarray] = []
    for folder in _one_based_dirs(LAPLACE_ROOT):
        path = folder / "test_predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            valid = data["valid_mask"].astype(bool)
            entropy_values.append(data["predictive_entropy"][valid])
            expected_values.append(data["expected_entropy"][valid])
            mi_values.append(data["mutual_information"][valid])
    distributions = [np.concatenate(values) for values in (entropy_values, expected_values, mi_values)]
    distributions = [rng.choice(values, min(values.size, 300_000), replace=False) for values in distributions]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for axis, values, label in zip(axes, distributions, ("Predictive entropy", "Expected entropy", "MI/BALD")):
        axis.hist(values, bins=70, density=True, alpha=0.8, color="#4472c4")
        axis.set_title(label)
        axis.set_xlabel("nats")
        axis.grid(alpha=0.25)
    fig.suptitle("Laplace Redux — distribuciones timestep-level de test")
    fig.savefig(output / "laplace_entropy_mi_distributions.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    diagnostic_test = diagnostics_frame[diagnostics_frame["split"] == "test"]
    axes[0].bar(diagnostic_test["fold"].astype(str), diagnostic_test["sample_probability_std_mean"], color="#70ad47")
    axes[0].set_title("Std. media entre las 30 muestras")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Desviación de probabilidad")
    axes[1].bar(diagnostic_test["fold"].astype(str), diagnostic_test["sample_probability_range_p95"], color="#ed7d31")
    axes[1].set_title("P95 del rango entre muestras")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Máximo − mínimo")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Diagnóstico de MI pequeña: dispersión de la probabilidad predictiva")
    fig.savefig(output / "laplace_posterior_dispersion.png", dpi=180)
    plt.close(fig)
    (output / "manifest.json").write_text(json.dumps({"method": "Laplace Redux last-layer", "source_root": str(LAPLACE_ROOT), "posterior_samples": 30, "posterior_mean_field": "mean_probability", "no_training_or_inference": True, "ece_bins": ECE_BINS}, indent=2), encoding="utf-8")
    return frame, summary, diagnostics_frame


def run_conformal_report(report_root: Path = REPORT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = report_root / "conformal"
    output.mkdir(parents=True, exist_ok=True)
    primary_rows: list[dict] = []
    sweep_rows: list[dict] = []
    base_rows: list[dict] = []
    for folder in _one_based_dirs(CONFORMAL_ROOT):
        fold = fold_number_from_path(folder) - 1
        with np.load(folder / "conformal_results.npz", allow_pickle=False) as data:
            p = data["probabilities"]
            y = data["y_true"]
            mask = data["valid_mask"].astype(bool)
            base_test = timestep_metrics(p, y, mask)
            base_test.update(event_metrics(data["event_iou"]))
            base_rows.append({"method": "SEED predictor base", "fold": fold, "split": "test", **base_test})
            seed_folder = SEED_ROOT / f"train_eval_0_fold_{fold}_seed_42"
            with np.load(seed_folder / "validation_predictions.npz", allow_pickle=False) as calibration:
                base_validation = timestep_metrics(calibration["probabilities"], calibration["y_true"], calibration["valid_mask"])
                base_validation.update(event_metrics(calibration["event_iou_values"]))
            base_rows.append({"method": "SEED predictor base", "fold": fold, "split": "validation/calibration", **base_validation})
            alpha = float(np.asarray(data["alpha"]).reshape(()))
            target_coverage = float(np.asarray(data["target_coverage"]).reshape(()))
            empirical = float(np.asarray(data["empirical_coverage"]).reshape(()))
            primary_rows.append({
                "method": "Conformal Prediction",
                "fold": fold,
                "alpha": alpha,
                "target_coverage": target_coverage,
                "q_hat": float(np.asarray(data["q_hat"]).reshape(())),
                "empirical_coverage": empirical,
                "coverage_gap": empirical - target_coverage,
                "mean_set_size": float(np.asarray(data["mean_set_size"]).reshape(())),
                "singleton_rate": float(np.asarray(data["singleton_rate"]).reshape(())),
                "ambiguous_rate": float(np.asarray(data["ambiguous_rate"]).reshape(())),
                "empty_rate": float(np.asarray(data["empty_rate"]).reshape(())),
            })
            for index, sweep_alpha in enumerate(data["sweep_alpha"]):
                sweep_rows.append({
                    "fold": fold,
                    "alpha": float(sweep_alpha),
                    "target_coverage": float(data["sweep_target_coverage"][index]),
                    "q_hat": float(data["sweep_q_hat"][index]),
                    "empirical_coverage": float(data["sweep_empirical_coverage"][index]),
                    "coverage_gap": float(data["sweep_empirical_coverage"][index] - data["sweep_target_coverage"][index]),
                    "mean_set_size": float(data["sweep_mean_set_size"][index]),
                })
    primary = _write_csv(primary_rows, output / "conformal_primary_metrics_by_fold.csv")
    sweep = _write_csv(sweep_rows, output / "conformal_alpha_sweep.csv")
    base = _write_csv(base_rows, output / "conformal_seed_base_metrics_by_fold.csv")
    base_summary = _summary(base_rows)
    base_summary.to_csv(output / "conformal_seed_base_metrics_summary.csv", index=False, float_format="%.10g")
    sweep_summary = sweep.groupby("alpha", as_index=False).agg(
        target_coverage=("target_coverage", "mean"),
        target_coverage_std=("target_coverage", "std"),
        empirical_coverage=("empirical_coverage", "mean"),
        empirical_coverage_std=("empirical_coverage", "std"),
        coverage_gap=("coverage_gap", "mean"),
        mean_set_size=("mean_set_size", "mean"),
        mean_set_size_std=("mean_set_size", "std"),
    )
    sweep_summary.to_csv(output / "conformal_alpha_sweep_summary.csv", index=False, float_format="%.10g")
    fig, axis = plt.subplots(figsize=(9, 5))
    for fold in sorted(sweep["fold"].unique()):
        subset = sweep[sweep["fold"] == fold]
        axis.plot(subset["alpha"], subset["empirical_coverage"], marker="o", label=f"Fold {int(fold)}")
    target = sweep_summary["target_coverage"].iloc[0]
    axis.axhline(target, color="#d62728", linestyle="--", label=f"Target={target:.2f}")
    axis.set(xlabel="Alpha", ylabel="Empirical coverage", title="Conformal Prediction — coverage sweep")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(output / "conformal_coverage_sweep.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    for fold in sorted(sweep["fold"].unique()):
        subset = sweep[sweep["fold"] == fold]
        axis.plot(subset["alpha"], subset["mean_set_size"], marker="o", label=f"Fold {int(fold)}")
    axis.set(xlabel="Alpha", ylabel="Mean prediction set size", title="Conformal Prediction — set-size sweep")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(output / "conformal_set_size_sweep.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5))
    primary_plot = primary.set_index("fold")[["singleton_rate", "ambiguous_rate", "empty_rate"]]
    primary_plot.plot(kind="bar", ax=axis, color=["#70ad47", "#ed7d31", "#a5a5a5"])
    axis.set(xlabel="Fold", ylabel="Rate", title=f"Conformal set composition at saved alpha={primary['alpha'].iloc[0]:.2f}")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "conformal_set_composition.png", dpi=180)
    plt.close(fig)
    (output / "manifest.json").write_text(json.dumps({"method": "Split Conformal Prediction", "source_root": str(CONFORMAL_ROOT), "base_predictor": "saved SEED probabilities", "saved_alpha_values": sorted(sweep["alpha"].unique().tolist()), "primary_alpha_in_artifacts": sorted(primary["alpha"].unique().tolist()), "no_training_or_inference": True, "ece_bins": ECE_BINS}, indent=2), encoding="utf-8")
    return primary, sweep, base, base_summary


def run_all_reports(report_root: Path = REPORT_ROOT) -> None:
    run_seed_report(report_root)
    run_laplace_report(report_root)
    run_conformal_report(report_root)


if __name__ == "__main__":
    run_all_reports()
    print(f"Reportes escritos en {REPORT_ROOT}")
