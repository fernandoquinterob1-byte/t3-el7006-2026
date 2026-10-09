"""Every analysis discussed in the project, for one uncertainty method.

``method_results`` reads the exported test NPZ files of evaluations 0 and 1
(no inference, no training) and returns one table per analysis, with an
``evaluación`` and a ``fold`` column.  Results are cached as a pickle so the
comparison notebook does not recompute them.  Plot helpers draw each analysis
per evaluation and for both evaluations together.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .method_summary import (
    CALIBRATION_BINS,
    entropy_bits,
    event_uncertainty,
    load_fold,
    positive_class_ece,
)
from .metrics import SAMPLE_RATE_HZ, calculate_metrics_simple

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs"
CACHE = OUT / "results_reports" / "analysis"
CACHE_VERSION = 1
EVALUATIONS = (0, 1)
FOLDS = range(5)
NEAR = SAMPLE_RATE_HZ  # ±1 s around true or predicted events
BORDER = SAMPLE_RATE_HZ // 4  # ±0.25 s around true onsets/offsets
ROC_GRID = np.linspace(0.0, 1.0, 201)
DURATION_BINS = np.array([0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
DEFERRAL = (0.0, 0.05, 0.10, 0.20, 0.30)
REJECTION = (0.5, 1, 2, 5, 10, 20)
GROUPS = ["TP (predicho)", "FP (predicho)", "TP (real)", "FN (real)"]

METHODS = {
    "SEED": ("seed", None),
    "Laplace": ("sampling", "laplace_redux"),
    "Temporal Ensemble": ("sampling", "temporal_ensemble"),
    "MC Dropout": ("sampling", "mc_dropout"),
    "Neural Process": ("neural_process", None),
    "Deep Ensemble": ("sampling", "deep_ensemble"),
}
COLORS = {
    "SEED": "#52514e", "Laplace": "#2a78d6", "Temporal Ensemble": "#eda100",
    "MC Dropout": "#1baf7a", "Neural Process": "#e87ba4", "Deep Ensemble": "#4a3aa7",
}
GROUP_COLORS = {"TP (predicho)": "#2a78d6", "FP (predicho)": "#eb6834", "TP (real)": "#7fb3ec", "FN (real)": "#e34948"}
CHANCE = "#52514e"


def method_path(method: str, evaluation: int, fold: int, split: str = "test") -> Path:
    kind, folder = METHODS[method]
    if kind == "seed":
        return OUT / "seed_batch8_earlystop" / f"train_eval_{evaluation}_fold_{fold}_seed_42" / f"{split}_predictions.npz"
    if kind == "neural_process":
        return OUT / "neural_process" / f"train_eval_{evaluation}_fold_{fold}_seed_42" / f"{split}_predictions.npz"
    return OUT / folder / f"eval_{evaluation + 1}_fold_{fold + 1}" / f"{split}_predictions.npz"


# ---------------------------------------------------------------- per fold

def _auc(target, score, average=False):
    target = np.asarray(target, dtype=bool)
    if target.all() or not target.any():
        return np.nan
    return float(average_precision_score(target, score) if average else roc_auc_score(target, score))


def _roc(target, score):
    fpr, tpr, _ = roc_curve(target, score)
    curve = np.interp(ROC_GRID, fpr, tpr)
    curve[0], curve[-1] = 0.0, 1.0
    return curve


def _measures(fold, seed_entropy):
    """Uncertainty arrays available for this method (MI only for sampling methods)."""
    measures = {"total": fold["total"], "aleatoria": fold["aleatoric"], "SEED H(p)": seed_entropy}
    if fold["mi"] is not None:
        measures["MI"] = fold["mi"]
    return measures


def _fold_analysis(method: str, evaluation: int, fold_index: int) -> dict[str, object]:
    kind = METHODS[method][0]
    fold = load_fold(method_path(method, evaluation, fold_index), kind)
    seed = load_fold(method_path("SEED", evaluation, fold_index), "seed")
    if not np.array_equal(seed["sample_id"], fold["sample_id"]):
        raise ValueError(f"{method} and SEED test records differ (eval {evaluation}, fold {fold_index}).")
    seed_entropy = seed["total"]
    valid = fold["valid_mask"].astype(bool)
    truth = fold["y_true"].astype(bool)
    error = fold["y_pred"].astype(bool) != truth
    near = valid & binary_dilation(truth | fold["processed_pred"].astype(bool), structure=np.ones((1, 2 * NEAR + 1), bool))
    transitions = np.zeros_like(truth)
    transitions[:, 1:] = truth[:, 1:] != truth[:, :-1]
    border = valid & binary_dilation(transitions, structure=np.ones((1, 2 * BORDER + 1), bool))
    zones = {"Borde del huso": border, "Interior del huso": valid & truth & ~border, "Fondo": valid & ~truth & ~border}
    measures = _measures(fold, seed_entropy)
    key = "MI" if "MI" in measures else "total"  # main uncertainty score of the method
    p, y = fold["probability"][valid], truth[valid]
    curve = calculate_metrics_simple(fold["event_iou_values"].tolist())
    index = curve["selected_index"]
    tag = {"evaluación": evaluation, "fold": fold_index}

    row = {
        **tag,
        "F1 evento (%)": 100 * float(curve["f1"][index]),
        "precisión (%)": 100 * float(curve["precision"][index]),
        "recall (%)": 100 * float(curve["recall"][index]),
        "AUROC detección": float(roc_auc_score(y, p)),
        "AUPRC detección": float(average_precision_score(y, p)),
        "NLL": float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (~y) * np.log(np.clip(1 - p, 1e-6, 1)))),
        "Brier": float(np.mean((p - y) ** 2)),
        "ECE clase positiva": positive_class_ece(p, y.astype(float)),
        "azar errores": float(error[near].mean()),
    }
    for zone_name, mask in (("registro", valid), ("cerca de eventos", near)):
        row[f"total ({zone_name})"] = float(fold["total"][mask].mean())
        row[f"aleatoria ({zone_name})"] = float(fold["aleatoric"][mask].mean())
        row[f"SEED H(p) ({zone_name})"] = float(seed_entropy[mask].mean())
        if fold["mi"] is not None:
            row[f"MI ({zone_name})"] = float(fold["mi"][mask].mean())
            row[f"% epistémica ({zone_name})"] = 100 * row[f"MI ({zone_name})"] / row[f"total ({zone_name})"]
    for name, values in measures.items():
        row[f"AUROC errores: {name}"] = _auc(error[near], values[near])
        row[f"AUPRC errores: {name}"] = _auc(error[near], values[near], average=True)

    rocs = pd.DataFrame({"curva": "detección", "fpr": ROC_GRID, "tpr": _roc(y, p)})
    for name in ("MI", "total", "SEED H(p)"):
        if name in measures:
            rocs = pd.concat([rocs, pd.DataFrame({"curva": f"errores: {name}", "fpr": ROC_GRID, "tpr": _roc(error[near], measures[name][near])})])
    rocs = rocs.assign(**tag)

    bins = np.clip(np.digitize(np.clip(p, 1e-6, 1 - 1e-6), CALIBRATION_BINS[1:-1]), 0, len(CALIBRATION_BINS) - 2)
    counts = np.bincount(bins, minlength=len(CALIBRATION_BINS) - 1)
    reliability = pd.DataFrame({
        "bin": np.arange(len(counts)),
        "probabilidad predicha": np.bincount(bins, weights=p, minlength=len(counts)) / np.maximum(counts, 1),
        "frecuencia observada": np.bincount(bins, weights=y, minlength=len(counts)) / np.maximum(counts, 1),
        "n": counts,
    }).assign(**tag)
    seed_p = seed["probability"][valid]
    seed_bins = np.clip(np.digitize(np.clip(seed_p, 1e-6, 1 - 1e-6), CALIBRATION_BINS[1:-1]), 0, len(CALIBRATION_BINS) - 2)
    seed_counts = np.bincount(seed_bins, minlength=len(counts))
    reliability["SEED probabilidad"] = np.bincount(seed_bins, weights=seed_p, minlength=len(counts)) / np.maximum(seed_counts, 1)
    reliability["SEED observada"] = np.bincount(seed_bins, weights=y, minlength=len(counts)) / np.maximum(seed_counts, 1)
    reliability["SEED n"] = seed_counts

    zone_rows = []
    for zone_name, mask in zones.items():
        zone_row = {**tag, "zona": zone_name, "% del registro": 100 * mask.sum() / valid.sum(),
                    "SEED H(p)": float(seed_entropy[mask].mean()), "aleatoria": float(fold["aleatoric"][mask].mean())}
        if fold["mi"] is not None:
            zone_row["MI"] = float(fold["mi"][mask].mean())
        zone_rows.append(zone_row)

    events = event_uncertainty(fold, extra={"SEED H(p)": seed_entropy}).assign(**tag)
    events["grupo"] = events["tipo"] + " (" + events["tramo"] + ")"
    score_name = "epistémica (MI)" if fold["mi"] is not None else "incertidumbre total"
    predicted = events[events["tramo"] == "predicho"]
    rho = spearmanr(predicted["duración (s)"], predicted[score_name])[0] if len(predicted) > 2 else np.nan
    row["ρ duración-incertidumbre"] = float(rho)
    duration = predicted.assign(rango=pd.cut(predicted["duración (s)"], DURATION_BINS, include_lowest=True))
    duration = duration.groupby("rango", observed=False)[score_name].mean().rename("incertidumbre").reset_index()
    duration["centro (s)"] = 0.5 * (DURATION_BINS[:-1] + DURATION_BINS[1:])
    duration = duration.drop(columns="rango").assign(**tag)

    # Deferral: send the most uncertain detections to review.
    deferral_rows = []
    fn_count = int(np.isin(fold["event_status"], ["FN", "LOW_IOU"]).sum())
    is_fp = (predicted["tipo"] == "FP").to_numpy()
    for name in ("incertidumbre total", "epistémica (MI)", "SEED H(p)"):
        scores = predicted[name].to_numpy()
        if np.isnan(scores).all():
            continue
        order = np.argsort(-scores)
        for fraction in DEFERRAL:
            deferred = np.zeros(len(order), bool)
            deferred[order[: int(round(fraction * len(order)))]] = True
            deferral_rows.append({**tag, "medida": name, "% derivado": 100 * fraction,
                                  "precisión aceptado (%)": 100 * (~is_fp[~deferred]).mean(),
                                  "% de FP capturados": 100 * (is_fp & deferred).sum() / max(is_fp.sum(), 1)})
        # Automatic rejection of the most uncertain detections (TP removed become misses).
        tp0, fp0 = (~is_fp).sum(), is_fp.sum()
        base = 2 * tp0 / (2 * tp0 + fp0 + fn_count)
        for percent in REJECTION:
            removed = np.zeros(len(order), bool)
            removed[order[: max(1, int(round(percent / 100 * len(order))))]] = True
            tp_rm, fp_rm = (removed & ~is_fp).sum(), (removed & is_fp).sum()
            f1 = 2 * (tp0 - tp_rm) / (2 * (tp0 - tp_rm) + (fp0 - fp_rm) + fn_count + tp_rm)
            deferral_rows.append({**tag, "medida": name, "% descartado": percent, "FP descartados": fp_rm,
                                  "TP descartados": tp_rm, "% FP en lo descartado": 100 * fp_rm / removed.sum(),
                                  "cambio F1 (puntos)": 100 * (f1 - base)})
    deferral = pd.DataFrame(deferral_rows)

    return {"metrics": row, "roc": rocs, "reliability": reliability, "zones": pd.DataFrame(zone_rows),
            "events": events, "duration": duration, "deferral": deferral}


# ---------------------------------------------------------------- per method

def method_info(method: str) -> pd.DataFrame | None:
    """Method-specific settings per fold (prior, τ, epochs, members)."""
    rows = []
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            tag = {"evaluación": evaluation, "fold": fold}
            if method == "Laplace":
                fit = json.loads((OUT / "laplace_redux" / f"eval_{evaluation + 1}_fold_{fold + 1}" / "metrics.json").read_text(encoding="utf-8"))["fit"]
                rows.append({**tag, "τ (autocorrelación)": fit["likelihood_temperature"], "puntos efectivos": fit["effective_train_points"],
                             "prior elegido": fit["prior_precision"]})
            elif method == "Neural Process":
                info = json.loads((OUT / "neural_process" / f"train_eval_{evaluation}_fold_{fold}_seed_42" / "metrics.json").read_text(encoding="utf-8"))
                rows.append({**tag, "mejor época": info["best_epoch"], "épocas entrenadas": info["epochs_completed"]})
            elif method == "Deep Ensemble":
                row = dict(tag)
                for seed in (42, 1, 2, 3, 4):
                    report = json.loads((OUT / "seed_batch8_earlystop" / f"train_eval_{evaluation}_fold_{fold}_seed_{seed}" / "test_metrics.json").read_text(encoding="utf-8"))
                    row[f"F1 miembro seed {seed}"] = report["f1_score"]
                rows.append(row)
            elif method == "SEED":
                report = json.loads((OUT / "seed_batch8_earlystop" / f"train_eval_{evaluation}_fold_{fold}_seed_42" / "metrics.json").read_text(encoding="utf-8"))
                rows.append({**tag, "épocas entrenadas": len(report.get("history", []))})
    return pd.DataFrame(rows) if rows else None


def method_results(method: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """All analyses of one method for evaluations 0 and 1 (cached)."""
    cache = CACHE / f"{method.lower().replace(' ', '_')}.pkl"
    if cache.is_file() and not refresh:
        with cache.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored["results"]
    parts: dict[str, list] = {}
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            for name, value in _fold_analysis(method, evaluation, fold).items():
                parts.setdefault(name, []).append(value)
        print(f"{method}: evaluación {evaluation} lista", flush=True)
    results = {name: pd.DataFrame(values) if name == "metrics" else pd.concat(values, ignore_index=True) for name, values in parts.items()}
    results["info"] = method_info(method)
    CACHE.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "results": results}, handle)
    return results


def by_evaluation(table: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mean ± std per evaluation and for both evaluations together (10 folds)."""
    columns = columns or [c for c in table.columns if c not in ("evaluación", "fold")]
    blocks = {}
    for evaluation in EVALUATIONS:
        part = table[table["evaluación"] == evaluation][columns]
        blocks[f"eval {evaluation}"] = part.mean().map("{:.4g}".format) + " ± " + part.std().map("{:.2g}".format)
    both = table[columns]
    blocks["promedio eval 0 y 1"] = both.mean().map("{:.4g}".format) + " ± " + both.std().map("{:.2g}".format)
    return pd.DataFrame(blocks).replace("nan ± nan", "—")


def event_separation_folds(events: pd.DataFrame, negative: str, positive: str) -> pd.DataFrame:
    """Per-fold AUROC and AUPRC of each measure to separate two event groups."""
    measures = [m for m in ("probabilidad media", "incertidumbre total", "aleatoria", "epistémica (MI)",
                            "dispersión entre muestras (std)", "SEED H(p)") if m in events and events[m].notna().any()]
    rows = []
    for (evaluation, fold), group in events.groupby(["evaluación", "fold"]):
        group = group[group["grupo"].isin([negative, positive])]
        target = (group["grupo"] == positive).to_numpy()
        row = {"evaluación": evaluation, "fold": fold, "azar AUPRC": target.mean()}
        for measure in measures:
            score = group[measure].to_numpy()
            score = -score if measure == "probabilidad media" else score
            row[f"AUROC {measure}"] = _auc(target, score)
            row[f"AUPRC {measure}"] = _auc(target, score, average=True)
        rows.append(row)
    return pd.DataFrame(rows)


def event_separation(events: pd.DataFrame, negative: str, positive: str) -> pd.DataFrame:
    """AUROC and AUPRC of each measure to separate two event groups, per evaluation and together."""
    return by_evaluation(event_separation_folds(events, negative, positive))


def event_medians(events: pd.DataFrame) -> pd.DataFrame:
    measures = [m for m in ("probabilidad media", "incertidumbre total", "aleatoria", "epistémica (MI)",
                            "dispersión entre muestras (std)") if events[m].notna().any()]
    blocks = {}
    for label, part in [(f"eval {e}", events[events["evaluación"] == e]) for e in EVALUATIONS] + [("promedio eval 0 y 1", events)]:
        medians = part.groupby("grupo")[measures].median().loc[GROUPS]
        medians.columns = pd.MultiIndex.from_product([[label], medians.columns])
        blocks[label] = medians
    return pd.concat(blocks.values(), axis=1).round(4)


# ---------------------------------------------------------------- plots

def _panels(title, ncols=3, width=5.2, height=4.4, sharey=True):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), sharey=sharey, constrained_layout=True)
    fig.suptitle(title)
    return fig, np.atleast_1d(axes)


def _scopes(table):
    return [(f"Evaluación {e}", table[table["evaluación"] == e]) for e in EVALUATIONS] + [("Promedio eval 0 y 1", table)]


def plot_roc(results, method):
    roc = results["roc"]
    curves = [c for c in ("detección", "errores: MI", "errores: total", "errores: SEED H(p)") if c in set(roc["curva"])]
    styles = {"detección": ("#2a78d6", "-"), "errores: MI": (COLORS.get(method, "#eb6834"), "-"),
              "errores: total": ("#1baf7a", "--"), "errores: SEED H(p)": (CHANCE, ":")}
    metrics = results["metrics"]
    fig, axes = _panels(f"{method}: curvas ROC promedio de los folds (errores: solo cerca de eventos)")
    for ax, (label, part) in zip(axes, _scopes(roc)):
        scope_metrics = metrics if label.startswith("Promedio") else metrics[metrics["evaluación"] == int(label[-1])]
        for curve in curves:
            mean = part[part["curva"] == curve].groupby("fpr")["tpr"].mean()
            auc_key = "AUROC detección" if curve == "detección" else "AUROC errores: " + curve.split(": ")[1]
            color, style = styles[curve]
            ax.plot(mean.index, mean.values, style, color=color, lw=2, label=f"{curve} (AUC {scope_metrics[auc_key].mean():.3f})")
        ax.plot([0, 1], [0, 1], "--", color=CHANCE, lw=1, label="azar")
        ax.set(title=label, xlabel="Tasa de falsos positivos")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Tasa de verdaderos positivos")
    return fig


def plot_reliability(results, method):
    rel = results["reliability"]
    fig, axes = _panels(f"{method}: diagrama de confiabilidad de la clase positiva (promedio de folds)")
    for ax, (label, part) in zip(axes, _scopes(rel)):
        own = part[part["n"] > 0].groupby("bin")[["probabilidad predicha", "frecuencia observada"]].mean()
        base = part[part["SEED n"] > 0].groupby("bin")[["SEED probabilidad", "SEED observada"]].mean()
        ax.plot([0, 1], [0, 1], "--", color=CHANCE, lw=1, label="calibración perfecta")
        if method != "SEED":
            ax.plot(base["SEED probabilidad"], base["SEED observada"], "o-", color=COLORS["SEED"], ms=4, label="SEED")
        ax.plot(own["probabilidad predicha"], own["frecuencia observada"], "o-", color=COLORS.get(method, "#2a78d6"), ms=4, label=method)
        ax.set(title=label, xlabel="Probabilidad predicha", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frecuencia observada de huso")
    return fig


def plot_decomposition(results, method):
    metrics = results["metrics"]
    fig, axes = _panels(f"{method}: descomposición de la incertidumbre (bits, promedio de folds)")
    for ax, (label, part) in zip(axes, _scopes(metrics)):
        x = np.arange(2)
        zones = ["registro", "cerca de eventos"]
        seed = [part[f"SEED H(p) ({z})"].mean() for z in zones]
        ax.bar(x - 0.2, seed, 0.38, color=COLORS["SEED"], label="SEED H(p)")
        if method != "SEED":
            ale = [part[f"aleatoria ({z})"].mean() for z in zones]
            ax.bar(x + 0.2, ale, 0.38, color="#7fb3ec", label=f"{method}: aleatoria")
            if f"MI ({zones[0]})" in part:
                mi = [part[f"MI ({z})"].mean() for z in zones]
                ax.bar(x + 0.2, mi, 0.38, bottom=ale, color=COLORS.get(method, "#eb6834"), label=f"{method}: epistémica (MI)")
                for xi, a, m in zip(x, ale, mi):
                    ax.text(xi + 0.2, a + m + 0.004, f"{100 * m / (a + m):.1f}% epist.", ha="center", fontsize=8)
        ax.set_xticks(x, ["Todo el registro", "Cerca de eventos"])
        ax.set(title=label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Entropía media [bits]")
    return fig


def plot_error_detection(results, method):
    metrics = results["metrics"]
    names = ["SEED H(p)"] if method == "SEED" else [n for n in ("SEED H(p)", "total", "aleatoria", "MI") if f"AUROC errores: {n}" in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    fig.suptitle(f"{method}: detección de errores cerca de eventos (barra = ± desv. est. de los 10 folds)")
    width = 0.26
    for ax, kind in zip(axes, ("AUROC", "AUPRC")):
        for offset, (label, part) in zip((-width, 0, width), _scopes(metrics)):
            means = [part[f"{kind} errores: {n}"].mean() for n in names]
            stds = [part[f"{kind} errores: {n}"].std() for n in names]
            ax.bar(np.arange(len(names)) + offset, means, width, yerr=stds, capsize=3, label=label,
                   color={"Evaluación 0": "#7fb3ec", "Evaluación 1": "#2a78d6", "Promedio eval 0 y 1": "#1f4e8c"}[label])
        chance = 0.5 if kind == "AUROC" else metrics["azar errores"].mean()
        ax.axhline(chance, ls="--", color=CHANCE, lw=1, label=f"azar ({chance:.2f})")
        ax.set_xticks(range(len(names)), names)
        ax.set(title=kind, ylim=(0, 1))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    return fig


def plot_zones(results, method):
    zones = results["zones"]
    names = [n for n in ("SEED H(p)", "aleatoria", "MI") if n in zones and not (method == "SEED" and n == "aleatoria")]
    order = ["Borde del huso", "Interior del huso", "Fondo"]
    fig, axes = _panels(f"{method}: incertidumbre por zona (bits, promedio de folds; barra = ± desv. est.)")
    colors = {"SEED H(p)": COLORS["SEED"], "aleatoria": "#7fb3ec", "MI": COLORS.get(method, "#eb6834")}
    for ax, (label, part) in zip(axes, _scopes(zones)):
        stats = part.groupby("zona")[names].agg(["mean", "std"]).loc[order]
        share = part.groupby("zona")["% del registro"].mean().loc[order]
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            ax.bar(np.arange(3) + (i - (len(names) - 1) / 2) * width, stats[(name, "mean")], width,
                   yerr=stats[(name, "std")], capsize=3, color=colors[name], label=name)
        ax.set_xticks(range(3), [f"{z}\n{s:.0f}% del registro" for z, s in zip(order, share)], fontsize=8)
        ax.set(title=label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Entropía media [bits]")
    return fig


def plot_duration(results, method):
    duration, metrics = results["duration"], results["metrics"]
    measure = "MI" if method != "SEED" else "incertidumbre total"
    fig, axes = _panels(f"{method}: duración del evento predicho vs {measure} (promedio de folds; banda = ± desv. est.)")
    for ax, (label, part) in zip(axes, _scopes(duration)):
        scope = metrics if label.startswith("Promedio") else metrics[metrics["evaluación"] == int(label[-1])]
        stats = part.groupby("centro (s)")["incertidumbre"].agg(["mean", "std"])
        ax.plot(stats.index, stats["mean"], "o-", color=COLORS.get(method, "#d81b60"), lw=2)
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=COLORS.get(method, "#d81b60"), alpha=0.2)
        ax.set(title=f"{label}: ρ = {scope['ρ duración-incertidumbre'].mean():.2f}", xlabel="Duración del evento predicho [s]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"{measure} media del evento [bits]")
    return fig


def plot_event_boxes(results, method):
    """Box plots by event type, as percentile within each fold (0.5 = chance)."""
    events = results["events"].copy()
    measures = [m for m in ("aleatoria", "epistémica (MI)") if events[m].notna().any()]
    if method == "SEED":
        measures = ["incertidumbre total"]
    figures = []
    for measure in measures:
        events["percentil"] = events.groupby(["evaluación", "fold"])[measure].rank(pct=True)
        fig, axes = _panels(f"{method}: {measure} por tipo de evento (percentil dentro del fold; línea = azar)")
        for ax, (label, part) in zip(axes, _scopes(events)):
            values = [part.loc[part["grupo"] == g, "percentil"].dropna().to_numpy() for g in GROUPS]
            box = ax.boxplot(values, patch_artist=True, showmeans=True, meanline=True, showfliers=False,
                             medianprops={"color": "black"}, meanprops={"color": "#d62728", "linestyle": "-"},
                             tick_labels=[f"{g.replace(' (', chr(10) + '(')}\nn={len(v)}" for g, v in zip(GROUPS, values)])
            for patch, group in zip(box["boxes"], GROUPS):
                patch.set_facecolor(GROUP_COLORS[group])
                patch.set_alpha(0.75)
            ax.axhline(0.5, ls="--", color=CHANCE, lw=1.2)
            fp_tp = part[part["grupo"].isin(GROUPS[:2])]
            fn_tp = part[part["grupo"].isin(GROUPS[2:])]
            text = []
            for name, sub, positive in (("FP vs TP", fp_tp, "FP (predicho)"), ("FN vs TP", fn_tp, "FN (real)")):
                target = (sub["grupo"] == positive).to_numpy()
                text.append(f"{name}: AUROC {_auc(target, sub['percentil'].to_numpy()):.2f}, "
                            f"AUPRC {_auc(target, sub['percentil'].to_numpy(), True):.2f} (azar {target.mean():.2f})")
            ax.text(0.02, 0.98, "\n".join(text), transform=ax.transAxes, va="top", fontsize=7.5,
                    bbox={"facecolor": "white", "edgecolor": "#d9d8d3", "boxstyle": "round"})
            ax.set(title=label, ylim=(0, 1.12))
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Percentil dentro del fold")
        figures.append(fig)
    return figures


def plot_deferral(results, method):
    deferral = results["deferral"].dropna(subset=["% derivado"])
    if method == "SEED":
        deferral = deferral[deferral["medida"] == "SEED H(p)"]
    fig, axes = _panels(f"{method}: derivar a revisión las detecciones más inciertas (promedio de folds)")
    styles = {"incertidumbre total": ("#1baf7a", "-"), "epistémica (MI)": (COLORS.get(method, "#eb6834"), "-"), "SEED H(p)": (CHANCE, ":")}
    for ax, (label, part) in zip(axes, _scopes(deferral)):
        for measure, sub in part.groupby("medida"):
            stats = sub.groupby("% derivado")["precisión aceptado (%)"].mean()
            color, style = styles[measure]
            ax.plot(stats.index, stats.values, style, marker="o", color=color, lw=2, label=measure)
        ax.set(title=label, xlabel="% de detecciones derivadas a revisión")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Precisión de lo aceptado automáticamente (%)")
    return fig


def plot_rejection(results, method):
    rejection = results["deferral"].dropna(subset=["% descartado"])
    measure = "epistémica (MI)" if (rejection["medida"] == "epistémica (MI)").any() else "incertidumbre total"
    part = rejection[rejection["medida"] == measure]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    fig.suptitle(f"{method}: descartar automáticamente las detecciones con mayor {measure}")
    for label, sub in _scopes(part):
        stats = sub.groupby("% descartado")[["% FP en lo descartado", "cambio F1 (puntos)"]].mean()
        axes[0].plot(stats.index, stats["% FP en lo descartado"], "o-", label=label)
        axes[1].plot(stats.index, stats["cambio F1 (puntos)"], "o-", label=label)
    axes[0].axhline(58, ls="--", color=CHANCE, lw=1, label="equilibrio para ganar F1 (≈58%)")
    axes[1].axhline(0, ls="--", color=CHANCE, lw=1)
    axes[0].set(xscale="log", xlabel="% de detecciones descartadas", ylabel="% de FP entre lo descartado")
    axes[1].set(xscale="log", xlabel="% de detecciones descartadas", ylabel="Cambio en F1 (puntos)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    return fig


# ---------------------------------------------------------------- conformal

def conformal_results(alphas=(0.01, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Split conformal on SEED, per class, for evaluations 0 and 1."""
    from .uncertainty.conformal import build_prediction_sets, compute_nonconformity_scores, fit_conformal_threshold

    rows = []
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            with np.load(method_path("SEED", evaluation, fold, "validation")) as calibration, np.load(method_path("SEED", evaluation, fold)) as test:
                scores = compute_nonconformity_scores(calibration["probabilities"], calibration["y_true"], calibration["valid_mask"])
                valid = test["valid_mask"].astype(bool)
                truth = test["y_true"].astype(bool)
                for alpha in alphas:
                    sets = build_prediction_sets(test["probabilities"], fit_conformal_threshold(scores, alpha), test["valid_mask"])
                    covered = np.where(truth, sets["contains_event"], sets["contains_background"]).astype(bool)
                    size = sets["contains_event"].astype(int) + sets["contains_background"].astype(int)
                    rows.append({"evaluación": evaluation, "fold": fold, "alpha": alpha, "cobertura objetivo": 1 - alpha,
                                 "cobertura total": covered[valid].mean(), "cobertura husos": covered[valid & truth].mean(),
                                 "cobertura fondo": covered[valid & ~truth].mean(), "vacíos (abstención)": (size[valid] == 0).mean(),
                                 "ambiguos": (size[valid] == 2).mean(), "tamaño medio": size[valid].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- comparison

SCOPES = ["Evaluación 0", "Evaluación 1", "Promedio eval 0 y 1"]


def comparison_table(all_results: dict[str, dict]) -> pd.DataFrame:
    """One numeric row per method and evaluation scope (fold means)."""
    rows = []
    for method, results in all_results.items():
        metrics = results["metrics"].copy()
        events = results["events"]
        main = "epistémica (MI)" if events["epistémica (MI)"].notna().any() else "incertidumbre total"
        for name, negative, positive in (("FP vs TP", "TP (predicho)", "FP (predicho)"), ("FN vs TP", "TP (real)", "FN (real)")):
            separation = event_separation_folds(events, negative, positive)
            for measure, tag in (("incertidumbre total", "total"), (main, "MI")):
                metrics[f"{name} AUROC {tag}"] = np.nan if (tag == "MI" and main != "epistémica (MI)") else separation[f"AUROC {measure}"].to_numpy()
                metrics[f"{name} AUPRC {tag}"] = np.nan if (tag == "MI" and main != "epistémica (MI)") else separation[f"AUPRC {measure}"].to_numpy()
            metrics[f"{name} azar AUPRC"] = separation["azar AUPRC"].to_numpy()
        for label, part in _scopes(metrics):
            row = part.drop(columns=["evaluación", "fold"]).mean()
            row["método"], row["alcance"] = method, label
            rows.append(row)
    return pd.DataFrame(rows).set_index(["método", "alcance"])


def plot_method_bars(table: pd.DataFrame, metrics: list, title: str):
    """Methods on x; one point per evaluation and a larger one for both together.

    ``metrics`` = (column, label, chance) where chance is a number, a column name or None.
    Points, not bars, because the y axis is zoomed on the values.
    """
    methods = list(dict.fromkeys(table.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(title)
    styles = {"Evaluación 0": ("o", 45, -0.18, 0.55), "Evaluación 1": ("s", 45, 0.0, 0.55), "Promedio eval 0 y 1": ("D", 110, 0.18, 1.0)}
    for ax, (column, label, chance) in zip(axes, metrics):
        values_all = []
        for scope, (marker, size, offset, alpha) in styles.items():
            values = np.array([table.loc[(m, scope), column] for m in methods], dtype=float)
            values_all += [v for v in values if np.isfinite(v)]
            ax.scatter(np.arange(len(methods)) + offset, values, s=size, marker=marker, c=[COLORS[m] for m in methods],
                       alpha=alpha, edgecolors="white", linewidths=0.8, zorder=3, label=scope)
        if chance is not None:
            reference = table[chance].mean() if isinstance(chance, str) else chance
            values_all.append(reference)
            ax.axhline(reference, ls="--", color=CHANCE, lw=1.2, label=f"azar ({reference:.2f})")
        low, high = min(values_all), max(values_all)
        pad = 0.12 * (high - low) + 1e-3
        ax.set_ylim(low - pad, high + pad)
        ax.set_xticks(range(len(methods)), [m.replace(" ", "\n", 1) for m in methods], fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([], [], marker=m, ls="", color=CHANCE, ms=7 if s < 100 else 9, label=scope)
               for scope, (m, s, _, _) in styles.items()]
    axes[0].legend(handles=handles + [plt.Line2D([], [], ls="--", color=CHANCE, label="azar")], fontsize=8, loc="best")
    return fig


def plot_event_boxes_all(all_results: dict[str, dict], measure: str, scope: str = "Promedio eval 0 y 1"):
    """One panel per method: ``measure`` by event type, percentile within fold (0.5 = chance)."""
    methods = [m for m, r in all_results.items() if r["events"][measure].notna().any()]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 4.9), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"{measure} por tipo de evento — {scope} (percentil dentro del fold; línea = azar)")
    for ax, method in zip(axes, methods):
        events = all_results[method]["events"].copy()
        if scope != "Promedio eval 0 y 1":
            events = events[events["evaluación"] == int(scope[-1])]
        events["percentil"] = events.groupby(["evaluación", "fold"])[measure].rank(pct=True)
        values = [events.loc[events["grupo"] == g, "percentil"].dropna().to_numpy() for g in GROUPS]
        box = ax.boxplot(values, patch_artist=True, showmeans=True, meanline=True, showfliers=False,
                         medianprops={"color": "black"}, meanprops={"color": "#d62728", "linestyle": "-"},
                         tick_labels=["TP\npred.", "FP\npred.", "TP\nreal", "FN\nreal"])
        for patch, group in zip(box["boxes"], GROUPS):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_alpha(0.75)
        ax.axhline(0.5, ls="--", color=CHANCE, lw=1.2)
        fp_tp = events[events["grupo"].isin(GROUPS[:2])]
        fn_tp = events[events["grupo"].isin(GROUPS[2:])]
        auc_fp = _auc((fp_tp["grupo"] == "FP (predicho)").to_numpy(), fp_tp["percentil"].to_numpy())
        auc_fn = _auc((fn_tp["grupo"] == "FN (real)").to_numpy(), fn_tp["percentil"].to_numpy())
        ax.set_title(f"{method}\nAUROC FP {auc_fp:.2f} · FN {auc_fn:.2f}", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Percentil dentro del fold")
    return fig


def plot_reliability_all(all_results: dict[str, dict]):
    fig, axes = _panels("Diagrama de confiabilidad de la clase positiva — todos los métodos (promedio de folds)")
    for ax, label in zip(axes, SCOPES):
        ax.plot([0, 1], [0, 1], "--", color=CHANCE, lw=1, label="calibración perfecta")
        for method, results in all_results.items():
            rel = results["reliability"]
            if not label.startswith("Promedio"):
                rel = rel[rel["evaluación"] == int(label[-1])]
            curve = rel[rel["n"] > 0].groupby("bin")[["probabilidad predicha", "frecuencia observada"]].mean()
            ax.plot(curve["probabilidad predicha"], curve["frecuencia observada"], "o-", ms=3, lw=1.5, color=COLORS[method], label=method)
        ax.set(title=label, xlabel="Probabilidad predicha", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Frecuencia observada de huso")
    return fig


def plot_deferral_all(all_results: dict[str, dict]):
    fig, axes = _panels("Derivar a revisión las detecciones más inciertas (incertidumbre total) — todos los métodos")
    for ax, label in zip(axes, SCOPES):
        for method, results in all_results.items():
            deferral = results["deferral"].dropna(subset=["% derivado"])
            deferral = deferral[deferral["medida"] == "incertidumbre total"]
            if not label.startswith("Promedio"):
                deferral = deferral[deferral["evaluación"] == int(label[-1])]
            stats = deferral.groupby("% derivado")["precisión aceptado (%)"].mean()
            ax.plot(stats.index, stats.values, "o-", ms=4, color=COLORS[method], label=method)
        ax.set(title=label, xlabel="% de detecciones derivadas a revisión")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Precisión de lo aceptado (%)")
    return fig


def plot_decomposition_all(table: pd.DataFrame):
    scope = "Promedio eval 0 y 1"
    methods = [m for m in dict.fromkeys(table.index.get_level_values(0)) if m != "SEED"]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ale = [table.loc[(m, scope), "aleatoria (cerca de eventos)"] for m in methods]
    mi = [table.loc[(m, scope), "MI (cerca de eventos)"] for m in methods]
    seed = table.loc[("SEED", scope), "SEED H(p) (cerca de eventos)"]
    ax.bar(range(len(methods)), ale, 0.6, color="#7fb3ec", label="aleatoria")
    ax.bar(range(len(methods)), mi, 0.6, bottom=ale, color=[COLORS[m] for m in methods], label="epistémica (MI)")
    for i, (a, m) in enumerate(zip(ale, mi)):
        ax.text(i, a + m + 0.004, f"{100 * m / (a + m):.1f}% epist.", ha="center", fontsize=9)
    ax.axhline(seed, ls="--", color=COLORS["SEED"], lw=1.2, label=f"SEED H(p) ({seed:.3f})")
    ax.set_xticks(range(len(methods)), methods)
    ax.set(ylabel="Entropía media cerca de eventos [bits]", title="Descomposición de la incertidumbre — promedio eval 0 y 1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    return fig


def plot_rejection_all(all_results: dict[str, dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    fig.suptitle("Descartar automáticamente las detecciones con mayor MI (SEED: entropía) — promedio eval 0 y 1")
    for method, results in all_results.items():
        rejection = results["deferral"].dropna(subset=["% descartado"])
        measure = "epistémica (MI)" if (rejection["medida"] == "epistémica (MI)").any() else "SEED H(p)"
        stats = rejection[rejection["medida"] == measure].groupby("% descartado")[["% FP en lo descartado", "cambio F1 (puntos)"]].mean()
        axes[0].plot(stats.index, stats["% FP en lo descartado"], "o-", color=COLORS[method], label=method)
        axes[1].plot(stats.index, stats["cambio F1 (puntos)"], "o-", color=COLORS[method], label=method)
    axes[0].axhline(58, ls="--", color=CHANCE, lw=1, label="equilibrio para ganar F1 (≈58%)")
    axes[1].axhline(0, ls="--", color=CHANCE, lw=1)
    axes[0].set(xscale="log", xlabel="% de detecciones descartadas", ylabel="% de FP entre lo descartado")
    axes[1].set(xscale="log", xlabel="% de detecciones descartadas", ylabel="Cambio en F1 (puntos)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    return fig
