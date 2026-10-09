"""One set of test metrics for every uncertainty method of the project.

Reads the exported test NPZ files (no inference) and computes, per fold:
event F1, MI as an error detector near events (AUROC/AUPRC), MI for false
positives vs hits (AUPRC), total-entropy AUROC, % epistemic, NLL and the
positive-class ECE.  Every method goes through the same definitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.special import entr
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import SAMPLE_RATE_HZ, calculate_metrics_simple

NEAR_SAMPLES = SAMPLE_RATE_HZ  # ±1 s around true or predicted events
CALIBRATION_BINS = np.linspace(0.0, 1.0, 16)
LN2 = np.log(2.0)


def entropy_bits(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return (entr(p) + entr(1.0 - p)) / LN2


def load_fold(path: str | Path, kind: str) -> dict[str, np.ndarray | None]:
    """Normalise one test NPZ to probabilities, total entropy and MI in bits.

    ``kind``: ``"sampling"`` (Laplace layout: Laplace, Temporal Ensemble, MC
    Dropout, Deep Ensemble), ``"neural_process"`` or ``"seed"`` (no MI).
    """
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in (
            "y_true", "valid_mask", "y_pred", "processed_pred", "sample_id", "event_iou_values",
            "event_sample_id", "event_pred_start", "event_pred_end", "event_status",
        )}
        for key in ("event_true_start", "event_true_end"):
            out[key] = data[key]
        out["std"] = None
        if kind == "sampling":
            out["probability"] = data["mean_probability"].astype(np.float64)
            out["total"] = data["predictive_entropy"] / LN2
            out["mi"] = data["mutual_information"] / LN2
            if "predictive_probabilities" in data.files:
                out["std"] = data["predictive_probabilities"].std(axis=0)
        elif kind == "neural_process":
            out["probability"] = data["probabilities"].astype(np.float64)
            samples = data["predictive_samples"]  # (records, samples, time)
            out["total"] = entropy_bits(out["probability"])
            expected = np.mean([entropy_bits(samples[:, k]) for k in range(samples.shape[1])], axis=0)
            out["mi"] = np.maximum(out["total"] - expected, 0.0)
            out["std"] = samples.std(axis=1)
        elif kind == "seed":
            out["probability"] = data["probabilities"].astype(np.float64)
            out["total"] = entropy_bits(out["probability"])
            out["mi"] = None
        else:
            raise ValueError(f"Unknown kind {kind!r}.")
    # Aleatoric part: expected entropy = total - MI (all of it for a single model).
    out["aleatoric"] = out["total"] if out["mi"] is None else out["total"] - out["mi"]
    return out


EVENT_MEASURES = {
    "probabilidad media": "probability",
    "incertidumbre total": "total",
    "aleatoria": "aleatoric",
    "epistémica (MI)": "mi",
    "dispersión entre muestras (std)": "std",
}


def event_uncertainty(fold: dict, extra: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Mean of every uncertainty measure inside each event, by event type.

    TP and FP are measured on the predicted span (what the detector reports).
    Missed events (FN, including pairs with IoU < 0.2, which the F1 counts as
    misses) have no usable prediction, so they are measured on the true span,
    together with the TP on their true span for a like-for-like comparison.
    ``extra`` adds more per-timestep arrays (e.g. SEED's entropy) by name.
    """
    arrays = {name: fold[key] for name, key in EVENT_MEASURES.items()}
    arrays.update(extra or {})
    valid = fold["valid_mask"].astype(bool)
    row_of = {int(sample): row for row, sample in enumerate(fold["sample_id"])}
    spans = []
    for sample, status, p_start, p_end, t_start, t_end in zip(
        fold["event_sample_id"], fold["event_status"], fold["event_pred_start"],
        fold["event_pred_end"], fold["event_true_start"], fold["event_true_end"],
    ):
        row = row_of[int(sample)]
        if status in ("TP", "FP"):
            spans.append((row, "TP" if status == "TP" else "FP", "predicho", p_start, p_end))
        if status == "TP":
            spans.append((row, "TP", "real", t_start, t_end))
        if status in ("FN", "LOW_IOU"):
            spans.append((row, "FN", "real", t_start, t_end))
    records = []
    for row, kind, span, start, end in spans:
        inside = valid[row, start:end]
        if end <= start or not inside.any():
            continue
        record = {"tipo": kind, "tramo": span, "duración (s)": (end - start) / SAMPLE_RATE_HZ}
        for name, values in arrays.items():
            record[name] = np.nan if values is None else float(values[row, start:end][inside].mean())
        records.append(record)
    return pd.DataFrame(records)


def method_event_uncertainty(paths: list[str | Path], kind: str) -> pd.DataFrame:
    """Event-level uncertainty for every fold of one method."""
    frames = []
    for fold_index, path in enumerate(paths):
        frame = event_uncertainty(load_fold(path, kind))
        frame.insert(0, "fold", fold_index)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def positive_class_ece(probability: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(probability, 1e-6, 1.0 - 1e-6)
    bins = np.clip(np.digitize(p, CALIBRATION_BINS[1:-1]), 0, len(CALIBRATION_BINS) - 2)
    counts = np.bincount(bins, minlength=len(CALIBRATION_BINS) - 1)
    mean_p = np.bincount(bins, weights=p, minlength=len(counts)) / np.maximum(counts, 1)
    observed = np.bincount(bins, weights=target, minlength=len(counts)) / np.maximum(counts, 1)
    filled = counts > 0
    return float(np.sum(counts[filled] / counts.sum() * np.abs(mean_p[filled] - observed[filled])))


def _event_means(fold: dict, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` inside every predicted TP/FP event, and its FP flag."""
    valid = fold["valid_mask"].astype(bool)
    row_of = {int(sample): row for row, sample in enumerate(fold["sample_id"])}
    means, is_fp = [], []
    for sample, start, end, status in zip(
        fold["event_sample_id"], fold["event_pred_start"], fold["event_pred_end"], fold["event_status"]
    ):
        if status not in ("TP", "FP"):
            continue
        row = row_of[int(sample)]
        inside = valid[row, start:end]
        if inside.any():
            means.append(float(values[row, start:end][inside].mean()))
            is_fp.append(status == "FP")
    return np.asarray(means), np.asarray(is_fp)


def fold_metrics(fold: dict) -> dict[str, float]:
    valid = fold["valid_mask"].astype(bool)
    truth = fold["y_true"].astype(bool)
    error = fold["y_pred"].astype(bool) != truth
    near = valid & binary_dilation(
        truth | fold["processed_pred"].astype(bool), structure=np.ones((1, 2 * NEAR_SAMPLES + 1), dtype=bool)
    )
    curve = calculate_metrics_simple(fold["event_iou_values"].tolist())
    p, y = fold["probability"][valid], truth[valid].astype(np.float64)
    row = {
        "F1 evento (%)": 100.0 * float(curve["f1"][curve["selected_index"]]),
        "AUROC MI errores": np.nan,
        "AUPRC MI errores": np.nan,
        "azar AUPRC errores": float(error[near].mean()),
        "AUPRC MI FP vs TP": np.nan,
        "azar AUPRC FP": np.nan,
        "AUROC total errores": float(roc_auc_score(error[near], fold["total"][near])),
        "% epistémica": np.nan,
        "NLL": float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))),
        "ECE clase positiva": positive_class_ece(p, y),
    }
    if fold["mi"] is not None:
        mi = fold["mi"]
        event_mi, is_fp = _event_means(fold, mi)
        row.update({
            "AUROC MI errores": float(roc_auc_score(error[near], mi[near])),
            "AUPRC MI errores": float(average_precision_score(error[near], mi[near])),
            "AUPRC MI FP vs TP": float(average_precision_score(is_fp, event_mi)),
            "azar AUPRC FP": float(is_fp.mean()),
            "% epistémica": 100.0 * float(mi[valid].mean() / fold["total"][valid].mean()),
        })
    return row


def method_table(paths: list[str | Path], kind: str) -> pd.DataFrame:
    """Per-fold metrics plus mean and std rows."""
    rows = [fold_metrics(load_fold(path, kind)) for path in paths]
    table = pd.DataFrame(rows, index=pd.Index([f"fold {k}" for k in range(len(rows))], name=""))
    return pd.concat([table, table.agg(["mean", "std"]).rename(index={"mean": "media", "std": "desv. est."})])
