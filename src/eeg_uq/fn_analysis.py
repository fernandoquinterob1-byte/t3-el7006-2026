"""What do the spindles the Deep Ensemble misses look like?

Every annotated spindle of the test folds is put in one group by how close
the Deep Ensemble came to reporting it, and every false positive gets its
own group:

* **TP**: detected (IoU >= 0.2).
* **FN visto pero corto**: probability >= 0.425 inside, but the legacy
  post-processing removed it (shorter than 0.3 s above the threshold).
* **FN casi visto**: maximum probability between 0.25 and 0.425.
* **FN invisible**: maximum probability below 0.25.
* **FP**: detection with no annotated spindle (measured on the predicted span).

For each event, simple signal features are measured on the raw EEG (channel
C3, 200 Hz): duration, peak-to-peak amplitude and RMS in the sigma band
(11-16 Hz), and the sigma contrast against ±2 s of surrounding signal.  Plus
the probability and uncertainty of the Deep Ensemble and Neural Process.

Reads the exported NPZ files and the raw HDF5 (``EEG_DATA_ROOT`` or
``~/Downloads/data/data``); nothing is trained.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .method_summary import load_fold
from .metrics import DURATION_THRESHOLD, SAMPLE_RATE_HZ
from .report import CACHE, EVALUATIONS, FOLDS, method_path

CACHE_FILE = CACHE / "fn_analysis.pkl"
CACHE_VERSION = 1
SIGMA_BAND = (11.0, 16.0)
CONTEXT_S = 2.0
GROUPS = ["TP", "FN visto pero corto", "FN casi visto", "FN invisible", "FP", "Fondo"]
FEATURES = ["duración (s)", "amplitud sigma pico a pico (µV)", "RMS sigma (µV)", "contraste sigma (veces el fondo)",
            "RMS banda ancha (µV)", "p máx DE", "p media DE", "MI media DE", "MI media NP", "aleatoria media DE"]


def data_file() -> Path:
    root = Path(os.environ.get("EEG_DATA_ROOT", Path.home() / "Downloads" / "data" / "data"))
    return root / "moda_moda.h5"


def _sigma(signals: np.ndarray) -> np.ndarray:
    sos = butter(4, SIGMA_BAND, btype="bandpass", fs=SAMPLE_RATE_HZ, output="sos")
    return sosfiltfilt(sos, signals, axis=-1)


def _group_of_miss(p_max: float) -> str:
    if p_max >= DURATION_THRESHOLD:
        return "FN visto pero corto"
    if p_max >= 0.25:
        return "FN casi visto"
    return "FN invisible"


def fold_events(evaluation: int, fold: int, signals: np.ndarray, sigma: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    de = load_fold(method_path("Deep Ensemble", evaluation, fold), "sampling")
    np_ = load_fold(method_path("Neural Process", evaluation, fold), "neural_process")
    if not np.array_equal(de["sample_id"], np_["sample_id"]):
        raise ValueError(f"Deep Ensemble and Neural Process records differ (eval {evaluation}, fold {fold}).")
    row_of = {int(s): r for r, s in enumerate(de["sample_id"])}
    context = int(CONTEXT_S * SAMPLE_RATE_HZ)
    records = []
    for sample, status, t_start, t_end, p_start, p_end in zip(
        de["event_sample_id"], de["event_status"], de["event_true_start"], de["event_true_end"],
        de["event_pred_start"], de["event_pred_end"],
    ):
        row, sample = row_of[int(sample)], int(sample)
        if status == "FP":
            start, end, group = int(p_start), int(p_end), "FP"
        elif status in ("TP", "FN", "LOW_IOU"):
            start, end = int(t_start), int(t_end)
            if end <= start:
                continue
            group = "TP" if status == "TP" else _group_of_miss(float(de["probability"][row, start:end].max()))
        else:
            continue
        if end <= start:
            continue
        segment = sigma[sample, start:end]
        lo, hi = max(0, start - context), min(sigma.shape[1], end + context)
        around = np.concatenate([sigma[sample, lo:start], sigma[sample, end:hi]])
        rms_inside = float(np.sqrt(np.mean(segment ** 2)))
        rms_around = float(np.sqrt(np.mean(around ** 2))) if len(around) else np.nan
        records.append({
            "evaluación": evaluation, "fold": fold, "registro": sample, "sujeto": subjects[sample], "grupo": group,
            "inicio": start, "fin": end,
            "duración (s)": (end - start) / SAMPLE_RATE_HZ,
            "amplitud sigma pico a pico (µV)": float(segment.max() - segment.min()),
            "RMS sigma (µV)": rms_inside,
            "contraste sigma (veces el fondo)": rms_inside / rms_around if rms_around else np.nan,
            "RMS banda ancha (µV)": float(np.sqrt(np.mean((signals[sample, start:end] - signals[sample, start:end].mean()) ** 2))),
            "p máx DE": float(de["probability"][row, start:end].max()),
            "p media DE": float(de["probability"][row, start:end].mean()),
            "MI media DE": float(de["mi"][row, start:end].mean()),
            "MI media NP": float(np_["mi"][row, start:end].mean()),
            "aleatoria media DE": float(de["aleatoric"][row, start:end].mean()),
            "p máx por miembro DE": float(de["probability"][row, start:end].max()),
        })
    return pd.DataFrame(records)


def background_windows(evaluation: int, fold: int, signals: np.ndarray, sigma: np.ndarray, subjects: np.ndarray,
                       per_record: int = 3, seed: int = 0) -> pd.DataFrame:
    """Random 0.5 s windows at least 1 s away from any annotated or detected spindle (reference group)."""
    de = load_fold(method_path("Deep Ensemble", evaluation, fold), "sampling")
    rng = np.random.default_rng(seed + 100 * evaluation + fold)
    length, context = int(0.5 * SAMPLE_RATE_HZ), int(CONTEXT_S * SAMPLE_RATE_HZ)
    margin = SAMPLE_RATE_HZ
    records = []
    for row, sample in enumerate(de["sample_id"]):
        sample = int(sample)
        busy = (de["y_true"][row] > 0) | (de["processed_pred"][row] > 0)
        busy = np.convolve(busy.astype(float), np.ones(2 * margin + 1), mode="same") > 0
        valid = de["valid_mask"][row].astype(bool) & ~busy
        starts = np.flatnonzero(valid[: len(valid) - length])
        starts = [s for s in rng.permutation(starts)[:50] if valid[s:s + length].all()][:per_record]
        for start in starts:
            end = start + length
            segment = sigma[sample, start:end]
            lo, hi = max(0, start - context), min(sigma.shape[1], end + context)
            around = np.concatenate([sigma[sample, lo:start], sigma[sample, end:hi]])
            rms_inside = float(np.sqrt(np.mean(segment ** 2)))
            records.append({
                "evaluación": evaluation, "fold": fold, "registro": sample, "sujeto": subjects[sample], "grupo": "Fondo",
                "inicio": start, "fin": end, "duración (s)": length / SAMPLE_RATE_HZ,
                "amplitud sigma pico a pico (µV)": float(segment.max() - segment.min()),
                "RMS sigma (µV)": rms_inside,
                "contraste sigma (veces el fondo)": rms_inside / float(np.sqrt(np.mean(around ** 2))),
                "RMS banda ancha (µV)": float(np.sqrt(np.mean((signals[sample, start:end] - signals[sample, start:end].mean()) ** 2))),
                "p máx DE": float(de["probability"][row, start:end].max()),
                "p media DE": float(de["probability"][row, start:end].mean()),
                "MI media DE": float(de["mi"][row, start:end].mean()),
                "MI media NP": np.nan,
                "aleatoria media DE": float(de["aleatoric"][row, start:end].mean()),
            })
    return pd.DataFrame(records)


def run_all(refresh: bool = False, progress=print) -> pd.DataFrame:
    if CACHE_FILE.is_file() and not refresh:
        with CACHE_FILE.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored["results"]
    with h5py.File(data_file(), "r") as source:
        signals = np.asarray(source["signals"][:], dtype=np.float64)
        subjects = np.asarray([s.decode() if isinstance(s, bytes) else str(s) for s in source["subject_ids"][:]])
    sigma = _sigma(signals)
    frames = []
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            frames.append(fold_events(evaluation, fold, signals, sigma, subjects).drop(columns=["p máx por miembro DE"]))
            frames.append(background_windows(evaluation, fold, signals, sigma, subjects))
        progress(f"evaluación {evaluation} lista")
    results = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "results": results}, handle)
    return results


SEPARABILITY_FILE = CACHE / "fn_separability.pkl"


def near_miss_separability(refresh: bool = False) -> pd.DataFrame:
    """Per fold AUROC of probability, MI and sigma features on the notebook-09 event table (test).

    Near misses: real spindle vs not.  Detections: FP vs TP.
    """
    from sklearn.metrics import roc_auc_score

    from .hybrid_threshold import event_table, load_pair

    if SEPARABILITY_FILE.is_file() and not refresh:
        with SEPARABILITY_FILE.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored["results"]
    with h5py.File(data_file(), "r") as source:
        sigma = _sigma(np.asarray(source["signals"][:], dtype=np.float64))
    context = int(CONTEXT_S * SAMPLE_RATE_HZ)
    rows = []
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            pair = load_pair(evaluation, fold, "test")
            events = event_table(pair, DURATION_THRESHOLD)
            ids = pair["de"]["sample_id"]
            feats = []
            for row, start, end in events[["fila", "inicio", "fin"]].itertuples(index=False):
                sample = int(ids[int(row)])
                segment = sigma[sample, start:end]
                around = np.concatenate([sigma[sample, max(0, start - context):start], sigma[sample, end:end + context]])
                rms = float(np.sqrt(np.mean(segment ** 2)))
                feats.append({"amplitud sigma": float(segment.max() - segment.min()), "RMS sigma": rms,
                              "contraste sigma": rms / float(np.sqrt(np.mean(around ** 2)))})
            events = pd.concat([events.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
            for flag, group in ((1, "casi detectados: huso real vs no"), (0, "detecciones: FP vs TP")):
                part = events[events["candidato"] == flag]
                target = part["real"].to_numpy() == (1 if flag else 0)
                if target.all() or not target.any():
                    continue
                row = {"evaluación": evaluation, "fold": fold, "grupo": group, "azar AUPRC": float(target.mean())}
                for measure in ("p media DE", "MI NP", "amplitud sigma", "RMS sigma", "contraste sigma"):
                    score = part[measure].to_numpy()
                    if flag == 0 and measure != "MI NP":
                        score = -score  # FP: lower probability and weaker sigma
                    row[f"AUROC {measure}"] = float(roc_auc_score(target, score))
                rows.append(row)
    results = pd.DataFrame(rows)
    with SEPARABILITY_FILE.open("wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "results": results}, handle)
    return results


def load_example(sample: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw and sigma-filtered signal of one record (for the example plots)."""
    with h5py.File(data_file(), "r") as source:
        signal = np.asarray(source["signals"][sample], dtype=np.float64)
    return signal, _sigma(signal[None, :])[0]
