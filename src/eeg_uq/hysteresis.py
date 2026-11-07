"""Post-processing variants: hysteresis (two thresholds) and minimum duration.

The legacy post-processing (``metrics.process_spindle_simple``, kept intact)
binarises the probability at 0.425, merges gaps < 0.3 s, removes events
< 0.3 s and trims events > 5 s.  About 11% of the Deep Ensemble misses have
a probability above 0.425 for less than 0.3 s, so that rule deletes them.

Variants (no training; probabilities read from the exported NPZ files):

* **Hysteresis**: an event is a run with ``p >= t_low`` that contains at least
  one sample with ``p >= t_high``; then the legacy clean-up is applied.
  ``t_high = t_low = 0.425`` reproduces the legacy output exactly.
* **Minimum duration**: legacy threshold, different minimum event length.
* **Both** together.

Every parameter is chosen per fold with the event F1 on validation and
scored once on test.
"""

from __future__ import annotations

import pickle
from itertools import product

import numpy as np
import pandas as pd
from scipy import ndimage

from .hybrid_threshold import METHOD_KINDS, _change_points, event_scores
from .method_summary import load_fold
from .metrics import DURATION_THRESHOLD, SAMPLE_RATE_HZ
from .report import CACHE, EVALUATIONS, FOLDS, method_path

CACHE_FILE = CACHE / "hysteresis.pkl"
CACHE_VERSION = 1
HIGH_GRID = (0.425, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
LOW_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.425)
MIN_DURATION_GRID = (0.20, 0.25, 0.30, 0.35, 0.40)  # seconds
LEGACY_MIN_S, MAX_S = 0.3, 5.0
ROW_STRUCTURE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])  # runs never cross records


def clean_up(mask_row: np.ndarray, min_s: float = LEGACY_MIN_S, max_s: float = MAX_S) -> np.ndarray:
    """Legacy clean-up (merge gaps < min, drop events < min, trim > max) with a configurable minimum.

    With ``min_s = 0.3`` it is identical to ``metrics.process_spindle_simple``.
    """
    min_samples, max_samples = int(SAMPLE_RATE_HZ * min_s), int(SAMPLE_RATE_HZ * max_s)
    dur_out = np.asarray(mask_row)
    new_output = dur_out.copy()
    last_label, in_event = 0, False
    for index in _change_points(dur_out):
        label = dur_out[index]
        if label == 1 and not in_event:
            if last_label != 0 and index - last_label < min_samples:
                new_output[last_label:index] = 1
            in_event = True
        elif label == 0 and in_event:
            last_label, in_event = index, False
    start_label, in_event = 0, False
    length = len(new_output)
    reference = new_output.copy()
    for index in _change_points(reference):
        label = reference[index]
        if label == 1 and not in_event:
            start_label, in_event = index, True
        elif (label == 0 or index + 1 == length) and in_event:
            if index - start_label < min_samples:
                new_output[start_label:index] = 0
            elif index - start_label > max_samples:
                start_trim = (index - start_label - max_samples) // 2 + start_label
                end_trim = start_trim + max_samples
                new_output[start_label:start_trim] = 0
                new_output[end_trim:index] = 0
            in_event = False
    return new_output


def hysteresis_mask(probability: np.ndarray, t_high: float, t_low: float) -> np.ndarray:
    """Runs of ``p >= t_low`` (within each record) that reach ``p >= t_high`` somewhere."""
    if t_low >= t_high:
        return (probability >= t_high).astype(np.int8)
    low = probability >= t_low
    labels, count = ndimage.label(low, structure=ROW_STRUCTURE)
    if count == 0:
        return np.zeros(probability.shape, np.int8)
    peak = ndimage.maximum(probability, labels=labels, index=np.arange(1, count + 1))
    keep = np.concatenate([[False], peak >= t_high])
    return keep[labels].astype(np.int8)


def detect(probability: np.ndarray, t_high: float = DURATION_THRESHOLD, t_low: float = DURATION_THRESHOLD,
           min_s: float = LEGACY_MIN_S) -> np.ndarray:
    """Hysteresis mask followed by the clean-up, for every record."""
    mask = hysteresis_mask(probability, t_high, t_low)
    return np.stack([clean_up(row, min_s) for row in mask])


VARIANTS = {
    "Histéresis": [(h, l, LEGACY_MIN_S) for h, l in product(HIGH_GRID, LOW_GRID) if l <= h],
    "Duración mínima": [(DURATION_THRESHOLD, DURATION_THRESHOLD, m) for m in MIN_DURATION_GRID],
    # Built per fold: the best hysteresis pair with every minimum duration (two-step search).
    "Histéresis + duración mínima": None,
}


def fold_analysis(method: str, evaluation: int, fold: int) -> tuple[list[dict], list[dict]]:
    kind = METHOD_KINDS[method]
    val = load_fold(method_path(method, evaluation, fold, "validation"), kind)
    test = load_fold(method_path(method, evaluation, fold, "test"), kind)
    tag = {"método": method, "evaluación": evaluation, "fold": fold}
    cache: dict[tuple, dict] = {}

    def val_score(params):
        if params not in cache:
            cache[params] = event_scores(val["y_true"], detect(val["probability"], *params))
        return cache[params]

    legacy = (DURATION_THRESHOLD, DURATION_THRESHOLD, LEGACY_MIN_S)
    rows = [{**tag, "variante": "Original", "umbral alto": legacy[0], "umbral bajo": legacy[1],
             "duración mínima (s)": legacy[2], "F1 val (%)": val_score(legacy)["F1 (%)"],
             **event_scores(test["y_true"], detect(test["probability"], *legacy))}]
    grid_rows = []
    chosen: dict[str, tuple] = {}
    for variant, grid in VARIANTS.items():
        if grid is None:
            high, low, _ = chosen["Histéresis"]
            grid = [(high, low, m) for m in MIN_DURATION_GRID]
        for params in grid:
            grid_rows.append({**tag, "variante": variant, "umbral alto": params[0], "umbral bajo": params[1],
                              "duración mínima (s)": params[2], "F1 val (%)": val_score(params)["F1 (%)"]})
        # Ties: prefer the setting closest to the legacy one.
        best = max(grid, key=lambda p: (round(val_score(p)["F1 (%)"], 6),
                                        -sum(abs(a - b) for a, b in zip(p, legacy))))
        chosen[variant] = best
        rows.append({**tag, "variante": variant, "umbral alto": best[0], "umbral bajo": best[1],
                     "duración mínima (s)": best[2], "F1 val (%)": val_score(best)["F1 (%)"],
                     **event_scores(test["y_true"], detect(test["probability"], *best))})
    return rows, grid_rows


def run_all(refresh: bool = False, progress=print) -> dict[str, pd.DataFrame]:
    if CACHE_FILE.is_file() and not refresh:
        with CACHE_FILE.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored["results"]
    rows, grid_rows = [], []
    for method in ("Deep Ensemble", "SEED", "Neural Process"):
        for evaluation in EVALUATIONS:
            for fold in FOLDS:
                fold_rows, fold_grid = fold_analysis(method, evaluation, fold)
                rows += fold_rows
                grid_rows += fold_grid
        progress(f"{method} listo")
    results = {"scores": pd.DataFrame(rows), "grid": pd.DataFrame(grid_rows)}
    CACHE.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "results": results}, handle)
    return results


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Per method: mean ± std and the paired change vs the original post-processing."""
    from scipy.stats import wilcoxon

    rows = []
    for method, by_method in scores.groupby("método", sort=False):
        base = by_method[by_method["variante"] == "Original"].set_index(["evaluación", "fold"])
        for variant, part in by_method.groupby("variante", sort=False):
            part = part.set_index(["evaluación", "fold"])
            delta = part["F1 (%)"] - base["F1 (%)"]
            try:
                p_value = wilcoxon(delta).pvalue if (delta != 0).any() else np.nan
            except ValueError:
                p_value = np.nan
            rows.append({
                "método": method, "variante": variant,
                "F1 (%)": f"{part['F1 (%)'].mean():.2f} ± {part['F1 (%)'].std():.2f}",
                "precisión (%)": f"{part['precisión (%)'].mean():.2f} ± {part['precisión (%)'].std():.2f}",
                "recall (%)": f"{part['recall (%)'].mean():.2f} ± {part['recall (%)'].std():.2f}",
                "FP": round(part["FP"].mean(), 1), "FN": round(part["FN"].mean(), 1),
                "Δ F1 (puntos)": round(delta.mean(), 2),
                "Δ precisión (puntos)": round((part["precisión (%)"] - base["precisión (%)"]).mean(), 2),
                "Δ recall (puntos)": round((part["recall (%)"] - base["recall (%)"]).mean(), 2),
                "folds que mejoran F1": f"{int((delta > 0).sum())}/{len(delta)}",
                "p Wilcoxon": round(p_value, 4) if not np.isnan(p_value) else np.nan,
            })
    return pd.DataFrame(rows).set_index(["método", "variante"])
