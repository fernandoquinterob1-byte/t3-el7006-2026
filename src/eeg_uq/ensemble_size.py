"""Deep Ensemble F1 as a function of the number of members (1 to 5).

Uses the per-member probabilities saved in the Deep Ensemble test NPZ: for
each ensemble size, every subset of the five seeds is averaged, post-processed
with the legacy rule and scored; the subsets of one size are then averaged.
Nothing is trained.  A fit ``F1(M) = a - b / M`` extrapolates to more seeds.
"""

from __future__ import annotations

import pickle
from itertools import combinations

import numpy as np
import pandas as pd

from .hybrid_threshold import event_scores, process_masks
from .metrics import DURATION_THRESHOLD
from .report import CACHE, EVALUATIONS, FOLDS, method_path

CACHE_FILE = CACHE / "ensemble_size.pkl"
CACHE_VERSION = 1
SCORES = ["F1 (%)", "precisión (%)", "recall (%)", "FP", "FN"]


def run_all(refresh: bool = False, progress=print) -> pd.DataFrame:
    """Per fold and ensemble size: mean scores over every subset of that size."""
    if CACHE_FILE.is_file() and not refresh:
        with CACHE_FILE.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored["results"]
    rows = []
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            with np.load(method_path("Deep Ensemble", evaluation, fold), allow_pickle=False) as data:
                members, y_true = data["predictive_probabilities"].astype(np.float32), data["y_true"]
            for size in range(1, len(members) + 1):
                subset_scores = [event_scores(y_true, process_masks(members[list(s)].mean(axis=0), DURATION_THRESHOLD))
                                 for s in combinations(range(len(members)), size)]
                frame = pd.DataFrame(subset_scores)
                rows.append({"evaluación": evaluation, "fold": fold, "miembros": size,
                             "subconjuntos": len(subset_scores), **frame[SCORES].mean().to_dict(),
                             "desv. F1 entre subconjuntos": frame["F1 (%)"].std() if size < len(members) else np.nan})
        progress(f"evaluación {evaluation} lista")
    results = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "results": results}, handle)
    return results


def extrapolate(results: pd.DataFrame, sizes=(10, 20)) -> dict[str, float]:
    """Least-squares fit F1(M) = a - b / M on the mean curve."""
    curve = results.groupby("miembros")["F1 (%)"].mean()
    m = curve.index.to_numpy(float)
    a, b = np.linalg.lstsq(np.vstack([np.ones_like(m), -1 / m]).T, curve.to_numpy(), rcond=None)[0]
    return {"a": float(a), "b": float(b), **{f"M={s}": float(a - b / s) for s in sizes}, "límite": float(a)}
