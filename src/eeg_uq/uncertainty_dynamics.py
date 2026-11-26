"""Uncertainty curves on top of probability curves, by event type and spindle duration.

Questions (tutor meeting):

* How do the uncertainty curves u(t) move together with the probability p(t)?
* How is the uncertainty distributed for short / medium / long spindles?
* Is the uncertainty of our methods (Deep Ensemble, Neural Process) the same
  as the one of the base SEED model ("zero sum") or does it add something?

Events are classified by the **base SEED model** (the frozen baseline):

* **TP** / **FN** (FN includes pairs with IoU < 0.2): measured on the
  annotated span, so the duration is the spindle duration.
* **FP**: measured on the predicted span.
* **TN**: random windows of the median spindle length, at least 1 s away from
  any annotated spindle or SEED detection.

SEED is a single model, so its only uncertainty is the entropy of its
probability (aleatoric).  Reads the exported test NPZ files only; nothing is
trained and no raw EEG is read.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .method_summary import load_fold
from .metrics import SAMPLE_RATE_HZ
from .report import CACHE, EVALUATIONS, FOLDS, method_path

CACHE_FILE = CACHE / "uncertainty_dynamics.pkl"
CACHE_VERSION = 1

DURATION_CUTS = (0.6, 0.9)  # seconds; ~terciles of the annotated spindle durations (0.62, 0.90)
DURATION_CLASSES = ["corto", "mediano", "largo"]
EVENT_TYPES = ["TP", "FN", "FP", "TN"]
TN_SECONDS = 0.75  # median annotated spindle duration
TN_PER_RECORD = 3

PROBABILITIES = ["p SEED", "p DE", "p NP"]
UNCERTAINTIES = ["u SEED", "u total DE", "aleatoria DE", "MI DE", "u total NP", "MI NP"]
SERIES = PROBABILITIES + UNCERTAINTIES

ALIGN_SECONDS = 1.5  # window around the event start / end
ALIGN_STEP = 8  # samples (SEED's output resolution: 0.04 s)
ALIGN_OFFSETS = np.arange(-int(ALIGN_SECONDS * SAMPLE_RATE_HZ), int(ALIGN_SECONDS * SAMPLE_RATE_HZ) + 1, ALIGN_STEP)


def duration_class(seconds: float) -> str:
    if seconds < DURATION_CUTS[0]:
        return "corto"
    if seconds < DURATION_CUTS[1]:
        return "mediano"
    return "largo"


def fold_series(evaluation: int, fold: int, split: str = "test") -> dict:
    """Per-timestep probability and uncertainty of SEED, DE and NP, aligned by record."""
    seed = load_fold(method_path("SEED", evaluation, fold, split), "seed")
    de = load_fold(method_path("Deep Ensemble", evaluation, fold, split), "sampling")
    np_ = load_fold(method_path("Neural Process", evaluation, fold, split), "neural_process")
    for name, other in (("Deep Ensemble", de), ("Neural Process", np_)):
        if not np.array_equal(seed["sample_id"], other["sample_id"]):
            raise ValueError(f"SEED and {name} records differ (eval {evaluation}, fold {fold}).")
    series = {
        "p SEED": seed["probability"], "p DE": de["probability"], "p NP": np_["probability"],
        "u SEED": seed["total"], "u total DE": de["total"], "aleatoria DE": de["aleatoric"], "MI DE": de["mi"],
        "u total NP": np_["total"], "MI NP": np_["mi"],
    }
    return {
        "series": {k: np.asarray(v, dtype=np.float32) for k, v in series.items()},
        "valid": seed["valid_mask"].astype(bool),
        "y_true": seed["y_true"],
        "seed_pred": seed["processed_pred"],
        "seed": seed,
    }


def base_events(data: dict, evaluation: int, fold: int, seed: int = 0) -> list[tuple]:
    """(row, type, start, end) of every TP / FN / FP of base SEED plus random TN windows."""
    fold_data = data["seed"]
    row_of = {int(s): r for r, s in enumerate(fold_data["sample_id"])}
    spans = []
    for sample, status, t_start, t_end, p_start, p_end in zip(
        fold_data["event_sample_id"], fold_data["event_status"], fold_data["event_true_start"],
        fold_data["event_true_end"], fold_data["event_pred_start"], fold_data["event_pred_end"],
    ):
        row = row_of[int(sample)]
        if status == "TP":
            spans.append((row, "TP", int(t_start), int(t_end)))
        elif status in ("FN", "LOW_IOU"):
            spans.append((row, "FN", int(t_start), int(t_end)))
        elif status == "FP":
            spans.append((row, "FP", int(p_start), int(p_end)))
    rng = np.random.default_rng(seed + 100 * evaluation + fold)
    length, margin = int(TN_SECONDS * SAMPLE_RATE_HZ), SAMPLE_RATE_HZ
    for row in range(len(fold_data["sample_id"])):
        busy = (data["y_true"][row] > 0) | (data["seed_pred"][row] > 0)
        busy = np.convolve(busy.astype(float), np.ones(2 * margin + 1), mode="same") > 0
        free = data["valid"][row] & ~busy
        starts = np.flatnonzero(free[: len(free) - length])
        chosen = [s for s in rng.permutation(starts)[:50] if free[s:s + length].all()][:TN_PER_RECORD]
        spans.extend((row, "TN", int(s), int(s) + length) for s in chosen)
    return [s for s in spans if s[3] > s[2]]


def _event_rows(data: dict, spans: list[tuple], evaluation: int, fold: int) -> pd.DataFrame:
    series, valid = data["series"], data["valid"]
    sample_ids = data["seed"]["sample_id"]
    dt = 1.0 / SAMPLE_RATE_HZ
    records = []
    for row, kind, start, end in spans:
        inside = valid[row, start:end]
        if not inside.any():
            continue
        duration = (end - start) / SAMPLE_RATE_HZ
        record = {"evaluación": evaluation, "fold": fold, "registro": int(sample_ids[row]), "fila": row,
                  "tipo": kind, "inicio": start, "fin": end, "duración (s)": duration,
                  "clase duración": duration_class(duration) if kind != "TN" else "—"}
        for name in SERIES:
            values = series[name][row, start:end][inside]
            record[name] = float(values.mean())
            if name in UNCERTAINTIES:
                record[f"masa {name}"] = float(values.sum() * dt)  # area under u(t), bit·s
        for name in ("u SEED", "MI DE", "MI NP"):
            values = np.where(inside, series[name][row, start:end], -np.inf)
            record[f"posición pico {name}"] = float(np.argmax(values) / max(1, end - start - 1))
        records.append(record)
    return pd.DataFrame(records)


def _aligned(data: dict, spans: list[tuple]) -> dict:
    """Curves of every series around the start and the end of each TP / FN / FP (NaN outside the record)."""
    out = {}
    series, valid = data["series"], data["valid"]
    width = valid.shape[1]
    for anchor in ("inicio", "fin"):
        rows, times, kinds, durations = [], [], [], []
        for row, kind, start, end in spans:
            if kind == "TN":
                continue
            rows.append(row)
            times.append((start if anchor == "inicio" else end) + ALIGN_OFFSETS)
            kinds.append(kind)
            durations.append(duration_class((end - start) / SAMPLE_RATE_HZ))
        rows, times = np.asarray(rows)[:, None], np.asarray(times)
        inside = (times >= 0) & (times < width)
        clipped = np.clip(times, 0, width - 1)
        ok = inside & valid[rows, clipped]
        curves = {}
        for name in SERIES:
            values = series[name][rows, clipped].astype(np.float32)
            values[~ok] = np.nan
            curves[name] = values
        out[anchor] = {"curves": curves, "tipo": np.asarray(kinds), "clase duración": np.asarray(durations)}
    return out


def _timestep_correlations(data: dict, rng: np.random.Generator, n: int = 200_000) -> pd.DataFrame:
    """Spearman of each u with |p - 0.5| (same model) on a random subsample of valid timesteps."""
    from scipy.stats import spearmanr

    series = data["series"]
    valid = data["valid"].ravel()
    active = valid & (np.maximum.reduce([series[p].ravel() for p in PROBABILITIES]) >= 0.1)
    pairs = {"u SEED": "p SEED", "u total DE": "p DE", "aleatoria DE": "p DE", "MI DE": "p DE",
             "u total NP": "p NP", "MI NP": "p NP"}
    rows = []
    for zone, mask in (("todos los instantes", valid), ("zona activa (alguna p ≥ 0.1)", active)):
        candidates = np.flatnonzero(mask)
        pick = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        flat = {k: v.ravel()[pick] for k, v in series.items()}
        for u, p in pairs.items():
            rows.append({"zona": zone, "medida": u, "vs": f"|{p} − 0.5|",
                         "Spearman": float(spearmanr(flat[u], np.abs(flat[p] - 0.5)).statistic)})
        for u in ("u total DE", "MI DE", "u total NP", "MI NP"):
            rows.append({"zona": zone, "medida": u, "vs": "u SEED",
                         "Spearman": float(spearmanr(flat[u], flat["u SEED"]).statistic)})
    return pd.DataFrame(rows)


def run_all(refresh: bool = False, progress=print) -> dict:
    if CACHE_FILE.is_file() and not refresh:
        with CACHE_FILE.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored
    rng = np.random.default_rng(0)
    events, correlations = [], []
    aligned = {"inicio": [], "fin": []}
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            data = fold_series(evaluation, fold)
            spans = base_events(data, evaluation, fold)
            events.append(_event_rows(data, spans, evaluation, fold))
            for anchor, part in _aligned(data, spans).items():
                aligned[anchor].append(part)
            corr = _timestep_correlations(data, rng)
            corr["evaluación"], corr["fold"] = evaluation, fold
            correlations.append(corr)
        progress(f"evaluación {evaluation} lista")
    summary = {}
    for anchor, parts in aligned.items():
        kinds = np.concatenate([p["tipo"] for p in parts])
        classes = np.concatenate([p["clase duración"] for p in parts])
        curves = {name: np.concatenate([p["curves"][name] for p in parts]) for name in SERIES}
        rows = []
        groups = [(k, "todos", kinds == k) for k in ("TP", "FN", "FP")]
        groups += [(k, c, (kinds == k) & (classes == c)) for k in ("TP", "FN", "FP") for c in DURATION_CLASSES]
        for kind, cls, mask in groups:
            for name in SERIES:
                block = curves[name][mask]
                with np.errstate(all="ignore"):
                    mean = np.nanmean(block, axis=0)
                    q25, q50, q75 = np.nanpercentile(block, [25, 50, 75], axis=0)
                rows.append({"tipo": kind, "clase duración": cls, "serie": name, "n": int(mask.sum()),
                             "media": mean, "q25": q25, "mediana": q50, "q75": q75})
        summary[anchor] = pd.DataFrame(rows)
    stored = {
        "version": CACHE_VERSION,
        "events": pd.concat(events, ignore_index=True),
        "aligned": summary,
        "offsets_s": ALIGN_OFFSETS / SAMPLE_RATE_HZ,
        "correlations": pd.concat(correlations, ignore_index=True),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump(stored, handle)
    return stored


def rescue_table(events: pd.DataFrame, measures=("u total DE", "MI DE", "u total NP", "MI NP"),
                 quantile: float = 0.8) -> pd.DataFrame:
    """Per event type: % where SEED is calm but the method is alert (and the reverse).

    "Alert" = above the ``quantile`` of that measure over all events.  If the
    method only re-ranks the same events as SEED (zero sum), both columns are small.
    """
    seed_cut = events["u SEED"].quantile(quantile)
    rows = []
    for measure in measures:
        cut = events[measure].quantile(quantile)
        for kind in EVENT_TYPES:
            part = events[events["tipo"] == kind]
            seed_alert, method_alert = part["u SEED"] >= seed_cut, part[measure] >= cut
            rows.append({"medida": measure, "tipo": kind, "n": len(part),
                         "% SEED alerta": 100 * seed_alert.mean(),
                         "% método alerta": 100 * method_alert.mean(),
                         "% solo el método alerta (rescate)": 100 * (method_alert & ~seed_alert).mean(),
                         "% solo SEED alerta": 100 * (seed_alert & ~method_alert).mean()})
    return pd.DataFrame(rows)


def error_auroc_by_duration(events: pd.DataFrame) -> pd.DataFrame:
    """``error_auroc`` inside each duration class (removes the "errors are shorter" effect)."""
    frames = []
    for cls in DURATION_CLASSES:
        part = events[(events["clase duración"] == cls) | (events["tipo"] == "TN")]
        table = error_auroc(part[part["tipo"] != "TN"])
        table["clase duración"] = cls
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def error_auroc(events: pd.DataFrame) -> pd.DataFrame:
    """AUROC for FP vs TP and FN vs TP, per measure and combined with SEED's u (leave-one-fold-out logistic).

    Measured per test fold and averaged.  The combined rows answer whether a
    method's uncertainty adds information on top of SEED's own uncertainty.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    single = ["u SEED", "u total DE", "MI DE", "u total NP", "MI NP"]
    combos = {"u SEED + MI DE": ["u SEED", "MI DE"], "u SEED + MI NP": ["u SEED", "MI NP"],
              "u SEED + MI DE + MI NP": ["u SEED", "MI DE", "MI NP"]}
    rows = []
    keys = events[["evaluación", "fold"]].drop_duplicates().itertuples(index=False)
    for evaluation, fold in keys:
        held = (events["evaluación"] == evaluation) & (events["fold"] == fold)
        for error in ("FP", "FN"):
            data = events[events["tipo"].isin(["TP", error])]
            target = (data["tipo"] == error).to_numpy()
            test, train = held[data.index].to_numpy(), ~held[data.index].to_numpy()
            row = {"evaluación": evaluation, "fold": fold, "error vs TP": f"{error} vs TP"}
            for measure in single:
                row[measure] = roc_auc_score(target[test], data.loc[test, measure])
            for name, cols in combos.items():
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                model.fit(data.loc[train, cols], target[train])
                row[name] = roc_auc_score(target[test], model.predict_proba(data.loc[test, cols])[:, 1])
            rows.append(row)
    return pd.DataFrame(rows)
