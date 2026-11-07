"""Mixed uncertainty threshold: Deep Ensemble detects, Neural Process MI corrects.

Everything runs on the exported validation/test NPZ files (no training, no
inference).  Per fold, every threshold or model is fitted on the validation
split and applied once to the test split of the same fold.

Steps (see ``notebooks/09_umbral_mixto.ipynb``):

1. Baseline with tuned thresholds: the event post-processing only depends on
   the *duration* threshold (``process_spindle_simple`` ignores the detection
   mask), so that one threshold is tuned on validation.
2. Event table: Deep Ensemble detections plus "near misses" (spans with
   probability above a low threshold that the detector did not report), with
   features from both methods and a label from one-to-one IoU matching.
3. Complementarity: does the Neural Process MI add information over the Deep
   Ensemble probability?
4. Rules: (1) hard thresholds, (2) an MI-adaptive probability threshold,
   (3) an event-level logistic regression.
5. Final F1 / precision / recall with the legacy pairing on the edited masks.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .method_summary import load_fold
from .metrics import (
    DURATION_THRESHOLD,
    EVENT_IOU_THRESHOLD,
    SAMPLE_RATE_HZ,
    calculate_metrics_simple,
)
from .report import CACHE, EVALUATIONS, FOLDS, method_path

CACHE_FILE = CACHE / "hybrid_threshold.pkl"
CACHE_VERSION = 1

MIN_SAMPLES = int(SAMPLE_RATE_HZ * 0.3)
MAX_SAMPLES = int(SAMPLE_RATE_HZ * 5)
DURATION_GRID = np.round(np.arange(0.25, 0.651, 0.025), 3)
CANDIDATE_THRESHOLD = 0.25  # near misses: DE probability >= this, not detected
BETA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
PROB_GRID = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01)  # 1.01 = no probability condition
QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98)
DE_FEATURES = ["p media DE", "p máx DE", "std DE", "log MI DE", "duración (s)", "candidato"]
NP_FEATURES = ["log MI NP", "p media NP", "std NP"]
METHOD_KINDS = {"Deep Ensemble": "sampling", "Neural Process": "neural_process", "SEED": "seed"}


# ------------------------------------------------------------ fast legacy code
# ``metrics.process_spindle_simple`` and ``metrics._legacy_event_stamps`` walk
# every sample in Python.  Their state only changes where the input changes,
# so the loops below visit only those indices (plus the first/second/last
# sample, where the legacy code has edge cases).  ``tests`` check that the
# outputs are identical to the legacy functions.

def _change_points(*arrays: np.ndarray) -> np.ndarray:
    length = len(arrays[0])
    points = {0, min(1, length - 1), length - 1}
    for values in arrays:
        points.update((np.flatnonzero(np.diff(values.astype(np.int8)) != 0) + 1).tolist())
    return np.asarray(sorted(points), dtype=np.int64)


def fast_process_spindle(dur_out: np.ndarray) -> np.ndarray:
    """Same output as ``metrics.process_spindle_simple(det, dur_out)`` (``det`` is unused there)."""
    dur_out = np.asarray(dur_out)
    new_output = dur_out.copy()
    last_label, in_event = 0, False
    for index in _change_points(dur_out):
        label = dur_out[index]
        if label == 1 and not in_event:
            if last_label != 0 and index - last_label < MIN_SAMPLES:
                new_output[last_label:index] = 1
            in_event = True
        elif label == 0 and in_event:
            last_label, in_event = index, False
    start_label, in_event = 0, False
    length = len(new_output)
    reference = new_output.copy()  # later edits are always behind the cursor
    for index in _change_points(reference):
        label = reference[index]
        if label == 1 and not in_event:
            start_label, in_event = index, True
        elif (label == 0 or index + 1 == length) and in_event:
            if index - start_label < MIN_SAMPLES:
                new_output[start_label:index] = 0
            elif index - start_label > MAX_SAMPLES:
                start_trim = (index - start_label - MAX_SAMPLES) // 2 + start_label
                end_trim = start_trim + MAX_SAMPLES
                new_output[start_label:start_trim] = 0
                new_output[end_trim:index] = 0
            in_event = False
    return new_output


def fast_event_ious(labels: np.ndarray, output: np.ndarray) -> list[float]:
    """Same list as ``metrics.get_results_simple(labels, output)``."""
    labels, output = np.asarray(labels), np.asarray(output)
    length = len(labels)
    label_stamps, predict_stamps = [], []
    in_predict, in_label = 0, 0
    for index in _change_points(labels, output):
        label, predict = labels[index], output[index]
        if not in_label and label == 1:
            in_label = index
        elif in_label and (label == 0 or index + 1 == length):
            label_stamps.append((in_label, index))
            in_label = 0
            if (not in_predict and len(predict_stamps) < len(label_stamps)) or len(predict_stamps) + 1 < len(label_stamps):
                predict_stamps.append(0)
        if not in_predict and predict == 1:
            in_predict = index
        elif in_predict and (predict == 0 or index + 1 == length):
            predict_stamps.append((in_predict, index))
            in_predict = 0
            if (not in_label and len(label_stamps) < len(predict_stamps)) or len(label_stamps) + 1 < len(predict_stamps):
                label_stamps.append(0)
    results = []
    for truth, prediction in zip(label_stamps, predict_stamps):
        if not truth and prediction:
            results.append(-1.0)
        elif not prediction and truth:
            results.append(0.0)
        else:
            intersection = min(truth[1], prediction[1]) - max(truth[0], prediction[0])
            union = max(truth[1], prediction[1]) - min(truth[0], prediction[0])
            results.append(intersection / union)
    return results


def process_masks(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Post-processed detection mask for every record."""
    binary = (probability >= threshold).astype(np.int8)
    return np.stack([fast_process_spindle(row) for row in binary])


def event_scores(y_true: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    """Legacy event F1, precision, recall and counts (IoU >= 0.2), micro over records."""
    ious = [value for truth, mask in zip(y_true, masks) for value in fast_event_ious(truth, mask)]
    curve = calculate_metrics_simple(ious, thresholds=np.array([EVENT_IOU_THRESHOLD]))
    return {
        "F1 (%)": 100 * float(curve["f1"][0]),
        "precisión (%)": 100 * float(curve["precision"][0]),
        "recall (%)": 100 * float(curve["recall"][0]),
        "TP": int(curve["true_positives"][0]),
        "FP": int(curve["false_positives"][0]),
        "FN": int(curve["false_negatives"][0]),
    }


# ------------------------------------------------------------------ loading

def load_pair(evaluation: int, fold: int, split: str) -> dict:
    """Deep Ensemble and Neural Process outputs of one fold, aligned by record."""
    de = load_fold(method_path("Deep Ensemble", evaluation, fold, split), "sampling")
    np_ = load_fold(method_path("Neural Process", evaluation, fold, split), "neural_process")
    if not np.array_equal(de["sample_id"], np_["sample_id"]):
        raise ValueError(f"Deep Ensemble and Neural Process records differ (eval {evaluation}, fold {fold}, {split}).")
    if not np.array_equal(de["y_true"], np_["y_true"]):
        raise ValueError(f"Labels differ between methods (eval {evaluation}, fold {fold}, {split}).")
    return {"y_true": de["y_true"].astype(np.int8), "de": de, "np": np_}


# --------------------------------------------------------------- step 1

def tune_duration_threshold(method: str, evaluation: int, fold: int) -> dict:
    """Best duration threshold on validation, applied to test (legacy F1)."""
    kind = METHOD_KINDS[method]
    val = load_fold(method_path(method, evaluation, fold, "validation"), kind)
    test = load_fold(method_path(method, evaluation, fold, "test"), kind)
    val_f1 = {t: event_scores(val["y_true"], process_masks(val["probability"], t))["F1 (%)"] for t in DURATION_GRID}
    best = max(val_f1, key=lambda t: (val_f1[t], -abs(t - DURATION_THRESHOLD)))
    original = event_scores(test["y_true"], process_masks(test["probability"], DURATION_THRESHOLD))
    tuned = event_scores(test["y_true"], process_masks(test["probability"], best))
    return {"método": method, "evaluación": evaluation, "fold": fold, "umbral elegido": float(best),
            "F1 val (original)": val_f1[min(DURATION_GRID, key=lambda t: abs(t - DURATION_THRESHOLD))],
            "F1 val (ajustado)": val_f1[best],
            **{f"{k} original": v for k, v in original.items()},
            **{f"{k} ajustado": v for k, v in tuned.items()}}


# --------------------------------------------------------------- step 2

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _match_labels(events: pd.DataFrame, truth_runs: dict[int, list[tuple[int, int]]]) -> pd.DataFrame:
    """One-to-one greedy IoU matching of the event pool to true spindles."""
    pairs = []
    for index, event in events.iterrows():
        for true_index, (ts, te) in enumerate(truth_runs[event["fila"]]):
            inter = min(te, event["fin"]) - max(ts, event["inicio"])
            if inter > 0:
                union = max(te, event["fin"]) - min(ts, event["inicio"])
                pairs.append((inter / union, index, event["fila"], true_index))
    label = pd.Series(0, index=events.index)
    iou = pd.Series(0.0, index=events.index)
    used_events, used_truth = set(), set()
    for value, index, row, true_index in sorted(pairs, reverse=True):
        if value < EVENT_IOU_THRESHOLD or index in used_events or (row, true_index) in used_truth:
            continue
        used_events.add(index)
        used_truth.add((row, true_index))
        label[index], iou[index] = 1, value
    events = events.assign(real=label.to_numpy(), IoU=iou.to_numpy())
    events.attrs["n_real"] = int(sum(len(runs) for runs in truth_runs.values()))
    return events


def event_table(pair: dict, duration_threshold: float, candidate_threshold: float = CANDIDATE_THRESHOLD) -> pd.DataFrame:
    """Detections and near misses of the Deep Ensemble with features from both methods."""
    de, np_ = pair["de"], pair["np"]
    detections = process_masks(de["probability"], duration_threshold)
    loose = process_masks(de["probability"], candidate_threshold)
    log_mi_de, log_mi_np = np.log10(de["mi"] + 1e-6), np.log10(np_["mi"] + 1e-6)
    rows = []
    for row in range(len(detections)):
        detected = _runs(detections[row])
        spans = [(s, e, 0) for s, e in detected]
        for s, e in _runs(loose[row]):
            if not detections[row, s:e].any():
                spans.append((s, e, 1))
        for s, e, candidate in spans:
            rows.append({
                "fila": row, "inicio": s, "fin": e, "candidato": candidate,
                "duración (s)": (e - s) / SAMPLE_RATE_HZ,
                "p media DE": de["probability"][row, s:e].mean(),
                "p máx DE": de["probability"][row, s:e].max(),
                "std DE": de["std"][row, s:e].mean(),
                "log MI DE": log_mi_de[row, s:e].mean(),
                "MI DE": de["mi"][row, s:e].mean(),
                "log MI NP": log_mi_np[row, s:e].mean(),
                "MI NP": np_["mi"][row, s:e].mean(),
                "p media NP": np_["probability"][row, s:e].mean(),
                "std NP": np_["std"][row, s:e].mean(),
            })
    events = pd.DataFrame(rows)
    truth_runs = {row: _runs(pair["y_true"][row]) for row in range(len(pair["y_true"]))}
    events = _match_labels(events, truth_runs)
    events.attrs["n_real"] = int(sum(len(r) for r in truth_runs.values()))
    return events


def pool_f1(events: pd.DataFrame, keep: np.ndarray, n_real: int) -> tuple[float, float]:
    """Event F1 and precision of a kept subset (one-to-one labels of the pool)."""
    kept = int(keep.sum())
    hits = int(events["real"].to_numpy()[keep].sum())
    f1 = 2 * hits / (kept + n_real) if kept + n_real else 0.0
    precision = hits / kept if kept else 0.0
    return f1, precision


# --------------------------------------------------------------- step 4

def _fit_logistic(events: pd.DataFrame, features: list[str]):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight=None))
    model.fit(events[features].to_numpy(), events["real"].to_numpy())
    return model


def _choose(options: list[tuple], min_precision: float | None) -> tuple:
    """Best (f1, precision, params...) option, optionally with a precision floor."""
    if min_precision is not None:
        allowed = [o for o in options if o[1] >= min_precision]
        options = allowed or options
    return max(options, key=lambda o: (round(o[0], 6), o[1]))


def rule_hard(val: pd.DataFrame, min_precision: float | None) -> dict:
    """Remove detections with p < t_p and MI_NP > τ_FP; add near misses with MI_NP > τ_FN."""
    det, cand = val["candidato"].to_numpy() == 0, val["candidato"].to_numpy() == 1
    mi, p = val["MI NP"].to_numpy(), val["p media DE"].to_numpy()
    fp_cuts = [np.inf] + [float(np.quantile(mi[det], q)) for q in QUANTILES]
    fn_cuts = [np.inf] + [float(np.quantile(mi[cand], q)) for q in QUANTILES] if cand.any() else [np.inf]
    options = []
    for t_p in PROB_GRID:
        for tau_fp in fp_cuts:
            remove = det & (p < t_p) & (mi > tau_fp)
            for tau_fn in fn_cuts:
                keep = (det & ~remove) | (cand & (mi > tau_fn))
                f1, precision = pool_f1(val, keep, val.attrs["n_real"])
                options.append((f1, precision, t_p, tau_fp, tau_fn))
    _, _, t_p, tau_fp, tau_fn = _choose(options, min_precision)
    return {"t_p": t_p, "tau_fp": tau_fp, "tau_fn": tau_fn}


def apply_hard(events: pd.DataFrame, params: dict) -> np.ndarray:
    det, cand = events["candidato"].to_numpy() == 0, events["candidato"].to_numpy() == 1
    mi, p = events["MI NP"].to_numpy(), events["p media DE"].to_numpy()
    remove = det & (p < params["t_p"]) & (mi > params["tau_fp"])
    return (det & ~remove) | (cand & (mi > params["tau_fn"]))


CUT_GRID = np.round(np.arange(0.10, 0.901, 0.02), 3)


def rule_logistic(val: pd.DataFrame, features: list[str], min_precision: float | None) -> dict:
    """One logistic model for detections and another for near misses.

    The MI means opposite things in the two groups (high MI on a detection
    points to a false positive; on a near miss, to a missed spindle), so a
    single shared coefficient cannot use it.  Each group gets its own model
    and its own probability cut, both chosen on validation.
    """
    features = [f for f in features if f != "candidato"]
    groups = {"det": val["candidato"].to_numpy() == 0, "cand": val["candidato"].to_numpy() == 1}
    models, scores = {}, np.zeros(len(val))
    for name, mask in groups.items():
        target = val["real"].to_numpy()[mask]
        if mask.sum() < 10 or target.min() == target.max():
            models[name] = None
            scores[mask] = 1.0 if name == "det" else 0.0
            continue
        models[name] = _fit_logistic(val[mask], features)
        scores[mask] = models[name].predict_proba(val.loc[mask, features].to_numpy())[:, 1]
    options = []
    for cut_det in CUT_GRID:
        keep_det = groups["det"] & (scores >= cut_det)
        for cut_cand in list(CUT_GRID) + [1.01]:
            keep = keep_det | (groups["cand"] & (scores >= cut_cand))
            f1, precision = pool_f1(val, keep, val.attrs["n_real"])
            options.append((f1, precision, cut_det, cut_cand))
    _, _, cut_det, cut_cand = _choose(options, min_precision)
    return {"models": models, "features": features, "cut_det": float(cut_det), "cut_cand": float(cut_cand)}


def logistic_scores(events: pd.DataFrame, params: dict) -> np.ndarray:
    scores = np.zeros(len(events))
    for name, flag in (("det", 0), ("cand", 1)):
        mask = events["candidato"].to_numpy() == flag
        model = params["models"][name]
        if model is None:
            scores[mask] = 1.0 if name == "det" else 0.0
        elif mask.any():
            scores[mask] = model.predict_proba(events.loc[mask, params["features"]].to_numpy())[:, 1]
    return scores


def apply_logistic(events: pd.DataFrame, params: dict) -> np.ndarray:
    scores = logistic_scores(events, params)
    det = events["candidato"].to_numpy() == 0
    return np.where(det, scores >= params["cut_det"], scores >= params["cut_cand"])


def edited_masks(shape: tuple[int, int], events: pd.DataFrame, keep: np.ndarray) -> np.ndarray:
    masks = np.zeros(shape, dtype=np.int8)
    for row, start, end in events.loc[keep, ["fila", "inicio", "fin"]].itertuples(index=False):
        masks[row, start:end] = 1
    return masks


def _mi_cdf(reference: np.ndarray):
    grid = np.quantile(reference, np.linspace(0, 1, 1001))
    return lambda values: np.interp(values, grid, np.linspace(0, 1, 1001))


def adaptive_masks(pair: dict, base: float, beta: float, cdf) -> np.ndarray:
    """Probability threshold lowered where Neural Process is unsure: p >= base - beta * F(MI_NP)."""
    threshold = base - beta * cdf(pair["np"]["mi"])
    binary = (pair["de"]["probability"] >= threshold).astype(np.int8)
    return np.stack([fast_process_spindle(row) for row in binary])


# --------------------------------------------------------------- per fold

def _rules_on_base(val_pair: dict, test_pair: dict, base_t: float, base_name: str, tag: dict) -> dict:
    """Every rule on top of one Deep Ensemble base threshold (fitted on validation, scored on test)."""
    tag = {**tag, "base": base_name}
    shape = test_pair["y_true"].shape
    base_val = event_scores(val_pair["y_true"], process_masks(val_pair["de"]["probability"], base_t))
    base_val_precision = base_val["precisión (%)"] / 100

    # Step 2: event tables (detections + near misses).
    val_events = event_table(val_pair, base_t)
    test_events = event_table(test_pair, base_t)

    results = {"Deep Ensemble": event_scores(test_pair["y_true"], process_masks(test_pair["de"]["probability"], base_t))}
    params = {"umbral DE": float(base_t)}
    floors = (("", None), (" (precisión ≥ base)", base_val_precision))

    # Rule 1: hard thresholds.
    for suffix, floor in floors:
        hard = rule_hard(val_events, floor)
        results["Regla 1: cortes fijos" + suffix] = event_scores(
            test_pair["y_true"], edited_masks(shape, test_events, apply_hard(test_events, hard)))
        params.update({f"regla 1{suffix}: {k}": v for k, v in hard.items()})

    # Rule 2: MI-adaptive probability threshold (timestep level).
    cdf = _mi_cdf(val_pair["np"]["mi"][val_pair["de"]["valid_mask"].astype(bool)])
    beta_scores = {b: event_scores(val_pair["y_true"], adaptive_masks(val_pair, base_t, b, cdf)) for b in BETA_GRID}
    for suffix, floor in floors:
        allowed = [b for b in BETA_GRID if floor is None or beta_scores[b]["precisión (%)"] / 100 >= floor] or list(BETA_GRID)
        beta = max(allowed, key=lambda b: (beta_scores[b]["F1 (%)"], -b))
        results["Regla 2: umbral adaptativo" + suffix] = event_scores(test_pair["y_true"], adaptive_masks(test_pair, base_t, beta, cdf))
        params[f"regla 2{suffix}: beta"] = beta

    # Rule 3: logistic regression (DE features only vs DE + NP).
    test_scores = {}
    for name, features in (("solo DE", DE_FEATURES), ("DE + NP", DE_FEATURES + NP_FEATURES)):
        for suffix, floor in floors:
            fitted = rule_logistic(val_events, features, floor)
            results[f"Regla 3: logística {name}{suffix}"] = event_scores(
                test_pair["y_true"], edited_masks(shape, test_events, apply_logistic(test_events, fitted)))
            params[f"regla 3 {name}{suffix}: corte detecciones"] = fitted["cut_det"]
            params[f"regla 3 {name}{suffix}: corte casi detectados"] = fitted["cut_cand"]
            if not suffix:
                test_scores[name] = logistic_scores(test_events, fitted)
                for group, model in fitted["models"].items():
                    if model is not None:
                        params.update({f"coef {name} [{group}]: {f}": float(c)
                                       for f, c in zip(fitted["features"], model[-1].coef_[0])})

    scores = pd.DataFrame([{**tag, "regla": rule, **values} for rule, values in results.items()])

    # Step 3: complementarity on test events (models fitted on validation).
    separation = []
    for group, mask in (("detecciones: TP vs FP", test_events["candidato"] == 0),
                        ("casi detectados: huso real vs no", test_events["candidato"] == 1)):
        part = test_events[mask]
        is_detection = group.startswith("detecciones")
        target = part["real"].to_numpy() == (0 if is_detection else 1)  # positive = the error to find
        row = {**tag, "grupo": group, "eventos": len(part), "positivos": int(target.sum()),
               "ρ Spearman MI DE vs MI NP": float(spearmanr(part["MI DE"], part["MI NP"])[0]) if len(part) > 2 else np.nan}
        if 0 < target.sum() < len(target):
            # FP: low probability / high MI.  Real near miss: high probability / high MI.
            for measure in ("p media DE", "MI DE", "MI NP"):
                values = part[measure].to_numpy()
                score = -values if (is_detection and measure.startswith("p")) else values
                row[f"AUROC {measure}"] = roc_auc_score(target, score)
            for name, prob in test_scores.items():
                prob = prob[mask.to_numpy()]
                score = -prob if is_detection else prob
                row[f"AUROC logística {name}"] = roc_auc_score(target, score)
                row[f"AUPRC logística {name}"] = average_precision_score(target, score)
            row["azar AUPRC"] = float(target.mean())
        separation.append(row)

    near = test_events["candidato"] == 1
    counts = {**tag, "detecciones (test)": int((~near).sum()), "casi detectados (test)": int(near.sum()),
              "% reales en casi detectados": 100 * float(test_events.loc[near, "real"].mean()) if near.any() else np.nan,
              "% de los FN que están en casi detectados":
                  100 * int(test_events.loc[near, "real"].sum()) / max(results["Deep Ensemble"]["FN"], 1)}
    events_out = test_events.drop(columns=["fila", "inicio", "fin"]).assign(**tag)
    return {"scores": scores, "params": pd.DataFrame([{**tag, **params}]), "separation": pd.DataFrame(separation),
            "counts": pd.DataFrame([counts]), "events": events_out}


def fold_analysis(evaluation: int, fold: int) -> dict:
    """Rules on the original (0.425) and on the validation-tuned DE threshold."""
    tag = {"evaluación": evaluation, "fold": fold}
    val_pair, test_pair = load_pair(evaluation, fold, "validation"), load_pair(evaluation, fold, "test")
    val_f1 = {t: event_scores(val_pair["y_true"], process_masks(val_pair["de"]["probability"], t))["F1 (%)"]
              for t in DURATION_GRID}
    tuned = max(val_f1, key=lambda t: (val_f1[t], -abs(t - DURATION_THRESHOLD)))
    parts: dict[str, list] = {}
    for base_name, base_t in (("original 0.425", DURATION_THRESHOLD), ("ajustado en validación", float(tuned))):
        for name, value in _rules_on_base(val_pair, test_pair, base_t, base_name, tag).items():
            parts.setdefault(name, []).append(value)
    return {name: pd.concat(values, ignore_index=True) for name, values in parts.items()}


def tuned_baselines() -> pd.DataFrame:
    rows = [tune_duration_threshold(method, e, f) for method in METHOD_KINDS for e in EVALUATIONS for f in FOLDS]
    return pd.DataFrame(rows)


def run_all(refresh: bool = False, progress=print) -> dict[str, pd.DataFrame]:
    if CACHE_FILE.is_file() and not refresh:
        with CACHE_FILE.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored["results"]
    parts: dict[str, list] = {}
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            for name, value in fold_analysis(evaluation, fold).items():
                parts.setdefault(name, []).append(value)
            progress(f"evaluación {evaluation}, fold {fold} listo")
    results = {name: pd.concat(values, ignore_index=True) for name, values in parts.items()}
    progress("líneas base de SEED, Deep Ensemble y Neural Process…")
    results["baselines"] = tuned_baselines()
    CACHE.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "results": results}, handle)
    return results


# ------------------------------------------- correct only one type of error

SINGLE_CACHE_FILE = CACHE / "hybrid_single_error.pkl"


def single_error_fold(evaluation: int, fold: int) -> pd.DataFrame:
    """Rules that only remove detections (FP) or only add near misses (FN), on the 0.425 base.

    Fewer parameters than the joint rules: each one is fitted on validation
    with the event F1 and scored once on test.
    """
    tag = {"evaluación": evaluation, "fold": fold, "base": "original 0.425"}
    val_pair, test_pair = load_pair(evaluation, fold, "validation"), load_pair(evaluation, fold, "test")
    val, test = event_table(val_pair, DURATION_THRESHOLD), event_table(test_pair, DURATION_THRESHOLD)
    shape, n_val = test_pair["y_true"].shape, val.attrs["n_real"]
    vdet, vcand = val["candidato"].to_numpy() == 0, val["candidato"].to_numpy() == 1
    tdet, tcand = test["candidato"].to_numpy() == 0, test["candidato"].to_numpy() == 1
    vmi, vp = val["MI NP"].to_numpy(), val["p media DE"].to_numpy()
    tmi, tp = test["MI NP"].to_numpy(), test["p media DE"].to_numpy()

    def score(keep):
        return event_scores(test_pair["y_true"], edited_masks(shape, test, keep))

    results = {"Deep Ensemble": score(tdet)}

    # Rule 1, only FP: remove detections with p_DE < t_p and MI_NP > tau.
    options = []
    for t_p in PROB_GRID:
        for q in (None,) + QUANTILES:
            tau = np.inf if q is None else float(np.quantile(vmi[vdet], q))
            options.append((*pool_f1(val, vdet & ~((vp < t_p) & (vmi > tau)), n_val), t_p, tau))
    _, _, t_p, tau = _choose(options, None)
    results["Solo FP: cortes fijos (MI de NP)"] = score(tdet & ~((tp < t_p) & (tmi > tau)))

    # Rule 1, only FN: add near misses with MI_NP > tau.
    options = []
    for q in (None,) + QUANTILES:
        tau = np.inf if q is None else float(np.quantile(vmi[vcand], q))
        options.append((*pool_f1(val, vdet | (vcand & (vmi > tau)), n_val), tau))
    _, _, tau = _choose(options, None)
    results["Solo FN: cortes fijos (MI de NP)"] = score(tdet | (tcand & (tmi > tau)))

    # Rule 3, one logistic model per group.
    for name, features in (("solo DE", DE_FEATURES), ("DE + NP", DE_FEATURES + NP_FEATURES)):
        features = [f for f in features if f != "candidato"]
        model = _fit_logistic(val[vdet], features)
        s_val = model.predict_proba(val.loc[vdet, features].to_numpy())[:, 1]
        s_test = model.predict_proba(test.loc[tdet, features].to_numpy())[:, 1]
        options = []
        for cut in np.concatenate([[0.0], CUT_GRID]):
            keep = vdet.copy()
            keep[vdet] = s_val >= cut
            options.append((*pool_f1(val, keep, n_val), cut))
        _, _, cut = _choose(options, None)
        keep = tdet.copy()
        keep[tdet] = s_test >= cut
        results[f"Solo FP: logística {name}"] = score(keep)

        model = _fit_logistic(val[vcand], features)
        s_val = model.predict_proba(val.loc[vcand, features].to_numpy())[:, 1]
        s_test = model.predict_proba(test.loc[tcand, features].to_numpy())[:, 1]
        options = []
        for cut in np.concatenate([CUT_GRID, [1.01]]):
            keep = vdet.copy()
            keep[vcand] = s_val >= cut
            options.append((*pool_f1(val, keep, n_val), cut))
        _, _, cut = _choose(options, None)
        keep = tdet.copy()
        keep[tcand] = s_test >= cut
        results[f"Solo FN: logística {name}"] = score(keep)

    return pd.DataFrame([{**tag, "regla": rule, **values} for rule, values in results.items()])


def run_single(refresh: bool = False, progress=print) -> pd.DataFrame:
    if SINGLE_CACHE_FILE.is_file() and not refresh:
        with SINGLE_CACHE_FILE.open("rb") as handle:
            stored = pickle.load(handle)
        if stored.get("version") == CACHE_VERSION:
            return stored["results"]
    frames = []
    for evaluation in EVALUATIONS:
        for fold in FOLDS:
            frames.append(single_error_fold(evaluation, fold))
            progress(f"evaluación {evaluation}, fold {fold} listo")
    results = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with SINGLE_CACHE_FILE.open("wb") as handle:
        pickle.dump({"version": CACHE_VERSION, "results": results}, handle)
    return results


# --------------------------------------------------------------- summaries

def summary(scores: pd.DataFrame, reference: str = "Deep Ensemble") -> pd.DataFrame:
    """Mean ± std over the 15 folds, paired change vs ``reference`` (same base) and Wilcoxon p-value."""
    rows = []
    for base_name, by_base in scores.groupby("base", sort=False):
        base = by_base[by_base["regla"] == reference].set_index(["evaluación", "fold"])
        for rule, part in by_base.groupby("regla", sort=False):
            part = part.set_index(["evaluación", "fold"])
            delta = (part["F1 (%)"] - base["F1 (%)"]).dropna()
            delta_p = (part["precisión (%)"] - base["precisión (%)"]).dropna()
            try:
                p_value = wilcoxon(delta).pvalue if (delta != 0).any() else np.nan
            except ValueError:
                p_value = np.nan
            rows.append({
                "base": base_name, "regla": rule,
                "F1 (%)": f"{part['F1 (%)'].mean():.2f} ± {part['F1 (%)'].std():.2f}",
                "precisión (%)": f"{part['precisión (%)'].mean():.2f} ± {part['precisión (%)'].std():.2f}",
                "recall (%)": f"{part['recall (%)'].mean():.2f} ± {part['recall (%)'].std():.2f}",
                "FP": round(part["FP"].mean(), 1), "FN": round(part["FN"].mean(), 1),
                "Δ F1 (puntos)": round(delta.mean(), 2),
                "Δ precisión (puntos)": round(delta_p.mean(), 2),
                "folds que mejoran F1": f"{int((delta > 0).sum())}/{len(delta)}",
                "p Wilcoxon": round(p_value, 4) if not np.isnan(p_value) else np.nan,
            })
    return pd.DataFrame(rows).set_index(["base", "regla"])
