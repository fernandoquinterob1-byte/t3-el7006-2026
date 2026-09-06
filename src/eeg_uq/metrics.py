"""Legacy-compatible event post-processing and event-level metrics.

The functions already used by the notebook are intentionally kept as the
reference implementation.  The record helpers below expose the same pairing
for analysis; they do not replace it with an alternative matching algorithm.
"""

import numpy as np


SAMPLE_RATE_HZ = 200
DETECTION_THRESHOLD = 0.5
DURATION_THRESHOLD = 0.425
EVENT_IOU_THRESHOLD = 0.2


def _legacy_event_stamps(labels: np.ndarray, output: np.ndarray) -> tuple[list, list]:
    """Copy the notebook's sequential placeholder-based event pairing setup."""
    label_stamps, predict_stamps = [], []
    in_predict_event = 0
    in_label_event = 0
    for index, (label, predict) in enumerate(zip(labels, output)):
        if not in_label_event and label == 1:
            in_label_event = index
        elif in_label_event and (label == 0 or index + 1 == len(labels)):
            label_stamps.append((in_label_event, index))
            in_label_event = 0
            if (not in_predict_event and len(predict_stamps) < len(label_stamps)) or (
                len(predict_stamps) + 1 < len(label_stamps)
            ):
                predict_stamps.append(0)
        if not in_predict_event and predict == 1:
            in_predict_event = index
        elif in_predict_event and (predict == 0 or index + 1 == len(output)):
            predict_stamps.append((in_predict_event, index))
            in_predict_event = 0
            if (not in_label_event and len(label_stamps) < len(predict_stamps)) or (
                len(label_stamps) + 1 < len(predict_stamps)
            ):
                label_stamps.append(0)
    return label_stamps, predict_stamps


def process_spindle_simple(det_out: np.ndarray, dur_out: np.ndarray) -> np.ndarray:
    """Replicates ``process_spindle_simple`` from the legacy notebook."""
    new_output = dur_out.copy()
    min_duration = int(SAMPLE_RATE_HZ * 0.3)
    max_duration = int(SAMPLE_RATE_HZ * 5)
    last_label = 0
    in_event = False
    for index, label in enumerate(dur_out):
        if label == 1 and not in_event:
            if last_label != 0 and index - last_label < min_duration:
                new_output[last_label:index] = 1
            in_event = True
        elif label == 0 and in_event:
            last_label = index
            in_event = False

    start_label = 0
    in_event = False
    for index, label in enumerate(new_output):
        if label == 1 and not in_event:
            start_label = index
            in_event = True
        elif (label == 0 or index + 1 == len(new_output)) and in_event:
            if index - start_label < min_duration:
                new_output[start_label:index] = 0
            elif index - start_label > max_duration:
                start_trim = (index - start_label - max_duration) // 2 + start_label
                end_trim = start_trim + max_duration
                new_output[start_label:start_trim] = 0
                new_output[end_trim:index] = 0
            in_event = False
    return new_output


def get_results_simple(labels: np.ndarray, output: np.ndarray) -> list[float]:
    """Replicates the notebook's event pairing and IoU/error representation."""
    label_stamps, predict_stamps = _legacy_event_stamps(labels, output)

    results = []
    for label, prediction in zip(label_stamps, predict_stamps):
        if not label and prediction:
            results.append(-1.0)
        elif not prediction and label:
            results.append(0.0)
        else:
            intersection = min(label[1], prediction[1]) - max(label[0], prediction[0])
            union = max(label[1], prediction[1]) - min(label[0], prediction[0])
            results.append(intersection / union)
    return results


def legacy_event_records(labels: np.ndarray, output: np.ndarray) -> list[dict]:
    """Return analysis records following exactly the legacy pairing order.

    ``status`` is evaluated at the historic IoU=0.2 point; the raw IoU stays
    available so later analyses can use another evaluation threshold.
    """
    label_stamps, predict_stamps = _legacy_event_stamps(labels, output)
    records = []
    for truth, prediction in zip(label_stamps, predict_stamps):
        if not truth and prediction:
            iou, status = -1.0, "FP"
        elif not prediction and truth:
            iou, status = 0.0, "FN"
        else:
            intersection = min(truth[1], prediction[1]) - max(truth[0], prediction[0])
            union = max(truth[1], prediction[1]) - min(truth[0], prediction[0])
            iou = intersection / union
            status = "TP" if iou >= EVENT_IOU_THRESHOLD else "LOW_IOU"
        records.append(
            {
                "true_start": -1 if not truth else truth[0],
                "true_end": -1 if not truth else truth[1],
                "pred_start": -1 if not prediction else prediction[0],
                "pred_end": -1 if not prediction else prediction[1],
                "iou": float(iou),
                "status": status,
            }
        )
    return records


def calculate_metrics_simple(
    results: list[float], thresholds: np.ndarray | None = None, threshold: float = EVENT_IOU_THRESHOLD
) -> dict:
    """Notebook-compatible micro-averaged precision, recall and F1 curve."""
    thresholds = np.linspace(0, 1, 100) if thresholds is None else thresholds
    true_positives = np.zeros_like(thresholds, dtype=float)
    false_positives = np.zeros_like(thresholds, dtype=float)
    false_negatives = np.zeros_like(thresholds, dtype=float)
    selected_index = int(np.argmin(np.abs(thresholds - threshold)))
    for index, value in enumerate(thresholds):
        for iou in results:
            if iou == -1:
                false_positives[index] += 1
            elif iou == 0:
                false_negatives[index] += 1
            elif iou >= value:
                true_positives[index] += 1
            else:
                false_negatives[index] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(true_positives / (true_positives + false_positives), nan=0.0)
        recall = np.nan_to_num(true_positives / (true_positives + false_negatives), nan=0.0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(recall),
            where=(precision + recall) != 0,
        )
    return {
        "thresholds": thresholds,
        "selected_index": selected_index,
        "selected_iou_threshold": float(thresholds[selected_index]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
