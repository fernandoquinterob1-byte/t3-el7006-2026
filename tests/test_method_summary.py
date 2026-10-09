import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_uq.method_summary import entropy_bits, fold_metrics, positive_class_ece  # noqa: E402


def _fold(mi):
    truth = np.zeros((1, 4000), dtype=np.int8)
    truth[0, 1000:1400] = 1
    probability = np.full((1, 4000), 0.02)
    probability[0, 1000:1400] = 0.9
    probability[0, 2500:2700] = 0.7  # false positive
    prediction = (probability >= 0.5).astype(np.int8)
    return {
        "y_true": truth, "valid_mask": np.ones_like(truth), "y_pred": prediction, "processed_pred": prediction,
        "sample_id": np.array([0]), "event_iou_values": np.array([1.0, -1.0]),
        "event_sample_id": np.array([0, 0]), "event_pred_start": np.array([1000, 2500]),
        "event_pred_end": np.array([1400, 2700]), "event_status": np.array(["TP", "FP"]),
        "probability": probability, "total": entropy_bits(probability), "mi": mi,
    }


def test_metrics_with_and_without_mi():
    fold = _fold(None)
    row = fold_metrics(fold)
    assert np.isnan(row["AUROC MI errores"]) and np.isnan(row["% epistémica"])
    assert 0 < row["F1 evento (%)"] <= 100
    mi = np.zeros((1, 4000))
    mi[0, 2500:2700] = 0.5  # MI only on the false positive
    row = fold_metrics(_fold(mi))
    assert row["AUPRC MI FP vs TP"] == 1.0 and row["azar AUPRC FP"] == 0.5
    assert 0 < row["% epistémica"] < 100


def test_positive_class_ece_is_zero_when_calibrated():
    probability = np.repeat([0.25, 0.75], 400)
    target = np.r_[np.repeat([1, 0, 0, 0], 100), np.repeat([1, 1, 1, 0], 100)].astype(float)
    assert positive_class_ece(probability, target) < 1e-9


def test_event_uncertainty_by_type():
    from eeg_uq.method_summary import event_uncertainty

    fold = _fold(np.zeros((1, 4000)))
    fold["event_true_start"] = np.array([1000, -1])
    fold["event_true_end"] = np.array([1400, -1])
    fold["aleatoric"] = fold["total"]
    fold["std"] = None
    # Add a missed event: true spindle at 3200-3400 with no detection.
    fold["event_sample_id"] = np.r_[fold["event_sample_id"], 0]
    fold["event_status"] = np.append(fold["event_status"], "FN")
    fold["event_pred_start"] = np.r_[fold["event_pred_start"], -1]
    fold["event_pred_end"] = np.r_[fold["event_pred_end"], -1]
    fold["event_true_start"] = np.r_[fold["event_true_start"], 3200]
    fold["event_true_end"] = np.r_[fold["event_true_end"], 3400]
    frame = event_uncertainty(fold)
    assert sorted(zip(frame["tipo"], frame["tramo"])) == [("FN", "real"), ("FP", "predicho"), ("TP", "predicho"), ("TP", "real")]
    fp = frame[frame["tipo"] == "FP"].iloc[0]
    assert np.isclose(fp["probabilidad media"], 0.7) and np.isnan(fp["dispersión entre muestras (std)"])
    assert np.isclose(frame[frame["tipo"] == "FN"].iloc[0]["probabilidad media"], 0.02)
