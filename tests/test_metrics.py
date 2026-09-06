import numpy as np

from eeg_uq.metrics import get_results_simple, legacy_event_records, process_spindle_simple


def test_event_records_preserve_legacy_iou_representation():
    truth = np.asarray([0, 1, 1, 0, 0], dtype=np.int8)
    prediction = np.asarray([0, 1, 1, 0, 0], dtype=np.int8)
    assert get_results_simple(truth, prediction) == [1.0]
    record = legacy_event_records(truth, prediction)[0]
    assert record["status"] == "TP"
    assert record["iou"] == 1.0


def test_spindle_postprocessing_removes_short_event():
    duration = np.zeros(100, dtype=np.int8)
    duration[10:20] = 1
    output = process_spindle_simple(duration, duration)
    assert not output.any()
