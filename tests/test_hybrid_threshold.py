import numpy as np
import pandas as pd

from eeg_uq.hybrid_threshold import _match_labels, fast_event_ious, fast_process_spindle, pool_f1
from eeg_uq.metrics import get_results_simple, process_spindle_simple


def _random_masks(seed: int, count: int = 200, length: int = 3000) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(count):
        # Smoothed noise gives runs of very different lengths (short, long, > 5 s).
        noise = np.convolve(rng.normal(size=length), np.ones(rng.integers(5, 400)), mode="same")
        mask = (noise > rng.normal(0, 3)).astype(np.int8)
        if rng.random() < 0.2:
            mask[0] = 1  # events touching the first sample
        if rng.random() < 0.2:
            mask[-rng.integers(1, 5):] = 1  # events touching the last sample
        masks.append(mask)
    return masks


def test_fast_process_spindle_matches_legacy():
    for mask in _random_masks(0):
        expected = process_spindle_simple(mask, mask)
        np.testing.assert_array_equal(fast_process_spindle(mask), expected)


def test_fast_event_ious_matches_legacy():
    labels = _random_masks(1)
    outputs = _random_masks(2)
    for label, output in zip(labels, outputs):
        processed = process_spindle_simple(output, output)
        assert fast_event_ious(label, processed) == get_results_simple(label, processed)
        assert fast_event_ious(label, output) == get_results_simple(label, output)


def test_pool_f1_counts_one_to_one_matches():
    events = pd.DataFrame({"fila": [0, 0, 0], "inicio": [10, 15, 100], "fin": [50, 45, 140]})
    labelled = _match_labels(events, {0: [(12, 48), (300, 340)]})
    # Two events overlap the same true spindle (IoU 0.90 and 0.83): only the best one counts as real.
    assert labelled["real"].tolist() == [1, 0, 0]
    f1, precision = pool_f1(labelled, np.array([True, True, True]), labelled.attrs["n_real"])
    assert np.isclose(f1, 2 * 1 / (3 + 2))
    assert np.isclose(precision, 1 / 3)
