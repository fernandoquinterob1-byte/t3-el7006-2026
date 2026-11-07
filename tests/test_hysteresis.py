import numpy as np

from eeg_uq.hysteresis import clean_up, detect, hysteresis_mask
from eeg_uq.metrics import process_spindle_simple


def test_clean_up_with_legacy_minimum_matches_legacy():
    rng = np.random.default_rng(3)
    for _ in range(100):
        noise = np.convolve(rng.normal(size=2000), np.ones(rng.integers(5, 300)), mode="same")
        mask = (noise > rng.normal(0, 3)).astype(np.int8)
        np.testing.assert_array_equal(clean_up(mask), process_spindle_simple(mask, mask))


def test_detect_with_equal_thresholds_matches_legacy():
    rng = np.random.default_rng(4)
    probability = np.clip(np.convolve(rng.random(3000), np.ones(80) / 80, mode="same")[None, :] * 1.2, 0, 1)
    expected = process_spindle_simple(None, (probability[0] >= 0.425).astype(np.int8))
    np.testing.assert_array_equal(detect(probability)[0], expected)


def test_hysteresis_keeps_only_runs_that_reach_the_high_threshold():
    p = np.array([[0.1, 0.35, 0.6, 0.35, 0.1, 0.35, 0.35, 0.1]])
    np.testing.assert_array_equal(hysteresis_mask(p, 0.5, 0.3)[0], [0, 1, 1, 1, 0, 0, 0, 0])
    # Runs never continue across records.
    two = np.array([[0.1, 0.35, 0.35], [0.6, 0.35, 0.1]])
    np.testing.assert_array_equal(hysteresis_mask(two, 0.5, 0.3), [[0, 0, 0], [1, 1, 0]])
