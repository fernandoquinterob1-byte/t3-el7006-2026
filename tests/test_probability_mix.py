import numpy as np

from eeg_uq.probability_mix import FEATURE_NAMES, _features, _logit, grid_probability


def _pair(seed: int = 0, records: int = 3, length: int = 50) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "y_true": rng.integers(0, 2, (records, length)).astype(np.int8),
        "de": {"probability": rng.random((records, length)), "mi": rng.random((records, length)) * 0.1,
               "valid_mask": np.ones((records, length), np.uint8)},
        "np": {"probability": rng.random((records, length)), "mi": rng.random((records, length)) * 0.1},
    }


def test_grid_weight_endpoints_are_each_method():
    pair = _pair()
    np.testing.assert_allclose(grid_probability(pair, 0.0), pair["de"]["probability"])
    np.testing.assert_allclose(grid_probability(pair, 1.0), pair["np"]["probability"])
    np.testing.assert_allclose(grid_probability(pair, 0.5),
                               0.5 * pair["de"]["probability"] + 0.5 * pair["np"]["probability"])


def test_features_match_names_and_logit_inverts_sigmoid():
    pair = _pair()
    for with_mi in (False, True):
        assert _features(pair, with_mi).shape[-1] == len(FEATURE_NAMES[with_mi])
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(1 / (1 + np.exp(-_logit(p))), p)
