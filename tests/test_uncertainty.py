import numpy as np

from eeg_uq.uncertainty.baseline import analyse_prediction_file, binary_uncertainty


def test_binary_entropy_and_confidence_are_correct_at_extremes():
    scores = binary_uncertainty(np.asarray([0.5, 0.9, 0.1]))
    assert np.allclose(scores["confidence"], [0.5, 0.9, 0.9])
    assert np.allclose(scores["uncertainty"], [0.5, 0.1, 0.1])
    assert scores["entropy"][0] > scores["entropy"][1]


def test_prediction_analysis_uses_binary_temporal_arrays(tmp_path):
    path = tmp_path / "predictions.npz"
    np.savez_compressed(
        path,
        probabilities=np.asarray([[0.9, 0.2], [0.6, 0.4]], dtype=np.float32),
        y_true=np.asarray([[1, 0], [0, 1]], dtype=np.int8),
        y_pred=np.asarray([[1, 0], [1, 0]], dtype=np.int8),
    )
    report, arrays = analyse_prediction_file(path)
    assert report["correct_entropy"]["count"] == 2
    assert report["incorrect_entropy"]["count"] == 2
    assert arrays["temporal_error"].sum() == 2
