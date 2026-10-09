import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_deep_ensemble_stacks_member_probabilities(tmp_path):
    from eeg_uq.uncertainty.deep_ensemble import combine_member_predictions

    truth = np.zeros((2, 2000), dtype=np.int8)
    truth[:, 500:800] = 1
    files = []
    for index, level in enumerate((0.9, 0.7)):
        probabilities = np.full((2, 2000), 0.05, dtype=np.float32)
        probabilities[:, 500:800] = level
        path = tmp_path / f"member_{index}.npz"
        np.savez(path, sample_id=np.array([3, 9]), subject_id=np.array(["a", "b"]), y_true=truth,
                 valid_mask=np.ones_like(truth, dtype=np.uint8), probabilities=probabilities)
        files.append(path)
    result = combine_member_predictions(files)
    assert result["predictive_probabilities"].shape == (2, 2, 2000)
    assert np.allclose(result["predictive_probabilities"][:, 0, 600], [0.9, 0.7])
    assert result["sample_id"].tolist() == [3, 9]
    assert list(result["event_status"]) == ["TP", "TP"]
