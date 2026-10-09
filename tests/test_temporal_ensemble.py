import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_uq.evaluation import infer_full_signal_logits  # noqa: E402
from eeg_uq.model import SEEDModel  # noqa: E402
from eeg_uq.uncertainty.laplace_redux import compute_laplace_uncertainty  # noqa: E402
from eeg_uq.uncertainty.temporal_ensemble import phase_shifted_probabilities  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def seed_and_signal():
    torch.manual_seed(0)
    signal = np.random.default_rng(0).standard_normal(23000).astype(np.float32)
    return SEEDModel().eval(), signal


def test_unshifted_member_is_plain_seed(seed_and_signal):
    model, signal = seed_and_signal
    members = phase_shifted_probabilities(model, signal, device=CPU, shifts=[0])
    expected = 1.0 / (1.0 + np.exp(-infer_full_signal_logits(model, signal, CPU)))
    assert members.shape == (1, 23000)
    assert np.allclose(members[0], expected, atol=1e-6)


def test_shifted_member_is_aligned_to_absolute_time(seed_and_signal):
    model, signal = seed_and_signal
    members = phase_shifted_probabilities(model, signal, device=CPU, shifts=[0, 3])
    # The unshifted member is plain SEED (previous test); applied to the
    # shifted signal it must reproduce member 3 placed from sample 3 onwards.
    shifted = phase_shifted_probabilities(model, signal[3:], device=CPU, shifts=[0])[0]
    assert np.allclose(members[1, 3:], shifted, atol=1e-6)
    assert not np.allclose(members[1], members[0], atol=1e-6)
    assert np.array_equal(members[1, :3], members[0, :3])


def test_identical_members_have_zero_mutual_information():
    member = np.random.default_rng(1).uniform(0.05, 0.95, size=(1, 2, 50))
    samples = np.repeat(member, 4, axis=0)
    truth = (member[0] > 0.5).astype(np.int8)
    metrics, arrays = compute_laplace_uncertainty(samples, truth, np.ones_like(truth, dtype=bool))
    assert metrics["mean_mutual_information"] < 1e-12
    assert np.allclose(arrays["mutual_information"], 0.0)


def test_shifts_must_include_plain_seed(seed_and_signal):
    model, signal = seed_and_signal
    with pytest.raises(ValueError):
        phase_shifted_probabilities(model, signal, device=CPU, shifts=[1, 2])

