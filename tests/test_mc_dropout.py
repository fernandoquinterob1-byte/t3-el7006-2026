import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_uq.evaluation import infer_full_signal_logits  # noqa: E402
from eeg_uq.model import SEEDModel  # noqa: E402
from eeg_uq.uncertainty.mc_dropout import active_dropout, mc_dropout_probabilities  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def seed_and_signal():
    torch.manual_seed(0)
    signal = np.random.default_rng(0).standard_normal(23000).astype(np.float32)
    return SEEDModel().eval(), signal


def test_passes_differ_and_cover_record(seed_and_signal):
    model, signal = seed_and_signal
    torch.manual_seed(1)
    passes = mc_dropout_probabilities(model, signal, device=CPU, n_samples=4, chunk_samples=3)
    assert passes.shape == (4, 23000)
    assert not np.allclose(passes[0], passes[1])
    assert not model.training


def test_mean_of_many_passes_is_close_to_seed(seed_and_signal):
    model, signal = seed_and_signal
    torch.manual_seed(2)
    passes = mc_dropout_probabilities(model, signal, device=CPU, n_samples=40, groups=["classifier"])
    seed = 1.0 / (1.0 + np.exp(-infer_full_signal_logits(model, signal, CPU)))
    assert np.abs(passes.mean(axis=0) - seed).mean() < 0.05


def test_only_selected_dropout_is_active(seed_and_signal):
    model, _ = seed_and_signal
    with active_dropout(model, ["classifier"]):
        assert model.classifier.dropout.training
        assert not model.transformer.dropout.training
        assert not model.transformer.lstm.training
        assert not model.embedding.conv_layer.bn.training
    assert not model.training
    with pytest.raises(ValueError):
        with active_dropout(model, ["unknown"]):
            pass
