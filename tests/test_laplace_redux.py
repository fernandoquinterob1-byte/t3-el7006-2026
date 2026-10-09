import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eeg_uq.evaluation import infer_full_signal_logits  # noqa: E402
from eeg_uq.model import RESOL_FACTOR, SEEDModel  # noqa: E402
from eeg_uq.uncertainty.laplace_redux import (  # noqa: E402
    SeedLastLayerHead,
    compute_laplace_uncertainty,
    integrated_autocorrelation_time,
    seed_penultimate_features,
    stitched_penultimate_features,
)


def test_head_preserves_original_binary_probability_at_map():
    torch.manual_seed(0)
    seed_model = SEEDModel().eval()
    head = SeedLastLayerHead(seed_model.classifier.logits)
    signal = torch.randn(2, 800)
    with torch.no_grad():
        features = seed_penultimate_features(seed_model, signal)
        probability = torch.softmax(head(features), dim=-1)[..., 1]
        expected = torch.sigmoid(seed_model(signal))
    assert features.shape == (2, 800 // RESOL_FACTOR, head.last_layer.in_features)
    assert torch.allclose(probability.repeat_interleave(RESOL_FACTOR, dim=1), expected, atol=1e-6)
    assert all(parameter.requires_grad for parameter in head.last_layer.parameters())


def test_head_wraps_seed_last_layer():
    seed_model = SEEDModel().eval()
    head = SeedLastLayerHead(seed_model.classifier.logits)
    source = seed_model.classifier.logits
    assert head.last_layer.in_features == source.in_features
    assert torch.equal(head.last_layer.weight[1], source.weight[0])
    assert torch.equal(head.last_layer.bias[1], source.bias[0])
    assert not head.last_layer.weight[0].any() and head.last_layer.bias[0] == 0


def test_stitched_features_match_legacy_full_signal_inference():
    torch.manual_seed(0)
    seed_model = SEEDModel().eval()
    head = SeedLastLayerHead(seed_model.classifier.logits)
    signal = np.random.default_rng(0).standard_normal(23000).astype(np.float32)
    device = torch.device("cpu")
    with torch.no_grad():
        features = stitched_penultimate_features(seed_model, signal, device=device)
        logits = (head(features)[:, 1] - head(features)[:, 0]).repeat_interleave(RESOL_FACTOR)
    expected = infer_full_signal_logits(seed_model, signal, device)
    assert logits.shape == (23000,)
    assert np.allclose(logits.numpy(), expected, atol=1e-5)


def test_autocorrelation_time_counts_correlated_points():
    rng = np.random.default_rng(0)
    white = [rng.standard_normal(5000) for _ in range(4)]
    assert integrated_autocorrelation_time(white) < 1.2
    phi = 0.9
    smooth = []
    for _ in range(4):
        values = np.zeros(20000)
        for index in range(1, len(values)):
            values[index] = phi * values[index - 1] + rng.standard_normal()
        smooth.append(values)
    # Exact AR(1) value (1 + phi) / (1 - phi) = 19; the cutoff truncates the tail.
    assert 15 < integrated_autocorrelation_time(smooth) < 20


def test_laplace_metrics_report_entropy_mi_nll_and_ece():
    samples = np.asarray(
        [
            [[0.90, 0.10], [0.80, 0.20]],
            [[0.80, 0.20], [0.70, 0.30]],
            [[0.85, 0.15], [0.75, 0.25]],
        ]
    )
    truth = np.asarray([[1, 0], [1, 0]], dtype=np.int8)
    mask = np.ones_like(truth, dtype=bool)
    metrics, arrays = compute_laplace_uncertainty(samples, truth, mask)
    assert metrics["posterior_samples"] == 3
    assert metrics["n_valid_timesteps"] == 4
    assert metrics["nll"] >= 0.0
    assert metrics["ece"] >= 0.0
    assert np.all(arrays["mutual_information"] >= 0.0)


def test_summarize_records_detects_events_from_mean_probability():
    from eeg_uq.uncertainty.laplace_redux import summarize_records

    truth = np.zeros(2000, dtype=np.int8)
    truth[500:800] = 1
    probabilities = np.full((3, 2000), 0.05)
    probabilities[:, 500:800] = [[0.9], [0.8], [0.7]]
    result = summarize_records(
        [(7, "s1", truth, np.ones(2000, dtype=np.uint8), probabilities)], mask_source="test"
    )
    assert result["predictive_probabilities"].shape == (3, 1, 2000)
    assert result["y_pred"][0, 500:800].all() and not result["y_pred"][0, :500].any()
    assert list(result["event_status"]) == ["TP"]
    assert result["event_sample_id"].tolist() == [7]
