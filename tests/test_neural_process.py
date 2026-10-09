import torch

from eeg_uq.model import SEEDModel
from eeg_uq.neural_process import SeedNeuralProcess, binary_predictive_distribution
from eeg_uq.npf_compat import ensure_npf_importable


def test_npf_source_is_importable():
    ensure_npf_importable()
    from npf import AttnLNP  # noqa: F401


def test_binary_distribution_accepts_soft_targets_and_backpropagates():
    logits = torch.zeros(2, 3, 1, requires_grad=True)
    distribution = binary_predictive_distribution(logits, torch.zeros_like(logits))
    target = torch.tensor([[[0.25], [0.5], [0.75]], [[0.0], [1.0], [0.5]]])
    loss = -distribution.log_prob(target).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_seed_neural_process_shapes():
    torch.manual_seed(123)
    model = SeedNeuralProcess(
        SEEDModel(),
        kind="attn_lnp",
        context_fraction=0.25,
        r_dim=16,
        z_dim=16,
        latent_samples_train=1,
        latent_samples_test=3,
    )
    signal = torch.randn(2, 4000)
    target = torch.randint(0, 2, (2, 4000), dtype=torch.float32)
    target_reduced = model.reduce_targets(target, 500)
    outputs = model(signal, target)
    assert outputs[0].mean.shape == (1, 2, 500, 1)
    assert outputs[0].log_prob(target_reduced).shape == (1, 2, 500)
    samples, mean, seed_logits = model.predict(signal)
    assert samples.shape == (3, 2, 500)
    assert mean.shape == (3, 2, 500)
    assert seed_logits.shape == (2, 500)



def _small_np(seed_model):
    return SeedNeuralProcess(
        seed_model,
        kind="attn_lnp",
        context_fraction=0.25,
        r_dim=16,
        z_dim=16,
        latent_samples_train=1,
        latent_samples_test=3,
    )


def test_stitched_record_matches_legacy_seed_inference():
    import numpy as np

    from eeg_uq.evaluation import infer_full_signal_logits
    from eeg_uq.np_evaluation import predict_stitched_record

    torch.manual_seed(0)
    seed_model = SEEDModel().eval()
    model = _small_np(seed_model).eval()
    signal = np.random.default_rng(0).standard_normal(23000).astype(np.float32)
    samples, seed_logits = predict_stitched_record(model, signal, device=torch.device("cpu"))
    assert samples.shape == (3, 23000 // 8)
    expected = infer_full_signal_logits(seed_model, signal, torch.device("cpu"))
    assert np.allclose(np.repeat(seed_logits, 8), expected, atol=1e-5)


def test_validation_loss_does_not_use_labels():
    from eeg_uq.np_training import NeuralProcessConfig, _batch_loss, _criterion

    torch.manual_seed(0)
    model = _small_np(SEEDModel())
    config = NeuralProcessConfig(
        evaluation=0, fold=0, seed=0, batch_size=2, learning_rate=1e-3, weight_decay=0.0,
        epochs=1, num_workers=0, early_stopping_patience=1, device="cpu",
    )
    criterion = _criterion(config)
    batch = {"signal": torch.randn(2, 4000), "target": torch.randint(0, 2, (2, 4000)).float()}
    model.eval()
    criterion.eval()
    with torch.no_grad():
        loss, probabilities, _ = _batch_loss(model, criterion, batch, torch.device("cpu"), training=False)
        flipped = {"signal": batch["signal"], "target": 1.0 - batch["target"]}
        torch.manual_seed(1)
        _, probabilities_flipped, _ = _batch_loss(model, criterion, flipped, torch.device("cpu"), training=False)
        torch.manual_seed(1)
        _, probabilities_again, _ = _batch_loss(model, criterion, batch, torch.device("cpu"), training=False)
    assert torch.isfinite(loss)
    # Predictions must not depend on the labels during validation.
    assert torch.allclose(probabilities_flipped, probabilities_again)


def test_legacy_window_plan_covers_record_once():
    from eeg_uq.windowing import legacy_window_plan

    plan = legacy_window_plan(23000, resolution=8)
    assert len(plan.starts) == 11
    assert sum(last - first for _, first, last in plan.pieces) * 8 == 23000
