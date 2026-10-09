import numpy as np
import pytest

from eeg_uq.uncertainty.conformal import (
    build_prediction_sets,
    compute_conformal_metrics,
    compute_nonconformity_scores,
    compute_rejection_curve,
    fit_conformal_threshold,
    summarize_event_conformal,
)


def test_binary_nonconformity_score_uses_probability_of_true_class():
    probability = np.asarray([0.9, 0.9])
    truth = np.asarray([1, 0])
    assert np.allclose(compute_nonconformity_scores(probability, truth), [0.1, 0.9])


def test_finite_sample_quantile_uses_exact_order_statistic():
    scores = np.arange(100, dtype=float) / 100.0
    # ceil(101 * .95) = 96, hence the 96th (one-based) score is .95.
    assert fit_conformal_threshold(scores, alpha=0.05) == pytest.approx(0.95)


def test_tiny_calibration_set_uses_bounded_equivalent_of_infinite_quantile():
    assert fit_conformal_threshold(np.asarray([0.2]), alpha=0.1) == 1.0


def test_prediction_sets_encode_zero_one_both_and_empty():
    only_zero = build_prediction_sets(np.asarray([0.1]), q_hat=0.2)
    only_one = build_prediction_sets(np.asarray([0.9]), q_hat=0.2)
    both = build_prediction_sets(np.asarray([0.5]), q_hat=0.6)
    empty = build_prediction_sets(np.asarray([0.5]), q_hat=0.4)
    assert (only_zero["contains_background"][0], only_zero["contains_event"][0], only_zero["set_size"][0]) == (1, 0, 1)
    assert (only_one["contains_background"][0], only_one["contains_event"][0], only_one["set_size"][0]) == (0, 1, 1)
    assert (both["contains_background"][0], both["contains_event"][0], both["set_size"][0]) == (1, 1, 2)
    assert (empty["contains_background"][0], empty["contains_event"][0], empty["set_size"][0]) == (0, 0, 0)


def test_metrics_compute_coverage_and_set_size_on_valid_timesteps_only():
    truth = np.asarray([0, 1, 1, 0])
    background = np.asarray([1, 0, 1, 0])
    event = np.asarray([0, 1, 1, 0])
    mask = np.asarray([1, 1, 0, 1], dtype=bool)
    metrics = compute_conformal_metrics(
        truth, background, event, mask, alpha=0.05
    )
    assert metrics["n_valid_timesteps"] == 3
    assert metrics["empirical_coverage"] == pytest.approx(2 / 3)
    assert metrics["mean_set_size"] == pytest.approx(2 / 3)
    assert metrics["singleton_rate"] == pytest.approx(2 / 3)
    assert metrics["empty_rate"] == pytest.approx(1 / 3)
    assert metrics["ambiguous_rate"] == 0.0
    assert metrics["target_coverage"] == 0.95


def test_mask_excludes_padding_from_calibration_and_prediction_sets():
    probability = np.asarray([0.9, 0.9, np.nan])
    truth = np.asarray([1, 0, 7])
    mask = np.asarray([1, 1, 0], dtype=bool)
    scores = compute_nonconformity_scores(probability, truth, mask)
    sets = build_prediction_sets(probability, q_hat=0.2, valid_mask=mask)
    assert np.allclose(scores, [0.1, 0.9])
    assert sets["valid_mask"].tolist() == [1, 1, 0]
    assert sets["set_size"].tolist() == [1, 1, 0]


def test_alpha_is_configurable_and_lower_alpha_cannot_lower_threshold():
    scores = np.linspace(0.0, 1.0, 101)
    conservative = fit_conformal_threshold(scores, alpha=0.01)
    permissive_error = fit_conformal_threshold(scores, alpha=0.20)
    assert conservative >= permissive_error
    with pytest.raises(ValueError, match="alpha"):
        fit_conformal_threshold(scores, alpha=0.0)


def test_event_summary_and_rejection_use_saved_legacy_event_spans():
    probability = np.asarray([[0.1, 0.5, 0.6, 0.9, 0.1, 0.1]])
    sets = build_prediction_sets(probability, q_hat=0.6)
    records = {
        "event_sample_id": np.asarray([10, 10]),
        "event_subject_id": np.asarray(["s", "s"]),
        "event_true_start": np.asarray([1, -1]),
        "event_true_end": np.asarray([3, -1]),
        "event_pred_start": np.asarray([1, 3]),
        "event_pred_end": np.asarray([3, 5]),
        "event_status": np.asarray(["TP", "FP"]),
        "event_iou": np.asarray([1.0, -1.0]),
    }
    summary = summarize_event_conformal(
        probability,
        sets["set_size"],
        sets["contains_event"],
        records,
        sample_ids=np.asarray([10]),
        subject_ids=np.asarray(["s"]),
    )
    assert summary["event_probability_seed"][0] == pytest.approx(0.575)
    assert summary["fraction_ambiguous_timesteps"][0] == 1.0
    assert summary["event_is_tp"].tolist() == [1, 0]
    assert summary["event_is_fp"].tolist() == [0, 1]
    rejection = compute_rejection_curve(summary, thresholds=[0.5, 1.0])
    assert rejection["acceptance_rate"][0] < rejection["acceptance_rate"][1]
