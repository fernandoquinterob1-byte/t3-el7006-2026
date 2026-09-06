import pytest

from eeg_uq.folds import get_fold, validate_folds


def test_fold_validation_detects_subject_leakage():
    folds = {"eval_1": {"fold_1": {"val": ["a"], "test": ["a"]}}}
    with pytest.raises(ValueError, match="leakage"):
        get_fold(folds, 1, 1)


def test_fold_validation_accepts_complete_subject_partition():
    folds = {"eval_1": {"fold_1": {"val": ["a"], "test": ["b"]}}}
    report = validate_folds(folds, {"a", "b"})
    assert report["subjects_in_folds"] == 2
