import pytest

from eeg_uq.paths import resolve_data_paths, validate_data_paths


def test_paths_require_explicit_root(monkeypatch):
    monkeypatch.delenv("EEG_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="Data root"):
        resolve_data_paths()


def test_path_validation_mentions_expected_layout(tmp_path):
    paths = resolve_data_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="Expected layout"):
        validate_data_paths(paths, require_augmented=True)
