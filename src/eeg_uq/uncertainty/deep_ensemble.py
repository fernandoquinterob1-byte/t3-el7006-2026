"""Deep Ensemble of independently trained SEED models.

Each member is a full SEED trained by ``train.py`` with a different seed
(initialisation and batch order).  ``train.py`` already exports raw validation
and test probabilities with the legacy stitching, so the ensemble only stacks
them: the spread across members is epistemic uncertainty over the whole
network, including the BiLSTM.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from ..metrics import DETECTION_THRESHOLD, DURATION_THRESHOLD
from .laplace_redux import summarize_records


def _member_records(
    member_files: Sequence[Path],
) -> tuple[Iterator[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]], str]:
    members = [np.load(path, allow_pickle=False) for path in member_files]
    reference = members[0]
    for path, member in zip(member_files[1:], members[1:]):
        if not np.array_equal(member["sample_id"], reference["sample_id"]):
            raise ValueError(f"{path} does not contain the same records as {member_files[0]}.")
        if not np.array_equal(member["y_true"], reference["y_true"]):
            raise ValueError(f"{path} has different labels than {member_files[0]}.")
    mask_source = str(reference["mask_source"]) if "mask_source" in reference.files else "valid_mask"

    def records():
        probabilities = [member["probabilities"] for member in members]
        for row, sample_id in enumerate(reference["sample_id"]):
            yield (
                int(sample_id),
                str(reference["subject_id"][row]),
                reference["y_true"][row].astype(np.int8),
                reference["valid_mask"][row].astype(np.uint8),
                np.stack([member[row] for member in probabilities]).astype(np.float32),
            )

    return records(), mask_source


def combine_member_predictions(
    member_files: Sequence[str | Path],
    *,
    detection_threshold: float = DETECTION_THRESHOLD,
    duration_threshold: float = DURATION_THRESHOLD,
) -> dict[str, np.ndarray | str]:
    """Ensemble predictions from members' exported split NPZ files."""
    member_files = [Path(path) for path in member_files]
    if len(member_files) < 2:
        raise ValueError("A deep ensemble needs at least two members.")
    records, mask_source = _member_records(member_files)
    return summarize_records(
        records,
        mask_source=mask_source,
        detection_threshold=detection_threshold,
        duration_threshold=duration_threshold,
    )
