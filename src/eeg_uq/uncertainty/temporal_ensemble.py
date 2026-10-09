"""Phase-shift Temporal Ensemble around the frozen SEED checkpoint.

Each member runs SEED with the legacy 20 s / 10 s central stitching on the
record shifted by ``d`` samples.  Shifts that are not multiples of SEED's x8
downsampling change how samples are pooled inside the network, which is the
only window displacement SEED is sensitive to: inside the central half of a
window, moving the timestep by whole seconds barely changes its prediction.
Member outputs are aligned back to absolute time, so their spread per timestep
measures sensitivity to the temporal cut, not uncertainty over weights.

No training is performed; the source checkpoint is never modified.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ..metrics import SAMPLE_RATE_HZ
from ..model import SEEDModel
from ..windowing import legacy_window_plan, split_windows, stitch_windows

DEFAULT_PHASE_SHIFTS = tuple(range(8))


@torch.no_grad()
def phase_shifted_probabilities(
    model: SEEDModel,
    signal: np.ndarray,
    *,
    device: torch.device,
    shifts: Sequence[int] = DEFAULT_PHASE_SHIFTS,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: int = 20,
    stride_seconds: int = 10,
) -> np.ndarray:
    """Member probabilities ``(len(shifts), len(signal))`` in absolute time.

    Member ``d`` stitches SEED over ``signal[d:]``; its first ``d`` samples
    (at most a few milliseconds at the record start) are taken from the
    unshifted member so every member covers the whole record.
    """
    signal = np.asarray(signal, dtype=np.float32)
    shifts = [int(shift) for shift in shifts]
    if not shifts or min(shifts) < 0 or 0 not in shifts:
        raise ValueError("Shifts must be non-negative and include 0 (the plain SEED member).")
    geometry = dict(sample_rate_hz=sample_rate_hz, window_seconds=window_seconds, stride_seconds=stride_seconds)
    plans = [legacy_window_plan(len(signal) - shift, **geometry) for shift in shifts]
    windows = np.concatenate([split_windows(signal[shift:], plan) for shift, plan in zip(shifts, plans)])
    probabilities = torch.sigmoid(model(torch.from_numpy(windows).to(device))).cpu().numpy()

    members = np.empty((len(shifts), len(signal)), dtype=np.float32)
    offset = 0
    for index, (shift, plan) in enumerate(zip(shifts, plans)):
        member_windows = probabilities[offset : offset + len(plan.starts)]
        offset += len(plan.starts)
        members[index, shift:] = stitch_windows(member_windows, plan)
    reference = members[shifts.index(0)]
    for index, shift in enumerate(shifts):
        members[index, :shift] = reference[:shift]
    return members
