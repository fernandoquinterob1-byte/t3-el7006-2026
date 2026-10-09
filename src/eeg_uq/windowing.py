"""Legacy SEED window geometry: 20 s windows, 10 s stride, central stitching.

Mirrors :func:`eeg_uq.evaluation.infer_full_signal_logits` so any model run
per window (SEED features, Laplace, Neural Process) covers a whole record the
same way: every timestep comes from the central half of some window, except
the record's first and last stretch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import SAMPLE_RATE_HZ


@dataclass(frozen=True)
class WindowPlan:
    window_samples: int
    starts: tuple[int, ...]  # start sample of every window, tail window included
    # (window index, first, last) of each stitched piece, in output units
    pieces: tuple[tuple[int, int, int], ...]
    length: int


def legacy_window_plan(
    length: int,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: int = 20,
    stride_seconds: int = 10,
    resolution: int = 1,
) -> WindowPlan:
    """Windows and stitched pieces for a record of ``length`` samples.

    ``resolution`` is the model's downsampling factor; the pieces are then
    expressed in its output units (e.g. 8 for SEED's reduced rate).
    """
    window = sample_rate_hz * window_seconds
    stride = sample_rate_hz * stride_seconds
    central_start = sample_rate_hz * (window_seconds - stride_seconds) // 2
    central_end = central_start + stride
    segments = int(length / sample_rate_hz - window_seconds) // stride_seconds + 1
    if segments < 1:
        raise ValueError("Signal is shorter than the required 20-second inference window.")
    rest = (length / sample_rate_hz - window_seconds) % stride_seconds
    rest_samples = int(round(rest * sample_rate_hz))
    for value in (length, window, stride, central_start, rest_samples):
        if value % resolution:
            raise ValueError(f"Window geometry must be a multiple of {resolution} samples.")
    starts = [stride * index for index in range(segments)]
    pieces = []
    for index in range(segments):
        if index == 0:
            pieces.append((index, 0, central_end))
        elif index + 1 == segments:
            pieces.append((index, central_start, window))
        else:
            pieces.append((index, central_start, central_end))
    if rest_samples:
        starts.append(length - window)
        pieces.append((segments, window - rest_samples, window))
    pieces = [(index, first // resolution, last // resolution) for index, first, last in pieces]
    covered = sum(last - first for _, first, last in pieces) * resolution
    if covered != length:
        raise RuntimeError(f"Stitched length {covered} differs from signal length {length}.")
    return WindowPlan(window, tuple(starts), tuple(pieces), length)


def split_windows(signal: np.ndarray, plan: WindowPlan) -> np.ndarray:
    """``(windows, window_samples)`` array of the plan's windows."""
    signal = np.asarray(signal, dtype=np.float32)
    return np.stack([signal[start : start + plan.window_samples] for start in plan.starts])


def stitch_windows(values, plan: WindowPlan):
    """Concatenate the central pieces along axis 1 of ``(windows, time, ...)``.

    Works on NumPy arrays and torch tensors alike.
    """
    parts = [values[index, first:last] for index, first, last in plan.pieces]
    if isinstance(values, np.ndarray):
        return np.concatenate(parts, axis=0)
    import torch

    return torch.cat(parts, dim=0)
