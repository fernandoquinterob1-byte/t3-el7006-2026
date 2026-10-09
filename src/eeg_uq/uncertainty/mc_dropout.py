"""MC Dropout on the frozen SEED checkpoint.

SEED's dropout layers stay active at prediction time, so every forward pass
samples a different thinned network; the spread of the passes approximates
epistemic uncertainty over the whole model, including the BiLSTM.  BatchNorm
keeps its running statistics (eval mode).  Each pass uses the legacy 20 s /
10 s central stitching.  No training is performed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from ..metrics import SAMPLE_RATE_HZ
from ..model import SEEDModel
from ..windowing import legacy_window_plan, split_windows, stitch_windows

# SEED dropout layers: input to the BiLSTM (p=0.2), between its two layers
# (p=0.5, inside nn.LSTM) and before the classifier (p=0.5).  The embedding
# dropout has p=0 and is ignored.
DROPOUT_GROUPS = {
    "bilstm_input": lambda model: [model.transformer.dropout],
    "bilstm_internal": lambda model: [model.transformer.lstm],
    "classifier": lambda model: [model.classifier.dropout],
}


@contextmanager
def active_dropout(model: SEEDModel, groups: Sequence[str]) -> Iterator[SEEDModel]:
    """Eval mode everywhere except the selected dropout layers."""
    unknown = set(groups) - set(DROPOUT_GROUPS)
    if unknown or not groups:
        raise ValueError(f"Choose dropout groups from {sorted(DROPOUT_GROUPS)}, got {list(groups)}.")
    was_training = model.training
    model.eval()
    modules: list[nn.Module] = [module for group in groups for module in DROPOUT_GROUPS[group](model)]
    for module in modules:
        module.train(True)
    try:
        yield model
    finally:
        model.train(was_training)


@torch.no_grad()
def mc_dropout_probabilities(
    model: SEEDModel,
    signal: np.ndarray,
    *,
    device: torch.device,
    n_samples: int = 30,
    groups: Sequence[str] = tuple(DROPOUT_GROUPS),
    chunk_samples: int = 10,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    window_seconds: int = 20,
    stride_seconds: int = 10,
) -> np.ndarray:
    """Positive-class probabilities ``(n_samples, len(signal))``, one per pass."""
    signal = np.asarray(signal, dtype=np.float32)
    plan = legacy_window_plan(
        len(signal), sample_rate_hz=sample_rate_hz, window_seconds=window_seconds, stride_seconds=stride_seconds
    )
    windows = torch.from_numpy(split_windows(signal, plan)).to(device)
    passes = []
    with active_dropout(model, groups):
        for start in range(0, n_samples, chunk_samples):
            count = min(chunk_samples, n_samples - start)
            # Every repeated window gets its own dropout masks: one pass per copy.
            batch = windows.repeat(count, 1)
            probabilities = torch.sigmoid(model(batch)).reshape(count, len(plan.starts), -1)
            passes.append(stitch_windows(probabilities.permute(1, 2, 0), plan).T.cpu().numpy())
    return np.concatenate(passes).astype(np.float32)
