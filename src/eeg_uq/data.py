"""HDF5 datasets and the legacy above/below-median batch balancing rule."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])


class H5TemporalDataset(Dataset):
    """One HDF5 row per item; subject filtering is performed before data loading."""

    def __init__(self, h5_path: str | Path, subject_ids: list[str], *, limit: int | None = None):
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as source:
            available = _decode(source["subject_ids"][:])
            self.valid_indices = np.flatnonzero(np.isin(available, subject_ids)).astype(np.int64)
            self.subject_ids = available[self.valid_indices]
            self.signal_shape = tuple(source["signals"].shape[1:])
            self.label_shape = tuple(source["labels"].shape[1:])
            if limit is not None:
                self.valid_indices = self.valid_indices[:limit]
                self.subject_ids = self.subject_ids[:limit]
            self.positive_counts = np.asarray(
                [np.asarray(source["labels"][index]).sum() for index in self.valid_indices], dtype=np.int64
            )
        if not len(self.valid_indices):
            raise ValueError("Subject split did not select any HDF5 records.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> dict:
        sample_id = int(self.valid_indices[index])
        with h5py.File(self.h5_path, "r") as source:
            signal = np.asarray(source["signals"][sample_id], dtype=np.float32)
            target = np.asarray(source["labels"][sample_id], dtype=np.float32)
        return {
            "sample_id": sample_id,
            "subject_id": self.subject_ids[index],
            "signal": torch.from_numpy(signal),
            "target": torch.from_numpy(target),
        }


class LegacyBalancedBatchSampler(Sampler[list[int]]):
    """Faithful batching intent from the notebook: alternate low/high event activity rows."""

    def __init__(self, positive_counts: np.ndarray, batch_size: int, seed: int = 42):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Legacy balanced batching requires an even batch size of at least 2.")
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        median = np.median(positive_counts)
        self.low = np.flatnonzero(positive_counts <= median)
        self.high = np.flatnonzero(positive_counts > median)
        if not len(self.low) or not len(self.high):
            raise ValueError("Cannot create legacy balanced batches: one median activity group is empty.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        count = max(len(self.low), len(self.high))
        low = rng.choice(self.low, size=count, replace=len(self.low) < count)
        high = rng.choice(self.high, size=count, replace=len(self.high) < count)
        interleaved = np.column_stack((low, high)).ravel()
        for start in range(0, len(interleaved), self.batch_size):
            yield interleaved[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return int(np.ceil(2 * max(len(self.low), len(self.high)) / self.batch_size))
