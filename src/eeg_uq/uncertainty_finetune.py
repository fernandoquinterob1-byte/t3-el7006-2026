"""Fine-tune SEED with an uncertainty-weighted loss (variant F) and a plain control (variant A).

Idea: the two kinds of uncertainty ask for opposite things during training.

* **Epistemic (MI of the Deep Ensemble)**: the model has not learned this yet,
  so those timesteps get **more** weight.
* **Aleatoric (expected entropy of the members)**: the example is ambiguous by
  nature (e.g. weak spindles the experts disagree on), so insisting would fit
  label noise; those timesteps get **less** weight.

Per-timestep weight on the training windows::

    w = (1 + ALPHA * min(MI / MI_ref, 1)) * (1 - BETA * aleatoric)

``MI_ref`` is the 99th percentile of the MI on timesteps where the ensemble
probability is >= 0.05 (spindle-like zones), so ``ALPHA`` means "at most this
much extra weight".  Aleatoric entropy is in bits (0..1).  The weighted BCE is
normalised by the sum of weights, so the loss scale matches plain BCE.

Variant **A** is the same fine-tuning with every weight equal to 1: it tells
apart "uncertainty helped" from "a few more epochs at a low learning rate helped".

Everything else is the original training: same checkpoint start (seed 42),
AdamW, gradient clipping, legacy balanced batches, augmented 20 s windows.
The epoch is chosen on **validation** event F1 (epoch 0 = no fine-tuning is a
valid choice); test is exported once, for the chosen epoch only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import H5TemporalDataset, LegacyBalancedBatchSampler
from .folds import get_fold, load_folds
from .model import SEEDModel
from .trained_evaluation import evaluate_trained_split, load_trained_checkpoint, save_trained_split_predictions
from .training import set_seed

ALPHA = 2.0
BETA = 0.5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 3
BATCH_SIZE = 8
SEED = 42
MEMBER_SEEDS = (42, 1, 2, 3, 4)
REPO = Path(__file__).resolve().parents[2]
SEED_RUNS = REPO / "outputs" / "seed_batch8_earlystop"
OUTPUT_ROOT = REPO / "outputs" / "uncertainty_finetune"
LN2 = float(np.log(2.0))


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / LN2


def checkpoint_of(evaluation: int, fold: int, seed: int) -> Path:
    return SEED_RUNS / f"train_eval_{evaluation}_fold_{fold}_seed_{seed}" / "checkpoints" / "best_full.pt"


def subjects_of(folds_path: Path, augmented_h5: Path, evaluation: int, fold: int) -> tuple[list[str], list[str]]:
    folds = load_folds(folds_path)
    validation, test = get_fold(folds, evaluation + 1, fold + 1)
    with h5py.File(augmented_h5, "r") as source:
        subjects = {s.decode("utf-8") for s in source["subject_ids"][:]}
    return sorted(subjects - set(validation) - set(test)), validation


@torch.no_grad()
def ensemble_uncertainty(evaluation: int, fold: int, augmented_h5: Path, indices: np.ndarray,
                         device: torch.device, cache: Path, batch: int = 32) -> dict[str, np.ndarray]:
    """MI and aleatoric entropy (bits) of the 5 Deep Ensemble members on the training windows."""
    if cache.is_file():
        with np.load(cache) as data:
            if np.array_equal(data["indices"], indices):
                return {k: data[k] for k in data.files}
    with h5py.File(augmented_h5, "r") as source:
        signals = np.asarray(source["signals"][np.sort(indices)], dtype=np.float32)
    order = np.argsort(np.argsort(indices))  # back to the dataset order
    signals = signals[order]
    sum_p = np.zeros(signals.shape, dtype=np.float32)
    sum_h = np.zeros(signals.shape, dtype=np.float32)
    for seed in MEMBER_SEEDS:
        model, _ = load_trained_checkpoint(checkpoint_of(evaluation, fold, seed), evaluation=evaluation, fold=fold,
                                           device=device)
        for start in range(0, len(signals), batch):
            x = torch.from_numpy(signals[start:start + batch]).to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            sum_p[start:start + batch] += p
            sum_h[start:start + batch] += _entropy_bits(p)
        del model
        torch.cuda.empty_cache()
    mean_p = sum_p / len(MEMBER_SEEDS)
    aleatoric = sum_h / len(MEMBER_SEEDS)
    mi = np.maximum(_entropy_bits(mean_p) - aleatoric, 0.0)
    out = {"indices": indices, "mean_p": mean_p.astype(np.float16), "mi": mi.astype(np.float16),
           "aleatoric": aleatoric.astype(np.float16)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def uncertainty_weights(unc: dict[str, np.ndarray], alpha: float = ALPHA, beta: float = BETA) -> tuple[np.ndarray, dict]:
    mi = unc["mi"].astype(np.float32)
    aleatoric = unc["aleatoric"].astype(np.float32)
    active = unc["mean_p"].astype(np.float32) >= 0.05
    mi_ref = float(np.quantile(mi[active], 0.99)) if active.any() else 1.0
    weights = (1 + alpha * np.minimum(mi / mi_ref, 1.0)) * (1 - beta * np.clip(aleatoric, 0, 1))
    info = {"alpha": alpha, "beta": beta, "mi_ref": mi_ref,
            "weight_mean": float(weights.mean()), "weight_min": float(weights.min()), "weight_max": float(weights.max()),
            "weight_mean_active": float(weights[active].mean()) if active.any() else None,
            "fraction_active": float(active.mean())}
    return weights.astype(np.float32), info


class WeightedWindows(Dataset):
    """``H5TemporalDataset`` items plus a per-timestep loss weight."""

    def __init__(self, base: H5TemporalDataset, weights: np.ndarray | None):
        self.base, self.weights = base, weights
        self.positive_counts = base.positive_counts

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        w = np.ones(item["target"].shape, np.float32) if self.weights is None else self.weights[index]
        item["weight"] = torch.from_numpy(np.ascontiguousarray(w))
        return item


def _train_epoch(model: SEEDModel, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        signal, target, weight = batch["signal"].to(device), batch["target"].to(device), batch["weight"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signal)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (weight * bce).sum() / weight.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def _validation_loss(model: SEEDModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        logits = model(batch["signal"].to(device))
        losses.append(float(model.criterion(logits.flatten(), batch["target"].to(device).flatten()).cpu()))
    return float(np.mean(losses))


def _event_scores(npz_path: Path) -> dict:
    from .experiments_summary import _scores_from_ious

    with np.load(npz_path, allow_pickle=False) as data:
        return _scores_from_ious(data["event_iou_values"])


def _save_checkpoint(model: SEEDModel, path: Path, evaluation: int, fold: int, epoch: int, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "epoch": epoch,
                "config": {"evaluation": evaluation, "fold": fold, "seed": SEED, "variant": variant}}, path)


def run_variant(variant: str, *, evaluation: int, fold: int, raw_h5: Path, augmented_h5: Path, folds_path: Path,
                weights: np.ndarray | None, train_data: H5TemporalDataset, val_loader: DataLoader,
                device: torch.device, log=print, train_seed: int = SEED) -> dict:
    """``train_seed`` changes batch order and dropout only; the start checkpoint is always seed 42."""
    if train_seed != SEED:
        variant = f"{variant}_s{train_seed}"
    out_dir = OUTPUT_ROOT / f"eval_{evaluation}_fold_{fold}" / variant
    set_seed(train_seed)
    model, _ = load_trained_checkpoint(checkpoint_of(evaluation, fold, SEED), evaluation=evaluation, fold=fold,
                                       device=device)
    dataset = WeightedWindows(train_data, weights)
    sampler = LegacyBalancedBatchSampler(dataset.positive_counts, BATCH_SIZE, train_seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []

    def evaluate(epoch: int) -> dict:
        checkpoint = out_dir / "checkpoints" / f"epoch_{epoch}.pt"
        _save_checkpoint(model, checkpoint, evaluation, fold, epoch, variant)
        predictions = evaluate_trained_split(data_path=raw_h5, folds_path=folds_path, checkpoint_path=checkpoint,
                                             run_dir=out_dir, evaluation=evaluation, fold=fold, split="validation",
                                             device=str(device))
        npz, _ = save_trained_split_predictions(predictions, out_dir / f"validation_epoch_{epoch}.npz")
        record = {"epoch": epoch, "val_loss": _validation_loss(model, val_loader, device),
                  **{f"val {k}": v for k, v in _event_scores(npz).items()}}
        log(f"[{variant}] epoch {epoch}: val loss {record['val_loss']:.4f}, val F1 {record['val F1 (%)']:.2f}")
        return record

    started = time.time()
    history.append(evaluate(0))
    for epoch in range(1, EPOCHS + 1):
        sampler.set_epoch(epoch)
        train_loss = _train_epoch(model, loader, optimizer, device)
        history.append({**evaluate(epoch), "train_loss": train_loss})
    best = max(history, key=lambda r: r["val F1 (%)"])
    checkpoint = out_dir / "checkpoints" / f"epoch_{best['epoch']}.pt"
    predictions = evaluate_trained_split(data_path=raw_h5, folds_path=folds_path, checkpoint_path=checkpoint,
                                         run_dir=out_dir, evaluation=evaluation, fold=fold, split="test",
                                         device=str(device))
    test_npz, _ = save_trained_split_predictions(predictions, out_dir / "test_predictions.npz")
    result = {"variant": variant, "train_seed": train_seed, "chosen_epoch": best["epoch"], "history": history,
              "test": _event_scores(test_npz), "minutes": (time.time() - started) / 60}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"[{variant}] chosen epoch {best['epoch']} -> test F1 {result['test']['F1 (%)']:.2f} "
        f"({result['minutes']:.1f} min)")
    return result


def member_noise(evaluation: int, fold: int) -> list[dict]:
    """Test scores of the 5 independently trained SEED (Deep Ensemble members): the retraining noise band."""
    rows = []
    for seed in MEMBER_SEEDS:
        path = SEED_RUNS / f"train_eval_{evaluation}_fold_{fold}_seed_{seed}" / "test_predictions.npz"
        rows.append({"seed": seed, **_event_scores(path)})
    return rows


def control_seeds(evaluation: int = 0, fold: int = 0, seeds=(1, 2, 3)) -> None:
    """Repeat the plain control A with other training seeds: how noisy is a 3-epoch fine-tune?"""
    main(evaluation, fold, variants=("A",), train_seeds=seeds, summary_name="summary_control_seeds.json")


def main(evaluation: int = 0, fold: int = 0, variants=("F", "A"), train_seeds=(SEED,),
         summary_name: str = "summary.json") -> None:
    import os

    root = Path(os.environ.get("EEG_DATA_ROOT", Path.home() / "Downloads" / "data" / "data"))
    raw_h5, augmented_h5 = root / "moda_moda.h5", root / "moda_augmented_s5.h5"
    folds_path = REPO / "eeg_code" / "data" / "moda_folds.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = OUTPUT_ROOT / f"eval_{evaluation}_fold_{fold}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = (out_root / "log.txt").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    train_subjects, validation_subjects = subjects_of(folds_path, augmented_h5, evaluation, fold)
    train_data = H5TemporalDataset(augmented_h5, train_subjects)
    val_data = H5TemporalDataset(augmented_h5, validation_subjects)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    log(f"train windows {len(train_data)}, validation windows {len(val_data)}, device {device}")

    results = {"noise": member_noise(evaluation, fold)}
    if "F" in variants:
        log("computing Deep Ensemble uncertainty on the training windows...")
        unc = ensemble_uncertainty(evaluation, fold, augmented_h5, train_data.valid_indices, device,
                                   out_root / "train_uncertainty.npz")
        weights, info = uncertainty_weights(unc)
        log(f"weights: {json.dumps(info)}")
        results["weights"] = info
        results["F"] = run_variant("F", evaluation=evaluation, fold=fold, raw_h5=raw_h5, augmented_h5=augmented_h5,
                                   folds_path=folds_path, weights=weights, train_data=train_data,
                                   val_loader=val_loader, device=device, log=log)
        del unc, weights
    if "A" in variants:
        for train_seed in train_seeds:
            key = "A" if train_seed == SEED else f"A_s{train_seed}"
            results[key] = run_variant("A", evaluation=evaluation, fold=fold, raw_h5=raw_h5,
                                       augmented_h5=augmented_h5, folds_path=folds_path, weights=None,
                                       train_data=train_data, val_loader=val_loader, device=device, log=log,
                                       train_seed=train_seed)
    (out_root / summary_name).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log("done")


if __name__ == "__main__":
    import sys

    if "--control-seeds" in sys.argv:
        control_seeds()
    else:
        main()
