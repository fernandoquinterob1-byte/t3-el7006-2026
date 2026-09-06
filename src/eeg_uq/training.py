"""One-fold training of the legacy PyTorch SEED implementation.

This module deliberately keeps the legacy architecture, BCEWithLogits loss,
AdamW optimizer, gradient clipping and above/below-median sampling.  It does
not claim to reproduce every preprocessing operation described in the paper:
the supplied HDF5 artifacts are the preprocessing boundary for this project.
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import H5TemporalDataset, LegacyBalancedBatchSampler
from .model import SEEDModel


@dataclass
class TrainConfig:
    evaluation: int  # zero-based external index
    fold: int  # zero-based external index
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    num_workers: int = 0
    early_stopping_patience: int = 6
    device: str = "auto"
    debug: bool = False
    overfit_small_subset: int | None = None
    logger: str = "none"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metrics_from_logits(logits: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probability = 1 / (1 + np.exp(-logits))
    prediction = (probability >= 0.5).astype(np.int8)
    truth = target.astype(np.int8)
    tp = int(np.logical_and(prediction == 1, truth == 1).sum())
    fp = int(np.logical_and(prediction == 1, truth == 0).sum())
    fn = int(np.logical_and(prediction == 0, truth == 1).sum())
    accuracy = float((prediction == truth).mean())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"temporal_accuracy": accuracy, "temporal_precision": precision, "temporal_recall": recall, "temporal_f1": f1}
    if truth.min() != truth.max():
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            metrics["temporal_auroc"] = float(roc_auc_score(truth.reshape(-1), probability.reshape(-1)))
            metrics["temporal_auprc"] = float(average_precision_score(truth.reshape(-1), probability.reshape(-1)))
        except ImportError:
            pass
    return metrics


def _run_epoch(model: SEEDModel, loader: DataLoader, optimizer, device: torch.device) -> tuple[float, dict[str, float]]:
    is_training = optimizer is not None
    model.train(is_training)
    losses, logits_parts, target_parts = [], [], []
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            signal, target = batch["signal"].to(device), batch["target"].to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(signal)
            loss = model.criterion(logits.flatten(), target.flatten())
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            logits_parts.append(logits.detach().cpu().numpy())
            target_parts.append(target.detach().cpu().numpy())
    return float(np.mean(losses)), _metrics_from_logits(np.concatenate(logits_parts), np.concatenate(target_parts))


@torch.no_grad()
def export_window_predictions(model: SEEDModel, loader: DataLoader, device: torch.device, output_path: Path, metadata: dict) -> None:
    """Export dense binary temporal predictions for the validation split."""
    model.eval()
    ids, subjects, targets, logits_values = [], [], [], []
    for batch in loader:
        logits = model(batch["signal"].to(device)).cpu().numpy().astype(np.float32)
        ids.append(batch["sample_id"].numpy())
        subjects.append(np.asarray(batch["subject_id"], dtype="U"))
        targets.append(batch["target"].numpy().astype(np.int8))
        logits_values.append(logits)
    logits = np.concatenate(logits_values)
    probability = 1 / (1 + np.exp(-logits))
    np.savez_compressed(
        output_path,
        sample_id=np.concatenate(ids),
        subject_id=np.concatenate(subjects),
        y_true=np.concatenate(targets),
        logits=logits,
        probabilities=probability.astype(np.float32),
        y_pred=(probability >= 0.5).astype(np.int8),
        **{key: np.asarray(value) for key, value in metadata.items()},
    )


def train_one_fold(
    config: TrainConfig,
    *,
    augmented_h5: Path,
    train_subjects: list[str],
    validation_subjects: list[str],
    output_dir: Path,
) -> dict:
    """Train one new model; fold IDs are external zero-based CLI indices."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    subset = config.overfit_small_subset
    train_data = H5TemporalDataset(augmented_h5, train_subjects, limit=subset)
    # Overfit mode is a diagnostic, not a validation estimate: deliberately
    # measure on the same tiny training subset so failure to memorize is visible.
    evaluation_data_subjects = train_subjects if subset is not None else validation_subjects
    val_data = H5TemporalDataset(augmented_h5, evaluation_data_subjects, limit=subset if subset else None)
    sampler = LegacyBalancedBatchSampler(train_data.positive_counts, config.batch_size, config.seed)
    train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=config.num_workers)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    model = SEEDModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, stale_epochs, history = float("inf"), 0, []
    run = None
    if config.logger == "wandb":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B requested but not installed. Install requirements.txt or use --logger none.") from error
        run = wandb.init(project="t3-el7006-2026", config=asdict(config), mode="online")
    for epoch in range(1, config.epochs + 1):
        sampler.set_epoch(epoch)
        train_loss, train_metrics = _run_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = _run_epoch(model, val_loader, None, device)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"], **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        if run:
            run.log(record, step=epoch)
        if val_loss < best_loss:
            best_loss, stale_epochs = val_loss, 0
            torch.save(model.embedding.state_dict(), checkpoints / "embedding_best.pth")
            torch.save(model.transformer.state_dict(), checkpoints / "lstm_best.pth")
            torch.save(model.classifier.state_dict(), checkpoints / "classifier_best.pth")
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "config": asdict(config)}, checkpoints / "best_full.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    torch.save(model.state_dict(), checkpoints / "final_full.pt")
    model.load_state_dict(torch.load(checkpoints / "best_full.pt", map_location=device, weights_only=True)["model_state"])
    metadata = {
        "evaluation": config.evaluation,
        "fold": config.fold,
        "seed": config.seed,
        "split": "overfit_train_subset" if subset is not None else "validation",
    }
    export_window_predictions(model, val_loader, device, output_dir / "predictions.npz", metadata)
    summary = {"config": asdict(config), "git_commit": git_commit(), "device": str(device), "best_val_loss": best_loss, "epochs_completed": len(history), "history": history, "train_records": len(train_data), "validation_records": len(val_data), "overfit_mode_uses_training_subset_for_evaluation": subset is not None}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps({**asdict(config), "git_commit": git_commit()}, indent=2))
    if run:
        run.finish()
    return summary
