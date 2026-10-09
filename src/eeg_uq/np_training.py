"""Training loop for the SEED-backed Neural Process model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import H5TemporalDataset, LegacyBalancedBatchSampler
from .neural_process import NPModelKind, SeedNeuralProcess, load_seed_checkpoint
from .trained_evaluation import checkpoint_sha256
from .training import git_commit, resolve_device, set_seed


@dataclass
class NeuralProcessConfig:
    evaluation: int
    fold: int
    seed: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    num_workers: int
    early_stopping_patience: int
    device: str
    model_kind: NPModelKind = "attn_lnp"
    context_fraction: float = 0.25
    r_dim: int = 64
    z_dim: int = 64
    latent_samples_train: int = 1
    latent_samples_test: int = 16
    freeze_seed_backbone: bool = True
    debug: bool = False
    overfit_small_subset: int | None = None


def _criterion(config: NeuralProcessConfig):
    from npf import CNPFLoss, ELBOLossLNPF

    return ELBOLossLNPF(reduction="mean") if config.model_kind == "attn_lnp" else CNPFLoss(reduction="mean")


def _batch_loss(
    model: SeedNeuralProcess,
    criterion,
    batch: dict,
    device: torch.device,
    *,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signal = batch["signal"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    target_reduced = model.reduce_targets(target, model.reduced_time)
    # Training samples z from q(z | context, targets) for the ELBO.  Validation
    # predicts like test: no labels, z from q(z | context), and the criterion in
    # eval mode scores the predictive likelihood instead of the ELBO.
    outputs = model(signal, target if training else None, randomise_context=training)
    loss = criterion(outputs, target_reduced)
    probabilities = outputs[0].mean.mean(0).squeeze(-1).detach()
    return loss, probabilities, target_reduced.squeeze(-1).detach()


def _metrics(probabilities: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    probability = probabilities.detach().cpu().numpy().reshape(-1)
    truth = target.detach().cpu().numpy().reshape(-1)
    prediction = (probability >= 0.5).astype(np.int8)
    truth_binary = (truth >= 0.5).astype(np.int8)
    tp = int(np.logical_and(prediction == 1, truth_binary == 1).sum())
    fp = int(np.logical_and(prediction == 1, truth_binary == 0).sum())
    fn = int(np.logical_and(prediction == 0, truth_binary == 1).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = float(np.mean((probability - truth) ** 2))
    return {
        "temporal_precision": precision,
        "temporal_recall": recall,
        "temporal_f1": f1,
        "brier": brier,
    }


def _run_epoch(
    model: SeedNeuralProcess,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    *,
    training: bool,
) -> tuple[float, dict[str, float]]:
    model.train(training)
    criterion.train(training)
    losses: list[float] = []
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, batch_probabilities, batch_targets = _batch_loss(
                model, criterion, batch, device, training=training
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.npf.parameters(), max_norm=1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            probabilities.append(batch_probabilities)
            targets.append(batch_targets)
    all_probabilities = torch.cat(probabilities, dim=0)
    all_targets = torch.cat(targets, dim=0)
    return float(np.mean(losses)), _metrics(all_probabilities, all_targets)


def train_neural_process_one_fold(
    config: NeuralProcessConfig,
    *,
    augmented_h5: str | Path,
    train_subjects: list[str],
    validation_subjects: list[str],
    seed_checkpoint: str | Path,
    output_dir: str | Path,
) -> dict:
    """Train one NP fold while preserving the source SEED checkpoint."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    subset = config.overfit_small_subset
    train_data = H5TemporalDataset(augmented_h5, train_subjects, limit=subset)
    val_data = H5TemporalDataset(augmented_h5, validation_subjects, limit=subset if subset else None)
    sampler = LegacyBalancedBatchSampler(train_data.positive_counts, config.batch_size, config.seed)
    train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=config.num_workers)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    seed_model, seed_payload = load_seed_checkpoint(
        seed_checkpoint, device=device, evaluation=config.evaluation, fold=config.fold
    )
    model = SeedNeuralProcess(
        seed_model,
        kind=config.model_kind,
        context_fraction=config.context_fraction,
        r_dim=config.r_dim,
        z_dim=config.z_dim,
        latent_samples_train=config.latent_samples_train,
        latent_samples_test=config.latent_samples_test,
        freeze_seed_backbone=config.freeze_seed_backbone,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = _criterion(config)
    source_sha256 = checkpoint_sha256(seed_checkpoint)
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    history: list[dict] = []
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        sampler.set_epoch(epoch)
        train_loss, train_metrics = _run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, criterion, None, device, training=False
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        if val_loss < best_loss:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            torch.save(
                {
                    "np_state": model.npf.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": asdict(config),
                    "source_seed_checkpoint": str(Path(seed_checkpoint).resolve()),
                    "source_seed_sha256": source_sha256,
                    "source_seed_epoch": int(seed_payload.get("epoch", -1)),
                    "git_commit": git_commit(),
                },
                checkpoints / "best_np.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                stopped_early = True
                break

    torch.save(model.npf.state_dict(), checkpoints / "final_np_state.pt")
    best_payload = torch.load(checkpoints / "best_np.pt", map_location=device, weights_only=True)
    model.npf.load_state_dict(best_payload["np_state"], strict=True)
    summary = {
        "config": asdict(config),
        "git_commit": git_commit(),
        "device": str(device),
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "history": history,
        "train_records": len(train_data),
        "validation_records": len(val_data),
        "source_seed_checkpoint": str(Path(seed_checkpoint).resolve()),
        "source_seed_sha256": source_sha256,
        "source_seed_epoch": int(seed_payload.get("epoch", -1)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    return summary


def load_neural_process_checkpoint(
    checkpoint_path: str | Path,
    *,
    seed_checkpoint: str | Path,
    evaluation: int,
    fold: int,
    device: torch.device,
) -> tuple[SeedNeuralProcess, dict]:
    """Build the wrapper from the immutable SEED checkpoint and NP weights."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = payload.get("config", {})
    if int(config.get("evaluation", -1)) != evaluation or int(config.get("fold", -1)) != fold:
        raise ValueError("Neural Process checkpoint evaluation/fold metadata does not match the request.")
    seed_model, _ = load_seed_checkpoint(
        seed_checkpoint, device=device, evaluation=evaluation, fold=fold
    )
    recorded_source_sha256 = payload.get("source_seed_sha256")
    if recorded_source_sha256 and recorded_source_sha256 != checkpoint_sha256(seed_checkpoint):
        raise ValueError(
            "The Neural Process checkpoint was trained from a different SEED checkpoint."
        )
    model = SeedNeuralProcess(
        seed_model,
        kind=config.get("model_kind", "attn_lnp"),
        context_fraction=float(config.get("context_fraction", 0.25)),
        r_dim=int(config.get("r_dim", 64)),
        z_dim=int(config.get("z_dim", 64)),
        latent_samples_train=int(config.get("latent_samples_train", 1)),
        latent_samples_test=int(config.get("latent_samples_test", 16)),
        freeze_seed_backbone=bool(config.get("freeze_seed_backbone", True)),
    ).to(device)
    model.npf.load_state_dict(payload["np_state"], strict=True)
    return model.eval(), payload
