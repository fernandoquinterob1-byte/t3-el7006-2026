"""SEED-backed Neural Process models.

The upstream Neural Process Family implementation is used for the set encoder,
attention mechanism and latent path.  This module supplies the binary temporal
segmentation adapter needed by this project and keeps the saved SEED checkpoint
as an immutable feature extractor by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Independent, constraints

from .model import SEEDModel
from .npf_compat import ensure_npf_importable

ensure_npf_importable()
from npf import AttnCNP, AttnLNP  # noqa: E402


NPModelKind = Literal["attn_cnp", "attn_lnp"]
REDUCED_FACTOR = 8
SEED_FEATURE_DIM = 512


class SoftBernoulli(Distribution):
    """Bernoulli log likelihood that also accepts soft targets in ``[0, 1]``."""

    arg_constraints = {"logits": constraints.real}
    support = constraints.unit_interval
    has_rsample = False

    def __init__(self, logits: torch.Tensor, validate_args: bool | None = None):
        self.logits = logits
        super().__init__(
            batch_shape=logits.shape,
            event_shape=torch.Size(),
            validate_args=validate_args,
        )

    @property
    def probs(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    @property
    def mean(self) -> torch.Tensor:
        return self.probs

    @property
    def variance(self) -> torch.Tensor:
        probability = self.probs
        return probability * (1.0 - probability)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        value = value.to(dtype=self.logits.dtype)
        value = torch.broadcast_to(value, self.logits.shape)
        return -F.binary_cross_entropy_with_logits(self.logits, value, reduction="none")

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        shape = sample_shape + self.probs.shape
        return torch.bernoulli(self.probs.expand(shape))


def binary_predictive_distribution(
    loc: torch.Tensor, _: torch.Tensor
) -> Independent:
    """Adapter for NPF's ``(loc, scale)`` decoder contract."""

    return Independent(SoftBernoulli(loc, validate_args=False), 1)


def _build_npf(
    *,
    kind: NPModelKind,
    x_dim: int,
    r_dim: int,
    z_dim: int,
    latent_samples_train: int,
    latent_samples_test: int,
) -> nn.Module:
    common = dict(
        x_dim=x_dim,
        y_dim=1,
        r_dim=r_dim,
        PredictiveDistribution=binary_predictive_distribution,
        p_y_loc_transformer=nn.Identity(),
        p_y_scale_transformer=nn.Identity(),
    )
    if kind == "attn_cnp":
        return AttnCNP(**common)
    if kind == "attn_lnp":
        return AttnLNP(
            **common,
            z_dim=z_dim,
            is_q_zCct=True,
            n_z_samples_train=latent_samples_train,
            n_z_samples_test=latent_samples_test,
        )
    raise ValueError(f"Unknown Neural Process model kind: {kind}")


class SeedNeuralProcess(nn.Module):
    """A Neural Process adapter conditioned on a saved SEED checkpoint."""

    def __init__(
        self,
        seed_model: SEEDModel,
        *,
        kind: NPModelKind = "attn_lnp",
        context_fraction: float = 0.25,
        r_dim: int = 64,
        z_dim: int = 64,
        latent_samples_train: int = 1,
        latent_samples_test: int = 16,
        freeze_seed_backbone: bool = True,
    ):
        super().__init__()
        if not 0.0 < context_fraction <= 1.0:
            raise ValueError("context_fraction must be in (0, 1].")
        self.seed_model = seed_model
        self.kind = kind
        self.context_fraction = float(context_fraction)
        self.r_dim = int(r_dim)
        self.z_dim = int(z_dim)
        self.latent_samples_train = int(latent_samples_train)
        self.latent_samples_test = int(latent_samples_test)
        self.freeze_seed_backbone = bool(freeze_seed_backbone)
        self.npf = _build_npf(
            kind=kind,
            x_dim=1 + SEED_FEATURE_DIM + 1,
            r_dim=self.r_dim,
            z_dim=self.z_dim,
            latent_samples_train=self.latent_samples_train,
            latent_samples_test=self.latent_samples_test,
        )
        if self.freeze_seed_backbone:
            for parameter in self.seed_model.parameters():
                parameter.requires_grad_(False)
            self.seed_model.eval()

    def train(self, mode: bool = True) -> "SeedNeuralProcess":
        super().train(mode)
        if self.freeze_seed_backbone:
            self.seed_model.eval()
        return self

    @property
    def reduced_time(self) -> int:
        return 500

    @staticmethod
    def _normalise_seed_features(hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # NPF validates X in [-1, 1].  Tanh keeps the pretrained representation
        # bounded without estimating statistics from validation or test records.
        hidden = torch.tanh(hidden / 4.0)
        seed_logit = torch.tanh(logits.unsqueeze(-1) / 4.0)
        return torch.cat((hidden, seed_logit), dim=-1)

    def encode_signal(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bounded target features, SEED probabilities and logits."""
        if signal.ndim != 2:
            raise ValueError(f"signal must have shape (batch, time), got {tuple(signal.shape)}")
        context = torch.no_grad() if self.freeze_seed_backbone else torch.enable_grad()
        with context:
            hidden = self.seed_model.transformer(self.seed_model.embedding(signal))
            logits = self.seed_model.classifier(hidden)
        if hidden.shape[-1] != SEED_FEATURE_DIM:
            raise RuntimeError(
                f"Expected SEED hidden dimension {SEED_FEATURE_DIM}, got {hidden.shape[-1]}"
            )
        features = self._normalise_seed_features(hidden, logits)
        time = torch.linspace(-1.0, 1.0, hidden.shape[1], device=signal.device, dtype=signal.dtype)
        time = time.view(1, -1, 1).expand(signal.shape[0], -1, -1)
        x = torch.cat((time, features), dim=-1)
        return x, torch.sigmoid(logits), logits

    def reduce_targets(self, target: torch.Tensor, n_points: int) -> torch.Tensor:
        if target.ndim == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if target.ndim != 2:
            raise ValueError(f"target must have shape (batch, time), got {tuple(target.shape)}")
        if target.shape[1] % n_points:
            raise ValueError(
                f"Target length {target.shape[1]} is not divisible by reduced length {n_points}."
            )
        factor = target.shape[1] // n_points
        return target[:, : n_points * factor].reshape(target.shape[0], n_points, factor).mean(-1).unsqueeze(-1)

    def context_indices(
        self, n_points: int, *, device: torch.device, randomise: bool
    ) -> torch.Tensor:
        n_context = max(1, min(n_points, round(n_points * self.context_fraction)))
        if randomise:
            return torch.randperm(n_points, device=device)[:n_context]
        return torch.linspace(0, n_points - 1, n_context, device=device).round().long().unique()

    def make_episode(
        self,
        signal: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        randomise_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        x, seed_probability, seed_logits = self.encode_signal(signal)
        indices = self.context_indices(x.shape[1], device=x.device, randomise=randomise_context)
        x_context = x[:, indices]
        y_context = seed_probability[:, indices].unsqueeze(-1)
        y_target = None if target is None else self.reduce_targets(target, x.shape[1])
        return x_context, y_context, x, y_target, seed_logits

    def forward(
        self,
        signal: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        randomise_context: bool = True,
    ):
        x_context, y_context, x_target, y_target, _ = self.make_episode(
            signal, target, randomise_context=randomise_context
        )
        return self.npf(x_context, y_context, x_target, y_target)

    @torch.no_grad()
    def predict(
        self,
        signal: torch.Tensor,
        *,
        randomise_context: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``samples``, mean probability and SEED logits at reduced rate."""
        self.eval()
        x_context, y_context, x_target, _, seed_logits = self.make_episode(
            signal, None, randomise_context=randomise_context
        )
        distribution = self.npf(x_context, y_context, x_target)[0]
        # The latent samples are the uncertainty samples.  Returning the
        # Bernoulli draws themselves would mix epistemic and observation noise.
        probability_samples = distribution.mean.squeeze(-1)
        mean = probability_samples
        return probability_samples, mean, seed_logits


def load_seed_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    evaluation: int,
    fold: int,
) -> tuple[SEEDModel, dict]:
    """Load an existing ``best_full.pt`` without changing it."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SEED checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid SEED checkpoint payload: {checkpoint_path}")
    config = payload.get("config", {})
    if int(config.get("evaluation", -1)) != evaluation or int(config.get("fold", -1)) != fold:
        raise ValueError("SEED checkpoint evaluation/fold metadata does not match the requested fold.")
    model = SEEDModel().to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), payload
