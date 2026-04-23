from __future__ import annotations
from typing import Literal, TypedDict
import torch
from torch import Tensor


class TrajectoryBatch(TypedDict):
    """Batch of trajectory data for PPO training."""
    advantages: Tensor   # [B, T]
    returns: Tensor      # [B, T]
    values: Tensor       # [B, T]
    logp_old: Tensor     # [B, T]
    logp_new: Tensor     # [B, T]
    logp_ref: Tensor     # [B, T]
    mask: Tensor         # [B, T] 1=valid, 0=pad


class PPOHyperParams(TypedDict):
    """Hyperparameters for PPO loss computation."""
    clip_eps: float
    vf_coef: float
    kl_coef: float
    normalize_adv: bool

class PPOLosses(TypedDict):
    """Output losses from PPO computation."""
    policy_loss: Tensor
    value_loss: Tensor
    kl_loss: Tensor
    total_loss: Tensor


# -----------------------------------------------------------------------------
# PPO Loss
# -----------------------------------------------------------------------------

def ppo_losses(
    batch: TrajectoryBatch,
    hparams: PPOHyperParams,
) -> PPOLosses:
    """
    Compute PPO losses for a batch of trajectories.

    Args:
        batch: Trajectory data containing advantages, returns, values, and log probs
        hparams: PPO hyperparameters (clip_eps, vf_coef, kl_coef, normalize_adv)

    Returns:
        PPOLosses
    """
    advantages = batch["advantages"]
    returns = batch["returns"]
    values = batch["values"]
    logp_old = batch["logp_old"]
    logp_new = batch["logp_new"]
    logp_ref = batch["logp_ref"]
    mask = batch["mask"]

    clip_eps = hparams["clip_eps"]
    normalize_adv = hparams["normalize_adv"]
    vf_coef = hparams["vf_coef"]
    kl_coef = hparams["kl_coef"]
    
    # normalize advantages (only over valid tokens)
    if normalize_adv:
        adv_mean = (advantages * mask).sum() / mask.sum()
        adv_var = ((advantages - adv_mean) ** 2 * mask).sum()/mask.sum()
        advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

    # policy loss 
    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    policy_loss = (policy_loss * mask).sum() / mask.sum()

    # values loss
    value_loss = (returns - values) ** 2
    value_loss = (value_loss * mask).sum() / mask.sum()

    # kl loss 
    kl = logp_new - logp_ref
    kl = (kl * mask).sum() / mask.sum()

    # total loss
    total_loss = policy_loss + vf_coef * value_loss + kl * kl_coef

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "kl_loss": kl,
        "total_loss": total_loss,
    }


def run_basic_tests(loss_fn):
    B, T = 4, 8

    batch = {
        "advantages": torch.randn(B, T, requires_grad=True),
        "returns": torch.randn(B, T),
        "values": torch.randn(B, T, requires_grad=True),
        "logp_old": torch.randn(B, T),
        "logp_new": torch.randn(B, T, requires_grad=True),
        "logp_ref": torch.randn(B, T),
        "mask": torch.ones(B, T),
    }

    hparams = {
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "kl_coef": 0.1,
        "normalize_adv": True,
    }

    out = loss_fn(batch, hparams)

    for k in ["policy_loss", "value_loss", "kl_loss", "total_loss"]:
        assert k in out, f"Missing key {k}"
        assert torch.isfinite(out[k]).all(), f"{k} has NaN/Inf"

    out["total_loss"].backward()

    print("✅ Basic test passed")

run_basic_tests(ppo_losses)