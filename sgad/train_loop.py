# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Any, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric

from sgad.components.soc import (
    adjoint_score_target,
    adjoint_score_target_torsion,
)
from sgad.sampletorsion.torsion import check_torsions

from sgad.utils.data_utils import cycle


def get_lr(optimizer: Optimizer) -> float:
    """Get the current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate from the first parameter group
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_one_epoch(
    controller: torch.nn.Module,
    noise_schedule: Any,
    clipper: Any,
    train_dataloader: DataLoader,
    optimizer: Optimizer,
    warmup_scheduler: Any,
    lr_schedule: Optional[Any],
    device: torch.device,
    cfg: Any,
    pretrain_mode: bool = False,
) -> Dict[str, float]:
    """Train the model for one epoch using score matching loss.

    Implements training with a combination of adjoint score matching loss and
    Bridge Matching (BM) loss. The score is learned by matching the adjoint
    score target computed from energy gradients and the noise schedule.

    Args:
        controller: Neural network model that predicts scores
        noise_schedule: Noise schedule object defining diffusion process
        clipper: Gradient clipper for energy gradients
        train_dataloader: DataLoader providing training batches
        optimizer: PyTorch optimizer for model parameters
        warmup_scheduler: Learning rate warmup scheduler
        lr_schedule: Optional learning rate schedule for cosine annealing
        device: Device to run training on (CPU/GPU)
        cfg: Configuration object with training hyperparameters
        pretrain_mode: If True, uses BM-only loss during pretraining

    Returns:
        Dictionary containing the average loss for the epoch
    """
    # Set model to training mode
    controller.train(True)
    # Initialize metric to track epoch loss
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    # Create infinite iterator over training data
    loader = iter(cycle(train_dataloader))

    for i in range(cfg.num_batches_per_epoch):
        optimizer.zero_grad()

        # Get batch of clean structures (t=1) and their energy gradients
        graph_state_1, grad_E = next(loader)
        graph_state_1 = graph_state_1.to(device)

        # Sample random time steps for each system in batch
        n_systems = len(graph_state_1["ptr"]) - 1
        t = torch.rand(n_systems).to(device)

        # Sample noisy structures at time t using posterior p_t(x|x_1)
        graph_state_t = noise_schedule.sample_posterior(t, graph_state_1)

        # Predict score at time t
        predicted_score = controller(t, graph_state_t)

        # Get noise schedule functions g(t) and alpha(t) per atom
        g_t = noise_schedule.g(t)[graph_state_1["batch"], None]
        alpha_t = noise_schedule.alpha(t)[graph_state_1["batch"], None]

        # Compute adjoint score target from energy gradients
        score_target = adjoint_score_target(
            graph_state_1, grad_E, noise_schedule, clipper, no_pbase=cfg.no_pbase
        )

        # For Adjoint Matching SDE parameterization, scale by g(t)
        if cfg.use_adjointmatching_sde:
            predicted_score = predicted_score / g_t

        # Adjoint score matching loss: ||s_theta - s_target||^2
        adjoint_loss = (predicted_score - score_target).pow(2).sum(-1).mean(0)

        # Bridge Matching loss (denoising score matching)
        if cfg.scale_bridgematching_loss:
            # Scaled formulation: ||s_theta * alpha^2 - (x_1 - x_t)||^2
            bridgematching_loss = (
                (
                    predicted_score * alpha_t.pow(2)
                    - (graph_state_1["positions"] - graph_state_t["positions"])
                )
                .pow(2)
                .sum(-1)
                .mean(0)
            )
        else:
            # Standard formulation: ||s_theta - (x_1 - x_t) / alpha^2||^2
            bridgematching_loss = (
                (
                    predicted_score
                    - 1
                    / alpha_t.pow(2)
                    * (graph_state_1["positions"] - graph_state_t["positions"])
                )
                .pow(2)
                .sum(-1)
                .mean(0)
            )

        # Select loss based on pretraining mode
        if pretrain_mode and cfg.BM_only_pretrain:
            loss = bridgematching_loss
        else:
            loss = adjoint_loss + cfg.bridgematching_loss_weight * bridgematching_loss

        # Backpropagation and optimization step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.grad_clip)
        optimizer.step()
        epoch_loss.update(loss.item())

        # Update learning rate with warmup dampening
        with warmup_scheduler.dampening():
            if lr_schedule:
                lr_schedule.step()

    return {"loss": float(epoch_loss.compute().detach().cpu())}
