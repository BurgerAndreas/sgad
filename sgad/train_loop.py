# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torchmetrics.aggregation import MeanMetric

from sgad.components.soc import (
    adjoint_score_target,
    adjoint_score_target_torsion,
)
from sgad.sampletorsion.torsion import check_torsions

from sgad.utils.data_utils import cycle


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_one_epoch(
    controller,
    noise_schedule,
    clipper,
    train_dataloader,
    optimizer,
    warmup_scheduler,
    lr_schedule,
    device,
    cfg,
    pretrain_mode=False,
):
    epoch_loss = 0
    controller.train(True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    loader = iter(cycle(train_dataloader))
    for i in range(cfg.num_batches_per_epoch):
        # print(get_lr(optimizer))
        optimizer.zero_grad()
        graph_state_1, grad_E = next(loader)
        graph_state_1 = graph_state_1.to(device)
        n_systems = len(graph_state_1["ptr"]) - 1
        t = torch.rand(n_systems).to(device)
        if cfg.learn_torsions:
            check_torsions(
                graph_state_1["positions"],
                graph_state_1["tor_index"],
                graph_state_1["torsions"],
            )
        graph_state_t = noise_schedule.sample_posterior(t, graph_state_1)

        predicted_score = controller(t, graph_state_t)

        if cfg.learn_torsions:
            g_t = torch.repeat_interleave(
                noise_schedule.g(t), graph_state_1["n_torsions"]
            )
            alpha_t = torch.repeat_interleave(
                noise_schedule.alpha(t), graph_state_1["n_torsions"]
            )
            score_target = adjoint_score_target_torsion(grad_E, clipper)
        else:
            g_t = noise_schedule.g(t)[graph_state_1["batch"], None]
            alpha_t = noise_schedule.alpha(t)[graph_state_1["batch"], None]
            score_target = adjoint_score_target(
                graph_state_1, grad_E, noise_schedule, clipper, no_pbase=cfg.no_pbase
            )

        if cfg.use_AM_SDE:
            predicted_score = predicted_score / g_t

        adjoint_loss = (predicted_score - score_target).pow(2).sum(-1).mean(0)

        if cfg.scaled_BM_loss and cfg.learn_torsions:
            bm_loss = (
                (
                    predicted_score * alpha_t.pow(2)
                    - (graph_state_1["torsions"] - graph_state_t["torsions"])
                )
                .pow(2)
                .mean(0)
            )
        elif cfg.scaled_BM_loss and not cfg.learn_torsions:
            bm_loss = (
                (
                    predicted_score * alpha_t.pow(2)
                    - (graph_state_1["positions"] - graph_state_t["positions"])
                )
                .pow(2)
                .sum(-1)
                .mean(0)
            )
        elif not cfg.scaled_BM_loss and cfg.learn_torsions:
            bm_loss = (
                (
                    predicted_score
                    - 1
                    / alpha_t.pow(2)
                    * (graph_state_1["torsions"] - graph_state_t["torsions"])
                )
                .pow(2)
                .mean(0)
            )
        else:  # not cfg.scaled_BM_loss and not cfg.learn_torsions:
            bm_loss = (
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

        if pretrain_mode and cfg.BM_only_pretrain:
            loss = bm_loss
        else:
            loss = adjoint_loss + cfg.BM_loss_weight * bm_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.grad_clip)
        optimizer.step()
        epoch_loss.update(loss.item())

        with warmup_scheduler.dampening():
            if lr_schedule:
                lr_schedule.step()

    return {"loss": float(epoch_loss.compute().detach().cpu())}
