# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch

from sgad.utils.data_utils import graph_to_vector, subtract_com_batch


@torch.no_grad
def adjoint_score_target_torsion(grad_E, clipper):
    adjoint_state = grad_E["energy_grad_tor"] + grad_E["grad_tor_p1base"]
    adjoint_state = clipper.clip_scores(adjoint_state)
    return -adjoint_state


def adjoint_score_target(
    graph_state_1, grad_E, noise_schedule, clipper, no_pbase=False
):
    sigma_1 = noise_schedule.h(
        torch.Tensor([1.0]).to(graph_state_1["positions"].device)
    )
    x1 = graph_state_1["positions"]
    if no_pbase:
        adjoint_state = grad_E["energy_grad"].to(graph_state_1["positions"].device)
    else:
        adjoint_state = (
            grad_E["energy_grad"].to(graph_state_1["positions"].device)
            - 1 / sigma_1**2 * x1
        )
    adjoint_state = clipper.clip_scores(adjoint_state)
    adjoint_state = adjoint_state + clipper.clip_scores(
        grad_E["reg_grad"].to(graph_state_1["positions"].device)
    )
    adjoint_state = subtract_com_batch(adjoint_state, graph_state_1["batch"])

    # later we expect the negative
    return -adjoint_state


# Currently computing SOC loss for one type of molecule unless we define to also take expectation over molecule types
@torch.no_grad()
def SOC_loss(controls, graph_state, energies, noise_schedule):
    # should only be called when using batches of the same molecule type
    n_spatial_dim = graph_state["positions"].shape[-1]
    n_particles = graph_state["ptr"][1]
    x1 = graph_to_vector(graph_state["positions"], n_particles, n_spatial_dim)
    controls = controls.reshape(controls.shape[0], *x1.shape)
    # approximate running cost: 1/2 * int_0^1 ||u_t||_2^2 dt \approx 1/2 * sum^N_i 1/N ||u_{t_i}||^2_2
    stage_cost = 0.5 * (controls * controls).sum(-1).sum(0) / controls.shape[0]
    n = x1.shape[-1]
    sigma_1 = noise_schedule.h(torch.Tensor([1.0]).to(graph_state["positions"].device))
    log_pb_1 = (
        -(0.5 / sigma_1**2 * (x1 * x1)).sum(-1)
        - n * torch.log(sigma_1)
        - n / 2 * np.log(2 * torch.pi)
    )
    terminal_cost = energies + log_pb_1
    # unnormalized cost, only average after accumulation
    return (stage_cost + terminal_cost).sum()


@torch.no_grad()
def SOC_loss_torsion(controls, graph_state, energies, noise_schedule):
    # should only be called when using batches of the same molecule type
    assert torch.all(graph_state["n_torsions"][0] == graph_state["n_torsions"])
    n_tors_per_mol = graph_state["n_torsions"][0].item()
    n_systems = len(graph_state["ptr"]) - 1
    tor1 = graph_state["torsions"].reshape(n_systems, n_tors_per_mol)
    controls = controls.reshape(controls.shape[0], *tor1.shape)
    # approximate running cost: 1/2 * int_0^1 ||u_t||_2^2 dt \approx 1/2 * sum^N_i 1/N ||u_{t_i}||^2_2

    return torch.tensor([0.0])  # since we did not write the loss, TODO
