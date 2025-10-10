# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch

from sgad.sampletorsion.rotate import set_torsions
from sgad.utils.data_utils import subtract_com_batch


# adds a constant to the position of a molecular system graph
def graph_add(graph, delta):
    graph["positions"] = graph["positions"] + delta
    return graph


# scale position vector of a molecular system graph
def graph_scale(graph, scalar):
    graph["positions"] = graph["positions"] * scalar
    return graph


class ControlledGraphSDE(torch.nn.Module):
    learn_torsions = False

    def __init__(self, control, noise_schedule, use_AM_SDE=False):
        super().__init__()
        self.control = control
        self.noise_schedule = noise_schedule
        self.use_AM_SDE = use_AM_SDE

    def g(self, t):
        g = self.noise_schedule.g(t)
        return g

    def f(self, t, graph_state):
        g_t = self.g(t)
        if t.dim() == 0:
            n_systems = len(graph_state["ptr"]) - 1
            t = t * torch.ones(n_systems).to(graph_state["positions"].device)
        u = self.control(t, graph_state)

        if self.use_AM_SDE:
            return u, g_t * u
        else:
            return g_t * u, g_t**2 * u


class ControlledGraphTorsionSDE(ControlledGraphSDE):
    learn_torsions = True


def euler_maruyama_step(
    sde,
    t,
    graph_state,
    dt,
):
    # Calculate drift and diffusion terms
    u, f = sde.f(t, graph_state)
    drift = f * dt

    diffusion = sde.g(t) * np.sqrt(dt) * torch.randn_like(graph_state["positions"])
    # Update the graph state and substract com
    graph_state_next = graph_add(graph_state, drift + diffusion)
    graph_state_next["positions"] = subtract_com_batch(
        graph_state_next["positions"], graph_state_next["batch"]
    )
    return u, graph_state_next


@torch.no_grad()
def integrate_sde(
    sde,
    graph0,
    num_integration_steps,
    only_final_state,
    discretization_scheme="uniform",
):
    if discretization_scheme == "uniform":
        times = uniform_discretization(num_steps=num_integration_steps)
    elif discretization_scheme == "ql":
        times = quadratic_linear_discretization(num_steps=num_integration_steps)
    else:
        raise ValueError(
            f"Unknown discretization_scheme option {discretization_scheme}"
        )

    graph_state = graph0.clone()

    controls = []
    for t, t_next in zip(times[:-1], times[1:]):
        dt = t_next - t
        u, graph_state = euler_maruyama_step(sde, t, graph_state, dt)
        controls.append(u)
    if only_final_state:
        return graph_state
    else:
        controls = torch.stack(controls)
        return graph_state, controls


def uniform_discretization(num_steps, time_limits=(0, 1)):
    return torch.linspace(time_limits[0], time_limits[1], num_steps + 1)


def quadratic_linear_discretization(
    num_steps=50,
    time_limits=(0, 1),
    fraction_of_linear_steps=0.5,
):
    num_steps = num_steps + 2
    start_steps = torch.linspace(time_limits[0], time_limits[1], 1000)
    num_start_ts = int(num_steps * fraction_of_linear_steps)
    start_ts = start_steps[:num_start_ts]

    x = torch.linspace(
        time_limits[0],
        time_limits[1],
        num_steps - num_start_ts - 1,
    )
    timesteps = x**2

    scale = 1 - start_steps[num_start_ts]
    timesteps = start_steps[num_start_ts] + timesteps / timesteps.max() * scale
    timesteps = torch.cat((start_ts, timesteps), dim=0)

    return torch.flip(1.0 - timesteps, [0])
