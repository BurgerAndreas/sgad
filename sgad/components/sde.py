# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch

from sgad.sampletorsion.rotate import set_torsions
from sgad.utils.data_utils import subtract_com_batch


def graph_add(graph: Dict[str, Any], delta: torch.Tensor) -> Dict[str, Any]:
    """Add a constant tensor to the positions in a molecular system graph.

    Args:
        graph: Graph dictionary containing 'positions' key
        delta: Tensor to add to positions

    Returns:
        Modified graph with updated positions
    """
    graph["pos"] = graph["pos"] + delta
    return graph


def graph_scale(graph: Dict[str, Any], scalar: float) -> Dict[str, Any]:
    """Scale the position vectors in a molecular system graph by a scalar.

    Args:
        graph: Graph dictionary containing 'positions' key
        scalar: Scaling factor to multiply positions by

    Returns:
        Modified graph with scaled positions
    """
    graph["pos"] = graph["pos"] * scalar
    return graph


class ControlledGraphSDE(torch.nn.Module):
    """Controlled Stochastic Differential Equation for molecular graphs.

    Implements a controlled SDE of the form:
        dx = f(t, x) dt + g(t) dW
    where f is the drift (control-dependent) and g is the diffusion coefficient.

    Supports two parameterizations:
    - Standard SDE: drift = g(t) * u(t,x), diffusion = g(t)
    - Adjoint Matching (AM) SDE: drift = u(t,x), diffusion = g(t)

    Attributes:
        learn_torsions: Whether this SDE learns torsion angles (False for base class)
    """

    learn_torsions = False

    def __init__(
        self,
        control: torch.nn.Module,
        noise_schedule: Any,
        use_adjointmatching_sde: bool = False,
    ):
        """Initialize the controlled SDE.

        Args:
            control: Neural network that outputs control signal u(t, x)
            noise_schedule: Noise schedule defining g(t) and related functions
            use_adjointmatching_sde: If True, use Adjoint Matching SDE parameterization
        """
        super().__init__()
        self.control = control
        self.noise_schedule = noise_schedule
        self.use_adjointmatching_sde = use_adjointmatching_sde

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Get diffusion coefficient g(t) from noise schedule.

        Args:
            t: Time value(s)

        Returns:
            Diffusion coefficient g(t)
        """
        g = self.noise_schedule.g(t)
        return g

    def f(
        self, t: torch.Tensor, graph_state: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute control and drift terms of the SDE.

        Args:
            t: Current time (scalar or per-system tensor)
            graph_state: Graph dictionary with molecular structure

        Returns:
            Tuple of (control u, drift f):
            - For AM SDE: (u, g*u)
            - For standard SDE: (g*u, g^2*u)
        """
        g_t = self.g(t)
        # Convert scalar time to per-system tensor if needed
        if t.dim() == 0:
            n_systems = len(graph_state["ptr"]) - 1
            t = t * torch.ones(n_systems).to(graph_state["pos"].device)

        # Compute control signal from neural network
        u = self.control(t, graph_state)

        # Return (control, drift) based on parameterization
        if self.use_adjointmatching_sde:
            return u, g_t * u
        else:
            return g_t * u, g_t**2 * u


class ControlledGraphTorsionSDE(ControlledGraphSDE):
    """Controlled SDE for molecular graphs with torsion angle learning.

    Extends ControlledGraphSDE to enable learning of torsional degrees of freedom
    in addition to the Cartesian positions.

    Attributes:
        learn_torsions: Set to True to enable torsion learning
    """

    learn_torsions = True


def euler_maruyama_step(
    sde: ControlledGraphSDE,
    t: torch.Tensor,
    graph_state: Dict[str, Any],
    dt: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Perform one Euler-Maruyama integration step for the SDE.

    Implements the discrete-time update:
        x_{t+dt} = x_t + f(t,x_t)*dt + g(t)*sqrt(dt)*epsilon
    where epsilon ~ N(0,I).

    Args:
        sde: The controlled SDE defining drift and diffusion
        t: Current time
        graph_state: Current graph state with positions
        dt: Time step size

    Returns:
        Tuple of (control, next_graph_state):
        - control: Control signal u at time t
        - next_graph_state: Graph state at time t+dt with COM subtracted
    """
    # Calculate drift and diffusion terms
    u, f = sde.f(t, graph_state)
    drift = f * dt

    # Diffusion term: g(t) * sqrt(dt) * noise
    diffusion = sde.g(t) * np.sqrt(dt) * torch.randn_like(graph_state["pos"])

    # Update the graph state: x_{t+dt} = x_t + drift + diffusion
    graph_state_next = graph_add(graph_state, drift + diffusion)
    # Subtract center of mass to maintain COM = 0 constraint
    graph_state_next["pos"] = subtract_com_batch(
        graph_state_next["pos"], graph_state_next["batch"]
    )
    return u, graph_state_next


@torch.no_grad()
def integrate_sde(
    sde: ControlledGraphSDE,
    graph0: Dict[str, Any],
    num_integration_steps: int,
    only_final_state: bool,
    discretization_scheme: str = "uniform",
) -> Any:
    """Integrate the controlled SDE from t=0 to t=1 using Euler-Maruyama method.

    Numerically solves the SDE by discretizing time and taking Euler-Maruyama steps.
    Supports different time discretization schemes for improved accuracy.

    Args:
        sde: The controlled SDE to integrate
        graph0: Initial graph state at t=0
        num_integration_steps: Number of integration steps
        only_final_state: If True, return only final state; if False, return controls too
        discretization_scheme: Time discretization scheme ("uniform" or "ql")

    Returns:
        If only_final_state=True: final graph state at t=1
        If only_final_state=False: tuple of (final_graph_state, controls)
        where controls is a stacked tensor of all control signals

    Raises:
        ValueError: If discretization_scheme is not recognized
    """
    # Select time discretization scheme
    if discretization_scheme == "uniform":
        times = uniform_discretization(num_steps=num_integration_steps)
    elif discretization_scheme == "ql":
        times = quadratic_linear_discretization(num_steps=num_integration_steps)
    else:
        raise ValueError(
            f"Unknown discretization_scheme option {discretization_scheme}"
        )

    # Start from initial state
    graph_state = graph0.clone()

    # Integrate through time
    controls = []
    for t, t_next in zip(times[:-1], times[1:]):
        dt = t_next - t
        u, graph_state = euler_maruyama_step(sde, t, graph_state, dt)
        controls.append(u)

    # Return results based on flag
    if only_final_state:
        return graph_state
    else:
        controls = torch.stack(controls)
        return graph_state, controls


def uniform_discretization(
    num_steps: int, time_limits: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    """Create uniformly spaced time steps.

    Args:
        num_steps: Number of integration steps (creates num_steps+1 time points)
        time_limits: Tuple of (start_time, end_time)

    Returns:
        Tensor of uniformly spaced time points from start to end
    """
    return torch.linspace(time_limits[0], time_limits[1], num_steps + 1)


def quadratic_linear_discretization(
    num_steps: int = 50,
    time_limits: Tuple[float, float] = (0, 1),
    fraction_of_linear_steps: float = 0.5,
) -> torch.Tensor:
    """Create quadratic-linear time discretization for improved sampling.

    Combines linear steps at the beginning with quadratic spacing later.
    This provides finer resolution at early times where dynamics are often
    more critical, while using coarser steps later.

    Args:
        num_steps: Number of integration steps
        time_limits: Tuple of (start_time, end_time), default (0, 1)
        fraction_of_linear_steps: Fraction of steps that are linearly spaced

    Returns:
        Tensor of time points with quadratic-linear spacing (reversed)
    """
    num_steps = num_steps + 2
    # Create fine linear grid to sample from
    start_steps = torch.linspace(time_limits[0], time_limits[1], 1000)

    # Take linear steps at the beginning
    num_start_ts = int(num_steps * fraction_of_linear_steps)
    start_ts = start_steps[:num_start_ts]

    # Create quadratic spacing for remaining steps
    x = torch.linspace(
        time_limits[0],
        time_limits[1],
        num_steps - num_start_ts - 1,
    )
    timesteps = x**2

    # Scale quadratic steps to fit after linear portion
    scale = 1 - start_steps[num_start_ts]
    timesteps = start_steps[num_start_ts] + timesteps / timesteps.max() * scale
    timesteps = torch.cat((start_ts, timesteps), dim=0)

    # Flip to reverse time direction (1 to 0)
    return torch.flip(1.0 - timesteps, [0])
