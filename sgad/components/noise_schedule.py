# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.distributions import Normal

from sgad.sampletorsion.rotate import set_torsions

from sgad.utils.data_utils import subtract_com_batch


class BaseNoiseSchedule(ABC):
    @abstractmethod
    def g(self, t) -> torch.Tensor:
        # Returns g(t)
        pass

    @abstractmethod
    def h(self, t) -> torch.Tensor:
        # Returns \sqrt( \int_0^t g(t)^2 dt )
        pass

    @abstractmethod
    def alpha(self, t) -> torch.Tensor:
        # Returns alpha(t)^2 = \int_t^1 g(z)^2 dz
        pass

    @abstractmethod
    def sample_posterior(self, t, graph) -> torch.Tensor:
        # Returns p_t(x | x_1)
        pass


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma):
        self.sigma = sigma

    def g(self, t) -> torch.Tensor:
        return torch.full_like(t, self.sigma)

    def h(self, t) -> torch.Tensor:
        return self.sigma * torch.sqrt(t)

    @torch.no_grad()
    def alpha(self, t) -> torch.Tensor:
        # alpha(t)^2 = int_t^1 g(z)^2 dz
        return self.sigma * torch.sqrt(1 - t)

    @torch.no_grad()
    def sample_posterior(self, t, graph):
        graph_t = graph.clone()
        x1 = graph["positions"]
        batch_index = graph["batch"]
        t = t[batch_index, None]
        mean = x1 * t
        var = self.sigma**2 * (1 - t) * t
        xt = mean + torch.randn(x1.shape).to(x1.device) * torch.sqrt(var)
        graph_t["positions"] = subtract_com_batch(xt, graph_t["batch"])
        return graph_t


class GeometricNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_max / sigma_min

    def g(self, t) -> torch.Tensor:
        g_t = (
            self.sigma_min
            * (self.sigma_diff ** (1 - t))
            * ((2 * np.log(self.sigma_diff)) ** 0.5)
        )
        return g_t

    def h(self, t) -> torch.Tensor:
        return self.sigma_max * torch.sqrt(1 - self.sigma_diff ** (-2 * t))

    @torch.no_grad()
    def alpha(self, t) -> torch.Tensor:
        return self.sigma_max * torch.sqrt(
            self.sigma_diff ** (-2 * t) - self.sigma_diff ** (-2)
        )

    @torch.no_grad()
    def sample_posterior(self, t, graph) -> torch.Tensor:
        graph = graph.clone()
        x1 = graph["positions"]
        batch_index = graph["batch"]
        t = t[batch_index, None]
        c_s = self.sigma_diff ** (-2 * t)
        c_1 = self.sigma_diff ** (-2)
        mean = (c_s - 1) / (c_1 - 1) * x1
        var = self.sigma_max**2 * (c_s - 1) * (c_s - c_1) / (c_1 - 1)
        x_t = mean + torch.randn(x1.shape).to(x1.device) * torch.sqrt(var)
        graph["positions"] = subtract_com_batch(x_t, graph["batch"])
        return graph


def normal_log_prob(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # compute the variance
    var = scale**2
    log_scale = scale.log()
    return (
        # -((value - self.loc) ** 2) / (2 * var)
        -(value**2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    )


@torch.compile
def logprob_diffusion_wrapped_normal(
    ntiles: int, sigma: float, t: torch.Tensor, torsions: torch.Tensor
) -> torch.Tensor:
    std = torch.sqrt(t) * sigma

    tor = torsions[None, ...]
    k = (
        torch.linspace(-ntiles, ntiles, 2 * ntiles + 1, device=tor.device).unsqueeze(-1)
        * 2
        * math.pi
    )

    # logp = Normal(loc=loc, scale=std, validate_args=False).log_prob(tor + k)
    logp = normal_log_prob(tor + k, scale=std)
    return torch.logsumexp(logp, dim=0).squeeze()


def get_tor1_in_ambient_space(
    ntiles: int,
    t: torch.Tensor,
    scale_at_t1: float,
    torsions: torch.Tensor,
) -> torch.Tensor:
    """we make the logprob calculations in repeated ambient space"""
    # ambient space repeats
    tor1 = torsions
    k = (
        torch.linspace(-ntiles, ntiles, 2 * ntiles + 1, device=tor1.device)
        * 2
        * math.pi
    )
    tor1pk = tor1[:, None] + k[None, :]

    logp1 = Normal(
        loc=torch.zeros((1,), device=t.device),
        scale=torch.ones((1,), device=t.device) * scale_at_t1,
    ).log_prob(tor1pk)

    # determine which tile
    idx = torch.multinomial(torch.exp(logp1), num_samples=1)
    tor1 = torch.gather(tor1pk, dim=1, index=idx).squeeze(1)
    return tor1


class LinearTorsionNoiseSchedule(LinearNoiseSchedule):
    def __init__(self, sigma: float, ntiles: int):
        self.sigma = sigma
        self.ntiles = ntiles

    @torch.no_grad()
    def sample_posterior(self, t: torch.Tensor, graph) -> torch.Tensor:
        graph_t = graph.clone()
        t = torch.repeat_interleave(t, graph_t["n_torsions"])
        tor1 = get_tor1_in_ambient_space(
            ntiles=self.ntiles,
            t=t,
            scale_at_t1=self.sigma,
            torsions=graph_t["torsions"],
        )
        # Backward conditional in ambient space.
        tort = tor1 * t + torch.sqrt((1 - t) * t) * self.sigma * torch.randn_like(tor1)
        # project to flat torus
        tort = (tort + math.pi) % (2 * math.pi) - math.pi
        # put into graph
        graph_t["positions"] = set_torsions(
            tort,
            graph_t["tor_index"],
            graph_t["index_to_rotate"],
            graph_t["positions"],
            graph_t["n_torsions"].max().item(),
            graph_t["tor_per_mol_label"],
        )
        graph_t["positions"] = subtract_com_batch(
            graph_t["positions"], graph_t["batch"]
        )
        graph_t["torsions"] = tort

        return graph_t

    def logprob_p1base(self, torsions: torch.Tensor) -> torch.Tensor:
        return logprob_diffusion_wrapped_normal(
            ntiles=self.ntiles,
            sigma=self.sigma,
            t=torch.ones_like(torsions),
            torsions=torsions,
        )


class GeometricTorsionNoiseSchedule(GeometricNoiseSchedule):
    def __init__(self, sigma_min: float, sigma_max: float, ntiles: int):
        super().__init__(sigma_min, sigma_max)
        self.ntiles = ntiles
        self.p1base_sigma = self.h(t=torch.Tensor([1.0])).item()

    def logprob_p1base(self, torsions: torch.Tensor) -> torch.Tensor:
        return logprob_diffusion_wrapped_normal(
            ntiles=self.ntiles,
            sigma=self.p1base_sigma,
            t=torch.ones_like(torsions),
            torsions=torsions,
        )

    @torch.no_grad()
    def sample_posterior(self, t, graph) -> torch.Tensor:
        graph_t = graph.clone()
        t = torch.repeat_interleave(t, graph_t["n_torsions"])
        tor1 = get_tor1_in_ambient_space(
            ntiles=self.ntiles,
            t=t,
            scale_at_t1=self.p1base_sigma,
            torsions=graph_t["torsions"],
        )
        # Backward conditional in ambient space.
        c_s = self.sigma_diff ** (-2 * t)
        c_1 = self.sigma_diff ** (-2)
        mean = (c_s - 1) / (c_1 - 1) * tor1
        var = self.sigma_max**2 * (c_s - 1) * (c_s - c_1) / (c_1 - 1)
        tort = mean + torch.randn(tor1.shape, device=tor1.device) * torch.sqrt(var)
        # project to flat torus
        tort = (tort + math.pi) % (2 * math.pi) - math.pi
        # put into graph
        graph_t["positions"] = set_torsions(
            tort,
            graph_t["tor_index"],
            graph_t["index_to_rotate"],
            graph_t["positions"],
            graph_t["n_torsions"].max().item(),
            graph_t["tor_per_mol_label"],
        )
        graph_t["positions"] = subtract_com_batch(
            graph_t["positions"], graph_t["batch"]
        )
        graph_t["torsions"] = tort

        return graph_t
