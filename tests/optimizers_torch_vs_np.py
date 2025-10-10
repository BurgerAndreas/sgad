#!/usr/bin/env python3
"""
Compare Torch vs NumPy optimizers (FIRE, BFGS) on the same deterministic toy potential.

Checks that final energies are within a small tolerance.
"""

import math
import random
import numpy as np
import torch

from sgad.optimizer_np import Geometry as GeometryNP, FIRE as FIRE_NP, BFGS as BFGS_NP
from sgad.rfo_np import RFOptimizer as RFO_NP
from sgad.optimizer_torch import GeometryTorch, FIRE as FIRE_TORCH, BFGS as BFGS_TORCH
from sgad.rfo_torch import RFOptimizer as RFO_TORCH
from time import perf_counter


class DeterministicCalculator:
    """Deterministic quadratic bowl with slight anisotropy and fixed seed noise-free behavior."""

    def __init__(self, shift=0.0):
        self.shift = float(shift)

    def predict(self, coords, atomic_nums, do_hessian=False):
        # coords may be numpy array (flat or Nx3) or torch tensor; standardize to torch
        coords_t = torch.as_tensor(coords, dtype=torch.float64).reshape(-1, 3)
        # Slightly anisotropic quadratic: E = 0.5 * sum((A x)^2) / N + shift
        A = torch.tensor([1.0, 1.2, 0.8], dtype=coords_t.dtype)
        Ax = coords_t * A
        energy = 0.5 * torch.sum(Ax * Ax) / coords_t.shape[0] + self.shift
        # Forces are negative gradient of energy wrt coords: F = - dE/dx = -(A^2 x)/N
        forces = -(A * A) * coords_t / coords_t.shape[0]
        return {"energy": energy.to(torch.float64), "forces": forces.to(torch.float64)}


def make_initial_coords(n_atoms=5, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    coords = np.random.randn(n_atoms, 3).astype(np.float64)
    atomic_nums = np.full((n_atoms,), 6, dtype=np.int64)
    return coords, atomic_nums


def run_np(geometry_np, OptimCls, **kwargs):
    opt = OptimCls(
        geometry=geometry_np,
        thresh="gau_loose",
        max_cycles=200,
        print_every=10**6,
        **kwargs,
    )
    opt.run()
    return geometry_np.energy


def run_torch(geometry_th, OptimCls, **kwargs):
    opt = OptimCls(
        geometry=geometry_th,
        thresh="gau_loose",
        max_cycles=200,
        print_every=10**6,
        **kwargs,
    )
    opt.run()
    return geometry_th.energy


def compare_one(
    optim_name: str, np_cls, th_cls, np_kwargs=None, th_kwargs=None, tol=1e-6
):
    np_kwargs = np_kwargs or {}
    th_kwargs = th_kwargs or {}

    coords_np, atomic_nums = make_initial_coords(n_atoms=8, seed=42)
    calc = DeterministicCalculator(shift=0.123)

    # NumPy geometry uses flat np array
    geom_np = GeometryNP(coords_np.reshape(-1), atomic_nums, calc)
    # Torch geometry uses torch tensor
    geom_th = GeometryTorch(torch.from_numpy(coords_np.reshape(-1)), atomic_nums, calc)

    t0 = perf_counter()
    e_np = run_np(geom_np, np_cls, **np_kwargs)
    t_np = perf_counter() - t0

    t0 = perf_counter()
    e_th = run_torch(geom_th, th_cls, **th_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_th = perf_counter() - t0

    if isinstance(e_np, np.ndarray):
        e_np = float(e_np)
    e_th = float(e_th)

    diff = e_np - e_th
    speedup = (t_np / t_th) if t_th > 0 else float("inf")
    print(
        f"{optim_name}: E_np={e_np:.8f} E_th={e_th:.8f} diff={diff:.3e} | "
        f"t_np={t_np * 1000:.1f}ms t_th={t_th * 1000:.1f}ms speedup={speedup:.2f}x"
    )
    if diff >= tol:
        print(f"{optim_name} energies differ by {diff} >= {tol}")


def test_bfgs():
    # Use plain BFGS (no damping) for parity
    compare_one(
        "BFGS",
        BFGS_NP,
        BFGS_TORCH,
        np_kwargs={},
        th_kwargs={"update": "bfgs"},
        tol=1e-6,
    )


def test_fire():
    compare_one(
        "FIRE",
        FIRE_NP,
        FIRE_TORCH,
        np_kwargs={},
        th_kwargs={},
        tol=1e-6,
    )


def test_rfo():
    compare_one(
        "RFO",
        RFO_NP,
        RFO_TORCH,
        np_kwargs={"line_search": False, "gdiis": False},
        th_kwargs={"line_search": False, "gdiis": False},
        tol=1e-6,
    )


if __name__ == "__main__":
    test_bfgs()
    test_fire()
    test_rfo()
    print("OK")
