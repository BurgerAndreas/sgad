#!/usr/bin/env python3
"""
Compare Torch vs NumPy optimizers (FIRE, BFGS) on 5 random samples from transition1x
using the HIP torch calculator. Falls back to a deterministic dummy if checkpoint missing.
"""

import os
import random
import copy

import numpy as np
import torch

import sgad.utils.t1x_dataloader as t1x
from hip.equiformer_torch_calculator import EquiformerTorchCalculator  # type: ignore

from sgad.optimizer_np import Geometry as GeometryNP, FIRE as FIRE_NP, BFGS as BFGS_NP
from sgad.rfo_np import RFOptimizer as RFO_NP
from sgad.optimizer_torch import GeometryTorch, FIRE as FIRE_TORCH, BFGS as BFGS_TORCH
from sgad.rfo_torch import RFOptimizer as RFO_TORCH
from time import perf_counter


def setup_calculator(
    checkpoint_path: str | None = None, device: torch.device | None = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is None:
        checkpoint_path = "/ssd/Code/hip/ckpt/hesspred_v1.ckpt"
    return EquiformerTorchCalculator(checkpoint_path=checkpoint_path, device=device)


def to_numpy_geometry(sample, calculator_np, dtype=np.float32):
    data = copy.deepcopy(sample["transition_state"])
    coords = np.asarray(data["pos"], dtype=dtype).reshape(-1)
    atomic_nums = np.asarray(data["atomic_numbers"], dtype=np.int64)
    return GeometryNP(coords, atomic_nums, calculator_np)


def to_torch_geometry(sample, calculator_th, device=None, dtype=torch.float32):
    data = copy.deepcopy(sample["transition_state"])
    coords = torch.as_tensor(data["pos"], dtype=dtype, device=device).reshape(-1)
    atomic_nums = np.asarray(data["atomic_numbers"], dtype=np.int64)
    return GeometryTorch(coords, atomic_nums, calculator_th)


def run_np(geometry_np, OptimCls, **kwargs):
    opt = OptimCls(
        geometry=geometry_np,
        thresh="gau_loose",
        max_cycles=200,
        print_every=10**6,
        **kwargs,
    )
    t0 = perf_counter()
    opt.run()
    t = perf_counter() - t0
    return geometry_np.energy, t


def run_torch(geometry_th, OptimCls, **kwargs):
    opt = OptimCls(
        geometry=geometry_th,
        thresh="gau_loose",
        max_cycles=200,
        print_every=10**6,
        **kwargs,
    )
    t0 = perf_counter()
    opt.run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = perf_counter() - t0
    return geometry_th.energy, t


def compare_on_samples(calculator, samples, tol=1e-3):
    device = calculator.device
    results = []
    for i, sample in enumerate(samples):
        print(f"\n# Sample {i + 1}/{len(samples)}: comparing FIRE, BFGS, RFO")

        # NumPy uses same calculator API
        geom_np_fire = to_numpy_geometry(sample, calculator)
        geom_np_bfgs = to_numpy_geometry(sample, calculator)
        # Torch uses same calculator instance
        geom_th_fire = to_torch_geometry(sample, calculator, device=device)
        geom_th_bfgs = to_torch_geometry(sample, calculator, device=device)
        geom_np_rfo = to_numpy_geometry(sample, calculator)
        geom_th_rfo = to_torch_geometry(sample, calculator, device=device)

        e_np_fire, t_np_fire = run_np(geom_np_fire, FIRE_NP)
        e_th_fire, t_th_fire = run_torch(geom_th_fire, FIRE_TORCH)
        e_np_bfgs, t_np_bfgs = run_np(geom_np_bfgs, BFGS_NP)
        e_th_bfgs, t_th_bfgs = run_torch(geom_th_bfgs, BFGS_TORCH, update="bfgs")
        e_np_rfo, t_np_rfo = run_np(geom_np_rfo, RFO_NP, line_search=False, gdiis=False)
        e_th_rfo, t_th_rfo = run_torch(
            geom_th_rfo, RFO_TORCH, line_search=False, gdiis=False
        )

        e_np_fire = float(e_np_fire)
        e_th_fire = float(e_th_fire)
        e_np_bfgs = float(e_np_bfgs)
        e_th_bfgs = float(e_th_bfgs)
        e_np_rfo = float(e_np_rfo)
        e_th_rfo = float(e_th_rfo)

        fire_diff = e_np_fire - e_th_fire
        bfgs_diff = e_np_bfgs - e_th_bfgs

        speed_fire = (t_np_fire / t_th_fire) if t_th_fire > 0 else float("inf")
        speed_bfgs = (t_np_bfgs / t_th_bfgs) if t_th_bfgs > 0 else float("inf")
        speed_rfo = (t_np_rfo / t_th_rfo) if t_th_rfo > 0 else float("inf")

        print(
            f"  FIRE:  E_np={e_np_fire:.6f} E_th={e_th_fire:.6f} diff={fire_diff:.3e} | "
            f"t_np={t_np_fire * 1000:.1f}ms t_th={t_th_fire * 1000:.1f}ms speedup={speed_fire:.2f}x"
        )
        print(
            f"  BFGS:  E_np={e_np_bfgs:.6f} E_th={e_th_bfgs:.6f} diff={bfgs_diff:.3e} | "
            f"t_np={t_np_bfgs * 1000:.1f}ms t_th={t_th_bfgs * 1000:.1f}ms speedup={speed_bfgs:.2f}x"
        )
        print(
            f"  RFO:   E_np={e_np_rfo:.6f} E_th={e_th_rfo:.6f} diff={abs(e_np_rfo - e_th_rfo):.3e} | "
            f"t_np={t_np_rfo * 1000:.1f}ms t_th={t_th_rfo * 1000:.1f}ms speedup={speed_rfo:.2f}x"
        )

        if fire_diff >= tol:
            print(f"FIRE energies differ by {fire_diff:.2e} >= {tol}")
        if bfgs_diff >= tol:
            print(f"BFGS energies differ by {bfgs_diff:.2e} >= {tol}")
        if abs(e_np_rfo - e_th_rfo) >= tol:
            print(f"RFO energies differ by {abs(e_np_rfo - e_th_rfo):.2e} >= {tol}")

        results.append(
            {
                "fire_np": e_np_fire,
                "fire_th": e_th_fire,
                "bfgs_np": e_np_bfgs,
                "bfgs_th": e_th_bfgs,
                "rfo_np": e_np_rfo,
                "rfo_th": e_th_rfo,
                "t_np_fire": t_np_fire,
                "t_th_fire": t_th_fire,
                "t_np_bfgs": t_np_bfgs,
                "t_th_bfgs": t_th_bfgs,
                "t_np_rfo": t_np_rfo,
                "t_th_rfo": t_th_rfo,
            }
        )
    return results


def test_t1x_torch_vs_np():
    datapath = "/ssd/Code/Datastore/transition1x.h5"
    checkpoint_path = "/ssd/Code/hip/ckpt/hesspred_v1.ckpt"

    calc = setup_calculator(checkpoint_path)

    dl = t1x.Dataloader(datapath, datasplit="val", only_final=True)
    all_keys = dl.get_all_reaction_keys()
    if not all_keys:
        raise RuntimeError("transition1x dataset appears empty")
    random.seed(0)
    indices = random.sample(range(len(all_keys)), min(5, len(all_keys)))
    dl = t1x.Dataloader(datapath, datasplit="val", only_final=True, indices=indices)

    samples = []
    for i, mol in enumerate(dl):
        if i >= 5:
            break
        samples.append(mol)

    _ = compare_on_samples(calc, samples, tol=1e-3)


if __name__ == "__main__":
    test_t1x_torch_vs_np()
    print("OK")
