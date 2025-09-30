#!/usr/bin/env python3
"""
Test script to compare RFO optimizer with and without GDIIS on transition state structures.

This script:
1. Loads 10 transition state structures from transition1x dataset
2. Optimizes each using RFO with GDIIS=True and GDIIS=False
3. Runs for up to 150 steps for each optimization
4. Calculates energy deltas (final - initial) for each case
5. Reports mean and median energy deltas for both configurations
"""

import argparse
import sys
import torch
import numpy as np
import random

# Import required modules
import sgad.utils.t1x_dataloader as t1x
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from sgad.optimizer import Geometry
from sgad.rfo import RFOptimizer


def setup_calculator(checkpoint_path=None, device=None):
    """Set up the calculator for energy and force predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = "/ssd/Code/hip/ckpt/hesspred_v1.ckpt"

    print(f"Using device: {device}")

    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return calculator


def load_test_samples(datapath, n_samples=10):
    """Load test samples from transition1x dataset."""
    print(f"Loading transition1x data from: {datapath}")

    # Create dataloader for validation set
    dataloader = t1x.Dataloader(datapath, datasplit="val", only_final=True)
    all_keys = dataloader.get_all_reaction_keys()
    total_reactions = len(all_keys)

    if total_reactions == 0:
        raise ValueError("No reactions found in dataset")

    # Sample random indices for reproducibility
    random.seed(42)
    sampled_indices = random.sample(
        range(min(total_reactions, 50)), min(n_samples, total_reactions)
    )

    # Create dataloader with sampled indices
    dataloader = t1x.Dataloader(
        datapath, datasplit="val", only_final=True, indices=sampled_indices
    )

    return dataloader


def create_geometry_from_sample(sample, calculator):
    """Create a Geometry object from a transition1x sample."""
    ts_data = sample["transition_state"]
    coords = np.array(ts_data["positions"], dtype=np.float32)
    atomic_nums = np.array(ts_data["atomic_numbers"], dtype=np.int64)

    # Flatten coordinates for optimizer
    coords_flat = coords.flatten()

    geometry = Geometry(coords_flat, atomic_nums, calculator)
    return geometry


def run_rfo_optimization(geometry, gdiis_enabled, max_cycles=150):
    """Run RFO optimization with or without GDIIS."""
    gdiis_label = "WITH GDIIS" if gdiis_enabled else "WITHOUT GDIIS"
    print(f"\n--- Running RFO {gdiis_label} ---")

    # Get initial energy
    initial_energy = geometry.energy
    print(f"Initial energy: {initial_energy:.6f} eV")

    # Create RFO optimizer
    optimizer = RFOptimizer(
        geometry=geometry,
        thresh="never",
        max_cycles=max_cycles,
        print_every=1000,
        gdiis=gdiis_enabled,
    )

    # Run optimization
    optimizer.run()

    # Get final energy
    final_energy = geometry.energy
    print(f"Final energy: {final_energy:.6f} eV")

    # Calculate energy change
    if isinstance(initial_energy, torch.Tensor):
        initial_val = initial_energy.item()
    elif isinstance(initial_energy, np.ndarray):
        initial_val = initial_energy[0]
    else:
        initial_val = float(initial_energy)

    if isinstance(final_energy, torch.Tensor):
        final_val = final_energy.item()
    elif isinstance(final_energy, np.ndarray):
        final_val = final_energy[0]
    else:
        final_val = float(final_energy)

    energy_delta = final_val - initial_val
    print(f"Energy delta: {energy_delta:.6f} eV")

    # Check convergence
    is_converged = optimizer.is_converged
    cycles_taken = optimizer.cur_cycle

    print(f"Converged: {is_converged} after {cycles_taken} cycles")

    return {
        "initial_energy": initial_val,
        "final_energy": final_val,
        "energy_delta": energy_delta,
        "is_converged": is_converged,
        "cycles_taken": cycles_taken,
        "gdiis_enabled": gdiis_enabled,
    }


def test_gdiis_comparison(startkey="transition_state"):
    """Main test function to compare RFO with and without GDIIS."""
    print("=" * 80)
    print("RFO GDIIS COMPARISON TEST")
    print("=" * 80)

    # Setup
    checkpoint_path = "/ssd/Code/hip/ckpt/hesspred_v1.ckpt"
    datapath = "/ssd/Code/Datastore/transition1x.h5"
    n_samples = 10
    max_cycles = 150

    # Initialize calculator
    calculator = setup_calculator(checkpoint_path)

    # Load test samples
    dataloader = load_test_samples(datapath, n_samples)

    # Results storage
    gdiis_results = []
    no_gdiis_results = []

    # Test each sample with both configurations
    for i, sample in enumerate(dataloader):
        sample_name = sample[startkey].get("formula", f"Sample_{i + 1}")
        rxn_name = sample[startkey].get("rxn", f"reaction_{i + 1}")
        n_atoms = len(sample[startkey]["atomic_numbers"])

        print(f"\n{'=' * 60}")
        print(f"SAMPLE {i + 1}/{n_samples}: {sample_name} ({rxn_name})")
        print(f"Number of atoms: {n_atoms}")
        print(f"{'=' * 60}")

        # Test RFO with GDIIS
        geometry_gdiis = create_geometry_from_sample(sample, calculator)
        gdiis_result = run_rfo_optimization(
            geometry_gdiis, gdiis_enabled=True, max_cycles=max_cycles
        )

        if gdiis_result:
            gdiis_result["sample_name"] = sample_name
            gdiis_result["rxn_name"] = rxn_name
            gdiis_result["n_atoms"] = n_atoms
            gdiis_results.append(gdiis_result)

        # Test RFO without GDIIS
        geometry_no_gdiis = create_geometry_from_sample(sample, calculator)
        no_gdiis_result = run_rfo_optimization(
            geometry_no_gdiis, gdiis_enabled=False, max_cycles=max_cycles
        )

        if no_gdiis_result:
            no_gdiis_result["sample_name"] = sample_name
            no_gdiis_result["rxn_name"] = rxn_name
            no_gdiis_result["n_atoms"] = n_atoms
            no_gdiis_results.append(no_gdiis_result)

    # Analyze results
    print(f"\n{'=' * 80}")
    print("GDIIS COMPARISON ANALYSIS")
    print(f"{'=' * 80}")

    if not gdiis_results or not no_gdiis_results:
        print("Insufficient results for analysis.")
        return False

    # Extract energy deltas
    gdiis_deltas = [result["energy_delta"] for result in gdiis_results]
    no_gdiis_deltas = [result["energy_delta"] for result in no_gdiis_results]

    # Calculate statistics
    gdiis_mean = np.mean(gdiis_deltas)
    gdiis_median = np.median(gdiis_deltas)
    no_gdiis_mean = np.mean(no_gdiis_deltas)
    no_gdiis_median = np.median(no_gdiis_deltas)

    # Count successful optimizations
    successful_gdiis = sum(1 for result in gdiis_results if result["is_converged"])
    successful_no_gdiis = sum(
        1 for result in no_gdiis_results if result["is_converged"]
    )

    # Print detailed results
    print("\nDetailed Results:")
    print(
        f"{'Sample':<15} {'GDIIS':<8} {'ΔE (GDIIS)':<15} {'ΔE (No GDIIS)':<15} {'Cycles (GDIIS)':<15} {'Cycles (No GDIIS)':<15} {'Conv (GDIIS)':<12} {'Conv (No GDIIS)':<12}"
    )
    print("-" * 130)

    for i in range(min(len(gdiis_results), len(no_gdiis_results))):
        gdiis_res = gdiis_results[i]
        no_gdiis_res = no_gdiis_results[i]

        gdiis_conv = "✓" if gdiis_res["is_converged"] else "✗"
        no_gdiis_conv = "✓" if no_gdiis_res["is_converged"] else "✗"

        print(
            f"{gdiis_res['sample_name']:<15} {'Yes':<8} {gdiis_res['energy_delta']:<15.6f} "
            f"{no_gdiis_res['energy_delta']:<15.6f} {gdiis_res['cycles_taken']:<15} "
            f"{no_gdiis_res['cycles_taken']:<15} {gdiis_conv:<12} {no_gdiis_conv:<12}"
        )

    print("-" * 130)

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(
        f"{'Configuration':<15} {'Mean ΔE':<15} {'Median ΔE':<15} {'Samples':<10} {'Converged':<10}"
    )
    print("-" * 70)
    print(
        f"{'RFO with GDIIS':<15} {gdiis_mean:<15.6f} {gdiis_median:<15.6f} {len(gdiis_deltas):<10} {successful_gdiis:<10}"
    )
    print(
        f"{'RFO w/o GDIIS':<15} {no_gdiis_mean:<15.6f} {no_gdiis_median:<15.6f} {len(no_gdiis_deltas):<10} {successful_no_gdiis:<10}"
    )

    # Success rate analysis
    gdiis_success_rate = (
        100 * successful_gdiis / len(gdiis_results) if gdiis_results else 0
    )
    no_gdiis_success_rate = (
        100 * successful_no_gdiis / len(no_gdiis_results) if no_gdiis_results else 0
    )

    print("\nCONVERGENCE ANALYSIS:")
    print(
        f"GDIIS success rate: {successful_gdiis}/{len(gdiis_results)} ({gdiis_success_rate:.1f}%)"
    )
    print(
        f"No GDIIS success rate: {successful_no_gdiis}/{len(no_gdiis_results)} ({no_gdiis_success_rate:.1f}%)"
    )

    # Additional analysis
    improvement_count = sum(
        1 for gd, ngd in zip(gdiis_deltas, no_gdiis_deltas) if gd < ngd
    )
    total_pairs = min(len(gdiis_deltas), len(no_gdiis_deltas))

    print("\nENERGY COMPARISON:")
    print(
        f"GDIIS achieved better energy reduction in {improvement_count}/{total_pairs} cases ({100 * improvement_count / total_pairs:.1f}%)"
    )
    print(f"Mean improvement with GDIIS: {no_gdiis_mean - gdiis_mean:.6f} eV")
    print(f"Median improvement with GDIIS: {no_gdiis_median - gdiis_median:.6f} eV")

    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare RFO with and without GDIIS")
    parser.add_argument(
        "--datapath",
        type=str,
        default="/ssd/Code/Datastore/transition1x.h5",
        help="Path to transition1x HDF5 file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/ssd/Code/hip/ckpt/hesspred_v1.ckpt",
        help="Path to HIP model checkpoint",
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of samples to test"
    )
    parser.add_argument(
        "--max_cycles", type=int, default=150, help="Maximum optimization cycles"
    )
    parser.add_argument(
        "--startkey", type=str, default="ts", help="Starting key for the sample"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    startkeymap = {
        "ts": "transition_state",
        "r": "reactant",
        "p": "product",
        "transition_state": "transition_state",
        "reactant": "reactant",
        "product": "product",
    }
    success = test_gdiis_comparison(startkey=startkeymap[args.startkey])

    if success:
        print("\n✅ GDIIS comparison test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ GDIIS comparison test failed!")
        sys.exit(1)
