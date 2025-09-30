#!/usr/bin/env python3
"""
Test script to compare RFO optimizer with and without adapt_step_func on transition state structures.

This script:
1. Loads 10 transition state structures from transition1x dataset
2. Optimizes each using RFO with adapt_step_func=True and adapt_step_func=False
3. Runs for up to 150 steps for each optimization
4. Calculates energy deltas (final - initial) for each case
5. Reports mean and median energy deltas and convergence rates for both configurations

adapt_step_func allows the optimizer to adaptively switch between:
- Newton steps (when gradient is small and Hessian is positive definite)
- Standard RFO steps (otherwise)
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


def run_rfo_optimization(geometry, adapt_step_func_enabled, max_cycles=150):
    """Run RFO optimization with or without adapt_step_func."""
    adapt_label = (
        "WITH ADAPT_STEP_FUNC" if adapt_step_func_enabled else "WITHOUT ADAPT_STEP_FUNC"
    )
    print(f"\n--- Running RFO {adapt_label} ---")

    # Get initial energy
    initial_energy = geometry.energy
    print(f"Initial energy: {initial_energy:.6f} eV")

    # Create RFO optimizer
    optimizer = RFOptimizer(
        geometry=geometry,
        thresh="never",
        max_cycles=max_cycles,
        print_every=1000,
        adapt_step_func=adapt_step_func_enabled,
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
        "adapt_step_func_enabled": adapt_step_func_enabled,
    }


def test_adapt_step_func_comparison(startkey="transition_state"):
    """Main test function to compare RFO with and without adapt_step_func."""
    print("=" * 80)
    print("RFO ADAPT_STEP_FUNC COMPARISON TEST")
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
    adapt_results = []
    no_adapt_results = []

    # Test each sample with both configurations
    for i, sample in enumerate(dataloader):
        sample_name = sample[startkey].get("formula", f"Sample_{i + 1}")
        rxn_name = sample[startkey].get("rxn", f"reaction_{i + 1}")
        n_atoms = len(sample[startkey]["atomic_numbers"])

        print(f"\n{'=' * 60}")
        print(f"SAMPLE {i + 1}/{n_samples}: {sample_name} ({rxn_name})")
        print(f"Number of atoms: {n_atoms}")
        print(f"{'=' * 60}")

        # Test RFO with adapt_step_func
        geometry_adapt = create_geometry_from_sample(sample, calculator)
        adapt_result = run_rfo_optimization(
            geometry_adapt, adapt_step_func_enabled=True, max_cycles=max_cycles
        )

        if adapt_result:
            adapt_result["sample_name"] = sample_name
            adapt_result["rxn_name"] = rxn_name
            adapt_result["n_atoms"] = n_atoms
            adapt_results.append(adapt_result)

        # Test RFO without adapt_step_func
        geometry_no_adapt = create_geometry_from_sample(sample, calculator)
        no_adapt_result = run_rfo_optimization(
            geometry_no_adapt, adapt_step_func_enabled=False, max_cycles=max_cycles
        )

        if no_adapt_result:
            no_adapt_result["sample_name"] = sample_name
            no_adapt_result["rxn_name"] = rxn_name
            no_adapt_result["n_atoms"] = n_atoms
            no_adapt_results.append(no_adapt_result)

    # Analyze results
    print(f"\n{'=' * 80}")
    print("ADAPT_STEP_FUNC COMPARISON ANALYSIS")
    print(f"{'=' * 80}")

    if not adapt_results or not no_adapt_results:
        print("Insufficient results for analysis.")
        return False

    # Extract energy deltas
    adapt_deltas = [result["energy_delta"] for result in adapt_results]
    no_adapt_deltas = [result["energy_delta"] for result in no_adapt_results]

    # Calculate statistics
    adapt_mean = np.mean(adapt_deltas)
    adapt_median = np.median(adapt_deltas)
    no_adapt_mean = np.mean(no_adapt_deltas)
    no_adapt_median = np.median(no_adapt_deltas)

    # Count successful optimizations
    successful_adapt = sum(1 for result in adapt_results if result["is_converged"])
    successful_no_adapt = sum(
        1 for result in no_adapt_results if result["is_converged"]
    )

    # Print detailed results
    print("\nDetailed Results:")
    print(
        f"{'Sample':<15} {'AdaptStep':<10} {'ΔE (Adapt)':<15} {'ΔE (No Adapt)':<15} {'Cycles (Adapt)':<15} {'Cycles (No Adapt)':<15} {'Conv (Adapt)':<12} {'Conv (No Adapt)':<12}"
    )
    print("-" * 130)

    for i in range(min(len(adapt_results), len(no_adapt_results))):
        adapt_res = adapt_results[i]
        no_adapt_res = no_adapt_results[i]

        adapt_conv = "✓" if adapt_res["is_converged"] else "✗"
        no_adapt_conv = "✓" if no_adapt_res["is_converged"] else "✗"

        print(
            f"{adapt_res['sample_name']:<15} {'Yes':<10} {adapt_res['energy_delta']:<15.6f} "
            f"{no_adapt_res['energy_delta']:<15.6f} {adapt_res['cycles_taken']:<15} "
            f"{no_adapt_res['cycles_taken']:<15} {adapt_conv:<12} {no_adapt_conv:<12}"
        )

    print("-" * 130)

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(
        f"{'Configuration':<20} {'Mean ΔE':<15} {'Median ΔE':<15} {'Samples':<10} {'Converged':<10}"
    )
    print("-" * 75)
    print(
        f"{'RFO with AdaptStep':<20} {adapt_mean:<15.6f} {adapt_median:<15.6f} {len(adapt_deltas):<10} {successful_adapt:<10}"
    )
    print(
        f"{'RFO w/o AdaptStep':<20} {no_adapt_mean:<15.6f} {no_adapt_median:<15.6f} {len(no_adapt_deltas):<10} {successful_no_adapt:<10}"
    )

    # Success rate analysis
    adapt_success_rate = (
        100 * successful_adapt / len(adapt_results) if adapt_results else 0
    )
    no_adapt_success_rate = (
        100 * successful_no_adapt / len(no_adapt_results) if no_adapt_results else 0
    )

    # Additional analysis
    improvement_count = sum(
        1 for ad, nad in zip(adapt_deltas, no_adapt_deltas) if ad < nad
    )
    total_pairs = min(len(adapt_deltas), len(no_adapt_deltas))

    print("\nENERGY COMPARISON:")
    print(
        f"AdaptStep achieved better energy reduction in {improvement_count}/{total_pairs} cases ({100 * improvement_count / total_pairs:.1f}%)"
    )
    print(f"Mean improvement with AdaptStep: {no_adapt_mean - adapt_mean:.6f} eV")
    print(f"Median improvement with AdaptStep: {no_adapt_median - adapt_median:.6f} eV")

    # Performance insights
    print("\nPERFORMACE INSIGHTS:")
    print("AdaptStep allows switching between Newton steps (when gradient is small")
    print("and Hessian is positive definite) and RFO steps (otherwise).")
    print(
        "This may improve convergence near minima where Newton steps are more efficient."
    )

    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare RFO with and without adapt_step_func"
    )
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
    success = test_adapt_step_func_comparison(startkey=startkeymap[args.startkey])

    if success:
        print("\n✅ AdaptStep comparison test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ AdaptStep comparison test failed!")
        sys.exit(1)
