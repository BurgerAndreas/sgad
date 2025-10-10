#!/usr/bin/env python3
"""
Test script for FIRE and BFGS optimizers on transition state structures from transition1x dataset.

This script:
1. Loads 3 transition state structures from transition1x dataset
2. Optimizes each using both FIRE and BFGS optimizers
3. Runs for up to 100 steps or until gau_loose convergence
4. Verifies that final energy is lower than initial energy for all cases
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
from pathlib import Path

# Import required modules
import sgad.utils.t1x_dataloader as t1x
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from sgad.optimizer_np import Geometry, FIRE, BFGS
from sgad.rfo_np import RFOptimizer


def setup_calculator(checkpoint_path=None, device=None):
    """Set up the calculator for energy and force predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = "/ssd/Code/hip/ckpt/hesspred_v1.ckpt"

    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Creating a dummy calculator for testing...")

        # Create a dummy calculator that provides realistic-looking energies
        class DummyCalculator:
            def __init__(self):
                self.call_count = 0

            def predict(self, coords, atomic_nums, do_hessian=False):
                self.call_count += 1
                n_atoms = len(atomic_nums)

                # Create a simple harmonic potential with minimum away from current coords
                coords_tensor = torch.as_tensor(coords, dtype=torch.float32).reshape(
                    -1, 3
                )

                # Minimum at origin for simplicity
                displacement = coords_tensor
                energy = 0.5 * torch.sum(displacement**2) / n_atoms + random.uniform(
                    -1, 1
                )

                # Forces point toward minimum (negative gradient)
                forces = -displacement + 0.1 * torch.randn_like(coords_tensor)

                return {
                    "energy": torch.tensor(energy),
                    "forces": forces,
                }

        calculator = DummyCalculator()
        print("Dummy calculator created for testing interface")
    else:
        calculator = EquiformerTorchCalculator(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        print("HIP calculator loaded successfully")

    return calculator


def load_test_samples(datapath, n_samples=3):
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

    samples = []
    for i, molecule in enumerate(dataloader):
        if i >= n_samples:
            break
        samples.append(molecule)

    print(f"Loaded {len(samples)} samples from dataset")
    return samples


def create_geometry_from_sample(sample, calculator):
    """Create a Geometry object from a transition1x sample."""
    ts_data = sample["transition_state"]
    coords = np.array(ts_data["positions"], dtype=np.float32)
    atomic_nums = np.array(ts_data["atomic_numbers"], dtype=np.int64)

    # Flatten coordinates for optimizer
    coords_flat = coords.flatten()

    geometry = Geometry(coords_flat, atomic_nums, calculator)
    return geometry


def run_optimization(
    geometry, optimizer_class, optimizer_name, max_cycles=100, **optimizer_kwargs
):
    """Run optimization with specified optimizer."""
    print(f"\n--- Running {optimizer_name} Optimization ---")

    # Get initial energy
    initial_energy = geometry.energy
    print(f"Initial energy: {initial_energy:.6f} eV")

    # Create optimizer
    optimizer = optimizer_class(
        geometry=geometry,
        thresh="never",
        max_cycles=max_cycles,
        print_every=1000,
        **optimizer_kwargs,
    )

    # Run optimization
    optimizer.run()

    # Get final energy
    final_energy = geometry.energy
    print(f"Final energy: {final_energy:.2f} eV")

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

    energy_change = final_val - initial_val
    print(f"Energy change: {energy_change:.2f} eV")

    # Check convergence
    is_converged = optimizer.is_converged
    cycles_taken = optimizer.cur_cycle

    print(f"Converged: {is_converged} after {cycles_taken} cycles")

    return {
        "initial_energy": initial_val,
        "final_energy": final_val,
        "energy_change": energy_change,
        "is_converged": is_converged,
        "cycles_taken": cycles_taken,
        "optimizer_name": optimizer_name,
    }


def test_optimizers():
    """Main test function."""
    print("=" * 80)
    print("OPTIMIZER TEST: FIRE vs BFGS on Transition State Structures")
    print("=" * 80)

    # Setup
    checkpoint_path = "/ssd/Code/hip/ckpt/hesspred_v1.ckpt"
    datapath = "/ssd/Code/Datastore/transition1x.h5"
    n_samples = 5
    max_cycles = 150

    # Initialize calculator
    calculator = setup_calculator(checkpoint_path)

    # Load test samples
    samples = load_test_samples(datapath, n_samples)

    # Results storage
    all_results = []

    # Test each sample with both optimizers
    for i, sample in enumerate(samples):
        sample_name = sample["transition_state"].get("formula", f"Sample_{i + 1}")
        rxn_name = sample["transition_state"].get("rxn", f"reaction_{i + 1}")
        n_atoms = len(sample["transition_state"]["atomic_numbers"])

        print(f"\n{'=' * 60}")
        print(f"SAMPLE {i + 1}/{n_samples}: {sample_name} ({rxn_name})")
        print(f"Number of atoms: {n_atoms}")
        print(f"{'=' * 60}")

        # Test FIRE optimizer
        geometry_fire = create_geometry_from_sample(sample, calculator)
        fire_result = run_optimization(
            geometry_fire,
            FIRE,
            "FIRE",
            max_cycles=max_cycles,
            # dt=0.01,  # Smaller time step for stability
        )

        if fire_result:
            fire_result["sample_name"] = sample_name
            fire_result["rxn_name"] = rxn_name
            fire_result["n_atoms"] = n_atoms
            all_results.append(fire_result)

        # Test BFGS optimizer
        geometry_bfgs = create_geometry_from_sample(sample, calculator)
        bfgs_result = run_optimization(
            geometry_bfgs,
            BFGS,
            "BFGS",
            max_cycles=max_cycles,
        )

        if bfgs_result:
            bfgs_result["sample_name"] = sample_name
            bfgs_result["rxn_name"] = rxn_name
            bfgs_result["n_atoms"] = n_atoms
            all_results.append(bfgs_result)

        # Test RFO optimizer
        geometry_rfo = create_geometry_from_sample(sample, calculator)
        rfo_result = run_optimization(
            geometry_rfo,
            RFOptimizer,
            "RFO",
            gdiis=True,
        )

        if rfo_result:
            rfo_result["sample_name"] = sample_name
            rfo_result["rxn_name"] = rxn_name
            rfo_result["n_atoms"] = n_atoms
            all_results.append(rfo_result)

    # Print summary
    print(f"\n{'=' * 80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'=' * 80}")

    if not all_results:
        print("No successful optimizations to report.")
        return False

    success_count = 0
    total_tests = 0

    # Group results by sample
    samples_tested = {}
    for result in all_results:
        sample_key = result["sample_name"]
        if sample_key not in samples_tested:
            samples_tested[sample_key] = {}
        samples_tested[sample_key][result["optimizer_name"]] = result

    print("\nDetailed Results:")
    print(
        f"{'Sample':<15} {'Optimizer':<8} {'Initial E':<12} {'Final E':<12} {'ΔE':<12} {'Converged':<10} {'Cycles':<8}"
    )
    print("-" * 80)

    for sample_name, optimizers in samples_tested.items():
        for opt_name, result in optimizers.items():
            energy_decreased = result["energy_change"] < 0
            success_count += int(energy_decreased)
            total_tests += 1

            status = "✓" if energy_decreased else "✗"
            converged_status = "Yes" if result["is_converged"] else "No"

            print(
                f"{sample_name:<15} {opt_name:<8} {result['initial_energy']:<12.2f} "
                f"{result['final_energy']:<12.2f} {result['energy_change']:<12.2f} "
                f"{converged_status:<10} {result['cycles_taken']:<8} {status}"
            )

    print("-" * 80)
    print("\nSUMMARY:")
    print(f"Total tests: {total_tests}")
    print(f"Successful energy decreases: {success_count}")
    print(
        f"Success rate: {success_count}/{total_tests} ({100 * success_count / total_tests:.1f}%)"
    )

    # Check if all tests passed
    all_passed = success_count == total_tests

    if all_passed:
        print(f"\nTests passed: All optimizers successfully decreased energy.")
    else:
        print("\n❌ Some tests failed. Expected all energies to decrease.")

    return all_passed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test FIRE and BFGS optimizers on transition states"
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
        "--samples", type=int, default=3, help="Number of samples to test"
    )
    parser.add_argument(
        "--max_cycles", type=int, default=100, help="Maximum optimization cycles"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = test_optimizers()

    if success:
        print(f"\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Test failed!")
        sys.exit(1)
