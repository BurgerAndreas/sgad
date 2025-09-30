#!/usr/bin/env python3
"""
Simple test script to verify the corrected Geometry class works with EquiformerTorchCalculator
"""

import torch
import numpy as np
import sys
import os


from sgad.optimizer import Geometry
from hip.equiformer_torch_calculator import EquiformerTorchCalculator


def test_geometry_class():
    """Test the Geometry class with EquiformerTorchCalculator"""
    print("Testing Geometry class with EquiformerTorchCalculator...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the calculator
    # Note: This would need the actual checkpoint path in a real test
    checkpoint_path = "/ssd/Code/hip/ckpt/hesspred_v1.ckpt"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Creating a dummy calculator for testing interface...")

        # Create a dummy calculator for testing
        class DummyCalculator:
            def predict(self, coords, atomic_nums, do_hessian=False):
                n_atoms = len(atomic_nums)
                return {
                    "energy": torch.tensor(42.0),
                    "forces": torch.randn(n_atoms, 3),
                    "hessian": torch.randn(n_atoms * 3, n_atoms * 3)
                    if do_hessian
                    else None,
                }

        calculator = DummyCalculator()
    else:
        calculator = EquiformerTorchCalculator(
            checkpoint_path=checkpoint_path,
            device=device,
        )

    # Create test molecule (water)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [0.96, 0.0, 0.0],  # H
            [-0.24, 0.93, 0.0],  # H
        ]
    )
    atomic_nums = np.array([8, 1, 1])  # O, H, H

    # Create Geometry object
    geom = Geometry(coords.flatten(), atomic_nums, calculator)

    print(f"Created geometry with {len(atomic_nums)} atoms")
    print(f"Coordinate shape: {geom.coords.shape}")
    print(f"3D coordinates shape: {geom.coords3d().shape}")

    # Test property access
    print("\nTesting property access...")

    # Test energy
    energy = geom.energy
    print(f"Energy: {energy}")
    print(f"Energy type: {type(energy)}")

    # Test forces
    forces = geom.forces
    print(f"Forces shape: {forces.shape}")
    print(f"Forces type: {type(forces)}")

    # Test Cartesian forces
    cart_forces = geom.cart_forces
    print(f"Cart forces shape: {cart_forces.shape}")

    # Test coordinate modification
    print("\nTesting coordinate modification...")
    old_energy = geom.energy

    # Modify coordinates slightly
    new_coords = coords.flatten() + 0.01 * np.random.randn(len(coords.flatten()))
    geom.coords = new_coords

    # This should trigger recalculation
    new_energy = geom.energy
    print(f"Old energy: {old_energy}")
    print(f"New energy: {new_energy}")
    print(
        f"Energy changed: {not torch.equal(old_energy, new_energy) if isinstance(old_energy, torch.Tensor) else old_energy != new_energy}"
    )

    print("\nGeometry class test completed successfully!")
    return True


if __name__ == "__main__":
    test_geometry_class()
