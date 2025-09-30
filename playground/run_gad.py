#!/usr/bin/env python3
"""
GAD Integration Script for Transition1x Data

This script loads transition1x validation samples and follows the GAD (Gentlest Ascent Dynamics)
vector using Euler integration for a specified number of steps. At the end, it performs
frequency analysis to check if the final state is a transition state.

Usage:
    python run_gad.py --nsamples 10 --nsteps 100 [--datapath path/to/transition1x.h5]
"""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Import required modules
# import transition1x as t1x
import sgad.utils.t1x_dataloader as t1x

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import from example files
from playground.example_hip import EquiformerTorchCalculator
from hip.frequency_analysis import analyze_frequencies_torch
from sgad.utils.graph_utils import coord_atoms_to_torch_geometric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GAD integration on transition1x data"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=5,
        help="Number of validation samples to process",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=1_000,
        help="Number of integration steps to perform",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="../Datastore/transition1x.h5",
        help="Path to transition1x HDF5 file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../hip/ckpt/hesspred_v1.ckpt",
        help="Path to HIP model checkpoint",
    )
    parser.add_argument(
        "--dt", type=float, default=0.001, help="Time step for Euler integration"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detect if not specified)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./plots", help="Directory to save plots"
    )
    parser.add_argument(
        "--startgeom",
        type=str,
        default="ts",
        choices=["ts", "r", "p"],
        help="Starting geometry: ts (transition state), r (reactant), p (product)",
    )
    parser.add_argument(
        "--tresh",
        type=float,
        default=None,
        help="Threshold of GAD magnitude for early stopping",
    )
    parser.add_argument(
        "--eckart",
        type=bool,
        default=False,
        help="Whether to use Eckart projection for Hessian/eigenvector",
    )
    parser.add_argument(
        "--eckartgad",
        action="store_true",
        help="Project GAD into vibrational subspace (Eckart)",
    )
    return parser.parse_args()


def compute_linearity_metric(positions: torch.Tensor) -> float:
    """Return a simple linearity score in [0, 1].

    1.0 means positions lie almost on a line (dominant first singular value),
    lower values indicate more planar or 3D spread.
    """
    pos = positions.reshape(-1, 3).to(dtype=torch.float64)
    pos_centered = pos - pos.mean(dim=0, keepdim=True)
    # Singular values of centered coordinates
    s = torch.linalg.svdvals(pos_centered)
    denom = s.sum()
    if denom.item() == 0.0:
        return 0.0
    return (s[0] / denom).item()


def convert_t1x_to_torch_geo(molecule_data, calculator, device):
    """Convert transition1x molecule data to PyTorch Geometric format."""
    positions = torch.tensor(molecule_data["positions"], dtype=torch.float32)
    atomic_numbers = torch.tensor(molecule_data["atomic_numbers"], dtype=torch.long)

    # Convert to torch geometric batch format
    batch = coord_atoms_to_torch_geometric(
        positions,
        atomic_numbers,
    )
    batch = batch.to(device)
    # Store atomic numbers separately for later use
    batch.atomic_numbers = atomic_numbers.to(device)
    return batch


def euler_integration_gad(
    calculator,
    initial_batch,
    atomic_numbers,
    nsteps,
    dt,
    device,
    tresh,
    eckart,
    eckartgad,
):
    """
    Perform Euler integration following GAD vector.

    GAD equation: dx/dt = -∇V(x) + 2(∇V, v(x))v(x) = F + 2(-F, v(x))v(x)
    where v(x) is the eigenvector with smallest eigenvalue.
    """
    print(f"Starting Euler integration for {nsteps} steps with dt={dt}")

    # Start with initial positions
    current_pos = initial_batch.pos.clone().detach()
    n_atoms = current_pos.shape[0]

    trajectory = [current_pos.cpu().numpy()]
    energies = []
    gad_magnitudes = []

    for step in range(nsteps):
        # Create batch with current positions
        batch = coord_atoms_to_torch_geometric(
            current_pos,
            atomic_numbers,
        ).to(device)

        # Get GAD vector
        results = calculator.get_gad(batch, eckart=eckart, eckartgad=eckartgad)
        gad_vector = results["gad"].reshape(n_atoms, 3)

        # Store energy and GAD magnitude
        energies.append(results["energy"].item())
        gad_magnitude = torch.norm(gad_vector).item()
        gad_magnitudes.append(gad_magnitude)

        if tresh and gad_magnitude < tresh:
            break

        # Euler integration step: x_{n+1} = x_n + dt * dx/dt
        current_pos = current_pos + dt * gad_vector

        # Store trajectory
        trajectory.append(current_pos.cpu().numpy())

        if (step + 1) % 10 == 0:
            print(
                f"  Step {step + 1}/{nsteps}, E: {energies[-1]:.2f} eV, |GAD|: {gad_magnitude:.2f}"
            )

    return current_pos, trajectory, energies, gad_magnitudes


def analyze_final_state(calculator, final_pos, atomic_numbers, device):
    """Analyze the final state to check if it's a transition state."""
    print("\nPerforming frequency analysis on final state...")

    # Create batch for final state
    final_batch = coord_atoms_to_torch_geometric(
        final_pos,
        atomic_numbers,
    ).to(device)

    # Get prediction with Hessian
    results = calculator.predict(final_batch, do_hessian=True)
    hessian = results["hessian"]

    # Perform frequency analysis
    frequency_analysis = analyze_frequencies_torch(hessian, final_pos, atomic_numbers)

    # Check if it's a transition state (exactly one negative frequency)
    neg_freqs = frequency_analysis["neg_num"]
    is_transition_state = neg_freqs == 1

    print(f"  Number of negative frequencies: {neg_freqs}")
    print(f"  Is transition state: {is_transition_state}")
    print(f"  Final energy: {results['energy'].item():.6f} eV")

    # Print some eigenvalues for inspection
    eigenvals = frequency_analysis["eigvals"]
    print(f"  Smallest eigenvalues: {eigenvals[:5].cpu().numpy()}")

    return {
        "is_transition_state": is_transition_state,
        "neg_freqs": neg_freqs,
        "final_energy": results["energy"].item(),
        "eigenvals": eigenvals,
        "frequency_analysis": frequency_analysis,
    }


def plot_trajectories(all_sample_data, output_dir="./plots", name_suffix=""):
    """Plot energy and GAD magnitude trajectories for all samples."""
    os.makedirs(output_dir, exist_ok=True)

    # Set seaborn theme for better aesthetics
    sns.set_theme(style="whitegrid", palette="deep")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot energy trajectories
    for i, sample_data in enumerate(all_sample_data):
        energies = sample_data["energies"]
        steps = range(len(energies))
        start_geom = sample_data.get("start_geom", "ts")
        label = f"Sample {sample_data['sample_id']} ({sample_data['formula']}, start={start_geom})"
        ax1.plot(steps, energies, label=label, alpha=0.8, linewidth=2)

    ax1.set_xlabel("Integration Step")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title("Energy Evolution During GAD Integration")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot GAD magnitude trajectories
    for i, sample_data in enumerate(all_sample_data):
        gad_magnitudes = sample_data["gad_magnitudes"]
        steps = range(len(gad_magnitudes))
        start_geom = sample_data.get("start_geom", "ts")
        label = f"Sample {sample_data['sample_id']} ({sample_data['formula']}, start={start_geom})"
        ax2.plot(steps, gad_magnitudes, label=label, alpha=0.8, linewidth=2)

    ax2.set_xlabel("Integration Step")
    ax2.set_ylabel("GAD Vector Magnitude")
    ax2.set_title("GAD Vector Magnitude During Integration")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")  # Log scale often better for magnitudes

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, f"gad_trajectories_{name_suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nTrajectory plots saved to: {plot_path}")

    # Also create separate plots for better visibility
    plot_energy_trajectories(all_sample_data, output_dir, name_suffix)
    plot_gad_magnitude_trajectories(all_sample_data, output_dir, name_suffix)

    return fig


def plot_energy_trajectories(all_sample_data, output_dir="./plots", name_suffix=""):
    """Plot only energy trajectories."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(12, 6))

    # Define a color cycle for different samples
    colors = plt.cm.tab10(range(len(all_sample_data)))

    for i, sample_data in enumerate(all_sample_data):
        energies = sample_data["energies"]
        steps = range(len(energies))
        start_geom = sample_data.get("start_geom", "ts")
        ts_status = "TS" if sample_data["is_transition_state"] else "NOT-TS"
        label = f"Sample {sample_data['sample_id']} ({sample_data['formula']}, start={start_geom}, {ts_status})"

        # Use unique color per sample, line style indicates TS status
        color = colors[i]
        linestyle = "-" if sample_data["is_transition_state"] else "--"

        plt.plot(
            steps,
            energies,
            label=label,
            alpha=0.8,
            linewidth=2,
            color=color,
            linestyle=linestyle,
        )

    plt.xlabel("Integration Step")
    plt.ylabel("Energy (eV)")
    plt.title(
        "Energy Evolution During GAD Integration\n(Solid line=Final TS, Dashed line=Not TS)"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"energy_trajectories_{name_suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Energy trajectory plot saved to: {plot_path}")
    plt.close()


def plot_gad_magnitude_trajectories(
    all_sample_data, output_dir="./plots", name_suffix=""
):
    """Plot only GAD magnitude trajectories."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(12, 6))

    # Define a color cycle for different samples
    colors = plt.cm.tab10(range(len(all_sample_data)))

    for i, sample_data in enumerate(all_sample_data):
        gad_magnitudes = sample_data["gad_magnitudes"]
        steps = range(len(gad_magnitudes))
        start_geom = sample_data.get("start_geom", "ts")
        ts_status = "TS" if sample_data["is_transition_state"] else "NOT-TS"
        label = f"Sample {sample_data['sample_id']} ({sample_data['formula']}, start={start_geom}, {ts_status})"

        # Use unique color per sample, line style indicates TS status
        color = colors[i]
        linestyle = "-" if sample_data["is_transition_state"] else "--"

        plt.plot(
            steps,
            gad_magnitudes,
            label=label,
            alpha=0.8,
            linewidth=2,
            color=color,
            linestyle=linestyle,
        )

    plt.xlabel("Integration Step")
    plt.ylabel("GAD Vector Magnitude")
    plt.title(
        "GAD Vector Magnitude During Integration\n(Solid line=Final TS, Dashed line=Not TS)"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()

    plot_path = os.path.join(
        output_dir, f"gad_magnitude_trajectories_{name_suffix}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"GAD magnitude trajectory plot saved to: {plot_path}")
    plt.close()


def main():
    """Main function.
    Examples:
    uv run playground/run_gad.py --nsamples 10 --nsteps 100 --startgeom ts --tresh 0.1 --dt 0.0001
    uv run playground/run_gad.py --nsamples 10 --nsteps 100 --startgeom ts --tresh 0.1 --dt 0.0001 --eckart True
    """
    args = parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Initialize calculator
    print(f"Loading HIP model from: {args.checkpoint}")
    calculator = EquiformerTorchCalculator(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    print("HIP model loaded successfully")

    # Load transition1x data and handle random sampling
    print(f"Loading transition1x data from: {args.datapath}")

    # Create temporary dataloader to get all available indices
    temp_dataloader = t1x.Dataloader(args.datapath, datasplit="train", only_final=True)
    all_keys = temp_dataloader.get_all_reaction_keys()
    total_reactions = len(all_keys)

    # Sample indices if requested
    random.seed(0)
    sampled_indices = random.sample(
        range(total_reactions), max(args.nsamples * 2, total_reactions)
    )

    # Create final dataloader with sampled indices
    dataloader = t1x.Dataloader(
        args.datapath, datasplit="val", only_final=True, indices=sampled_indices
    )

    geom_map = {"ts": "transition_state", "r": "reactant", "p": "product"}

    name_suffix = args.startgeom
    if args.eckart:
        name_suffix += "_eckart"
    if args.eckartgad:
        name_suffix += "_eckartgad"

    # Process samples
    effective_subset_size = len(sampled_indices) if sampled_indices else total_reactions
    subset_info = (
        f" from random subset of {effective_subset_size}" if sampled_indices else ""
    )
    print(
        f"\nProcessing {min(args.nsamples, effective_subset_size)} validation samples{subset_info} starting from {args.startgeom}..."
    )
    results_summary = []
    all_sample_data = []  # Store trajectory data for plotting

    for i, molecule in enumerate(dataloader):
        if i >= args.nsamples:
            break
        geom_key = geom_map[args.startgeom]

        print(f"\n=== Sample {i + 1}/{args.nsamples} ===")
        print(f"Reaction: {molecule[geom_key].get('rxn', 'unknown')}")
        print(f"Formula: {molecule[geom_key].get('formula', 'unknown')}")

        # Select starting geometry based on flag
        start_data = molecule[geom_key]
        print(f"Starting from: {geom_key} ({args.startgeom})")

        # Convert to torch geometric format
        initial_batch = convert_t1x_to_torch_geo(start_data, calculator, device)
        print(f"Number of atoms: {initial_batch.pos.shape[0]}")

        # Perform GAD integration
        final_pos, trajectory, energies, gad_magnitudes = euler_integration_gad(
            calculator,
            initial_batch,
            initial_batch.atomic_numbers,
            args.nsteps,
            args.dt,
            device,
            args.tresh,
            args.eckart,
            args.eckartgad,
        )

        # Analyze final state
        analysis = analyze_final_state(
            calculator, final_pos, initial_batch.atomic_numbers, device
        )

        # Store results
        sample_result = {
            "sample_id": i + 1,
            "rxn": molecule.get("rxn", "unknown"),
            "formula": molecule.get("formula", "unknown"),
            "n_atoms": initial_batch.pos.shape[0],
            "initial_energy": energies[0] if energies else None,
            "final_energy": analysis["final_energy"],
            "energy_change": analysis["final_energy"] - energies[0]
            if energies
            else None,
            "is_transition_state": analysis["is_transition_state"],
            "neg_freqs": analysis["neg_freqs"],
            "linearity": compute_linearity_metric(torch.as_tensor(final_pos)),
        }
        results_summary.append(sample_result)

        # Store trajectory data for plotting
        sample_trajectory_data = {
            "sample_id": i + 1,
            "rxn": molecule[geom_key].get("rxn", "unknown"),
            "formula": molecule[geom_key].get("formula", "unknown"),
            "start_geom": args.startgeom,
            "energies": energies,
            "gad_magnitudes": gad_magnitudes,
            "trajectory": trajectory,
            "is_transition_state": analysis["is_transition_state"],
            "neg_freqs": analysis["neg_freqs"],
        }
        all_sample_data.append(sample_trajectory_data)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if results_summary:
        transition_states = sum(1 for r in results_summary if r["is_transition_state"])
        print(f"Total samples processed: {len(results_summary)}")
        print(f"Final states identified as transition states: {transition_states}")
        print(
            f"Transition state success rate: {transition_states / len(results_summary) * 100:.1f}%"
        )

        print("\nDetailed results:")
        for result in results_summary:
            print(
                f"  Sample {result['sample_id']} ({result['formula']}): "
                f"{'TS' if result['is_transition_state'] else 'NOT TS'} "
                f"(neg_freqs={result['neg_freqs']}, "
                f"ΔE={result['energy_change']:.4f} eV)"
            )

        # Create plots if we have data
        if all_sample_data:
            print("\nCreating trajectory plots...")
            plot_trajectories(all_sample_data, args.output_dir, name_suffix=name_suffix)
        # Group-wise summary for TS vs NOT-TS
        print("\nGroup statistics (final geometry):")
        ts_group = [r for r in results_summary if r["is_transition_state"]]
        nts_group = [r for r in results_summary if not r["is_transition_state"]]

        def summarize(group, name):
            if not group:
                print(f"  {name}: n=0")
                return
            n = len(group)
            mean_atoms = sum(r["n_atoms"] for r in group) / n
            mean_lin = sum(r.get("linearity", 0.0) for r in group) / n
            print(
                f"  {name}: n={n}, mean n_atoms={mean_atoms:.2f}, mean linearity={mean_lin:.3f}"
            )

        summarize(ts_group, "TS")
        summarize(nts_group, "NOT-TS")
    else:
        print("No samples were successfully processed.")


if __name__ == "__main__":
    main()
