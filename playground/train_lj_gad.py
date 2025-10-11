"""
Score-matching with GAD targets from Lennard-Jones potential (PyTorch).

This script trains a model to predict the GAD (Gradient-adjusted descent) field
for a system of 4 particles interacting via Lennard-Jones potential.

GAD Field Computation: Calculates v = -g + 2(g·v_min)v_min where:
g = forces
v_min = eigenvector with smallest eigenvalue from Hessian

classic LJ potential 4ε[(1/r)^12 - (1/r)^6] for all particle pairs

It:
- Implements Lennard-Jones potential energy
- Uses torch autograd to compute forces (negative gradient) and Hessian
- Forms the GAD vector field: v = -g + 2 (g·v_min) v_min,
  where g = forces and v_min is the eigenvector with smallest eigenvalue
- Trains a simple MLP on flattened coordinates to predict the GAD field
- Integrates the learned field deterministically and with Langevin-like noise
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from tqdm import tqdm
from matplotlib import pyplot as plt

from torch_geometric.data import Batch as TGBatch
from sgad.utils.data_utils import get_atomic_graph


# -----------------------
# Config and directories
# -----------------------
base_dir = "plots/train_lj_gad"
os.makedirs(base_dir, exist_ok=True)


# ---------------------------------------------
# Lennard-Jones potential energy
# ---------------------------------------------
def lennard_jones_energy(
    positions: torch.Tensor, epsilon: float = 1.0, sigma: float = 1.0
) -> torch.Tensor:
    """
    Compute Lennard-Jones potential energy for a batch of configurations.

    Args:
        positions: (B, N, 3) tensor of particle positions

    Returns:
        energy: (B,) tensor of total potential energies
    """
    B, N, _ = positions.shape
    energy = torch.zeros(B, device=positions.device, dtype=positions.dtype)

    # Compute pairwise distances and energies
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[:, i, :] - positions[:, j, :]  # (B, 3)
            r = torch.norm(r_vec, dim=1)  # (B,)

            # LJ potential: 4 * [(1/r)^12 - (1/r)^6]
            sr6 = (1 / r) ** 6
            sr12 = sr6**2
            energy += 4.0 * (sr12 - sr6)

    return energy


# ---------------------------------------------
# Compute forces and Hessian using autograd (for comparison/verification)
# ---------------------------------------------
def compute_forces_hessian_autograd(
    positions: torch.Tensor, epsilon: float = 1.0, sigma: float = 1.0
):
    """
    Compute forces (negative gradients) and Hessian of LJ energy using autograd.
    This is slower but useful for verification.

    Args:
        positions: (B, N, 3) tensor, requires_grad=True
        epsilon, sigma: LJ parameters

    Returns:
        forces: (B, N, 3) tensor
        hessian: (B, 3N, 3N) tensor
    """
    B, N, _ = positions.shape
    D = 3 * N

    # Compute energy
    positions.requires_grad_(True)
    energy = lennard_jones_energy(positions, epsilon, sigma)  # (B,)

    # Compute gradient (negative forces)
    grad = torch.autograd.grad(
        energy.sum(),
        positions,
        create_graph=True,
        retain_graph=True,
    )[0]  # (B, N, 3)

    forces = -grad  # (B, N, 3)

    # Compute Hessian
    grad_flat = grad.reshape(B, D)  # (B, 3N)
    hessian = torch.zeros(B, D, D, device=positions.device, dtype=positions.dtype)

    for i in range(D):
        # Compute second derivatives
        grad2 = torch.autograd.grad(
            grad_flat[:, i].sum(),
            positions,
            retain_graph=True,
            allow_unused=True,
        )[0]  # (B, N, 3)

        if grad2 is not None:
            hessian[:, i, :] = grad2.reshape(B, D)

    return forces, hessian


# ---------------------------------------------
# VECTORIZED/PARALLELIZED versions (much faster!)
# ---------------------------------------------
def lennard_jones_energy_vectorized(
    positions: torch.Tensor, epsilon: float = 1.0, sigma: float = 1.0
) -> torch.Tensor:
    """
    Vectorized computation of LJ energy - computes all pairwise interactions in parallel.

    Args:
        positions: (B, N, 3) tensor of particle positions

    Returns:
        energy: (B,) tensor of total potential energies
    """
    B, N, _ = positions.shape
    device = positions.device

    # Compute all pairwise displacement vectors: (B, N, N, 3)
    # Note: for energy, only distance matters, not direction
    # But for consistency with forces, use r_ij[i,j] = r_j - r_i
    r_ij = positions[:, None, :, :] - positions[:, :, None, :]  # (B, N, N, 3)

    # Compute pairwise distances: (B, N, N)
    r = torch.norm(r_ij, dim=-1)  # (B, N, N)

    # Create mask for upper triangle (i < j) to avoid double counting
    mask = torch.triu(
        torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1
    )  # (N, N)

    # Avoid division by zero for diagonal elements (i == i)
    r = torch.where(mask.unsqueeze(0), r, torch.ones_like(r))

    # LJ potential: 4 * [(1/r)^12 - (1/r)^6]
    r_inv = 1.0 / r  # (B, N, N)
    sr6 = r_inv**6
    sr12 = sr6**2

    # Compute pairwise energies
    pair_energies = 4.0 * (sr12 - sr6)  # (B, N, N)

    # Sum only upper triangle pairs
    energy = (pair_energies * mask.unsqueeze(0)).sum(dim=(1, 2))  # (B,)

    return energy


def compute_forces_hessian_vectorized(
    positions: torch.Tensor, epsilon: float = 1.0, sigma: float = 1.0
):
    """
    Vectorized computation of forces and Hessian - computes all pairwise interactions in parallel.

    Args:
        positions: (B, N, 3) tensor of particle positions

    Returns:
        forces: (B, N, 3) tensor
        hessian: (B, 3N, 3N) tensor
    """
    B, N, _ = positions.shape
    D = 3 * N
    device = positions.device
    dtype = positions.dtype

    # Compute all pairwise displacement vectors: (B, N, N, 3)
    # r_ij[i, j] should point from i to j, so it's positions[j] - positions[i]
    r_ij = positions[:, None, :, :] - positions[:, :, None, :]  # (B, N, N, 3)

    # Compute pairwise distances: (B, N, N)
    r = torch.norm(r_ij, dim=-1, keepdim=True)  # (B, N, N, 1)

    # Create mask for upper triangle (i < j)
    mask = torch.triu(
        torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1
    )  # (N, N)

    # Avoid division by zero - set diagonal to 1 (will be masked out anyway)
    r_safe = torch.where(r > 0, r, torch.ones_like(r))
    r_unit = r_ij / r_safe  # (B, N, N, 3)

    # Compute derivatives
    r_inv = 1.0 / r_safe  # (B, N, N, 1)
    sr6 = r_inv**6
    sr12 = sr6**2

    # First derivative: dV/dr = 24 * [(1/r)^6 - 2(1/r)^12] / r
    dV_dr = 24.0 * r_inv * (sr6 - 2.0 * sr12)  # (B, N, N, 1)

    # Second derivative: d²V/dr² = 24 * [26(1/r)^12 - 7(1/r)^6] / r²
    d2V_dr2 = 24.0 * (r_inv**2) * (26.0 * sr12 - 7.0 * sr6)  # (B, N, N, 1)

    # Apply mask to derivatives (only upper triangle)
    mask_4d = mask.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
    dV_dr_masked = torch.where(mask_4d, dV_dr, torch.zeros_like(dV_dr))
    d2V_dr2 = torch.where(mask_4d, d2V_dr2, torch.zeros_like(d2V_dr2))

    # Compute pairwise forces: f_ij = dV/dr * r_unit
    # r_ij[i,j] points from i to j, so f_ij[i,j] is force on i pushing it away from j
    # Only computed for upper triangle (i < j)
    f_ij_upper = dV_dr_masked * r_unit  # (B, N, N, 3)

    # For forces, we need contributions from both upper and lower triangle
    # f_ij[i,j] = force on i due to j (for i < j)
    # f_ji[j,i] = force on j due to i = -f_ij[i,j] by Newton's 3rd law
    #
    # Create full force matrix using symmetry: f_ji = -f_ij
    f_full = f_ij_upper - f_ij_upper.transpose(1, 2)  # (B, N, N, 3)

    # Force on each particle: sum over all other particles
    forces = f_full.sum(dim=2)  # (B, N, 3) - sum over j dimension

    # Compute Hessian
    # H_ij = (d²V/dr²)(r_ij ⊗ r_ij)/r² + (dV/dr)/r (I - (r_ij ⊗ r_ij)/r²)
    hessian = torch.zeros(B, D, D, device=device, dtype=dtype)

    # Precompute outer products: r_unit ⊗ r_unit
    outer = r_unit.unsqueeze(-1) * r_unit.unsqueeze(-2)  # (B, N, N, 3, 3)

    # Identity matrix
    identity = torch.eye(3, device=device, dtype=dtype).view(
        1, 1, 1, 3, 3
    )  # (1, 1, 1, 3, 3)

    # Hessian blocks for each pair
    # H_blocks = d2V_dr2.unsqueeze(-1) * outer + (dV_dr * r_inv).unsqueeze(-1) * (identity - outer)  # (B, N, N, 3, 3)
    H_blocks = -(
        d2V_dr2.unsqueeze(-1) * outer
        + (dV_dr * r_inv).unsqueeze(-1) * (identity - outer)
    )  # (B, N, N, 3, 3)

    # Fill Hessian matrix
    for i in range(N):
        for j in range(i + 1, N):
            i_start, i_end = 3 * i, 3 * (i + 1)
            j_start, j_end = 3 * j, 3 * (j + 1)

            H_block = H_blocks[:, i, j, :, :]  # (B, 3, 3)

            # Off-diagonal blocks
            hessian[:, i_start:i_end, j_start:j_end] = H_block
            hessian[:, j_start:j_end, i_start:i_end] = H_block

            # Diagonal blocks (accumulate negative contributions)
            hessian[:, i_start:i_end, i_start:i_end] -= H_block
            hessian[:, j_start:j_end, j_start:j_end] -= H_block

    return forces, hessian


# ---------------------------------------------
# GAD construction from forces and Hessian
# ---------------------------------------------
def compute_gad_from_forces_hessian(
    forces: torch.Tensor,  # (B, N, 3)
    hessian: torch.Tensor,  # (B, 3N, 3N)
) -> torch.Tensor:
    """
    Compute GAD vector field v for a batch.

    We use g = forces (negative gradient), and v_min is the eigenvector of the
    Hessian corresponding to the smallest eigenvalue.

    Returns:
        v: (B, 3N) GAD vectors
    """
    B, N, _ = forces.shape
    D = 3 * N

    g_flat = forces.reshape(B, D)  # (B, 3N)

    # Eigendecomposition of Hessian
    eig_vals, eig_vecs = torch.linalg.eigh(hessian)  # (B, 3N), (B, 3N, 3N)
    v_min = eig_vecs[:, :, 0]  # (B, 3N) - eigenvector with smallest eigenvalue

    # GAD formula: v = -g + 2 (g·v_min) v_min
    dots = (g_flat * v_min).sum(dim=1, keepdim=True)  # (B, 1)
    v = -g_flat + 2.0 * dots * v_min  # (B, 3N)

    return v


# --------------
# Torch MLP model
# --------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.silu(h)
        h = self.fc3(h)
        h = F.silu(h)
        h = self.fc4(h)
        return h


# ---------------------------------------
# Unified controller: MLP or EGNN_dynamics
# ---------------------------------------
class UnifiedController(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        atomic_nums: torch.Tensor,
        use_egnn: bool = False,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.use_egnn = use_egnn
        self.atomic_nums = atomic_nums
        input_dim = 3 * num_atoms

        if use_egnn:
            from sgad.components.controllers import EGNN_dynamics

            # Build one-hot encoding for atomic numbers
            unique_z = torch.unique(atomic_nums).tolist()
            n_atom_types = len(unique_z)
            self.model = EGNN_dynamics(
                n_atoms=n_atom_types,
                hidden_nf=hidden_dim,
                n_layers=n_layers,
                uniform=False,
            )
            # Create z_table mapping from atomic number to one-hot index
            self.z_table = {int(z): i for i, z in enumerate(unique_z)}
            self.atom_list = [int(z) for z in atomic_nums.tolist()]
        else:
            self.model = MLP(
                input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim
            )

    def _build_batch_dict(self, coords_flat: torch.Tensor, t: torch.Tensor):
        """Convert flat coords (B, 3N) to batch dict for EGNN_dynamics using get_atomic_graph."""
        B = coords_flat.shape[0]
        N = self.num_atoms
        device = coords_flat.device

        # Reshape to (B, N, 3)
        positions = coords_flat.reshape(B, N, 3)

        # Build a list of Data objects using get_atomic_graph
        data_list = []
        for b in range(B):
            data = get_atomic_graph(
                atom_list=self.atom_list,
                positions=positions[b],
                z_table=self.z_table,
            )
            data_list.append(data)
        # Batch the graphs together
        batched_data = TGBatch.from_data_list(data_list)

        batched_data = batched_data.to(device)

        # Extract edge attrs
        n_edges = batched_data.edge_index.shape[1]
        edge_attrs = torch.zeros(n_edges, 2, dtype=torch.long, device=device)
        node_attrs_tensor = batched_data["node_attrs"]

        return {
            "positions": batched_data.positions,
            "edge_index": batched_data.edge_index,
            "batch": batched_data.batch,
            "node_attrs": node_attrs_tensor,
            "edge_attrs": edge_attrs,
        }

    def forward(self, t: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) time values (unused for now)
            x_flat: (B, 3N) flattened coordinates
        Returns:
            (B, 3N) vector field prediction
        """
        if self.use_egnn:
            batch_dict = self._build_batch_dict(x_flat, t)
            output = self.model(t, batch_dict)  # (B*N, 3)
            return output.reshape(x_flat.shape[0], -1)  # (B, 3N)
        else:
            return self.model(x_flat)


def verify_vectorized_implementations(num_samples: int = 5) -> None:
    """
    Verify that vectorized implementations match autograd-based implementations.
    """
    print("\n" + "=" * 60)
    print("Verifying vectorized implementations...")
    print("=" * 60)

    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 4

    # Generate test positions (ensure they're not too close)
    positions = 1.5 + 1.5 * torch.randn(num_samples, N, 3, device=device)

    # ---- Test Energy ----
    energy_autograd = lennard_jones_energy(positions)
    energy_vec = lennard_jones_energy_vectorized(positions)
    energy_diff = (energy_autograd - energy_vec).abs().max().item()
    energy_rel_diff = (
        ((energy_autograd - energy_vec).abs() / (energy_autograd.abs() + 1e-8))
        .max()
        .item()
    )

    print("\nEnergy comparison:")
    print(f"  Max absolute difference: {energy_diff:.2e}")
    print(f"  Max relative difference: {energy_rel_diff:.2e}")

    # ---- Test Forces and Hessian ----
    forces_autograd, hessian_autograd = compute_forces_hessian_autograd(positions)
    forces_vec, hessian_vec = compute_forces_hessian_vectorized(positions)

    force_diff = (forces_autograd - forces_vec).abs().max().item()
    hessian_diff = (hessian_autograd - hessian_vec).abs().max().item()
    force_rel_diff = (
        ((forces_autograd - forces_vec).abs() / (forces_autograd.abs() + 1e-8))
        .max()
        .item()
    )
    hessian_rel_diff = (
        ((hessian_autograd - hessian_vec).abs() / (hessian_autograd.abs() + 1e-8))
        .max()
        .item()
    )

    print("\nForces comparison:")
    print(f"  Max absolute difference: {force_diff:.2e}")
    print(f"  Max relative difference: {force_rel_diff:.2e}")

    # Debug: print sample values if difference is large
    if force_diff > 1e-4:
        print("\n  WARNING: Large force difference detected!")
    print(f"  Sample forces (autograd)[0,0]: {forces_autograd[0, 0]}")
    print(f"  Sample forces (vec)[0,0]:  {forces_vec[0, 0]}")
    print(f"  Difference: {(forces_autograd[0, 0] - forces_vec[0, 0])}")

    print("\nHessian comparison:")
    print(f"  Max absolute difference: {hessian_diff:.2e}")
    print(f"  Max relative difference: {hessian_rel_diff:.2e}")

    # ---- Timing comparison ----
    import time

    # Warm up
    for _ in range(3):
        _ = compute_forces_hessian_autograd(positions)
        _ = compute_forces_hessian_vectorized(positions)

    # Time autograd version
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for _ in range(10):
        forces_autograd, hessian_autograd = compute_forces_hessian_autograd(positions)
    torch.cuda.synchronize() if device == "cuda" else None
    t_autograd = (time.time() - t0) / 10

    # Time vectorized version
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for _ in range(10):
        forces_vec, hessian_vec = compute_forces_hessian_vectorized(positions)
    torch.cuda.synchronize() if device == "cuda" else None
    t_vec = (time.time() - t0) / 10

    print("\nTiming (avg over 10 runs):")
    print(f"  autograd-based:   {t_autograd * 1000:.3f} ms")
    print(f"  Vectorized:   {t_vec * 1000:.3f} ms")
    print(f"  Speedup:      {t_autograd / t_vec:.2f}x")

    # Verification
    tolerance = 1e-5
    if energy_diff < tolerance and force_diff < tolerance and hessian_diff < tolerance:
        print("\n✓ Vectorized implementations verified successfully!")
    else:
        print("\n✗ Warning: Implementations differ more than expected")
    print("=" * 60 + "\n")


def main(
    load_checkpoint: bool = False,
    use_egnn: bool = True,
    verify_derivatives: bool = True,
    score_clip: float = 1e6,
    initial_lr=1e-3,
    final_lr=5e-5,
    bs=128,
    pos_range=3.0,  # Sample positions in range [-pos_range, pos_range]
    num_iterations=50_000,
    loss_type: str = "l2",  # "l1" or "l2",
    log_every=5_000,
    use_wandb=True,
) -> None:
    # -----------------------------
    # Setup seeds
    # -----------------------------
    seed = 0
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Verify analytical derivatives and vectorized implementations
    if verify_derivatives:
        verify_vectorized_implementations()

    # -----------------------------
    # System setup: 4 particles
    # -----------------------------
    N = 4  # Number of particles
    D = 3 * N  # Coordinate dimension

    # All particles are hydrogen (atomic number 1)
    atomic_nums = torch.ones(N, dtype=torch.int64)

    # -----------------------------
    # Model and optimizer setup
    # -----------------------------
    model_name = "egnn" if use_egnn else "mlp"
    hidden_dim = 64 if use_egnn else 256
    model = UnifiedController(
        num_atoms=N,
        atomic_nums=atomic_nums,
        use_egnn=use_egnn,
        hidden_dim=hidden_dim,
        n_layers=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # -----------------------------
    # Training loop: learn GAD(x)
    # -----------------------------
    loss_plot = np.zeros(num_iterations, dtype=np.float32)

    # Linear learning rate scheduler: decay from initial_lr to final_lr
    def lr_schedule(iteration):
        return (
            final_lr + (initial_lr - final_lr) * (1 - iteration / num_iterations)
        ) / initial_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="train-lj-gad",
            config={
                "model": model_name,
                "n_particles": N,
                "hidden_dim": hidden_dim,
                "n_layers": 4,
                "batch_size": bs,
                "num_iterations": num_iterations,
                "initial_lr": initial_lr,
                "final_lr": final_lr,
                "pos_range": pos_range,
                "score_clip": score_clip,
                "use_egnn": use_egnn,
                "seed": seed,
                "loss_type": loss_type,
            },
        )

    # Optional: load existing checkpoint
    os.makedirs(base_dir, exist_ok=True)
    ckpt_path = os.path.join(base_dir, f"{model_name}.pt")
    if load_checkpoint and os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model checkpoint from {ckpt_path}")
    else:
        print(f"Training for {num_iterations} iterations...")
        for iter_idx in tqdm(range(num_iterations)):
            # Sample random positions
            positions = 2.0 * pos_range * (torch.rand((bs, N, 3), generator=g) - 0.5)
            positions = positions.to(device)

            with torch.no_grad():
                # Compute forces and Hessian using vectorized implementation (fastest!)
                forces, hessian = compute_forces_hessian_vectorized(positions)

                # Compute GAD targets (B, 3N)
                gad_targets = compute_gad_from_forces_hessian(forces, hessian)

                # Clip GAD targets to avoid extreme values
                gad_targets = torch.clamp(gad_targets, min=-score_clip, max=score_clip)

            # Model prediction on flattened coords
            x_flat = positions.reshape(bs, D).detach()
            t_dummy = torch.zeros(bs, device=device)  # Time-independent for now
            pred = model(t_dummy, x_flat)

            # Clip predicted scores
            pred = torch.clamp(pred, min=-score_clip, max=score_clip)

            # Compute loss (L1 or L2)
            if loss_type == "l1":
                loss = (pred - gad_targets.detach()).abs().sum(dim=1).mean()
            elif loss_type == "l2":
                loss = ((pred - gad_targets.detach()) ** 2).sum(dim=1).mean()
            else:
                raise ValueError(
                    f"Invalid loss_type: {loss_type}. Must be 'l1' or 'l2'."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Compute gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0
            )

            optimizer.step()
            scheduler.step()  # Update learning rate

            loss_plot[iter_idx] = float(loss.item())

            # Log to wandb
            if use_wandb:
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "loss": loss.item(),
                        "grad_norm": grad_norm.item(),
                        "learning_rate": current_lr,
                        "iteration": iter_idx,
                    },
                    step=iter_idx,
                )

            if iter_idx % log_every == 0 and iter_idx > 0:
                avg_loss = loss_plot[iter_idx - log_every : iter_idx].mean()
                tqdm.write(
                    f"Iteration {iter_idx}: loss={loss.item():.4f}, avg={avg_loss:.4f}"
                )

        # Save training loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(loss_plot)
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss")
        plt.grid(alpha=0.3)
        fname = os.path.join(base_dir, "loss.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {fname}")
        plt.close()

        # Save trained model checkpoint
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint to {ckpt_path}")

    # ---------------------------------
    # Sampling with learned GAD field
    # ---------------------------------
    print("\nSampling with learned GAD field...")

    # Deterministic integration
    with torch.no_grad():
        dt = 1e-2
        n_steps = 1000
        n_samples = 16
        x_det = torch.zeros(
            (n_samples, n_steps + 1, D), dtype=torch.float32, device=device
        )

        # Random initial coordinates
        x_det[:, 0, :] = (
            2.0 * pos_range * (torch.rand((n_samples, D), device=device) - 0.5)
        )
        t_dummy = torch.zeros(n_samples, device=device)

        for i in tqdm(range(n_steps), desc="Deterministic integration"):
            v = model(t_dummy, x_det[:, i, :])
            v = torch.clamp(v, min=-score_clip, max=score_clip)  # Clip scores
            x_det[:, i + 1, :] = x_det[:, i, :] + dt * v

    # Stochastic integration (Langevin-like)
    with torch.no_grad():
        dt_s = 1e-2
        n_steps_s = 1000
        eta = 0.3
        x_stoch = torch.zeros(
            (n_samples, n_steps_s + 1, D), dtype=torch.float32, device=device
        )
        x_stoch[:, 0, :] = (
            2.0 * pos_range * (torch.rand((n_samples, D), device=device) - 0.5)
        )
        t_dummy = torch.zeros(n_samples, device=device)

        for i in tqdm(range(n_steps_s), desc="Stochastic integration"):
            v = model(t_dummy, x_stoch[:, i, :])
            v = torch.clamp(v, min=-score_clip, max=score_clip)  # Clip scores
            noise = torch.randn((n_samples, D), device=device)
            x_next = x_stoch[:, i, :] + dt_s * v + math.sqrt(2.0 * eta * dt_s) * noise
            x_stoch[:, i + 1, :] = x_next

    # -----------------------------
    # Compute energies along trajectories
    # -----------------------------
    def compute_energies(x_traj: torch.Tensor) -> np.ndarray:
        """Compute LJ energy for each point in trajectory."""
        n_samples, n_steps, _ = x_traj.shape
        energies = np.zeros((n_samples, n_steps))

        for i in range(n_steps):
            pos = x_traj[:, i, :].reshape(n_samples, N, 3)
            e = lennard_jones_energy_vectorized(pos)
            energies[:, i] = e.detach().cpu().numpy()

        return energies

    print("Computing energies...")
    energies_det = compute_energies(x_det)
    energies_stoch = compute_energies(x_stoch)

    # Plot energy trajectories
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(min(5, n_samples)):
        plt.plot(energies_det[i], alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("LJ Energy")
    plt.title("Deterministic Trajectories")
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    for i in range(min(5, n_samples)):
        plt.plot(energies_stoch[i], alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("LJ Energy")
    plt.title("Stochastic Trajectories")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(base_dir, "energy_trajectories.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fname}")
    if use_wandb:
        wandb.log({"energy_trajectories": wandb.Image(fname)})
    plt.close()

    # Plot RMS displacement
    def rmsd_traj(x_traj: torch.Tensor) -> np.ndarray:
        x0 = x_traj[:, 0:1, :]
        diff = x_traj - x0
        rms = torch.sqrt((diff * diff).mean(dim=2))  # (B, T)
        return rms.mean(dim=0).detach().cpu().numpy()

    rms_det = rmsd_traj(x_det)
    rms_stoch = rmsd_traj(x_stoch)

    plt.figure(figsize=(8, 5))
    plt.plot(rms_det, label="Deterministic", linewidth=2)
    plt.plot(rms_stoch, label="Stochastic", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Avg RMS Displacement")
    plt.title("Trajectory Displacement")
    plt.legend()
    plt.grid(alpha=0.3)
    fname = os.path.join(base_dir, "rms_displacement.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fname}")
    if use_wandb:
        wandb.log({"rms_displacement": wandb.Image(fname)})
    plt.close()

    # Final statistics
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"System: {N} particles with Lennard-Jones potential")
    print(f"Coordinate dimension: {D}")
    print("\nDeterministic sampling:")
    print(
        f"  Initial energy (avg): {energies_det[:, 0].mean():.4f} ± {energies_det[:, 0].std():.4f}"
    )
    print(
        f"  Final energy (avg):   {energies_det[:, -1].mean():.4f} ± {energies_det[:, -1].std():.4f}"
    )
    print(f"  Final RMS:            {rms_det[-1]:.4f}")
    print("\nStochastic sampling:")
    print(
        f"  Initial energy (avg): {energies_stoch[:, 0].mean():.4f} ± {energies_stoch[:, 0].std():.4f}"
    )
    print(
        f"  Final energy (avg):   {energies_stoch[:, -1].mean():.4f} ± {energies_stoch[:, -1].std():.4f}"
    )
    print(f"  Final RMS:            {rms_stoch[-1]:.4f}")
    print("=" * 60)

    # Visualize final configurations in 3D
    final_det = x_det[:, -1, :].reshape(n_samples, N, 3).cpu().numpy()
    final_stoch = x_stoch[:, -1, :].reshape(n_samples, N, 3).cpu().numpy()

    # Remove center of mass and compute average distances
    def remove_center_of_mass(positions):
        """Remove center of mass from positions. Shape: (n_samples, N, 3)"""
        com = positions.mean(axis=1, keepdims=True)  # (n_samples, 1, 3)
        return positions - com

    def compute_avg_distance(positions):
        """Compute average pairwise distance for each sample. Shape: (n_samples, N, 3)"""
        n_samples, N, _ = positions.shape
        avg_dists = []
        for i in range(n_samples):
            pos = positions[i]
            dists = []
            for j in range(N):
                for k in range(j + 1, N):
                    dist = np.linalg.norm(pos[j] - pos[k])
                    dists.append(dist)
            avg_dists.append(np.mean(dists))
        return np.array(avg_dists)

    # Apply transformations
    final_det = remove_center_of_mass(final_det)
    final_stoch = remove_center_of_mass(final_stoch)

    # Compute average distances
    avg_dist_det = compute_avg_distance(final_det)
    avg_dist_stoch = compute_avg_distance(final_stoch)

    print("\nAverage interparticle distances (after removing COM):")
    print(f"  Deterministic: {avg_dist_det.mean():.4f} ± {avg_dist_det.std():.4f}")
    print(f"  Stochastic:    {avg_dist_stoch.mean():.4f} ± {avg_dist_stoch.std():.4f}")

    # Log to wandb
    if use_wandb:
        wandb.log(
            {
                "final/avg_distance_det_mean": avg_dist_det.mean(),
                "final/avg_distance_det_std": avg_dist_det.std(),
                "final/avg_distance_stoch_mean": avg_dist_stoch.mean(),
                "final/avg_distance_stoch_std": avg_dist_stoch.std(),
            }
        )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Define colors for different samples
    colors = plt.cm.tab10(np.linspace(0, 1, min(8, n_samples)))

    # Plot deterministic samples
    for i in range(min(8, n_samples)):
        pos = final_det[i]
        color = colors[i]

        # XY projection (top view)
        axes[0, 0].scatter(pos[:, 0], pos[:, 1], s=100, alpha=0.7, color=color)
        axes[0, 0].set_xlabel("X")
        axes[0, 0].set_ylabel("Y")
        axes[0, 0].set_title("XY Projection (Top View)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_aspect("equal")

        # XZ projection (front view)
        axes[0, 1].scatter(pos[:, 0], pos[:, 2], s=100, alpha=0.7, color=color)
        axes[0, 1].set_xlabel("X")
        axes[0, 1].set_ylabel("Z")
        axes[0, 1].set_title("XZ Projection (Front View)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_aspect("equal")

        # YZ projection (side view)
        axes[0, 2].scatter(pos[:, 1], pos[:, 2], s=100, alpha=0.7, color=color)
        axes[0, 2].set_xlabel("Y")
        axes[0, 2].set_ylabel("Z")
        axes[0, 2].set_title("YZ Projection (Side View)")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_aspect("equal")

    # Add row label for deterministic
    fig.text(
        0.02,
        0.75,
        "Deterministic\n(COM removed)",
        rotation=90,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Plot stochastic samples
    for i in range(min(8, n_samples)):
        pos = final_stoch[i]
        color = colors[i]

        # XY projection (top view)
        axes[1, 0].scatter(pos[:, 0], pos[:, 1], s=100, alpha=0.7, color=color)
        axes[1, 0].set_xlabel("X")
        axes[1, 0].set_ylabel("Y")
        axes[1, 0].set_title("XY Projection (Top View)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect("equal")

        # XZ projection (front view)
        axes[1, 1].scatter(pos[:, 0], pos[:, 2], s=100, alpha=0.7, color=color)
        axes[1, 1].set_xlabel("X")
        axes[1, 1].set_ylabel("Z")
        axes[1, 1].set_title("XZ Projection (Front View)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect("equal")

        # YZ projection (side view)
        axes[1, 2].scatter(pos[:, 1], pos[:, 2], s=100, alpha=0.7, color=color)
        axes[1, 2].set_xlabel("Y")
        axes[1, 2].set_ylabel("Z")
        axes[1, 2].set_title("YZ Projection (Side View)")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_aspect("equal")

    # Add row label for stochastic
    fig.text(
        0.02,
        0.25,
        "Stochastic\n(COM removed)",
        rotation=90,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Add overall title with statistics
    fig.suptitle(
        f"Final Configurations\nDeterministic: avg dist={avg_dist_det.mean():.2f}±{avg_dist_det.std():.2f} | "
        f"Stochastic: avg dist={avg_dist_stoch.mean():.2f}±{avg_dist_stoch.std():.2f}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    fname = os.path.join(base_dir, "final_configurations.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {fname}")
    if use_wandb:
        wandb.log({"final_configurations": wandb.Image(fname)})
    plt.close()

    # Log final summary metrics
    if use_wandb:
        wandb.log(
            {
                "final/det_initial_energy_mean": energies_det[:, 0].mean(),
                "final/det_initial_energy_std": energies_det[:, 0].std(),
                "final/det_final_energy_mean": energies_det[:, -1].mean(),
                "final/det_final_energy_std": energies_det[:, -1].std(),
                "final/det_final_rms": rms_det[-1],
                "final/stoch_initial_energy_mean": energies_stoch[:, 0].mean(),
                "final/stoch_initial_energy_std": energies_stoch[:, 0].std(),
                "final/stoch_final_energy_mean": energies_stoch[:, -1].mean(),
                "final/stoch_final_energy_std": energies_stoch[:, -1].std(),
                "final/stoch_final_rms": rms_stoch[-1],
            }
        )

        # Finish wandb run
        wandb.finish()

    print("\nTraining and sampling complete!")


"""
uv run playground/train_lj_gad.py;
uv run playground/train_lj_gad.py --score-clip 1e12;
uv run playground/train_lj_gad.py --final-lr 1e-6;
uv run playground/train_lj_gad.py --loss-type l1;
echo "Done"
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GAD field on Lennard-Jones potential"
    )
    parser.add_argument(
        "--use-egnn", type=bool, default=True, help="Use EGNN instead of MLP"
    )
    parser.add_argument(
        "--score-clip", type=float, default=1e6, help="Clipping value for scores"
    )
    parser.add_argument(
        "--initial-lr", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument(
        "--final-lr", type=float, default=5e-5, help="Final learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--pos-range", type=float, default=3.0, help="Range of positions"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=500_000, help="Number of iterations"
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="l2",
        choices=["l1", "l2"],
        help="Loss function type",
    )
    parser.add_argument(
        "--load-checkpoint", action="store_true", help="Load existing checkpoint"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip derivative verification"
    )

    args = parser.parse_args()

    main(
        load_checkpoint=args.load_checkpoint,
        use_egnn=args.use_egnn,
        verify_derivatives=not args.no_verify,
        score_clip=args.score_clip,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
        bs=args.batch_size,
        pos_range=args.pos_range,
        num_iterations=args.num_iterations,
        loss_type=args.loss_type,
    )
