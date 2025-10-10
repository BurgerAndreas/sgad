"""
Score-matching with GAD targets from HIP Equiformer (PyTorch).

This script mirrors the structure of `examples/score_matching_gad_torch.py`
but obtains GAD targets from a trained HIP Equiformer torch calculator.

It:
- Loads a single molecule from a HIP LMDB dataset
- Uses the calculator to compute forces (negative energy gradient) and Hessian
- Forms the GAD vector field per sample: v = -g + 2 (g·v_min) v_min,
  where g = forces and v_min is the eigenvector of the Hessian with the
  smallest eigenvalue
- Trains a simple MLP on flattened 3N coordinates to predict the GAD field
- Integrates the learned field deterministically and with Langevin-like noise

Dependencies: torch, torch_geometric, matplotlib, numpy, tqdm, and the HIP
project with a valid checkpoint and dataset accessible via relative paths
like in `playground/example_hip.py`.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from matplotlib import pyplot as plt

from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData
from tqdm import tqdm

from sgad.utils.data_utils import get_atomic_graph

import pymol
from pymol import cmd

# HIP utilities and calculator
from nets.prediction_utils import compute_extra_props
from hip.inference_utils import get_dataloader
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.frequency_analysis import analyze_frequencies_torch


# -----------------------
# Config and directories
# -----------------------
base_dir = "plots/score_matching_gad_hip_torch"
os.makedirs(base_dir, exist_ok=True)


# ---------------------------------------------
# Torch Geometric batching for raw coords/z
# ---------------------------------------------
def batch_coords_atomicnums(coords_b: torch.Tensor, atomic_nums: torch.Tensor) -> TGBatch:
    """
    Build a torch_geometric Batch from a batch of coordinate tensors and a single
    atomic number vector.

    Args:
        coords_b: (B, N, 3) tensor of coordinates
        atomic_nums: (N,) int64 tensor
    Returns:
        TGBatch suitable for the HIP calculator
    """
    B, N, _ = coords_b.shape
    data_list = []
    for b in range(B):
        data = TGData(
            pos=coords_b[b].to(torch.float32).reshape(-1, 3),
            z=atomic_nums.to(torch.int64),
            charges=atomic_nums.to(torch.int64),
            natoms=torch.tensor([N], dtype=torch.int64),
            cell=None,
            pbc=torch.tensor(False, dtype=torch.bool),
        )
        data_list.append(data)
    return TGBatch.from_data_list(data_list)


# ---------------------------------------------
# GAD construction from forces and Hessian
# ---------------------------------------------
def compute_gad_from_forces_hessian(
    forces_flat: torch.Tensor,  # (B*N, 3)
    hessian_flat: torch.Tensor,  # (B*3*N*3*N)
    batch_size: int,
    num_atoms: int,
) -> torch.Tensor:
    """
    Compute GAD vector field v for a batch.

    We use g = forces (shape aligned to 3N), and v_min is the eigenvector of the
    Hessian corresponding to the smallest eigenvalue.

    Returns:
        v: (B, 3N) GAD vectors
    """
    B = batch_size
    N = num_atoms
    # reshape inputs from flattened HIP outputs
    forces = forces_flat.reshape(B, N, 3)
    D = 3 * N
    hessian = hessian_flat.reshape(B, D, D)

    g_flat = forces.reshape(B, D)  # (B, 3N)
    eig_vals, eig_vecs = torch.linalg.eigh(hessian)  # (B, 3N), (B, 3N, 3N)
    v_min = eig_vecs[..., :, 0]  # (B, 3N)
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
        hidden_dim: int = 512,
        n_layers: int = 4,
        potential: torch.nn.Module = None,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.use_egnn = use_egnn
        self.atomic_nums = atomic_nums
        input_dim = 3 * num_atoms
        self.potential = potential
        
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
            self.model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    
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
        
        # overwrite graph
        # self.potential.generate_graph(data=data, otf_graph=True)
        
        # Extract bond types from edge_index for edge_attrs
        # Since get_atomic_graph doesn't include edge_attrs, we create dummy ones
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
            t: (B,) time values
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


def main(
        load_score_ckpt = True,
        use_egnn = True  # Set to True to use EGNN_dynamics instead of MLP
    ) -> None:
    # -----------------------------
    # Setup seeds and HIP calculator
    # -----------------------------
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "../hip/ckpt/hesspred_v1.ckpt")
    dataset_path = os.path.join(project_root, "../hip/data/sample_100.lmdb")

    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
        device=device,
    )

    # Load a single molecule (batch_size=1) to fix N and z for the whole run
    dataloader = get_dataloader(
        dataset_path, calculator.potential, batch_size=1, shuffle=False
    )
    first_batch = next(iter(dataloader))
    first_batch = compute_extra_props(first_batch)
    print("Keys in first batch:", first_batch.keys())
    atomic_nums = first_batch.z.to(torch.int64)  # (N,)
    ref_pos = first_batch.pos.reshape(-1, 3).to(torch.float32)  # (N, 3)
    N = ref_pos.shape[0]

    # -----------------------------
    # Model and optimizer setup
    # -----------------------------
    model_name = "egnn" if use_egnn else "mlp"
    model = UnifiedController(
        num_atoms=N,
        atomic_nums=atomic_nums.cpu(),
        use_egnn=use_egnn,
        hidden_dim=512 if not use_egnn else 64,
        n_layers=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


    # -----------------------------
    # Training loop: learn GAD(x)
    # -----------------------------
    bs = 32
    pos_noise_std = 0.05
    num_iterations = 2_000
    loss_plot = np.zeros(num_iterations, dtype=np.float32)

    # Optional: load existing checkpoint
    score_ckpt_path = os.path.join(base_dir, f"{model_name}.pt")
    if load_score_ckpt and os.path.isfile(score_ckpt_path):
        state_dict = torch.load(score_ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model checkpoint from {score_ckpt_path}")
    
    else:
        for iter_idx in tqdm(range(num_iterations)):
            # Build a batch of noisy positions around the reference geometry
            noise = pos_noise_std * torch.randn((bs, N, 3), generator=g)
            coords_b = (ref_pos.unsqueeze(0) + noise).to(torch.float32).to(device)

            # Construct TG batch for HIP calculator and move to device as needed
            tg_batch = batch_coords_atomicnums(coords_b, atomic_nums)
            tg_batch = tg_batch.to(device)

            # Predict energy derivatives
            results = calculator.predict(tg_batch)
            forces = results["forces"].to(torch.float32)  # (B*N, 3)
            hessian = results["hessian"].to(torch.float32)  # (B*3*N*3*N)

            # Compute GAD targets (B, 3N)
            gad_targets = compute_gad_from_forces_hessian(forces, hessian, bs, N)

            # Model prediction on flattened coords
            x_flat = coords_b.reshape(bs, 3 * N)
            t_dummy = torch.zeros(bs, device=device)  # Time-independent for now
            pred = model(t_dummy, x_flat)
            pred = torch.clamp(pred, min=-10.0, max=10.0)  # Clip score to [-10, 10]

            # MSE loss
            loss = ((pred - gad_targets) ** 2).sum(dim=1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_plot[iter_idx] = float(loss.detach().item())
            
            log_every = 100
            if iter_idx % log_every == 0 and iter_idx > 0:
                tqdm.write(f"{iter_idx} loss: {loss.item():.2f} avg: {loss_plot[iter_idx-log_every:iter_idx].mean():.2f}")

        # Save training loss plot
        plt.figure(figsize=(6, 4))
        plt.plot(loss_plot)
        plt.grid()
        fname = os.path.join(base_dir, "loss.png")
        plt.savefig(fname)
        print(f"Saved plot to {fname}")
        plt.close()

        # Save trained MLP checkpoint
        torch.save(model.state_dict(), score_ckpt_path)
        print(f"Saved model checkpoint to {score_ckpt_path}")

    # ---------------------------------
    # Sampling with learned GAD field
    # ---------------------------------
    # Deterministic integration
    with torch.no_grad():
        dt = 1e-2
        n_steps = 500
        x_det = torch.zeros((bs, n_steps + 1, 3 * N), dtype=torch.float32, device=device)
        # random initial coordinates
        x_det[:, 0, :] = torch.randn((bs, 3 * N), device=device)
        t_dummy = torch.zeros(bs, device=device)
        for i in tqdm(range(n_steps), desc="Deterministic integration"):
            v = model(t_dummy, x_det[:, i, :]).detach()
            v = torch.clamp(v, min=-10.0, max=10.0)  # Clip score to [-10, 10]
            x_det[:, i + 1, :] = x_det[:, i, :] + dt * v

    # Stochastic integration (Langevin-like)
    with torch.no_grad():
        dt_s = 1e-2
        n_steps_s = 500
        eta = 0.5
        x_stoch = torch.zeros((bs, n_steps_s + 1, 3 * N), dtype=torch.float32, device=device)
        x_stoch[:, 0, :] = torch.randn((bs, 3 * N), device=device)
        for i in tqdm(range(n_steps_s), desc="Stochastic integration"):
            v = model(t_dummy, x_stoch[:, i, :]).detach()
            v = torch.clamp(v, min=-10.0, max=10.0)  # Clip score to [-10, 10]
            noise = torch.randn((bs, 3 * N), device=device)
            x_next = x_stoch[:, i, :] + dt_s * v + math.sqrt(2.0 * eta * dt_s) * noise
            x_stoch[:, i + 1, :] = x_next

    # Save simple snapshots (RMS displacement vs step) for quick sanity check
    def rmsd_traj(x_traj: torch.Tensor) -> np.ndarray:
        # x_traj: (B, T, 3N)
        x0 = x_traj[:, 0:1, :]
        diff = x_traj - x0
        rms = torch.sqrt((diff * diff).mean(dim=2))  # (B, T)
        return rms.mean(dim=0).detach().cpu().numpy()

    rms_det = rmsd_traj(x_det)
    rms_stoch = rmsd_traj(x_stoch)

    plt.figure(figsize=(6, 4))
    plt.plot(rms_det, label="deterministic")
    plt.plot(rms_stoch, label="stochastic")
    plt.xlabel("step")
    plt.ylabel("avg RMS displacement (arb)")
    plt.legend()
    plt.grid()
    fname = os.path.join(base_dir, "rms_displacement.png")
    plt.savefig(fname)
    print(f"Saved plot to {fname}")
    plt.close()

    # Final prints
    print("Training completed.")
    print(f"Molecule atoms: N={N}, coord dim={3*N}")
    print(f"Deterministic final RMS: {rms_det[-1]:.4f}")
    print(f"Stochastic final RMS: {rms_stoch[-1]:.4f}")

    # ---------------------------------
    # Frequency analysis on final samples
    # ---------------------------------
    print("\n" + "="*60)
    print("Frequency analysis on sampled structures")
    print("="*60)
    
    # Analyze final deterministic samples
    final_det_coords = x_det[:, -1, :].reshape(bs, N, 3).to(device)
    
    ts_count_det = 0
    min_count_det = 0
    higher_count_det = 0
    
    for b in tqdm(range(bs), desc="Frequency analysis on deterministic samples"):
        try:
            # Predict Hessian one sample at a time
            coords_b1 = final_det_coords[b:b+1, :, :]
            tg_b = batch_coords_atomicnums(coords_b1, atomic_nums).to(device)
            res_b = calculator.predict(tg_b)
            N = atomic_nums.shape[0]
            hessian_b = res_b["hessian"].reshape(3 * N, 3 * N)
            pos_b = final_det_coords[b].cpu()
            freq_analysis = analyze_frequencies_torch(hessian_b.cpu(), pos_b, atomic_nums.cpu())
            neg_num = freq_analysis["neg_num"]
            
            if neg_num == 0:
                min_count_det += 1
            elif neg_num == 1:
                ts_count_det += 1
            else:
                higher_count_det += 1
        except Exception as e:
            print(f"Error for batch {b}: {e}")
            # higher_count_det += 1
    
    print(f"\nDeterministic samples (n={bs}):")
    print(f"  Minima (0 negative eigenvalues):     {min_count_det} ({100*min_count_det/bs:.1f}%)")
    print(f"  Transition states (1 negative):      {ts_count_det} ({100*ts_count_det/bs:.1f}%)")
    print(f"  Higher-order saddles (2+ negative):  {higher_count_det} ({100*higher_count_det/bs:.1f}%)")
    
    
    # Analyze final stochastic samples
    ts_count_stoch = 0
    min_count_stoch = 0
    higher_count_stoch = 0
    
    final_stoch_coords = x_stoch[:, -1, :].reshape(bs, N, 3).to(device)
    tg_batch_stoch = batch_coords_atomicnums(final_stoch_coords, atomic_nums)
    tg_batch_stoch = tg_batch_stoch.to(device)
    for b in tqdm(range(bs), desc="Frequency analysis on stochastic samples"):
        try:
            # Predict Hessian one sample at a time and aggregate
            coords_b1 = final_stoch_coords[b:b+1, :, :]
            tg_b = batch_coords_atomicnums(coords_b1, atomic_nums).to(device)
            res_b = calculator.predict(tg_b)
            N = atomic_nums.shape[0]
            hessian_b = res_b["hessian"].reshape(3 * N, 3 * N)
            pos_b = final_stoch_coords[b].cpu()
            freq_analysis = analyze_frequencies_torch(hessian_b.cpu(), pos_b, atomic_nums.cpu())
            neg_num = freq_analysis["neg_num"]
            
            if neg_num == 0:
                min_count_stoch += 1
            elif neg_num == 1:
                ts_count_stoch += 1
            else:
                higher_count_stoch += 1
        except Exception as e:
            print(f"Error for batch {b}: {e}")
            # higher_count_stoch += 1
    
    print(f"\nStochastic samples (n={bs}):")
    print(f"  Minima (0 negative eigenvalues):     {min_count_stoch} ({100*min_count_stoch/bs:.1f}%)")
    print(f"  Transition states (1 negative):      {ts_count_stoch} ({100*ts_count_stoch/bs:.1f}%)")
    print(f"  Higher-order saddles (2+ negative):  {higher_count_stoch} ({100*higher_count_stoch/bs:.1f}%)")
    print("="*60)

    # ---------------------------------
    # Save structures and render with PyMOL
    # ---------------------------------
    atomic_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
    
    def save_xyz(filename, coords, atomic_nums):
        """Save structure in XYZ format."""
        with open(filename, 'w') as f:
            f.write(f"{len(atomic_nums)}\n")
            f.write("Structure from score matching\n")
            for i, (z, pos) in enumerate(zip(atomic_nums, coords)):
                symbol = atomic_symbols.get(int(z), 'X')
                f.write(f"{symbol}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}\n")
    
    # Save 3 deterministic samples
    print(f"\nSaving structures to {base_dir}/")
    xyz_files_det = []
    for i in range(min(3, bs)):
        coords = final_det_coords[i].cpu().numpy()
        filename = os.path.join(base_dir, f"det_sample_{i}.xyz")
        save_xyz(filename, coords, atomic_nums.cpu().numpy())
        xyz_files_det.append(filename)
        print(f"  Saved {filename}")
    
    # Save 3 stochastic samples
    xyz_files_stoch = []
    for i in range(min(3, bs)):
        coords = final_stoch_coords[i].cpu().numpy()
        filename = os.path.join(base_dir, f"stoch_sample_{i}.xyz")
        save_xyz(filename, coords, atomic_nums.cpu().numpy())
        xyz_files_stoch.append(filename)
        print(f"  Saved {filename}")
    
    # Launch PyMOL in headless mode and render structures
    print("\nRendering structures with PyMOL...")
    pymol.finish_launching(["pymol", "-c"])
    
    def setup_pymol():
        """Configure PyMOL settings for nice rendering."""
        cmd.set("ray_opaque_background", 1)
        cmd.set("opaque_background", 1)
        cmd.bg_color("white")
        cmd.set("antialias", 2)
        cmd.set("orthoscopic", 0)
        cmd.set("connect_mode", 4)
        cmd.set("ray_shadows", 0)
        cmd.set("specular", 0)
        cmd.set("depth_cue", 0)
        
        # Pastel element colors
        cmd.set_color("pastel_H", [240, 240, 240])
        cmd.set_color("pastel_C", [0, 142, 109])
        cmd.set_color("pastel_N", [0, 107, 247])
        cmd.set_color("pastel_O", [217, 2, 0])
    
    def render_structure(xyz_file, output_png, obj_name):
        """Load and render a single structure."""
        cmd.load(xyz_file, obj_name)
        cmd.hide("everything", obj_name)
        cmd.show("sticks", obj_name)
        cmd.show("spheres", obj_name)
        cmd.set("sphere_scale", 0.25, obj_name)
        cmd.set("stick_radius", 0.15, obj_name)
        cmd.color("pastel_H", f"{obj_name} and elem H")
        cmd.color("pastel_C", f"{obj_name} and elem C")
        cmd.color("pastel_N", f"{obj_name} and elem N")
        cmd.color("pastel_O", f"{obj_name} and elem O")
        cmd.zoom(obj_name, buffer=0.5, complete=1)
        cmd.png(output_png, width=800, height=800, dpi=300, ray=1)
    
    # Render individual structures
    setup_pymol()
    
    for i, xyz_file in enumerate(xyz_files_det):
        cmd.reinitialize()
        setup_pymol()
        output_png = os.path.join(base_dir, f"det_sample_{i}.png")
        render_structure(xyz_file, output_png, f"det_{i}")
        print(f"  Rendered {output_png}")
    
    for i, xyz_file in enumerate(xyz_files_stoch):
        cmd.reinitialize()
        setup_pymol()
        output_png = os.path.join(base_dir, f"stoch_sample_{i}.png")
        render_structure(xyz_file, output_png, f"stoch_{i}")
        print(f"  Rendered {output_png}")
    
    # Create a grid view with all structures
    cmd.reinitialize()
    setup_pymol()
    
    # Load all structures with spatial offsets
    for i, xyz_file in enumerate(xyz_files_det):
        obj_name = f"det_{i}"
        cmd.load(xyz_file, obj_name)
        cmd.hide("everything", obj_name)
        cmd.show("sticks", obj_name)
        cmd.show("spheres", obj_name)
        cmd.set("sphere_scale", 0.25, obj_name)
        cmd.set("stick_radius", 0.15, obj_name)
        cmd.color("pastel_C", f"{obj_name} and elem C")
        cmd.color("pastel_H", f"{obj_name} and elem H")
        cmd.color("pastel_N", f"{obj_name} and elem N")
        cmd.color("pastel_O", f"{obj_name} and elem O")
        cmd.translate([i * 10, 0, 0], obj_name)
    
    for i, xyz_file in enumerate(xyz_files_stoch):
        obj_name = f"stoch_{i}"
        cmd.load(xyz_file, obj_name)
        cmd.hide("everything", obj_name)
        cmd.show("sticks", obj_name)
        cmd.show("spheres", obj_name)
        cmd.set("sphere_scale", 0.25, obj_name)
        cmd.set("stick_radius", 0.15, obj_name)
        cmd.color("pastel_C", f"{obj_name} and elem C")
        cmd.color("pastel_H", f"{obj_name} and elem H")
        cmd.color("pastel_N", f"{obj_name} and elem N")
        cmd.color("pastel_O", f"{obj_name} and elem O")
        cmd.translate([i * 10, -10, 0], obj_name)
    
    cmd.zoom("all", buffer=1.0, complete=1)
    grid_png = os.path.join(base_dir, "all_structures_grid.png")
    cmd.png(grid_png, width=2400, height=1600, dpi=300, ray=1)
    print(f"  Rendered grid view: {grid_png}")
    
    print("\nPyMOL rendering complete!")


if __name__ == "__main__":
    main()