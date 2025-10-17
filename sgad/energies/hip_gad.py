import torch
from typing import Optional
import numpy as np

from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData
# from torch_geometric.loader import DataLoader as TGDataLoader


from hip.inference_utils import get_model_from_checkpoint, get_dataloader
from hip.frequency_analysis import (
    analyze_frequencies_torch,
    eckart_projection_notmw_torch,
    get_trans_rot_projector_torch,
    mass_weigh_hessian_torch,
)
from hip.masses import MASS_DICT

from nets.scatter_utils import scatter_mean

GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8], dtype=torch.long)
GLOBAL_ATOM_SYMBOLS = np.array(["H", "C", "N", "O"])
Z_TO_ATOM_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


def onehot_convert(atomic_numbers, device):
    """
    Convert a list of atomic numbers into an one-hot matrix
    """
    encoder = {
        1: [1, 0, 0, 0, 0],
        6: [0, 1, 0, 0, 0],
        7: [0, 0, 1, 0, 0],
        8: [0, 0, 0, 1, 0],
    }
    onehot = [encoder[i] for i in atomic_numbers]
    return torch.tensor(onehot, dtype=torch.int64, device=device)


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


def coord_atoms_to_torch_geometric(
    coords,  # (N, 3)
    atomic_nums,  # (N,)
):
    """
    Convert ASE Atoms object to torch_geometric Data format expected by Equiformer.
    with_grad=True ensures there are gradients of the energy and forces w.r.t. the positions,
    through the graph generation.

    Args:
        atoms: ASE Atoms object

    Returns:
        Data: torch_geometric Data object with required attributes
    """

    # Convert to torch tensors
    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return TGBatch.from_data_list(
        [data],
        # follow_batch=["diag_ij", "edge_index", "message_idx_ij"]
    )


from hip.equiformer_torch_calculator import EquiformerTorchCalculator


class HIPGADEnergy(torch.nn.Module):
    def __init__(
        self,
        model_ckpt: Optional[str] = None,
        tau: float = 1e-3,
        alpha: float = 1e3,
        device: str = "cpu",
    ):
        super().__init__()
        # if model_ckpt is None:
        #     model_ckpt = hf_hub_download(
        #         repo_id="facebook/sgad", filename="esen_spice.pt"
        #     )
        self.calculator = EquiformerTorchCalculator(
            checkpoint_path=model_ckpt,
            hessian_method="predict",
            device=device,
        )

        self.tau = tau  # temperature
        self.alpha = alpha  # regularization strength
        self.device = device

        # Set atomic numbers for compatibility with existing code
        self.atomic_numbers = torch.tensor([1, 6, 7, 8], dtype=torch.long)

    def __call__(
        self,
        batch,
        coords=None,
        atomic_nums=None,
        with_grad=False,
        samples_same_shape=True,
    ):
        """Either pass batch or coords and atomic_nums"""
        if batch is None:
            assert coords is not None and atomic_nums is not None, (
                "coords and atomic_nums must be provided if batch is not provided"
            )
            batch = coord_atoms_to_torch_geometric(
                coords,
                atomic_nums,
            )
        assert samples_same_shape, "samples_same_shape must be True for GAD prediction"

        output_dict = {}

        batch = batch.to(self.calculator.potential.device)
        batch.pos = remove_mean_batch(batch.pos, batch.batch)

        energy, forces, out = self.calculator.potential.forward(
            batch,
            otf_graph=True,
        )

        B = batch.batch.max() + 1
        N = batch.natoms.max()

        hessian = out["hessian"].detach().reshape(B, 3 * N, 3 * N)
        output_dict["hessian"] = hessian

        output_dict["energy"] = energy.detach()

        # Forces shape: [n_atoms, 3]
        forces = forces.detach().reshape(B, N * 3)

        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        v = eigenvectors[:, 0]  # [B, 3*N]
        assert v.shape == (B, 3 * N), f"v.shape: {v.shape}, expected: (B, 3*N)"
        dotprod = torch.einsum("bi,bi->b", -forces, v)  # [B]
        dotprod = dotprod.unsqueeze(-1)  # [B, 1]
        # -∇V(x) + 2(∇V, v(x))v(x)
        gad = forces + 2 * dotprod * v  # [B, 3*N]
        output_dict["gad"] = gad.reshape(B * N, 3)
        output_dict["forces"] = forces.reshape(B * N, 3)
        output_dict["eigvalprod"] = eigenvalues[:, 0] * eigenvalues[:, 1]  # [B]

        # Apply temperature scaling to forces
        output_dict["energy_grad"] = output_dict["gad"] / self.tau

        return output_dict
