from typing import Optional
import os
import torch

from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData
# from torch_geometric.loader import DataLoader as TGDataLoader

from nets.prediction_utils import compute_extra_props

from hip.hessian_utils import compute_hessian
from hip.inference_utils import get_model_from_checkpoint, get_dataloader
from hip.frequency_analysis import (
    analyze_frequencies_torch,
    # eckart_projection_notmw_torch,
)
from hip.equiformer_torch_calculator import EquiformerTorchCalculator


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # you might need to change this
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "../hip/ckpt/hesspred_v1.ckpt")
    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
    )

    # Example 1: load a dataset file and predict the first batch
    dataset_path = os.path.join(project_root, "../hip/data/sample_100.lmdb")
    dataloader = get_dataloader(
        dataset_path, calculator.potential, batch_size=1, shuffle=False
    )
    batch = next(iter(dataloader))
    results = calculator.predict(batch)
    print("\nExample 1:")
    print(f"  Energy: {results['energy'].shape}")
    print(f"  Forces: {results['forces'].shape}")
    print(f"  Hessian: {results['hessian'].shape}")

    print("\nGAD:")
    gad = calculator.get_gad(batch)
    print(f"  GAD: {gad['gad'].shape}")

    # Example 2: create a random data object with random positions and predict
    n_atoms = 10
    elements = torch.tensor([1, 6, 7, 8])  # H, C, N, O
    pos = torch.randn(n_atoms, 3)  # (N, 3)
    atomic_nums = elements[torch.randint(0, 4, (n_atoms,))]  # (N,)
    results = calculator.predict(coords=pos, atomic_nums=atomic_nums)
    print("\nExample 2:")
    print(f"  Energy: {results['energy'].shape}")
    print(f"  Forces: {results['forces'].shape}")
    print(f"  Hessian: {results['hessian'].shape}")

    print("\nFrequency analysis:")
    hessian = results["hessian"]
    frequency_analysis = analyze_frequencies_torch(hessian, pos, atomic_nums)
    print(f"eigvals: {frequency_analysis['eigvals'].shape}")
    print(f"eigvecs: {frequency_analysis['eigvecs'].shape}")
    print(f"neg_num: {frequency_analysis['neg_num']}")
    print(f"natoms: {frequency_analysis['natoms']}")
