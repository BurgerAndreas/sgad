# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Iterable, List, Tuple, Dict, Any, Callable

import torch

import torch_geometric
from rdkit import Chem
from rdkit.Chem import AllChem

from torch_geometric.data import Batch, Data

from sgad.components.sde import ControlledGraphTorsionSDE, integrate_sde
from sgad.utils.data_utils import cycle, subtract_com_batch


ATOMIC_NUMBERS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
MAX_RDKIT_ITERATIONS = 10


# DEPRECATED: Not used with Transition1x dataset - T1x provides molecular structures directly
# def get_rdkit_conformer_positions(
#     smiles: str, duplicates: int
# ) -> Optional[List[torch.Tensor]]:
#     """
#     Generate up to `duplicates` RDKit 3D conformers for a molecule.

#     Args:
#         smiles: SMILES string of the molecule.
#         duplicates: Number of conformers requested; attempts may exceed this
#             to account for pruning of similar conformers.

#     Returns:
#         List[torch.Tensor] or None: List of conformer position tensors
#         shaped (n_atoms, 3), or None if generation failed or produced fewer
#         than `duplicates` conformers.
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     mol = Chem.AddHs(mol)  # add hydrogen to end
#     if mol is None:
#         print(f"Failed to convert SMILES to RDKit molecule: {smiles}")
#         return None

#     AllChem.EmbedMultipleConfs(
#         mol,
#         numConfs=duplicates + 2,  # try to generate 2 extra mols
#         pruneRmsThresh=-1,  # Remove similar conformers
#         enforceChirality=True,
#     )
#     confs = mol.GetConformers()
#     if len(confs) < duplicates:
#         return None

#     return [torch.from_numpy(conf.GetPositions()) for conf in confs]


# DEPRECATED: Not used with Transition1x dataset - T1x provides molecular structures directly
# @torch.no_grad()
# def sample_from_loader_rdkit(
#     sample_loader: Iterable[Dict[str, Any]],
#     energy_model: Callable[[Batch], Dict[str, torch.Tensor]],
#     sde: Any,
#     n_batches: int,
#     batch_size: int,
#     device: torch.device,
#     duplicates: int = 1,
#     float_dtype: Optional[torch.dtype] = None,
# ) -> List[Batch]:
#     """
#     Build a list of batched graph states seeded with RDKit conformer coordinates.

#     Args:
#         sample_loader: Iterable of per-system dictionaries produced by the
#             data pipeline.
#         energy_model: Unused here; kept for API symmetry with other samplers.
#         sde: SDE object (unused); kept for signature consistency.
#         n_batches: Number of batches to produce.
#         batch_size: Total number of systems per batch (including duplicates).
#         device: Torch device to move the batch onto.
#         duplicates: Number of conformers per unique system in a batch.
#         float_dtype: Optional dtype override for floating tensors.

#     Returns:
#         List[Batch]: Each entry is a `torch_geometric.data.Batch` with fields
#         needed by the model.
#     """
#     batch_list = []
#     loader = iter(cycle(sample_loader))
#     duplicates = min(batch_size, duplicates)
#     n_systems = int(batch_size // duplicates)
#     for i in range(n_batches):
#         graph_state = data_to_graph_batch(
#             loader,
#             duplicates,
#             n_systems,
#             use_rdkit_conformers=True,
#             float_dtype=float_dtype,
#         ).to(device)

#         graph_state["positions"] = subtract_com_batch(
#             graph_state["positions"], graph_state["batch"]
#         )
#         batch_list.append(graph_state)

#     return batch_list


# DEPRECATED: Not used with Transition1x dataset - T1x provides molecular structures directly
# @torch.no_grad()
# def populate_buffer_with_rdkit_guesses(
#     energy_model: Callable[[Batch], Dict[str, torch.Tensor]],
#     sample_loader: Iterable[Dict[str, Any]],
#     sde: Any,
#     n_batches: int,
#     batch_size: int,
#     device: torch.device,
#     duplicates: int = 1,
# ) -> Tuple[List[Batch], List[Dict[str, torch.Tensor]]]:
#     """
#     Populate a replay buffer using RDKit-seeded samples and compute energy gradients.

#     Args:
#         energy_model: Callable mapping a graph state Batch -> dict with 'forces'
#             and 'reg_forces'.
#         sample_loader: Iterable yielding systems to sample from.
#         sde: SDE instance; controls dtype selection for torsion-controlled SDEs.
#         n_batches: Number of batches to generate.
#         batch_size: Batch size (including duplicates).
#         device: Target torch device.
#         duplicates: Conformers per system.

#     Returns:
#         Tuple[List[Batch], List[dict]]: Graph states and corresponding gradient
#         dicts with 'energy_grad' and 'reg_grad'.
#     """
#     if isinstance(sde, ControlledGraphTorsionSDE):
#         float_dtype = torch.float64
#     else:
#         float_dtype = None

#     graph_state_list = sample_from_loader_rdkit(
#         sample_loader,
#         energy_model,
#         sde,
#         n_batches,
#         batch_size,
#         device,
#         duplicates=duplicates,
#         float_dtype=float_dtype,
#     )

#     with torch.enable_grad():
#         grad_list = []
#         for graph_state in graph_state_list:
#             F = energy_model(graph_state)
#             grad_dict = {"energy_grad": -F["forces"], "reg_grad": -F["reg_forces"]}
#             grad_list.append(grad_dict)
#     return graph_state_list, grad_list


def cast_floats_to_dtype(container: Batch, dtype: torch.dtype) -> Batch:
    """
    Cast all floating-point tensors contained in a `torch_geometric.data.Batch`
    to a given dtype.

    Args:
        container: The Batch whose floating tensors will be converted.
        dtype: Target torch dtype (e.g., torch.float64).

    Returns:
        Batch: The same container with converted tensors.
    """
    if isinstance(container, Batch):
        keys = container.keys()
    else:
        raise ValueError("did not recognize Batch type. only use torch_geometric")
    for k in keys:
        v = container[k]
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            container[k] = v.to(dtype=dtype)
    return container


# we want to populate a batch with several samples of the same system.
def data_to_graph_batch(
    loader: Iterable[Dict[str, Any]],
    duplicates: int,
    n_systems: int,
    float_dtype: Optional[torch.dtype] = None,
) -> Batch:
    """
    Assemble a `torch_geometric.data.Batch` containing `n_systems * duplicates` samples.
    Each unique system is duplicated `duplicates` times using the same positions.

    Args:
        loader: Infinite/cycled loader yielding per-system dicts.
        duplicates: Number of duplicates per unique system in the batch.
        n_systems: Number of unique systems to include.
        float_dtype: Optional dtype override for floating tensors.

    Returns:
        Batch: A `torch_geometric.data.Batch` with concatenated samples and metadata.
    """
    data_list = []
    i = 0
    while i < n_systems:
        sys = next(loader)  # contains the graph

        # Use the same positions for all duplicates (T1x provides actual structures)
        base_positions = sys["positions"]

        if float_dtype is not None:
            sys = cast_floats_to_dtype(sys, float_dtype)
            base_positions = base_positions.to(dtype=float_dtype)

        i += 1
        for j in range(duplicates):
            # Use the same positions for all duplicates
            positions = base_positions.clone()

            # copy over graph (fully connected anyway in adjoint sampling)
            data_j = Data(
                edge_index=sys["edge_index"],
                positions=positions,
                shifts=sys["shifts"],
                unit_shifts=sys["unit_shifts"],
                cell=sys["cell"],
                node_attrs=sys["node_attrs"],
                weight=sys["weight"][0],
                head=sys["head"][0],
                energy_weight=sys["energy_weight"][0],
                forces_weight=sys["forces_weight"][0],
                stress_weight=sys["stress_weight"][0],
                virials_weight=sys["virials_weight"][0],
                forces=sys["forces"],
                energy=sys["energy"][0],
                stress=sys["stress"],
                virials=sys["virials"],
                dipole=sys["dipole"],
                charges=sys["charges"],
            )
            data_j["edge_attrs"] = sys["edge_attrs"]
            # No SMILES field for T1x data
            data_list.append(data_j)

    graph_batch_loader = torch_geometric.loader.DataLoader(
        dataset=data_list,
        batch_size=int(n_systems * duplicates),
        shuffle=False,
    )
    return next(iter(graph_batch_loader))


@torch.no_grad()
def sample_from_loader(
    sample_loader: Iterable[Dict[str, Any]],
    sde: Any,
    n_batches: int,
    batch_size: int,
    device: torch.device,
    duplicates: int = 1,
    nfe: int = 1000,
    controlled: bool = True,
    float_dtype: Optional[torch.dtype] = None,
    discretization_scheme: str = "uniform",
) -> List[Batch]:
    """
    Sample batches from a loader and, if `controlled`, integrate an SDE to generate positions.

    Args:
        sample_loader: Iterable of per-system dictionaries.
        sde: SDE to integrate for controlled sampling.
        n_batches: Number of batches to produce.
        batch_size: Total samples in each batch (including duplicates).
        device: Torch device.
        duplicates: Duplicates per unique system.
        nfe: Number of function evaluations for SDE integration.
        controlled: Whether to integrate the SDE; if False, add noise at final time.
        float_dtype: Optional dtype override for floating tensors.
        discretization_scheme: Name of SDE discretization scheme.

    Returns:
        List[Batch]: List of graph states (final SDE state if controlled).
    """
    batch_list = []
    if len(sample_loader) == 0:
        raise ValueError(
            "if sample_loader is empty, `iter(cycle(sample_loader))` will hang"
        )
    loader = iter(cycle(sample_loader))
    duplicates = min(batch_size, duplicates)
    n_systems = int(batch_size // duplicates)
    for i in range(n_batches):
        batch = data_to_graph_batch(
            loader, duplicates, n_systems, float_dtype=float_dtype
        ).to(device)

        if controlled:
            # integrate the SDE to generate the final state
            graph_state = integrate_sde(
                sde=sde,
                graph0=batch,
                num_integration_steps=nfe,
                only_final_state=True,
                discretization_scheme=discretization_scheme,
            )
        else:
            # random
            graph_state = batch
            graph_state["positions"] = subtract_com_batch(
                positions=torch.randn_like(graph_state["positions"])
                * sde.noise_schedule.h(torch.Tensor([1.0]).to(device)),
                batch_index=graph_state["batch"],
            )
        batch_list.append(graph_state)

    return batch_list


@torch.no_grad()
def populate_buffer_with_samples_and_energy_gradients(
    energy_model: Callable[[Batch], Dict[str, torch.Tensor]],
    sample_loader: Iterable[Dict[str, Any]],
    sde: Any,
    n_batches: int,
    batch_size: int,
    device: torch.device,
    duplicates: int = 1,
    nfe: int = 1000,
    controlled: bool = True,
    discretization_scheme: str = "uniform",
) -> Tuple[List[Batch], List[Dict[str, torch.Tensor]]]:
    """
    Populate a replay buffer by sampling batches (optionally via SDE) and computing gradients.

    Args:
        energy_model: Callable mapping graph state Batch -> dict with 'forces' and
            'reg_forces'.
        sample_loader: Iterable yielding systems to sample from.
        sde: SDE instance used for controlled sampling.
        n_batches: Number of batches to generate.
        batch_size: Batch size (including duplicates).
        device: Target torch device.
        duplicates: Duplicates per unique system.
        nfe: Number of function evaluations for SDE integration.
        controlled: Whether to integrate SDE or use noise at final time.
        discretization_scheme: Name of SDE discretization scheme.

    Returns:
        Tuple[List[Batch], List[dict]]: Graph states and corresponding gradient dicts with
        keys 'energy_grad' and 'reg_grad'.
    """
    if isinstance(sde, ControlledGraphTorsionSDE):
        float_dtype = torch.float64
    else:
        float_dtype = None

    graph_state_list = sample_from_loader(
        sample_loader,
        sde,
        n_batches,
        batch_size,
        device,
        duplicates=duplicates,
        nfe=nfe,
        controlled=controlled,
        float_dtype=float_dtype,
        discretization_scheme=discretization_scheme,
    )
    with torch.enable_grad():
        grad_list = []
        for graph_state in graph_state_list:
            F = energy_model(graph_state)
            grad_dict = {"energy_grad": -F["forces"], "reg_grad": -F["reg_forces"]}
            grad_list.append(grad_dict)
    return graph_state_list, grad_list
