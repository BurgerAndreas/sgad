# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from typing import Optional

import torch

import torch_geometric
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.func import grad, vmap

from torch_geometric.data import Batch, Data

from sgad.components.sde import ControlledGraphTorsionSDE, integrate_sde
from sgad.sampletorsion.rotate import grad_pos_E_to_grad_tor_E, set_torsions
from sgad.utils.data_utils import cycle, subtract_com_batch


ATOMIC_NUMBERS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
MAX_RDKIT_ITERATIONS = 10


def get_rdkit_conformer_positions(smiles, duplicates):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # add hydrogen to end
    if mol is None:
        print(f"Failed to convert SMILES to RDKit molecule: {smiles}")
        return None

    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=duplicates + 2,  # try to generate 2 extra mols
        pruneRmsThresh=-1,  # Remove similar conformers
        enforceChirality=True,
    )
    confs = mol.GetConformers()
    if len(confs) < duplicates:
        return None

    return [torch.from_numpy(conf.GetPositions()) for conf in confs]


@torch.no_grad()
def sample_from_loader_rdkit(
    sample_loader,
    energy_model,
    sde,
    n_batches,
    batch_size,
    device,
    duplicates=1,
    float_dtype: Optional[torch.dtype] = None,
):
    batch_list = []
    loader = iter(cycle(sample_loader))
    duplicates = min(batch_size, duplicates)
    n_systems = int(batch_size // duplicates)
    for i in range(n_batches):
        graph_state = data_to_graph_batch(
            loader,
            duplicates,
            n_systems,
            use_rdkit_conformers=True,
            float_dtype=float_dtype,
        ).to(device)

        graph_state["positions"] = subtract_com_batch(
            graph_state["positions"], graph_state["batch"]
        )
        batch_list.append(graph_state)

    return batch_list


@torch.no_grad()
def populate_buffer_from_loader_rdkit(
    energy_model,
    sample_loader,
    sde,
    n_batches,
    batch_size,
    device,
    duplicates=1,
):
    if isinstance(sde, ControlledGraphTorsionSDE):
        float_dtype = torch.float64
    else:
        float_dtype = None

    graph_state_list = sample_from_loader_rdkit(
        sample_loader,
        energy_model,
        sde,
        n_batches,
        batch_size,
        device,
        duplicates=duplicates,
        float_dtype=float_dtype,
    )

    with torch.enable_grad():
        grad_list = []
        for graph_state in graph_state_list:
            F = energy_model(graph_state)
            grad_dict = {"energy_grad": -F["forces"], "reg_grad": -F["reg_forces"]}
            grad_list.append(grad_dict)
    return graph_state_list, grad_list


def cast_floats_to_dtype(container: Batch, dtype: torch.dtype) -> Batch:
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
    loader,
    duplicates,
    n_systems,
    use_rdkit_conformers: bool = False,
    float_dtype: Optional[torch.dtype] = None,
):
    data_list = []
    i = 0
    while i < n_systems:
        sys = next(loader)
        smiles = sys["smiles"][0]
        if use_rdkit_conformers:
            conformer_positions = get_rdkit_conformer_positions(smiles, duplicates)
            if not conformer_positions:
                print(
                    "RDKIT failed to generate conformers, skipping molecule with smiles {}".format(
                        smiles
                    )
                )
        else:
            conformer_positions = [
                torch.zeros_like(sys["positions"]) for i in range(duplicates)
            ]

        if conformer_positions:
            if float_dtype is not None:
                sys = cast_floats_to_dtype(sys, float_dtype)
                conformer_positions = [
                    cp.to(dtype=float_dtype) for cp in conformer_positions
                ]
            i += 1
            for j in range(duplicates):
                positions = conformer_positions[j][: sys["batch"].shape[0]].to(
                    sys["positions"]
                )

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
                data_j["smiles"] = smiles
                data_list.append(data_j)

    graph_batch_loader = torch_geometric.loader.DataLoader(
        dataset=data_list,
        batch_size=int(n_systems * duplicates),
        shuffle=False,
    )
    return next(iter(graph_batch_loader))


@torch.no_grad()
def sample_from_loader(
    sample_loader,
    sde,
    n_batches,
    batch_size,
    device,
    duplicates=1,
    nfe=1000,
    controlled=True,
    float_dtype: Optional[torch.dtype] = None,
    discretization_scheme="uniform",
):
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
            graph_state = integrate_sde(
                sde,
                batch,
                nfe,
                only_final_state=True,
                discretization_scheme=discretization_scheme,
            )
        else:
            graph_state = batch
            graph_state["positions"] = subtract_com_batch(
                torch.randn_like(graph_state["positions"])
                * sde.noise_schedule.h(torch.Tensor([1.0]).to(device)),
                graph_state["batch"],
            )
        batch_list.append(graph_state)

    return batch_list


@torch.no_grad()
def populate_buffer_from_loader(
    energy_model,
    sample_loader,
    sde,
    n_batches,
    batch_size,
    device,
    duplicates=1,
    nfe=1000,
    controlled=True,
    discretization_scheme="uniform",
):
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
