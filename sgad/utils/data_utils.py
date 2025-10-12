# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Iterable, Dict

from torch_geometric.data import Data


def subtract_com_batch(positions, batch_index):
    K = batch_index.max().item() + 1
    means = torch.zeros(K, positions.shape[1], dtype=positions.dtype).to(
        positions.device
    )
    means.index_reduce_(0, batch_index, positions, reduce="mean", include_self=False)
    return positions - means[batch_index]


# this assumes every system in the same molecule type. This is just for interfacing with standard torch models
def graph_to_vector(positions, n_particles, n_spatial_dim):
    n_systems = int(positions.shape[0] // n_particles)

    batch = positions.reshape(n_systems, n_spatial_dim * n_particles)
    return batch


def vector_to_graph(batch, n_particles, n_spatial_dim):
    n_systems = batch.shape[0]
    return batch.reshape(n_systems * n_particles, n_spatial_dim)


def subtract_com_vector(samples, n_particles, n_dimensions):
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples


class PreBatchedDataset(Dataset):
    def __init__(self, batched_data):
        """
        Args:
            batched_data (list of tuples): Each tuple is a batch, typically (inputs, targets)
        """
        self.batched_data = batched_data

    def __len__(self):
        return len(self.batched_data)

    def __getitem__(self, index):
        # Return the pre-batched data
        return self.batched_data[index]


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class AtomicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)



# construct a molecule graph state with fully connected edge index for diffusion controller
def get_tgdata_with_graph(atom_list, positions, z_table):
    n_atoms = len(atom_list)
    n_edges = n_atoms**2 - n_atoms
    source = torch.zeros(n_edges, dtype=torch.long)
    sink = torch.zeros(n_edges, dtype=torch.long)
    k = 0
    # fully connected graph w/o self connections
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                source[k] = i
                sink[k] = j
                k += 1
    indices = torch.tensor([z_table[z] for z in atom_list])
    one_hot_atomic_species = torch.nn.functional.one_hot(
        indices, num_classes=len(z_table)
    )

    if isinstance(positions, torch.Tensor):
        positions = positions.to(dtype=torch.get_default_dtype())
    elif isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions).to(dtype=torch.get_default_dtype())
    else:
        positions = torch.tensor(positions, dtype=torch.get_default_dtype())

    return Data(
        cell=torch.eye(3) * 50.0,  # was 25.0 by default in MACE
        charges=torch.zeros(n_atoms),
        dipole=torch.zeros(1, 3),
        edge_index=torch.stack([source, sink]),
        energy=0.0,
        energy_weight=0.0,
        forces=torch.zeros(n_atoms, 3),
        forces_weight=0.0,
        head=0,
        node_attrs=one_hot_atomic_species.to(torch.get_default_dtype()),
        positions=positions,
        shifts=torch.zeros(n_edges, 3),
        stress=torch.zeros(1, 3, 3),
        stress_weight=0.0,
        unit_shifts=torch.zeros(n_edges, 3),
        virials=torch.zeros(1, 3, 3),
        virials_weight=0.0,
        weight=1.0,  # not sure what this weight is for
    )


def get_atomic(
    atom_list: Iterable[int], positions: torch.Tensor, z_table: Dict[int, int]
) -> Data:
    atomic_nums = [z_table[z] for z in atom_list]
    return Data(
        pos=torch.as_tensor(positions, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        # not used
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
