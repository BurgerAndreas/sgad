# Copyright (c) Meta Platforms, Inc. and affiliates.

from functools import partial
from typing import Optional

import networkx as nx

import torch

from torch import BoolTensor, LongTensor, Tensor
from torch.func import vjp

from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask, to_networkx

from sgad.sampletorsion.py3d import axis_angle_to_matrix

from sgad.sampletorsion.torsion import check_torsions, dihedral


def get_index_to_rotate(
    tor_index: LongTensor,  # noqa: F722, F821
    edge_index: LongTensor,  # noqa: F722, F821
) -> Optional[LongTensor]:  # noqa: F722, F821
    """returns indexes of connected components after splitting bond `tor_index[1:3]`
    uses RDKit index convention.

    ...after breaking the bond between tor_index[1] and tor_index[2]
    `ind` included implies connected to `tor_index[3]` and `[2]`
    `ind` NOT included implies connected to `tor_index[0]` and `[1]`

    we intend to rotate at `origin` == `tor_index[1]`
    we intend to rotate all atoms connected to `tor_index[2]`
    """
    data = Data(edge_index=edge_index)
    out = []
    for n_torsion, tor_ind in enumerate(tor_index.T):
        graph = to_networkx(data, to_undirected=False)
        ugraph = graph.to_undirected()
        ugraph.remove_edge(*tor_ind[1:3].tolist())
        assert not nx.is_connected(ugraph)
        s1, s2 = tuple(nx.connected_components(ugraph))
        assert s1.intersection(s2) == set()
        if tor_ind[2].item() in s1:
            inds = torch.tensor(list(s1))
        else:
            inds = torch.tensor(list(s2))
        ntor = torch.tensor([n_torsion] * len(inds))
        out.append(torch.stack([ntor, inds], dim=0))
    if len(out) == 0:
        raise ValueError("no torsions")
    else:
        return torch.cat(out, dim=-1)


@torch.compile
def set_torsion_on_multiple_molecules(
    delta: Tensor,  # noqa: F722, F821
    tor_index: LongTensor,  # noqa: F722, F821
    mask_to_rotate: BoolTensor,  # noqa: F722, F821
    positions: Tensor,  # noqa: F722, F821
    n_to_rot: BoolTensor,  # noqa: F722, F821
) -> Tensor:  # noqa: F722, F821
    """rotate one torsion per molecule.

    returns subset of positions
    """
    rot_axes_begin = positions[tor_index[1]]
    rot_axes_end = positions[tor_index[2]]
    rot_axes = rot_axes_end - rot_axes_begin

    # only rotate the delta so we get the target angle
    rot_per_mol = axis_angle_to_matrix(
        delta[:, None]
        * rot_axes
        / torch.linalg.vector_norm(rot_axes, dim=-1, keepdim=True)
    )
    rots = torch.repeat_interleave(rot_per_mol, n_to_rot, dim=0)
    rab = torch.repeat_interleave(rot_axes_begin, n_to_rot, dim=0)

    pos = positions[mask_to_rotate] - rab
    pos = torch.einsum("bij,bj->bi", rots, pos)
    positions[mask_to_rotate] = pos + rab
    return positions


@torch.compile
def set_torsions(
    torsions: Tensor,  # noqa: F722, F821
    tor_index: LongTensor,  # noqa: F722, F821
    index_to_rotate: LongTensor,  # noqa: F722, F821
    positions: Tensor,  # noqa: F722, F821
    max_n_torsions: int,  # noqa: F722, F821
    tor_per_mol_label: LongTensor,  # noqa: F722, F821
) -> Tensor:  # noqa: F722, F821
    """uses rdkit rotation and angle conventions"""
    positions = positions.clone()
    # only rotate the delta so we get the target angle
    delta = torsions - dihedral(positions[tor_index])
    uniq_ind_to_rot, count_ind_to_rot = index_to_rotate[0, :].unique(return_counts=True)
    for n_tor in range(max_n_torsions):
        # select the torsions per molecule with index n_tor
        mask_inds_rotate_with_n_tor = n_tor == tor_per_mol_label
        delt = delta[mask_inds_rotate_with_n_tor]
        ti = tor_index.T[mask_inds_rotate_with_n_tor].T

        # calc repeats of the rot matrix for the number of atoms that rotate
        n_to_rot = count_ind_to_rot[mask_inds_rotate_with_n_tor]

        # identify the atoms that will rotate
        mask_index_to_rotate = torch.isin(
            index_to_rotate[0], uniq_ind_to_rot[mask_inds_rotate_with_n_tor]
        )
        inds_to_rot = index_to_rotate[1][mask_index_to_rotate]
        mask_to_rot = index_to_mask(inds_to_rot, positions.shape[0])

        # do the rotation
        positions = set_torsion_on_multiple_molecules(
            delt,
            ti,
            mask_to_rot,
            positions,
            n_to_rot,
        )
    return positions


def grad_pos_E_to_grad_tor_E(
    grad_pos_E: Tensor,  # noqa: F722, F821
    torsions: Tensor,  # noqa: F722, F821
    tor_index: LongTensor,  # noqa: F722, F821
    index_to_rotate: LongTensor,  # noqa: F722, F821
    positions: Tensor,  # noqa: F722, F821
    max_n_torsions: LongTensor,  # noqa: F722, F821
    tor_per_mol_label: LongTensor,  # noqa: F722, F821
    check: bool = True,
    retain_graph: bool = False,
) -> Tensor:  # noqa: F722, F821
    check_torsions(positions, tor_index, torsions)
    # project forces?
    to_coords = partial(
        set_torsions,
        tor_index=tor_index,
        index_to_rotate=index_to_rotate,
        positions=positions,
        max_n_torsions=max_n_torsions,
        tor_per_mol_label=tor_per_mol_label,
    )
    _, vjpfunc = vjp(to_coords, torsions)
    (out,) = vjpfunc(grad_pos_E, retain_graph=retain_graph)  # returns a tuple
    return out
