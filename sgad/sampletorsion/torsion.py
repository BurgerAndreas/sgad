# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.linalg

from rdkit import Chem
from rdkit.Chem import Mol
from torch import Tensor


T_TOR_INDEX = tuple[int, int, int, int]
T_TOR_INDEXES = list[T_TOR_INDEX]


def reverse_dihedral_by_size(mol: Mol, at1: int, at2: int) -> bool:
    """this introduces the convention that the smaller piece ends up in position [0]"""
    mol_copy = Chem.RWMol(mol)
    mol_copy.RemoveBond(at1, at2)
    frags = Chem.GetMolFrags(mol_copy)
    # if this torsion is not in a ring, then this will separate the molecule in two
    assert len(frags) == 2
    reverse = False
    # If they are tied then put the lower index one first
    if len(frags[0]) == len(frags[1]):
        if at1 > at2:
            reverse = True
    elif at1 not in min(frags, key=lambda x: len(x)):
        reverse = True
    else:
        reverse = False
    return reverse


def get_tor_indexes_daniel(mol: Mol) -> list[T_TOR_INDEX]:
    """returns torsion indexes. if broken, the atom attached to the smaller piece is in index [0]"""
    tors_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    tors_list = mol.GetSubstructMatches(tors_smarts)
    full_tors_list = []
    for at1, at2 in tors_list:
        at0 = max(
            (at for at in mol.GetAtomWithIdx(at1).GetNeighbors() if at.GetIdx() != at2),
            key=lambda x: x.GetMass(),
        )
        at3 = max(
            (at for at in mol.GetAtomWithIdx(at2).GetNeighbors() if at.GetIdx() != at1),
            key=lambda x: x.GetMass(),
        )
        full_tors = [at0.GetIdx(), at1, at2, at3.GetIdx()]
        if reverse_dihedral_by_size(mol, at1, at2):
            full_tors.reverse()
        full_tors_list.append(full_tors)
    return full_tors_list


def dihedral(pos: Tensor) -> Tensor:  # noqa: F722
    """uses rdkit rotation and angle conventions"""
    dx1 = pos[1] - pos[0]
    dx2 = pos[2] - pos[1]
    dx3 = pos[3] - pos[2]
    xdx1dx2 = torch.linalg.cross(dx1, dx2)
    xdx2dx3 = torch.linalg.cross(dx2, dx3)
    xddd = torch.linalg.cross(xdx1dx2, xdx2dx3)
    numer = torch.einsum("...i,...i->...", dx2, xddd)
    ndx = torch.linalg.vector_norm(dx2, dim=-1, keepdim=True)
    denom = torch.einsum("...i,...i->...", ndx * xdx1dx2, xdx2dx3)
    return torch.arctan2(numer, denom)


@torch.no_grad
def check_torsions(
    positions: torch.Tensor, tor_index: torch.Tensor, torsions: torch.Tensor
) -> None:
    # check that torsions are correct
    tora = dihedral(positions[tor_index])
    torb = torsions.clone()
    allc_s = torch.allclose(tora.sin(), torb.sin(), atol=1e-4)
    allc_c = torch.allclose(tora.cos(), torb.cos(), atol=1e-4)
    assert allc_s and allc_c, "computed and represented torsions do not agree!"


# # quick test that our definitions are the same as RDKit
# from rdkit import Chem
# for ii in range(graph.batch.max() + 1):
#     s = graph["smiles"][ii]
#     p = graph["pos"][graph.batch == ii].detach().cpu().numpy()
#     tor_inds = graph["tor_index"][:, 0].cpu().tolist()
#     mol = Chem.MolFromSmiles(s[0])
#     mol = Chem.AddHs(mol)
#     Chem.AllChem.EmbedMultipleConfs(
#         mol,
#         numConfs=1,
#         randomSeed=42,  # for reproducibility
#         pruneRmsThresh=-1,  # Remove similar conformers
#         enforceChirality=True,
#     )
#     conf = mol.GetConformer(0)
#     conf.SetPositions(p)
#     rda = Chem.rdMolTransforms.GetDihedralRad(conf, *tor_inds)
#     assert np.allclose(dhi[ii].detach().cpu().numpy(), rda)
