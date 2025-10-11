# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import pickle
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np

import torch
from ase import Atoms
from ase.io import read
from ase.optimize.lbfgs import LBFGS

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.AllChem import EmbedMultipleConfs
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from sgad.components.ourjoblib import joblib_map
from sgad.energies.fairchem_energy import FairChemEnergy
from sgad.sampletorsion.featurize import drugs_types, featurize_mol
from sgad.sampletorsion.rotate import get_index_to_rotate
from sgad.sampletorsion.torsion import get_tor_indexes_daniel, T_TOR_INDEXES

from sgad.utils.data_utils import get_atomic_graph
from sgad.utils.t1x_dataloader import Dataloader as T1xDataloader


# Define covalent radii (in Å) for relevant elements
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "O": 0.66,
    "N": 0.71,
    "F": 0.57,
    "Cl": 0.99,
    "Br": 1.14,
    "S": 1.05,
    "I": 1.33,
    "P": 1.07,
}

VAN_DER_WAALS_RADII = {
    "H": 1.10,
    "C": 1.70,
    "O": 1.52,
    "N": 1.55,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.83,
    "S": 1.80,
    "I": 1.98,
    "P": 1.80,
}

# Atom-specific tolerances (in Å) for each element
ATOM_SPECIFIC_TOLERANCES = {
    "H": 0.1,
    "C": 0.15,
    "O": 0.12,
    "N": 0.12,
    "F": 0.1,
    "Cl": 0.2,
    "Br": 0.2,
    "S": 0.18,
    "I": 0.2,
    "P": 0.18,
}

# Empirical multipliers for bond types
BOND_MULTIPLIERS = {
    Chem.rdchem.BondType.SINGLE: 1.0,
    Chem.rdchem.BondType.DOUBLE: 0.85,
    Chem.rdchem.BondType.TRIPLE: 0.75,
    Chem.rdchem.BondType.AROMATIC: 0.95,
}


def get_covalent_radius(atom_symbol):
    """Returns the covalent radius for a given atom symbol."""
    return COVALENT_RADII.get(atom_symbol, None)


def get_van_der_waals_radius(atom_symbol):
    """Returns the covalent radius for a given atom symbol."""
    return VAN_DER_WAALS_RADII.get(atom_symbol, None)


def get_atom_specific_tolerance(atom_symbol):
    """Returns the tolerance specific to a given atom symbol."""
    return ATOM_SPECIFIC_TOLERANCES.get(atom_symbol, 0.1)


# def bond_length_matrix(rdmol):
#     if rdmol is None:
#         raise ValueError("Invalid SMILES string.")

#     # Get number of atoms (including hydrogens)
#     N = rdmol.GetNumAtoms()

#     # Initialize the NxN matrix with 'inf' for non-bonded pairs
#     length_matrix = torch.full((N, N), fill_value=np.nan)
#     length_matrix.fill_diagonal_(0.0)
#     type_matrix = torch.full((N, N), fill_value=0)
#     # type_matrix.fill_diagonal_(0.0)
#     # Create a list to store atom symbols in order
#     atomic_numbers = [rdmol.GetAtomWithIdx(i).GetAtomicNum() for i in range(N)]

#     # Iterate over bonds in the molecule

#     for bond in rdmol.GetBonds():
#         # Get the indices of the bonded atoms
#         i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         # Get the atom symbols for each atom
#         atom1, atom2 = (
#             rdmol.GetAtomWithIdx(i).GetSymbol(),
#             rdmol.GetAtomWithIdx(j).GetSymbol(),
#         )

#         # Get bond multiplier
#         multiplier = 1.5  # * BOND_MULTIPLIERS.get(bond.GetBondType(), 1.0)

#         # Get covalent radii and atom-specific tolerances
#         # Get radii and tolerances
#         radius1, radius2 = get_covalent_radius(atom1), get_covalent_radius(atom2)
#         # Calculate the upper limit of bond length based on bond type
#         if radius1 is not None and radius2 is not None:
#             # check with the chemistry team
#             bond_length = (radius1 + radius2) * multiplier
#             length_matrix[i, j] = bond_length
#             length_matrix[j, i] = bond_length  # Matrix is symmetric

#             type_matrix[i, j] = 1
#             type_matrix[j, i] = 1

#     # fill out non-bond edges with limits
#     for i in range(length_matrix.shape[0]):
#         for j in range(i, length_matrix.shape[0]):
#             if torch.isnan(length_matrix[i, j]):
#                 atom1, atom2 = (
#                     rdmol.GetAtomWithIdx(i).GetSymbol(),
#                     rdmol.GetAtomWithIdx(j).GetSymbol(),
#                 )
#                 multiplier = (
#                     1.0 / 1.5
#                 )  # (1.2  * BOND_MULTIPLIERS.get(bond.GetBondType(), 1.0) )

#                 # Get covalent radii and atom-specific tolerances
#                 # Get radii and tolerances
#                 radius1, radius2 = (
#                     get_van_der_waals_radius(atom1),
#                     get_van_der_waals_radius(atom2),
#                 )

#                 # Calculate the upper limit of bond length based on bond type
#                 if radius1 is not None and radius2 is not None:
#                     # check with the chemistry team
#                     bond_length = (radius1 + radius2) * multiplier
#                     length_matrix[i, j] = bond_length
#                     length_matrix[j, i] = bond_length  # Matrix is symmetric
#     # print(matrix)
#     return length_matrix, type_matrix, atomic_numbers


def get_positions(
    mol: Chem.Mol, tor_indexes: T_TOR_INDEXES, relax: bool, energy_model=None
) -> np.ndarray:
    """provides the positions of a conformation with dihedrals set to zero"""
    # find a conformation
    EmbedMultipleConfs(
        mol,
        numConfs=1,
        randomSeed=42,  # for reproducibility
        pruneRmsThresh=-1,  # Remove similar conformers
        enforceChirality=True,
    )
    confId = 0
    conf = mol.GetConformer(confId)

    # relax structure
    if relax:
        atoms = Atoms(
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
            positions=conf.GetPositions(),
        )
        if energy_model is None:
            raise ValueError(
                "you must provide an energy model with an ASE calculator in energy_model.calc"
            )
        atoms.calc = energy_model.calc
        out = LBFGS(atoms)
        out.run()
        conf.SetPositions(out.atoms.get_positions())

    # set dihedrals to zero
    for tor_ind in tor_indexes:
        rdMolTransforms.SetDihedralRad(conf, *tor_ind, 0.0)
    return conf.GetPositions()


def tordiff_featurizer(
    smile: Union[str, bytes],
    mol: Chem.Mol,
) -> Data:
    # our additions
    if isinstance(smile, bytes):
        smile = smile.decode("utf-8")

    # mostly copied
    if "." in smile:
        raise ValueError("tordiff cant have dots in smiles")

    N = mol.GetNumAtoms()

    if N < 4:
        raise ValueError("mol_too_small")

    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=False)

    data = featurize_mol(mol, drugs_types)
    if not data:
        raise ValueError("featurize_mol_failed")

    data.canonical_smi = canonical_smi
    # data.mol = mol
    return data


# generates dataloader of all zeros for a single molecule type. Only need to generate one batch.
def create_molecule_copies_from_pointcloud(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    energy_model,
    duplicate: int,
) -> list[Data]:
    """
    Create molecular graphs from 3D point clouds without bonds.
    
    Args:
        atomic_numbers: Tensor of atomic numbers [N]
        positions: Tensor of 3D positions [N, 3]
        energy_model: Energy model with atomic_numbers attribute
        duplicate: Number of copies to create
        
    Returns:
        List of Data objects representing molecular graphs
    """
    atomic_number_table = energy_model.atomic_numbers
    atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}

    dataset = []
    for _ in range(duplicate):
        # TODO@Andreas: get graph using energy_model
        # Create atomic graph without bond information
        data_i = get_atomic_graph(atomic_numbers.tolist(), positions, atomic_index_table)
        
        # Remove bond-related edge attributes - only keep distance-based features
        edge_index = data_i["edge_index"]
        # Calculate distances between all atom pairs
        distances = torch.norm(
            positions[edge_index[0]] - positions[edge_index[1]], 
            dim=1, 
            keepdim=True
        )
        data_i["edge_attrs"] = distances.float()
        
        dataset.append(data_i)

    return dataset


def pointcloud_to_graph(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    energy_model,
    duplicate: int,
    atomic_index_table,
    r_max,
) -> Optional[Data]:
    """
    Convert 3D point cloud to molecular graph without bonds.
    
    Args:
        atomic_numbers: Tensor of atomic numbers [N]
        positions: Tensor of 3D positions [N, 3]
        energy_model: Energy model
        duplicate: Number of duplicates
        atomic_index_table: Mapping from atomic numbers to indices
        r_max: Maximum distance (not used for bonds anymore)
        
    Returns:
        Data object representing molecular graph
    """
    # Create atomic graph without bond information
    data_i = get_atomic_graph(atomic_numbers.tolist(), positions, atomic_index_table)

    # Calculate distances between all atom pairs for edge attributes
    edge_index = data_i["edge_index"]
    distances = torch.norm(
        positions[edge_index[0]] - positions[edge_index[1]], 
        dim=1, 
        keepdim=True
    )
    data_i["edge_attrs"] = distances.float()
    
    # Remove SMILES reference
    # data_i["smiles"] = smiles  # Removed

    return data_i


def create_t1x_dataset(
    datapath: str,
    datasplit: str,
    energy_model,
    duplicate: int,
) -> list[Data]:
    """
    Create dataset from Transition1x data.
    
    Args:
        datapath: Path to T1x HDF5 file
        datasplit: Data split to use ('train', 'val', 'test')
        energy_model: Energy model with atomic_numbers attribute
        duplicate: Number of copies to create per molecule
        
    Returns:
        List of Data objects representing molecular graphs
    """
    dataloader = T1xDataloader(datapath, datasplit=datasplit, only_final=False)
    dataset = []
    
    for molecule in dataloader:
        atomic_numbers = torch.tensor(molecule["atomic_numbers"], dtype=torch.long)
        positions = torch.tensor(molecule["positions"], dtype=torch.float32)
        
        # Create duplicates using the pointcloud function
        molecule_data = create_molecule_copies_from_pointcloud(
            atomic_numbers=atomic_numbers,
            positions=positions,
            energy_model=energy_model,
            duplicate=duplicate,
        )
        dataset.extend(molecule_data)
    
    return dataset


# def get_spice_dataset(
#     dataset_path: str,
#     energy_model,
#     duplicate: int,
#     cache_only: bool = False,
# ):
#     atomic_number_table = energy_model.atomic_numbers
#     atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}

#     r_max = energy_model.r_max

#     if not os.path.isabs(dataset_path):
#         repo_dir = Path(__file__).resolve().parent.parent.parent
#         dataset_path = os.path.join(repo_dir, dataset_path)

#     if isinstance(energy_model, FairChemEnergy):
#         energy_model_suffix = "_fc"
#     else:
#         ValueError("energy_model is unrecognized")
#     cache_path = os.path.splitext(dataset_path)[0] + energy_model_suffix + ".pkl"
#     if Path(cache_path).is_file():
#         print("found cached dataset!")
#         if cache_only:
#             return
#         else:
#             print(f"loading from {cache_path}")
#             with open(cache_path, "rb") as handle:
#                 dataset = pickle.load(handle)
#             return dataset
#     with open(dataset_path, "r") as f:
#         next(f)
#         train_mols = f.readlines()

#     _, smiles_list_in, _ = zip(
#         *[line.strip().split() for line in tqdm(train_mols, desc="mol text")]
#     )

#     print(f"loading from {dataset_path}")

#     todo = partial(
#         smiles_to_graph,
#         energy_model=energy_model,
#         duplicate=duplicate,
#         atomic_index_table=atomic_index_table,
#         r_max=r_max,
#     )
#     dataset = joblib_map(
#         todo,
#         smiles_list_in,
#         n_jobs=int(os.environ["SLURM_CPUS_ON_NODE"])
#         if os.environ.get("SLURM_CPUS_ON_NODE", None)
#         else 1,
#         inner_max_num_threads=1,
#         desc="reading smiles",
#         total=len(smiles_list_in),
#     )
#     dataset = [item for item in dataset if item is not None]

#     print("caching data to ", cache_path)
#     with open(cache_path, "wb") as handle:
#         pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     return dataset


# def test_dataset(
#     smiles_list,
#     energy_model,
#     duplicate: int = 1,
# ):
#     atomic_number_table = energy_model.atomic_numbers
#     atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}

#     dataset = []
#     for smiles in smiles_list:
#         rdmol = Chem.MolFromSmiles(smiles)
#         rdmol = Chem.AddHs(rdmol)
#         length_matrix, type_matrix, atom_list = bond_length_matrix(rdmol)

#         positions = torch.zeros((len(atom_list), 3))
#         tor_data = {}
#         data_tordiff = Data()

#         for _ in range(duplicate):
#             data_i = get_atomic_graph(atom_list, positions, atomic_index_table)
#             edge_index = data_i["edge_index"]
#             length_attr = length_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
#             type_attr = type_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
#             edge_attr = torch.cat([length_attr, type_attr], dim=-1).float()
#             data_i["edge_attrs"] = edge_attr
#             data_i["smiles"] = smiles

#             data_tordiff_dup = data_tordiff.clone()

#             # no overlap between these keys
#             assert set(data_i.keys()).intersection(tor_data.keys()) == set()
#             for k, v in tor_data.items():
#                 data_i[k] = v

#             # no overlap between these keys either
#             assert (
#                 set(data_tordiff_dup.keys()).intersection(set(data_i.keys())) == set()
#             )
#             for k in data_tordiff_dup.keys():
#                 if isinstance(data_tordiff_dup[k], np.ndarray):
#                     data_i[k] = torch.from_numpy(data_tordiff_dup[k])
#                 else:
#                     data_i[k] = data_tordiff_dup[k]

#             dataset.append(data_i)

#     return dataset


