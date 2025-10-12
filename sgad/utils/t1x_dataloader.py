import h5py
import torch

from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch as TGBatch
from torch_geometric.loader import DataLoader as TGDataLoader

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}


def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def generator(formula, rxn, grp):
    """Iterates through a h5 group"""

    energies = grp["wB97x_6-31G(d).energy"]
    forces = grp["wB97x_6-31G(d).forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["positions"]
    molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)

    for energy, force, positions in zip(energies, forces, positions):
        d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": positions,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }

        yield d


class T1xDataloader:
    """
    Can iterate through h5 data set

    hdf5_file: path to data
    only_final: if True, the iterator will only loop through reactant, product and transition
    state instead of all configurations for each reaction and return them in dictionaries.
    indices: if specified, only iterate over these specific reaction indices
    """

    def __init__(self, hdf5_file, datasplit="data", only_final=False, indices=None):
        self.hdf5_file = hdf5_file
        self.only_final = only_final
        self.indices = indices

        self.datasplit = datasplit
        if datasplit:
            assert datasplit in [
                "data",
                "train",
                "val",
                "test",
            ], "datasplit must be one of 'all', 'train', 'val' or 'test'"

        self.reaction_keys = self.get_reaction_keys(indices)

    def get_reaction_keys(self, indices):
        """Get all (formula, rxn) pairs from the dataset."""
        all_keys = []
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]
            for formula, grp in split.items():
                for rxn, subgrp in grp.items():
                    all_keys.append((formula, rxn))
        # Use specified indices or all reactions
        if indices is not None:
            return [all_keys[i] for i in indices if i < len(all_keys)]
        else:
            return all_keys

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]

            for formula, rxn in self.reaction_keys:
                grp = split[formula]
                subgrp = grp[rxn]

                reactant = next(generator(formula, rxn, subgrp["reactant"]))
                product = next(generator(formula, rxn, subgrp["product"]))

                if self.only_final:
                    transition_state = next(
                        generator(formula, rxn, subgrp["transition_state"])
                    )
                    yield {
                        "rxn": rxn,
                        "reactant": reactant,
                        "product": product,
                        "transition_state": transition_state,
                    }
                else:
                    yield reactant
                    yield product
                    for molecule in generator(formula, rxn, subgrp):
                        yield molecule


def t1x_generator(formula, rxn, grp):
    """Iterates through a h5 group"""

    energies = grp["wB97x_6-31G(d).energy"]
    forces = grp["wB97x_6-31G(d).forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["positions"]
    # molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)

    for energy, force, positions in zip(energies, forces, positions):
        # d = {
        #     "rxn": rxn,
        #     "wB97x_6-31G(d).energy": energy.__float__(),
        #     "wB97x_6-31G(d).atomization_energy": energy
        #     - molecular_reference_energy.__float__(),
        #     "wB97x_6-31G(d).forces": force.tolist(),
        #     "positions": positions,
        #     "formula": formula,
        #     "atomic_numbers": atomic_numbers,
        # }
        data = TGData(
            pos=torch.as_tensor(positions, dtype=torch.float32).reshape(-1, 3),
            z=torch.as_tensor(atomic_numbers, dtype=torch.int64),
            charges=torch.as_tensor(atomic_numbers, dtype=torch.int64),
            natoms=torch.tensor([len(atomic_numbers)], dtype=torch.int64),
            cell=None,
            pbc=torch.tensor(False, dtype=torch.bool),
        )

        yield data


class T1xTGDataloader(T1xDataloader):
    """Torch geometric dataloader for T1x data"""

    def __init__(
        self,
        which: str = "transition_state",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.which = {
            "transition_state": "transition_state",
            "reactant": "reactant",
            "product": "product",
            "ts": "transition_state",
            "r": "reactant",
            "p": "product",
        }[which]

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]
            for formula, rxn in self.reaction_keys:
                grp = split[formula]
                subgrp = grp[rxn]
                yield next(t1x_generator(formula, rxn, subgrp[self.which]))


if __name__ == "__main__":
    dataloader = T1xTGDataloader(
        hdf5_file="../Datastore/transition1x.h5",
        datasplit="val",
        indices=None,
        which="ts",
    )
    dataset = [molecule for molecule in dataloader]
    dataloader = TGDataLoader(dataset, batch_size=4, shuffle=True)
    for molecule in dataloader:
        print(molecule)
        print(molecule.batch)
        break
