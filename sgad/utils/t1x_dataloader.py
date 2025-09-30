import h5py

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


class Dataloader:
    """
    Can iterate through h5 data set for paper ####

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

    def get_all_reaction_keys(self):
        """Get all (formula, rxn) pairs from the dataset."""
        reaction_keys = []
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]
            for formula, grp in split.items():
                for rxn, subgrp in grp.items():
                    reaction_keys.append((formula, rxn))
        return reaction_keys

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]

            # Get all reaction keys
            all_keys = self.get_all_reaction_keys()

            # Use specified indices or all reactions
            if self.indices is not None:
                reaction_keys = [all_keys[i] for i in self.indices if i < len(all_keys)]
            else:
                reaction_keys = all_keys

            for formula, rxn in reaction_keys:
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
