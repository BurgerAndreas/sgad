import transition1x as t1x
import numpy as np

"""
The elements in the data loader each represent a single molecule. It is a dictionary that has the following keys available:

rxn:                               the name of the reaction that the molecule is coming from
formula:                           chemical formula for the molecule.
positions:                         list of x, y, z coordinates of all atoms in the molecule in Å.
atomic_numbers:                    list of atomic numbers ordered in the same way as positions.
wB97x_6-31G(d).energy:             total energy of molecule in eV.
wB97x_6-31G(d).atomization_energy: atomization energy of molecule in eV.
wB97x_6-31G(d).forces:             list of x, y, z forces on each atom in eV/Å - atoms are ordered in the same way as in positions.

It is possible to provide a datasplit key to the dataloader from 'train', 'val' or 'test' to only iterate through the training, validation or test data, respectively.

Finally, it is possible to go through the reactant, transition state and product only by setting 'only_final' kwarg to True when instantiating the data loader.
In this case the data loader will return dictionaries where the configurations can be accessed under 'product', 'transition_state' or 'reactant'.
"""

datapath = "../Datastore/transition1x.h5"

dataloader = t1x.Dataloader(datapath, only_final=True)
for molecule in dataloader:
    ts_energy = molecule["transition_state"]["wB97x_6-31G(d).energy"]
    r_energy = molecule["reactant"]["wB97x_6-31G(d).energy"]
    activation_energy = ts_energy - r_energy
    atomic_numbers = molecule["transition_state"]["atomic_numbers"]
    formula = molecule["formula"]
    print(molecule["transition_state"].keys())
    break

# Sample indices 
np.random.seed(0)
sampled_indices = np.random.sample(
    range(1000), 10
)

# Create final dataloader with sampled indices
dataloader = t1x.Dataloader(
    datapath, datasplit="val", only_final=True, indices=sampled_indices
)

for molecule in dataloader:
    ts_energy = molecule["transition_state"]["wB97x_6-31G(d).energy"]
    r_energy = molecule["reactant"]["wB97x_6-31G(d).energy"]
    activation_energy = ts_energy - r_energy
    atomic_numbers = molecule["transition_state"]["atomic_numbers"]
    formula = molecule["formula"]
    print(molecule["transition_state"].keys())
    break