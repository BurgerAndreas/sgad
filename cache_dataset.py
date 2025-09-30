# Copyright (c) Meta Platforms, Inc. and affiliates.

import warnings
from argparse import ArgumentParser

from sgad.components.datasets import get_spice_dataset
from sgad.energies.fairchem_energy import FairChemEnergy


warnings.simplefilter(action="ignore", category=FutureWarning)

known_datasets = {
    "spice_train": "data/spice_train.txt",
    "spice_test": "data/spice_test.txt",
    "drugs": "data/drugs_test.txt",
}


energy_models = ["fairchem"]


parser = ArgumentParser("Dataset Caching Options")
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["spice_train", "spice_test", "drugs"],
    choices=list(known_datasets.keys()),
)
parser.add_argument(
    "--energy_model", type=str, default="fairchem", choices=energy_models
)
parser.add_argument("--torsion_cache_suffix", type=str, default="_torsion")
parser.add_argument("--learn_torsions", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    device = "cpu"
    if args.energy_model == "fairchem":
        energy_model = FairChemEnergy(device=device)
    else:
        ValueError("wrong energy model")

    for dataset_name in args.datasets:
        if dataset_name in known_datasets:
            dataset_path = known_datasets[dataset_name]
            print("Loading {} dataset ...".format(dataset_name))
            get_spice_dataset(
                dataset_path,
                energy_model,
                duplicate=1,
                torsion_cache_suffix=args.torsion_cache_suffix,
                learn_torsions=args.learn_torsions,
                relax=False,
                cache_only=True,
            )
        else:
            print("dataset name {} not registered, continuing ...".format(dataset_name))
    print("Caching completed.")
