# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import traceback
import warnings
from argparse import ArgumentParser
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch

from sgad.components.controllers import EGNN_dynamics
from sgad.components.datasets import test_dataset
from sgad.components.noise_schedule import GeometricNoiseSchedule
from sgad.components.sde import (
    ControlledGraphSDE,
    ControlledGraphTorsionSDE,
    integrate_sde,
)

from sgad.energies.fairchem_energy import (
    FairChemEnergy,
    wandb_mode_disabled,
)
from sgad.sampletorsion.randomtorsion import get_random_torsions
from sgad.utils.eval_utils import (
    calc_performance_stats,
    # calc_rmsd,
    calc_rmsd_parallel,
    generate_conformers,
    read_rdkit_mols,
    read_xyz_files,
    safe_remove_hydrogens,
    save_metrics,
    setup_device,
    visualize_conformers,
)
from fairchem.core import OCPCalculator

from huggingface_hub import hf_hub_download

from rdkit import Chem
from torch_geometric.loader import DataLoader


# Constants
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
SYM_LIST = {v: k for k, v in ATOMIC_NUMBERS.items()}


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Evaluate conformer generation")
    parser.add_argument(
        "--test_mols",
        type=str,
        default="data/spice_test.txt",
        help="Path to list of SMILES of test molecules",
    )
    parser.add_argument(
        "--true_confs",
        type=str,
        default="data/spice_test_conformers",
        help="Path/directory to xyz file with ground truth conformers",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the results"
    )
    parser.add_argument("--num_integration_steps", type=int, default=1000)
    parser.add_argument("--discretization", type=str, default="uniform")
    parser.add_argument("--num_eval_samples", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--rotatable_bonds", type=int, default=10)
    parser.add_argument(
        "--rdkit", action="store_true", help="Use RDKit to generate conformers"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Optimize RDKit conformers"
    )
    parser.add_argument("--xyz_dir", type=str, default=None)
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument(
        "--energy",
        type=str,
        default="fairchem",
        choices=["fairchem"],
        help="This only has an effect if cfg is None (rdkit)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--relax",
        action="store_true",
        help="use FairChem energy to relax sampled conformations",
    )
    parser.add_argument("--fmax", type=float, default=0.0154)
    parser.add_argument("--num_ranks", type=int, default=1)
    parser.add_argument("--max_n_refs", type=int, default=sys.maxsize)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save visualization of generated conformers",
    )
    parser.add_argument(
        "--random_torsions",
        action="store_true",
        help="create baseline where samples have torsions set randomly",
    )
    parser.add_argument(
        "--relax_steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # find cfg from checkpoint
    if args.checkpoint_path is None:
        cfg = None
    else:
        cfg_file = Path(args.checkpoint_path).parent.parent / ".hydra/config.yaml"
        if os.path.exists(cfg_file):
            cfg = omegaconf.OmegaConf.load(cfg_file)
        else:
            raise ValueError(
                f"cfg file not found at {cfg_file}, but you also provided a checkpoint. That's not going to work!"
            )
    device = setup_device(args)

    energy_map = {
        "sgad.energies.fairchem_energy.FairChemEnergy": "fairchem",
    }
    energy = args.energy if cfg is None else energy_map[cfg.energy._target_]

    # load energy model
    if energy == "fairchem":
        energy_model = FairChemEnergy(device=device)

    if not args.rdkit or not args.random_torsions:
        # load model
        if not cfg:
            controller = EGNN_dynamics(
                n_atoms=100 if args.energy == "fairchem" else 10,
                hidden_nf=128,
                n_layers=5,
                agg="sum",
            )
            noise_schedule = GeometricNoiseSchedule(sigma_min=1e-3, sigma_max=1.0)
            if args.random_torsions:
                sde = ControlledGraphTorsionSDE(controller, noise_schedule).to("cpu")
            else:
                sde = ControlledGraphSDE(controller, noise_schedule).to("cpu")
        else:
            noise_schedule = hydra.utils.instantiate(cfg.noise_schedule)
            controller = hydra.utils.instantiate(cfg.controller)
            controller.load_state_dict(
                torch.load(args.checkpoint_path, weights_only=False)[
                    "controller_state_dict"
                ]
            )
            sde = ControlledGraphSDE(
                controller, noise_schedule, use_AM_SDE=cfg.use_AM_SDE
            ).to(device)

    if args.relax:
        if args.energy == "fairchem":
            model_ckpt = hf_hub_download(
                repo_id="facebook/sgad",
                filename="esen_spice.pt",
            )
            with wandb_mode_disabled():
                calc = OCPCalculator(
                    checkpoint_path=model_ckpt,
                    seed=0,
                )
        else:
            raise NotImplementedError
    else:
        calc = None

    # load test mols
    with open(args.test_mols, "r") as f:
        next(f)
        test_mols = f.readlines()

    num_rot_bonds_list, smiles_list_in, xyz_dir_list_in = zip(
        *[line.strip().split() for line in test_mols]
    )

    smiles_list = []
    xyz_dir_list = []

    if args.smiles and args.xyz_dir:
        smiles_list = [args.smiles]
        xyz_dir_list = [args.xyz_dir]
    else:
        for num_rot_bonds, smiles, xyz_dir in zip(
            num_rot_bonds_list, smiles_list_in, xyz_dir_list_in
        ):
            if int(num_rot_bonds) <= args.rotatable_bonds:
                smiles_list.append(smiles)
                xyz_dir_list.append(xyz_dir)
            else:
                print(f"Skipping molecule with {num_rot_bonds} rotatable bonds")

    rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    print(
        f"eval over {len(smiles_list)} molecules over {args.num_ranks} ranks: RANK {rank + 1}/{args.num_ranks}"
    )
    smiles_list = smiles_list[rank :: args.num_ranks]
    xyz_dir_list = xyz_dir_list[rank :: args.num_ranks]
    n_mols_this_rank = len(smiles_list)

    threshold_ranges = np.arange(0, 2.5, 0.125)
    if (Path(args.save_path) / f"results_{rank}.json").exists():
        all_results = dict(
            pd.read_json(Path(args.save_path) / f"results_{rank}.json").T
        )
    else:
        all_results = {}

    for index, smiles in enumerate(smiles_list):
        if (xyz_dir_list[index] in all_results) and not args.overwrite:
            continue

        info_dict = {"smiles": smiles}
        print(f"Evaluating {index} of {n_mols_this_rank} molecules ...")
        # load reference mols
        xyz_path = os.path.join(
            args.true_confs, xyz_dir_list[index], "crest_conformers.xyz"
        )
        mol = Chem.MolFromSmiles(smiles)

        try:
            ref_mols = read_xyz_files(xyz_path)
        except ValueError:
            continue
        ref_mols = ref_mols[: args.max_n_refs]
        n_ref_mols = len(ref_mols)
        if args.num_eval_samples:
            num_eval_samples = args.num_eval_samples
        else:
            num_eval_samples = int(2 * n_ref_mols)

        print()
        print()
        print("num_eval_samples", num_eval_samples)
        print("n_ref_mols", n_ref_mols)
        print()
        print()

        if n_ref_mols == 0:
            continue

        print("n_ref_mols", n_ref_mols)
        print("num_eval_samples", num_eval_samples)

        if n_ref_mols == 0:
            continue

        if args.rdkit:
            gen_mols = read_rdkit_mols(
                smiles,
                num_eval_samples,
                optimize=args.optimize,
                relax=args.relax,
                calc=calc,
                fmax=args.fmax,
            )
        else:
            test_set = test_dataset(
                [smiles],
                energy_model,
                duplicate=num_eval_samples,
            )
            test_dataloader = DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )
            gen_mols = []
            sde.to(device)
            for batch in test_dataloader:
                batch = batch.to(device)
                if args.random_torsions:
                    graph_state = get_random_torsions(batch)
                else:
                    graph_state = integrate_sde(
                        sde, batch, args.num_integration_steps, only_final_state=True
                    )

                gen_mols.extend(
                    generate_conformers(
                        graph_state,
                        energy_model.atomic_numbers,
                        relax=args.relax,
                        calc=calc,
                        fmax=args.fmax,
                        steps=args.relax_steps,
                    )
                )
            sde.to("cpu")

            try:
                del graph_state, batch
            except UnboundLocalError:
                pass

            del test_dataloader, test_set

        # calculate rmsd
        correct_mols = []
        for _, mol in enumerate(gen_mols):
            mol_noHs = safe_remove_hydrogens(mol)
            smi1 = Chem.MolToSmiles(mol_noHs, isomericSmiles=False, canonical=True)
            smi2 = Chem.MolToSmiles(mol_noHs, isomericSmiles=True, canonical=True)
            if smi1 == smiles or smi2 == smiles:
                correct_mols.append(mol)

        # rmsd_array_gen_ = calc_rmsd(gen_mols, ref_mols)
        rmsd_array_gen = calc_rmsd_parallel(gen_mols, ref_mols)
        # assert np.allclose(rmsd_array_gen, rmsd_array_gen_)
        stats_gen_ = calc_performance_stats(
            rmsd_array_gen, threshold_ranges, "gen_mols vs ref_mols"
        )
        # rmsd_array_crr_ = calc_rmsd(correct_mols, ref_mols)
        rmsd_array_crr = calc_rmsd_parallel(correct_mols, ref_mols)
        # assert np.allclose(rmsd_array_crr, rmsd_array_crr_)
        stats_crr_ = calc_performance_stats(
            rmsd_array_crr, threshold_ranges, "correct_mols vs ref_mols"
        )

        if stats_gen_ is not None:
            cr, mr, cp, mp = stats_gen_
            print(threshold_ranges)
            print(f"coverage recall for mol {smiles}, {xyz_dir_list[index]}: {cr}")
            print(f"amr recall for mol {smiles}, {xyz_dir_list[index]}: {mr}")
            print(f"coverage precision for mol {smiles}, {xyz_dir_list[index]}: {cp}")
            print(f"amr precision for mol {smiles}, {xyz_dir_list[index]}: {mp}")
            info_dict.update(
                {
                    "threshold_ranges": threshold_ranges,
                    "recall": cr,
                    "precision": cp,
                    "mr": mr,
                    "mp": mp,
                    "samples_correct_smiles": len(correct_mols),
                    "samples_valid_rdkit": len(gen_mols),
                    "num_samples": args.num_eval_samples,
                }
            )
            if stats_crr_ is not None:
                cr, mr, cp, mp = stats_crr_
                info_dict.update(
                    {"recall_crr": cr, "precision_crr": cp, "mr_crr": mr, "mp_crr": mp}
                )
            else:
                info_dict.update(
                    {
                        "recall_crr": [np.nan] * len(threshold_ranges),
                        "precision_crr": [np.nan] * len(threshold_ranges),
                        "mr_crr": np.nan,
                        "mp_crr": np.nan,
                    }
                )
        else:
            info_dict.update(
                {
                    "threshold_ranges": threshold_ranges,
                    "recall": [np.nan] * len(threshold_ranges),
                    "precision": [np.nan] * len(threshold_ranges),
                    "mr": np.nan,
                    "mp": np.nan,
                    "samples_correct_smiles": len(correct_mols),
                    "samples_valid_rdkit": len(gen_mols),
                    "num_samples": args.num_eval_samples,
                    "recall_crr": [np.nan] * len(threshold_ranges),
                    "precision_crr": [np.nan] * len(threshold_ranges),
                    "mr_crr": np.nan,
                    "mp_crr": np.nan,
                }
            )

        all_results[xyz_dir_list[index]] = info_dict

        # visualize
        os.makedirs(args.save_path, exist_ok=True)
        if args.visualize:
            conformer_save_path = os.path.join(
                args.save_path, f"conformers_mol_{xyz_dir_list[index]}.png"
            )
            visualize_conformers(
                ref_mols, gen_mols, n_samples=16, save_path=conformer_save_path
            )

        del ref_mols, gen_mols
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_results = pd.DataFrame(all_results).T
    all_results.index.name = "id"
    all_results.to_json(Path(args.save_path) / f"results_{rank}.json")

    all_results_files = list(Path(args.save_path).glob("results_*"))
    if len(all_results_files) == args.num_ranks:
        final_results = pd.concat(
            [pd.read_json(f) for f in all_results_files]
        ).sort_index()
        print("final results", len(final_results))
        if len(final_results) != 100:
            warnings.warn(
                "Final results does not contain exactly 100 molecules!", UserWarning
            )
        if len(final_results) != 500:
            warnings.warn(
                "Final results does not contain exactly 500 molecules!", UserWarning
            )
        final_results.to_json(Path(args.save_path) / "final_results.json")

        save_metrics(final_results, Path(args.save_path) / "metrics.csv")
        save_metrics(
            final_results,
            Path(args.save_path) / "metrics_exclude_bad.csv",
            none_processing="nan",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr)
        raise e
