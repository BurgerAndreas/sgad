# Copyright (c) Meta Platforms, Inc. and affiliates.

import io

import matplotlib.pyplot as plt
import PIL
import torch

from rdkit import Chem

from rdkit.Chem import Draw, rdDetermineBonds

from sgad.utils.eval_utils import safe_remove_hydrogens


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def interatomic_dist(samples):
    # Compute the pairwise interatomic distances
    # removes duplicates and diagonal
    n_particles = samples.shape[1]
    distances = samples[:, None, :, :] - samples[:, :, None, :]
    distances = distances[
        :,
        torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1,
    ]
    dist = torch.linalg.norm(distances, dim=-1)
    return dist


@torch.no_grad()
def get_dataset_fig(graph_state, energies, cfg, outputs=None):
    n_systems = len(graph_state["ptr"]) - 1
    n_particles = int(len(graph_state["batch"]) // n_systems)
    n_spatial_dim = graph_state["positions"].shape[-1]
    x = graph_state["positions"].view(n_systems, n_particles, n_spatial_dim)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    dist_samples = interatomic_dist(x).detach().cpu()

    bins = 200

    axs[0].hist(
        dist_samples.view(-1),
        bins=bins,
        alpha=0.5,
        density=True,
        histtype="step",
        linewidth=4,
    )

    axs[0].set_xlabel("Interatomic distance")
    axs[0].legend(["generated data"])
    energies = energies.detach().cpu()
    min_energy = energies.min().item() if cfg.min_energy is None else cfg.min_energy
    max_energy = energies.max().item() if cfg.max_energy is None else cfg.max_energy

    axs[1].hist(
        energies,
        bins=100,
        density=True,
        alpha=0.4,
        range=(min_energy, max_energy),
        color="r",
        histtype="step",
        linewidth=4,
        label="generated data",
    )
    if outputs is not None:
        for x in outputs["energy"]:
            axs[1].axvline(x=x.cpu(), color="red", linestyle="--")
    axs[1].set_xlabel("Energy")
    axs[1].legend()

    fig.canvas.draw()
    PIL_im = fig2img(fig)
    plt.close()
    return PIL_im


def to_mol(positions, atom_types):
    """Convert an XYZ file to an RDKit Mol object"""
    # SYM_LIST = {1: "H", 6: "C", 8: "O", 53: "I", 7: "N", 17: "Cl"}
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

    num_atoms = positions.shape[0]
    mol = Chem.Mol()
    edit_mol = Chem.EditableMol(mol)
    conf = Chem.Conformer(num_atoms)
    for i in range(num_atoms):
        atom_symbol = SYM_LIST[atom_types[i]]
        x = positions[i, 0]
        y = positions[i, 1]
        z = positions[i, 2]
        atom = Chem.Atom(atom_symbol)
        atom_idx = edit_mol.AddAtom(atom)
        conf.SetAtomPosition(int(atom_idx), (float(x), float(y), float(z)))

    mol = edit_mol.GetMol()
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(mol)
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
        cm = safe_remove_hydrogens(mol)
        smi = Chem.MolToSmiles(cm)
        return mol, smi
    except:  # noqa: E722
        return mol, "NA"

    # Chem.SanitizeMol(mol)


def visualize_conformations(graph_state, outputs, atomic_number_table, n_samples=16):
    n_samples = min(n_samples, len(graph_state["ptr"]) - 1)
    # rows = math.ceil(n_samples / 4)
    # fig, ax = plt.subplots(
    #     rows, 4, figsize=(10 * rows / 4, 10), gridspec_kw={"wspace": 0.5, "hspace": 0.5}
    # )
    # ij = 0
    ptr = graph_state["ptr"]

    mols = []
    smis = []
    for i in range(n_samples):
        indices = torch.nonzero(graph_state["node_attrs"][ptr[i] : ptr[i + 1]])[
            :, 1
        ].int()
        atomic_numbers = (
            atomic_number_table[indices.detach().cpu()].detach().cpu().numpy()
        )
        positions = graph_state["positions"][ptr[i] : ptr[i + 1]].detach().cpu().numpy()
        mol, smi = to_mol(positions, atomic_numbers)
        # mol = AllChem.AddHs(mol)
        # try:
        #     AllChem.EmbedMolecule(mol)
        # except:
        #     print("failed to embed molecule")

        mols.append(mol)
        smis.append(smi)

    try:
        legends = [
            smi + "\n\n reg energy:{:.2f}".format(outputs["reg_energy"][j])
            for j, smi in enumerate(smis)
        ]
    except TypeError:
        legends = [smi + "\n\n reg energy: NA" for j, smi in enumerate(smis)]
        
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(200, 200),
        legends=legends,
    )
    img.save("rdkit_vis.png")

    # for i in range(rows):
    #     for j in range(4):
    #         indices = torch.nonzero(graph_state["node_attrs"][ptr[ij] : ptr[ij + 1]])[
    #             :, 1
    #         ].int()
    #         atomic_numbers = atomic_number_table[indices].detach().cpu().numpy()
    #         positions = (
    #             graph_state["positions"][ptr[ij] : ptr[ij + 1]].detach().cpu().numpy()
    #         )
    #         atom = Atoms(numbers=atomic_numbers, positions=positions)
    #         ax[i, j] = plot_atoms(atom, ax[i, j])
    #         ax[i, j].set_title(
    #             "E: {:.2f}, E Reg.: {:.2f}".format(
    #                 outputs["energy"][ij], outputs["reg_energy"][ij]
    #             ),
    #             fontsize=10,
    #         )
    #         ij += 1

    # fig.savefig("test_conformation_arrangment.png")
    # plt.close()
