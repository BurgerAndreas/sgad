# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

from hip.frequency_analysis import analyze_frequencies_torch

from sgad.components.sde import integrate_sde
from sgad.components.soc import SOC_loss

from sgad.utils.visualize_utils import (
    get_dataset_fig,
    visualize_conformations,
)

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


def save_to_xyz(batches, outputs, atomic_numbers, rank, dir):
    k = 0
    for graph_state, output in zip(batches, outputs):
        positions = graph_state["positions"].detach().cpu()
        atom_list = graph_state["node_attrs"].detach().cpu()
        ij = 0
        for i in range(len(graph_state["ptr"]) - 1):
            sys_size = graph_state["ptr"][i + 1] - graph_state["ptr"][i]
            with open(dir + "/sample_{}_{}.xyz".format(rank, k), "w") as f:
                f.write("{}\n".format(sys_size.item()))
                f.write("   {}\n".format(output["energy"][i]))
                for j in range(sys_size):
                    atomic_number = atomic_numbers[torch.nonzero(atom_list[ij])]
                    symbol = SYM_LIST[atomic_number[0, 0].item()]
                    f.write(
                        "{0:s}    {1:>f}    {2:>f}    {3:>f}\n".format(
                            symbol,
                            positions[ij, 0].item(),
                            positions[ij, 1].item(),
                            positions[ij, 2].item(),
                        )
                    )
                    ij += 1
            k += 1


def evaluation(
    sde,
    energy_model,
    eval_sample_loader,
    noise_schedule,
    atomic_numbers,
    rank,
    device,
    cfg,
):
    states = []
    outputs = []
    soc_loss = 0.0
    num_neg_freqs = []
    num_transition_states = 0  # Count molecules with exactly 1 negative frequency
    all_frequency_analyses = []

    for batch in eval_sample_loader:
        batch = batch.to(device)
        graph_state, controls = integrate_sde(
            sde, batch, cfg.eval_nfe, only_final_state=False
        )
        output = energy_model(graph_state)

        # generated_energies = outputs["energy"]
        states.append(graph_state)
        outputs.append(output)
        soc_loss += SOC_loss(controls, graph_state, output["energy"], noise_schedule)

        # Perform frequency analysis for each molecule in the batch
        batch_ptr = graph_state["ptr"]
        positions = graph_state["positions"]
        atomic_nums = graph_state["node_attrs"].argmax(dim=-1)

        for i in range(len(batch_ptr) - 1):
            start_idx = batch_ptr[i]
            end_idx = batch_ptr[i + 1]

            mol_positions = positions[start_idx:end_idx]
            mol_atomic_nums = atomic_nums[start_idx:end_idx]

            # Get hessian from energy model if available
            if "hessian" in output:
                hessian = output["hessian"][i]  # Assuming hessian is per-molecule
            else:
                # If no hessian available, skip frequency analysis for this molecule
                continue

            # Perform frequency analysis
            frequency_analysis = analyze_frequencies_torch(
                hessian, mol_positions, mol_atomic_nums
            )
            all_frequency_analyses.append(frequency_analysis)
            neg_freq_count = frequency_analysis["neg_num"]
            num_neg_freqs.append(neg_freq_count)

            # Count transition states (exactly 1 negative frequency)
            if neg_freq_count == 1:
                num_transition_states += 1

    soc_loss = (soc_loss / cfg.num_eval_samples).detach().cpu().item()

    # Calculate frequency statistics
    if num_neg_freqs:
        avg_neg_freqs = torch.stack(num_neg_freqs).float().mean().item()
        max_neg_freqs = torch.stack(num_neg_freqs).float().max().item()
        min_neg_freqs = torch.stack(num_neg_freqs).float().min().item()
        total_molecules = len(num_neg_freqs)
        transition_state_ratio = (
            num_transition_states / total_molecules if total_molecules > 0 else 0.0
        )
    else:
        avg_neg_freqs = 0.0
        max_neg_freqs = 0.0
        min_neg_freqs = 0.0
        transition_state_ratio = 0.0

    conformer_outputs = None
    Im = get_dataset_fig(
        states[0], outputs[0]["energy"], cfg, outputs=conformer_outputs
    )

    if cfg.visualize_conformations:
        visualize_conformations(states[0], outputs[0], atomic_numbers, n_samples=16)

    return {
        "soc_loss": soc_loss,
        "energy_vis": Im,
        "avg_neg_freqs": avg_neg_freqs,
        "max_neg_freqs": max_neg_freqs,
        "min_neg_freqs": min_neg_freqs,
        "num_transition_states": num_transition_states,
        "transition_state_ratio": transition_state_ratio,
        "frequency_analyses": all_frequency_analyses,
    }
