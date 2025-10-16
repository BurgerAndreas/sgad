# Copyright (c) Meta Platforms, Inc. and affiliates.

import time
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
        positions = graph_state["pos"].detach().cpu()
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
    atomic_numbers_allowed,
    rank,
    device,
    cfg,
    global_step,
):
    start_time = time.time()
    states = []
    outputs = []
    soc_loss = 0.0
    num_neg_freqs = []
    num_transition_states = 0  # Count molecules with exactly 1 negative frequency
    num_minima = 0  # Count molecules with 0 negative frequencies (minima)
    all_frequency_analyses = []
    energy_values = []  # per-molecule energies
    force_norm_values = []  # per-molecule mean force norms
    energy_grad_norm_values = []  # per-molecule mean energy-gradient norms
    eigvalprod_values = []  # optional per-batch or per-molecule scalar(s)

    print(f"Evaluation at global step: {global_step}")
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
        positions = graph_state["pos"]
        # atomic_nums = graph_state["node_attrs"].argmax(dim=-1)
        atomic_nums = graph_state["z"]
        
        # Collect per-molecule energy and force-norm statistics
        forces = output["forces"]  # shape: (num_atoms_total, 3)
        energy_grad = output.get("energy_grad", None)  # optional, same shape as forces
        # Optionally collect eigvalprod if present (assumed scalar or 1D tensor)
        if "eigvalprod" in output:
            evp = output["eigvalprod"].detach().cpu().float().reshape(-1)
            eigvalprod_values.extend(evp)
        for i in range(len(batch_ptr) - 1):
            start_idx = batch_ptr[i]
            end_idx = batch_ptr[i + 1]
            energy_values.append(output["energy"][i].detach().cpu())
            mol_forces = forces[start_idx:end_idx]
            mol_force_norm = mol_forces.norm(dim=1).mean()
            force_norm_values.append(mol_force_norm.detach().cpu())
            if energy_grad is not None:
                mol_energy_grad = energy_grad[start_idx:end_idx]
                mol_energy_grad_norm = mol_energy_grad.norm(dim=1).mean()
                energy_grad_norm_values.append(mol_energy_grad_norm.detach().cpu())

        for i in range(len(batch_ptr) - 1):
            start_idx = batch_ptr[i]
            end_idx = batch_ptr[i + 1]

            mol_positions = positions[start_idx:end_idx]
            mol_atomic_nums = atomic_nums[start_idx:end_idx]

            # Get hessian from energy model if available
            hessian = output["hessian"][i]  # Assuming hessian is per-molecule
            # if "hessian" in output:
            #     hessian = output["hessian"][i]  # Assuming hessian is per-molecule
            # else:
            #     # If no hessian available, skip frequency analysis for this molecule
            #     continue

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
            # Count minima (no negative frequencies)
            if neg_freq_count == 0:
                num_minima += 1

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

    # Compute and log energy and force-norm statistics
    if energy_values:
        energy_tensor = torch.stack(energy_values).float()
        avg_energy = energy_tensor.mean().item()
        min_energy = energy_tensor.min().item()
        max_energy = energy_tensor.max().item()
    else:
        avg_energy = 0.0
        min_energy = 0.0
        max_energy = 0.0

    if force_norm_values:
        force_norm_tensor = torch.stack(force_norm_values).float()
        avg_force_norm = force_norm_tensor.mean().item()
        min_force_norm = force_norm_tensor.min().item()
        max_force_norm = force_norm_tensor.max().item()
    else:
        avg_force_norm = 0.0
        min_force_norm = 0.0
        max_force_norm = 0.0

    # Compute and log energy-gradient norm statistics
    if energy_grad_norm_values:
        energy_grad_norm_tensor = torch.stack(energy_grad_norm_values).float()
        avg_energy_grad_norm = energy_grad_norm_tensor.mean().item()
        min_energy_grad_norm = energy_grad_norm_tensor.min().item()
        max_energy_grad_norm = energy_grad_norm_tensor.max().item()
    else:
        avg_energy_grad_norm = 0.0
        min_energy_grad_norm = 0.0
        max_energy_grad_norm = 0.0

    # Aggregate eigvalprod if present
    eigvalprod_present = len(eigvalprod_values) > 0
    if eigvalprod_present:
        eigvalprod_tensor = torch.stack(eigvalprod_values).float()
        eigvalprod_avg = eigvalprod_tensor.mean().item()
        eigvalprod_num_negative = (eigvalprod_tensor < 0).sum().item()

    print(
        f"Energy stats: avg={avg_energy:.6f}, min={min_energy:.6f}, max={max_energy:.6f}"
    )
    print(
        f"Force-norm stats: avg={avg_force_norm:.6f}, min={min_force_norm:.6f}, max={max_force_norm:.6f}"
    )
    print(
        f"Energy-gradient-norm stats: avg={avg_energy_grad_norm:.6f}, min={min_energy_grad_norm:.6f}, max={max_energy_grad_norm:.6f}"
    )
    if eigvalprod_present:
        print(f"Eigvalprod: avg={eigvalprod_avg:.6f}, num_negative={int(eigvalprod_num_negative)}")

    conformer_outputs = None
    Im = get_dataset_fig(
        states[0], outputs[0]["energy"], cfg, outputs=conformer_outputs
    )
    Im.save("test_im.png")

    if cfg.visualize_conformations:
        visualize_conformations(
            states[0], outputs[0], atomic_numbers_allowed, n_samples=16
        )

    return {
        "soc_loss": soc_loss,
        # "avg_neg_freqs": avg_neg_freqs,
        # "max_neg_freqs": max_neg_freqs,
        # "min_neg_freqs": min_neg_freqs,
        "num_transition_states": num_transition_states,
        "num_minima": num_minima,
        "transition_state_ratio": transition_state_ratio,
        # "frequency_analyses": all_frequency_analyses,
        "eval_time": time.time() - start_time,
        "avg_energy": avg_energy,
        "min_energy": min_energy,
        "max_energy": max_energy,
        "avg_force_norm": avg_force_norm,
        "min_force_norm": min_force_norm,
        "max_force_norm": max_force_norm,
        "avg_energy_grad_norm": avg_energy_grad_norm,
        "min_energy_grad_norm": min_energy_grad_norm,
        "max_energy_grad_norm": max_energy_grad_norm,
        **(
            {
                "eigvalprod_avg": eigvalprod_avg,
                "eigvalprod_num_negative": int(eigvalprod_num_negative),
            }
            if eigvalprod_present
            else {}
        ),
    }
