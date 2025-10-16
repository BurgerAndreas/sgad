# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from contextlib import contextmanager
from typing import Optional

import torch

from fairchem.core import OCPCalculator

from huggingface_hub import hf_hub_download


def bond_structure_regularizer(
    positions, bond_limits, bond_types, edge_index, batch_ptr, alpha=1.0
):
    bond_norms = torch.sqrt(
        torch.sum(
            (positions[edge_index[1]] - positions[edge_index[0]]) ** 2,
            dim=1,
            keepdim=True,
        )
    )
    bond_mask = bond_types == 1
    no_bond_mask = bond_types == 0

    bond_constraint = torch.nn.functional.relu(bond_mask * (bond_norms - bond_limits))[
        :, 0
    ]
    no_bond_constraint = torch.nn.functional.relu(
        no_bond_mask * (bond_limits - bond_norms)
    )[:, 0]

    # bond_constraint = torch.nn.functional.huber_loss(bond_constraint, torch.zeros_like(bond_constraint), reduction=
    # 'none')

    # no_bond_constraint = torch.nn.functional.huber_loss(no_bond_constraint, torch.zeros_like(no_bond_constraint), reduction=
    # 'none')

    constraints = bond_constraint + no_bond_constraint
    input = torch.zeros(len(batch_ptr) - 1).to(positions.device)

    # assuming fully-connected graph
    sys_sizes = batch_ptr[1:] - batch_ptr[:-1]
    n_edges = sys_sizes * (sys_sizes - 1)
    edge_index = torch.arange(n_edges.shape[0]).to(positions.device)
    edge_index = edge_index.repeat_interleave(n_edges)
    input = input.scatter_reduce(0, edge_index, constraints, reduce="sum")

    return input * alpha


@contextmanager
def wandb_mode_disabled():
    """
    Temporarily sets the WANDB_MODE environment variable to 'disabled'.
    Saves the original value of WANDB_MODE, sets it to 'disabled' within the context,
    and restores the original value when exiting the context.
    """
    original_mode = os.environ.get("WANDB_MODE")
    try:
        os.environ["WANDB_MODE"] = "disabled"
        yield
    finally:
        if original_mode is None:
            del os.environ["WANDB_MODE"]
        else:
            os.environ["WANDB_MODE"] = original_mode


class FairChemEnergy(torch.nn.Module):
    def __init__(
        self,
        model_ckpt: Optional[str] = None,
        tau: float = 1e-3,
        alpha: float = 1e3,
        device: str = "cpu",
        default_regularize: bool = True,
    ):
        super().__init__()
        if model_ckpt is None:
            model_ckpt = hf_hub_download(
                repo_id="facebook/sgad", filename="esen_spice.pt"
            )
        with wandb_mode_disabled():
            calculator = OCPCalculator(
                checkpoint_path=model_ckpt, cpu=(device == "cpu"), seed=0
            )
        self.predictor = calculator.trainer
        self.predictor.model.device = device

        self.predictor.model.backbone.use_pbc = True
        self.predictor.model.backbone.use_pbc_single = False

        self.device = device
        self.tau = tau  # temperature
        self.alpha = alpha  # regularization strength
        self.r_max = self.predictor.model.backbone.cutoff
        self.atomic_numbers = torch.arange(100)
        self.default_regularize = default_regularize

    def bond_regularizer(self, batch):
        energy_reg = bond_structure_regularizer(
            batch["pos"],
            batch["edge_attrs"][:, 0].unsqueeze(-1),  # bond limits
            batch["edge_attrs"][:, 1].unsqueeze(-1),  # bond types
            batch["edge_index"],
            batch["ptr"],
            alpha=self.alpha,
        )
        grad_outputs = [torch.ones_like(energy_reg)]
        gradient = torch.autograd.grad(
            outputs=[energy_reg],  # [n_graphs, ]
            inputs=[batch["pos"]],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=False,  # Make sure the graph is not destroyed during training
            create_graph=False,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]
        return energy_reg, -gradient

    def __call__(self, batch, regularize: Optional[bool] = None):
        if regularize is None:
            regularize = self.default_regularize

        # rename/add required input fields for fairchem model
        batch.natoms = batch.ptr[1:] - batch.ptr[:-1]
        batch.atomic_numbers = batch["node_attrs"].argmax(dim=-1)

        # TODO use fake large cell
        batch.cell = (
            batch.cell.view(-1, 3, 3)
            + torch.eye(3).to(batch.cell.device).unsqueeze(0) * 1e3
        ).float()
        batch.pos = batch.positions.float()  # wrap?
        # note that our model has otf graph. edge_index here is not used.

        output_dict = {}
        preds = self.predictor._forward(batch)
        for target_key in self.predictor.config["outputs"]:
            output_dict[target_key] = self.predictor._denorm_preds(
                target_key, preds[target_key], batch
            )

        if regularize:
            reg_energy, reg_force = self.bond_regularizer(batch)
            output_dict["reg_forces"] = reg_force.detach()  # / self.tau
            output_dict["reg_energy"] = reg_energy.detach()
        else:
            output_dict["reg_forces"] = torch.zeros_like(output_dict["forces"])
            output_dict["reg_energy"] = torch.zeros_like(output_dict["energy"])

        output_dict["forces"] = (output_dict["forces"].detach()) / self.tau
        output_dict["energy_grad"] = output_dict["forces"]
        return output_dict
