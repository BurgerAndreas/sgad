# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


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
