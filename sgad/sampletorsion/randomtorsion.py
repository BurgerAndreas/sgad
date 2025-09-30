# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import torch

from sgad.sampletorsion.rotate import set_torsions
from sgad.utils.data_utils import subtract_com_batch


def get_random_torsions(graph0):
    graph_state = graph0.clone()
    graph_state["torsions"] = (
        torch.rand_like(graph_state["torsions"]) * 2 * math.pi - math.pi
    )
    graph_state["positions"] = set_torsions(
        graph_state["torsions"],
        graph_state["tor_index"],
        graph_state["index_to_rotate"],
        graph_state["positions"],
        graph_state["n_torsions"].max().item(),
        graph_state["tor_per_mol_label"],
    )
    graph_state["positions"] = subtract_com_batch(
        graph_state["positions"], graph_state["batch"]
    )
    return graph_state
