# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from typing import Tuple, Union

import torch

from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from sgad.sampletorsion.rotate import grad_pos_E_to_grad_tor_E
from sgad.sampletorsion.score_model import TensorProductScoreModel
from sgad.sampletorsion.torsion import check_torsions

from sgad.utils.data_utils import subtract_com_batch, subtract_com_vector


class OurTensorProductScoreModel(TensorProductScoreModel):
    def forward(self, t: torch.Tensor, data) -> torch.Tensor:
        data = data.clone()  # this makes it safe to adjust the graph
        _, counts = data.batch.unique(return_counts=True)
        # data.old_edge_index = data.edge_index.clone()  # this should be unecessary since we clone the input
        data.edge_index = data.rdkit_edge_index
        data.t = torch.repeat_interleave(t, counts)
        shf = math.log(self.sigma_min)
        scl = math.log(self.sigma_max) - shf
        # data.node_sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        data.node_sigma = torch.exp(data.t * scl + shf)
        data = super().forward(data)
        # data.edge_index = data.old_edge_index  # this should be unecessary since we clone the input
        return data.edge_pred


class SimpleMLP(nn.Module):
    def __init__(self, n_particles, n_spatial_dim, hid_dim, n_hid_layers, freqs=3):
        super().__init__()
        self.n_particles = n_particles
        self.n_spatial_dim = n_spatial_dim
        self.state_dim = int(n_particles * n_spatial_dim)

        activation = nn.GELU

        # add 1 for time parameterization, should use something better like time embedding
        layers = []
        layers.append(nn.Linear(self.state_dim + 2 * freqs, hid_dim))
        layers.append(activation())

        for i in range(n_hid_layers):
            layers.append(nn.Linear(hid_dim, hid_dim))
            layers.append(activation())

        layers.append(nn.Linear(hid_dim, self.state_dim))

        self.u_t = nn.Sequential(*layers)
        self.register_buffer("freqs", torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t, z):
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        z = torch.cat((z, t), dim=-1)
        ut = self.u_t(z)
        return subtract_com_vector(ut, self.n_particles, self.n_spatial_dim)


class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        n_atoms=10,
        hidden_nf=64,
        n_layers=4,
        agg="sum",
        uniform=False,
    ):
        super().__init__()
        in_node_nf = n_atoms + 1 if not uniform else 1
        in_edge_nf = 3 if not uniform else 1
        self.uniform = uniform
        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            agg=agg,
        )

    def forward(self, t, batch):
        x = batch["positions"]
        edge_index = batch["edge_index"]
        batch_index = batch["batch"]
        h = torch.ones(x.shape[0], 1).to(x.device)
        h = h * t[batch_index, None]

        edge_attr = torch.sum(
            (x[edge_index[0]] - x[edge_index[1]]) ** 2, dim=1, keepdim=True
        )  # .double()
        if not self.uniform:
            h_atom_types = batch["node_attrs"]  # .double()
            h = torch.cat([h_atom_types, h], dim=-1)  # .double()
            bond_one_hot = torch.nn.functional.one_hot(batch["edge_attrs"][:, 1].long())
            edge_attr = torch.cat([bond_one_hot, edge_attr], dim=-1)
        x_final, _ = self.egnn(x, h, edge_index, edge_attr)
        return subtract_com_batch(x_final - x, batch_index)


class EGNN_dynamics_torsion(EGNN_dynamics):
    def __init__(self, check: bool = True, *args, **kwargs):
        self.check = check
        super().__init__(*args, **kwargs)

    def forward(self, t, batch):
        check_torsions(batch["positions"], batch["tor_index"], batch["torsions"])
        ut_x = super().forward(t, batch)
        return grad_pos_E_to_grad_tor_E(
            ut_x,
            batch["torsions"],
            batch["tor_index"],
            batch["index_to_rotate"],
            batch["positions"],
            batch["n_torsions"].max().item(),
            batch["tor_per_mol_label"],
        )


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        n_layers=4,
        out_node_nf=None,
        # coords_range=15,
        agg="sum",
    ):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        # self.coords_range_layer = float(coords_range) / self.n_layers
        # if agg == "mean":
        #     self.coords_range_layer = self.coords_range_layer * 19
        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    in_edge_nf,
                    hidden_channels=self.hidden_nf,
                    aggr=agg,
                ),
            )

    def forward(self, x, h, edges, edge_attr):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            x, h = self._modules["gcl_%d" % i](x, h, edge_attr, edges)
        h = self.embedding_out(h)

        # # Important, the bias of the last linear might be non-zero
        # if node_mask is not None:
        #     h = h * node_mask
        return x, h


class ResWrapper(torch.nn.Module):
    def __init__(self, module, dim_res=2):
        super(ResWrapper, self).__init__()
        self.module = module
        self.dim_res = dim_res

    def forward(self, x):
        res = x[:, : self.dim_res]
        out = self.module(x)
        return out + res


class E_GCL(MessagePassing):
    """EGNN layer from https://arxiv.org/pdf/2102.09844.pdf"""

    def __init__(
        self,
        channels_h: Union[int, Tuple[int, int]],
        channels_m: Union[int, Tuple[int, int]],
        channels_a: Union[int, Tuple[int, int]],
        aggr: str = "add",
        hidden_channels: int = 64,
        **kwargs,
    ):
        super(E_GCL, self).__init__(aggr=aggr, **kwargs)

        self.phi_e = nn.Sequential(
            nn.Linear(2 * channels_h + 1 + channels_a, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, channels_m),
            nn.LayerNorm(channels_m),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(channels_m, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.phi_h = ResWrapper(
            nn.Sequential(
                nn.Linear(channels_h + channels_m, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, channels_h),
            ),
            dim_res=channels_h,
        )

    def forward(self, x, h, edge_attr, edge_index, c=None):
        if c is None:
            c = degree(edge_index[0], x.shape[0]).unsqueeze(-1)
        return self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=edge_attr, c=c)

    def message(self, x_i, x_j, h_i, h_j, edge_attr):
        mh_ij = self.phi_e(
            torch.cat(
                [h_i, h_j, torch.norm(x_i - x_j, dim=-1, keepdim=True) ** 2, edge_attr],
                dim=-1,
            )
        )
        mx_ij = (x_i - x_j) * self.phi_x(mh_ij)
        return torch.cat((mx_ij, mh_ij), dim=-1)

    def update(self, aggr_out, x, h, edge_attr, c):
        m_len = 3
        m_x, m_h = aggr_out[:, :m_len], aggr_out[:, m_len:]
        h_l1 = self.phi_h(torch.cat([h, m_h], dim=-1))
        x_l1 = x + (m_x / c)
        return x_l1, h_l1


def update_batch_for_fairchem(batch):
    natoms = batch.batch.new_zeros(batch.batch.max().item() + 1)
    natoms.scatter_add_(0, batch.batch, torch.ones_like(batch.batch))
    batch.natoms = natoms
    batch.pos = batch.positions
    # One hot encoding to atomic numbers
    batch.atomic_numbers = torch.argmax(batch["node_attrs"], dim=-1) + 1
    # Hack: Add random jitter to the positions since all
    # positions are equal to zero once in a while
    batch.pos = batch.pos + torch.randn_like(batch.pos) * 1e-6
    batch.tags = torch.zeros_like(batch.batch)
    return batch
