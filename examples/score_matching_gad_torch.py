"""
Score-matching toy example in PyTorch.

This script mirrors `examples/score_matching_gad.py` but replaces JAX with PyTorch.
It trains a time-conditional MLP to predict a GAD vector field derived from the
analytical log-density of a simple 2D four-mode dataset, then integrates the
learned field deterministically and with Langevin-like noise. Forward process
snapshots and generated samples are saved as plots.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, hessian, vmap

from tqdm import trange
from matplotlib import pyplot as plt

base_dir = "plots/score_matching_gad_torch"
os.makedirs(base_dir, exist_ok=True)


# -----------------------
# Data generation (Torch)
# -----------------------
def sample_data(batch_size: int) -> torch.Tensor:
    """Sample a 2D four-mode dataset on a grid with small Gaussian noise."""
    x = torch.randint(low=0, high=2, size=(batch_size, 2))
    x = 3.0 * (x.to(torch.float32) - 0.5)
    x = x + 4e-1 * torch.randn((batch_size, 2))
    return x


# ---------------------------------------------
# Time and noise schedule (kept simple, Torch)
# ---------------------------------------------
t_0, t_1 = 0.0, 1.0
beta_0 = 0.1
beta_1 = 20.0


def log_alpha(t: torch.Tensor) -> torch.Tensor:
    return -0.5 * t * beta_0 - 0.25 * t * t * (beta_1 - beta_0)


def log_sigma(t: torch.Tensor) -> torch.Tensor:
    # log(t); exp(log_sigma(0)) == 0 is fine here
    return torch.log(t)


def q_t(data: torch.Tensor, t: torch.Tensor):
    """Forward noising: x_t = alpha(t) * x_0 + sigma(t) * eps."""
    eps = torch.randn_like(data)
    x_t = torch.exp(log_alpha(t)) * data + torch.exp(log_sigma(t)) * eps
    return eps, x_t


# --------------------------
# Analytical q_0 and its GAD
# --------------------------
centers = torch.tensor(
    [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]], dtype=torch.float32
)
sigma = 0.4


def q0_logpdf(x: torch.Tensor) -> torch.Tensor:
    """Analytical log-PDF of q_0 for a single point x in R^2.

    Args:
        x: shape (2,)
    Returns:
        scalar tensor: log q_0(x)
    """
    diffs = x - centers  # (4, 2)
    sq = (diffs * diffs).sum(dim=1)  # (4,)
    # log-sum-exp for numerical stability
    log_terms = -sq / (2.0 * (sigma**2))  # (4,)
    norm_const = math.log(4.0) + math.log(2.0 * math.pi * (sigma**2))
    return torch.logsumexp(log_terms, dim=0) - torch.tensor(
        norm_const, dtype=torch.float32
    )


def gad_vector_field_batch(x_batch: torch.Tensor) -> torch.Tensor:
    """Compute GAD vector field for a batch of points (B, 2).

    Uses batched jacobians/hessians when torch.func is available; otherwise
    falls back to per-sample autograd.functional.hessian.
    """
    grad_f = jacrev(q0_logpdf)
    hess_f = hessian(q0_logpdf)
    grad_x = vmap(grad_f)(x_batch)  # (B, 2)
    H = vmap(hess_f)(x_batch)  # (B, 2, 2)
    eig_vals, eig_vecs = torch.linalg.eigh(H)  # batched eigen-decomp
    smallest_eigvec = eig_vecs[..., :, 0]  # (B, 2)
    dots = (grad_x * smallest_eigvec).sum(dim=1, keepdim=True)  # (B, 1)
    x_dot = -grad_x + 2.0 * dots * smallest_eigvec  # (B, 2)
    return x_dot.detach()


# --------------
# Torch MLP model
# --------------
class MLP(nn.Module):
    def __init__(self, num_hid: int, num_out: int):
        super().__init__()
        self.fc1 = nn.Linear(3, num_hid)
        self.fc2 = nn.Linear(num_hid, num_hid)
        self.fc3 = nn.Linear(num_hid, num_hid)
        self.fc4 = nn.Linear(num_hid, num_out)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([t, x], dim=1)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.silu(h)
        h = self.fc3(h)
        h = F.silu(h)
        h = self.fc4(h)
        return h


# -----------------------------
# Setup seeds, data and training
# -----------------------------
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
g = torch.Generator().manual_seed(seed)

bs = 512
t_axis = np.linspace(0.0, 1.0, 6)

# Visualize the forward process q_t(x|x_0) at a few time points
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    data = sample_data(bs)
    t_val = torch.full((bs, 1), float(t_axis[i]), dtype=torch.float32)
    _, x_t = q_t(data, t_val)
    x_np = x_t.detach().numpy()
    plt.scatter(x_np[:, 0], x_np[:, 1], alpha=0.3)
    plt.title(f"t={t_axis[i]}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()


# Model and optimizer setup
with torch.no_grad():
    # one pass to get dimensionality
    x_dummy = sample_data(bs)
num_out = x_dummy.shape[1]
model = MLP(num_hid=512, num_out=num_out)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def gad_loss(batch_size: int) -> torch.Tensor:
    x0 = sample_data(batch_size)
    t0 = torch.zeros((batch_size, 1), dtype=torch.float32)
    target = gad_vector_field_batch(x0)
    pred = model(t0, x0)
    loss_vec = (pred - target) ** 2
    loss = loss_vec.sum(dim=1).mean()
    return loss


num_iterations_gad = 5_000
loss_plot_gad = np.zeros(num_iterations_gad, dtype=np.float32)
for iter_idx in trange(num_iterations_gad):
    optimizer.zero_grad(set_to_none=True)
    loss = gad_loss(bs)
    loss.backward()
    optimizer.step()
    loss_plot_gad[iter_idx] = float(loss.detach().item())

plt.figure(figsize=(6, 4))
plt.plot(loss_plot_gad)
plt.grid()
fname = os.path.join(base_dir, "loss.png")
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()


# -----------------------------
# Sampling with learned GAD
# -----------------------------

# Deterministic integration along the learned GAD field
dt_gad = 1e-2
n_gad = 500
x_gen_gad = torch.zeros((bs, n_gad + 1, 2), dtype=torch.float32)
x_gen_gad[:, 0, :] = torch.randn((bs, 2))
for i in trange(n_gad):
    t0 = torch.zeros((bs, 1), dtype=torch.float32)
    v = model(t0, x_gen_gad[:, i, :])
    x_gen_gad[:, i + 1, :] = x_gen_gad[:, i, :] + dt_gad * v

# Plot snapshots across integration
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    step_idx = int(n_gad * t_axis[i])
    x_np = x_gen_gad[:, step_idx, :].detach().numpy()
    plt.scatter(x_np[:, 0], x_np[:, 1], label="gad_gen", alpha=0.7)
    plt.title(f"step={step_idx}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = os.path.join(base_dir, "samples.png")
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()


# -----------------------------
# Stochastic GAD variant (SDE)
# -----------------------------

# Langevin-like integration: x_{k+1} = x_k + dt * v(x_k) + sqrt(2*eta*dt) * N(0, I)
dt_gad_s = 1e-2
n_gad_s = 500
eta = 0.5
x_gen_gad_stoch = torch.zeros((bs, n_gad_s + 1, 2), dtype=torch.float32)
x_gen_gad_stoch[:, 0, :] = torch.randn((bs, 2))
for i in trange(n_gad_s):
    t0 = torch.zeros((bs, 1), dtype=torch.float32)
    v = model(t0, x_gen_gad_stoch[:, i, :])
    noise = torch.randn((bs, 2))
    x_next = (
        x_gen_gad_stoch[:, i, :]
        + dt_gad_s * v
        + math.sqrt(2.0 * eta * dt_gad_s) * noise
    )
    x_gen_gad_stoch[:, i + 1, :] = x_next

plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    step_idx = int(n_gad_s * t_axis[i])
    x_np = x_gen_gad_stoch[:, step_idx, :].detach().numpy()
    plt.scatter(x_np[:, 0], x_np[:, 1], label="gad_gen_stoch", alpha=0.7)
    plt.title(f"step={step_idx}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = os.path.join(base_dir, "samples_stoch.png")
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()
