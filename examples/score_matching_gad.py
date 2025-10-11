# @title
"""
Score-matching toy example in JAX.

This script trains a time-conditional MLP to predict the score (\nabla_x log q_t)
under a simple forward noising process, then samples with a reverse-time vector
field (probability flow when xi=0, Langevin-like when xi>0). It also visualizes
the forward diffusion and the generated samples across time.
"""

import jax
import numpy as np

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import flax

import optax

from flax import linen as nn
from flax.training import train_state

from tqdm import trange
from functools import partial
from matplotlib import pyplot as plt

# Data generation

from typing import NamedTuple, Any


def sample_data(key, bs):
    """Sample a 2D four-mode dataset on a grid with small Gaussian noise."""
    keys = random.split(key, 3)
    x_1 = random.randint(keys[0], minval=0, maxval=2, shape=(bs, 2))
    x_1 = 3 * (x_1.astype(jnp.float32) - 0.5)
    x_1 += 4e-1 * random.normal(keys[1], shape=(bs, 2))
    return x_1


# Time and noise schedule (simple closed-form choices for demo purposes)
t_0, t_1 = 0.0, 1.0
beta_0 = 0.1
beta_1 = 20.0
log_alpha = lambda t: -0.5 * t * beta_0 - 0.25 * t**2 * (beta_1 - beta_0)
# log_sigma = lambda t: jnp.log(jnp.sqrt(-jnp.expm1(-t*beta_0-0.5*t**2*(beta_1-beta_0))))
log_sigma = lambda t: jnp.log(t)
dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())
# beta_t = s_t d/dt log(s_t/alpha_t)
# beta = lambda t: jnp.exp(log_sigma(t))*(dlog_sigmadt(t) - dlog_alphadt(t))
beta = lambda t: (1 + 0.5 * t * beta_0 + 0.5 * t**2 * (beta_1 - beta_0))


def q_t(key, data, t):
    """Forward noising: x_t = alpha(t) * x_0 + sigma(t) * eps."""
    eps = random.normal(key, shape=data.shape)
    x_t = jnp.exp(log_alpha(t)) * data + jnp.exp(log_sigma(t)) * eps
    return eps, x_t


def q_0(state):
    """Analytical PDF of q_t at t=0 for array input state = [x, y]"""
    centers = jnp.array([[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]])
    sigma = 0.4

    # Sum of 4 Gaussians
    pdf = jnp.zeros_like(state[0])
    for c in centers:
        pdf += jnp.exp(
            -((state[0] - c[0]) ** 2 + (state[1] - c[1]) ** 2) / (2 * sigma**2)
        )
    pdf /= 2 * jnp.pi * sigma**2 * 4  # Normalize
    return jnp.log(pdf)


seed = 0
np.random.seed(seed)
key = random.PRNGKey(seed)
bs = 512
t_axis = np.linspace(0.0, 1.0, 6)
TRAIN_ON_GAD = True  # train directly on GAD field; skip epsilon loss and sampling

# Visualize the forward process q_t(x|x_0) at a few time points
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs), t_axis[i])
    plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.3)
    plt.title(f"t={t_axis[i]}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()


class MLP(nn.Module):
    num_hid: int
    num_out: int

    @nn.compact
    def __call__(self, t, x):
        h = jnp.hstack([t, x])
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.relu(h)
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=self.num_hid)(h)
        h = nn.swish(h)
        h = nn.Dense(features=self.num_out)(h)
        return h


# Model and optimizer setup
model = MLP(num_hid=512, num_out=x_t.shape[1])

key, init_key = random.split(key)
optimizer = optax.adam(learning_rate=2e-4)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=model.init(init_key, np.ones([bs, 1]), x_t),
    tx=optimizer,
)


# def sm_loss(state, key, params, bs):
#     """Score matching loss"""
#     keys = random.split(
#         key,
#     )
#     sdlogqdx = lambda _t, _x: state.apply_fn(params, _t, _x)
#     # Sample minibatch and time; construct noisy inputs and target noise
#     data = sample_data(keys[0], bs)
#     t = random.uniform(keys[1], [bs, 1])
#     eps, x_t = q_t(keys[2], data, t)
#     loss = ((eps + sdlogqdx(t, x_t)) ** 2).sum(1)
#     return loss.mean()


q_0_vmap = jax.vmap(q_0)
q_0_vmap = jax.jit(q_0_vmap)

grad_q0 = jax.grad(q_0)
q0_hess = jax.hessian(q_0)


def gad_vector_field(state):
    """Compute vector field based on smallest eigenvector of Hessian"""
    grad_x = grad_q0(state)

    H = q0_hess(state)
    eig_vals, eig_vecs = jnp.linalg.eigh(H)
    smallest_eigvec = eig_vecs[0]

    x_dot = -grad_x + 2 * jnp.dot(grad_x, smallest_eigvec) * smallest_eigvec

    return x_dot


gad_vector_field_vmap = jax.vmap(gad_vector_field)
gad_vector_field_vmap = jax.jit(gad_vector_field_vmap)


def gad_loss(state, key, params, bs):
    keys = random.split(key)
    x0 = sample_data(keys[0], bs)
    t0 = jnp.zeros((bs, 1))
    target = gad_vector_field_vmap(x0)
    pred = state.apply_fn(params, t0, x0)
    loss = ((pred - target) ** 2).sum(1)
    return loss.mean()


@partial(jax.jit, static_argnums=1)
def gad_train_step(state, bs, key):
    grad_fn = jax.value_and_grad(gad_loss, argnums=2)
    loss, grads = grad_fn(state, key, state.params, bs)
    state = state.apply_gradients(grads=grads)
    return state, loss


num_iterations_gad = 5_000
loss_plot_gad = np.zeros(num_iterations_gad)
key, loop_key = random.split(key)
for iter in trange(num_iterations_gad):
    state, loss = gad_train_step(state, bs, random.fold_in(loop_key, iter))
    loss_plot_gad[iter] = loss

plt.figure(figsize=(6, 4))
plt.plot(loss_plot_gad)
plt.grid()
fname = "plots/sm_simple_jax_gad_loss.png"
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()


#############################
# Sampling with learned GAD #
#############################

# Deterministic integration along the learned GAD field
dt_gad = 1e-2
n_gad = 500
key, ikey = random.split(key, num=2)
x_gen_gad = jnp.zeros((bs, n_gad + 1, 2))
x_gen_gad = x_gen_gad.at[:, 0, :].set(random.normal(ikey, shape=(bs, 2)))
for i in trange(n_gad):
    t0 = jnp.zeros((bs, 1))
    v = state.apply_fn(state.params, t0, x_gen_gad[:, i, :])
    x_gen_gad = x_gen_gad.at[:, i + 1, :].set(x_gen_gad[:, i, :] + dt_gad * v)

# Plot snapshots across integration
plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    step_idx = int(n_gad * t_axis[i])
    plt.scatter(
        x_gen_gad[:, step_idx, 0],
        x_gen_gad[:, step_idx, 1],
        label="gad_gen",
        alpha=0.7,
    )
    plt.title(f"step={step_idx}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = "plots/sm_simple_jax_gad_samples.png"
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()


###############################
# Stochastic GAD variant (SDE)
###############################

# Langevin-like integration: x_{k+1} = x_k + dt * v(x_k) + sqrt(2*eta*dt) * N(0, I)
dt_gad_s = 1e-2
n_gad_s = 500
eta = 0.5  # noise scale; smaller -> closer to deterministic
key, ikey = random.split(key, num=2)
x_gen_gad_stoch = jnp.zeros((bs, n_gad_s + 1, 2))
x_gen_gad_stoch = x_gen_gad_stoch.at[:, 0, :].set(random.normal(ikey, shape=(bs, 2)))
key, loop_key = random.split(key)
for i in trange(n_gad_s):
    t0 = jnp.zeros((bs, 1))
    v = state.apply_fn(state.params, t0, x_gen_gad_stoch[:, i, :])
    noise = random.normal(random.fold_in(loop_key, i), shape=(bs, 2))
    x_next = (
        x_gen_gad_stoch[:, i, :] + dt_gad_s * v + jnp.sqrt(2 * eta * dt_gad_s) * noise
    )
    x_gen_gad_stoch = x_gen_gad_stoch.at[:, i + 1, :].set(x_next)

plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    step_idx = int(n_gad_s * t_axis[i])
    plt.scatter(
        x_gen_gad_stoch[:, step_idx, 0],
        x_gen_gad_stoch[:, step_idx, 1],
        label="gad_gen_stoch",
        alpha=0.7,
    )
    plt.title(f"step={step_idx}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = "plots/sm_simple_jax_gad_samples_stoch.png"
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()
