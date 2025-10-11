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


def sm_loss(state, key, params, bs):
    """Score matching loss"""
    keys = random.split(
        key,
    )
    sdlogqdx = lambda _t, _x: state.apply_fn(params, _t, _x)
    # Sample minibatch and time; construct noisy inputs and target noise
    data = sample_data(keys[0], bs)
    t = random.uniform(keys[1], [bs, 1])
    eps, x_t = q_t(keys[2], data, t)
    loss = ((eps + sdlogqdx(t, x_t)) ** 2).sum(1)
    return loss.mean()


@partial(jax.jit, static_argnums=1)
def train_step(state, bs, key):
    """One optimization step on the score matching objective."""
    grad_fn = jax.value_and_grad(sm_loss, argnums=2)
    loss, grads = grad_fn(state, key, state.params, bs)
    state = state.apply_gradients(grads=grads)
    return state, loss


key, loc_key = random.split(key)
state, loss = train_step(state, bs, loc_key)


# Training loop
num_iterations = 20_000

loss_plot = np.zeros(num_iterations)
key, loop_key = random.split(key)
for iter in trange(num_iterations):
    state, loss = train_step(state, bs, random.fold_in(loop_key, iter))
    loss_plot[iter] = loss


plt.figure(figsize=(6, 4))
plt.plot(loss_plot)
plt.grid()
fname = "plots/sm_simple_jax_loss.png"
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()


# Reverse-time vector field
# v_t(x) = dlog(alpha)/dt x - s^2_t d/dt log(s_t/alpha_t) dlog q_t(x)/dx
@jax.jit
def vector_field(t, x, xi=0.0):
    """Time-dependent drift used for sampling (xi>0 adds Langevin noise)."""
    sdlogqdx = lambda _t, _x: state.apply_fn(state.params, _t, _x)
    dxdt = (
        dlog_alphadt(t) * x
        - beta(t) * sdlogqdx(t, x)
        - xi * beta(t) / jnp.exp(log_sigma(t)) * sdlogqdx(t, x)
    )
    return dxdt


# Euler–Maruyama sampling of the reverse dynamics
dt = 1e-2
xi = 1.0  # xi=0: deterministic flow; xi>0: stochastic (adds noise)
t = 1.0
n = int(t / dt)
t = t * jnp.ones((bs, 1))
key, ikey = random.split(key, num=2)
x_gen = jnp.zeros((bs, n + 1, x_t.shape[1]))
x_gen = x_gen.at[:, 0, :].set(random.normal(ikey, shape=(bs, x_t.shape[1])))
for i in trange(n):
    key, ikey = random.split(key, num=2)
    dx = -dt * vector_field(t, x_gen[:, i, :], xi) + jnp.sqrt(
        2 * xi * beta(t) * dt
    ) * random.normal(ikey, shape=(bs, 2))
    x_gen = x_gen.at[:, i + 1, :].set(x_gen[:, i, :] + dx)
    t += -dt


plt.figure(figsize=(23, 5))
for i in range(len(t_axis)):
    plt.subplot(1, len(t_axis), i + 1)
    key, *ikey = random.split(key, 3)
    t = t_axis[len(t_axis) - 1 - i]
    _, x_t = q_t(ikey[1], sample_data(ikey[0], bs), t)
    plt.scatter(x_t[:, 0], x_t[:, 1], label="noise_data", alpha=0.3)
    plt.scatter(
        x_gen[:, int(n * (t_axis[i])), 0],
        x_gen[:, int(n * (t_axis[i])), 1],
        label="gen_data",
    )
    plt.title(f"t={t}")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    if i == 0:
        plt.legend(fontsize=15)
fname = "plots/sm_simple_jax.png"
plt.savefig(fname)
print(f"Saved plot to {fname}")
plt.close()
