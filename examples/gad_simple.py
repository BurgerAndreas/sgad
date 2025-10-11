import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# jax.config.update("jax_enable_x64", True)
def V(x):
    return 0.25 * (x[0] ** 2 - 1) ** 2 + 3 * x[1] ** 2


N = 500
# Create a 2D grid
lim = 2
x = jnp.linspace(-lim, lim, N)
y = jnp.linspace(-lim, lim, N)
X, Y = jnp.meshgrid(x, y)
grid_flat = jnp.stack([X.flatten(), Y.flatten()], axis=1)

V_vmap = jax.vmap(V)
V_vmap = jax.jit(V_vmap)
V = jax.jit(V)

# Calculate gradient using JAX for a single point
grad_V = lambda x: jax.grad(V)(x)
grad_V_vmap = jax.vmap(grad_V)
grad_V_vmap = jax.jit(grad_V_vmap)
grad_V = jax.jit(grad_V)


# Calculate the potential values
Z = V_vmap(grid_flat).reshape(N, N)

# Calculate gradients
V_grad = grad_V_vmap(grid_flat).reshape(N, N, 2)


V_hess = jax.hessian(V)
V_hess_vmap = jax.vmap(V_hess)
V_hess_vmap = jax.jit(V_hess_vmap)
V_hess = jax.jit(V_hess)

##################################################################
# GAD with one state variable
##################################################################


# Define the dynamics
def eigvec_dynamics(state):
    """GAD field"""
    grad_x = grad_V(state)

    H = V_hess(state)
    eig_vals, eig_vecs = jnp.linalg.eigh(H)
    smallest_eigvec = eig_vecs[0]

    x_dot = -grad_x + 2 * jnp.dot(grad_x, smallest_eigvec) * smallest_eigvec

    return x_dot


eigvec_dynamics = jax.jit(eigvec_dynamics)
