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
# GAD with two state variables
##################################################################


def v_dot_fn(x, v):
    # Calculate Hessian
    H_x = V_hess(x)

    # Calculate v' terms
    Hv = H_x @ v
    vHv = jnp.dot(v, Hv)

    return -Hv + vHv * v / jnp.dot(v, v)


def x_dot_fn(x, v):
    grad_x = grad_V(x)
    return -grad_x + 2 * jnp.dot(grad_x, v) * v / jnp.dot(v, v)


# Define the dynamics
def dynamics(state):
    x, v = state[:2], state[2:]

    x_dot = x_dot_fn(x, v)

    # Calculate v' = -Hv + vHv * v
    v_dot = v_dot_fn(x, v)

    return jnp.array([x_dot[0], x_dot[1], v_dot[0], v_dot[1]])


dynamics = jax.jit(dynamics)

# Set up integration
dt = 0.01
T = 600
times = jnp.arange(T) * dt

# Initial conditions
initial_direction = jnp.array([-0.9, 0.1])
initial_direction = initial_direction / jnp.linalg.norm(initial_direction)
x0 = jnp.array([-0.9, 0.9, *initial_direction])


# Simple Euler integration
def integrate():
    trajectory = [x0]
    state = x0

    for _ in range(T - 1):
        # Euler step
        state = state + dt * dynamics(state)
        trajectory.append(state)

    return jnp.array(trajectory)


# Run integration
trajectory = integrate()

# Plot results
plt.close()
plt.figure(figsize=(12, 10))

# Plot potential surface
plt.contour(X, Y, Z, levels=20)
plt.contourf(X, Y, Z, levels=20, alpha=0.7)
plt.colorbar(label="Potential Energy")

# Plot trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", label="Trajectory")
plt.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start")
plt.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", label="End")

plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Particle Trajectory in Double Well Potential")
plt.legend()
plt.savefig("plots/gad2d.png")
print("Saved plot to plots/gad2d.png")
plt.close()


##################################################################
# GAD with one state variable
##################################################################


# Define the dynamics
def eigvec_dynamics(state):
    grad_x = grad_V(state)

    H = V_hess(state)
    eig_vals, eig_vecs = jnp.linalg.eigh(H)
    smallest_eigvec = eig_vecs[0]

    x_dot = -grad_x + 2 * jnp.dot(grad_x, smallest_eigvec) * smallest_eigvec

    return x_dot


eigvec_dynamics = jax.jit(eigvec_dynamics)

# Set up integration
dt = 0.01
T = 500
times = jnp.arange(T) * dt

# Initial conditions
x0 = jnp.array([-0.6, 0.6])


# Simple Euler integration
def integrate():
    trajectory = [x0]
    state = x0

    for _ in range(T - 1):
        # Euler step
        state = state + dt * eigvec_dynamics(state)
        trajectory.append(state)

    return jnp.array(trajectory)


eigvec_dynamics_vmap = jax.vmap(eigvec_dynamics)
eigvec_dynamics_grid = eigvec_dynamics_vmap(grid_flat).reshape(N, N, 2)

# Run integration
trajectory = integrate()

# Plot results
plt.close()
skip = 10
plt.figure(figsize=(12, 10))

# Plot potential surface
plt.contour(X, Y, Z, levels=20)
plt.contourf(X, Y, Z, levels=20, alpha=0.7)
plt.colorbar(label="Potential Energy")

# Plot trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], "r-", label="Trajectory")
plt.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start")
plt.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", label="End")

plt.quiver(
    X[::skip, ::skip],
    Y[::skip, ::skip],
    eigvec_dynamics_grid[::skip, ::skip, 0],
    eigvec_dynamics_grid[::skip, ::skip, 1],
    color="white",
    alpha=0.8,
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Particle Trajectory in Double Well Potential")
plt.legend()
plt.savefig("plots/gad2d_eigvec.png")
print("Saved plot to plots/gad2d_eigvec.png")
plt.close()
