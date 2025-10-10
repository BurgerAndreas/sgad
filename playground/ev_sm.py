import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- score field (analytical via autograd) -----

def softplus(z):
    return torch.log1p(torch.exp(-z.abs())) + z.clamp_min(0)

def score_star(x, energy_fn, mu=0.1):
    """
    x: (N,3) tensor with requires_grad=False/True
    energy_fn: callable(x)-> scalar tensor (must be autograd-compatible)
    returns s*(x): (N,3) tensor
    """
    x = x.clone().detach().requires_grad_(True)
    N, Dp = x.shape[0], 3
    D = 3 * N

    # gradient g = ∇V
    E = energy_fn(x)
    g = torch.autograd.grad(E, x, create_graph=True)[0].reshape(D)

    # Hg without forming H explicitly: J_g^T g
    Hg = torch.autograd.grad(g, x, grad_outputs=g.view_as(x), create_graph=True)[0].reshape(D)

    # Hessian H to get eigenpairs (small to moderate D)
    H = torch.autograd.functional.hessian(lambda y: energy_fn(y), x)
    H = H.reshape(D, D)

    w, Q = torch.linalg.eigh(H)              # symmetric eigendecomp
    idx = torch.argsort(w)                   # order ascending
    w = w[idx]; Q = Q[:, idx]
    v1 = Q[:, 0].detach()
    v2 = Q[:, 1].detach()

    # ∇λ_k = ∂/∂x (v_k^T H v_k) with v_k treated constant
    Hv1 = torch.autograd.grad(g, x, grad_outputs=v1.view_as(x), create_graph=True)[0].reshape(D)
    q1 = (Hv1 * v1).sum()
    grad_l1 = torch.autograd.grad(q1, x, create_graph=True)[0].reshape(D)

    Hv2 = torch.autograd.grad(g, x, grad_outputs=v2.view_as(x), create_graph=True)[0].reshape(D)
    q2 = (Hv2 * v2).sum()
    grad_l2 = torch.autograd.grad(q2, x, create_graph=True)[0].reshape(D)

    r1p  = softplus(w[0]) * torch.sigmoid(w[0])     # r'(λ1)
    r2mp = softplus(-w[1]) * torch.sigmoid(-w[1])   # r'(-λ2)

    gradU = Hg + mu * (r1p * grad_l1 - r2mp * grad_l2)
    s = -gradU
    return s.view_as(x)

# ----- small score net and training (unchanged API) -----

class ScoreNet(nn.Module):
    def __init__(self, dim, width=256, depth=3):
        super().__init__()
        layers = [nn.Linear(dim, width), nn.SiLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def train_on_field(energy_fn, N, steps=2000, batch_size=8, lr=2e-4, mu=0.1, init_std=1.0, device="cpu"):
    torch.manual_seed(0)
    D = 3 * N
    model = ScoreNet(D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for t in range(steps):
        x = init_std * torch.randn(batch_size, N, 3, device=device)
        with torch.no_grad():
            S = torch.stack([score_star(x[b], energy_fn, mu).reshape(-1) for b in range(batch_size)], dim=0)
        pred = model(x.reshape(batch_size, D))
        loss = F.mse_loss(pred, S)
        opt.zero_grad(); loss.backward(); opt.step()
        if (t+1) % max(steps//10,1) == 0:
            print(f"step {t+1}/{steps}  loss {loss.item():.4e}")
    return model

@torch.no_grad()
def langevin_sample(model, n_samples, N, steps=2000, step_size=1e-3, noise=1.0, device="cpu"):
    D = 3 * N
    x = noise * torch.randn(n_samples, D, device=device)
    for _ in range(steps):
        score = model(x)
        x = x + step_size * score + (2.0 * step_size) ** 0.5 * torch.randn_like(x)
    return x.view(n_samples, N, 3).cpu()

# Usage:
# Define energy_fn(x)->scalar Tensor with higher-order derivatives.
# model = train_on_field(energy_fn, N=NUM_ATOMS, steps=1000, batch_size=4, mu=0.1)
# X = langevin_sample(model, n_samples=32, N=NUM_ATOMS, steps=5000, step_size=5e-4)
