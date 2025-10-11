# torch>=2.1
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- Utilities ----------


@torch.no_grad()
def gad_field_from_calc(forces_flat, hess, eps=1e-12):
    """
    forces_flat: (B, D) physical forces  (i.e., -∇V). D=3N
    hess:       (B, D, D) Hessian of V at same configs
    Returns F_GAD(x) in flat (B, D).
    """
    # ∇V = -forces
    gradV = -forces_flat
    # smallest-eig eigenvector of Hessian
    # symmetrize for numerical stability
    Hs = 0.5 * (hess + hess.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(Hs)  # (B,D), (B,D,D)
    v = evecs[..., :, 0]  # (B,D) eigenvector with smallest eigenvalue
    vv = (v * v).sum(dim=-1, keepdim=True).clamp_min(eps)  # (B,1)
    gv = (gradV * v).sum(dim=-1, keepdim=True)  # (B,1)
    Fgad = -gradV + 2.0 * (gv / vv) * v  # (B,D)
    return Fgad


def flatten_xyz(x):  # x: (B,N,3) -> (B,3N)
    return x.reshape(x.shape[0], -1)


def unflatten_xyz(x_flat, N):  # (B,3N) -> (B,N,3)
    return x_flat.reshape(x_flat.shape[0], N, 3)


# ---------- Model ----------


class ScoreNet(nn.Module):
    """Simple MLP that predicts a vector field; conditioned on noise σ."""

    def __init__(self, dim, hidden=512, k=8):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(dim + 2 * k, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def sigma_embed(self, sigma):  # sigma: (B,)
        # Fourier features of log σ
        B = sigma.shape[0]
        logsig = sigma.log().unsqueeze(1)  # (B,1)
        freqs = (2.0 ** torch.arange(self.k, device=sigma.device)).view(1, -1)
        s = torch.sin(freqs * logsig)
        c = torch.cos(freqs * logsig)
        return torch.cat([s, c], dim=1)  # (B,2k)

    def forward(self, x_flat, sigma):  # x_flat: (B,D)
        emb = self.sigma_embed(sigma)
        return self.net(torch.cat([x_flat, emb], dim=1))  # (B,D)


# ---------- Data stub ----------


class ConfigDataset(Dataset):
    """
    Replace `self.X` with your configurations tensor of shape (M,N,3).
    """

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i]


# ---------- Calculator adapter ----------
# Expect a user-supplied `calculator(x_one)` that accepts (N,3) tensor (or numpy)
# and returns dict: { "forces": (N,3), "energy": float, "hessian": (3N,3N) }.


def batch_calculator(x_batch, calculator):
    B, N, _ = x_batch.shape
    forces = torch.zeros(B, N, 3, dtype=x_batch.dtype, device=x_batch.device)
    hess = torch.zeros(B, 3 * N, 3 * N, dtype=x_batch.dtype, device=x_batch.device)
    # loop; many external calculators are non-batched
    for b in range(B):
        out = calculator(x_batch[b])  # user-provided
        f = torch.as_tensor(out["forces"], dtype=x_batch.dtype, device=x_batch.device)
        H = torch.as_tensor(out["hessian"], dtype=x_batch.dtype, device=x_batch.device)
        forces[b] = f
        hess[b] = H
    return forces, hess


# ---------- Training ----------


def train_on_gad(
    calculator,  # external oracle
    init_confs,  # (M,N,3) training configurations
    batch_size=32,
    epochs=50,
    lr=2e-4,
    sigma_min=1e-3,
    sigma_max=1e-1,
    num_sigmas=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    M, N, _ = init_confs.shape
    D = 3 * N
    model = ScoreNet(D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ds = ConfigDataset(init_confs.to(device))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    sigmas = torch.exp(
        torch.linspace(
            math.log(sigma_max), math.log(sigma_min), num_sigmas, device=device
        )
    )

    for epoch in range(epochs):
        for x in dl:  # x: (B,N,3)
            x = x.to(device)
            B = x.shape[0]
            # pick noise level per-sample
            idx = torch.randint(len(sigmas), (B,), device=device)
            sigma = sigmas[idx]
            noise = torch.randn_like(x) * sigma.view(B, 1, 1)
            x_noisy = x + noise

            # oracle GAD field at x_noisy
            forces, hess = batch_calculator(x_noisy, calculator)
            Fgad = gad_field_from_calc(flatten_xyz(forces), hess)  # (B,D)

            # predict
            pred = model(flatten_xyz(x_noisy), sigma)  # (B,D)
            loss = F.mse_loss(pred, Fgad)

            opt.zero_grad()
            loss.backward()
            opt.step()
        # optional: print(loss.item())

    return model, sigmas


# ---------- Sampling ----------


@torch.no_grad()
def annealed_langevin_sampling(
    model, sigmas, N, num_samples=64, steps_per_sigma=50, step_size=1e-2, device=None
):
    """
    Not a valid reverse-SDE unless the field is a true score. It is a heuristic generator.
    """
    device = device or next(model.parameters()).device
    D = 3 * N
    x = (
        torch.randn(num_samples, D, device=device) * sigmas[0]
    )  # init from wide Gaussian

    for sigma in sigmas:
        s = sigma.repeat(num_samples)
        for _ in range(steps_per_sigma):
            v = model(x, s)  # (B,D)
            noise = torch.randn_like(x)
            x = x + step_size * v + math.sqrt(2.0 * step_size) * noise
    return unflatten_xyz(x, N)  # (num_samples,N,3)


# ---------- Post-filter to index-1 saddles (optional) ----------


@torch.no_grad()
def project_with_gad(calculator, X, iters=200, eta=0.05):
    """
    Run explicit GAD flow from X to approach GAD fixed points.
    """
    B, N, _ = X.shape
    x = X.clone()
    for _ in range(iters):
        forces, hess = batch_calculator(x, calculator)
        Fgad = gad_field_from_calc(flatten_xyz(forces), hess)
        x = unflatten_xyz(flatten_xyz(x) + eta * Fgad, N)
    return x


@torch.no_grad()
def is_index1_saddle(calculator, x, tol=1e-8):
    out = calculator(x)
    H = torch.as_tensor(out["hessian"], dtype=torch.float64)
    Hs = 0.5 * (H + H.T)
    evals = torch.linalg.eigvalsh(Hs)
    neg = (evals < -tol).sum().item()
    return neg == 1


# ---------- Example usage (placeholders) ----------
# Provide your own `calculator` and `init_confs` before running.
# model, sigmas = train_on_gad(calculator, init_confs)           # train
# samples = annealed_langevin_sampling(model, sigmas, N)         # generate candidates
# samples = project_with_gad(calculator, samples)                 # drive to GAD fixed points
# flags = [is_index1_saddle(calculator, s) for s in samples]      # optional check
