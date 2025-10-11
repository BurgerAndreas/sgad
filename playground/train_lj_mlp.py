import argparse
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb


def lennard_jones_energy(r: torch.Tensor, sigma: float, epsilon: float) -> torch.Tensor:
    """Compute Lennard-Jones potential energy for pairwise distance r.

    V(r) = 4 * epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
    r must be strictly positive.
    """
    sr = sigma / r
    sr6 = sr.pow(6)
    sr12 = sr6.pow(2)
    return 4.0 * epsilon * (sr12 - sr6)


def lennard_jones_derivative(
    r: torch.Tensor, sigma: float, epsilon: float
) -> torch.Tensor:
    """d/dr of the Lennard-Jones potential.

    dV/dr = 4 * epsilon * ( -12 * sigma^12 / r^13 + 6 * sigma^6 / r^7 )
    """
    sigma6 = sigma**6
    sigma12 = sigma6**2
    term1 = -12.0 * sigma12 / r.pow(13)
    term2 = 6.0 * sigma6 / r.pow(7)
    return 4.0 * epsilon * (term1 + term2)


def lennard_jones_second_derivative(
    r: torch.Tensor, sigma: float, epsilon: float
) -> torch.Tensor:
    """d^2/dr^2 of the Lennard-Jones potential.

    d2V/dr2 = 4 * epsilon * ( 156 * sigma^12 / r^14 - 42 * sigma^6 / r^8 )
    """
    sigma6 = sigma**6
    sigma12 = sigma6**2
    term1 = 156.0 * sigma12 / r.pow(14)
    term2 = -42.0 * sigma6 / r.pow(8)
    return 4.0 * epsilon * (term1 + term2)


def rastrigin_energy(x: torch.Tensor, _sigma: float, _epsilon: float) -> torch.Tensor:
    """Rastrigin function (1D): f(x) = A + x^2 - A cos(2πx), with A=10.

    Signature matches Lennard-Jones helpers; sigma/epsilon are unused.
    """
    A = 10.0
    return A + x.pow(2) - A * torch.cos(2 * torch.pi * x)


def rastrigin_derivative(
    x: torch.Tensor, _sigma: float, _epsilon: float
) -> torch.Tensor:
    """First derivative of 1D Rastrigin: f'(x) = 2x + 2πA sin(2πx), A=10."""
    A = 10.0
    return 2.0 * x + 2.0 * torch.pi * A * torch.sin(2 * torch.pi * x)


def rastrigin_second_derivative(
    x: torch.Tensor, _sigma: float, _epsilon: float
) -> torch.Tensor:
    """Second derivative of 1D Rastrigin: f''(x) = 2 + 4π^2 A cos(2πx), A=10."""
    A = 10.0
    return 2.0 + 4.0 * (torch.pi**2) * A * torch.cos(2 * torch.pi * x)


def styblinski_tang_energy(
    x: torch.Tensor, _sigma: float, _epsilon: float
) -> torch.Tensor:
    """Styblinski–Tang (1D): f(x) = 0.5 * (x^4 - 16 x^2 + 5 x)."""
    return 0.5 * (x.pow(4) - 16.0 * x.pow(2) + 5.0 * x)


def styblinski_tang_derivative(
    x: torch.Tensor, _sigma: float, _epsilon: float
) -> torch.Tensor:
    """First derivative: f'(x) = 2 x^3 - 16 x + 2.5."""
    return 2.0 * x.pow(3) - 16.0 * x + 2.5


def styblinski_tang_second_derivative(
    x: torch.Tensor, _sigma: float, _epsilon: float
) -> torch.Tensor:
    """Second derivative: f''(x) = 6 x^2 - 16."""
    return 6.0 * x.pow(2) - 16.0


def get_energy_functions(target: str):
    """Return (energy_fn, derivative_fn, second_derivative_fn) for the chosen target."""
    if target == "lj":
        return (
            lennard_jones_energy,
            lennard_jones_derivative,
            lennard_jones_second_derivative,
        )
    if target == "rastrigin":
        return (
            rastrigin_energy,
            rastrigin_derivative,
            rastrigin_second_derivative,
        )
    if target == "styblinski_tang":
        return (
            styblinski_tang_energy,
            styblinski_tang_derivative,
            styblinski_tang_second_derivative,
        )
    raise ValueError(f"Unknown target function: {target}")


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers = []
        input_dim = 1
        output_dim = 1

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Config:
    epochs: int = 2000
    lr: float = 1e-3
    batch_size: int = 256
    hidden_dim: int = 64
    hidden_layers: int = 2
    target: str = "lj"
    sigma: float = 1.0
    epsilon: float = 1.0
    r_min: float = 0.8
    r_max: float = 3.0
    num_samples: int = 10000
    seed: int = 42
    device: str = "cpu"
    wandb_project: str = "lj-mlp"
    wandb_run_name: str | None = None
    log_every: int = 50
    out_path: str = "examples/lj_mlp_dissociation.png"
    loss: str = "l2"
    deriv_weight: float = 0.0
    second_deriv_weight: float = 0.0
    curvature_weight: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(
    cfg: Config, device: torch.device
) -> tuple[TensorDataset, torch.Tensor, torch.Tensor]:
    r = torch.linspace(cfg.r_min, cfg.r_max, cfg.num_samples, device=device).unsqueeze(
        1
    )
    energy_fn, _, _ = get_energy_functions(cfg.target)
    y = energy_fn(r, cfg.sigma, cfg.epsilon)
    dataset = TensorDataset(r, y)
    return dataset, r.squeeze(1), y.squeeze(1)


def train(cfg: Config) -> tuple[MLP, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    dataset, r_all, y_all = build_dataset(cfg, device)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )

    model = MLP(cfg.hidden_dim, cfg.hidden_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.L1Loss() if cfg.loss == "l1" else nn.MSELoss()

    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.__dict__)

    # Select target functions
    energy_fn, deriv_fn, second_deriv_fn = get_energy_functions(cfg.target)

    # Precompute plotting grid and true curve for per-epoch visualization
    r_plot = torch.linspace(cfg.r_min, cfg.r_max, 1000, device=device).unsqueeze(1)
    with torch.no_grad():
        y_true_plot_epoch = energy_fn(r_plot, cfg.sigma, cfg.epsilon).squeeze(1).cpu()
    # Precompute validation grid (100 points) and true values
    r_val = torch.linspace(cfg.r_min, cfg.r_max, 100, device=device).unsqueeze(1)
    with torch.no_grad():
        y_true_val = energy_fn(r_val, cfg.sigma, cfg.epsilon)

    global_step = 0
    for epoch in tqdm(range(1, cfg.epochs + 1)):
        epoch_total_loss = 0.0
        epoch_value_loss = 0.0
        epoch_deriv_loss = 0.0
        epoch_second_deriv_loss = 0.0
        epoch_curvature_loss = 0.0
        for xb, yb in loader:
            needs_grads = (
                (cfg.deriv_weight > 0.0)
                or (cfg.second_deriv_weight > 0.0)
                or (cfg.curvature_weight > 0.0)
            )
            if needs_grads:
                xb = xb.requires_grad_(True)

            pred = model(xb)
            value_loss = loss_fn(pred, yb)

            deriv_loss = None
            second_deriv_loss = None
            curvature_loss = None

            if cfg.deriv_weight > 0.0 or cfg.second_deriv_weight > 0.0:
                grads = torch.autograd.grad(
                    outputs=pred,
                    inputs=xb,
                    grad_outputs=torch.ones_like(pred),
                    create_graph=True,
                    retain_graph=True,
                )[0]

            if cfg.deriv_weight > 0.0:
                yprime_true = deriv_fn(xb, cfg.sigma, cfg.epsilon)
                deriv_loss = loss_fn(grads, yprime_true)

            if (cfg.second_deriv_weight > 0.0) or (cfg.curvature_weight > 0.0):
                second_grads = torch.autograd.grad(
                    outputs=grads,
                    inputs=xb,
                    grad_outputs=torch.ones_like(grads),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                if cfg.second_deriv_weight > 0.0:
                    ysecond_true = second_deriv_fn(xb, cfg.sigma, cfg.epsilon)
                    second_deriv_loss = loss_fn(second_grads, ysecond_true)
                if cfg.curvature_weight > 0.0:
                    curvature_loss = torch.mean(torch.abs(second_grads))

            loss = value_loss
            if deriv_loss is not None:
                loss = loss + cfg.deriv_weight * deriv_loss
            if second_deriv_loss is not None:
                loss = loss + cfg.second_deriv_weight * second_deriv_loss
            if curvature_loss is not None:
                loss = loss + cfg.curvature_weight * curvature_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
            batch_size = xb.shape[0]
            epoch_total_loss += loss.item() * batch_size
            epoch_value_loss += value_loss.item() * batch_size
            if deriv_loss is not None:
                epoch_deriv_loss += deriv_loss.item() * batch_size
            if second_deriv_loss is not None:
                epoch_second_deriv_loss += second_deriv_loss.item() * batch_size
            if curvature_loss is not None:
                epoch_curvature_loss += curvature_loss.item() * batch_size
            global_step += 1

        epoch_total_loss /= cfg.num_samples
        epoch_value_loss /= cfg.num_samples
        if cfg.deriv_weight > 0.0:
            epoch_deriv_loss /= cfg.num_samples
        if cfg.second_deriv_weight > 0.0:
            epoch_second_deriv_loss /= cfg.num_samples
        if cfg.curvature_weight > 0.0:
            epoch_curvature_loss /= cfg.num_samples
        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            tqdm.write(f"Epoch {epoch} loss: {epoch_total_loss}")
            log_payload = {
                "epoch": epoch,
                "train/loss": epoch_total_loss,
                "train/value_loss": epoch_value_loss,
            }
            if cfg.deriv_weight > 0.0:
                log_payload["train/deriv_loss"] = epoch_deriv_loss
            if cfg.second_deriv_weight > 0.0:
                log_payload["train/second_deriv_loss"] = epoch_second_deriv_loss
            if cfg.curvature_weight > 0.0:
                log_payload["train/curvature_loss"] = epoch_curvature_loss
            # Plot dissociation curve this epoch and log to W&B (no file save per epoch)
            with torch.no_grad():
                y_pred_plot_epoch = model(r_plot).squeeze(1).cpu()
                y_pred_val = model(r_val)
                val_value_loss = loss_fn(y_pred_val, y_true_val).item()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                r_plot.squeeze(1).cpu().numpy(),
                y_true_plot_epoch.numpy(),
                label=f"{cfg.target} True",
                linewidth=2,
            )
            ax.plot(
                r_plot.squeeze(1).cpu().numpy(),
                y_pred_plot_epoch.numpy(),
                label="MLP Pred",
                linewidth=2,
            )
            ax.set_xlabel("Distance r")
            ax.set_ylabel("Energy V(r)")
            ax.set_title(f"{cfg.target} function (epoch {epoch})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            log_payload["plots/dissociation"] = wandb.Image(fig)
            log_payload["val/loss"] = val_value_loss
            plt.close(fig)
            wandb.log(log_payload)

    # Evaluate over a fine grid for plotting
    r_plot = torch.linspace(cfg.r_min, cfg.r_max, 1000, device=device).unsqueeze(1)
    with torch.no_grad():
        y_true_plot = energy_fn(r_plot, cfg.sigma, cfg.epsilon).squeeze(1).cpu()
        y_pred_plot = model(r_plot).squeeze(1).cpu()
    return model, r_plot.squeeze(1).cpu(), y_true_plot, y_pred_plot


def make_plot(
    cfg: Config, r: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor
) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r.numpy(), y_true.numpy(), label=f"{cfg.target} True", linewidth=2)
    ax.plot(r.numpy(), y_pred.numpy(), label="MLP Pred", linewidth=2)
    ax.set_xlabel("Distance r")
    ax.set_ylabel("Energy V(r)")
    ax.set_title(f"{cfg.target} function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = cfg.out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    wandb.log({"plots/dissociation": wandb.Image(fig)})
    plt.close(fig)
    return out_path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a small MLP on the Lennard-Jones potential (1D distance)"
    )
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument(
        "--target",
        type=str,
        choices=["lj", "rastrigin", "styblinski_tang"],
        default="lj",
    )
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=0.8)
    parser.add_argument("--r-max", type=float, default=3.0)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    # parser.add_argument("--wandb-project", type=str, default="lj-mlp")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--out-path", type=str, default="examples/lj_mlp_dissociation.png"
    )
    parser.add_argument("--loss", type=str, choices=["l1", "l2"], default="l2")
    parser.add_argument("--deriv-weight", type=float, default=0.0)
    parser.add_argument("--second-deriv-weight", type=float, default=0.0)
    parser.add_argument("--curvature-weight", type=float, default=0.0)
    args = parser.parse_args()

    return Config(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        target=args.target,
        sigma=args.sigma,
        epsilon=args.epsilon,
        r_min=args.r_min,
        r_max=args.r_max,
        num_samples=args.num_samples,
        seed=args.seed,
        device=args.device,
        wandb_project=f"{args.target}-mlp",
        wandb_run_name=args.wandb_run_name,
        log_every=args.log_every,
        out_path=args.out_path,
        loss=args.loss,
        deriv_weight=args.deriv_weight,
        second_deriv_weight=args.second_deriv_weight,
        curvature_weight=args.curvature_weight,
    )


def main() -> None:
    cfg = parse_args()
    model, r_plot, y_true_plot, y_pred_plot = train(cfg)
    out_path = make_plot(cfg, r_plot, y_true_plot, y_pred_plot)
    print(f"Saved dissociation plot to: {out_path}")


if __name__ == "__main__":
    main()
