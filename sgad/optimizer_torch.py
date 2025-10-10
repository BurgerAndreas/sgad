import abc
from dataclasses import dataclass
import time
from typing import Literal, Optional, List

import torch


# Length
M_PER_AU = 0.52917721067e-10  # (m / a_0)
M_TO_AU = 1.0 / M_PER_AU  # (a_0 / m)
ANGSTROM_TO_AU = 1.0e-10 * M_TO_AU  # (a_0 / A)
EV_PER_AU = 27.21138602  # (eV / E_h)
EV_TO_AU = 1.0 / EV_PER_AU  # (E_h / eV)


def convert_thresholds_to_ev_angstrom(atomic_threshs):
    force_conversion = EV_PER_AU / ANGSTROM_TO_AU
    step_conversion = 1.0 / ANGSTROM_TO_AU
    converted = {}
    for name, (max_force, rms_force, max_step, rms_step) in atomic_threshs.items():
        converted[name] = (
            max_force * force_conversion,
            rms_force * force_conversion,
            max_step * step_conversion,
            rms_step * step_conversion,
        )
    return converted


# in atomic units
CONV_THRESHS_ATOMIC = {
    "nwchem_loose": (4.5e-3, 3.0e-3, 5.4e-3, 3.6e-3),
    "gau_loose": (2.5e-3, 1.7e-3, 1.0e-2, 6.7e-3),
    "gau": (4.5e-4, 3.0e-4, 1.8e-3, 1.2e-3),
    "gau_tight": (1.5e-5, 1.0e-5, 6.0e-5, 4.0e-5),
    "gau_vtight": (2.0e-6, 1.0e-6, 6.0e-6, 4.0e-6),
    "baker": (3.0e-4, 2.0e-4, 3.0e-4, 2.0e-4),
    "never": (2.0e-6, 1.0e-6, 6.0e-6, 4.0e-6),
}


CONV_THRESHS = convert_thresholds_to_ev_angstrom(CONV_THRESHS_ATOMIC)
Thresh = Literal["gau_loose", "gau", "gau_tight", "gau_vtight", "baker", "never"]


def bfgs_multiply_torch(
    s_list: List[torch.Tensor],
    y_list: List[torch.Tensor],
    vector: torch.Tensor,
    beta: float = 1.0,
    P: Optional[torch.Tensor] = None,
    gamma_mult: bool = True,
    mu_reg: Optional[float] = None,
):
    assert len(s_list) == len(y_list)
    q = vector.clone()

    # Optional regularization of y
    if mu_reg is not None and mu_reg > 0.0:
        y_list = [
            (y + mu_reg * s)
            if torch.dot(y + mu_reg * s, s) > 0
            else (
                y
                + (max(0.0, -float(torch.dot(s, y)) / float(torch.dot(s, s))) + mu_reg)
                * s
            )
            for s, y in zip(s_list, y_list)
        ]

    alphas: List[torch.Tensor] = []
    rhos: List[torch.Tensor] = []

    cycles = len(s_list)
    for i in range(cycles - 1, -1, -1):
        s = s_list[i]
        y = y_list[i]
        rho = 1.0 / torch.dot(y, s)
        rhos.append(rho)
        alpha = rho * torch.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)

    alphas = alphas[::-1]
    rhos = rhos[::-1]

    if P is not None:
        if P.ndim == 1:
            r = q / P
        else:
            r = torch.linalg.solve(P, q)
    elif gamma_mult and cycles > 0:
        s = s_list[-1]
        y = y_list[-1]
        gamma = float(torch.dot(s, y) / torch.dot(y, y))
        r = q * gamma
    else:
        r = q * beta

    for i in range(cycles):
        s = s_list[i]
        y = y_list[i]
        beta_i = rhos[i] * torch.dot(y, r)
        r = r + s * (alphas[i] - beta_i)

    return r


def double_damp_torch(
    s: torch.Tensor,
    y: torch.Tensor,
    H: Optional[torch.Tensor] = None,
    s_list: Optional[List[torch.Tensor]] = None,
    y_list: Optional[List[torch.Tensor]] = None,
    mu_1: float = 0.2,
    mu_2: Optional[float] = 0.2,
):
    sy = torch.dot(s, y)
    if H is not None:
        Hy = H @ y
    else:
        Hy = bfgs_multiply_torch(s_list or [], y_list or [], y)
    yHy = torch.dot(y, Hy)

    theta_1 = 1.0
    if float(sy) < mu_1 * float(yHy):
        theta_1 = (1.0 - mu_1) * float(yHy) / (float(yHy) - float(sy))
        s = theta_1 * s + (1.0 - theta_1) * Hy

    if mu_2 is not None:
        sy = torch.dot(s, y)
        ss = torch.dot(s, s)
        theta_2 = 1.0
        if float(sy) < mu_2 * float(ss):
            theta_2 = (1.0 - mu_2) * float(ss) / (float(ss) - float(sy))
        y = theta_2 * y + (1.0 - theta_2) * s

    return s, y


class GeometryTorch:
    def __init__(self, coords: torch.Tensor, atomic_nums, calc, device=None):
        if device is None:
            device = coords.device
        self.device = device
        self._coords = coords.reshape(-1).detach().clone()
        self.atomic_nums = atomic_nums
        self.calc = calc
        self.N = len(atomic_nums)
        self._results = {}
        self.coords_changed = True

    @property
    def coords(self) -> torch.Tensor:
        return self._coords

    @coords.setter
    def coords(self, coords: torch.Tensor):
        self._coords = coords.reshape(-1)
        self.coords_changed = True

    def coords3d(self) -> torch.Tensor:
        return self.coords.reshape(-1, 3)

    @property
    def energy(self) -> float:
        if self.coords_changed:
            res = self.calc.predict(
                coords=self.coords3d(), atomic_nums=self.atomic_nums, do_hessian=False
            )
            self._results = {"energy": res["energy"], "forces": res["forces"]}
            self.coords_changed = False
        return float(self._results["energy"].detach().cpu().item())

    @property
    def forces(self) -> torch.Tensor:
        if self.coords_changed:
            res = self.calc.predict(
                coords=self.coords3d(), atomic_nums=self.atomic_nums, do_hessian=False
            )
            self._results = {"energy": res["energy"], "forces": res["forces"]}
            self.coords_changed = False
        return self._results["forces"].reshape(-1)

    @property
    def gradient(self) -> torch.Tensor:
        return -self.forces

    @property
    def cart_coords(self) -> torch.Tensor:
        return self.coords

    @property
    def cart_forces(self) -> torch.Tensor:
        return self.forces.reshape(-1)

    @property
    def coord_type(self) -> str:
        return "cart"


@dataclass(frozen=True)
class ConvInfo:
    cur_cycle: int
    energy_converged: bool
    max_force_converged: bool
    rms_force_converged: bool
    max_step_converged: bool
    rms_step_converged: bool
    desired_eigval_structure: bool

    def get_convergence(self):
        return (
            self.energy_converged,
            self.max_force_converged,
            self.rms_force_converged,
            self.max_step_converged,
            self.rms_step_converged,
            self.desired_eigval_structure,
        )

    def is_converged(self):
        return all(self.get_convergence())


class OptimizerTorch(metaclass=abc.ABCMeta):
    def __init__(
        self,
        geometry: GeometryTorch,
        thresh: Thresh = "gau_loose",
        max_step: float = 0.04 / ANGSTROM_TO_AU,
        max_cycles: int = 150,
        min_step_norm: float = 1e-8 / ANGSTROM_TO_AU,
        assert_min_step: bool = True,
        rms_force: Optional[float] = None,
        rms_force_only: bool = False,
        max_force_only: bool = False,
        force_only: bool = False,
        print_every: int = 100,
        overachieve_factor: float = 0.0,
        check_eigval_structure: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        assert thresh in CONV_THRESHS.keys()

        if device is None:
            device = geometry.coords.device
        self.device = device
        if dtype is None:
            dtype = geometry.coords.dtype
        self.dtype = dtype

        self.geometry = geometry
        self.thresh = thresh
        self.max_step = max_step
        self.min_step_norm = min_step_norm
        self.assert_min_step = assert_min_step
        self.rms_force_only = rms_force_only
        self.max_force_only = max_force_only
        self.force_only = force_only
        self.print_every = int(print_every)
        self.overachieve_factor = float(overachieve_factor)
        self.check_eigval_structure = check_eigval_structure

        # Convergence thresholds
        self.convergence = self.make_conv_dict(
            thresh, rms_force, rms_force_only, max_force_only, force_only
        )
        for key, value in self.convergence.items():
            setattr(self, key, value)

        self.max_cycles = max_cycles

        # Storage
        self.coords: List[torch.Tensor] = []
        self.cart_coords: List[torch.Tensor] = []
        self.energies: List[float] = []
        self.forces: List[torch.Tensor] = []
        self.steps: List[torch.Tensor] = []
        self.max_forces: List[float] = []
        self.rms_forces: List[float] = []
        self.max_steps: List[float] = []
        self.rms_steps: List[float] = []
        self.cycle_times: List[float] = []
        self.modified_forces: List[torch.Tensor] = []

        self.restarted = False
        self.last_cycle = 0
        self.cur_cycle = 0
        self.is_converged = False

    def make_conv_dict(
        self,
        key,
        rms_force=None,
        rms_force_only=False,
        max_force_only=False,
        force_only=False,
    ):
        if not rms_force:
            threshs = CONV_THRESHS[key]
        else:
            threshs = (
                1.5 * rms_force,
                rms_force,
                6 * rms_force,
                4 * rms_force,
            )
        keys = [
            "max_force_thresh",
            "rms_force_thresh",
            "max_step_thresh",
            "rms_step_thresh",
        ]
        conv_dict = {k: v for k, v in zip(keys, threshs)}

        if rms_force_only:
            keep_keys = ["rms_force_thresh"]
        elif max_force_only:
            keep_keys = ["max_force_thresh"]
        elif force_only:
            keep_keys = ["max_force_thresh", "rms_force_thresh"]
        else:
            keep_keys = keys
        conv_dict = {key: value for key, value in conv_dict.items() if key in keep_keys}
        return conv_dict

    def scale_by_max_step(self, steps: torch.Tensor) -> torch.Tensor:
        steps_max = torch.abs(steps).max()
        if float(steps_max) > self.max_step:
            steps = steps * (self.max_step / float(steps_max))
        return steps

    def check_convergence(
        self,
        step: Optional[torch.Tensor] = None,
        multiple: float = 1.0,
        overachieve_factor: Optional[float] = None,
    ):
        if step is None:
            step = self.steps[-1]
        if overachieve_factor is None:
            overachieve_factor = self.overachieve_factor

        forces = self.forces[-1]
        if len(self.modified_forces) == len(self.forces):
            forces = self.modified_forces[-1]

        rms_force = torch.sqrt(torch.mean(forces * forces))
        rms_step = torch.sqrt(torch.mean(step * step))

        max_force = torch.abs(forces).max()
        max_step = torch.abs(step).max()

        self.max_forces.append(float(max_force))
        self.rms_forces.append(float(rms_force))
        self.max_steps.append(float(max_step))
        self.rms_steps.append(float(rms_step))

        vals = {
            "max_force_thresh": float(max_force),
            "rms_force_thresh": float(rms_force),
            "max_step_thresh": float(max_step),
            "rms_step_thresh": float(rms_step),
        }

        def check(key):
            if key in self.convergence:
                return vals[key] <= self.convergence[key] * multiple
            return True

        convergence = {
            "energy_converged": True,
            "max_force_converged": check("max_force_thresh"),
            "rms_force_converged": check("rms_force_thresh"),
            "max_step_converged": check("max_step_thresh"),
            "rms_step_converged": check("rms_step_thresh"),
        }

        desired_eigval_structure = True
        if self.check_eigval_structure:
            if (
                hasattr(self, "ts_mode_eigvals")
                and hasattr(self, "small_eigval_thresh")
                and hasattr(self, "roots")
            ):
                desired_eigval_structure = int(
                    (self.ts_mode_eigvals < self.small_eigval_thresh).sum().item()
                ) == len(self.roots)
        convergence["desired_eigval_structure"] = desired_eigval_structure
        conv_info = ConvInfo(self.cur_cycle, **convergence)

        overachieved = False
        if overachieve_factor and overachieve_factor > 0:
            max_thresh = (
                self.convergence.get("max_force_thresh", 0) / overachieve_factor
            )
            rms_thresh = (
                self.convergence.get("rms_force_thresh", 0) / overachieve_factor
            )
            max_ok = float(max_force) < max_thresh
            rms_ok = float(rms_force) < rms_thresh
            overachieved = bool(max_ok and rms_ok)

        converged = all(convergence.values())
        not_never = self.thresh != "never"
        return (bool((converged or overachieved) and not_never), conv_info)

    @abc.abstractmethod
    def optimize(self) -> Optional[torch.Tensor]:
        pass

    def run(self):
        start_time = time.time()
        for self.cur_cycle in range(self.last_cycle, self.max_cycles):
            t0 = time.time()

            self.coords.append(self.geometry.coords.clone())
            self.cart_coords.append(self.geometry.cart_coords.clone())

            step = self.optimize()
            step_norm = float(torch.linalg.norm(step)) if step is not None else 0.0
            if step is None:
                self.coords.pop(-1)
                self.cart_coords.pop(-1)
                continue

            self.steps.append(step)

            self.is_converged, conv_info = self.check_convergence()

            t1 = time.time()
            self.cycle_times.append(t1 - t0)

            if self.is_converged:
                break
            elif self.assert_min_step and (step_norm <= self.min_step_norm):
                break

            new_coords = self.geometry.coords + step.reshape(-1)
            self.geometry.coords = new_coords
            self.steps[-1] = self.geometry.coords - self.coords[-1]

        _ = start_time  # silence unused


class BFGS(OptimizerTorch):
    """
    Differences between NumPy and Torch optimizers:
    1. Curvature handling (s·y ≤ 0)
    If s·y becomes nonpositive, plain BFGS corrupts H and you can climb.
    NumPy path uses plain BFGS by default;
    the torch path can use damped/double-damped updates.
    If NumPy hit s·y ≤ 0 more often, it can end much higher.
    2. Different initial H0 scaling
    Gamma scaling of H0 (γ = (s·y)/(y·y)) stabilizes early steps.
    Torch has a two-loop LBFGS with γ option;
    NumPy BFGS uses H = I.
    Without γ, NumPy can take poorer early steps.
    """

    def __init__(self, geometry: GeometryTorch, *args, update: str = "bfgs", **kwargs):
        super().__init__(geometry, *args, **kwargs)
        self.H = self.eye
        self.update = update

    @property
    def eye(self) -> torch.Tensor:
        size = self.geometry.coords.numel()
        return torch.eye(
            size, dtype=self.geometry.coords.dtype, device=self.geometry.coords.device
        )

    def bfgs_update(self, s: torch.Tensor, y: torch.Tensor):
        rho = 1.0 / torch.dot(s, y)
        V = self.eye - rho * torch.outer(s, y)
        self.H = V @ self.H @ V.T + rho * torch.outer(s, s)

    def double_damped_bfgs_update(
        self,
        s: torch.Tensor,
        y: torch.Tensor,
        mu_1: float = 0.2,
        mu_2: Optional[float] = 0.2,
    ):
        s_d, y_d = double_damp_torch(s, y, H=self.H, mu_1=mu_1, mu_2=mu_2)
        _ = torch.dot(s_d, y_d)
        self.bfgs_update(s_d, y_d)

    def damped_bfgs_update(self, s: torch.Tensor, y: torch.Tensor, mu_1: float = 0.2):
        self.double_damped_bfgs_update(s, y, mu_1=mu_1, mu_2=None)

    def optimize(self) -> Optional[torch.Tensor]:
        forces = self.geometry.forces
        energy = self.geometry.energy
        self.forces.append(forces)
        self.energies.append(float(energy))

        if self.cur_cycle > 0:
            y = self.forces[-2] - forces
            s = self.steps[-1]
            if self.update == "double":
                self.double_damped_bfgs_update(s, y)
            elif self.update == "damped":
                self.damped_bfgs_update(s, y)
            else:
                _ = torch.dot(s, y)
                self.bfgs_update(s, y)

        step = self.H @ forces
        step = self.scale_by_max_step(step)
        return step


class LBFGS(OptimizerTorch):
    def __init__(
        self,
        geometry: GeometryTorch,
        *args,
        memory: int = 10,
        gamma_mult: bool = True,
        mu_reg: Optional[float] = None,
        preconditioner: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(geometry, *args, **kwargs)
        self.memory = memory
        self.gamma_mult = gamma_mult
        self.mu_reg = mu_reg
        self.P = preconditioner
        self.s_list: List[torch.Tensor] = []
        self.y_list: List[torch.Tensor] = []

    def optimize(self) -> Optional[torch.Tensor]:
        forces = self.geometry.forces
        energy = self.geometry.energy
        self.forces.append(forces)
        self.energies.append(float(energy))

        if self.cur_cycle > 0:
            s = self.steps[-1]
            y = self.forces[-2] - forces
            # curvature check; if not positive, skip storing (simple safeguard)
            if float(torch.dot(s, y)) > 1e-12:
                self.s_list.append(s.detach().clone())
                self.y_list.append(y.detach().clone())
                if len(self.s_list) > self.memory:
                    self.s_list.pop(0)
                    self.y_list.pop(0)

        # SD on first iteration
        if len(self.s_list) == 0:
            step = forces.clone()
        else:
            step = bfgs_multiply_torch(
                self.s_list,
                self.y_list,
                forces,
                beta=1.0,
                P=self.P,
                gamma_mult=self.gamma_mult,
                mu_reg=self.mu_reg,
            )

        step = self.scale_by_max_step(step)
        return step


class FIRE(OptimizerTorch):
    def __init__(
        self,
        geometry: GeometryTorch,
        dt: float = 0.1,
        dt_max: float = 1.0,
        N_acc: int = 2,
        f_inc: float = 1.1,
        f_acc: float = 0.99,
        f_dec: float = 0.5,
        n_reset: int = 0,
        a_start: float = 0.1,
        **kwargs,
    ):
        self.dt = dt
        self.dt_max = dt_max
        self.N_acc = N_acc
        self.f_inc = f_inc
        self.f_acc = f_acc
        self.f_dec = f_dec
        self.n_reset = n_reset
        self.a_start = a_start
        self.a = self.a_start
        super().__init__(geometry, **kwargs)
        self.v = torch.zeros_like(geometry.coords, device=self.device, dtype=self.dtype)
        self.velocities: List[torch.Tensor] = [self.v]
        self.time_deltas: List[float] = [self.dt]

    def optimize(self) -> Optional[torch.Tensor]:
        forces = self.geometry.forces
        energy = self.geometry.energy
        self.forces.append(forces)
        self.energies.append(float(energy))

        v_norm_sq = torch.dot(self.v, self.v)
        f_norm_sq = torch.dot(forces, forces)
        mix_coeff = self.a * torch.sqrt(v_norm_sq / (f_norm_sq + 1e-30))
        mixed_v = (1.0 - self.a) * self.v + mix_coeff * forces

        last_v = self.velocities[-1]
        aligned = (self.cur_cycle > 0) and (torch.dot(last_v, forces) > 0)
        if aligned:
            if self.n_reset > self.N_acc:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self.a = self.a * self.f_acc
            self.n_reset += 1
        else:
            mixed_v = torch.zeros_like(forces, device=self.device, dtype=forces.dtype)
            self.a = self.a_start
            self.dt = self.dt * self.f_acc
            self.n_reset = 0

        v = mixed_v + self.dt * forces
        self.velocities.append(v)
        self.time_deltas.append(self.dt)
        steps = self.dt * v
        steps = self.scale_by_max_step(steps)
        return steps


class SteepestDescent(OptimizerTorch):
    def __init__(
        self,
        geometry: GeometryTorch,
        alpha: float = 0.1,
        bt_force: int = 5,
        dont_skip_after: int = 2,
        bt_max_scale: int = 4,
        bt_disable: bool = False,
        **kwargs,
    ):
        self.alpha = alpha
        self.bt_force = bt_force
        self.dont_skip_after = dont_skip_after
        self.bt_max_scale = bt_max_scale
        self.bt_disable = bt_disable
        self.cycles_since_backtrack = self.bt_force
        self.scale_factor = 0.5
        super().__init__(geometry, **kwargs)
        self.alpha0 = self.alpha
        self.alpha_max = self.bt_max_scale * self.alpha0
        self.skip_log: List[bool] = []

    def backtrack(self, cur_forces: torch.Tensor, prev_forces: torch.Tensor) -> bool:
        if self.bt_disable:
            return False
        epsilon = 1e-3
        cur_rms = torch.sqrt(torch.mean(cur_forces * cur_forces))
        prev_rms = torch.sqrt(torch.mean(prev_forces * prev_forces))
        rms_diff = float((cur_rms - prev_rms) / torch.abs(cur_rms + prev_rms))
        skip = False
        if rms_diff > epsilon:
            self.alpha = self.alpha * self.scale_factor
            skip = True
            self.cycles_since_backtrack = self.bt_force
        else:
            self.cycles_since_backtrack -= 1
            if self.cycles_since_backtrack < 0:
                self.cycles_since_backtrack = self.bt_force
                if self.alpha < self.alpha0:
                    self.alpha = self.alpha0
                    skip = True
                else:
                    self.alpha = self.alpha / self.scale_factor
        if self.alpha > self.alpha_max:
            self.alpha = self.alpha_max
        if (len(self.skip_log) >= self.dont_skip_after) and all(
            self.skip_log[-self.dont_skip_after :]
        ):
            skip = False
            if self.alpha > self.alpha0:
                self.alpha = self.alpha0
        self.skip_log.append(skip)
        return skip

    def optimize(self) -> Optional[torch.Tensor]:
        forces = self.geometry.forces
        energy = self.geometry.energy
        self.forces.append(forces)
        self.energies.append(float(energy))
        if self.cur_cycle > 0:
            _ = self.backtrack(self.forces[-1], self.forces[-2])
        step = self.alpha * self.forces[-1]
        step = self.scale_by_max_step(step)
        return step


class NaiveSteepestDescent(OptimizerTorch):
    def __init__(self, geometry: GeometryTorch, alpha: float = 0.1, **kwargs):
        super().__init__(geometry, **kwargs)
        self.alpha = alpha

    def optimize(self) -> Optional[torch.Tensor]:
        forces = self.geometry.forces
        self.forces.append(forces)
        step = self.alpha * self.forces[-1]
        step = self.scale_by_max_step(step)
        return step


__all__ = [
    "GeometryTorch",
    "OptimizerTorch",
    "BFGS",
    "bfgs_multiply_torch",
    "FIRE",
    "SteepestDescent",
    "NaiveSteepestDescent",
]
