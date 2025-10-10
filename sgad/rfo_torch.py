import torch
import numpy as np

from sgad.optimizer_torch import (
    OptimizerTorch,
    GeometryTorch,
    ANGSTROM_TO_AU,
    EV_PER_AU,
)


def bfgs_update_torch(
    H: torch.Tensor, dx: torch.Tensor, dg: torch.Tensor
) -> torch.Tensor:
    first_term = torch.outer(dg, dg) / torch.dot(dg, dx)
    denom = torch.dot(dx, H @ dx)
    second_term = H @ torch.outer(dx, dx) @ H / denom
    return first_term - second_term


def get_rms_torch(arr: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(arr * arr))


class RFOptimizer(OptimizerTorch):
    def __init__(
        self,
        geometry: GeometryTorch,
        line_search: bool = True,
        gdiis: bool = False,
        gdiis_thresh: float = 2.5e-3 * EV_PER_AU / ANGSTROM_TO_AU,
        gdiis_test_direction: bool = True,
        trust_radius: float = 0.5 / ANGSTROM_TO_AU,
        trust_update: bool = False,
        trust_min: float = 0.1 / ANGSTROM_TO_AU,
        trust_max: float = 1.0 / ANGSTROM_TO_AU,
        hessian_init: str = "unit",
        hessian_recalc: int | None = None,
        small_eigval_thresh: float = 1e-8,
        alpha0: float = 1.0,
        max_micro_cycles: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(geometry, **kwargs)

        self.line_search = bool(line_search)
        self.gdiis = bool(gdiis)
        self.trust_update = bool(trust_update)
        self.trust_min = float(trust_min)
        self.trust_max = float(trust_max)
        self.trust_radius = min(trust_radius, trust_max)
        self.hessian_init = hessian_init
        self.hessian_recalc = hessian_recalc
        self.small_eigval_thresh = float(small_eigval_thresh)
        self.alpha0 = float(alpha0)
        self.max_micro_cycles = int(max_micro_cycles)
        self.gdiis_thresh = float(gdiis_thresh)
        self.gdiis_test_direction = bool(gdiis_test_direction)
        self.predicted_energy_changes: list[float] = []
        self.successful_gdiis = 0
        self.successful_line_search = 0

        self.prepare_opt(self.hessian_init)

    def prepare_opt(self, hessian_init=None):
        if hessian_init is None:
            hessian_init = self.hessian_init

        if isinstance(hessian_init, torch.Tensor):
            self.H = hessian_init.to(device=self.device, dtype=self.dtype)
        elif hessian_init == "calc" and hasattr(self.geometry, "hessian"):
            self.H = torch.as_tensor(
                self.geometry.hessian, dtype=self.dtype, device=self.device
            )
        elif hessian_init == "unit":
            size = self.geometry.coords.numel()
            self.H = torch.eye(size, dtype=self.dtype, device=self.device)
        else:
            self.H = torch.eye(
                self.geometry.coords.numel(), dtype=self.dtype, device=self.device
            )

        self.hessian_recalc_in = (
            (self.hessian_recalc - 1) if self.hessian_recalc else None
        )

    def update_hessian(self):
        recalc = (
            (self.hessian_recalc_in == 0)
            if isinstance(self.hessian_recalc_in, int)
            else False
        )
        if recalc and hasattr(self.geometry, "hessian"):
            self.H = torch.as_tensor(
                self.geometry.hessian, dtype=self.dtype, device=self.device
            )
            self.hessian_recalc_in = self.hessian_recalc
        else:
            dx = self.steps[-1]
            dg = -(self.forces[-1] - self.forces[-2])
            dH = bfgs_update_torch(self.H, dx, dg)
            self.H = self.H + dH
            if isinstance(self.hessian_recalc_in, int):
                self.hessian_recalc_in = max(self.hessian_recalc_in - 1, 0)

    def filter_small_eigvals(self, eigvals: torch.Tensor, eigvecs: torch.Tensor):
        mask = torch.abs(eigvals) >= self.small_eigval_thresh
        eigvals_f = eigvals[mask]
        eigvecs_f = eigvecs[:, mask]
        return eigvals_f, eigvecs_f

    def get_augmented_hessian(
        self, eigvals: torch.Tensor, gradient: torch.Tensor, alpha: float = 1.0
    ):
        dim_ = eigvals.numel() + 1
        H_aug = torch.zeros((dim_, dim_), dtype=self.dtype, device=self.device)
        H_aug[: dim_ - 1, : dim_ - 1] = torch.diag(eigvals / alpha)
        H_aug[-1, :-1] = gradient
        H_aug[:-1, -1] = gradient
        H_aug[:-1, -1] = H_aug[:-1, -1] / alpha
        # make finite
        # Enforce symmetry, sanitize NaN/Inf, and add tiny diagonal jitter
        H_aug = (H_aug + H_aug.T) * 0.5
        H_aug = torch.nan_to_num(H_aug, nan=0.0, posinf=0.0, neginf=0.0)
        eps = torch.finfo(H_aug.dtype).eps
        H_aug = H_aug + eps * torch.eye(dim_, dtype=self.dtype, device=self.device)
        return H_aug

    def solve_rfo(self, H_aug: torch.Tensor):
        # Use symmetric solver for stability (constructed augmented matrix is symmetric)
        try:
            H_aug = (H_aug + H_aug.T) * 0.5
            eigvals, eigvecs = torch.linalg.eigh(H_aug)
        except:
            eigvals, eigvecs = torch.linalg.eig(H_aug)
            eigvals = eigvals.real
            eigvecs = eigvecs.real
        ind = torch.argmin(eigvals)
        follow = eigvecs[:, ind]
        nu = follow[-1]
        step = follow[:-1] / nu
        eigval = eigvals[ind]
        return step, eigval, nu

    def get_rs_step(
        self, eigvals: torch.Tensor, eigvecs: torch.Tensor, gradient: torch.Tensor
    ):
        # gradient in eigenspace
        gradient_ = eigvecs.T @ gradient
        alpha = self.alpha0
        for _ in range(self.max_micro_cycles):
            alpha = float(max(min(alpha, 1e8), 1e-8))
            H_aug = self.get_augmented_hessian(eigvals, gradient_, alpha)
            rfo_step_, eigval_min, nu = self.solve_rfo(H_aug)
            rfo_norm_ = torch.linalg.norm(rfo_step_)
            if (rfo_norm_ < self.trust_radius) or (
                abs(self.trust_radius - float(rfo_norm_)) <= 1e-3 / ANGSTROM_TO_AU
            ):
                step_ = rfo_step_
                break
            numer = gradient_ * gradient_
            denom = (eigvals - eigval_min * alpha) ** 3
            quot = torch.sum(numer / denom)
            dstep2_dalpha = 2 * eigval_min / (1 + (rfo_norm_**2) * alpha) * quot
            alpha_step = (
                2 * (self.trust_radius * rfo_norm_ - (rfo_norm_**2)) / dstep2_dalpha
            )
            alpha = float(alpha + alpha_step)
        else:
            # Hard-case/newton fallback: clamp by trust radius
            step_ = -(gradient_ / torch.clamp(eigvals, min=1e-12))
            step_norm = torch.linalg.norm(step_)
            if step_norm > self.trust_radius:
                step_ = step_ / step_norm * self.trust_radius
        # back to original basis
        step = eigvecs @ step_
        return step

    @staticmethod
    def rfo_model(
        gradient: torch.Tensor, hessian: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        quadratic = step @ gradient + 0.5 * (step @ (hessian @ step))
        return quadratic / (1 + step @ step)

    def optimize(self):
        forces = self.geometry.forces
        energy = self.geometry.energy
        self.forces.append(forces)
        self.energies.append(float(energy))

        # Update Hessian if we have previous cycle
        if (
            (len(self.forces) > 1)
            and (len(self.coords) > 0)
            and (len(self.energies) > 1)
        ):
            if self.trust_update and (
                len(self.predicted_energy_changes) == len(self.forces) - 1
            ):
                predicted_change = self.predicted_energy_changes[-1]
                actual_change = self.energies[-1] - self.energies[-2]
                coeff = (
                    actual_change / float(predicted_change)
                    if predicted_change != 0
                    else 0.0
                )
                last_step_norm = (
                    float(torch.linalg.norm(self.steps[-1])) if self.steps else 0.0
                )
                if coeff < 0.25:
                    self.trust_radius = max(self.trust_radius / 4, self.trust_min)
                elif (coeff > 0.75) and (
                    abs(self.trust_radius - last_step_norm) <= 1e-3 / ANGSTROM_TO_AU
                ):
                    self.trust_radius = min(self.trust_radius * 2, self.trust_max)
            self.update_hessian()

        # Eigen-decomposition (optionally in projected internal basis)
        H = self.H
        try:
            eigvals, eigvecs = torch.linalg.eigh(H)
        except:
            eigvals, eigvecs = torch.linalg.eig(H)
            eigvals = eigvals.real
            eigvecs = eigvecs.real
        eigvals, eigvecs = self.filter_small_eigvals(eigvals, eigvecs)

        gradient = -forces
        ref_gradient = gradient.clone()
        ref_step = self.get_rs_step(eigvals, eigvecs, ref_gradient)

        # Early convergence check
        if self.check_convergence(ref_step)[0]:
            return ref_step

        # Try GDIIS when small step and not resetted
        ip_gradient = None
        ip_step = None
        if self.gdiis:
            rms_step = float(torch.sqrt(torch.mean(ref_step * ref_step)))
            can_diis = (
                (rms_step <= self.gdiis_thresh)
                and (len(self.coords) > 0)
                and (len(self.forces) > 0)
            )
            if can_diis:
                diis_res = self._gdiis_numpy(ref_step)
                if diis_res is not None:
                    ip_step = torch.as_tensor(
                        diis_res["coords"]
                        - self.geometry.coords.detach().cpu().numpy(),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    ip_gradient = -torch.as_tensor(
                        diis_res["forces"], dtype=self.dtype, device=self.device
                    )
                    self.successful_gdiis += 1

        # Poly line search if requested and DIIS not used
        if (
            (ip_gradient is None)
            and self.line_search
            and (len(self.energies) > 1)
            and (len(self.steps) > 0)
        ):
            pl = self._poly_line_search_numpy(
                cur_energy=self.energies[-1],
                prev_energy=self.energies[-2],
                cur_grad=gradient.detach().cpu().numpy(),
                prev_grad=(-self.forces[-2]).detach().cpu().numpy(),
                prev_step=self.steps[-1].detach().cpu().numpy(),
            )
            if pl is not None:
                ip_energy, ip_gradient_np, ip_step_np = pl
                ip_gradient = torch.as_tensor(
                    ip_gradient_np, dtype=self.dtype, device=self.device
                )
                ip_step = torch.as_tensor(
                    ip_step_np, dtype=self.dtype, device=self.device
                )
                self.successful_line_search += 1

        # Recompute RS step with interpolated gradient if available
        if (ip_gradient is not None) and (ip_step is not None):
            rs_step = self.get_rs_step(eigvals, eigvecs, ip_gradient)
            step = rs_step + ip_step
        else:
            step = ref_step

        # Prediction uses reference gradient for parity with numpy implementation
        prediction = self.rfo_model(ref_gradient, H, step)
        self.predicted_energy_changes.append(float(prediction))

        return step

    # -------------------------- numpy helpers (no try/except) --------------------------
    def _poly_line_search_numpy(
        self,
        cur_energy,
        prev_energy,
        cur_grad,
        prev_grad,
        prev_step,
        cubic_max_x=2.0,
        quartic_max_x=4.0,
    ):
        prev_grad_proj = float(prev_step @ prev_grad)
        cur_grad_proj = float(prev_step @ cur_grad)
        cubic = self._cubic_fit(prev_energy, cur_energy, prev_grad_proj, cur_grad_proj)
        quartic = self._quartic_fit(
            prev_energy, cur_energy, prev_grad_proj, cur_grad_proj
        )

        fit_result = None
        if quartic is not None and (quartic[0] > 0.0) and (quartic[0] <= quartic_max_x):
            fit_result = ("quartic", quartic)
        elif cubic is not None and (cubic[0] > 0.0) and (cubic[0] < cubic_max_x):
            fit_result = ("cubic", cubic)

        if fit_result is None:
            return None

        _, (x, y, _) = fit_result
        if y >= prev_energy:
            return None

        # Interpolate coords and gradient along prev_step
        fit_step = (1 - x) * (-prev_step)
        fit_grad = (1 - x) * prev_grad + x * cur_grad
        return float(y), fit_grad, fit_step

    def _cubic_fit(self, e0, e1, g0, g1):
        d = e0
        c = g0
        b = -(g1 + 2 * g0 + 3 * e0 - 3 * e1)
        a = 2 * (e0 - e1) + g0 + g1
        coeffs = np.array([a, b, c, d], dtype=np.float64)
        # roots of derivative of cubic → quadratic
        deriv = np.polyder(coeffs)
        roots = np.roots(deriv)
        real_roots = np.real(roots[np.isreal(roots)])
        if real_roots.size == 0:
            return None
        poly = np.poly1d(coeffs)
        vals = poly(real_roots)
        min_ind = int(np.argmin(vals))
        return float(real_roots[min_ind]), float(vals[min_ind]), (poly,)

    def _quartic_fit(self, e0, e1, g0, g1, maximize=False):
        a0 = e0
        a1 = g0
        disc = -2 * (
            6 * (e0 - e1) ** 2
            + 6 * (e0 - e1) * (g0 + g1)
            + (g0 + g1) ** 2
            + 2 * g0 * g1
        )
        # Expression for sqrt term is -disc; require it >= 0
        sqrt_arg = -disc
        if sqrt_arg <= 0:
            return None
        sqrt_term = float(np.sqrt(sqrt_arg))
        a2_pre = -3 * (e0 - e1) - 2.5 * g0 - 0.5 * g1
        a3_pre = 2 * e0 - 2 * e1 + 2 * g0

        def get_poly(a3, a2, a1, a0):
            a4 = 3.0 / 8.0 * (a3**2) / a2
            return np.array([a4, a3, a2, a1, a0], dtype=np.float64)

        a2 = a2_pre - sqrt_term / 2
        a3 = a3_pre + sqrt_term
        poly0 = get_poly(a3, a2, a1, a0)

        a2 = a2_pre + sqrt_term / 2
        a3 = a3_pre - sqrt_term
        poly1 = get_poly(a3, a2, a1, a0)

        def get_extreme(coeffs, maximize=False):
            deriv = np.polyder(coeffs)
            roots = np.roots(deriv)
            real = np.real(roots[np.isreal(roots)])
            if real.size == 0:
                return None
            poly = np.poly1d(coeffs)
            vals = poly(real)
            idx = int(np.argmax(vals) if maximize else np.argmin(vals))
            return float(real[idx]), float(vals[idx])

        alt0 = get_extreme(poly0, maximize=maximize)
        alt1 = get_extreme(poly1, maximize=maximize)
        if (alt0 is None) and (alt1 is None):
            return None
        if alt0 is None:
            best = alt1
        elif alt1 is None:
            best = alt0
        else:
            best = (
                alt0
                if ((alt0[1] > alt1[1]) if maximize else (alt0[1] < alt1[1]))
                else alt1
            )
        x, y = best
        return x, y, (poly0, poly1)

    # ------- GDIIS helpers (numpy; use lstsq) -------
    COS_CUTOFFS = {
        2: 0.80,
        3: 0.75,
        4: 0.71,
        5: 0.67,
        6: 0.62,
        7: 0.56,
        8: 0.49,
        9: 0.41,
    }

    def _valid_diis_direction(
        self, diis_step: np.ndarray, ref_step: torch.Tensor, use: int
    ) -> bool:
        ref_np = ref_step.detach().cpu().numpy()
        ref_dir = ref_np / (np.linalg.norm(ref_np) + 1e-30)
        diis_dir = diis_step / (np.linalg.norm(diis_step) + 1e-30)
        cos = float(diis_dir @ ref_dir)
        cutoff = self.COS_CUTOFFS.get(use, 0.41)
        return (cos >= cutoff) and (cos >= 0.0)

    def _gdiis_numpy(self, ref_step: torch.Tensor, max_vecs: int = 5):
        # Build error vectors and stacks (as numpy)
        err_vecs = -np.array([f.detach().cpu().numpy() for f in self.forces])
        if err_vecs.ndim != 2:
            return None
        norms = np.linalg.norm(err_vecs, axis=1)
        if norms.size == 0 or np.min(norms) == 0:
            return None
        err_vecs = err_vecs / np.min(norms)

        best = None
        for use in range(2, min(max_vecs, len(err_vecs)) + 1):
            use_vecs = err_vecs[::-1][:use]
            A = use_vecs @ use_vecs.T
            # Solve A c = 1 in least squares sense
            ones = np.ones((use,), dtype=np.float64)
            coeffs, *_ = np.linalg.lstsq(A, ones, rcond=None)
            coeffs_norm = np.linalg.norm(coeffs)
            coeffs_sum = float(np.sum(coeffs))
            if coeffs_sum == 0:
                continue
            coeffs = coeffs / coeffs_sum
            if not (coeffs_norm <= 1e8):
                break

            # Direction and length checks
            coords_stack = np.array([c.detach().cpu().numpy() for c in self.coords])
            forces_stack = np.array([f.detach().cpu().numpy() for f in self.forces])
            diis_coords = np.sum(coeffs[:, None] * coords_stack[::-1][:use], axis=0)
            diis_forces = np.sum(coeffs[:, None] * forces_stack[::-1][:use], axis=0)
            diis_step = diis_coords - coords_stack[-1]
            valid_length = np.linalg.norm(diis_step) <= (
                10.0 * np.linalg.norm(ref_step.detach().cpu().numpy())
            )
            valid_direction = (
                self._valid_diis_direction(diis_step, ref_step, use)
                if self.gdiis_test_direction
                else True
            )
            pos_sum = abs(coeffs[coeffs > 0].sum())
            neg_sum = abs(coeffs[coeffs < 0].sum())
            valid_sums = (pos_sum <= 15) and (neg_sum <= 15)
            if valid_length and valid_direction and valid_sums:
                best = {"coords": diis_coords, "forces": diis_forces}
            else:
                break
        return best


__all__ = [
    "RFOptimizer",
]
