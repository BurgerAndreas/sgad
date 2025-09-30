from sgad.optimizer import Optimizer, Geometry, ANGSTROM_TO_AU, EV_PER_AU
from collections import namedtuple

import numpy as np

# import autograd.numpy as anp
# from autograd import grad
import scipy.optimize

import math

import logging

logger = logging.getLogger("optimizer")


def log(msg):
    logger.debug(msg)


def get_rms(arr):
    return np.sqrt(np.mean(arr**2))


def bfgs_update(H, dx, dg):
    first_term = np.outer(dg, dg) / dg.dot(dx)
    second_term = H.dot(np.outer(dx, dx)).dot(H) / dx.dot(H).dot(dx)
    return first_term - second_term, "BFGS"


class RFOptimizer(Optimizer):
    rfo_dict = {
        "min": (0, "min"),
        "max": (-1, "max"),
    }

    def __init__(
        self,
        geometry: Geometry,
        line_search: bool = True,
        gdiis: bool = True,
        gdiis_thresh: float = 2.5e-3
        * EV_PER_AU
        / ANGSTROM_TO_AU,  # 2.5e-3 Bohr -> Angstrom
        gdiis_test_direction: bool = True,
        trust_radius: float = 0.5 / ANGSTROM_TO_AU,  # 0.5 Bohr -> Angstrom
        trust_update: bool = True,
        trust_min: float = 0.1 / ANGSTROM_TO_AU,  # 0.1 Bohr -> Angstrom
        trust_max: float = 1.0 / ANGSTROM_TO_AU,  # 1.0 Bohr -> Angstrom
        hessian_update: str = "bfgs",
        hessian_init: str = "unit",
        hessian_recalc: int | None = None,
        small_eigval_thresh: float = 1e-8,
        alpha0: float = 1.0,
        max_micro_cycles: int = 25,
        **kwargs,
    ) -> None:
        """Rational function Optimizer.

        Parameters
        ----------
        geometry
            Geometry to be optimized.
        line_search
            Whether to carry out implicit line searches.
        gdiis
            Whether to enable GDIIS.
        gdiis_thresh
            Threshold for rms(forces) to enable GDIIS.
        gdiis_test_direction
            Whether to the overlap of the RFO step and the GDIIS step.
        max_micro_cycles
            Number of restricted-step microcycles. Disabled by default.
        trust_radius
            Initial trust radius in Angstrom.
        trust_update
            Whether to update the trust radius throughout the optimization.
        trust_min
            Minimum trust radius in Angstrom.
        trust_max
            Maximum trust radius in Angstrom.
        hessian_update
            Type of Hessian update. Defaults to BFGS for minimizations and Bofill
            for saddle point searches.
        hessian_init
            Type of initial model Hessian.
        hessian_recalc
            Recalculate exact Hessian every n-th cycle instead of updating it.
        small_eigval_thresh
            Threshold for small eigenvalues. Eigenvectors belonging to eigenvalues
            below this threshold are discardewd.
        alpha0
            Initial alpha for restricted-step (RS) procedure.
        max_micro_cycles
            Maximum number of RS iterations.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to the Optimizer baseclass.
        """
        super().__init__(geometry, **kwargs)

        self.line_search = line_search
        self.gdiis = gdiis
        self.gdiis_thresh = gdiis_thresh  # Will be compared to rms(step)
        self.gdiis_test_direction = gdiis_test_direction

        self.successful_gdiis = 0
        self.successful_line_search = 0
        self.trust_update = bool(trust_update)
        assert trust_min <= trust_max, "trust_min must be <= trust_max!"
        self.trust_min = float(trust_min)
        self.trust_max = float(trust_max)
        # Constrain initial trust radius if trust_max > trust_radius
        self.trust_radius = min(trust_radius, trust_max)
        self.log(f"Initial trust radius: {self.trust_radius:.6f}")
        self.hessian_update = hessian_update
        self.hessian_update_func = bfgs_update
        self.hessian_init = hessian_init
        self.hessian_recalc = hessian_recalc
        self.small_eigval_thresh = float(small_eigval_thresh)
        # Restricted-step related
        self.alpha0 = alpha0
        self.max_micro_cycles = int(max_micro_cycles)
        assert max_micro_cycles >= 0

        assert self.small_eigval_thresh > 0.0, "small_eigval_thresh must be > 0.!"
        if not self.restarted:
            self.hessian_recalc_in = None
            self.predicted_energy_changes = list()
        assert isinstance(self.hessian_init, np.ndarray) or self.hessian_init in (
            "calc",
            "unit",
        )

    def reset(self):
        # Don't recalculate the hessian if we have to reset the optimizer
        hessian_init = self.hessian_init
        self.prepare_opt(hessian_init)

    def prepare_opt(self, hessian_init=None):
        if hessian_init is None:
            hessian_init = self.hessian_init

        hessian_was_calc = isinstance(hessian_init, str) and hessian_init == "calc"
        if isinstance(hessian_init, np.ndarray):
            self.H = hessian_init
        elif hessian_init == "calc":
            self.H = self.geometry.hessian
        elif hessian_init == "unit":
            self.H = np.eye(self.geometry.coords.size)
        else:
            raise ValueError(f"Invalid Hessian initialization: {hessian_init}")

        if (
            hasattr(self.geometry, "coord_type")
            and self.geometry.coord_type == "dlc"
            # Calculated Hessian is already in DLC
            and not hessian_was_calc
        ):
            U = self.geometry.internal.U
            self.H = U.T.dot(self.H).dot(U)

        if self.hessian_recalc:
            # Already substract one, as we don't do a hessian update in
            # the first cycle.
            self.hessian_recalc_in = self.hessian_recalc - 1

    def _get_opt_restart_info(self):
        opt_restart_info = {
            "H": self.H.tolist(),
            "hessian_recalc_in": self.hessian_recalc_in,
            "predicted_energy_changes": self.predicted_energy_changes,
        }
        return opt_restart_info

    def _set_opt_restart_info(self, opt_restart_info):
        self.H = np.array(opt_restart_info["H"])
        self.hessian_recalc_in = opt_restart_info["hessian_recalc_in"]
        self.predicted_energy_changes = opt_restart_info["predicted_energy_changes"]

    def update_trust_radius(self):
        # The predicted change should be calculated at the end of optimize
        # of the previous cycle.
        assert len(self.predicted_energy_changes) == len(self.forces) - 1, (
            "Did you forget to append to self.predicted_energy_changes?"
        )
        predicted_change = self.predicted_energy_changes[-1]
        actual_change = self.energies[-1] - self.energies[-2]
        # Only report an unexpected increase if we actually predicted a
        # decrease.
        coeff = actual_change / predicted_change
        step = self.steps[-1]
        last_step_norm = np.linalg.norm(step)
        self.set_new_trust_radius(coeff, last_step_norm)

    def set_new_trust_radius(self, coeff, last_step_norm):
        # Nocedal, Numerical optimization Chapter 4, Algorithm 4.1

        # If actual and predicted energy change have different signs
        # coeff will be negative and lead to a decreased trust radius,
        # which is fine.
        if coeff < 0.25:
            self.trust_radius = max(self.trust_radius / 4, self.trust_min)
        # Only increase trust radius if last step norm was at least 80% of it
        # See [5], Appendix, step size and direction control
        # elif coeff > 0.75 and (last_step_norm >= .8*self.trust_radius):
        #
        # Only increase trust radius if last step norm corresponded approximately
        # to the trust radius.
        elif (
            coeff > 0.75
            and abs(self.trust_radius - last_step_norm) <= 1e-3 / ANGSTROM_TO_AU
        ):
            self.trust_radius = min(self.trust_radius * 2, self.trust_max)
        else:
            return

    def update_hessian(self):
        try:
            self.hessian_recalc_in = max(self.hessian_recalc_in - 1, 0)
        except TypeError:
            self.hessian_recalc_in = None

        recalc = self.hessian_recalc_in == 0

        if recalc:
            # Use xtb hessian
            self.H = self.geometry.hessian
            self.save_hessian()
            # Reset counter. It is also reset when the recalculation was initiated
            # by the adaptive formulation.
            self.hessian_recalc_in = self.hessian_recalc
        # Simple hessian update
        else:
            dx = self.steps[-1]
            dg = -(self.forces[-1] - self.forces[-2])
            dH, key = self.hessian_update_func(self.H, dx, dg)
            self.H = self.H + dH

    def solve_rfo(self, rfo_mat, kind="min"):
        # When using the restricted step variant of RFO the RFO matrix
        # may not be symmetric. Thats why we can't use eigh here.
        eigenvalues, eigenvectors = np.linalg.eig(rfo_mat)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        sorted_inds = np.argsort(eigenvalues)

        # Depending on wether we want to minimize (maximize) along
        # the mode(s) in the rfo mat we have to select the smallest
        # (biggest) eigenvalue and corresponding eigenvector.
        first_or_last, verbose = self.rfo_dict[kind]
        # Given sorted eigenvalue-indices (sorted_inds) use the first
        # (smallest eigenvalue) or the last (largest eigenvalue) index.
        ind = sorted_inds[first_or_last]
        follow_eigvec = eigenvectors.T[ind]
        step_nu = follow_eigvec.copy()
        nu = step_nu[-1]
        # Scale eigenvector so that its last element equals 1. The
        # final is step is the scaled eigenvector without the last element.
        step = step_nu[:-1] / nu
        eigval = eigenvalues[ind]
        return step, eigval, nu

    def filter_small_eigvals(self, eigvals, eigvecs, mask=False):
        small_inds = np.abs(eigvals) < self.small_eigval_thresh
        eigvals = eigvals[~small_inds]
        eigvecs = eigvecs[:, ~small_inds]
        small_num = sum(small_inds)
        assert small_num <= 6, (
            "Expected at most 6 small eigenvalues in cartesian hessian "
            f"but found {small_num}!"
        )
        if mask:
            return eigvals, eigvecs, small_inds
        else:
            return eigvals, eigvecs

    # def log_negative_eigenvalues(self, eigvals, pre_str=""):
    #     neg_inds = eigvals < -self.small_eigval_thresh
    #     neg_eigval_str = np.array2string(eigvals[neg_inds], precision=6)

    def housekeeping(self):
        """Calculate gradient and energy. Update trust radius and hessian
        if needed. Return energy, gradient and hessian for the current cycle."""
        gradient = self.geometry.gradient
        energy = self.geometry.energy
        self.forces.append(-gradient)
        self.energies.append(energy)

        can_update = (
            # Allows gradient differences
            len(self.forces) > 1
            and (self.forces[-2].shape == gradient.shape)
            and len(self.coords) > 1
            # Coordinates may have been rebuilt. Take care of that.
            and (self.coords[-2].shape == self.coords[1].shape)
            and len(self.energies) > 1
        )
        if can_update:
            if self.trust_update:
                self.update_trust_radius()
            self.update_hessian()

        H = self.H
        if self.geometry.internal:
            # Shift eigenvalues of orthogonal part to high values, so they
            # don't contribute to the actual step.
            H_proj = self.geometry.internal.project_hessian(self.H)
            # Symmetrize hessian, as the projection may break it?!
            H = (H_proj + H_proj.T) / 2

        eigvals, eigvecs = np.linalg.eigh(H)
        # Neglect small eigenvalues
        eigvals, eigvecs = self.filter_small_eigvals(eigvals, eigvecs)

        resetted = not can_update
        return energy, gradient, H, eigvals, eigvecs, resetted

    def get_augmented_hessian(self, eigvals, gradient, alpha=1.0):
        dim_ = eigvals.size + 1
        H_aug = np.zeros((dim_, dim_))
        H_aug[: dim_ - 1, : dim_ - 1] = np.diag(eigvals / alpha)
        H_aug[-1, :-1] = gradient
        H_aug[:-1, -1] = gradient

        H_aug[:-1, -1] /= alpha

        return H_aug

    def get_alpha_step(self, cur_alpha, rfo_eigval, step_norm, eigvals, gradient):
        # Derivative of the squared step w.r.t. alpha
        numer = gradient**2
        denom = (eigvals - rfo_eigval * cur_alpha) ** 3
        quot = np.sum(numer / denom)
        dstep2_dalpha = 2 * rfo_eigval / (1 + step_norm**2 * cur_alpha) * quot
        # Update alpha
        alpha_step = 2 * (self.trust_radius * step_norm - step_norm**2) / dstep2_dalpha
        assert (cur_alpha + alpha_step) > 0, "alpha must not be negative!"
        return alpha_step

    def get_rs_step(self, eigvals, eigvecs, gradient, name="RS"):
        # Transform gradient to basis of eigenvectors
        gradient_ = eigvecs.T.dot(gradient)

        alpha = self.alpha0
        for mu in range(self.max_micro_cycles):
            H_aug = self.get_augmented_hessian(eigvals, gradient_, alpha)
            rfo_step_, eigval_min, nu = self.solve_rfo(H_aug, "min")
            rfo_norm_ = np.linalg.norm(rfo_step_)

            if (rfo_norm_ < self.trust_radius) or abs(
                rfo_norm_ - self.trust_radius
            ) <= 1e-3 / ANGSTROM_TO_AU:
                step_ = rfo_step_
                break

            alpha_step = self.get_alpha_step(
                alpha, eigval_min, rfo_norm_, eigvals, gradient_
            )
            alpha += alpha_step
        # Otherwise, use trust region newton step
        else:
            H_aug = self.get_augmented_hessian(eigvals, gradient_, alpha=1.0)
            rfo_step_, eigval_min, nu = self.solve_rfo(H_aug, "min")
            rfo_norm_ = np.linalg.norm(rfo_step_)

            # This should always be True if the above algorithm failed but we
            # keep this line nonetheless,  to make it more obvious.
            if rfo_norm_ > self.trust_radius:
                step_ = self.get_newton_step_on_trust(
                    eigvals, eigvecs, gradient, transform=False
                )
                # Simple, downscaled RFO step
                # step_ = rfo_step_ / rfo_norm_ * self.trust_radius
            else:
                step_ = rfo_step_

        # Transform step back to original basis
        step = eigvecs.dot(step_)
        return step

    @staticmethod
    def get_shifted_step_trans(eigvals, gradient_trans, shift):
        return -gradient_trans / (eigvals + shift)

    @staticmethod
    def get_newton_step(eigvals, eigvecs, gradient):
        return eigvecs.dot(eigvecs.T.dot(gradient) / eigvals)

    def get_newton_step_on_trust(self, eigvals, eigvecs, gradient, transform=True):
        """Step on trust-radius.

        See Nocedal 4.3 Iterative solutions of the subproblem
        """
        min_ind = eigvals.argmin()
        min_eigval = eigvals[min_ind]
        pos_definite = (eigvals > 0.0).all()
        gradient_trans = eigvecs.T.dot(gradient)

        # This will be also be True when we come close to a minimizer,
        # but then the Hessian will also be positive definite and a
        # simple Newton step will be used.
        hard_case = abs(gradient_trans[min_ind]) <= 1e-6 * (EV_PER_AU / ANGSTROM_TO_AU)
        # self.log(f"Smallest eigenvalue: {min_eigval:.6f}")
        # self.log(f"Positive definite Hessian: {pos_definite}")
        # self.log(f"Hard case: {hard_case}")

        def get_step(shift):
            return -gradient_trans / (eigvals + shift)

        # Unshifted Newton step
        newton_step_trans = get_step(0.0)
        newton_norm = np.linalg.norm(newton_step_trans)

        def on_trust_radius_lin(step):
            return 1 / self.trust_radius - 1 / np.linalg.norm(step)

        def finalize_step(shift):
            step = get_step(shift)
            if transform:
                step = eigvecs.dot(step)
            return step

        # Simplest case. Positive definite Hessian and predicted step is
        # already in trust radius.
        if pos_definite and newton_norm <= self.trust_radius:
            # self.log("Using unshifted Newton step.")
            return eigvecs.dot(newton_step_trans)

        # If the Hessian is not positive definite or if the step is too
        # long we have to determine the shift parameter lambda.
        rs_kwargs = {
            "f": lambda shift: on_trust_radius_lin(get_step(shift)),
            "xtol": 1e-3,
            # Would otherwise be chosen automatically, but we set it
            # here explicitly for verbosity.
            "method": "brentq",
        }

        def root_search(bracket):
            rs_kwargs.update(
                {
                    "bracket": bracket,
                    "x0": bracket[0] + 1e-3,
                }
            )
            res = scipy.optimize.root_scalar(**rs_kwargs)
            return res

        BRACKET_END = 1e10
        if not hard_case:
            bracket_start = 0.0 if pos_definite else -min_eigval + 1e-2
            bracket = (bracket_start, BRACKET_END)
            try:
                res = root_search(bracket)
                assert res.converged
                return finalize_step(res.root)
            # ValueError may be raised when the function values for the
            # initial bracket have the same sign. If so, we continue with
            # treating it as a hard case.
            except ValueError:
                pass

        # Now we would try the bracket (-b2, -b1). The resulting step should have
        # a suitable length, but the (shifted) Hessian would have an incorrect
        # eigenvalue spectrum (not positive definite). To solve this we use a
        # different formula to calculate the step.
        mask = np.ones_like(gradient_trans)
        mask[min_ind] = 0
        mask = mask.astype(bool)
        without_min = gradient_trans[mask] / (eigvals[mask] - min_eigval)
        try:
            tau = math.sqrt(self.trust_radius**2 - (without_min**2).sum())
            step_trans = [tau] + (-without_min).tolist()
        # Hard case. Search in open interval (endpoints not included)
        # (-min_eigval, inf).
        except ValueError:
            bracket = (-min_eigval + 1e-6, BRACKET_END)
            res = root_search(bracket)
            if res.converged:
                return finalize_step(res.root)

        if not transform:
            return step_trans

        return eigvecs.dot(step_trans)

    @staticmethod
    def rfo_model(gradient, hessian, step):
        quadratic = step.dot(gradient) + 0.5 * step.dot(hessian).dot(step)
        return quadratic / (1 + step.dot(step))

    def optimize(self):
        energy, gradient, H, big_eigvals, big_eigvecs, resetted = self.housekeeping()
        # positive_definite = (eigvals < 0).sum() == 0
        # gradient_small = rms(gradient) < grad_rms_thresh
        step_func = self.get_rs_step
        pred_func = self.rfo_model

        ref_gradient = gradient.copy()
        # Reference step, used for judging the proposed GDIIS step
        ref_step = step_func(big_eigvals, big_eigvecs, gradient)

        # Right everything is in place to check for convergence.  If all values are below
        # the thresholds, there is no need to do additional inter/extrapolations.
        if self.check_convergence(ref_step)[0]:  # Drop conv_info
            self.log("Convergence achieved! Skipping inter/extrapolation.")
            return ref_step

        # Try to interpolate an intermediate geometry, either from GDIIS or line search.
        #
        # Set some defaults
        ip_gradient = None
        ip_step = None
        diis_result = None

        # Check if we can do GDIIS or GEDIIS. If we (can) do a line search is decided
        # after trying GDIIS.
        # rms_forces = get_rms(gradient)
        rms_step = get_rms(ref_step)
        can_diis = (rms_step <= self.gdiis_thresh) and (not resetted)

        # GDIIS / GEDIIS, prefer GDIIS over GEDIIS
        if self.gdiis and can_diis:
            # Gradients as error vectors
            err_vecs = -np.array(self.forces)
            diis_result = gdiis(
                err_vecs,
                self.coords,
                self.forces,
                ref_step,
                test_direction=self.gdiis_test_direction,
            )
            self.successful_gdiis += 1 if diis_result else 0

        try:
            ip_coords = diis_result.coords
            ip_step = ip_coords - self.geometry.coords
            ip_gradient = -diis_result.forces
        # When diis_result is None
        except AttributeError:
            pass

        # Try line search if GDIIS failed or not requested
        if self.line_search and (diis_result is None) and (not resetted):
            ip_energy, ip_gradient, ip_step = poly_line_search(
                energy,
                self.energies[-2],
                gradient,
                -self.forces[-2],
                self.steps[-1],
                cubic_max_x=-1,
                quartic_max_x=2,
                # logger=self.logger,
            )
            self.successful_line_search += 1 if ip_gradient is not None else 0

        # Use the interpolated gradient for the RFO step if interpolation succeeded
        if (ip_gradient is not None) and (ip_step is not None):
            gradient = ip_gradient
        # Keep the original gradient when the interpolation failed, but recreate
        # ip_step, as it will be returned as None from poly_line_search().
        else:
            ip_step = np.zeros_like(gradient)

        step = step_func(big_eigvals, big_eigvecs, gradient)
        # Form full step. If we did not interpolate or interpolation failed,
        # ip_step will be zero.
        step = step + ip_step

        # Use the original, actually calculated, gradient
        prediction = pred_func(ref_gradient, H, step)
        self.predicted_energy_changes.append(prediction)

        return step

    def postprocess_opt(self):
        msg = (
            f"Successful invocations:\n"
            f"\t      GDIIS: {self.successful_gdiis}\n"
            f"\tLine Search: {self.successful_line_search}\n"
        )
        self.log(msg)


# [1] http://aip.scitation.org/doi/10.1063/1.1515483 Optimization review
# [2] https://doi.org/10.1063/1.450914 Trust region method
# [3] 10.1007/978-0-387-40065-5 Numerical optimization
# [4] 10.1007/s00214-016-1847-3 Explorations of some refinements


def get_minimum(poly):
    roots = np.roots(np.polyder(poly))
    real_roots = np.real(roots[np.isreal(roots)])
    vals = poly(real_roots)
    min_ind = vals.argmin()
    min_root = real_roots[vals.argmin()]
    min_val = vals[min_ind]
    return min_root, min_val


def get_maximum(poly):
    roots = np.roots(np.polyder(poly))
    real_roots = np.real(roots[np.isreal(roots)])
    vals = poly(real_roots)
    max_ind = vals.argmax()
    max_root = real_roots[max_ind]
    max_val = vals[max_ind]
    return max_root, max_val


FitResult = namedtuple("FitResult", "x y polys")


def quintic_fit(e0, e1, g0, g1, H0, H1):
    a = -H0 / 2 + H1 / 2 - 6 * e0 + 6 * e1 - 3 * g0 - 3 * g1
    b = 3 * H0 / 2 - H1 + 15 * e0 - 15 * e1 + 8 * g0 + 7 * g1
    c = -3 * H0 / 2 + H1 / 2 - 10 * e0 + 10 * e1 - 6 * g0 - 4 * g1
    d = H0 / 2
    e = g0
    f = e0

    poly = np.poly1d((a, b, c, d, e, f))
    try:
        mr, mv = get_minimum(poly)
    except ValueError:
        return None

    fit_result = FitResult(mr, mv, (poly,))
    return fit_result


def quartic_fit(e0, e1, g0, g1, maximize=False):
    """See gen_solutions() for derivation."""
    a0 = e0
    a1 = g0
    try:
        sqrt_term = math.sqrt(
            -2
            * (
                6 * (e0 - e1) ** 2
                + 6 * (e0 - e1) * (g0 + g1)
                + (g0 + g1) ** 2
                + 2 * g0 * g1
            )
        )
    except ValueError:
        # In these cases there is no intermediate minimum between 0 and 1 and the term
        # under the square root becomes negative.
        return None

    a2_pre = -3 * (e0 - e1) - 5 * g0 / 2 - g1 / 2
    a3_pre = 2 * e0 - 2 * e1 + 2 * g0

    def get_poly(a3, a2, a1, a0):
        a4 = 3 / 8 * a3**2 / a2
        return np.poly1d((a4, a3, a2, a1, a0))

    a2 = a2_pre - sqrt_term / 2
    a3 = a3_pre + sqrt_term
    poly0 = get_poly(a3, a2, a1, a0)

    a2 = a2_pre + sqrt_term / 2
    a3 = a3_pre - sqrt_term
    poly1 = get_poly(a3, a2, a1, a0)

    get_func = get_maximum if maximize else get_minimum
    mr0, mv0 = get_func(poly0)
    mr1, mv1 = get_func(poly1)

    if maximize:
        mr, mv = (mr0, mv0) if mv0 > mv1 else (mr1, mv1)
    else:
        mr, mv = (mr0, mv0) if mv0 < mv1 else (mr1, mv1)

    fit_result = FitResult(mr, mv, (poly0, poly1))
    return fit_result


def cubic_fit(e0, e1, g0, g1):
    d = e0
    c = g0
    b = -(g1 + 2 * g0 + 3 * e0 - 3 * e1)
    a = 2 * (e0 - e1) + g0 + g1
    # np.testing.assert_allclose([a, b, c, d], coeffs, atol=1e-10)
    poly = np.poly1d((a, b, c, d))
    try:
        mr, mv = get_minimum(poly)
    except ValueError:
        return None

    fit_result = FitResult(mr, mv, (poly,))
    return fit_result


def poly_line_search(
    cur_energy,
    prev_energy,
    cur_grad,
    prev_grad,
    prev_step,
    cubic_max_x=2.0,
    quartic_max_x=4.0,
    logger=None,
):
    """Generate directional gradients by projecting them on the previous step."""
    prev_grad_proj = prev_step @ prev_grad
    cur_grad_proj = prev_step @ cur_grad
    cubic_result = cubic_fit(prev_energy, cur_energy, prev_grad_proj, cur_grad_proj)
    quartic_result = quartic_fit(prev_energy, cur_energy, prev_grad_proj, cur_grad_proj)
    accept = {
        "cubic": lambda x: (x > 0.0) and (x < cubic_max_x),
        "quartic": lambda x: (x > 0.0) and (x <= quartic_max_x),
    }

    fit_result = None
    if quartic_result and accept["quartic"](quartic_result.x):
        fit_result = quartic_result
        deg = "quartic"
    elif cubic_result and accept["cubic"](cubic_result.x):
        fit_result = cubic_result
        deg = "cubic"

    fit_energy = None
    fit_grad = None
    fit_step = None
    if fit_result and fit_result.y < prev_energy:
        x = fit_result.x
        fit_energy = fit_result.y

        # Interpolate coordinates and gradient. 'fit_step' applied to the current
        # coordinates yields interpolated coordinates.
        #
        # x == 0 would take us to the previous coordinates:
        #  (1-0) * -prev_step = -prev_step (we revert the last step)
        # x == 1 would preserve the current coordinates:
        #  (1-1) * -prev_step = 0 (we stay at the current coordinates)
        # x > 1 extrapolate along previous step direction:
        #  with x=2, (1-2) * -prev_step = -1*-prev_step = prev_step
        fit_step = (1 - x) * -prev_step
        fit_grad = (1 - x) * prev_grad + x * cur_grad
    return fit_energy, fit_grad, fit_step


# [1] https://doi.org/10.1016/S0022-2860(84)87198-7
#     Pulay, 1984
# [2] https://pubs.rsc.org/en/content/articlehtml/2002/cp/b108658h
#     Stabilized GDIIS
#     Farkas, Schlegel, 2002
# [3] https://pubs.acs.org/doi/abs/10.1021/ct050275a
#     GEDIIS/Hybrid method
#     Li, Frisch, 2006
# [4] https://aip.scitation.org/doi/10.1063/1.2977735
#     Sim-GEDIIS using hessian information
#     Moss, Li, 2008

# validate the direction of DIIS (Direct Inversion in the Iterative Subspace) steps
# Keys (2, 3, 4, 5, 6, 7, 8, 9): Number of error vectors used in the DIIS procedure
# Values (0.80, 0.75, 0.71, etc.): Minimum acceptable cosine values for the angle between DIIS and reference step directions
COS_CUTOFFS = {
    # Looser cutoffs
    2: 0.80,
    3: 0.75,
    # Original cutoffs, as published in [2]
    # 2: 0.97,
    # 3: 0.84,
    4: 0.71,
    5: 0.67,
    6: 0.62,
    7: 0.56,
    8: 0.49,
    9: 0.41,
}
DIISResult = namedtuple("DIISResult", "coeffs coords forces energy N type")


def valid_diis_direction(diis_step, ref_step, use):
    ref_direction = ref_step / np.linalg.norm(ref_step)
    diis_direction = diis_step / np.linalg.norm(diis_step)
    cos = diis_direction @ ref_direction
    return (cos >= COS_CUTOFFS[use]) and (cos >= 0)


def from_coeffs(vec, coeffs):
    return np.sum(coeffs[:, None] * vec[::-1][: len(coeffs)], axis=0)


def diis_result(coeffs, coords, forces, energy=None, prefix=""):
    diis_coords = from_coeffs(coords, coeffs)
    diis_forces = from_coeffs(forces, coeffs)
    diis_result = DIISResult(
        coeffs=coeffs,
        coords=diis_coords,
        forces=diis_forces,
        energy=energy,
        N=len(coeffs),
        type=f"{prefix}DIIS",
    )
    log(f"\tUsed {len(coeffs)} error vectors for {prefix}DIIS.")
    log("")
    return diis_result


def gdiis(err_vecs, coords, forces, ref_step, max_vecs=5, test_direction=True):
    # Scale error vectors so the smallest norm is 1
    norms = np.linalg.norm(err_vecs, axis=1)
    err_vecs = err_vecs / norms.min()

    valid_coeffs = None
    for use in range(2, min(max_vecs, len(err_vecs)) + 1):
        use_vecs = np.array(err_vecs[::-1][:use])

        A = np.einsum("ij,kj->ik", use_vecs, use_vecs)
        try:
            coeffs = np.linalg.solve(A, np.ones(use))
        except np.linalg.LinAlgError:
            break
        # Scale coeffs so that their sum equals 1
        coeffs_norm = np.linalg.norm(coeffs)
        valid_coeffs_norm = coeffs_norm <= 1e8
        coeffs /= np.sum(coeffs)
        # coeffs_str = np.array2string(coeffs, precision=4)

        # Check degree of extra- and interpolation.
        pos_sum = abs(coeffs[coeffs > 0].sum())
        neg_sum = abs(coeffs[coeffs < 0].sum())
        valid_sums = (pos_sum <= 15) and (neg_sum <= 15)

        # Calculate GDIIS step for comparison to the reference step
        diis_coords = from_coeffs(coords, coeffs)
        diis_step = diis_coords - coords[-1]
        valid_length = np.linalg.norm(diis_step) <= (10 * np.linalg.norm(ref_step))

        # Compare directions of GDIIS- and reference step
        valid_direction = (
            True
            if (not test_direction)
            else valid_diis_direction(diis_step, ref_step, use)
        )

        gdiis_valid = (
            valid_sums and valid_coeffs_norm and valid_direction and valid_length
        )
        if not gdiis_valid:
            break
        # Update valid DIIS coefficients
        valid_coeffs = coeffs

    if valid_coeffs is None:
        return None

    return diis_result(valid_coeffs, coords, forces, prefix="G")
