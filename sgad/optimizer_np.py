import abc
from dataclasses import dataclass
import functools
import logging
import os
from pathlib import Path
import sys
import textwrap
import time
from typing import Literal, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
import yaml
import torch

from scipy.sparse.linalg import spsolve


class Geometry:
    def __init__(self, coords, atomic_nums, calc):
        """Initialize a Geometry object.

        Args:
            coords: Coordinates as numpy array or torch tensor, shape (N*3,) or (N, 3)
            atomic_nums: Atomic numbers as numpy array or torch tensor, shape (N,)
            calc: Calculator object with predict() method (e.g., EquiformerTorchCalculator)
        """
        self._coords = coords.reshape(-1)
        self.atomic_nums = atomic_nums
        self.calc = calc
        self.N = len(atomic_nums)
        self.internal = None
        self.results = {}
        self.coords_changed = True

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        self._coords = coords.reshape(-1)
        self.coords_changed = True

    @property
    def energy(self):
        if self.coords_changed:
            self.results = self.calc.predict(
                coords=self.coords3d(), atomic_nums=self.atomic_nums, do_hessian=False
            )
            self.coords_changed = False
        return self.results["energy"].item()

    @property
    def forces(self):
        if self.coords_changed:
            self.results = self.calc.predict(
                coords=self.coords3d(), atomic_nums=self.atomic_nums, do_hessian=False
            )
            self.coords_changed = False
        return self.results["forces"].cpu().numpy().reshape(-1)

    @property
    def gradient(self):
        return -self.forces

    @property
    def cart_coords(self):
        """Return Cartesian coordinates (alias for coords for compatibility)"""
        return self.coords

    @property
    def cart_forces(self):
        """Return Cartesian forces (alias for forces for compatibility)"""
        return self.forces.reshape(-1)

    @property
    def coord_type(self):
        """Return coordinate type for compatibility with optimizer"""
        return "cart"

    def coords3d(self):
        """Return coordinates in 3D format (N, 3)"""
        coords = self.coords
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        return coords.reshape(-1, 3)


def interfragment_distance(
    frag1: List[int], frag2: List[int], coords3d: NDArray
) -> float:
    def mean_coords(frag):
        frag_coords = coords3d[frag]
        mean = frag_coords.mean(axis=0)
        return mean

    frag1_mean = mean_coords(frag1)
    frag2_mean = mean_coords(frag2)
    interfrag_dist = float(np.linalg.norm(frag1_mean - frag2_mean))
    return interfrag_dist


def get_data_model(geometry, max_cycles):
    dummy_geom = geometry

    # Define dataset shapes. As pysisyphus offers growing COS methods where
    # the number of images changes along the optimization we have to define
    # the shapes accordingly by considering the maximum number of images.
    _1d = (max_cycles,)
    _2d = (max_cycles, dummy_geom.coords.size)
    _image_inds = (max_cycles, 1)
    # Number of cartesian coordinates is probably different from the number
    # of internal coordinates.
    _2d_cart = (max_cycles, dummy_geom.cart_coords.size)

    data_model = {
        "image_nums": _1d,
        "image_inds": _image_inds,
        "cart_coords": _2d_cart,
        "coords": _2d,
        "energies": _1d,
        "forces": _2d,
        # AFIR related
        "true_energies": _1d,
        "true_forces": _2d_cart,
        "steps": _2d,
        # Convergence related
        "max_forces": _1d,
        "rms_forces": _1d,
        "max_steps": _1d,
        "rms_steps": _1d,
        # Misc
        "cycle_times": _1d,
        "modified_forces": _2d,
        # COS specific
        "tangents": _2d,
    }

    return data_model


# Length
M_PER_AU = 0.52917721067e-10  # (m / a_0)
M_TO_AU = 1.0 / M_PER_AU  # (a_0 / m)
ANGSTROM_TO_AU = 1.0e-10 * M_TO_AU  # (a_0 / A)
EV_PER_AU = 27.21138602  # (eV / E_h)
EV_TO_AU = 1.0 / EV_PER_AU  # (E_h / eV)

# in atomic units
CONV_THRESHS_ATOMIC = {
    #                max_force (Hartree/Bohr), rms_force (Hartree/Bohr), max_step (Bohr), rms_step (Bohr)
    "nwchem_loose": (4.5e-3, 3.0e-3, 5.4e-3, 3.6e-3),
    "gau_loose": (2.5e-3, 1.7e-3, 1.0e-2, 6.7e-3),
    "gau": (4.5e-4, 3.0e-4, 1.8e-3, 1.2e-3),
    "gau_tight": (1.5e-5, 1.0e-5, 6.0e-5, 4.0e-5),
    "gau_vtight": (2.0e-6, 1.0e-6, 6.0e-6, 4.0e-6),
    "baker": (3.0e-4, 2.0e-4, 3.0e-4, 2.0e-4),
    # Dummy thresholds
    "never": (2.0e-6, 1.0e-6, 6.0e-6, 4.0e-6),
}


# Convert atomic unit thresholds to eV/Angstrom units
# Force: Hartree/Bohr -> eV/Angstrom (multiply by EV_PER_AU / ANGSTROM_TO_AU)
# Step: Bohr -> Angstrom (divide by ANGSTROM_TO_AU)
def convert_thresholds_to_ev_angstrom(atomic_threshs):
    """Convert convergence thresholds from atomic units to eV/Angstrom."""
    force_conversion = EV_PER_AU / ANGSTROM_TO_AU  # eV/Angstrom per Hartree/Bohr
    step_conversion = 1.0 / ANGSTROM_TO_AU  # Angstrom per Bohr

    converted = {}
    for name, (max_force, rms_force, max_step, rms_step) in atomic_threshs.items():
        converted[name] = (
            max_force * force_conversion,  # max_force in eV/Angstrom
            rms_force * force_conversion,  # rms_force in eV/Angstrom
            max_step * step_conversion,  # max_step in Angstrom
            rms_step * step_conversion,  # rms_step in Angstrom
        )
    return converted


# Convergence thresholds in eV and Angstrom units
# max_force (eV/Angstrom), rms_force (eV/Angstrom), max_step (Angstrom), rms_step (Angstrom)
CONV_THRESHS = convert_thresholds_to_ev_angstrom(CONV_THRESHS_ATOMIC)
Thresh = Literal["gau_loose", "gau", "gau_tight", "gau_vtight", "baker", "never"]


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


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(
        self,
        geometry: Geometry,
        thresh: Thresh = "gau_loose",
        max_step: float = 0.04 / ANGSTROM_TO_AU,  # 0.04 Bohr -> Angstrom
        max_cycles: int = 150,
        min_step_norm: float = 1e-8 / ANGSTROM_TO_AU,  # 1e-8 Bohr -> Angstrom
        assert_min_step: bool = True,
        rms_force: Optional[float] = None,
        rms_force_only: bool = False,
        max_force_only: bool = False,
        force_only: bool = False,
        print_every: int = 100,
        prefix: str = "",
        overachieve_factor: float = 0.0,
        check_eigval_structure: bool = False,
        restart_info=None,
        check_coord_diffs: bool = True,
        coord_diff_thresh: float = 0.01,
        fragments: Optional[Tuple] = None,
        monitor_frag_dists: int = 0,
        out_dir: str = ".",
    ) -> None:
        """Optimizer baseclass. Meant to be subclassed.

        Parameters
        ----------
        geometry
            Geometry to be optimized.
        thresh
            Convergence threshold.
        max_step
            Maximum absolute component of the allowed step vector in Angstrom. Utilized in
            optimizers that don't support a trust region or line search.
        max_cycles
            Maximum number of allowed optimization cycles.
        min_step_norm
            Minimum norm of an allowed step in Angstrom. If the step norm drops below
            this value a ValueError exception is raised.
        assert_min_step
            Flag that controls whether the norm of the proposed step is check
            for being too small.
        rms_force
            Root-mean-square of the force in eV/Angstrom from which user-defined thresholds
            are derived. When 'rms_force' is given 'thresh' is ignored.
        rms_force_only
            When set, convergence is signalled only based on rms(forces).
        max_force_only
            When set, convergence is signalled only based on max(|forces|).
        force_only
            When set, convergence is signalled only based on max(|forces|) and rms(forces).
        print_every
            Report optimization progress every nth cycle.
        prefix
            Short string that is prepended to several files created by
            the optimizer. Allows distinguishing several optimizations carried
            out in the same directory.
        overachieve_factor
            Signal convergence when max(forces) and rms(forces) fall below the
            chosen threshold, divided by this factor. Convergence of max(step) and
            rms(step) is ignored.
        check_eigval_structure
            Check the eigenvalues of the modes we maximize along. Convergence requires
            them to be negative. Useful if TS searches are started from geometries close
            to a minimum.
        restart_info
            Restart information. Undocumented.
        check_coord_diffs
            Whether coordinates of chain-of-sates images are checked for being
            too similar.
        coord_diff_thresh
            Unitless threshold for similary checking of COS image coordinates.
            The first image is assigned 0, the last image is assigned to 1.
        fragments
            Tuple of lists containing atom indices, defining two fragments.
        monitor_frag_dists
            Monitor fragment distances for N cycles. The optimization is terminated
            when the interfragment distances falls below the initial value after N
            cycles.
        out_dir
            String poiting to a directory where optimization progress is
            dumped.
        """
        assert thresh in CONV_THRESHS.keys()

        self.geometry = geometry
        self.thresh = thresh
        self.max_step = max_step
        self.min_step_norm = min_step_norm
        self.assert_min_step = assert_min_step
        self.rms_force_only = rms_force_only
        self.max_force_only = max_force_only
        self.force_only = force_only
        print_every = int(print_every)
        assert print_every >= 1
        self.print_every = print_every
        self.prefix = f"{prefix}_" if prefix else prefix
        self.overachieve_factor = float(overachieve_factor)
        self.check_eigval_structure = check_eigval_structure
        self.check_coord_diffs = check_coord_diffs
        self.coord_diff_thresh = float(coord_diff_thresh)

        self.logger = logging.getLogger("optimizer")

        # Set up convergence thresholds
        self.convergence = self.make_conv_dict(
            thresh, rms_force, rms_force_only, max_force_only, force_only
        )
        for key, value in self.convergence.items():
            setattr(self, key, value)

        self.max_cycles = max_cycles

        self.fragments = fragments
        self.monitor_frag_dists = monitor_frag_dists
        if self.monitor_frag_dists:
            assert len(self.fragments) == 2, (
                "Interfragment monitoring requires two fragments!"
            )
            assert all([len(frag) > 0 for frag in self.fragments]), (
                "Fragments must not be empty!"
            )
        # Setting some default values
        self.monitor_frag_dists_counter = self.monitor_frag_dists
        self.interfrag_dists = list()
        self.resetted = False
        try:
            out_dir = Path(out_dir)
        except TypeError:
            out_dir = Path(".")
        self.out_dir = out_dir.resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Setting some empty lists as default. The actual shape of the respective
        # entries is not considered, which gives us some flexibility.
        self.data_model = get_data_model(self.geometry, self.max_cycles)
        for la in self.data_model.keys():
            setattr(self, la, list())

        if self.prefix:
            self.log(f"Created optimizer with prefix {self.prefix}")

        self.restarted = False
        self.last_cycle = 0
        self.cur_cycle = 0
        if restart_info is not None:
            if isinstance(restart_info, str):
                restart_info = yaml.load(restart_info, Loader=yaml.SafeLoader)
            self.set_restart_info(restart_info)
            self.restarted = True

        # header = "cycle Δ(energy) max(|force|) rms(force) max(|step|) rms(step) s/cycle".split()
        # col_fmts = "int float float float float float float_short".split()
        self.is_converged = False

    def get_path_for_fn(self, fn, with_prefix=True):
        prefix = self.prefix if with_prefix else ""
        return self.out_dir / (prefix + fn)

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
            # print(
            #     "Deriving convergence threshold from supplied "
            #     f"rms_force={rms_force}."
            # )
            threshs = (
                1.5 * rms_force,
                rms_force,
                6 * rms_force,
                4 * rms_force,
            )
        keys = keep_keys = [
            "max_force_thresh",
            "rms_force_thresh",
            "max_step_thresh",
            "rms_step_thresh",
        ]
        conv_dict = {k: v for k, v in zip(keys, threshs)}

        # Only used gradient information for COS optimizations
        if rms_force_only:
            keep_keys = ["rms_force_thresh"]
        elif max_force_only:
            keep_keys = ["max_force_thresh"]
        elif force_only:
            keep_keys = ["max_force_thresh", "rms_force_thresh"]

        # The dictionary should only contain pairs that are needed
        conv_dict = {key: value for key, value in conv_dict.items() if key in keep_keys}
        return conv_dict

    def report_conv_thresholds(self):
        oaf = self.overachieve_factor

        # Overachieved
        def oa(val):
            return f", ({val / oaf:.6f})" if oaf > 0.0 else ""

        internal_coords = self.geometry.coord_type not in (
            "cart",
            "cartesian",
            "mwcartesian",
        )
        fu = "E_h a_0⁻¹" + (" (rad⁻¹)" if internal_coords else "")  # forces unit
        su = "a_0" + (" (rad)" if internal_coords else "")  # step unit

        try:
            rms_thresh = f"\tmax(|force|) <= {self.max_force_thresh:.6f}{oa(self.max_force_thresh)} {fu}"
        except AttributeError:
            rms_thresh = None
        try:
            max_thresh = f"\t  rms(force) <= {self.rms_force_thresh:.6f}{oa(self.rms_force_thresh)} {fu}"
        except AttributeError:
            max_thresh = None
        threshs = (rms_thresh, max_thresh)

        if self.rms_force_only:
            use_threshs = (threshs[1],)
        elif self.max_force_only:
            use_threshs = (threshs[0],)
        elif self.force_only:
            use_threshs = threshs
        else:
            use_threshs = threshs + (
                f"\t max(|step|) <= {self.max_step_thresh:.6f} {su}",
                f"\t   rms(step) <= {self.rms_step_thresh:.6f} {su}",
            )
        print(
            "Convergence thresholds"
            + (", (overachieved when)" if oaf > 0.0 else "")
            + ":\n"
            + "\n".join(use_threshs)
            + "\n\t'Superscript' indicates convergence"
            + "\n"
        )

    def log(self, message, level=50):
        self.logger.log(level, message)

    def check_convergence(self, step=None, multiple=1.0, overachieve_factor=None):
        """Check if the current convergence of the optimization
        is equal to or below the required thresholds, or a multiple
        thereof. The latter may be used in initiating the climbing image.
        """

        if step is None:
            step = self.steps[-1]
        if overachieve_factor is None:
            overachieve_factor = self.overachieve_factor

        if len(self.modified_forces) == len(self.forces):
            self.log("Using modified forces to determine convergence!")
            forces = self.modified_forces[-1]
        else:
            forces = self.forces[-1]

        # The forces of fixed images may be zero and this may distort the RMS
        # values. So we take into account the number of moving images with
        # non-zero forces vectors.
        rms_force = np.sqrt(np.mean(np.square(forces)))
        rms_step = np.sqrt(np.mean(np.square(step)))

        max_force = np.abs(forces).max()
        max_step = np.abs(step).max()

        self.max_forces.append(max_force)
        self.rms_forces.append(rms_force)
        self.max_steps.append(max_step)
        self.rms_steps.append(rms_step)

        this_cycle = {
            "max_force_thresh": max_force,
            "rms_force_thresh": rms_force,
            "max_step_thresh": max_step,
            "rms_step_thresh": rms_step,
        }

        def check(key):
            # Always return True if given key is not checked
            key_is_checked = key in self.convergence
            if key_is_checked:
                result = this_cycle[key] <= getattr(self, key) * multiple
            else:
                result = True
            return result

        convergence = {
            "energy_converged": True,
            "max_force_converged": check("max_force_thresh"),
            "rms_force_converged": check("rms_force_thresh"),
            "max_step_converged": check("max_step_thresh"),
            "rms_step_converged": check("rms_step_thresh"),
        }
        # For TS optimizations we also try to check the eigenvalue structure of the
        # Hessian. A saddle point of order n must have exatcly only n significant negative
        # eigenvalues. We try to check this below.
        #
        # Currently, this is not totally strict,
        # as only the values in self.ts_mode_eigvals are checked but actually all eigenvalues
        # would have to be checked.
        desired_eigval_structure = True
        if self.check_eigval_structure:
            try:
                desired_eigval_structure = (
                    # Acutally all eigenvalues would have to be checked, but
                    # currently they are not stored anywhere.
                    self.ts_mode_eigvals < self.small_eigval_thresh
                ).sum() == len(self.roots)
            except AttributeError:
                self.log(
                    "Skipping check of eigenvalue structure, as information is unavailable."
                )
        convergence["desired_eigval_structure"] = desired_eigval_structure
        conv_info = ConvInfo(self.cur_cycle, **convergence)

        # Check if force convergence is overachieved. If the eigenvalue-structure is
        # checked, a wrong structure will prevent overachieved convergence.
        overachieved = False
        if overachieve_factor > 0 and desired_eigval_structure:
            max_thresh = (
                self.convergence.get("max_force_thresh", 0) / overachieve_factor
            )
            rms_thresh = (
                self.convergence.get("rms_force_thresh", 0) / overachieve_factor
            )
            max_ = max_force < max_thresh
            rms_ = rms_force < rms_thresh
            overachieved = max_ and rms_
            if max_:
                self.log("max(force) is overachieved")
            if rms_:
                self.log("rms(force) is overachieved")
            if max_ and rms_:
                self.log("Force convergence overachieved!")

        converged = all([val for val in convergence.values()])
        convergence["converged"] = converged
        not_never = self.thresh != "never"

        if self.thresh == "baker":
            energy_converged = False
            if self.cur_cycle > 0:
                cur_energy = self.energies[-1]
                prev_energy = self.energies[-2]
                energy_converged = (
                    abs(cur_energy - prev_energy) < 1e-6
                )  # eV energy difference
            # Convert atomic unit thresholds to eV/Angstrom units
            force_thresh_ev_ang = 3e-4 * (
                EV_PER_AU / ANGSTROM_TO_AU
            )  # 3e-4 Hartree/Bohr -> eV/Angstrom
            step_thresh_ang = 3e-4 / ANGSTROM_TO_AU  # 3e-4 Bohr -> Angstrom
            converged = (max_force < force_thresh_ev_ang) and (
                energy_converged or (max_step < step_thresh_ang)
            )
            convergence["converged"] = converged
        self._convergence_result = convergence

        return (
            any((converged, overachieved)) and not_never,
            conv_info,
        )

    def print_opt_progress(self, conv_info):
        try:
            energy_diff = self.energies[-1] - self.energies[-2]
        # ValueError: maybe raised when the number of images differ in cycles
        # IndexError: raised in first cycle when only one energy is present
        except (ValueError, IndexError):
            energy_diff = float("nan")

        # Try to sum COS energies
        try:
            energy_diff = sum(energy_diff)
        except TypeError:
            pass

        # desired_eigval_structure is also returned, but currently not reported.
        # marks = [False, *conv_info.get_convergence()[:-1], False]
        try:
            cycle_time = self.cycle_times[-1]
        except IndexError:
            cycle_time = 0.0
        print(
            self.cur_cycle,
            energy_diff,
            self.max_forces[-1],
            self.rms_forces[-1],
            self.max_steps[-1],
            self.rms_steps[-1],
            cycle_time,
        )

    def scale_by_max_step(self, steps):
        steps_max = np.abs(steps).max()
        if steps_max > self.max_step:
            steps *= self.max_step / steps_max
        return steps

    def prepare_opt(self):
        pass

    def postprocess_opt(self):
        pass

    @abc.abstractmethod
    def optimize(self):
        pass

    def write_to_out_dir(self, out_fn, content, mode="w"):
        out_path = self.out_dir / out_fn
        with open(out_path, mode) as handle:
            handle.write(content)
        print(f"{__file__} {self.__class__.__name__} wrote to {out_path}")

    def final_summary(self):
        # If the optimization was stopped _forces may not be set, so
        # then we force a calculation if it was not already set.
        _ = self.geometry.forces
        cart_forces = self.geometry.cart_forces
        max_cart_forces = np.abs(cart_forces).max()
        rms_cart_forces = np.sqrt(np.mean(cart_forces**2))
        int_str = ""
        if self.geometry.coord_type not in ("cart", "cartesian", "mwcartesian"):
            int_forces = self.geometry.forces
            max_int_forces = np.abs(int_forces).max()
            rms_int_forces = np.sqrt(np.mean(int_forces**2))
            int_str = f"""
            \tmax(forces, internal): {max_int_forces:.6f} hartree/(bohr,rad)
            \trms(forces, internal): {rms_int_forces:.6f} hartree/(bohr,rad)"""
        energy = self.geometry.energy
        cnt_hessian_autograd = None
        cnt_hessian_predict = None
        if hasattr(self.geometry.calculator, "cnt_hessian_autograd"):
            cnt_hessian_autograd = self.geometry.calculator.cnt_hessian_autograd
            cnt_hessian_predict = self.geometry.calculator.cnt_hessian_predict
        elif hasattr(self.geometry.calculator, "inner") and hasattr(
            self.geometry.calculator.inner, "cnt_hessian_autograd"
        ):
            cnt_hessian_autograd = self.geometry.calculator.inner.cnt_hessian_autograd
            cnt_hessian_predict = self.geometry.calculator.inner.cnt_hessian_predict
        elif hasattr(self.geometry.calculator, "model") and hasattr(
            self.geometry.calculator.model, "cnt_hessian_autograd"
        ):
            cnt_hessian_autograd = self.geometry.calculator.model.cnt_hessian_autograd
            cnt_hessian_predict = self.geometry.calculator.model.cnt_hessian_predict
        else:
            print(
                f"{__file__} {self.__class__.__name__} No Hessian count found in {self.geometry.calculator.__class__.__name__}."
            )
        final_summary = f"""
        Final summary:{int_str}
        \tmax(forces,cartesian): {max_cart_forces:.6f} hartree/bohr
        \trms(forces,cartesian): {rms_cart_forces:.6f} hartree/bohr
        \tenergy: {energy:.8f} hartree
        \tcnt_hessian_autograd: {cnt_hessian_autograd}
        \tcnt_hessian_predict: {cnt_hessian_predict}
        """
        return textwrap.dedent(final_summary.strip())

    def run(self):
        if not self.restarted:
            self.prepare_opt()
            # self.log(f"{self.geometry.coords.size} degrees of freedom.")
            # self.report_conv_thresholds()
            # print(f"{self.__class__.__name__} Spent {prep_time:.1f} s preparing the first cycle.")

        self.start_time = time.time()

        self.stopped = False
        # Actual optimization loop
        for self.cur_cycle in range(self.last_cycle, self.max_cycles):
            start_time = time.time()
            if self.cur_cycle % self.print_every == 0 and self.cur_cycle > 0:
                self.log("")
                self.log(f"Cycle {self.cur_cycle:03d}")

            # Check if something considerably changed in the optimization,
            # e.g. new images were added/interpolated. Then the optimizer
            # should be reset.
            reset_flag = False
            # Reset when number of coordinates changed
            if self.cur_cycle > 0:
                reset_flag = reset_flag or (
                    self.geometry.coords.size != self.coords[-1].size
                )

            if reset_flag:
                self.reset()

            self.coords.append(self.geometry.coords.copy())
            self.cart_coords.append(self.geometry.cart_coords.copy())

            # Determine and store number of currenctly actively optimized images
            try:
                image_inds = self.geometry.image_inds
                image_num = len(image_inds)
            except AttributeError:
                image_inds = [
                    0,
                ]
                image_num = 1
            self.image_inds.append(image_inds)
            self.image_nums.append(image_num)

            # Here the actual step is obtained from the actual optimizer class.
            step = self.optimize()
            step_norm = np.linalg.norm(step)
            if self.cur_cycle % self.print_every == 0 and self.cur_cycle > 0:
                self.log(f"norm(step)={step_norm:.6f} Ang (rad)")
            for source, target in (
                ("true_energy", "true_energies"),
                ("true_forces", "true_forces"),
            ):
                try:
                    if (value := getattr(self.geometry, source)) is not None:
                        getattr(self, target).append(value)
                except AttributeError:
                    pass

            if step is None:
                # Remove the previously added coords
                self.coords.pop(-1)
                self.cart_coords.pop(-1)
                continue

            self.steps.append(step)

            # Convergence check
            self.is_converged, conv_info = self.check_convergence()

            end_time = time.time()
            elapsed_seconds = end_time - start_time
            self.cycle_times.append(elapsed_seconds)

            if (
                (self.cur_cycle % self.print_every) == 0 and self.cur_cycle > 0
            ) or self.is_converged:
                self.print_opt_progress(conv_info)
            if self.is_converged:
                print("Converged!")
                # Added Andreas
                print(
                    f"Optimizer {self.__class__.__name__} converged at cycle {self.cur_cycle}!"
                )
                # for k, v in self._convergence_result.items():
                #     print(f"{k}: {v}")
                break
            # Allow convergence, before checking for too small steps
            elif self.assert_min_step and (step_norm <= self.min_step_norm):
                print(f"Step norm is too small: {step_norm:.2e} <= {self.min_step_norm:.2e}")
                break

            # Update coordinates
            new_coords = self.geometry.coords.copy() + step.reshape(-1)
            self.geometry.coords = new_coords
            # Use the actual step. It may differ from the proposed step
            # when internal coordinates are used, as the internal-Cartesian
            # transformation is done iteratively.
            self.steps[-1] = self.geometry.coords - self.coords[-1]

            # Alternative: calculate overlap of AFIR force and step. If this
            # overlap is negative the step is taken against the AFIR force.
            if self.monitor_frag_dists_counter > 0:
                interfrag_dist = interfragment_distance(
                    *self.fragments, self.geometry.coords3d
                )
                try:
                    prev_interfrag_dist = self.interfrag_dists[-1]
                    if interfrag_dist > prev_interfrag_dist:
                        print("Interfragment distances increased!")
                        self.stopped = True  # Can't use := in if clause
                        break
                except IndexError:
                    pass
                self.interfrag_dists.append(interfrag_dist)
                self.monitor_frag_dists_counter -= 1

            sys.stdout.flush()

        else:
            print(f"{self.__class__.__name__} pysis result: Number of cycles exceeded!")
            # print(f"{self.__class__.__name__} Cycles taken: {self.cur_cycle}")

        # Outside loop
        print()
        self.postprocess_opt()
        sys.stdout.flush()

    def _get_opt_restart_info(self):
        """To be re-implemented in the derived classes."""
        return dict()

    def _set_opt_restart_info(self, opt_restart_info):
        """To be re-implemented in the derived classes."""
        return

    def set_restart_info(self, restart_info):
        # Set restart information general to all optimizers
        self.last_cycle = restart_info["last_cycle"] + 1

        must_resize = self.last_cycle >= self.max_cycles
        if must_resize:
            self.max_cycles += restart_info["max_cycles"]

        self.coords = [np.array(coords) for coords in restart_info["coords"]]
        self.energies = restart_info["energies"]
        self.forces = [np.array(forces) for forces in restart_info["forces"]]
        self.steps = [np.array(step) for step in restart_info["steps"]]

        # Set subclass specific information
        self._set_opt_restart_info(restart_info)

        # Propagate restart information downwards to the geometry
        # self.geometry.set_restart_info(restart_info["geom_info"])


def scale_by_max_step(steps, max_step):
    steps_max = np.abs(steps).max()
    if steps_max > max_step:
        steps *= max_step / steps_max
    return steps


def get_scale_max(max_element):
    def scale_max(step):
        step_max = np.abs(step).max()
        if step_max > max_element:
            step *= max_element / step_max
        return step

    return scale_max


def restrict_step(steps, max_step):
    too_big = np.abs(steps) > max_step
    signs = np.sign(steps[too_big])
    steps[too_big] = signs * max_step
    return steps


# [1] https://arxiv.org/pdf/2101.04413.pdf
#     A Regularized Limited Memory BFGS method for Large-Scale Unconstrained
#     Optimization and itsefficient Implementations
#     Tankaria, Sugimoto, Yamashita, 2021
# [2] Regularization of Limited Memory Quasi-Newton Methods for Large-Scale
#     Nonconvex Minimization
#     https://arxiv.org/pdf/1911.04584.pdf
#     Kanzow, Steck 2021


def bfgs_multiply(
    s_list,
    y_list,
    vector,
    beta=1,
    P=None,
    logger=None,
    gamma_mult=True,
    mu_reg=None,
    inds=None,
    cur_size=None,
):
    """Matrix-vector product H·v.

    Multiplies given vector with inverse Hessian, obtained
    from repeated BFGS updates calculated from steps in 's_list'
    and gradient differences in 'y_list'.

    Based on algorithm 7.4 Nocedal, Num. Opt., p. 178."""

    assert len(s_list) == len(y_list), (
        "lengths of step list 's_list' and gradient list 'y_list' differ!"
    )

    cycles = len(s_list)
    q = vector.copy()
    alphas = list()
    rhos = list()

    if mu_reg is not None:
        # Regularized L-BFGS with ŷ = y + μs, see [1]
        assert mu_reg > 0.0
        y_list_reg = list()
        for i, si in enumerate(s_list):
            yi = y_list[i]
            y_hat = yi + mu_reg * si
            if y_hat.dot(si) <= 0:
                # See 2 in [1]
                y_hat = yi + (max(0, -si.dot(yi) / si.dot(si)) + mu_reg) + si
            y_list_reg.append(y_hat)
        y_list = y_list_reg

    # Store rho and alphas as they are also needed in the second loop
    for i in reversed(range(cycles)):
        s = s_list[i]
        y = y_list[i]
        rho = 1 / y.dot(s)
        rhos.append(rho)
        try:
            alpha = rho * s.dot(q)
            q -= alpha * y
        except ValueError:
            inds_i = inds[i]
            q_ = q.reshape(cur_size, -1)
            alpha = rho * s.dot(q_[inds_i].flatten())
            # This also modifies q!
            q_[inds_i] -= alpha * y.reshape(len(inds_i), -1)
        alphas.append(alpha)

    # Restore original order, so that rho[i] = 1/s_list[i].dot(y_list[i]) etc.
    alphas = alphas[::-1]
    rhos = rhos[::-1]

    if P is not None:
        r = spsolve(P, q)
        msg = "preconditioner"
    elif gamma_mult and (cycles > 0):
        s = s_list[-1]
        y = y_list[-1]
        gamma = s.dot(y) / y.dot(y)
        r = gamma * q
        msg = f"gamma={gamma:.4f}"
    else:
        r = beta * q
        msg = f"beta={beta:.4f}"

    if mu_reg is not None:
        msg += f" and μ_reg={mu_reg:.6f}"

    if logger is not None:
        msg = f"BFGS multiply using {cycles} previous cycles with {msg}."
        if len(s_list) == 0:
            msg += "\nProduced simple SD step."
        logger.debug(msg)

    for i in range(cycles):
        s = s_list[i]
        y = y_list[i]
        try:
            beta = rhos[i] * y.dot(r)
            r += s * (alphas[i] - beta)
        except ValueError:
            inds_i = inds[i]
            r_ = r.reshape(cur_size, -1)
            beta = rhos[i] * y.dot(r_[inds_i].flatten())
            # This also modifies r!
            r_[inds_i] += s.reshape(len(inds_i), -1) * (alphas[i] - beta)

    return r


# [1] https://link.springer.com/article/10.1007/s00214-016-1847-3
#     Birkholz, 2016
# [2] Geometry optimization in Cartesian coordinates: Constrained optimization
#     Baker, 1992
# [3] https://epubs.siam.org/doi/pdf/10.1137/S1052623496306450
#     BFGS WITH UPDATE SKIPPING AND VARYING MEMORY
#     Kolda, 1998
# [4] https://link.springer.com/article/10.1186/1029-242X-2012-241
#     New cautious BFGS algorithm based on modified Armijo-type line search
#     Wan, 2012
# [5] Numerical optimization, 2nd ed.
#     Nocedal, Wright
# [6] https://arxiv.org/abs/2006.08877
#     Goldfarb, 2020
# [7] https://pubs.acs.org/doi/10.1021/acs.jctc.9b00869
#     Hermes, Zádor, 2019
# [8] https://doi.org/10.1002/(SICI)1096-987X(199802)19:3<349::AID-JCC8>3.0.CO;2-T
#     Bofill, 1998
# [9] http://dx.doi.org/10.1016/S0166-1280(02)00209-9
#     Bungay, Poirier


def bfgs_update(H, dx, dg):
    first_term = np.outer(dg, dg) / dg.dot(dx)
    second_term = H.dot(np.outer(dx, dx)).dot(H) / dx.dot(H).dot(dx)
    return first_term - second_term, "BFGS"


def damped_bfgs_update(H, dx, dg):
    """See [5]"""
    dxdg = dx.dot(dg)
    dxHdx = dx.dot(H).dot(dx)
    theta = 1
    if dxdg < 0.2 * dxHdx:
        theta = 0.8 * dxHdx / (dxHdx - dxdg)
    r = theta * dg + (1 - theta) * H.dot(dx)

    first_term = np.outer(r, r) / r.dot(dx)
    second_term = H.dot(np.outer(dx, dx)).dot(H) / dxHdx
    return first_term - second_term, "damped BFGS"


def double_damp(
    s, y, H=None, s_list=None, y_list=None, mu_1=0.2, mu_2=0.2, logger=None
):
    """Double damped step 's' and gradient differences 'y'.

    H is the inverse Hessian!
    See [6]. Potentially updates s and y. y is only
    updated if mu_2 is not None.

    Parameters
    ----------
    s : np.array, shape (N, ), floats
        Coordiante differences/step.
    y : np.array, shape (N, ), floats
        Gradient differences
    H : np.array, shape (N, N), floats, optional
        Inverse Hessian.
    s_list : list of nd.array, shape (K, N), optional
        List of K previous steps. If no H is supplied and prev_ys is given
        the matrix-vector product Hy will be calculated through the
        two-loop LBFGS-recursion.
    y_list : list of nd.array, shape (K, N), optional
        List of K previous gradient differences. See s_list.
    mu_1 : float, optional
        Parameter for 's' damping.
    mu_2 : float, optional
        Parameter for 'y' damping.
    logger : logging.Logger, optional
        Logger to be used.

    Returns
    -------
    s : np.array, shape (N, ), floats
        Damped coordiante differences/step.
    y : np.array, shape (N, ), floats
        Damped gradient differences
    """
    sy = s.dot(y)
    # Calculate Hy directly
    if H is not None:
        Hy = H.dot(y)
    # Calculate Hy via BFGS_multiply as in LBFGS
    else:
        Hy = bfgs_multiply(s_list, y_list, y, logger=logger)
    yHy = y.dot(Hy)

    theta_1 = 1
    damped_s = ""
    if sy < mu_1 * yHy:
        theta_1 = (1 - mu_1) * yHy / (yHy - sy)
        s = theta_1 * s + (1 - theta_1) * Hy
        if theta_1 < 1.0:
            damped_s = ", damped s"
    msg = f"damped BFGS\n\ttheta_1={theta_1:.4f} {damped_s}"

    # Double damping
    damped_y = ""
    if mu_2 is not None:
        sy = s.dot(y)
        ss = s.dot(s)
        theta_2 = 1
        if sy < mu_2 * ss:
            theta_2 = (1 - mu_2) * ss / (ss - sy)
        y = theta_2 * y + (1 - theta_2) * s
        if theta_2 < 1.0:
            damped_y = ", damped y"
        msg = "double " + msg + f"\n\ttheta_2={theta_2:.4f} {damped_y}"

    if logger is not None:
        logger.debug(msg.capitalize())

    return s, y


# [1] Nocedal, Wright - Numerical Optimization, 2006
# [2] http://dx.doi.org/10.1016/j.jcp.2013.08.044
#     Badreddine, 2013
# [3] https://arxiv.org/abs/2006.08877
#     Goldfarb, 2020


class BFGS(Optimizer):
    def __init__(self, geometry, *args, update="bfgs", **kwargs):
        super().__init__(geometry, *args, **kwargs)

        self.update = update

        update_funcs = {
            "bfgs": self.bfgs_update,
            "damped": self.damped_bfgs_update,
            "double": self.double_damped_bfgs_update,
        }
        self.update_func = update_funcs[self.update]

    def prepare_opt(self):
        # Inverse Hessian
        self.H = self.eye

    @property
    def eye(self):
        size = self.geometry.coords.size
        return np.eye(size)

    def bfgs_update(self, s, y):
        rho = 1 / s.dot(y)
        V = self.eye - rho * np.outer(s, y)
        self.H = V.dot(self.H).dot(V.T) + rho * np.outer(s, s)

    def double_damped_bfgs_update(self, s, y, mu_1=0.2, mu_2=0.2):
        """Double damped BFGS update of inverse Hessian.

        See [3]. Potentially updates s and y."""

        # Call using the inverse Hessian 'H'
        s, y = double_damp(
            s,
            y,
            H=self.H,
            mu_1=mu_1,
            mu_2=mu_2,
            # logger=self.logger
        )
        if self.cur_cycle % self.print_every == 0 and self.cur_cycle > 0:
            self.log(f"s·y={s.dot(y):.6f} (damped)")
        self.bfgs_update(s, y)

    def damped_bfgs_update(self, s, y, mu_1=0.2):
        """Damped BFGS update of inverse Hessian.

        Potentially updates s.
        See Section 3.2 of [2], Eq. (30) - (33). There is a typo ;)
        It should be
            H_{k+1} = V_k H_k V_k^T + ...
        instead of
            H_{k+1} = V_k^T H_k V_k + ...
        """
        self.double_damped_bfgs_update(s, y, mu_2=None)

    def optimize(self):
        forces = self.geometry.forces
        energy = self.geometry.energy
        self.forces.append(forces)
        self.energies.append(energy)

        if self.cur_cycle > 0:
            # Gradient difference
            y = self.forces[-2] - forces
            # Coordinate difference / step
            s = self.steps[-1]
            # Curvature condition
            sy = s.dot(y)
            if self.cur_cycle % self.print_every == 0 and self.cur_cycle > 0:
                self.log(f"s·y={sy:.6f} (undamped)")
            # Hessian update
            self.update_func(s, y)

        # Results in simple SD step in the first cycle
        step = self.H.dot(forces)

        # Step restriction
        unscaled_norm = np.linalg.norm(step)
        step = scale_by_max_step(step, self.max_step)
        scaled_norm = np.linalg.norm(step)

        if self.cur_cycle % self.print_every == 0 and self.cur_cycle > 0:
            self.log(f"Calcualted {self.update} step")
            self.log(f"Unscaled norm(step)={unscaled_norm:.4f}")
            self.log(f"  Scaled norm(step)={scaled_norm:.4f}")

        return step


class FIRE(Optimizer):
    # https://doi.org/10.1103/PhysRevLett.97.170201
    """
    Structure optimization algorithm which is
    significantly faster than standard implementations of the conjugate gradient method
    and often competitive with more sophisticated quasi-Newton schemes
    It is based on conventional molecular dynamics
    with additional velocity modifications and adaptive time steps.
    """

    def __init__(
        self,
        # Geometry providing coords, forces, energy
        geometry,
        # Initial time step; adaptively scaled during optimization
        dt=0.1,
        # Maximum allowed time step when increasing dt
        dt_max=1,
        # Consecutive aligned steps before accelerating
        N_acc=2,
        # Factor to increase dt on acceleration
        f_inc=1.1,
        # Factor to reduce mixing a on acceleration; also shrinks dt on reset here
        f_acc=0.99,
        # Unused in this implementation; typical FIRE uses to reduce dt on reset
        f_dec=0.5,
        # Counter of aligned steps since last reset (start at 0)
        n_reset=0,
        # Initial mixing parameter for velocity/force mixing; restored on reset
        a_start=0.1,
        # Forwarded to base Optimizer
        **kwargs,
    ):
        self.dt = dt
        self.dt_max = dt_max
        # Accelerate after N_acc cycles
        self.N_acc = N_acc
        self.f_inc = f_inc
        self.f_acc = f_acc
        self.f_dec = f_dec
        self.n_reset = n_reset
        self.a_start = a_start

        self.a = self.a_start
        # The current velocity
        self.v = np.zeros_like(geometry.coords)
        # Store the velocities for every step
        self.velocities = [
            self.v,
        ]
        self.time_deltas = [
            self.dt,
        ]

        super().__init__(geometry, **kwargs)

    def _get_opt_restart_info(self):
        opt_restart_info = {
            "a": self.a,
            "dt": self.dt,
            "n_reset": self.n_reset,
            "time_delta": self.time_deltas[-1],
            "velocity": self.velocities[-1].tolist(),
        }
        return opt_restart_info

    def _set_opt_restart_info(self, opt_restart_info):
        self.a = opt_restart_info["a"]
        self.dt = opt_restart_info["dt"]
        self.n_reset = opt_restart_info["n_reset"]
        self.time_deltas = [
            opt_restart_info["time_delta"],
        ]
        velocity = np.array(opt_restart_info["velocity"], dtype=float)
        self.velocities = [
            velocity,
        ]

    def reset(self):
        pass

    def optimize(self):
        self.forces.append(self.geometry.forces)
        self.energies.append(self.geometry.energy)
        forces = self.forces[-1]
        mixed_v = (
            # As 'a' gets bigger we keep less old v.
            (1.0 - self.a) * self.v
            +
            # As 'a' gets bigger we use more new v
            # from the current forces.
            self.a * np.sqrt(np.dot(self.v, self.v) / np.dot(forces, forces)) * forces
        )
        last_v = self.velocities[-1]
        # Check if forces are still aligned with the last velocity
        if (self.cur_cycle > 0) and (np.dot(last_v, forces) > 0):
            if self.n_reset > self.N_acc:
                self.dt = min(self.dt * self.f_inc, self.dt_max)
                self.a *= self.f_acc
            self.n_reset += 1
        else:
            # Reset everything when 'forces' and 'last_v' aren't
            # aligned anymore.
            mixed_v = np.zeros_like(forces)
            # if self.cur_cycle % self.print_every == 0 and self.cur_cycle > 0:
            #     self.log("resetted velocities")
            self.a = self.a_start
            self.dt *= self.f_acc
            self.n_reset = 0

        v = mixed_v + self.dt * forces
        self.velocities.append(v)
        self.time_deltas.append(self.dt)
        steps = self.dt * v
        steps = self.scale_by_max_step(steps)

        velo_norm = np.linalg.norm(v)
        if self.cur_cycle % self.print_every == 0 and self.cur_cycle > 0:
            self.log(f"dt = {self.dt:.4f}, norm(v) {velo_norm:.4f}")

        return steps

    def __str__(self):
        return "FIRE optimizer"


class SteepestDescent(Optimizer):
    def __init__(
        self,
        geometry,
        alpha=0.1,
        bt_force=5,
        dont_skip_after=2,
        bt_max_scale=4,
        bt_disable=False,
        **kwargs,
    ):
        # Setting some default values
        self.alpha = alpha
        assert self.alpha > 0, "Alpha must be positive!"
        self.bt_force = bt_force
        self.dont_skip_after = dont_skip_after
        self.bt_max_scale = bt_max_scale
        self.bt_disable = bt_disable
        assert self.dont_skip_after >= 1
        self.cycles_since_backtrack = self.bt_force
        self.scale_factor = 0.5

        super(SteepestDescent, self).__init__(geometry, **kwargs)

        self.alpha0 = self.alpha
        self.alpha_max = self.bt_max_scale * self.alpha0

        # Keep the skipping history to avoid infinite skipping, e.g. always
        # return skip = False if we already skipped in the last n iterations.
        self.skip_log = list()

    def prepare_opt(self):
        # self.log("no backtracking in cycle 0")
        pass

    def optimize(self):
        self.forces.append(self.geometry.forces)
        self.energies.append(self.geometry.energy)

        if self.cur_cycle > 0:
            self.skip = self.backtrack(self.forces[-1], self.forces[-2])

        step = self.alpha * self.forces[-1]
        step = self.scale_by_max_step(step)
        return step

    def _get_opt_restart_info(self):
        opt_restart_info = {
            "alpha": self.alpha,
            "cycles_since_backtrack": self.cycles_since_backtrack,
        }
        return opt_restart_info

    def _set_opt_restart_info(self, opt_restart_info):
        self.alpha = opt_restart_info["alpha"]
        self.cycles_since_backtrack = opt_restart_info["cycles_since_backtrack"]

    def reset(self):
        if self.alpha > self.alpha0:
            self.alpha = self.alpha0
            self.log(
                f"Resetting! Current alpha is {self.alpha}. Lowering "
                f"it to {self.alpha0}."
            )

    def backtrack(self, cur_forces, prev_forces, reset_hessian=None):
        """Accelerated backtracking line search."""
        if self.bt_disable:
            return False

        epsilon = 1e-3

        _rms_func = lambda f: np.sqrt(np.mean(np.square(f)))
        cur_rms_force = _rms_func(cur_forces)
        prev_rms_force = _rms_func(prev_forces)

        rms_diff = (cur_rms_force - prev_rms_force) / np.abs(
            cur_rms_force + prev_rms_force
        )

        # Skip tells us if we overshot
        skip = False

        # When the optimiziation is converging cur_forces will
        # be smaller than prev_forces, so rms_diff will be negative
        # and hence smaller than epsilon, which is a positive number.

        # We went uphill, slow alpha
        # self.log(f"Backtracking: rms_diff = {rms_diff:.03f}")
        if rms_diff > epsilon:
            # self.log(f"Scaling alpha with {self.scale_factor:.03f}")
            # self.alpha = max(self.alpha0*.5, self.alpha*self.scale_factor)
            self.alpha *= self.scale_factor
            skip = True
            self.cycles_since_backtrack = self.bt_force
        # We continnue going downhill, rms_diff is smaller than epsilon
        else:
            self.cycles_since_backtrack -= 1
            # Check if we didn't accelerate in the previous cycles
            if self.cycles_since_backtrack < 0:
                self.cycles_since_backtrack = self.bt_force
                if self.alpha < self.alpha0:
                    # Reset alpha
                    self.alpha = self.alpha0
                    skip = True
                    # self.log(f"Reset alpha to alpha0 = {self.alpha0:.4f}")
                else:
                    # Accelerate alpha
                    self.alpha /= self.scale_factor
                    # self.log(f"Scaled alpha to {self.alpha:.4f}")

        # Avoid huge alphas
        if self.alpha > self.alpha_max:
            self.alpha = self.alpha_max
            # self.log(
            #     "Didn't accelerate as alpha would become too large. "
            #     f"keeping it at {self.alpha}."
            # )

        # Don't skip if we already skipped the previous iterations to
        # avoid infinite skipping.
        if (len(self.skip_log) >= self.dont_skip_after) and all(
            self.skip_log[-self.dont_skip_after :]
        ):
            # self.log(
            #     f"already skipped last {self.dont_skip_after} "
            #     "iterations don't skip now."
            # )
            skip = False
            if self.alpha > self.alpha0:
                self.alpha = self.alpha0
                # self.log("Resetted alpha to alpha0.")
        self.skip_log.append(skip)
        # self.log(f"alpha = {self.alpha:.4f}, skip = {skip}")

        return skip

class NaiveSteepestDescent(Optimizer):

    def __init__(self, geometry, alpha=0.1, **kwargs):
        super(NaiveSteepestDescent, self).__init__(geometry, **kwargs)
        self.alpha = alpha

    def optimize(self):

        self.forces.append(self.geometry.forces)

        step = self.alpha*self.forces[-1]
        step = self.scale_by_max_step(step)
        return step