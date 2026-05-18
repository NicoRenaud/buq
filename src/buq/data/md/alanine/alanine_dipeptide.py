"""
Alanine Dipeptide Target Function
"""
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from buq.data.md.gromacs_target_function import GROMACSTargetFunction


class AlanineDipeptideMD(GROMACSTargetFunction):
    """
    A class for running an MD simulation of an alanine dipeptide.
    """

    def __init__(
        self,
        kappa=200,
        ns=0.2,
        measure_after_ps=100,
        nsteps=500000,
        mode: str = "cluster",
        tpr_file="md.tpr",
    ):
        """
        Initialize an AlanineDipeptide object.

        Parameters
        ----------
        kappa : float, optional
            Force constant for harmonic restraints on phi and psi angles.
            Default is 200.
        ns : float, optional
            Number of steps between each measurement of the free energy.
            Default is 0.2.
        measure_after_ps : float, optional
            Time in ps after which to start measuring the free energy.
            Default is 100.
        nsteps : int, optional
            Total number of steps in the MD simulation.
            Default is 500000.
        mode: str, optional
            how to run the simulations, options are:
                cluster: run MD on a cluster
                local: run MD locally (this will still require GROMACS and PLUMED to be installed locally)
                debug: do not run actual MD simulations, but return the analytical forces from the alanine dipeptide target function.
        tpr_file : str, optional
            Path to the TPR file for the MD simulation.
            Default is 'md.tpr'.

        Notes
        -----
        If deploy_on_cluster is True, the task_id will be read from the environment variable SLURM_ARRAY_TASK_ID.
        This has been tested using GROMACS 2023 and PLUMED 2.9.0
        """

        super().__init__(tpr_file, mode)

        # Simulation setup
        self.ns = ns
        self.measure_after_ps = measure_after_ps  # this is above the maximum steering time, which is around 30 ps (2*pi * 1000 + 1500)*0.002 = 30 ps
        nsteps = nsteps * ns
        self.nsteps = int(nsteps)
        self.kappa_phi = kappa
        self.kappa_psi = kappa
        self.has_ground_truth = True

        # parameter space
        self.parameter_space = ParameterSpace(
            [
                ContinuousParameter("x", -np.pi, np.pi),
                ContinuousParameter("y", -np.pi, np.pi),
            ]
        )

        self.debug = self.mode == "debug"
        if self.debug:
            self._run_md = self._run_md_debug
        else:
            self._run_md = self._run_md_cluster

    def __call__(self, samples: np.ndarray):
        """
        Run an MD simulation with the given phi and psi angles and return the value of the free energy and its derivatives with respect to phi and psi.

        Parameters
        ----------
        samples : array
            Array of shape (2,) containing phi and psi angles in radians.

        Returns
        -------
        derivatives : array
            [dF/dphi, dF/dpsi]
        """
        samples = np.atleast_2d(samples)
        forces = []
        for spl in samples:
            fx, fy = self._run_md(spl)
            forces.append([fx, fy])
        return np.array(forces)

    def _run_md_debug(self, sample: np.ndarray):
        """
        Debug function to simulate running an MD simulation with the given phi and psi angles and return the value of the free energy and its derivatives with respect to phi and psi.

        Parameters
        ----------
        phi : float
            Target phi angle in radians.
        psi : float
            Target psi angle in radians.

        Returns
        -------
        derivatives : array
            [dF/dphi, dF/dpsi]
        """
        
        print(f"Debug mode: Simulating MD run for phi={sample[0]}, psi={sample[1]}")
        return self._get_force_debug(sample)

    def _run_md_cluster(self, sample: np.ndarray):
        """
        Runs an MD simulation with the given phi and psi angles and returns the value of the free energy and its derivatives with respect to phi and psi.

        Parameters
        ----------
        sample : array
            Array of shape (2,) containing phi and psi angles in radians.

        Returns
        -------
        derivatives : array
            [dF/dphi, dF/dpsi]
        """
        phi, psi = sample
        self.write_plumed_file(sample)

        # self.run_command("rm *#*") #remove old colvars etc
        command = (
            self.precommand
            + f" -plumed Colvars/plumed_{phi:.3f}_{psi:.3f}.dat -nsteps {self.nsteps} -x Colvars/traj_{phi:.3f}_{psi:.3f}.xtc"
        )
        self.run_command(command)
        derivatives = self.get_forces(sample, measure_after_ps=self.measure_after_ps)
        return derivatives

    def get_forces(
        self, sample: np.ndarray, kappa_phi=200, kappa_psi=200, measure_after_ps=1000
    ):
        """
        Gets the force after doing a restraint md simulation

        Parameters
        ----------
        phi_value : float
            Target phi angle in radians.
        psi_value : float
            Target psi angle in radians.
        kappa_phi : float, optional
            Force constant for phi.
            Default is 200.
        kappa_psi : float, optional
            Force constant for psi.
            Default is 200.
        measure_after_ps : float, optional
            Time in ps after which to start measuring the free energy.
            Default is 1000.

        Returns
        -------
        derivatives : array
            [dF/dphi, dF/dpsi]
        """
        phi_value, psi_value = sample
        data = np.genfromtxt(f"Colvars/COLVAR_{phi_value:.3f}_{psi_value:.3f}")
        data = data[data[:, 0] > measure_after_ps]

        # Mean values
        mean_vals = np.mean(data[:, 1:5], axis=0)
        sin_phi_real, cos_phi_real, sin_psi_real, cos_psi_real = mean_vals

        sin_phi_umbrella, cos_phi_umbrella = np.sin(phi_value), np.cos(phi_value)
        sin_psi_umbrella, cos_psi_umbrella = np.sin(psi_value), np.cos(psi_value)

        # Forces along sin/cos
        force_phi_vec = (
            np.array([sin_phi_real - sin_phi_umbrella, cos_phi_real - cos_phi_umbrella])
            * kappa_phi
        )
        force_psi_vec = (
            np.array([sin_psi_real - sin_psi_umbrella, cos_psi_real - cos_psi_umbrella])
            * kappa_psi
        )

        # Total forces with sign
        sign_phi = (
            -1
            if np.arctan2(sin_phi_real, cos_phi_real)
            < np.arctan2(sin_phi_umbrella, cos_phi_umbrella)
            else 1
        )
        sign_psi = (
            -1
            if np.arctan2(sin_psi_real, cos_psi_real)
            < np.arctan2(sin_psi_umbrella, cos_psi_umbrella)
            else 1
        )

        force_phi = np.linalg.norm(force_phi_vec) * sign_phi
        force_psi = np.linalg.norm(force_psi_vec) * sign_psi

        return np.array([-force_phi, -force_psi])

    def write_plumed_file(
        self,
        sample: np.ndarray,
        kappa_phi=200,
        kappa_psi=200,
        current_phi=0.0,
        current_psi=0.0,
    ):
        """
        Generates a PLUMED input file for torsional restraints on phi and psi angles.

        Args:
            phi (float): Target phi angle in radians.
            psi (float): Target psi angle in radians.
            kappa_phi (float): Force constant for phi.
            kappa_psi (float): Force constant for psi.
            current_phi (float): Current phi reference value.
            current_psi (float): Current psi reference value.
        """
        equisteps = 500
        moving_speed = 1000
        build_up_kappa_steps = 1000 + equisteps

        phi, psi = sample

        angles = {
            "phi": {"target": phi, "kappa": kappa_phi, "current": current_phi},
            "psi": {"target": psi, "kappa": kappa_psi, "current": current_psi},
        }

        filename = f"Colvars/plumed_{phi:.3f}_{psi:.3f}.dat"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("#vim:ft=plumed\n")
            f.write("MOLINFO STRUCTURE=simulations_essentials/diala.pdb\n")
            f.write("UNITS LENGTH=A TIME=ps ENERGY=kcal/mol\n")
            f.write("phi: TORSION ATOMS=@phi-2\n")
            f.write("psi: TORSION ATOMS=@psi-2\n")
            f.write("cos_phi: MATHEVAL arg=phi FUNC=cos(x) PERIODIC=NO\n")
            f.write("sin_phi: MATHEVAL arg=phi FUNC=sin(x) PERIODIC=NO\n")
            f.write("cos_psi: MATHEVAL arg=psi FUNC=cos(x) PERIODIC=NO\n")
            f.write("sin_psi: MATHEVAL arg=psi FUNC=sin(x) PERIODIC=NO\n")

            for angle_name, info in angles.items():
                distance = abs(info["current"] - info["target"])
                step_to = int(build_up_kappa_steps + distance * moving_speed)
                for trig in ["cos", "sin"]:
                    target_val = (
                        np.cos(info["target"])
                        if trig == "cos"
                        else np.sin(info["target"])
                    )
                    current_val = (
                        np.cos(info["current"])
                        if trig == "cos"
                        else np.sin(info["current"])
                    )
                    f.write(
                        f"restraint_{angle_name}_{trig}: MOVINGRESTRAINT ...\n"
                        f"ARG={trig}_{angle_name}\n"
                        f"STEP0={equisteps} AT0={current_val} KAPPA0=0\n"
                        f"STEP1={build_up_kappa_steps} AT1={current_val} KAPPA1={info['kappa']}\n"
                        f"STEP2={step_to} AT2={target_val} KAPPA2={info['kappa']}\n"
                        "...\n"
                    )

            f.write(
                f"PRINT ARG=sin_phi,cos_phi,sin_psi,cos_psi,*.* "
                f"FILE=Colvars/COLVAR_{phi:.3f}_{psi:.3f} STRIDE=100\n"
            )

    def get_ground_truth(self, samples: np.ndarray):
        """
        Debug function to get the ground truth free energy at given phi and psi angles.

        Args:
            samples (np.ndarray): Array of shape (n_samples, 2) containing phi and psi angles in radians.

        Returns:
            energy (np.ndarray): Array of shape (n_samples,) containing the free energy values.
            forces (np.ndarray): Array of shape (n_samples, 2) containing the forces [dF/dphi, dF/dpsi].
        """

        from buq.data.exact.alanine_dipeptide import AlanineDipeptide2D
        from buq.data.exact.interpolated_target_function import (
            Interpolated2DTargetFunction,
        )

        alanine = AlanineDipeptide2D()
        energy = alanine(samples)

        phi, psi, dx, dy = alanine.get_data(derivatives=True)
        target_function_dx = Interpolated2DTargetFunction(dx, phi, psi)
        target_function_dy = Interpolated2DTargetFunction(dy, phi, psi)
        forces = [target_function_dx(samples), target_function_dy(samples)]
        return energy, np.array(forces)

    def _get_force_debug(self, sample: np.ndarray):
        """
        Debug function to get the force at a given phi and psi angles.

        Args:
            phi_sample (float): Phi angle in radians.
            psi_sample (float): Psi angle in radians.
            measure_after_ps (float): Measure after ps (default is None)

        Returns:
            list: [dF/dphi, dF/dpsi]
        """
        from buq.data.exact.alanine_dipeptide import AlanineDipeptide2D
        from buq.data.exact.interpolated_target_function import (
            Interpolated2DTargetFunction,
        )

        phi_sample, psi_sample = sample
        alanine = AlanineDipeptide2D()
        phi, psi, dx, dy = alanine.get_data(derivatives=True)

        target_function_dx = Interpolated2DTargetFunction(dx, phi, psi)
        target_function_dy = Interpolated2DTargetFunction(dy, phi, psi)

        return [
            target_function_dx([phi_sample, psi_sample])[0, 0],
            target_function_dy([phi_sample, psi_sample])[0, 0],
        ]
