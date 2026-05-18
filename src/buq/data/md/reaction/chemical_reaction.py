"""Chemical Reaction Target Function"""

import numpy as np
import os
import glob
from emukit.core import ContinuousParameter, ParameterSpace
from ase import units
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read
from mace.calculators import MACECalculator
from ase.calculators.plumed import Plumed

from buq.data.md.mace_target_function import MACETargetFunction

PATH = os.path.dirname(os.path.abspath(__file__))


class ChemicalReaction(MACETargetFunction):
    """Chemical Reaction Target Function"""

    def __init__(
        self,
        kappa_d1=1000,
        kappa_d2=100,
        measure_after_ps=100,
        temperature=300,
        nsteps=80000,
        mode="cluster",
        coordinate_file="chemical_reaction.xyz",
        mace_model_file="MACE_2_swa.model",
        device="cpu",
        timesteps=None,
    ):
        """
        kappa_d2 : float, optional
            Harmonic force constant associated with d2. Default is 100.
        kappa_d1 : float, optional
            Harmonic force constant associated with d1. Default is 1000.
        measure_after_ps : float, optional
            Time threshold (in ps). Only data with time > measure_after_ps
            are used in the force estimate.
        tpr_file : str, optional
            Path to the .tpr file. Default is 'md.tpr'.
        mode : str, optional
            Mode for running the MD simulation. Default is 'cluster'.
        """

        super().__init__(coordinate_file, mode)
        self.name = "chemical_reaction"
        self.kappa_d1 = kappa_d1
        self.kappa_d2 = kappa_d2
        self.measure_after_ps = measure_after_ps
        self.coordinate_file = (
            os.path.join(PATH, coordinate_file)
            if not os.path.isabs(coordinate_file)
            else coordinate_file
        )
        self.mace_model_file = (
            os.path.join(PATH, mace_model_file)
            if not os.path.isabs(mace_model_file)
            else mace_model_file
        )
        self.device = device
        self.temperature = temperature
        self.nsteps = nsteps
        self.timesteps = timesteps
        if self.timesteps is None:
            self.timesteps = 0.5 * units.fs

        # parameter space
        self.parameter_space = ParameterSpace(
            [ContinuousParameter("d2", 1.8, 3.5), ContinuousParameter("d1", 1.2, 2.8)]
        )

        self.debug = self.mode == "debug"
        if self.debug:
            self._run_md = self._run_md_debug
        else:
            self._run_md = self._run_md_cluster

        self.has_ground_truth = True

    def __call__(self, samples: np.ndarray):
        """
        Run a biased molecular dynamics simulation with PLUMED using a MACE potential.

        The simulation steers two C–X bond distances independently rather than their
        difference, due to the stiffness of the C–F bond. Angular restraints are applied
        to control molecular geometry during the pulling process. Trajectories and
        collective variables are written to disk for post-processing.


        Parameters
        ----------
        samples : numpy.ndarray
            Array containing the values of the two collective variables [(d2, d1), ... ]
            with d2 : float Target value (in Å) for the second distance collective variable.
            d1 : float Target value (in Å) for the first distance collective variable.
        """
        samples = np.atleast_2d(samples)
        forces = []
        for spl in samples:
            fx, fy = self._run_md(spl)
            forces.append([fx, fy])
        return np.array(forces)

    def _run_md_debug(self, sample: np.ndarray):
        """
        Debug function to simulate running an MD simulation with the given x and y angles and return the value offree energy derivatives

        Parameters
        ----------
        sample : array
            Array of shape (2,) containing x and y

        Returns
        -------
        derivatives : array
            [dF/dphi, dF/dpsi]
        """

        print(f"Debug mode: Simulating MD run for x={sample[0]}, y={sample[1]}")
        return self._get_force_debug(sample)

    def _run_md_cluster(self, sample: np.ndarray):
        """
        Runs an MD simulation with the given x and y positions and returns the value of the free energy and its derivatives with respect to x and y.

        Parameters
        ----------
        sample : array
            Array of shape (2,) containing x and y

        Returns
        -------
        derivatives : array
            [dF/dphi, dF/dpsi]
        """
        d2, d1 = sample

        timestep = 0.5 * units.fs
        atoms = read(self.coordinate_file, "0")
        potential = MACECalculator(model_paths=self.mace_model_file, device=self.device)

        # pulling rc=d2-d1 does not work, C-F bond is too strong.
        # "steer: MOVINGRESTRAINT ARG=rc STEP0=5000 AT0=2.0 KAPPA0=50000.00 STEP1=255000 AT1=-1.0",
        # notice we start from the lowest energy, with C bound to F.
        bias = [
            "UNITS LENGTH=A TIME=ps ENERGY=kcal/mol",
            "d1: DISTANCE ATOMS=1,4 NOPBC",
            "d2: DISTANCE ATOMS=1,5 NOPBC",
            "rc: COMBINE ARG=d1,d2 COEFFICIENTS=-1,1 PERIODIC=NO",
            f"steer: MOVINGRESTRAINT ARG=d1,d2 STEP0=1000 AT0=2.64,1.84 KAPPA0=1000.0,100.0 STEP1=5000 AT1={d1},{d2}",
            "ener: ENERGY",
            "an: ANGLE ATOMS=1,2,4,5 NOPBC",
            "res: RESTRAINT ARG=an AT=pi*0.5 KAPPA=100.0",
            "an2: ANGLE ATOMS=1,5,4 NOPBC",
            "res2: RESTRAINT ARG=an2 AT=0.0 KAPPA=100.0",
            f"PRINT ARG=* STRIDE=100 FILE=colvars/COLVAR_{d1}_{d2}",
            "FLUSH STRIDE=500",
        ]

        atoms.calc = Plumed(calc=potential, input=bias, timestep=timestep, atoms=atoms)

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)

        dyn = Bussi(
            atoms,
            timestep,
            temperature_K=self.temperature,
            taut=100 * timestep,
            # logfile=f"colvars/log_{d1}_{d2}",
            logfile=None,
            loginterval=500,
        )

        def write_frame():
            dyn.atoms.write(f"colvars/t_{d1}_{d2}.xyz", append=True)

        dyn.attach(write_frame, interval=500)
        dyn.run(self.nsteps)

        return self.get_forces(sample)

    def get_forces(self, samples: np.ndarray):
        """
        Compute mean restoring forces for two the two distance collective variables.

        Parameters
        ----------
        samples : numpy.ndarray
            Array containing the values of the two collective variables [(d2, d1), ... ]

        Returns
        -------
        numpy.ndarray
            Array containing the negative mean restoring forces
            [F_d2, F_d1].
        """
        d2, d1 = samples
        pattern = f"colvars/COLVAR_{d1}_{d2}*"

        # Glob for the matching file
        files = glob.glob(pattern)

        if len(files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one COLVAR file, found {len(files)} matching: {pattern}"
            )

        filename = files[0]
        data = np.genfromtxt(filename)

        mask = data[:, 0] > self.measure_after_ps
        data = data[mask]
        d1_real = np.mean((data[:, 1]))
        d2_real = np.mean((data[:, 2]))

        force_d2 = np.mean((d2_real - d2) * self.kappa_d2)
        force_d1 = np.mean((d1_real - d1) * self.kappa_d1)
        return np.array([-force_d2, -force_d1])

    def get_ground_truth(self, samples):
        """
        Debug function to get the ground truth free energy at given x and y distance.

        Args:
            samples (np.ndarray): Array of shape (n_samples, 2) containing atomic distances

        Returns:
            energy (np.ndarray): Array of shape (n_samples,) containing the free energy values.
            forces (np.ndarray): Array of shape (n_samples, 2) containing the forces [dF/dphi, dF/dpsi].
        """

        from buq.data.exact.chemical_reaction import (
            ChemicalReaction as ChemicalReactionExact,
        )
        from buq.data.exact.interpolated_target_function import (
            Interpolated2DTargetFunction,
        )

        crex = ChemicalReactionExact()
        energy = crex(samples)

        xpts, ypts, dx, dy = crex._get_data(derivatives=True)
        target_function_dx = Interpolated2DTargetFunction(dx, xpts, ypts)
        target_function_dy = Interpolated2DTargetFunction(dy, xpts, ypts)
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
        from buq.data.exact.chemical_reaction import (
            ChemicalReaction as ChemicalReactionExact,
        )
        from buq.data.exact.interpolated_target_function import (
            Interpolated2DTargetFunction,
        )

        x_sample, y_sample = sample
        crex = ChemicalReactionExact()
        xpts, ypts, dx, dy = crex._get_data(derivatives=True)

        target_function_dx = Interpolated2DTargetFunction(dx, xpts, ypts)
        target_function_dy = Interpolated2DTargetFunction(dy, xpts, ypts)
        
        return [
            target_function_dx([x_sample, y_sample])[0, 0],
            target_function_dy([x_sample, y_sample])[0, 0],
        ]
