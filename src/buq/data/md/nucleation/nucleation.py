"""Nucleation Target Function"""

import numpy as np
import os
import glob
from emukit.core import ContinuousParameter, ParameterSpace
from buq.data.md.lammps_target_function import LAMMPSTargetFunction

PATH = os.path.dirname(os.path.abspath(__file__))


class Nucleation(LAMMPSTargetFunction):
    """Nucleation Target Function"""

    def __init__(
        self,
        name: str,
        tpr_file,
        steeringsteps=250000,
        equil_steps=250000,
        measure_after_ps=1000,
        kappa_es=1000,
        nsteps=250000,
        mode="cluster",
    ):
        """
        Initialize a Nucleation object.

        Parameters
        ----------
        tpr_file : str
            Path to the GROMACS topology file (.tpr).
        measure_after_ps : float, optional
            Time threshold (in ps). Only data with time > measure_after_ps
            are used in the force estimate. Default is 1000.
        kappa_es : float, optional
            Harmonic force constant associated with the bias. Default is 1000.
        mode : str, optional
            Execution mode. Options: "cluster", "local", "debug", "container".
            Defaults to "cluster".
        """
        super().__init__(mode)
        self.name = name
        self.tpr_file = tpr_file
        self.mode = mode
        self.measure_after_ps = measure_after_ps
        self.kappa_es = kappa_es
        self.steeringsteps = steeringsteps
        self.equil_steps = equil_steps
        self.nsteps = nsteps

        # parameter space
        self.parameter_space = ParameterSpace([ContinuousParameter("es", 0.0, 288.0)])

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
            es = spl[0]
            force = self._run_md(es)
            forces.append([force])
        return np.array(forces)
    
    def _run_md_debug(self, samples: np.ndarray):
        """
        Debug function to simulate running an MD simulation with the given phi and psi angles and return the value of the free energy and its derivatives with respect to phi and psi.

        Parameters
        ----------
        es : float
            Target value of the collective variable.

        Returns
        -------
        derivatives : array
            [dE/des
        """
        print(f"Debug mode: Simulating MD run for es={samples[0]}")
        return self._get_forces_debug(samples)

    def _get_forces_debug(self, samples: np.ndarray):
        """
        Debug function to simulate running an MD simulation with the given phi and psi angles and return the value of the free energy and its derivatives with respect to phi and psi.

        Parameters
        ----------
        es : float
            Target value of the collective variable.

        Returns
        -------
        derivatives : array
            [dF/dphi, dF/dpsi]
        """
        from buq.data.exact.nucleation import Nucleation as nucleation_exact
        from scipy.differentiate import derivative
        nucleation = nucleation_exact()
        return derivative(nucleation, samples)
    
    def get_ground_truth(self, samples):
        """Compute the ground truth values for the nucleation target function.

        This method evaluates the exact analytical nucleation function at the
        provided sample points, providing reference values for comparison
        against MD-based estimates.

        Parameters
        ----------
        samples : np.ndarray
            Array of sample points where the function should be evaluated.
            Typically contains values of the collective variable (e.g., cluster size).

        Returns
        -------
        np.ndarray
            Ground truth function values at the input sample points.
        """
        
        from buq.data.exact.nucleation import Nucleation as nucleation_exact
        nucleation = nucleation_exact()
        return nucleation(samples)
        
    
    def write_lammps_input(self, samples: np.ndarray):
        """
        Generate a LAMMPS input file for a biased MD simulation with PLUMED.

        Parameters
        ----------
        es : float
            Target value of the collective variable. Used to label output,
            dump, and PLUMED files.
        nsteps : int
            Number of MD integration steps to run.
        """
        es = samples[0]
        es_label = f"{es:.3f}".replace(".", "_")

        filename = f"colvars/input_{es}"
        content = f"""variable    temperature equal 300.0
        variable    tempDamp equal 100.0

        variable    pressure equal 1.
        variable    pressureDamp equal 1000.0 # This is 1 ps

        variable    seed equal 745821

        units       real
        atom_style  full

        read_data   water.data.0

        variable    out_freq equal 1000
        variable    out_freq2 equal 1000

        timestep    2.0

        neigh_modify    delay 7 every 1

        include     in.tip4p

        thermo          ${{out_freq}}
        thermo_style    custom step temp pe etotal epair emol press lx ly lz vol pxx pyy pzz pxy pxz pyz

        restart     ${{out_freq}} restart.lmp restart2.lmp

        dump            myDump all atom ${{out_freq2}} colvars/dump_{es_label}.water
        dump_modify     myDump append yes

        fix             1 all plumed plumedfile colvars/plumed_{es_label}.dat outfile plumed_out{es}
        fix             2 all shake 1e-6 200 0 b 1 a 1
        fix             3 all nph iso ${{pressure}} ${{pressure}} ${{pressureDamp}}
        fix             4 all temp/csvr ${{temperature}} ${{temperature}} ${{tempDamp}} ${{seed}}
        velocity        all create ${{temperature}} ${{seed}} dist gaussian

        run             {self.nsteps}

        write_data      data_{es_label}.final

        write_restart   restart.lmp
        """

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

    def write_plumed_file(self, samples: np.ndarray):
        """
        Generate a PLUMED input file with custom environment similarity restraints.

        Parameters:
        samples (np.ndarray): Array containing the target value for environment similarity.

        """
        es_target = samples[0]

        # Build file name
        es_label = f"{es_target:.3f}".replace(".", "_")
        file_path = f"colvars/plumed_{es_label}.dat"

        with open(file_path, "w", encoding="utf-8") as file:
            # VOLUME
            file.write("vol: VOLUME\n\n")

            # ENVIRONMENTSIMILARITY block 1
            file.write("ENVIRONMENTSIMILARITY ...\n")
            file.write(" SPECIES=1-864:3\n")
            file.write(" SIGMA=0.055\n")
            file.write(" CRYSTAL_STRUCTURE=CUSTOM\n")
            file.write(" LABEL=refcv\n")
            file.write(" REFERENCE_1=Environments/IceIhExtendedEnvironments/env1.pdb\n")
            file.write(" REFERENCE_2=Environments/IceIhExtendedEnvironments/env2.pdb\n")
            file.write(" REFERENCE_3=Environments/IceIhExtendedEnvironments/env3.pdb\n")
            file.write(" REFERENCE_4=Environments/IceIhExtendedEnvironments/env4.pdb\n")
            file.write(" MORE_THAN={RATIONAL R_0=0.5 NN=12 MM=24}\n")
            file.write(" MEAN\n")
            file.write("... ENVIRONMENTSIMILARITY\n\n")

            # ENVIRONMENTSIMILARITY block 2
            file.write("ENVIRONMENTSIMILARITY ...\n")
            file.write(" SPECIES=1-864:3\n")
            file.write(" SIGMA=0.055\n")
            file.write(" CRYSTAL_STRUCTURE=CUSTOM\n")
            file.write(" LABEL=refcv2\n")
            file.write(" REFERENCE_1=Environments/IceIcExtendedEnvironments/env1.pdb\n")
            file.write(" REFERENCE_2=Environments/IceIcExtendedEnvironments/env2.pdb\n")
            file.write(" MORE_THAN={RATIONAL R_0=0.5 NN=12 MM=24}\n")
            file.write(" MEAN\n")
            file.write("... ENVIRONMENTSIMILARITY\n\n")

            # diff MATHEVAL
            file.write(
                "diff: MATHEVAL ARG=refcv2.mean,refcv.mean FUNC=((x-0.26)/(0.58-0.26)-(y-0.29)/(0.80-0.29)) PERIODIC=NO\n\n"
            )

            # UPPER_WALLS
            file.write(
                "UPPER_WALLS ARG=diff AT=0.04 KAPPA=100000 EXP=2 LABEL=uwall\n\n"
            )

            # Q6
            file.write(
                "Q6 SPECIES=1-288:3 SWITCH={CUBIC D_0=0.3 D_MAX=0.35} VMEAN LABEL=q6\n\n"
            )

            # diff2 MATHEVAL
            file.write(
                "diff2: MATHEVAL ARG=q6.vmean,refcv.mean FUNC=((x-0.0668781995)/(0.39184059-0.0668781995)-(y-0.2899390548628429)/(0.7838534089775562-0.2899390548628429)) PERIODIC=NO\n\n"
            )

            # MOVINGRESTRAINT
            file.write("restraint_more_es: MOVINGRESTRAINT ...\n")
            file.write(" ARG=refcv.morethan\n")
            file.write(" STEP0=0 AT0=288 KAPPA0=0\n")
            file.write(f" STEP1={self.steeringsteps} AT1=288 KAPPA1={self.kappa_es}\n")
            file.write(
                f" STEP2={self.equil_steps + self.steeringsteps} AT2={es_target} KAPPA2={self.kappa_es}\n"
            )
            file.write("...\n\n")

            # PRINT
            file.write(
                f"PRINT STRIDE=500 ARG=refcv.morethan,vol,* FILE=colvars/COLVAR_{es_label}\n"
            )

    def get_forces(self, samples: np.ndarray):
        """
        Compute the mean force extered by the umbrella on the similairity parameter.

        The function locates a single COLVAR file corresponding to the target value
        of 'es', discards data before a given equilibration time, estimates the mean
        realized value of the collective variable, and computes the average harmonic
        restoring force.

        Parameters
        ----------
        es : float
            Target (setpoint) value of the collective variable. (environment similarity)
        Returns
        -------
        numpy.ndarray
            One-dimensional array containing the negative mean restoring force.
            Shape: (1,).
        """

        es = samples[0]

        # Format the target value to match the COLVAR filename convention
        es_label = f"{es:.3f}".replace(".", "_")
        pattern = f"colvars/COLVAR_{es_label}.*"

        # Find the COLVAR file matching the target value
        files = glob.glob(pattern)

        # Load COLVAR data (assumes time in column 0, CV value in column 1)
        filename = files[0]
        data = np.genfromtxt(filename)

        # Discard data prior to the equilibration time
        mask = data[:, 0] > self.measure_after_ps
        data = data[mask]

        # Mean realized value of the collective variable after equilibration
        es_real = np.mean(data[:, 1])

        # Mean harmonic restoring force
        force_es = np.mean((es_real - es) * self.kappa_es)

        # Return force with sign convention expected by the caller
        return np.array([-force_es])

    def _run_md_cluster(self, sample: np.ndarray):
        """Run a molecular dynamics job on a cluster using a prepared MPI command.

        This helper constructs and executes a cluster-ready command for running
        the LAMMPS MD engine with MPI parallelization. The command is executed
        in a shell while preserving the current environment variables.

        Parameters
        ----------
        sample : np.ndarray
            Input sample at which to evaluate the target function.

        Returns
        -------
        None
            The command is executed for its side effects; output is redirected
            to ``lmp.out``. Any failures are printed to standard output.
        """
        
        self.write_lammps_input(sample)
        self.write_plumed_file(sample)

        command = (
            self.precommand
            + f" -in colvars/input_{sample[0]}"
        )

        self.run_command(command)
        derivatives = self.get_forces(sample)
        return derivatives
