import os
import subprocess
import numpy as np
from emukit.core import ParameterSpace
from ..target_function import BaseTargetFunction


class LAMMPSTargetFunction(BaseTargetFunction):
    """Base class for MD-based target functions.
    
    This class provides the interface and shared functionality for molecular dynamics
    simulations used as target functions in Bayesian optimization/quadrature tasks.
    """

    def __init__(self,  mode: str = "cluster"):
        """Initialize the MD target function.
        
        Args:
            mode (str): Execution mode. Options: "cluster", "local", "debug", "container".
                       Defaults to "cluster".
        """

        
        self.parameter_space: ParameterSpace | None = None
        self.mode = mode
        self.available_modes = ["cluster", "local", "debug", "container"]
        self.has_ground_truth = False
        self._init_runner()

    @property
    def x(self):
        """Get the lower bounds of the parameter space.
        
        Returns:
            array-like: Lower bounds of the parameter space.
        """
        return self.parameter_space.get_bounds()[0]

    @property
    def y(self):
        """Get the upper bounds of the parameter space.
        
        Returns:
            array-like: Upper bounds of the parameter space.
        """
        return self.parameter_space.get_bounds()[1]

    def _init_runner(self):
        """Initialize the MD simulation runner based on the execution mode.
        
        Sets up the appropriate command prefix and environment for running
        MD simulations in the specified mode (cluster, local, debug, or container).
        
        Raises:
            ValueError: If the specified mode is not supported.
        """
        if self.mode not in self.available_modes:
            raise ValueError(
                f"mode {self.mode} not supported. The available modes are: {self.available_modes}"
            )

        if self.mode == "cluster":
            self.task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
            self.precommand = "mpirun -np 32 lmp -partition 1x32"  # for running on a cluster
            self.target_function = None

        elif self.mode == "container":
            self.task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
            self.precommand = (
                "lmp"  # for running in a container
            )
            self.target_function = None

        elif self.mode == "local":
            self.task_id = None
            self.precommand = "lmp"
            self.target_function = None

        elif self.mode == "debug":
            self.task_id = None
            self.precommand = "mock lmd"
            self.target_function = None

    def __call__(self, samples: np.ndarray):
        """Evaluate the target function at given sample points.
        
        Args:
            samples (np.ndarray): Input samples at which to evaluate the function.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_forces(self, samples: np.ndarray):
        """Compute or retrieve forces for the current MD target function.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _get_forces_debug(self, samples: np.ndarray):
        """Compute or retrieve forces for the current MD target function.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def write_lammps_input(self, samples: np.ndarray):
        """Write LAMMPS input file for enhanced sampling.
        
        This method must be implemented by subclasses to generate the appropriate
        LAMMPS input file for biased MD simulations.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def write_plumed_file(self, samples: np.ndarray):
        """Write PLUMED configuration file for enhanced sampling.
        
        This method must be implemented by subclasses to generate the appropriate
        PLUMED input file for biased MD simulations.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_ground_truth(self, samples: np.ndarray):
        """Retrieve the ground truth values for evaluation purposes.
        
        This method must be implemented by subclasses if ground truth data is available.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @staticmethod
    def run_command(command):
        """Run a molecular dynamics job on a cluster using a prepared MPI command.

        This helper constructs and executes a cluster-ready command for running
        the LAMMPS MD engine with MPI parallelization. The command is executed
        in a shell while preserving the current environment variables.

        Parameters
        ----------
        command : str
            Command-line arguments or input file specification for the MD
            executable.

        Returns
        -------
        None
            The command is executed for its side effects; output is redirected
            to ``lmp.out``. Any failures are printed to standard output.
        """

        # Build the full srun command for the cluster        
        # setup_cmds = f"mpirun -np 32 lmp -partition 1x32 {command} &> lmp.out"  # Example command

        # Use current environment variables (assuming conda env already active)
        env = os.environ.copy()

        try:
            subprocess.run(
                command, shell=True, check=True, executable="/bin/bash", env=env
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed: {command}")
            print(e)
