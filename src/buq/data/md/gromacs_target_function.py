import os
import subprocess
import numpy as np
from emukit.core import ParameterSpace
from ..target_function import BaseTargetFunction


class GROMACSTargetFunction(BaseTargetFunction):
    """Base class for MD-based target functions.
    
    This class provides the interface and shared functionality for molecular dynamics
    simulations used as target functions in Bayesian optimization/quadrature tasks.
    """

    def __init__(self, tpr_file, mode: str = "cluster"):
        """Initialize the MD target function.
        
        Args:
            tpr_file (str): Path to the GROMACS topology file (.tpr).
            mode (str): Execution mode. Options: "cluster", "local", "debug", "container".
                       Defaults to "cluster".
        """

        
        self.parameter_space: ParameterSpace | None = None
        self.tpr_file = tpr_file
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
            self.precommand = f"srun --mpi=pmix_v3 gmx_mpi mdrun -s {self.tpr_file}"  # for running on a cluster
            self.target_function = None

        elif self.mode == "container":
            self.task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
            self.precommand = (
                f"gmx_mpi mdrun -s {self.tpr_file}"  # for running in a container
            )
            self.target_function = None

        elif self.mode == "local":
            self.task_id = None
            self.precommand = f"gmx mdrun -s {self.tpr_file}"
            self.target_function = None

        elif self.mode == "debug":
            self.task_id = None
            self.precommand = f"mock mdrun -s {self.tpr_file}"
            self.target_function = None

    def __call__(self, samples: np.ndarray):
        """Evaluate the target function at given sample points.
        
        Args:
            samples (np.ndarray): Input samples at which to evaluate the function.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_forces(self, sample: np.ndarray):
        """Compute or retrieve forces for the current MD target function.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def write_plumed_file(self, sample: np.ndarray):
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
        """Execute an MD simulation command with GROMACS environment configuration.
        
        Configures GROMACS library paths and environment variables before executing
        the simulation command via subprocess.
        
        Args:
            command (str): Shell command to execute (typically a gmx mdrun command).
            
        Raises:
            subprocess.CalledProcessError: If the command execution fails.
        """
        # Set up GROMACS environment variables and preserve the existing PATH
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/local/gromacs/lib:" + env.get(
            "LD_LIBRARY_PATH", ""
        )
        env["PATH"] = "/usr/local/gromacs/bin:" + env.get(
            "PATH", "/bin:/usr/bin:/usr/local/bin"
        )  # Preserve system PATH

        # Execute the command
        try:
            subprocess.run(command, shell=True, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {command}")
            print(e)
