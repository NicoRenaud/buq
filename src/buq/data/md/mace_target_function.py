import numpy as np
from emukit.core import ParameterSpace
from ..target_function import BaseTargetFunction


class MACETargetFunction(BaseTargetFunction):
    """Base class for MD-based target functions.
    
    This class provides the interface and shared functionality for molecular dynamics
    simulations used as target functions in Bayesian optimization/quadrature tasks.
    """

    def __init__(self, coordinate_file, mode: str = "cluster"):
        """Initialize the MD target function.
        
        Args:
            coordinate_file (str): Path to the XYZ file.
            mode (str): Execution mode. Options: "cluster", "local", "debug", "container".
                       Defaults to "cluster".
        """

        self.coordinate_file = coordinate_file
        self.parameter_space: ParameterSpace | None = None
        self.mode = mode
        self.available_modes = ["cluster", "local", "debug", "container"]
        self.has_ground_truth = False

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


    def get_ground_truth(self, samples: np.ndarray):
        """Retrieve the ground truth values for evaluation purposes.
        
        This method must be implemented by subclasses if ground truth data is available.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

