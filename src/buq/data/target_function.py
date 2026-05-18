import numpy as np
from emukit.core import ParameterSpace, ContinuousParameter

class BaseTargetFunction:
    """
    Base class for target functions used in Bayesian optimization/quadrature tasks.

    This class provides a common interface for target functions and defines a parameter space
    that can be overridden by subclasses. The [__call__] method should be overridden by subclasses
    to define the behavior of the target function.

    Attributes:
        parameter_space (ParameterSpace): The parameter space of the target function. Defaults to a
            single continuous parameter with bounds [-pi, pi].

    Methods:
        __call__(sample: np.ndarray): Evaluates the target function at the given sample points.
s
    """

    parameter_space : ParameterSpace = ParameterSpace([ContinuousParameter("x", -np.pi, np.pi)])

    def __call__(self, sample: np.ndarray):
        """
        Evaluates the target function at the given sample points.

        Args:
            sample (np.ndarray): Input samples at which to evaluate the function.

        Returns:
            float: The value of the target function at the given sample points.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def generate_samples(self, nsamples):
        """generates samples according to the bounds of the function"""
        bounds = self.parameter_space.get_bounds()
        coords = []
        for b in bounds:
            coords.append(np.random.uniform(b[0], b[1], size=(nsamples,1)))
        return np.hstack(coords)
