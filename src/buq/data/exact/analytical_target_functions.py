"""
Analytical Target Functions
"""
import numpy as np
from typing import List
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.test_functions.quadrature.hennig1D import _hennig1D
from emukit.test_functions.forrester import forrester_function


class Forrester:
    """
    Initialize Forrester object.

    Parameters
    ----------
    noise_level : float
        The standard deviation of the Gaussian noise.
    """

    def __init__(self, noise_level: float = 0.0):
        """
        Initialize Forrester object.

        Parameters
        ----------
        noise_level : float
            The standard deviation of the Gaussian noise.
        """
        self.noise_level = noise_level
        self.fn, self.parameter_space = forrester_function(
            noise_standard_deviation=self.noise_level
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fn(x)


class Hennig1D:
    """benchmark class.
    """

    def __init__(self, noise_level: float = 0.0):
        """Init the class

        Args:
            noise_level (float, optional): noise level. Defaults to 0.0.
        """
        self.noise_level = noise_level
        self.parameter_space = ParameterSpace([ContinuousParameter("x", -3, 3)])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        return _hennig1D(x) + np.random.randn(*x.shape) * self.noise_level


class SinTanH:
    """Benchmark function."""
    def __init__(self, noise_level: float = 0.01):
        """
        Initialize SinTanH object.

        Parameters
        ----------
        noise_level : float
            The standard deviation of the Gaussian noise.

        """
        self.noise_level = noise_level
        self.parameter_space = ParameterSpace([ContinuousParameter("x", -2, 2)])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (
            np.sin(5 * x) * (1 - np.tanh(x**2))
            + np.random.randn(*x.shape) * self.noise_level
        )


class GaussianSin:
    """Benchmark function."""
    def __init__(
        self,
        x0: float = 0.0,
        std: float = 1.0,
        omega: float = 1.0,
        noise_level: float = 0.01,
        bounds: List | None = [-np.pi, np.pi],
    ):
        """
        A Gaussian-sinusoidal function, which is a product of a Gaussian
        distribution and a sinusoidal function.

        Parameters
        ----------
        x0 : float, optional
            Mean of the Gaussian distribution. Default is 0.0.
        std : float, optional
            Standard deviation of the Gaussian distribution. Default is 1.0.
        omega : float, optional
            Frequency of the sinusoidal function. Default is 1.0.
        noise_level : float, optional
            Standard deviation of Gaussian noise to add to the function output.
            Default is 0.01.
        """

        self.x0 = x0
        self.std = std
        self.omega = omega
        self.noise_level = noise_level
        self.bounds = bounds
        self.parameter_space = ParameterSpace(
            [ContinuousParameter("x", bounds[0], bounds[1])]
        )
        self.omega *= np.pi / bounds[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Gaussian-sinusoidal function at given points.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the function.

        Returns
        -------
        y : ndarray
            Function values at given points.
        """
        return (
            np.exp(-((x - self.x0) ** 2) / self.std) * np.sin(self.omega * x)
            + np.random.randn(*x.shape) * self.noise_level
        )
