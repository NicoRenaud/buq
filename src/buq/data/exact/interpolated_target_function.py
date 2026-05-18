from typing import Callable
import numpy as np
from .utils import interpolation_function_1d, interpolation_function_2d
from emukit.core import ContinuousParameter, ParameterSpace
from ..target_function import BaseTargetFunction
from .utils import normalize_1D


class Interpolated1DTargetFunction(BaseTargetFunction):

    def __init__(self, data, x, noise_level=0.0, bounds=[-np.pi, np.pi]):
        self.data = data
        self.x = np.sort(np.unique(x))  # sort and unique x
        self.bounds = bounds
        self._normalize_data()
        self.dimx = x.shape[0]
        self.interpolator = interpolation_function_1d(
            self.x, self.data, noise_level=noise_level
        )
        self.parameter_space = ParameterSpace(
            [ContinuousParameter("x", float(self.x.min()), float(self.x.max()))]
        )

    def _normalize_data(self):

        if self.bounds is not None:
            self.x = normalize_1D(self.x, self.bounds)
            self.data = normalize_1D(self.data, [0, 1])

    def _check_bounds(self, x):
        if x > self.x[-1] or x < self.x[0]:
            raise ValueError(f"x {x} out of bounds")

    def __call__(self, samples: np.ndarray):
        for x in samples:
            self._check_bounds(x)
        return self.interpolator(samples).reshape(-1, 1)


class Interpolated2DTargetFunction(BaseTargetFunction):
    """
    Interpolated 2D target function
    """
    def __init__(
        self, data: np.ndarray, x: np.ndarray, y: np.ndarray, noise_level: float = 0.0
    ):
        """
        Initialize the TargetFunction class.
        This function allows to pick the closest point to the given (x,y) coordinate give
        It does not interpolate the data

        Parameters
        ----------
        data : np.ndarray of shape (dimx, dimy)
            The data for the target function.
        x : np.ndarray of shape (dimx,)
            The x values of the data.
        y : np.ndarray of shape (dimy,)
            The y values of the data.

        Attributes
        ----------
        data : np.ndarray of shape (dimx, dimy)
            The data for the target function.
        x : np.ndarray of shape (dimx,)
            The sorted and unique x values of the data.
        y : np.ndarray of shape (dimy,)
            The sorted and unique y values of the data.
        dimx : int
            The number of x values.
        dimy : int
            The number of y values.
        """
        self.data = data
        self.x = np.sort(np.unique(x))
        self.y = np.sort(np.unique(y))
        self.grid = np.meshgrid(self.x, self.y)
        self.dimx = x.shape[0]
        self.dimy = y.shape[0]
        self.interpolator = interpolation_function_2d(
            self.x, self.y, self.data, noise_level=noise_level
        )
        self.parameter_space = ParameterSpace(
            [
                ContinuousParameter("x", self.x[0], self.x[-1]),
                ContinuousParameter("y", self.y[0], self.y[-1]),
            ]
        )

    def _check_bounds(self, x, y):
        """
        Check if x and y are within the bounds of the data.

        Parameters
        ----------
        x : float
            The x value to check.
        y : float
            The y value to check.

        Raises
        ------
        ValueError
            If x or y is out of bounds.
        """
        if x > self.x[-1] or x < self.x[0]:
            raise ValueError("x out of bounds")
        if y > self.y[-1] or y < self.y[0]:
            raise ValueError("y out of bounds")

    def __call__(self, samples: np.ndarray):
        """
        Evaluate the target function at the given point (x,y).

        Parameters
        ----------
        x : float
            The x value at which to evaluate the target function.
        y : float
            The y value at which to evaluate the target function.

        Returns
        -------
        float
            The value of the target function at (x,y).

        Raises
        ------
        ValueError
            If x or y is out of bounds.
        """
        return self.interpolator(samples).reshape(-1, 1)
