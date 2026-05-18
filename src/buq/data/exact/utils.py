from typing import Callable
import numpy as np 
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

def interpolation_function_1d(x_data, y_data, noise_level=0.0):
    """
    Construct an interpolation-based callable function from discrete data.

    Parameters
    ----------
    x_data : array_like
        1D array of input coordinates.
    y_data : array_like
        1D array of corresponding function values.
    noise_level : float, optional
        Standard deviation of Gaussian noise to add to interpolated outputs.
        Default is 0.0 (no noise).

    Returns
    -------
    callable
        Function f(x) that interpolates the input data using cubic interpolation.
        If noise_level > 0, the returned values are perturbed by additive Gaussian noise.

    Notes
    -----
    - Uses cubic interpolation via scipy.interpolate.interp1d with extrapolation enabled.
    - The returned function accepts scalar or array-like inputs.
    """
    interpolator = interp1d(x_data, y_data, kind="cubic", fill_value="extrapolate")
    def interpolated_function(x):
        x = np.atleast_1d(x).reshape(-1)
        y = interpolator(x)
        if noise_level > 0:
            y += np.random.randn(*y.shape) * noise_level
        return y
    return interpolated_function

def interpolation_function_2d(x_data: np.ndarray, y_data: np.ndarray, 
                              z_data: np.ndarray | None = None, 
                              z_func: Callable | None = None, 
                              noise_level: float = 0.0):
    """
    Construct an interpolation-based callable function from discrete data.

    Parameters
    ----------
    x_data : array_like
        1D array of input coordinates in x direction.
    y_data : array_like
        1D array of input coordinates in y direction.
    z_data : array_like
        1D array of corresponding function values.
    z_func : Callable
        Function to compute z_data from x_data and y_data.
    noise_level : float, optional
        Standard deviation of Gaussian noise to add to interpolated outputs.
        Default is 0.0 (no noise).

    Returns
    -------
    callable
        Function f(x) that interpolates the input data using cubic interpolation.
        If noise_level > 0, the returned values are perturbed by additive Gaussian noise.

    Raises
    ------
    ValueError
        If both z_data and z_func are provided.
        If neither z_data nor z_func are provided.
        
    Notes
    -----
    - Uses cubic interpolation via scipy.interpolate.interp1d with extrapolation enabled.
    - The returned function accepts scalar or array-like inputs.
    """
    
    if z_data is not None and z_func is not None:
        raise ValueError("Cannot provide both z_data and z_func")
    if z_data is None and z_func is None:
        raise ValueError("Must provide either z_data or z_func")
    if z_data is None:
        xg, yg = np.meshgrid(x_data, y_data, indexing='ij', sparse=True)
        z_data = z_func(xg, yg)

    interpolator = RegularGridInterpolator((x_data, y_data), z_data)
    def interpolated_function(xy_pts: np.ndarray):
        xy_pts = np.atleast_2d(xy_pts).reshape(-1, 2)
        z = interpolator(xy_pts)
        if noise_level > 0:
            z += np.random.randn(*z.shape) * noise_level
        return z
    return interpolated_function

def normalize_1D(x: np.ndarray, bounds: list) -> np.ndarray:
    x -= min(x)
    x /= max(x)
    x *= (bounds[1] - bounds[0])
    x += bounds[0]
    return x