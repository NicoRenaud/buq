"""Integration of forces on a 2D grid."""

import numpy as np
from scipy import optimize as scipy_optimize


def integration_2d_rgrid(forces, dx, dy, npts, minimization=True, max_iter=100, verbose=False):
    """Integrate forces on a regular 2D grid.

    Args:
        forces (np.ndarray): Force grid with shape (npts, npts, 2).
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.
        npts (int): Number of grid points in each direction.
        minimization (bool): Whether to apply gradient-based minimization.
        max_iter (int): Maximum number of optimizer iterations.
        verbose (bool): Whether to print callback messages.

    Returns:
        np.ndarray: Integrated energy grid.
    """

    energy = np.zeros((npts, npts))
    energy = simpson(energy, forces, npts, dx, dy)
    if minimization:
        return optimize(energy, forces, dx, dy, npts, max_iter=max_iter, verbose=verbose)
    return energy - np.min(energy)


def simpson_cusum(val_grid, dval_grid, n, dx, dy):
    """Compute Simpson-style cumulative sums for a 2D grid.

    This helper updates the provided grid in place by integrating the
    partial derivatives stored in ``dval_grid`` using a Simpson-like
    cumulative sum approach along both axes.

    Args:
        val_grid (np.ndarray): Target grid of shape (n, n) to populate.
        dval_grid (np.ndarray): Grid of partial derivatives with shape
            (n, n, 2), where the last dimension contains x- and y-derivatives.
        n (int): Number of grid points in each direction.
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.

    Returns:
        None: The result is written in place to ``val_grid``.
    """
    assert val_grid.shape == (n, n)
    assert dval_grid.shape == (n, n, 2)

    val_grid[0, 0] = 0  # corner point to zero
    val_grid[0, 1:] = np.cumsum((dval_grid[0, :-1, 0] + dval_grid[0, 1:, 0]) * dx / 2)
    val_grid[1:, 0] = np.cumsum((dval_grid[:-1, 0, 1] + dval_grid[1:, 0, 1]) * dx / 2)
    val_grid[1:, 1:] = 0.5 * (
        val_grid[1:, 0]
        + np.cumsum((dval_grid[1:, :-1, 0] + dval_grid[1:, 1:, 0]) * dx / 2, axis=1)
        + val_grid[0, 1:]
        + np.cumsum((dval_grid[:-1, 1:, 1] + dval_grid[1:, 1:, 1]) * dy / 2, axis=0)
    )


def simpson(val_grid, dval_grid, n, dx, dy):
    """Integrate a 2D grid of derivatives using Simpson-like averaging.

    The integration is performed on a regular grid by iterating through
    each point and combining neighboring derivative values to compute the
    integrated value at each cell.

    Args:
        val_grid (np.ndarray): Target grid of shape (n, n) to populate.
        dval_grid (np.ndarray): Grid of partial derivatives with shape
            (n, n, 2), where the last dimension contains x- and y-derivatives.
        n (int): Number of grid points in each direction.
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.

    Returns:
        np.ndarray: Integrated grid ``val_grid`` with the same shape as input.
    """

    assert val_grid.shape == (n, n)
    assert dval_grid.shape == (n, n, 2)

    for j in range(n):
        for i in range(n):
            if i == 0 and j == 0:
                val_grid[j, i] = 0  # corner point to zero
            elif i == 0:
                val_grid[j, i] = (
                    val_grid[j - 1, i]
                    + (dval_grid[j - 1, i, 1] + dval_grid[j, i, 1]) * dy / 2
                )
            elif j == 0:
                val_grid[j, i] = (
                    val_grid[j, i - 1]
                    + (dval_grid[j, i - 1, 0] + dval_grid[j, i, 0]) * dx / 2
                )
            else:
                val_grid[j, i] = (
                    val_grid[j - 1, i - 1]
                    + (
                        dval_grid[j - 1, i - 1, 0]
                        + dval_grid[j - 1, i, 0]
                        + dval_grid[j, i - 1, 0]
                        + dval_grid[j, i, 0]
                    )
                    * dx
                    / 4
                    + (
                        dval_grid[j - 1, i - 1, 1]
                        + dval_grid[j - 1, i, 1]
                        + dval_grid[j, i - 1, 1]
                        + dval_grid[j, i, 1]
                    )
                    * dy
                    / 4
                )

    return val_grid


def optimize(val_grid, dval_grid, dx, dy, npts, max_iter=100, verbose=False):
    """Optimize an integrated energy grid to match a derivative field.

    The function performs L-BFGS-B optimization on the integrated energy grid
    ``val_grid`` so that its numerical gradient best matches the target
    derivative field ``dval_grid``. The optimization objective minimizes the
    mean squared error between the gradient of the candidate energy grid and
    the provided partial derivatives.

    Args:
        val_grid (np.ndarray): Initial energy grid guess with shape (npts, npts).
        dval_grid (np.ndarray): Target derivative field with shape (npts, npts, 2).
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.
        npts (int): Number of grid points in each direction.
        max_iter (int): Maximum number of optimizer iterations.
        verbose (bool): If True, print loss values during optimization.

    Returns:
        np.ndarray: Optimized energy grid shifted so that its minimum value is zero.
    """

    def objective(func_vals):
        func_vals = func_vals.reshape(npts, npts)
        func_val_dy, func_vals_dx = np.gradient(func_vals, dy, dx)
        func_vals_der = np.stack((func_vals_dx, func_val_dy), axis=-1)
        return np.sum((dval_grid - func_vals_der) ** 2) / (npts * npts)

    def callback_func(func_vals):
        if verbose:
            print(f"Current loss: {objective(func_vals):.6f}")

    result = scipy_optimize.minimize(
        objective,
        val_grid.ravel(),
        method="L-BFGS-B",
        options={
            "maxfun": np.inf,
            "maxiter": max_iter,
            "maxls": 50,
            "iprint": 10,
        },
        callback=callback_func,
    )

    if not result.success:
        print("WARNING: Minimization could not converge")

    val_grid = result.x.reshape(npts, npts)
    if verbose:
        print(f"\n# Force Integration error: {objective(val_grid.ravel()):.2f}\n")
    return val_grid - np.min(val_grid)
