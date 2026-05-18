"""
Chemcial reaction free energy surface
"""

import numpy as np
import pathlib

from buq.data.md.integration import integration_2d_rgrid

from .interpolated_target_function import Interpolated2DTargetFunction


class ChemicalReaction(Interpolated2DTargetFunction):
    """
    Chemcial reaction free energy surface
    """

    def __init__(self,
                 noise_level: float = 0.0,
                 path=None,
                 minimize_integration = False,
                 num_iter_minimzation = 100):

        self.minimize_integration = minimize_integration
        self.num_iter_minimzation = num_iter_minimzation
        data = self._get_data(path=path)
        grid = self._get_grid()
        super().__init__(data, grid[0], grid[1], noise_level=noise_level)


    def _get_data(self, derivatives=False, path=None):
        """
        Load free energy landscape from the alanine Chemcial reaction

        Parameters
        ----------
        derivatives : bool
            If True, return gradients instead of free energy.

        Returns
        -------
        FES surface or derivatives as arrays.
        """
        if path is None:
            root = str(pathlib.Path(__file__).parent.resolve())
            path = root + "/data/chemical_reaction_fes.dat"

        with open(path, "r", encoding="utf-8") as metafile:
            data = np.genfromtxt(metafile)

        npts = 10
        xpts = data[:, 1]
        ypts = data[:, 2]
        dx = xpts[1] - xpts[0]
        dy = ypts[npts] - ypts[0]

        force_x = data[:, 3].reshape(npts, npts)
        force_y = data[:, 4].reshape(npts, npts)
        forces = np.stack((force_x, force_y), axis=-1)

        fes = integration_2d_rgrid(
            forces, dx, dy, npts, minimization=self.minimize_integration, max_iter=self.num_iter_minimzation
        )

        xpts = xpts.reshape(npts, npts)
        ypts = ypts.reshape(npts, npts)
        dforce_x = data[:, 3].reshape(npts, npts)
        dforce_y = data[:, 4].reshape(npts, npts)

        if derivatives:
            return xpts, ypts, dforce_x, dforce_y
        return (fes - np.min(fes)).reshape(npts, npts)

    def _get_grid(self, path=None) -> np.ndarray:
        """get the grid information

        Args:
            path (str, optional): the data path. Defaults to None.

        Returns:
            np.array: the grid 
        """
        xpts, ypts, _, _ = self._get_data(derivatives=True, path=path)
        x_grid = np.unique(xpts)
        y_grid = np.unique(ypts)
        return np.meshgrid(x_grid, y_grid, indexing="ij")
