"""
Alanine dipeptide free energy surface
"""
from typing import List
import numpy as np
import pathlib
from .interpolated_target_function import (
    Interpolated2DTargetFunction,
    Interpolated1DTargetFunction)



class AlanineDipeptide2D(Interpolated2DTargetFunction):
    """
    Alanine dipeptide free energy surface
    """
    def __init__(self, noise_level: float = 0.0, path=None):
        data = self.get_data(path=path)
        grid = self.get_grid()
        super().__init__(data, grid[0], grid[1], noise_level=noise_level)

    def get_data(self, derivatives=False, path=None):
        """
        Load free energy landscape from the alanine dipeptide
        
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
            path = root + "/data/ad_fes_total.dat"

        with open(path, "r", encoding="utf-8") as metafile:
            data = np.genfromtxt(metafile)
        
        fes = data[:, 2]
        phi=data[:,0].reshape(101,101)
        psi=data[:,1].reshape(101,101)
        dx = data[:,3].reshape(101,101)
        dy = data[:,4].reshape(101,101)
        if derivatives:
            return phi, psi, dx, dy
        return (fes - np.min(fes)).reshape(101,101)

    def get_grid(self, path=None):
        phi_metadynamics, psi_metadynamics, _, _ = self.get_data(derivatives=True, path=path)
        x_grid = np.unique(phi_metadynamics)
        y_grid = np.unique(psi_metadynamics)
        return np.meshgrid(x_grid, y_grid, indexing='ij')



class AlanineDipeptide1D(Interpolated1DTargetFunction):
    """
    Alanine dipeptide free energy surface
    """
    def __init__(self,
                 noise_level: float = 0.0, path=None,
                 index=None, bounds: List | None = [-np.pi, np.pi]):
        y_data, x_data = self.get_data(path, index)
        super().__init__(y_data, x_data, 
                         noise_level=noise_level, 
                         bounds=bounds)

    def get_data(self, path=None, index=None):
        """
        Load free energy landscape from file.
        
        Parameters
        ----------
        derivatives : bool
            If True, return gradients instead of free energy.
        index: int
            Index of the column to be loaded
            
        Returns
        -------
        FES surface or derivatives as arrays.
        """
        if path is None:
            root = str(pathlib.Path(__file__).parent.resolve())
            path = root + "/data/ad_fes_total.dat"

        if index is None:
            index = 50

        with open(path, "r", encoding="utf-8") as metafile:
            data = np.genfromtxt(metafile)
        
        fes = data[:,2]
        phi = data[:,0].reshape(101,101)
        fes = (fes - np.min(fes)).reshape(101,101)
        return fes[:,index].reshape(-1), phi[index,:].reshape(-1)

