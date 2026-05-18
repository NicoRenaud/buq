from typing import List
import numpy as np
import pathlib
from .interpolated_target_function import Interpolated1DTargetFunction


class Nucleation(Interpolated1DTargetFunction):
    """Nucleation FES for 1D collective variable"""
    def __init__(self, 
                 noise_level: float = 0.0, path=None,
                 bounds: List | None = [-np.pi, np.pi]):
        x, data = self._get_data(path)
        super().__init__(data, x, noise_level=noise_level, bounds=bounds)


    def _get_data(self, path=None):

        if path is None:
            root = str(pathlib.Path(__file__).parent.resolve())
            path = root + "/data/nucleation_fes_analytical_1d.dat"
        metafile = open(path, "r", encoding="utf-8")
        data = np.genfromtxt(metafile)
        metafile.close()
        x_data = data[:, 0]
        y_data = data[:, 1]
        return x_data, y_data