from typing import List
import numpy as np 
from emukit.core import ContinuousParameter, ParameterSpace
from scipy.interpolate import interp1d
import pathlib 
from .utils import interpolation_function_1d as interpolation_function
from .interpolated_target_function import Interpolated1DTargetFunction


class VECadherin(Interpolated1DTargetFunction):
    def __init__(self, 
                 noise_level: float = 0.0, path=None,
                 bounds: List | None = [-np.pi, np.pi]):
        x_data, y_data = self._get_data(path)
        super().__init__(y_data, x_data, noise_level, bounds)


    def _get_data(self, path=None):
        if path is None:
            root = str(pathlib.Path(__file__).parent.resolve())
            path = root + "/data/VECadherin_10PS_UI.dat"
        metafile = open(path)
        data = np.genfromtxt(metafile)
        metafile.close()
        x_data = data[:, 0]
        y_data = data[:, 3]
        return x_data, y_data
    
class VECadherin_without_plastic(Interpolated1DTargetFunction):
    def __init__(self, 
                 noise_level: float = 0.0, path=None,
                 bounds: List | None = [-np.pi, np.pi]):
        x_data, y_data = self._get_data(path)
        super().__init__(y_data, x_data, noise_level, bounds)

    def _get_data(self, path=None):
        if path is None:
            root = str(pathlib.Path(__file__).parent.resolve())
            path = root + "/data/VECadherin_noplastic_UI.dat"
        metafile = open(path)
        data = np.genfromtxt(metafile)
        metafile.close()
        x_data = data[:, 0]
        y_data = data[:, 3]
        return x_data, y_data