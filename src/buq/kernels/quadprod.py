import numpy as np
from typing import Union
import sys
from emukit.quadrature.kernels import QuadratureProductMatern52,LebesgueEmbedding
from emukit.quadrature.measures import LebesgueMeasure
from emukit.quadrature.interfaces import  IProductMatern52, IStandardKernel
from scipy import optimize as scipy_optimize
import matplotlib.pyplot as plt
import subprocess
import os

class QuadratureProductMatern52LebesgueMeasure(QuadratureProductMatern52, LebesgueEmbedding):
    """Product Matern52 kernel integrated over a standard Lebesgue measure.
    
    Combines the EmuKit QuadratureProductMatern52 kernel with LebesgueEmbedding
    to allow integration of functions with respect to the Lebesgue measure. Fixes
    an EmuKit bug related to handling multidimensional lengthscales.


    .. seealso::
       * :class:`emukit.quadrature.interfaces.IProductMatern52`
       * :class:`emukit.quadrature.kernels.QuadratureProductMatern52`
       * :class:`emukit.quadrature.measures.LebesgueMeasure`

    :param matern_kernel: The standard EmuKit product Matern52 kernel.
    :param measure: The Lebesgue measure.
    :param variable_names: The (variable) name(s) of the integral.

    """

    def __init__(self, matern_kernel: IProductMatern52, measure: LebesgueMeasure, variable_names: str = "") -> None:
        super().__init__(matern_kernel=matern_kernel, measure=measure, variable_names=variable_names)

    def _scale(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Scale input z by kernel variance."""
        return self.variance * z

    def _get_univariate_parameters(self, dim: int) -> dict:
        lengthscales = self.lengthscales
        # Handle isotropic vs anisotropic lengthscales
        if np.ndim(lengthscales) == 0 or lengthscales.size == 1:
            ls = float(lengthscales)
        else:
            ls = float(lengthscales[dim])
        return {
            "domain": self.measure.domain.bounds[dim], #this was the line i had to change
            "lengthscale": ls,
            "normalize": self.measure.is_normalized,
        }
    def _qK_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        """
        Analytical integration of the 1D Matern52 kernel against the Lebesgue measure.
        Returns an array of integrated values at x.
        """
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        s5 = np.sqrt(5)
        first_term = 16 * lengthscale / (3 * s5)
        second_term = (
            -np.exp(s5 * (x - b) / lengthscale)
            / (15 * lengthscale)
            * (8 * s5 * lengthscale**2 + 25 * lengthscale * (b - x) + 5 * s5 * (b - x) ** 2)
        )
        third_term = (
            -np.exp(s5 * (a - x) / lengthscale)
            / (15 * lengthscale)
            * (8 * s5 * lengthscale**2 + 25 * lengthscale * (x - a) + 5 * s5 * (a - x) ** 2)
        )
        return (first_term + second_term + third_term) * normalization

    def _qKq_1d(self, **parameters) -> float:
        """
        Analytical integration of the 1D kernel against itself over the domain.
        Returns a scalar.
        """
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        c = np.sqrt(5) * (b - a)
        bracket_term = 5 * a**2 - 10 * a * b + 5 * b**2 + 7 * c * lengthscale + 15 * lengthscale**2
        qKq = (2 * lengthscale * (8 * c - 15 * lengthscale) + 2 * np.exp(-c / lengthscale) * bracket_term) / 15
        return float(qKq) * normalization**2

    def _dqK_dx_1d(self, x: np.ndarray, **parameters) -> np.ndarray:
        """
        Analytical derivative of the 1D kernel integral with respect to x.
        Returns an array of derivatives.
        """
        a, b = parameters["domain"]
        lengthscale = parameters["lengthscale"]
        normalization = 1 / (b - a) if parameters["normalize"] else 1.0
        s5 = np.sqrt(5)
        first_exp = -np.exp(s5 * (x - b) / lengthscale) / (15 * lengthscale)
        first_term = first_exp * (15 * lengthscale - 15 * s5 * (x - b) + 25 / lengthscale * (x - b) ** 2)
        second_exp = -np.exp(s5 * (a - x) / lengthscale) / (15 * lengthscale)
        second_term = second_exp * (-15 * lengthscale + 15 * s5 * (a - x) - 25 / lengthscale * (a - x) ** 2)
        return (first_term + second_term) * normalization