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

class SumRBFWhiteGPy(IStandardKernel):
    """
    Wrapper for a sum of GPy RBF and White kernels to use with EmuKit quadrature.

    Parameters
    ----------
    gpy_kernel : GPy kernel
        A kernel composed of an RBF and White component.
    """
    def __init__(self, gpy_kernel):
        
        gpy_rbf = gpy_kernel.parts[0]
        gpy_white = gpy_kernel.parts[1]
        self.gpy_rbf = gpy_rbf
        self.gpy_white = gpy_white
        self.gpy_kernel = gpy_rbf + gpy_white 

    @property
    def lengthscales(self) -> np.ndarray:
       """Return array of lengthscales (supports ARD)."""

       if self.gpy_rbf.ARD:
           return self.gpy_rbf.lengthscale.values
       return np.full((self.gpy_rbf.input_dim,), self.gpy_rbf.lengthscale[0])

    @property
    def variance(self) -> float:
        """Return variance of  kernel"""
        return self.gpy_rbf.variance.values[0] + self.gpy_white.variance.values[0]


    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Computes the full kernel matrix (RBF + White)."""
        return self.gpy_kernel.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute derivative of kernel with respect to x1."""
        scaled_vector_diff = np.swapaxes((x1[None, :, :] - x2[:, None, :]) / self.lengthscales**2, 0, -1)
        return -self.K(x1, x2)[None, ...] * scaled_vector_diff

    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """Derivative of diagonal kernel is zero."""
        return np.zeros((x.shape[1], x.shape[0]))

class SumMatern52WhiteGPy(IStandardKernel):
    """
    Wrapper for a sum of GPy Matern52 and White kernels to use with EmuKit quadrature.

    Parameters
    ----------
    gpy_kernel : GPy kernel
        A kernel composed of a Matern52 and White component.
    """

    def __init__(self, gpy_kernel):
        
        gpy_matern = gpy_kernel.parts[0]
        gpy_white = gpy_kernel.parts[1]
        self.gpy_matern = gpy_matern
        self.gpy_white = gpy_white
        self.gpy_kernel = gpy_matern + gpy_white 

    @property
    def lengthscales(self) -> np.ndarray:
       """Return array of lengthscales (supports ARD)."""
       if self.gpy_matern.ARD:
           return self.gpy_matern.lengthscale.values
       return np.full((self.gpy_matern.input_dim,), self.gpy_matern.lengthscale[0])

    @property
    def variance(self) -> float:
        """Returns the variance of the total"""
        return self.gpy_matern.variance.values[0] + self.gpy_white.variance.values[0]


    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Computes the full kernel matrix (RBF + White)."""
        return self.gpy_kernel.K(x1, x2)

    def dK_dx1(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute derivative of kernel with respect to x1."""
        scaled_vector_diff = np.swapaxes((x1[None, :, :] - x2[:, None, :]) / self.lengthscales**2, 0, -1)
        return -self.K(x1, x2)[None, ...] * scaled_vector_diff


    def dKdiag_dx(self, x: np.ndarray) -> np.ndarray:
        """Derivative of diagonal kernel is zero."""
        return np.zeros((x.shape[1], x.shape[0]))