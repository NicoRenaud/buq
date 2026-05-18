import GPy
# from emukit.model_wrappers.gpy_quadrature_wrappers import (
#     QuadratureRBFLebesgueMeasure,
#     QuadratureProductMatern52LebesgueMeasure,
# )
from emukit.quadrature.measures import LebesgueMeasure
from .sum import SumMatern52WhiteGPy
from .quadprod import QuadratureProductMatern52LebesgueMeasure
from .base_kernel import BaseKernel


class MaternKernel(BaseKernel):
    """
    Wrapper for a GPy Matern52 and White kernel to use with EmuKit quadrature.

    Parameters
    ----------
    lengthscale : float
        The lengthscale parameter of the Matern52 kernel.
    noise : float
        The variance of the white kernel.
    """
    def __init__(self, lengthscale, noise):
        """
        Initialize a Matern kernel with the given lengthscale and noise.

        Parameters
        ----------
        lengthscale : float
            The lengthscale parameter of the Matern52 kernel.
        noise : float
            The variance of the white kernel.
        """
        self.lengthscale = lengthscale
        self.noise = noise
    
    def get_kernel(self, x_data, y_data, bounds):
        """
        Create an EmuKit kernel wrapper for the GPy Matern52 and White kernel.

        Parameters
        ----------
        x_data : np.ndarray
            The input data.
        y_data : np.ndarray
            The target data.
        bounds : tuple
            The bounds of the integration domain.

        Returns
        -------
        emukit_qrbf : QuadratureProductMatern52LebesgueMeasure
            The EmuKit kernel wrapper for the Matern52 and White kernel.
        """
        kernel1 = GPy.kern.Matern52(2, lengthscale=self.lengthscale, variance=1, ARD=True)
        kernel2 = GPy.kern.src.static.White(2,variance = self.noise)
        kernel = kernel1 + kernel2
        gpy_model = GPy.models.GPRegression(X=x_data, Y=y_data, kernel=kernel)
        emukit_kernel = SumMatern52WhiteGPy(gpy_model.kern)
        emukit_measure = LebesgueMeasure.from_bounds(bounds=bounds)
        emukit_qrbf = QuadratureProductMatern52LebesgueMeasure(emukit_kernel, emukit_measure)
        return emukit_qrbf
