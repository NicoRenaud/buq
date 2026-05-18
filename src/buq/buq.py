from ast import Tuple
import numpy as np
from typing import Callable
from scipy.optimize import minimize, OptimizeResult, Bounds

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy
from emukit.quadrature.methods import VanillaBayesianQuadrature
import emukit.quadrature.acquisitions as emu_acqui
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace

from buq.data.md.integration import integration_2d_rgrid
from buq.data.target_function import BaseTargetFunction

import matplotlib.pyplot as plt

from .kernels.base_kernel import BaseKernel

class ClassicalBayesianQuadratureMD:

    """
    Classical Bayesian Quadrature for Multi-Dimensional Target Functions.

    This class implements the classical Bayesian quadrature method for multi-dimensional
    target functions. It uses a surrogate model to approximate the target function and
    optimizes the acquisition function to select the next evaluation points.

    Parameters
    ----------
    surrogate_model : BaseGaussianProcessGPy
        The surrogate model used for approximation.
    target_function : BaseTargetFunction
        The target function to optimize.
    acquisition_function : str, optional
        The name of the acquisition function to use. Supported values are "IVR" and "US".
    acquisition_function_kwargs : dict, optional
        Additional keyword arguments for the acquisition function.
    num_init_samples : int, optional
        The number of initial samples to use for the surrogate model. Defaults to 2.
    npts_plot : int, optional
        The number of points to sample for the plotting. Defaults to 10.
    """

    def __init__(
        self,
        kernel: BaseKernel,
        target_function: BaseTargetFunction,
        acquisition_function: str = "IVR",
        acquisition_function_kwargs: dict = {},
        num_init_samples: int = 2,
        npts_plot: int = 10,
    ):
        # Initialize the class
        self.kernel = kernel
        self.target_function = target_function
        self.acquisition = acquisition_function
        self.acquisition_function_kwargs = acquisition_function_kwargs
        self.npts_plot = npts_plot
        self.num_init_samples = num_init_samples

        # Define parameter bounds and ground truth
        parameter_bounds = target_function.parameter_space.get_bounds()
        self.bounds = Bounds(
            [d[0] for d in parameter_bounds], [d[1] for d in parameter_bounds]
        )

        # define ground truth
        xplot = np.linspace(self.bounds.lb[0], self.bounds.ub[0], npts_plot)
        yplot = np.linspace(self.bounds.lb[1], self.bounds.ub[1], npts_plot)
        self.x_plot = np.array(np.meshgrid(xplot, yplot)).reshape(2, -1).T
        if self.target_function.has_ground_truth:
            energy, forces = self.target_function.get_ground_truth(self.x_plot)
            self.references_forces = forces.reshape(2, npts_plot, npts_plot).transpose(
                1, 2, 0
            )
            # self.y_plot = energy.reshape(npts_plot, npts_plot)
            self.y_plot = integration_2d_rgrid(
                self.references_forces[:, :, [1, 0]],
                xplot[1] - xplot[0],
                yplot[1] - yplot[0],
                npts_plot,
                minimization=True,
                max_iter=100,
            )
        else:
            print(
                "The target function does not have ground truth values. Setting y_plot to zeros."
            )
            self.y_plot = np.zeros((npts_plot, npts_plot))
            self.references_forces = np.zeros((npts_plot, npts_plot, 2))

        # Initialize the surrogate model and acquisition function
        self.x_data = self.target_function.generate_samples(self.num_init_samples)
        self.y_data = self.target_function(self.x_data)

        # initiate the surrogate model
        self.surrogate_model = self.kernel.get_kernel(
            self.x_data,
            self.y_data,
            bounds=self.target_function.parameter_space.get_bounds(),
        )

        self.emukit_method = VanillaBayesianQuadrature(
            base_gp=self.surrogate_model, X=self.x_data, Y=self.y_data
        )
        self.space = ParameterSpace(
            self.emukit_method.reasonable_box_bounds.convert_to_list_of_continuous_parameters()
        )
        self.optimizer = GradientAcquisitionOptimizer(self.space)

        # Set the acquisition function
        self.set_acquisition_functions(
            acquisition_function, **acquisition_function_kwargs
        )

    def set_acquisition_functions(self, acquisition: str, **kwargs):
        """
        Set the acquisition function to use.

        Parameters
        ----------
        acquisition : str
            The name of the acquisition function to use. Supported values are "IVR" and "US".

        Raises
        ------
        ValueError
            If the given acquisition function is not supported.

        """
        self.acquisition = acquisition
        if isinstance(self.acquisition, str):

            supported_acquisitions = [
                "IVR",
                "US",
                # "V3",
            ]

            if self.acquisition not in supported_acquisitions:
                raise ValueError(
                    f"Acquisition function {self.acquisition} not supported."
                )

            self.acq_func = {
                "IVR": emu_acqui.IntegralVarianceReduction,
                "US": emu_acqui.UncertaintySampling,
                # "V3": emu_acqui.VoronoiVerticesVariance,
            }[self.acquisition.upper()](self.emukit_method, **kwargs)

        elif issubclass(self.acquisition, BaseAcquisition):
            self.acq_func = self.acquisition(self.emukit_method, **kwargs)

        else:
            raise ValueError(
                "Acquisition function must be either a string or an instance of BaseAcquisition."
            )

    def evaluate_free_energy(
        self, minimization: bool = True, max_iter: int = 100
    ):
        """
        Compute the free energy surface using the surrogate model by integrating the value
        of the forces predicted by the surrogate model.

        Returns
        -------
        free_energy : np.ndarray
            The free energy surface evaluated on the grid defined by x_plot.
        """
        if self.x_plot is None:
            raise ValueError("x_plot is None. Free energy cannot be computed.")
        forces, _ = self.surrogate_model.predict(self.x_plot)
        print(forces)
        forces = forces.reshape(
            self.npts_plot, self.npts_plot, 2
        )
        dx = self.x_plot[1, 0] - self.x_plot[0, 0]
        dy = self.x_plot[self.npts_plot, 1] - self.x_plot[0, 1]
        energy = integration_2d_rgrid(
            forces[:, :, [1, 0]],
            dx,
            dy,
            self.npts_plot,
            minimization=minimization,
            max_iter=max_iter,
        )
        return energy, forces

    def find_optimal_location(
        self, acquisition: Callable, x_sample: np.ndarray
    ):
        """
        Find the optimal location to evaluate the acquisition function.

        Parameters
        ----------
        acquisition : Callable
            The acquisition function to use.
        x_sample : np.ndarray
            The points at which the surrogate model has been evaluated.

        Returns
        -------
        xy_new : np.ndarray
            The proposed next point to evaluate.
        acq_vals : np.ndarray
            The values of the acquisition function at each point in x_sample.
        """

        # Evaluate IVR on grid
        acq_vals = acquisition.evaluate(x_sample)
        acq_vals = acq_vals.reshape(self.x_plot.shape[0])

        # Determine next query point
        max_index = np.unravel_index(
            np.argmax(acq_vals), acq_vals.shape
        )
        new_x_ivr = self.x_plot[max_index, 0][0]
        new_y_ivr = self.x_plot[max_index, 1][0]

        xy_new = np.array([[new_x_ivr, new_y_ivr]])

        return xy_new, acq_vals

    def run(self, n_iter: int, plot_fit: bool = False) -> OptimizeResult:
        """
        Run the Bayesian optimization algorithm for the given number of steps.

        Parameters
        ----------
        n_iter : int
            The number of steps to run the algorithm.
        plot_fit : bool, optional
            Whether to plot the surrogate function at each step. Defaults to False.

        Returns
        -------
        result : OptimizeResult
            The result of the optimization algorithm.
        """

        rmse = []

        for i in range(n_iter):

            print(f" -------------- BaysOpt loop, query {i} ---------------")

            self.emukit_method.set_data(self.x_data, self.y_data)

            est_free_energy, est_forces = self.evaluate_free_energy(
                minimization=True, max_iter=100
            )

            x_next, acq_vals = self.find_optimal_location(self.acq_func, self.x_plot)
            y_next = self.target_function(x_next)

            # Update dataset
            print(self.x_data, x_next)
            self.x_data = np.vstack((self.x_data, x_next))
            self.y_data = np.vstack((self.y_data, y_next))

            # Plot surrogate and acquisition at each step
            if plot_fit:
                self.plot_surrogate(est_free_energy, est_forces, acq_vals, self.x_data)

            # integrate
            rmse_step = np.sqrt(np.mean((est_free_energy - self.y_plot) ** 2))
            rmse.append(rmse_step)

        result = OptimizeResult()
        result.estimated_forces = est_forces
        result.estimated_free_energy = est_free_energy
        result.reference_free_energy = self.y_plot
        result.reference_forces = self.references_forces
        result.bounds = self.bounds
        result.x_iters = self.x_data
        result.rmse = rmse
        return result

    def plot_surrogate(
        self,
        est_free_energy: np.ndarray,
        est_forces: np.ndarray,
        acq_vals: np.ndarray,
        x_sample: np.ndarray,
    ):
        """
        Plot the surrogate function at each step of the optimization algorithm.

        Parameters
        ----------
        step : int
            The current step of the optimization algorithm.
        x_sample : np.ndarray
            The current set of samples.

        Raises
        ------
        ValueError
            If x_plot is None, the surrogate function cannot be plotted.
        """
        if self.x_plot is None:
            raise ValueError("x_plot is None. Surrogate function cannot be plotted.")

        delta = est_free_energy - self.y_plot
        v = np.max(np.abs(delta))

        extent = [
            self.bounds.lb[0],
            self.bounds.ub[0],
            self.bounds.lb[1],
            self.bounds.ub[1],
        ]
        origin = "lower"

        plt.subplot(231)
        plt.imshow((est_forces[:, :, 0]), cmap="viridis", extent=extent, origin=origin)
        plt.scatter(x_sample[:-1, 0], x_sample[:-1, 1], color="black", marker="o", s=5)
        plt.title("Est Force X", fontsize=10)

        plt.subplot(232)
        plt.imshow(
            (self.references_forces[:, :, 0]),
            cmap="viridis",
            extent=extent,
            origin=origin,
        )
        plt.scatter(x_sample[:-1, 0], x_sample[:-1, 1], color="black", marker="o", s=5)
        plt.title("Force X", fontsize=10)

        delta_fx = est_forces[:, :, 0] - self.references_forces[:, :, 0]
        v_fx = np.max(np.abs(delta_fx))
        plt.subplot(233)
        plt.imshow(
            (delta_fx), cmap="RdBu", vmin=-v_fx, vmax=v_fx, extent=extent, origin=origin
        )
        plt.scatter(x_sample[:-1, 0], x_sample[:-1, 1], color="black", marker="o", s=5)
        plt.title("Error Force X", fontsize=10)

        plt.subplot(234)
        plt.imshow((est_forces[:, :, 1]), cmap="viridis", extent=extent, origin=origin)
        plt.scatter(x_sample[:-1, 0], x_sample[:-1, 1], color="black", marker="o", s=5)
        plt.title("Est Forces Y", fontsize=10)

        plt.subplot(235)
        plt.imshow(
            (self.references_forces[:, :, 1]),
            cmap="viridis",
            extent=extent,
            origin=origin,
        )
        plt.scatter(x_sample[:-1, 0], x_sample[:-1, 1], color="black", marker="o", s=5)
        plt.title("Force Y", fontsize=10)

        delta_fx = est_forces[:, :, 1] - self.references_forces[:, :, 1]
        v_fx = np.max(np.abs(delta_fx))
        plt.subplot(236)
        plt.imshow(
            (delta_fx), cmap="RdBu", vmin=-v_fx, vmax=v_fx, extent=extent, origin=origin
        )
        plt.scatter(x_sample[:-1, 0], x_sample[:-1, 1], color="black", marker="o", s=5)
        plt.title("Error Force Y", fontsize=10)

        plt.show()

        plt.subplot(132)
        plt.imshow(self.y_plot, cmap="viridis", extent=extent, origin=origin)
        plt.title("Ground truth", fontsize=10)

        plt.subplot(131)
        plt.imshow(est_free_energy, cmap="viridis", extent=extent, origin=origin)
        plt.scatter(x_sample[:-1, 0], x_sample[:-1, 1], color="black", marker="o", s=5)
        plt.scatter(x_sample[-1, 0], x_sample[-1, 1], color="red", marker="x", s=5)
        plt.title("Prediction", fontsize=10)

        plt.subplot(133)
        plt.imshow(delta, cmap="RdBu", vmin=-v, vmax=v, extent=extent, origin=origin)
        plt.scatter(x_sample[:-1, 0], x_sample[:-1, 1], color="black", marker="o", s=5)
        plt.title("Error", fontsize=10)
        plt.show()

        if self.acq_func.plot_acquisition_map:

            plt.subplot(121)
            plt.imshow(
                (acq_vals[:, :, 0]), cmap="viridis", extent=extent, origin=origin
            )
            plt.scatter(x_sample[-1, 0], x_sample[-1, 1], color="red", marker="x", s=20)
            plt.title("Acquisition X", fontsize=10)

            plt.subplot(122)
            plt.imshow(
                (acq_vals[:, :, 1]), cmap="viridis", extent=extent, origin=origin
            )
            plt.scatter(x_sample[-1, 0], x_sample[-1, 1], color="red", marker="x", s=20)
            plt.scatter(x_sample[-1, 0], x_sample[-1, 1], color="red", marker="x", s=20)
            plt.title("Acquisition Y", fontsize=10)

            plt.show()
