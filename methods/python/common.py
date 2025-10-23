"""
Common utilities and base class for Python derivative estimation methods.

This module provides shared imports, utilities, and the base MethodEvaluator class
that all method implementations inherit from.
"""

import json
import sys
import time
import numpy as np
from typing import Dict, List
from pathlib import Path
import os
import warnings

# Import hyperparameter selection module
try:
    # Adjust import path to find hyperparameters module in python/ directory
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
    from hyperparameters import (
        select_chebyshev_degree,
        select_fourier_harmonics,
        select_aaa_tolerance,
        select_fourier_filter_fraction_simple,
        estimate_noise_auto
    )
    ADAPTIVE_HYPERPARAMS = True
except ImportError:
    ADAPTIVE_HYPERPARAMS = False
    warnings.warn("hyperparameters module not found - using fixed hyperparameters")

# Import AAA (baryrat)
try:
    from baryrat import aaa
    BARYRAT_AVAILABLE = True
except ImportError:
    BARYRAT_AVAILABLE = False
    warnings.warn("baryrat not available - AAA methods disabled")

# Import JAX AAA (baryrat_jax) for automatic differentiation
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
    from baryrat_jax import aaa as aaa_jax
    JAX_AAA_AVAILABLE = True
except ImportError:
    JAX_AAA_AVAILABLE = False
    warnings.warn("JAX or baryrat_jax not available - JAX AAA methods with AD disabled")

# Standard scientific packages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
except Exception:
    pass
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial import hermite_e as herme
from sklearn.svm import SVR
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import UnivariateSpline

# Optional: autograd for AD-backed trig method
try:
    import autograd.numpy as anp
    from autograd import elementwise_grad as egrad
    AUTOGRAD_AVAILABLE = True
except Exception:
    AUTOGRAD_AVAILABLE = False

# Optional: TV regularized numerical differentiation
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
    from tvregdiff import TVRegDiff  # type: ignore
    TVREG_AVAILABLE = True
except Exception:
    TVREG_AVAILABLE = False


def _safe_orders(data_orders: List[int]) -> List[int]:
    """
    Convert and validate derivative orders list.

    Args:
        data_orders: List of derivative orders to compute

    Returns:
        List of integer derivative orders (defaults to 0-7 if None or invalid)
    """
    if data_orders is None:
        return list(range(8))
    try:
        return [int(o) for o in data_orders]
    except Exception:
        return list(range(8))


class MethodEvaluator:
    """
    Base class for evaluating derivative estimation methods.

    This class handles common initialization and data validation.
    Method-specific implementations should inherit from this class.

    Attributes:
        x_train: Training input points (1D array)
        y_train: Training output values (1D array)
        x_eval: Evaluation points where derivatives are computed (1D array)
        orders: List of derivative orders to compute
    """

    def __init__(self, x_train, y_train, x_eval, orders):
        """
        Initialize the method evaluator.

        Args:
            x_train: Training input points
            y_train: Training output values
            x_eval: Evaluation points for computing derivatives
            orders: List of derivative orders to compute

        Raises:
            ValueError: If training or evaluation data contains NaN or inf values
        """
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.x_eval = np.array(x_eval)
        self.orders = orders

        # Input validation: check for NaN/inf in training data
        if not np.all(np.isfinite(self.x_train)):
            raise ValueError("Training x values contain NaN or inf")
        if not np.all(np.isfinite(self.y_train)):
            raise ValueError("Training y values contain NaN or inf")
        if not np.all(np.isfinite(self.x_eval)):
            raise ValueError("Evaluation x values contain NaN or inf")

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Evaluate a single method (to be implemented by subclasses).

        Args:
            method_name: Name of the method to evaluate

        Returns:
            Dictionary with method results including predictions and timing
        """
        raise NotImplementedError("Subclasses must implement evaluate_method")
