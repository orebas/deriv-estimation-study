"""
ROM Utilities for Python Methods

This module provides utilities for densifying smoothed signals and saving them
for ROM (Reduced Order Model) processing in Julia.

The workflow:
1. Python method produces smoothed signal (order 0)
2. densify_and_save() evaluates on dense grid (~1000 points)
3. Saves to JSON in build/data/rom_input/
4. Julia reads file, builds ApproxFun interpolant, computes derivatives
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Optional
from scipy.interpolate import interp1d


def densify_signal(t_original, y_smooth, n_dense=1000, kind='cubic'):
    """
    Densify a smoothed signal to a fine grid for AAA interpolation.

    Args:
        t_original: Original time points
        y_smooth: Smoothed signal values at original points
        n_dense: Number of dense points (default: 1000)
        kind: Interpolation method ('linear', 'cubic', 'quintic')

    Returns:
        t_dense, y_dense: Densified grid and values
    """
    # Remove any NaN values
    valid_mask = np.isfinite(y_smooth)
    t_valid = t_original[valid_mask]
    y_valid = y_smooth[valid_mask]

    if len(t_valid) < 4:
        raise ValueError(f"Not enough valid points for densification: {len(t_valid)}")

    # Create dense grid
    t_min, t_max = t_valid[0], t_valid[-1]
    t_dense = np.linspace(t_min, t_max, n_dense)

    # Interpolate to dense grid
    interpolator = interp1d(t_valid, y_valid, kind=kind, bounds_error=False,
                            fill_value='extrapolate')
    y_dense = interpolator(t_dense)

    return t_dense, y_dense


def save_densified_data(method_name: str, t_dense, y_dense, t_eval,
                        metadata: Optional[Dict] = None):
    """
    Save densified data to JSON file for ROM processing.

    Args:
        method_name: Name of the method (e.g., "PyNumDiff-SavGol-Tuned")
        t_dense: Dense time grid
        y_dense: Smoothed signal on dense grid
        t_eval: Original evaluation points
        metadata: Optional metadata dictionary

    File format:
    {
        "method_name": "PyNumDiff-SavGol-Tuned",
        "t_dense": [...],
        "y_dense": [...],
        "t_eval": [...],
        "metadata": {...}
    }
    """
    # Create output directory if needed
    output_dir = Path(__file__).parent.parent.parent / "build" / "data" / "rom_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build data dictionary
    data = {
        "method_name": method_name,
        "t_dense": t_dense.tolist(),
        "y_dense": y_dense.tolist(),
        "t_eval": t_eval.tolist(),
        "metadata": metadata or {}
    }

    # Add default metadata
    data["metadata"].update({
        "n_dense": len(t_dense),
        "n_eval": len(t_eval),
        "t_range": [float(t_dense[0]), float(t_dense[-1])]
    })

    # Save to JSON
    output_file = output_dir / f"{method_name}_dense.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    return str(output_file)


def densify_and_save(method_name: str, t_original, y_smooth, t_eval,
                     n_dense=1000, kind='cubic', metadata: Optional[Dict] = None):
    """
    Convenience function: densify signal and save to JSON.

    Args:
        method_name: Name of the method
        t_original: Original time points where y_smooth is defined
        y_smooth: Smoothed signal values
        t_eval: Evaluation points (for metadata)
        n_dense: Number of dense points (default: 1000)
        kind: Interpolation method
        metadata: Optional metadata

    Returns:
        str: Path to saved file
    """
    # Densify
    t_dense, y_dense = densify_signal(t_original, y_smooth, n_dense, kind)

    # Save
    filepath = save_densified_data(method_name, t_dense, y_dense, t_eval, metadata)

    return filepath


class ROMEvaluator:
    """
    Wrapper that adds ROM capability to any method evaluator.

    Usage:
        # Wrap an existing method
        base_method = lambda t, y: some_smoothing_method(t, y)
        rom_method = ROMEvaluator(base_method, "MySmoother")

        # Densify and save
        rom_method.densify_and_save(t, y, t_eval)

        # Then call Julia to compute derivatives
    """

    def __init__(self, base_method: Callable, method_name: str):
        """
        Args:
            base_method: Function that takes (t, y) and returns smoothed signal
            method_name: Name for this method
        """
        self.base_method = base_method
        self.method_name = method_name

    def densify_and_save(self, t_train, y_train, t_eval,
                         n_dense=1000, kind='cubic', **method_kwargs):
        """
        Run base method, densify result, and save for AAA-ROM.

        Args:
            t_train: Training time points
            y_train: Training signal values
            t_eval: Evaluation points
            n_dense: Number of dense points
            kind: Interpolation method
            **method_kwargs: Passed to base_method

        Returns:
            str: Path to saved file
        """
        # Run base method to get smoothed signal
        y_smooth = self.base_method(t_train, y_train, **method_kwargs)

        # Densify and save
        metadata = {
            "source_method": self.method_name,
            "method_kwargs": method_kwargs
        }

        return densify_and_save(
            f"ROM-{self.method_name}",
            t_train, y_smooth, t_eval,
            n_dense, kind, metadata
        )


def add_rom_mode(evaluator_class):
    """
    Decorator to add ROM densification capability to a MethodEvaluator class.

    Adds a method: densify_method(method_name, n_dense=1000)
    """
    original_init = evaluator_class.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._rom_mode = False

    def enable_rom_mode(self):
        """Enable ROM densification mode."""
        self._rom_mode = True

    def disable_rom_mode(self):
        """Disable ROM densification mode."""
        self._rom_mode = False

    def densify_method(self, method_name: str, n_dense=1000, kind='cubic'):
        """
        Densify a method's smoothed output and save for ROM.

        Args:
            method_name: Name of method to densify
            n_dense: Number of dense points
            kind: Interpolation method

        Returns:
            str: Path to saved file
        """
        # Run the method to get smoothed signal (order 0)
        result = self.evaluate_method(method_name)

        if "predictions" not in result or 0 not in result["predictions"]:
            raise ValueError(f"Method {method_name} did not return order 0 (smoothed signal)")

        y_smooth = np.array(result["predictions"][0])

        # Densify and save
        metadata = {
            "source_method": method_name,
            "source_class": evaluator_class.__name__
        }

        return densify_and_save(
            f"ROM-{method_name}",
            self.x_train, y_smooth, self.x_eval,
            n_dense, kind, metadata
        )

    # Add methods to class
    evaluator_class.__init__ = new_init
    evaluator_class.enable_rom_mode = enable_rom_mode
    evaluator_class.disable_rom_mode = disable_rom_mode
    evaluator_class.densify_method = densify_method

    return evaluator_class
