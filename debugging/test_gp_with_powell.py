#!/usr/bin/env python3
"""
Test script with Powell optimizer enabled and debug output.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Enable Powell optimizer
os.environ["GP_OPTIMIZER"] = "powell"

# Add methods path
sys.path.insert(0, str(Path(__file__).parent.parent / "methods" / "python"))
from gp.gaussian_process import GPMethods, _heuristic_inits

def test_matern_with_debug():
    """Test Matérn with debug output."""
    print("=" * 80)
    print("Testing Matérn 2.5 with Powell optimizer and debug info")
    print("=" * 80)

    # Generate test data - simple sine
    x_train = np.linspace(0, 2*np.pi, 50)
    y_train = np.sin(x_train)
    x_eval = np.array([np.pi/4, np.pi/2, 3*np.pi/4, np.pi])

    # Check initialization
    ell0, amp0 = _heuristic_inits(x_train, y_train)
    print(f"\nInitialization:")
    print(f"  ell0 = {ell0:.6f}")
    print(f"  amp0 = {amp0:.6f}")
    print(f"  x_train range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  y_train std: {np.std(y_train):.6f}")
    print(f"  y_train var: {np.var(y_train):.6f}")

    # Test Matérn 2.5
    gp = GPMethods(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        orders=[0, 1, 2]
    )

    print(f"\nFitting GP_Matern_2.5_Python with Powell optimizer...")
    result = gp.evaluate_method("GP_Matern_2.5_Python")

    print(f"\nFitted hyperparameters:")
    print(f"  {result.get('meta', {})}")

    true_derivs = {
        0: np.sin(x_eval),
        1: np.cos(x_eval),
        2: -np.sin(x_eval),
    }

    print(f"\n{'Order':<8} {'RMSE':<20}")
    print("-" * 28)
    for order in [0, 1, 2]:
        preds = np.array(result['predictions'][order])
        true_vals = true_derivs[order]
        rmse = np.sqrt(np.mean((preds - true_vals)**2))
        print(f"{order:<8} {rmse:<20.6f}")

if __name__ == "__main__":
    test_matern_with_debug()
