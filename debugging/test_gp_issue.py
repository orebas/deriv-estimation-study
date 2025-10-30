#!/usr/bin/env python3
"""
Test script to reproduce and debug Python GP performance issues.
"""

import numpy as np
import sys
from pathlib import Path

# Add methods path
sys.path.insert(0, str(Path(__file__).parent.parent / "methods" / "python"))
from gp.gaussian_process import GPMethods

def test_simple_sine():
    """Test on a simple sine function where we know the derivatives."""
    print("=" * 80)
    print("Testing GP on simple sine function")
    print("=" * 80)

    # Generate test data
    x_train = np.linspace(0, 2*np.pi, 50)
    y_train = np.sin(x_train)
    x_eval = np.array([np.pi/4, np.pi/2, 3*np.pi/4, np.pi])

    # True derivatives at evaluation points
    true_derivs = {
        0: np.sin(x_eval),
        1: np.cos(x_eval),
        2: -np.sin(x_eval),
        3: -np.cos(x_eval),
        4: np.sin(x_eval),
    }

    # Test GP methods
    for method_name in ["GP_RBF_Python", "GP_Matern_2.5_Python"]:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        gp = GPMethods(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            orders=[0, 1, 2, 3, 4]
        )

        result = gp.evaluate_method(method_name)

        print(f"\nHyperparameters: {result.get('meta', {})}")

        if result.get('failures'):
            print(f"Failures: {result['failures']}")

        print(f"\n{'Order':<8} {'Predicted':<20} {'True':<20} {'Error':<20}")
        print("-" * 68)

        for order in [0, 1, 2, 3, 4]:
            if order in result['predictions']:
                preds = result['predictions'][order]
                true_vals = true_derivs[order]

                # Calculate RMSE
                rmse = np.sqrt(np.mean((np.array(preds) - true_vals)**2))

                # Show first eval point
                print(f"{order:<8} {preds[0]:<20.6f} {true_vals[0]:<20.6f} {abs(preds[0]-true_vals[0]):<20.6f}")
                print(f"         RMSE: {rmse:.6f}")

def test_lotka_volterra_simple():
    """Test on Lotka-Volterra-like data to mimic the real scenario."""
    print("\n" + "=" * 80)
    print("Testing GP on oscillatory data (mimics Lotka-Volterra)")
    print("=" * 80)

    # Generate synthetic oscillatory data
    np.random.seed(42)
    x_train = np.linspace(0, 10, 100)
    # Lotka-Volterra-like oscillation
    y_train = 2 + 0.5 * np.sin(2*x_train) + 0.3 * np.cos(3*x_train)
    # Add tiny noise
    y_train += np.random.normal(0, 1e-6, len(y_train))

    x_eval = np.linspace(1, 9, 20)

    # Test with higher order derivatives
    orders = [0, 1, 2, 3, 4, 5]

    for method_name in ["GP_RBF_Python", "GP_Matern_1.5_Python", "GP_Matern_2.5_Python"]:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        gp = GPMethods(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            orders=orders
        )

        result = gp.evaluate_method(method_name)

        print(f"\nHyperparameters: {result.get('meta', {})}")

        if result.get('failures'):
            print(f"Failures: {result['failures']}")

        print(f"\n{'Order':<8} {'Mean Pred':<20} {'Std Pred':<20} {'Min':<20} {'Max':<20}")
        print("-" * 88)

        for order in orders:
            if order in result['predictions']:
                preds = np.array(result['predictions'][order])
                if not np.all(np.isnan(preds)):
                    print(f"{order:<8} {np.mean(preds):<20.6f} {np.std(preds):<20.6f} "
                          f"{np.min(preds):<20.6f} {np.max(preds):<20.6f}")
                else:
                    print(f"{order:<8} ALL NaN")

def test_matern_kernel_derivative():
    """Test the Matern kernel derivative formula directly."""
    print("\n" + "=" * 80)
    print("Testing Matern 2.5 kernel derivative formula (order 7)")
    print("=" * 80)

    # Create a simple test case
    x_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_train = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    x_eval = np.array([1.5])

    gp = GPMethods(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        orders=[7]
    )

    result = gp.evaluate_method("GP_Matern_2.5_Python")

    print(f"\nHyperparameters: {result.get('meta', {})}")
    print(f"Order 7 prediction: {result['predictions'][7]}")
    print(f"Failures: {result.get('failures', {})}")

    # Compare with what the formula should give
    ell = result['meta']['length_scale']
    print(f"\nLength scale: {ell}")
    print(f"Check if result is reasonable...")
    if abs(result['predictions'][7][0]) > 1e6:
        print("WARNING: Order 7 prediction seems unreasonably large!")
        print("This could indicate a bug in the kernel derivative formula.")

if __name__ == "__main__":
    test_simple_sine()
    test_lotka_volterra_simple()
    test_matern_kernel_derivative()

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
