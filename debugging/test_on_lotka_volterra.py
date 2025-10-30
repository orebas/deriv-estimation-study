#!/usr/bin/env python3
"""
Test Python GP on actual Lotka-Volterra data matching the study setup.
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys
from pathlib import Path

# Add methods path
sys.path.insert(0, str(Path(__file__).parent.parent / "methods" / "python"))
from gp.gaussian_process import GPMethods

def lotka_volterra(t, y):
    """Lotka-Volterra ODE system."""
    x, y_val = y
    alpha, beta, gamma, delta = 1.5, 1.0, 3.0, 1.0
    dxdt = alpha * x - beta * x * y_val
    dydt = -gamma * y_val + delta * x * y_val
    return [dxdt, dydt]

def generate_lotka_volterra_data(noise_level=1e-8, n_points=1000):
    """Generate Lotka-Volterra test data matching study parameters."""
    # Initial conditions
    y0 = [1.0, 1.0]

    # Time span
    t_span = (0.0, 10.0)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # Solve ODE
    sol = solve_ivp(lotka_volterra, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)

    # Add noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, noise_level, sol.y.shape)
    y_noisy = sol.y + noise

    return sol.t, y_noisy

def test_gp_on_lotka_volterra():
    """Test GP RBF on Lotka-Volterra data."""
    print("=" * 80)
    print("Testing Python GP_RBF on Lotka-Volterra Data")
    print("=" * 80)

    # Generate data with noise level 1e-8 (matching the problematic results)
    t, y_noisy = generate_lotka_volterra_data(noise_level=1e-8, n_points=1000)

    # Use first variable (prey population)
    x_train = t
    y_train = y_noisy[0, :]

    # Use every 4th point as evaluation points (matching study's subsampling)
    eval_indices = np.arange(0, len(t), 4)
    x_eval = t[eval_indices]

    print(f"\nData info:")
    print(f"  Training points: {len(x_train)}")
    print(f"  Evaluation points: {len(x_eval)}")
    print(f"  Time range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  y range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  y std: {np.std(y_train):.6f}")

    # Test derivative orders
    orders = [0, 1, 2, 3]

    print(f"\nRunning GP_RBF_Python on derivative orders {orders}...")

    gp = GPMethods(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        orders=orders
    )

    result = gp.evaluate_method("GP_RBF_Python")

    print(f"\nFitted hyperparameters:")
    print(f"  {result.get('meta', {})}")

    if result.get('failures'):
        print(f"\nFailures: {result['failures']}")

    # Compute true derivatives numerically for comparison
    from scipy.interpolate import UnivariateSpline

    # Use clean data for "true" derivatives
    t_clean, y_clean = generate_lotka_volterra_data(noise_level=0, n_points=1000)
    y_clean_var = y_clean[0, :]

    # Create spline for smooth derivatives
    spline = UnivariateSpline(t_clean, y_clean_var, s=0, k=5)

    true_derivs = {}
    for order in orders:
        if order == 0:
            true_derivs[order] = spline(x_eval)
        else:
            true_derivs[order] = spline.derivative(order)(x_eval)

    print(f"\n{'Order':<8} {'RMSE':<15} {'NRMSE':<15} {'Max Error':<15}")
    print("-" * 53)

    for order in orders:
        if order in result['predictions']:
            preds = np.array(result['predictions'][order])
            true_vals = true_derivs[order]

            if not np.all(np.isnan(preds)):
                rmse = np.sqrt(np.mean((preds - true_vals)**2))

                # NRMSE: normalize by range of true values
                y_range = np.max(true_vals) - np.min(true_vals)
                nrmse = rmse / y_range if y_range > 0 else float('inf')

                max_error = np.max(np.abs(preds - true_vals))

                print(f"{order:<8} {rmse:<15.6e} {nrmse:<15.6f} {max_error:<15.6e}")
            else:
                print(f"{order:<8} ALL NaN")

    # Compare with the problematic results from the table
    print(f"\n{'='*80}")
    print("Comparison with problematic results from table:")
    print(f"{'='*80}")
    print("Order 3 from table: RMSE=47.06, NRMSE=0.498")
    print(f"Order 3 from our test (above): Check if NRMSE is much better than 0.498")
    print()
    print("If NRMSE is << 0.498, the fix worked!")
    print("If NRMSE is still ~0.5, there's still an issue.")

if __name__ == "__main__":
    test_gp_on_lotka_volterra()
