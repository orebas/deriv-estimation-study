"""
Investigate if we can extract polynomial coefficients from polydiff
to compute higher-order derivatives analytically.
"""
import numpy as np
from scipy.signal import savgol_filter

# Test signal
np.random.seed(42)
n = 101
t = np.linspace(0, 1, n)
dt = np.mean(np.diff(t))
y_true = np.sin(2 * np.pi * t)
y_noisy = y_true + 1e-3 * np.random.randn(n)

print("=" * 80)
print("Polydiff Coefficient Extraction Investigation")
print("=" * 80)

# Let's replicate polydiff's internal _polydiff function
def polydiff_with_coeffs(x, dt, degree):
    """
    Replicate PyNumDiff's _polydiff but return polynomial coefficients.

    Based on lines 86-95 of _polynomial_fit.py
    """
    t = np.arange(len(x)) * dt

    # Fit polynomial (returns coefficients highest order first)
    coeffs = np.polyfit(t, x, degree)

    print(f"\nPolynomial fit (degree {degree}):")
    print(f"  Coefficients shape: {coeffs.shape}")
    print(f"  Highest order: {coeffs[0]:.6e}")
    print(f"  Lowest order: {coeffs[-1]:.6e}")

    # Compute derivatives of all orders
    derivatives = {}
    for order in range(min(degree + 1, 8)):  # Up to order 7 or degree
        if order == 0:
            # Order 0: evaluate polynomial
            deriv_coeffs = coeffs
        else:
            # Order n: take nth derivative of polynomial
            deriv_coeffs = coeffs.copy()
            for _ in range(order):
                deriv_coeffs = np.polyder(deriv_coeffs)

        # Evaluate at all points
        y_deriv = np.polyval(deriv_coeffs, t)
        derivatives[order] = y_deriv

    return derivatives, coeffs

# Test with different polynomial degrees
for degree in [3, 5, 7]:
    print("\n" + "=" * 80)
    print(f"Testing with degree = {degree}")
    print("=" * 80)

    derivatives, coeffs = polydiff_with_coeffs(y_noisy, dt, degree)

    # Compute ground truth for comparison
    def ground_truth(order):
        omega = 2 * np.pi
        if order == 0:
            return np.sin(omega * t)
        elif order == 1:
            return omega * np.cos(omega * t)
        elif order == 2:
            return -(omega**2) * np.sin(omega * t)
        elif order == 3:
            return -(omega**3) * np.cos(omega * t)
        elif order == 4:
            return omega**4 * np.sin(omega * t)
        elif order == 5:
            return omega**5 * np.cos(omega * t)
        elif order == 6:
            return -(omega**6) * np.sin(omega * t)
        elif order == 7:
            return -(omega**7) * np.cos(omega * t)

    print(f"\nErrors (RMSE):")
    for order in range(min(degree + 1, 8)):
        if order in derivatives:
            y_exact = ground_truth(order)
            rmse = np.sqrt(np.mean((derivatives[order] - y_exact)**2))
            print(f"  Order {order}: {rmse:.4e}")

# Compare with PyNumDiff's polydiff (sliding window version)
print("\n" + "=" * 80)
print("PyNumDiff polydiff (sliding window)")
print("=" * 80)

from pynumdiff.polynomial_fit import polydiff

# polydiff requires window_size for sliding window
try:
    x_hat, dx_hat = polydiff(y_noisy, dt, degree=7, window_size=31, step_size=1, kernel='friedrichs')
    print(f"\nPyNumDiff polydiff returned:")
    print(f"  x_hat shape: {x_hat.shape}")
    print(f"  dx_hat shape: {dx_hat.shape}")

    # Can we get order 1?
    y_exact = ground_truth(1)
    rmse = np.sqrt(np.mean((dx_hat - y_exact)**2))
    print(f"  Order 1 RMSE: {rmse:.4e}")

    print("\nNote: PyNumDiff's polydiff uses sliding windows, so we can't easily")
    print("extract a single polynomial. Each window gets its own polynomial fit.")
    print("We'd need to modify it to return coefficients for each window.")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
For polydiff:
✓ Single polynomial fit (no sliding): Easy to get all derivatives
✗ Sliding window version: Each window has different coefficients
  - Would need to modify PyNumDiff to return coefficients per window
  - Then differentiate each polynomial separately

Options:
1. Implement our own "simple polydiff" without sliding windows
2. Modify PyNumDiff's polydiff to expose coefficients
3. Just use orders 0-1 for PyNumDiff's polydiff
4. Use scipy.signal.savgol_filter instead (it's similar and has native deriv)

Recommendation: Use savgol_filter instead of polydiff
  - Both fit polynomials on sliding windows
  - savgol_filter has native deriv parameter
  - savgol_filter is faster and well-tested
""")
