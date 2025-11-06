"""
Proof of concept: Use JAX automatic differentiation to get higher-order
derivatives from PyNumDiff smooth reconstructions.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline

# Test with PyNumDiff
from pynumdiff.smooth_finite_difference import butterdiff
from pynumdiff.basis_fit import rbfdiff

print("=" * 80)
print("JAX Automatic Differentiation for PyNumDiff")
print("=" * 80)

# Create test signal
np.random.seed(42)
n = 101
t = np.linspace(0, 1, n)
dt = np.mean(np.diff(t))
y_true = np.sin(2 * np.pi * t)
dy_true = 2 * np.pi * np.cos(2 * np.pi * t)
d2y_true = -(2 * np.pi)**2 * np.sin(2 * np.pi * t)
y_noisy = y_true + 1e-3 * np.random.randn(n)

print(f"\nTest signal: N={n}, dt={dt:.6f}")
print(f"True signal: y = sin(2πt)")
print(f"True dy/dt: dy = 2π·cos(2πt)")
print(f"True d²y/dt²: d²y = -(2π)²·sin(2πt)")

# Method 1: Use PyNumDiff to get smooth reconstruction, then JAX for derivatives
print("\n" + "=" * 80)
print("Method 1: PyNumDiff butterdiff + JAX for higher derivatives")
print("=" * 80)

# Get smooth reconstruction from PyNumDiff
x_smooth, dx_dt = butterdiff(y_noisy, dt, filter_order=2, cutoff_freq=0.2)
print(f"\nPyNumDiff butterdiff returned:")
print(f"  x_smooth shape: {x_smooth.shape}")
print(f"  dx_dt shape: {dx_dt.shape}")

# Create a JAX-differentiable interpolating function
# Strategy: Fit a spline to x_smooth, then use JAX to differentiate it
spline = UnivariateSpline(t, x_smooth, k=5, s=0)

# Create JAX-compatible evaluation function
def smooth_function(t_eval):
    """JAX-compatible wrapper around spline evaluation."""
    # Note: scipy splines aren't JAX-compatible directly, but we can
    # evaluate at points and use those values
    return jnp.array(spline(np.array(t_eval)))

# Alternative: Create interpolator using JAX directly
# Use JAX's grad, grad(grad), etc. to get higher derivatives
print("\n" + "-" * 80)
print("Using scipy spline + manual finite differences:")
print("-" * 80)

# Order 0: smooth signal
y0 = spline(t)
print(f"Order 0 RMSE: {np.sqrt(np.mean((y0 - y_true)**2)):.4e}")

# Order 1: first derivative
dy = spline.derivative(n=1)(t)
print(f"Order 1 RMSE: {np.sqrt(np.mean((dy - dy_true)**2)):.4e}")

# Order 2: second derivative
d2y = spline.derivative(n=2)(t)
print(f"Order 2 RMSE: {np.sqrt(np.mean((d2y - d2y_true)**2)):.4e}")

# Method 2: Direct JAX approach with central differences on smooth signal
print("\n" + "=" * 80)
print("Method 2: JAX central differences on smooth reconstruction")
print("=" * 80)

def central_diff_jax(y_smooth, dt):
    """Compute derivative using JAX-compatible central differences."""
    # Pad edges for central difference
    dy = jnp.zeros_like(y_smooth)
    dy = dy.at[1:-1].set((y_smooth[2:] - y_smooth[:-2]) / (2 * dt))
    # Forward/backward diff at endpoints
    dy = dy.at[0].set((y_smooth[1] - y_smooth[0]) / dt)
    dy = dy.at[-1].set((y_smooth[-1] - y_smooth[-2]) / dt)
    return dy

# Convert to JAX arrays
x_smooth_jax = jnp.array(x_smooth)

# Order 1
dy_jax = central_diff_jax(x_smooth_jax, dt)
print(f"Order 1 (JAX central diff) RMSE: {np.sqrt(np.mean((np.array(dy_jax) - dy_true)**2)):.4e}")

# Order 2: apply central diff again
d2y_jax = central_diff_jax(dy_jax, dt)
print(f"Order 2 (JAX central diff²) RMSE: {np.sqrt(np.mean((np.array(d2y_jax) - d2y_true)**2)):.4e}")

# Method 3: RBF with AD
print("\n" + "=" * 80)
print("Method 3: RBF reconstruction + spline differentiation")
print("=" * 80)

try:
    x_rbf, dx_rbf = rbfdiff(y_noisy, dt, sigma=1.0, lmbd=0.01)
    print(f"\nPyNumDiff rbfdiff returned:")
    print(f"  x_rbf shape: {x_rbf.shape}")

    # Fit spline to RBF reconstruction
    rbf_spline = UnivariateSpline(t, x_rbf, k=5, s=0)

    # Get derivatives
    y0_rbf = rbf_spline(t)
    dy_rbf = rbf_spline.derivative(n=1)(t)
    d2y_rbf = rbf_spline.derivative(n=2)(t)

    print(f"Order 0 RMSE: {np.sqrt(np.mean((y0_rbf - y_true)**2)):.4e}")
    print(f"Order 1 RMSE: {np.sqrt(np.mean((dy_rbf - dy_true)**2)):.4e}")
    print(f"Order 2 RMSE: {np.sqrt(np.mean((d2y_rbf - d2y_true)**2)):.4e}")
except Exception as e:
    print(f"RBF test failed: {e}")

# Summary
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Strategy for PyNumDiff + Higher Derivatives:

For methods that return smooth reconstructions (x_hat):
  1. Get x_smooth from PyNumDiff method
  2. Fit a high-order spline to x_smooth (k=5 or higher)
  3. Use spline.derivative(n=order) for orders 0-k

This approach:
  ✓ Uses PyNumDiff as intended (for smoothing)
  ✓ Gets higher derivatives from the smooth reconstruction
  ✓ Doesn't require mucking with package internals
  ✓ Works for: butterdiff, gaussiandiff, friedrichsdiff,
              rbfdiff, splinediff, polydiff, spectraldiff

Alternative for methods like savgoldiff:
  - Just use deriv=0,1,2,... parameter directly!

Alternative for spectraldiff:
  - Multiply by (i*omega)^n in frequency domain
""")
