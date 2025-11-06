"""
Test to verify PyNumDiff parameter passing.
"""
import numpy as np
import sys
from pathlib import Path

# Test direct API calls
import pynumdiff
from pynumdiff import smooth_finite_difference as sfd
from pynumdiff import total_variation_regularization as tvr
from pynumdiff import kalman_smooth
from pynumdiff.optimize import optimize as pnd_optimize

print("=" * 80)
print("PyNumDiff Parameter Verification Test")
print("=" * 80)

# Create test signal
np.random.seed(42)
n = 101
x = np.linspace(0, 1, n)
dt = np.mean(np.diff(x))
y = np.sin(2 * np.pi * x) + 1e-3 * np.random.randn(n)

print(f"\nTest signal: N={n}, dt={dt:.6f}")

# Test 1: butterdiff with explicit parameters
print("\n1. Testing butterdiff API:")
try:
    x_smooth, dx_dt = sfd.butterdiff(y, dt, filter_order=2, cutoff_freq=0.2)
    print(f"   ✓ Direct call works: filter_order=2, cutoff_freq=0.2")
    print(f"     Shape: {x_smooth.shape}, finite: {np.sum(np.isfinite(x_smooth))}/{len(x_smooth)}")
except Exception as e:
    print(f"   ✗ Direct call failed: {e}")

# Test 2: butterdiff with optimize
print("\n2. Testing butterdiff with optimize:")
try:
    # Estimate cutoff frequency heuristically
    f_cutoff = 0.5  # example
    tvgamma = np.exp(-1.6 * np.log(f_cutoff) - 0.71 * np.log(dt) - 5.1)
    params, val = pnd_optimize(sfd.butterdiff, y, dt, tvgamma=tvgamma,
                                metric='rmse', padding='auto',
                                opt_method='Nelder-Mead', maxiter=5)
    print(f"   ✓ Optimize works: params={params}")

    # Try using the optimized params
    x_smooth, dx_dt = sfd.butterdiff(y, dt, **params)
    print(f"   ✓ Can call with **params: {np.sum(np.isfinite(x_smooth))}/{len(x_smooth)} finite")
except Exception as e:
    print(f"   ✗ Optimize failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: TV velocity
print("\n3. Testing TV velocity:")
try:
    gamma = 10.0
    x_smooth, dx_dt = tvr.velocity(y, dt, gamma=gamma)
    print(f"   ✓ TV velocity works: gamma={gamma}")
    print(f"     Shape: {x_smooth.shape}, finite: {np.sum(np.isfinite(dx_dt))}/{len(dx_dt)}")
except Exception as e:
    print(f"   ✗ TV velocity failed: {e}")

# Test 4: Kalman constant_acceleration
print("\n4. Testing Kalman constant_acceleration:")
try:
    result = kalman_smooth.constant_acceleration(y, dt, r=1e-8, q=1e-6)
    print(f"   ✓ constant_acceleration works")
    print(f"     Returns {len(result)} values (should be 2: position, velocity)")
    if len(result) == 2:
        x_smooth, dx_smooth = result
        print(f"     Position shape: {x_smooth.shape}, Velocity shape: {dx_smooth.shape}")
    else:
        print(f"     WARNING: Expected 2 values, got {len(result)}")
except Exception as e:
    print(f"   ✗ constant_acceleration failed: {e}")

# Test 5: Check default parameters
print("\n5. Checking method signatures:")
import inspect

methods_to_check = [
    ("butterdiff", sfd.butterdiff),
    ("gaussiandiff", sfd.gaussiandiff),
    ("friedrichsdiff", sfd.friedrichsdiff),
    ("splinediff", sfd.splinediff),
]

for name, func in methods_to_check:
    sig = inspect.signature(func)
    print(f"\n   {name}:")
    for param_name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty and param_name not in ['x', 'dt', 'params', 'options']:
            print(f"     - {param_name} = {param.default}")

print("\n" + "=" * 80)
print("Parameter Verification Complete")
print("=" * 80)
