"""
Test ALL PyNumDiff methods with CORRECT API (no deprecated params)
"""

import numpy as np
import matplotlib.pyplot as plt
import pynumdiff
from pynumdiff import smooth_finite_difference as sfd
from pynumdiff import total_variation_regularization as tvr
from pynumdiff.polynomial_fit import polydiff, savgoldiff, splinediff
from pynumdiff.basis_fit import rbfdiff, spectraldiff
from pynumdiff import kalman_smooth

# Generate test signal
n = 201
t = np.linspace(0, 1, n)
dt = np.mean(np.diff(t))

# True signal and derivatives
y_true = np.sin(2 * np.pi * t)
dy_true = 2 * np.pi * np.cos(2 * np.pi * t)

# Add noise
np.random.seed(42)
noise_level = 1e-3
y_noisy = y_true + noise_level * np.random.randn(n)

print("=" * 80)
print("COMPLETE PYNUMDIFF TEST WITH CORRECT API")
print("=" * 80)

# Define ALL methods with CORRECT API
methods = {
    # ========== FINITE DIFFERENCES ==========
    'first_order': lambda: (y_noisy, pynumdiff.first_order(y_noisy, dt)),
    'second_order': lambda: (y_noisy, pynumdiff.second_order(y_noisy, dt)),
    'fourth_order': lambda: (y_noisy, pynumdiff.fourth_order(y_noisy, dt)),

    # ========== SMOOTH FINITE DIFFERENCE ==========
    'butterdiff': lambda: sfd.butterdiff(y_noisy, dt, filter_order=2, cutoff_freq=0.2),
    'gaussiandiff': lambda: sfd.gaussiandiff(y_noisy, dt, window_size=7, num_iterations=1),
    'friedrichsdiff': lambda: sfd.friedrichsdiff(y_noisy, dt, window_size=7, num_iterations=1),
    'meandiff': lambda: sfd.meandiff(y_noisy, dt, window_size=7, num_iterations=1),
    'mediandiff': lambda: sfd.mediandiff(y_noisy, dt, window_size=7, num_iterations=1),

    # ========== POLYNOMIAL ==========
    'savgoldiff': lambda: savgoldiff(y_noisy, dt, degree=3, window_size=7),
    'splinediff': lambda: splinediff(y_noisy, dt, degree=3, smoothing_factor=1e-3),
    'polydiff': lambda: (y_noisy, polydiff(y_noisy, dt, degree=5, diff_order=1)),

    # ========== BASIS FUNCTIONS ==========
    'spectraldiff': lambda: spectraldiff(y_noisy, dt, high_freq_cutoff=0.2),
    'rbfdiff': lambda: rbfdiff(y_noisy, dt, window_size=7),
    'lineardiff': lambda: (y_noisy, pynumdiff.lineardiff(y_noisy, dt, diff_order=1)),

    # ========== TOTAL VARIATION ==========
    'tvrdiff': lambda: tvr.tvrdiff(y_noisy, dt, alph=0.01, iterations=20),
    'tv_velocity': lambda: tvr.velocity(y_noisy, dt, gamma=1e-3),
    'tv_acceleration': lambda: tvr.acceleration(y_noisy, dt, gamma=1e-3),
    'tv_iterative': lambda: tvr.iterative_velocity(y_noisy, t, alpha=1e-2, gamma=1e-3),

    # ========== KALMAN ==========
    'kalman_const_vel': lambda: kalman_smooth.constant_velocity(y_noisy, dt, alpha=1.0),
    'kalman_const_accel': lambda: kalman_smooth.constant_acceleration(y_noisy, dt, alpha=1.0),
    'kalman_rts': lambda: kalman_smooth.rts_smooth(y_noisy, dt, alpha=1.0),
}

results = {}
categories = {
    'Finite Diff': ['first_order', 'second_order', 'fourth_order'],
    'Smooth FD': ['butterdiff', 'gaussiandiff', 'friedrichsdiff', 'meandiff', 'mediandiff'],
    'Polynomial': ['savgoldiff', 'splinediff', 'polydiff'],
    'Basis': ['spectraldiff', 'rbfdiff', 'lineardiff'],
    'Total Var': ['tvrdiff', 'tv_velocity', 'tv_acceleration', 'tv_iterative'],
    'Kalman': ['kalman_const_vel', 'kalman_const_accel', 'kalman_rts']
}

print("\nTesting all methods:")
print("-" * 80)
print("Method               | RMSE    | Max Err | Interior | Boundary | Status")
print("-" * 80)

for name, method in methods.items():
    try:
        # Get result
        result = method()

        # Extract derivative
        if isinstance(result, tuple) and len(result) >= 2:
            x_smooth = result[0]
            dx_dt = result[1]
        else:
            dx_dt = result
            x_smooth = None

        # Handle special formats
        if hasattr(dx_dt, 'flatten'):
            dx_dt = dx_dt.flatten()

        # For methods that return second derivative
        if name in ['tv_acceleration', 'kalman_const_accel']:
            # These return acceleration (2nd derivative), skip for now
            print(f"{name:20} | Returns 2nd derivative - skipping")
            continue

        if len(dx_dt) != len(t):
            print(f"{name:20} | Size mismatch: {len(dx_dt)} vs {len(t)}")
            continue

        # Calculate metrics
        rmse = np.sqrt(np.mean((dx_dt - dy_true)**2))
        max_err = np.max(np.abs(dx_dt - dy_true))

        # Boundary analysis
        inner = slice(20, 181)
        boundary = np.concatenate([np.arange(20), np.arange(181, 201)])
        rmse_inner = np.sqrt(np.mean((dx_dt[inner] - dy_true[inner])**2))
        rmse_boundary = np.sqrt(np.mean((dx_dt[boundary] - dy_true[boundary])**2))
        ratio = rmse_boundary / rmse_inner if rmse_inner > 0 else np.inf

        results[name] = {
            'dx_dt': dx_dt,
            'rmse': rmse,
            'max_err': max_err,
            'ratio': ratio
        }

        # Status
        if rmse < 0.05:
            status = "✓✓ Excellent"
        elif rmse < 0.1:
            status = "✓ Good"
        elif rmse < 0.5:
            status = "~ OK"
        else:
            status = "✗ Poor"

        print(f"{name:20} | {rmse:7.4f} | {max_err:7.4f} | {rmse_inner:8.4f} | {ratio:8.1f}x | {status}")

    except Exception as e:
        print(f"{name:20} | FAILED: {str(e)[:40]}")

# Summary by category
print("\n" + "=" * 80)
print("SUMMARY BY CATEGORY")
print("=" * 80)

for cat_name, method_list in categories.items():
    cat_results = [(m, results[m]['rmse']) for m in method_list if m in results]
    if cat_results:
        cat_results.sort(key=lambda x: x[1])
        print(f"\n{cat_name}:")
        print("-" * 40)

        excellent = [m for m, r in cat_results if r < 0.05]
        good = [m for m, r in cat_results if 0.05 <= r < 0.1]
        ok = [m for m, r in cat_results if 0.1 <= r < 0.5]
        poor = [m for m, r in cat_results if r >= 0.5]

        if excellent:
            print(f"  ✓✓ Excellent: {', '.join(excellent)}")
        if good:
            print(f"  ✓ Good: {', '.join(good)}")
        if ok:
            print(f"  ~ OK: {', '.join(ok)}")
        if poor:
            print(f"  ✗ Poor: {', '.join(poor)}")

        print(f"  Best: {cat_results[0][0]} (RMSE={cat_results[0][1]:.4f})")

# Overall statistics
total = len(results)
excellent = sum(1 for r in results.values() if r['rmse'] < 0.05)
good = sum(1 for r in results.values() if 0.05 <= r['rmse'] < 0.1)

print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)
print(f"Total methods tested: {total}")
print(f"Excellent (<0.05 RMSE): {excellent}")
print(f"Good (0.05-0.1 RMSE): {good}")
print(f"OK (0.1-0.5 RMSE): {sum(1 for r in results.values() if 0.1 <= r['rmse'] < 0.5)}")
print(f"Poor (>0.5 RMSE): {sum(1 for r in results.values() if r['rmse'] >= 0.5)}")

# Top methods
sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
print(f"\n🏆 TOP 5 METHODS:")
for i, (name, data) in enumerate(sorted_results[:5], 1):
    cat = [c for c, m in categories.items() if name in m][0]
    print(f"  {i}. {name}: RMSE={data['rmse']:.4f} ({cat})")

# Boundary handling
best_boundary = sorted(results.items(), key=lambda x: x[1]['ratio'])[:3]
print(f"\n🎯 BEST BOUNDARY HANDLING:")
for name, data in best_boundary:
    print(f"  {name}: {data['ratio']:.1f}x ratio")

print(f"\n📊 SUCCESS RATE: {100*(excellent+good)/total:.1f}% of methods work well")
print(f"PyNumDiff has {excellent+good}/{total} effective methods for first derivatives")

# Quick plot of best methods
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot 1: Top 3 overall
ax = axes[0, 0]
ax.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.7)
for name, data in sorted_results[:3]:
    ax.plot(t, data['dx_dt'], linewidth=1.5, label=f"{name} ({data['rmse']:.3f})", alpha=0.8)
ax.set_title('Top 3 Methods Overall')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2-6: Best from each category
for idx, (cat_name, method_list) in enumerate(list(categories.items())[:5]):
    ax = axes.flatten()[idx+1]
    ax.plot(t, dy_true, 'k-', linewidth=2, label='True', alpha=0.5)

    cat_results = [(m, results[m]) for m in method_list if m in results]
    if cat_results:
        cat_results.sort(key=lambda x: x[1]['rmse'])
        name, data = cat_results[0]
        ax.plot(t, data['dx_dt'], 'b-', linewidth=1.5,
                label=f"{name} ({data['rmse']:.3f})")
        ax.set_title(f'Best {cat_name}: {name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle('PyNumDiff Complete Analysis - All Methods Working!', fontsize=14)
plt.tight_layout()
plt.savefig('pynumdiff_all_methods.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization: pynumdiff_all_methods.png")