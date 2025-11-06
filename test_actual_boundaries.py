"""
Test boundary handling in actual PyNumDiff implementations
"""

import numpy as np
import matplotlib.pyplot as plt
from pynumdiff import finite_difference, savgol, trend_filtered, total_variation_regularization
from pynumdiff import kalman_smooth, spline_smooth, chebychev

# Generate test signal
n = 201
t = np.linspace(0, 1, n)
dt = t[1] - t[0]

# True signal and derivative
y_true = np.sin(2 * np.pi * t)
dy_true = 2 * np.pi * np.cos(2 * np.pi * t)

# Add noise
np.random.seed(42)
noise_level = 1e-3
y_noisy = y_true + noise_level * np.random.randn(n)

print("=" * 80)
print("TESTING ACTUAL PYNUMDIFF BOUNDARY HANDLING")
print("=" * 80)
print()

# Define regions
inner_80 = slice(20, 181)  # Middle 80%
boundary_left = slice(0, 20)
boundary_right = slice(181, 201)

results = {}

# Test different methods
methods = {
    'finite_diff': lambda: finite_difference.first_order(y_noisy, dt),
    'savgol': lambda: savgol.savgol(y_noisy, deriv_order=1, left=5, right=5, order=3),
    'trend_filtered': lambda: trend_filtered.iterative_velocity(y_noisy, t, alpha=1e-2, gamma=1e-3),
    'total_variation': lambda: total_variation_regularization.velocity(y_noisy, dt, alpha=1e-2),
    'kalman': lambda: kalman_smooth.kalman_smooth(y_noisy, t, alpha=1, return_data='dxdt'),
    'spline': lambda: spline_smooth.spline_smooth(y_noisy, t, s=1e-3),
    'chebychev': lambda: chebychev.chebychev(y_noisy, t, n=20, m=1)
}

print("Method               | Interior RMSE | Boundary RMSE | Ratio")
print("-" * 60)

for name, method in methods.items():
    try:
        # Get derivative estimate
        dy_est = method()

        # Handle different return types
        if isinstance(dy_est, tuple):
            dy_est = dy_est[0]  # Some methods return (derivative, other_data)

        # Ensure it's the right shape
        if len(dy_est) != len(t):
            print(f"{name:20} | Shape mismatch: {len(dy_est)} vs {len(t)}")
            continue

        # Calculate errors
        error_interior = np.sqrt(np.mean((dy_est[inner_80] - dy_true[inner_80])**2))
        error_left = np.sqrt(np.mean((dy_est[boundary_left] - dy_true[boundary_left])**2))
        error_right = np.sqrt(np.mean((dy_est[boundary_right] - dy_true[boundary_right])**2))
        error_boundary = (error_left + error_right) / 2

        ratio = error_boundary / error_interior if error_interior > 0 else float('inf')

        print(f"{name:20} | {error_interior:13.4f} | {error_boundary:13.4f} | {ratio:5.1f}x")

        results[name] = {
            'dy_est': dy_est,
            'error_interior': error_interior,
            'error_boundary': error_boundary,
            'ratio': ratio
        }

    except Exception as e:
        print(f"{name:20} | Failed: {str(e)[:30]}")

# Visualize worst offenders
print("\nGenerating visualization...")

# Sort by ratio
sorted_methods = sorted(results.items(), key=lambda x: x[1]['ratio'], reverse=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot worst 3 methods
for idx, (name, data) in enumerate(sorted_methods[:3]):
    ax = axes.flatten()[idx]
    ax.plot(t, dy_true, 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(t, data['dy_est'], 'r-', linewidth=1.5, label=f'{name}', alpha=0.8)

    # Mark boundaries
    ax.axvspan(0, t[19], color='gray', alpha=0.2)
    ax.axvspan(t[180], 1, color='gray', alpha=0.2)

    ax.set_title(f"{name} (boundary {data['ratio']:.1f}x worse)")
    ax.set_xlabel('t')
    ax.set_ylabel('dy/dt')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Error comparison in 4th subplot
ax = axes[1, 1]
for name, data in sorted_methods[:4]:
    errors = np.abs(data['dy_est'] - dy_true)
    ax.semilogy(t, errors, label=f"{name} ({data['ratio']:.1f}x)", linewidth=1.5, alpha=0.7)

ax.axvline(t[19], color='gray', linestyle='--', alpha=0.5)
ax.axvline(t[180], color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('t')
ax.set_ylabel('|Error|')
ax.set_title('Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pynumdiff_actual_boundaries.png', dpi=150)
print("Saved: pynumdiff_actual_boundaries.png")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

bad_methods = [name for name, data in results.items() if data['ratio'] > 10]
good_methods = [name for name, data in results.items() if data['ratio'] < 2]

if bad_methods:
    print(f"\nMethods with terrible boundaries (>10x worse): {', '.join(bad_methods)}")

if good_methods:
    print(f"\nMethods with good boundaries (<2x ratio): {', '.join(good_methods)}")

print(f"\nWorst boundary handler: {sorted_methods[0][0]} ({sorted_methods[0][1]['ratio']:.1f}x)")
if sorted_methods:
    best = sorted_methods[-1]
    print(f"Best boundary handler: {best[0]} ({best[1]['ratio']:.1f}x)")