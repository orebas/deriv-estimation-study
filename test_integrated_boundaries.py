"""
Test boundary handling in our integrated Python methods
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('python')
from python_methods_integrated import compute_derivative

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
print("TESTING BOUNDARY HANDLING IN OUR INTEGRATED METHODS")
print("=" * 80)
print()

# Define regions
inner_80 = slice(20, 181)  # Middle 80%
boundary_left = slice(0, 20)
boundary_right = slice(181, 201)

# Test methods that support first derivative
methods_to_test = [
    'pynumdiff_savitzky_golay',
    'pynumdiff_finite_difference',
    'pynumdiff_smoothed_finite_difference',
    'pynumdiff_iterative_fitting',
    'pynumdiff_spline',
    'pynumdiff_trend_filtered',
    'pynumdiff_total_variation',
    'pynumdiff_kalman',
    'pynumdiff_spectral',
    'scipy_savgol',
    'findiff_basic',
    'derivative_polynomial',
    'numdifftools_gradient',
    'cs_derivative'
]

results = {}

print("Method                        | Interior RMSE | Boundary RMSE | Ratio")
print("-" * 70)

for method in methods_to_test:
    try:
        # Get derivative estimate
        result = compute_derivative(t, y_noisy, method, max_order=1, alpha=1e-2)

        if 'predictions' not in result or 1 not in result['predictions']:
            print(f"{method:30} | No first derivative")
            continue

        dy_est = result['predictions'][1]

        # Calculate errors
        error_interior = np.sqrt(np.mean((dy_est[inner_80] - dy_true[inner_80])**2))
        error_left = np.sqrt(np.mean((dy_est[boundary_left] - dy_true[boundary_left])**2))
        error_right = np.sqrt(np.mean((dy_est[boundary_right] - dy_true[boundary_right])**2))
        error_boundary = (error_left + error_right) / 2

        ratio = error_boundary / error_interior if error_interior > 0 else float('inf')

        print(f"{method:30} | {error_interior:13.4f} | {error_boundary:13.4f} | {ratio:5.1f}x")

        results[method] = {
            'dy_est': dy_est,
            'error_interior': error_interior,
            'error_boundary': error_boundary,
            'error_left': error_left,
            'error_right': error_right,
            'ratio': ratio
        }

    except Exception as e:
        print(f"{method:30} | Failed: {str(e)[:25]}")

# Visualize results
print("\nGenerating visualization...")

# Sort by ratio
sorted_methods = sorted(results.items(), key=lambda x: x[1]['ratio'], reverse=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot worst 4 methods
for idx, (name, data) in enumerate(sorted_methods[:4]):
    ax = axes.flatten()[idx]
    ax.plot(t, dy_true, 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(t, data['dy_est'], 'r-', linewidth=1.5, label=f'{name.replace("pynumdiff_", "")}', alpha=0.8)

    # Mark boundaries
    ax.axvspan(0, t[19], color='gray', alpha=0.2, label='Boundary' if idx == 0 else '')
    ax.axvspan(t[180], 1, color='gray', alpha=0.2)

    ax.set_title(f"{name.replace('pynumdiff_', '')} (boundary {data['ratio']:.1f}x worse)")
    ax.set_xlabel('t')
    ax.set_ylabel('dy/dt')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

# Error comparison in 5th subplot
ax = axes[1, 2]
n_show = min(6, len(sorted_methods))
for name, data in sorted_methods[:n_show]:
    errors = np.abs(data['dy_est'] - dy_true)
    label = name.replace('pynumdiff_', '').replace('_', ' ')
    ax.semilogy(t, errors, label=f"{label} ({data['ratio']:.1f}x)",
                linewidth=1.5, alpha=0.7)

ax.axvline(t[19], color='gray', linestyle='--', alpha=0.5)
ax.axvline(t[180], color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('t')
ax.set_ylabel('|Error|')
ax.set_title('Error Distribution (6 worst methods)')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# Summary statistics in 6th subplot
ax = axes[1, 1]
ax.axis('off')

# Create summary text
summary_lines = ["SUMMARY", "="*30]

bad_methods = [(name.replace('pynumdiff_', ''), data['ratio'])
               for name, data in results.items() if data['ratio'] > 10]
good_methods = [(name.replace('pynumdiff_', ''), data['ratio'])
                for name, data in results.items() if data['ratio'] < 2]

if bad_methods:
    summary_lines.append("\nTerrible boundaries (>10x):")
    for name, ratio in bad_methods[:3]:
        summary_lines.append(f"  • {name}: {ratio:.1f}x")

if good_methods:
    summary_lines.append("\nGood boundaries (<2x):")
    for name, ratio in good_methods[:3]:
        summary_lines.append(f"  • {name}: {ratio:.1f}x")

if sorted_methods:
    worst = sorted_methods[0]
    best = sorted_methods[-1]
    summary_lines.append(f"\nWorst: {worst[0].replace('pynumdiff_', '')} ({worst[1]['ratio']:.1f}x)")
    summary_lines.append(f"Best: {best[0].replace('pynumdiff_', '')} ({best[1]['ratio']:.1f}x)")

ax.text(0.5, 0.5, '\n'.join(summary_lines), transform=ax.transAxes,
        fontsize=10, ha='center', va='center', family='monospace')

plt.tight_layout()
plt.savefig('integrated_methods_boundaries.png', dpi=150)
print("Saved: integrated_methods_boundaries.png")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Categorize methods
terrible = [m for m, d in results.items() if d['ratio'] > 20]
bad = [m for m, d in results.items() if 10 < d['ratio'] <= 20]
moderate = [m for m, d in results.items() if 2 <= d['ratio'] <= 10]
good = [m for m, d in results.items() if d['ratio'] < 2]

print(f"\nTerrible (>20x worse at boundaries): {len(terrible)} methods")
if terrible:
    for m in terrible[:3]:
        print(f"  - {m}: {results[m]['ratio']:.1f}x")

print(f"\nBad (10-20x worse): {len(bad)} methods")
if bad:
    for m in bad[:3]:
        print(f"  - {m}: {results[m]['ratio']:.1f}x")

print(f"\nModerate (2-10x worse): {len(moderate)} methods")
if moderate:
    for m in moderate[:3]:
        print(f"  - {m}: {results[m]['ratio']:.1f}x")

print(f"\nGood (<2x worse): {len(good)} methods")
if good:
    for m in good:
        print(f"  - {m}: {results[m]['ratio']:.1f}x")