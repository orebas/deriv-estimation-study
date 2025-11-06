"""
Comprehensive test of ALL PyNumDiff methods to see what actually works
"""

import numpy as np
import matplotlib.pyplot as plt
import pynumdiff

# Generate test signals with different characteristics
n = 201
t = np.linspace(0, 1, n)
dt = t[1] - t[0]

# Test case 1: Smooth sine wave (easy)
y1_true = np.sin(2 * np.pi * t)
dy1_true = 2 * np.pi * np.cos(2 * np.pi * t)

# Test case 2: Polynomial (should be perfect for some methods)
y2_true = 3*t**3 - 2*t**2 + t
dy2_true = 9*t**2 - 4*t + 1

# Add different noise levels
np.random.seed(42)
noise_low = 1e-4
noise_med = 1e-3
noise_high = 1e-2

y1_low = y1_true + noise_low * np.random.randn(n)
y1_med = y1_true + noise_med * np.random.randn(n)
y1_high = y1_true + noise_high * np.random.randn(n)

print("=" * 80)
print("COMPREHENSIVE PYNUMDIFF TEST")
print("=" * 80)

# All PyNumDiff methods we can test
methods = {
    # Finite differences
    'finite_diff': lambda y, dt: pynumdiff.finite_difference.first_order(y, dt),

    # Smoothing + finite difference
    'smooth_finite_diff': lambda y, dt: pynumdiff.smooth_finite_difference.butterdiff(
        y, dt, n=2, cutoff=10),

    # Total variation regularization
    'total_variation': lambda y, dt: pynumdiff.total_variation_regularization.velocity(
        y, dt, alpha=0.01),

    # Trend filtered
    'trend_filtered': lambda y, dt: pynumdiff.optimize.trend_filtered.iterative_velocity(
        y, t, alpha=1e-2, gamma=1e-3)[0],

    # Spectral methods
    'spectral': lambda y, dt: pynumdiff.linear_model.spectral.dxdt_spectral(
        y, t, cutoff_freq=10),

    # Savitzky-Golay polynomial
    'savitzky_golay': lambda y, dt: pynumdiff.linear_model.savitzky_golay.savgoldiff(
        y, order=3, left=5, right=5, iwindow=1)[0],

    # Friedrichs (smoothing kernel)
    'friedrichs': lambda y, dt: pynumdiff.linear_model.friedrichs.dxdt_friedrichs(
        y, dt, k=5),

    # Spline
    'spline': lambda y, dt: pynumdiff.smooth_finite_difference.splinediff(
        y, t, s=1e-4)[0],

    # ANN (Artificial Neural Network smoothing)
    'ann': lambda y, dt: pynumdiff.smooth_finite_difference.anndiff(
        y, t, alpha=0.01)[0],
}

# Test each method on different scenarios
results = {}

print("\nTesting all methods on medium noise (1e-3)...")
print("-" * 60)
print("Method               | RMSE    | Max Error | Boundary Ratio")
print("-" * 60)

for name, method in methods.items():
    try:
        # Get derivative
        dy_est = method(y1_med, dt)

        # Calculate overall RMSE
        rmse = np.sqrt(np.mean((dy_est - dy1_true)**2))
        max_err = np.max(np.abs(dy_est - dy1_true))

        # Check boundary vs interior
        inner = slice(20, 181)
        boundary = np.concatenate([np.arange(20), np.arange(181, 201)])

        rmse_inner = np.sqrt(np.mean((dy_est[inner] - dy1_true[inner])**2))
        rmse_boundary = np.sqrt(np.mean((dy_est[boundary] - dy1_true[boundary])**2))

        ratio = rmse_boundary / rmse_inner if rmse_inner > 0 else np.inf

        results[name] = {
            'dy': dy_est,
            'rmse': rmse,
            'max_err': max_err,
            'ratio': ratio,
            'rmse_inner': rmse_inner,
            'rmse_boundary': rmse_boundary
        }

        print(f"{name:20} | {rmse:7.4f} | {max_err:9.4f} | {ratio:6.1f}x")

    except Exception as e:
        print(f"{name:20} | FAILED: {str(e)[:30]}")

# Create comprehensive visualization
print("\nGenerating comprehensive visualization...")

fig = plt.figure(figsize=(16, 10))

# Sort methods by performance
sorted_by_rmse = sorted(results.items(), key=lambda x: x[1]['rmse'])
sorted_by_boundary = sorted(results.items(), key=lambda x: x[1]['ratio'], reverse=True)

# Plot 1: Best overall performers
ax1 = plt.subplot(3, 3, 1)
ax1.plot(t, dy1_true, 'k-', linewidth=2, label='True', alpha=0.6)
for name, data in sorted_by_rmse[:3]:
    ax1.plot(t, data['dy'], linewidth=1.5, label=name, alpha=0.8)
ax1.set_title('Best Overall (by RMSE)')
ax1.set_xlabel('t')
ax1.set_ylabel('dy/dt')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Worst boundary effects
ax2 = plt.subplot(3, 3, 2)
ax2.plot(t, dy1_true, 'k-', linewidth=2, label='True', alpha=0.6)
for name, data in sorted_by_boundary[:3]:
    ax2.plot(t, data['dy'], linewidth=1.5, label=f"{name} ({data['ratio']:.1f}x)", alpha=0.8)
ax2.axvspan(0, 0.1, color='red', alpha=0.1)
ax2.axvspan(0.9, 1, color='red', alpha=0.1)
ax2.set_title('Worst Boundary Effects')
ax2.set_xlabel('t')
ax2.set_ylabel('dy/dt')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Error distribution
ax3 = plt.subplot(3, 3, 3)
for name, data in sorted_by_rmse[:5]:
    errors = np.abs(data['dy'] - dy1_true)
    ax3.semilogy(t, errors, linewidth=1.5, label=name, alpha=0.7)
ax3.axvline(0.1, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(0.9, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('Error Distribution (Log Scale)')
ax3.set_xlabel('t')
ax3.set_ylabel('|Error|')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Performance vs noise level
ax4 = plt.subplot(3, 3, 4)
noise_levels = [noise_low, noise_med, noise_high]
noise_labels = ['1e-4', '1e-3', '1e-2']
selected_methods = ['savitzky_golay', 'total_variation', 'spectral', 'finite_diff']

for method_name in selected_methods:
    if method_name in methods:
        rmses = []
        for noise in noise_levels:
            y_noisy = y1_true + noise * np.random.randn(n)
            try:
                dy = methods[method_name](y_noisy, dt)
                rmse = np.sqrt(np.mean((dy - dy1_true)**2))
                rmses.append(rmse)
            except:
                rmses.append(np.nan)
        ax4.semilogy(noise_labels, rmses, 'o-', label=method_name, linewidth=2, markersize=8)

ax4.set_title('Performance vs Noise Level')
ax4.set_xlabel('Noise Level')
ax4.set_ylabel('RMSE (log scale)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Zoomed view of best method
ax5 = plt.subplot(3, 3, 5)
best_name, best_data = sorted_by_rmse[0]
zoom_range = slice(80, 120)
ax5.plot(t[zoom_range], dy1_true[zoom_range], 'k-', linewidth=3,
         label='True', marker='o', markersize=4)
ax5.plot(t[zoom_range], best_data['dy'][zoom_range], 'g-', linewidth=2,
         label=f'Best: {best_name}', marker='s', markersize=4)
ax5.plot(t[zoom_range], y1_med[zoom_range]*20 - 5, 'gray', alpha=0.3,
         label='Noisy data (scaled)', marker='.', markersize=2)
ax5.set_title(f'Zoomed: Best Method ({best_name})')
ax5.set_xlabel('t')
ax5.set_ylabel('dy/dt')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Test on polynomial (should be exact for some methods)
ax6 = plt.subplot(3, 3, 6)
poly_results = {}
for name in ['savitzky_golay', 'spline', 'spectral']:
    if name in methods:
        try:
            dy_poly = methods[name](y2_true, dt)  # No noise!
            error = np.sqrt(np.mean((dy_poly - dy2_true)**2))
            poly_results[name] = error
            ax6.plot(t, dy_poly - dy2_true, linewidth=1.5,
                    label=f"{name} (err={error:.2e})", alpha=0.8)
        except:
            pass

ax6.set_title('Polynomial Test (No Noise)')
ax6.set_xlabel('t')
ax6.set_ylabel('Error in derivative')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Plot 7-9: Individual method showcases
showcases = ['savitzky_golay', 'total_variation', 'spectral']
for idx, method_name in enumerate(showcases):
    ax = plt.subplot(3, 3, 7 + idx)
    if method_name in results:
        data = results[method_name]
        ax.plot(t, dy1_true, 'k-', linewidth=2, label='True', alpha=0.6)
        ax.plot(t, data['dy'], 'b-', linewidth=1.5, label=method_name)

        # Show error regions
        error = np.abs(data['dy'] - dy1_true)
        ax.fill_between(t, data['dy'] - error, data['dy'] + error,
                        alpha=0.2, color='red', label='Error band')

        ax.set_title(f"{method_name}\nRMSE={data['rmse']:.4f}, Boundary={data['ratio']:.1f}x")
        ax.set_xlabel('t')
        ax.set_ylabel('dy/dt')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle('PyNumDiff Comprehensive Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('pynumdiff_comprehensive.png', dpi=150, bbox_inches='tight')
print("Saved: pynumdiff_comprehensive.png")

# Final summary
print("\n" + "=" * 80)
print("VERDICT ON PYNUMDIFF")
print("=" * 80)

good_methods = [m for m, d in results.items() if d['rmse'] < 0.1]
ok_methods = [m for m, d in results.items() if 0.1 <= d['rmse'] < 0.5]
bad_methods = [m for m, d in results.items() if d['rmse'] >= 0.5]

print(f"\n✓ GOOD methods (RMSE < 0.1): {', '.join(good_methods) if good_methods else 'None'}")
print(f"~ OK methods (0.1 ≤ RMSE < 0.5): {', '.join(ok_methods) if ok_methods else 'None'}")
print(f"✗ BAD methods (RMSE ≥ 0.5): {', '.join(bad_methods) if bad_methods else 'None'}")

good_boundary = [m for m, d in results.items() if d['ratio'] < 2]
bad_boundary = [m for m, d in results.items() if d['ratio'] > 10]

print(f"\nGood boundary handling (<2x): {', '.join(good_boundary) if good_boundary else 'None'}")
print(f"Terrible boundaries (>10x): {', '.join(bad_boundary) if bad_boundary else 'None'}")

print("\nCONCLUSION:")
if good_methods:
    print(f"PyNumDiff has {len(good_methods)} good methods for first derivatives with medium noise")
    print(f"Best performer: {sorted_by_rmse[0][0]} (RMSE={sorted_by_rmse[0][1]['rmse']:.4f})")
else:
    print("PyNumDiff struggles with this noise level")

if bad_boundary:
    print(f"WARNING: {len(bad_boundary)} methods have severe boundary problems!")