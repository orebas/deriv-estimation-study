"""
Test PyNumDiff properly using the correct API from our existing implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import pynumdiff
from pynumdiff import smooth_finite_difference as sfd
from pynumdiff import total_variation_regularization as tvr
from pynumdiff.polynomial_fit import splinediff, savgoldiff
from pynumdiff.basis_fit import spectraldiff

# Generate test signals
n = 201
t = np.linspace(0, 1, n)
dt = np.mean(np.diff(t))

# Smooth sine wave
y_true = np.sin(2 * np.pi * t)
dy_true = 2 * np.pi * np.cos(2 * np.pi * t)

# Add noise
np.random.seed(42)
noise_levels = {'low': 1e-4, 'medium': 1e-3, 'high': 1e-2}
y_noisy = {level: y_true + noise * np.random.randn(n)
           for level, noise in noise_levels.items()}

print("=" * 80)
print("PYNUMDIFF: PROPER API TEST (Based on Our Implementation)")
print("=" * 80)

# Test with medium noise
test_y = y_noisy['medium']

# Define methods with correct API (all return (x_smooth, dx_dt))
methods = {
    # Butterworth filter
    'butterdiff': lambda: sfd.butterdiff(test_y, dt, filter_order=2, cutoff_freq=0.2),

    # Spline
    'splinediff': lambda: splinediff(test_y, dt, degree=3, s=1e-3),

    # Gaussian smoothing
    'gaussiandiff': lambda: sfd.gaussiandiff(test_y, dt, sigma=2.0, order=1),

    # Friedrichs
    'friedrichsdiff': lambda: sfd.friedrichsdiff(test_y, dt, k=5, p=1),

    # Savitzky-Golay (special - returns derivatives for different windows)
    'savgoldiff': lambda: savgoldiff(test_y, n=3, degree=3, diff_order=1),

    # Total variation
    'tv_velocity': lambda: tvr.velocity(test_y, dt, alpha=0.01, iterations=20),

    # Spectral
    'spectraldiff': lambda: spectraldiff(test_y, dt, cutoff_freq=0.2, tau=0),

    # Mean difference (if available)
    'meandiff': lambda: sfd.meandiff(test_y, dt, window_size=7),

    # Median difference (if available)
    'mediandiff': lambda: sfd.mediandiff(test_y, dt, window_size=7),
}

results = {}

print("\nTesting PyNumDiff methods with correct API (noise=1e-3):")
print("-" * 70)
print("Method               | RMSE    | Max Error | Interior | Boundary | Ratio")
print("-" * 70)

for name, method in methods.items():
    try:
        # Get result - methods return (x_smooth, dx_dt) tuple
        result = method()

        # Extract derivative (second element)
        if isinstance(result, tuple) and len(result) >= 2:
            x_smooth = result[0]
            dx_dt = result[1]
        else:
            print(f"{name:20} | Unexpected return type: {type(result)}")
            continue

        # Handle special case for savgoldiff which returns different format
        if name == 'savgoldiff':
            # savgoldiff returns derivatives for different window sizes
            # Use the first one (smallest window)
            if len(dx_dt) > 0 and hasattr(dx_dt[0], '__len__'):
                dx_dt = dx_dt[0]

        # Ensure proper shape
        dx_dt = np.asarray(dx_dt).flatten()

        if len(dx_dt) != len(t):
            print(f"{name:20} | Size mismatch: {len(dx_dt)} vs {len(t)}")
            continue

        # Calculate metrics
        rmse = np.sqrt(np.mean((dx_dt - dy_true)**2))
        max_err = np.max(np.abs(dx_dt - dy_true))

        # Interior vs boundary
        inner = slice(20, 181)
        boundary = np.concatenate([np.arange(20), np.arange(181, 201)])

        rmse_inner = np.sqrt(np.mean((dx_dt[inner] - dy_true[inner])**2))
        rmse_boundary = np.sqrt(np.mean((dx_dt[boundary] - dy_true[boundary])**2))
        ratio = rmse_boundary / rmse_inner if rmse_inner > 0 else np.inf

        results[name] = {
            'x_smooth': x_smooth,
            'dx_dt': dx_dt,
            'rmse': rmse,
            'max_err': max_err,
            'rmse_inner': rmse_inner,
            'rmse_boundary': rmse_boundary,
            'ratio': ratio
        }

        status = "✓✓" if rmse < 0.05 else ("✓" if rmse < 0.1 else ("~" if rmse < 0.5 else "✗"))
        print(f"{name:20} | {rmse:7.4f} | {max_err:9.4f} | {rmse_inner:8.4f} | {rmse_boundary:8.4f} | {ratio:5.1f}x {status}")

    except Exception as e:
        print(f"{name:20} | FAILED: {str(e)[:40]}")

# Create comprehensive visualization
print("\nGenerating visualization...")

# Sort by performance
sorted_rmse = sorted(results.items(), key=lambda x: x[1]['rmse'])
sorted_boundary = sorted(results.items(), key=lambda x: x[1]['ratio'], reverse=True)

fig = plt.figure(figsize=(16, 12))

# 1. Overall comparison - show both smoothed signal and derivative
ax1 = plt.subplot(3, 4, 1)
ax1.plot(t, y_true, 'k-', linewidth=3, label='True signal', alpha=0.7)
if sorted_rmse:
    best_name, best_data = sorted_rmse[0]
    ax1.plot(t, best_data['x_smooth'], 'g--', linewidth=2,
             label=f'{best_name} smoothed', alpha=0.8)
ax1.plot(t, test_y, '.', alpha=0.2, markersize=1, color='gray', label='Noisy')
ax1.set_title('Best Method: Signal')
ax1.set_xlabel('t')
ax1.set_ylabel('y')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# 2. Best derivatives
ax2 = plt.subplot(3, 4, 2)
ax2.plot(t, dy_true, 'k-', linewidth=3, label='True derivative', alpha=0.7)
for i, (name, data) in enumerate(sorted_rmse[:3]):
    ax2.plot(t, data['dx_dt'], linewidth=1.5,
             label=f"{name} ({data['rmse']:.3f})", alpha=0.8)
ax2.set_title('Best Methods: Derivatives')
ax2.set_xlabel('t')
ax2.set_ylabel('dy/dt')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# 3. Zoomed comparison
ax3 = plt.subplot(3, 4, 3)
zoom = slice(80, 120)
ax3.plot(t[zoom], dy_true[zoom], 'k-', linewidth=3, label='True', marker='o', markersize=3)
for i, (name, data) in enumerate(sorted_rmse[:3]):
    ax3.plot(t[zoom], data['dx_dt'][zoom], linewidth=1.5,
             label=name, marker='s', markersize=3, alpha=0.8)
ax3.set_title('Zoomed View (Best 3)')
ax3.set_xlabel('t')
ax3.set_ylabel('dy/dt')
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)

# 4. Error distribution
ax4 = plt.subplot(3, 4, 4)
for name, data in sorted_rmse[:5]:
    errors = np.abs(data['dx_dt'] - dy_true)
    ax4.semilogy(t, errors + 1e-10, linewidth=1.5, label=name, alpha=0.7)
ax4.axvline(0.1, color='gray', linestyle='--', alpha=0.5, label='10% boundary')
ax4.axvline(0.9, color='gray', linestyle='--', alpha=0.5)
ax4.set_title('Error Distribution')
ax4.set_xlabel('t')
ax4.set_ylabel('|Error| (log)')
ax4.legend(fontsize=6)
ax4.grid(True, alpha=0.3)

# 5. Boundary problems showcase
ax5 = plt.subplot(3, 4, 5)
ax5.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.7)
for name, data in sorted_boundary[:2]:
    ax5.plot(t, data['dx_dt'], linewidth=1.5,
             label=f"{name} ({data['ratio']:.1f}x)", alpha=0.8)
ax5.axvspan(0, 0.1, color='red', alpha=0.1, label='Boundary')
ax5.axvspan(0.9, 1, color='red', alpha=0.1)
ax5.set_title('Worst Boundary Effects')
ax5.set_xlabel('t')
ax5.set_ylabel('dy/dt')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# 6. Best boundary handling
ax6 = plt.subplot(3, 4, 6)
ax6.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.7)
sorted_boundary_best = sorted(results.items(), key=lambda x: x[1]['ratio'])
for name, data in sorted_boundary_best[:2]:
    ax6.plot(t, data['dx_dt'], linewidth=1.5,
             label=f"{name} ({data['ratio']:.1f}x)", alpha=0.8)
ax6.axvspan(0, 0.1, color='green', alpha=0.1, label='Boundary')
ax6.axvspan(0.9, 1, color='green', alpha=0.1)
ax6.set_title('Best Boundary Handling')
ax6.set_xlabel('t')
ax6.set_ylabel('dy/dt')
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)

# 7. Performance vs noise for selected methods
ax7 = plt.subplot(3, 4, 7)
selected = ['butterdiff', 'savgoldiff', 'tv_velocity', 'spectraldiff']
for method_name in selected:
    if method_name in methods:
        rmses = []
        for level_name, y_test in y_noisy.items():
            test_y = y_test  # Update for lambda
            try:
                result = methods[method_name]()
                if isinstance(result, tuple) and len(result) >= 2:
                    dx_dt = result[1]
                    if method_name == 'savgoldiff' and len(dx_dt) > 0:
                        dx_dt = dx_dt[0]
                    dx_dt = np.asarray(dx_dt).flatten()
                    rmse = np.sqrt(np.mean((dx_dt - dy_true)**2))
                    rmses.append(rmse)
                else:
                    rmses.append(np.nan)
            except:
                rmses.append(np.nan)
        ax7.semilogy(['low', 'medium', 'high'], rmses, 'o-',
                    label=method_name, linewidth=2, markersize=8)
ax7.set_title('Performance vs Noise')
ax7.set_xlabel('Noise Level')
ax7.set_ylabel('RMSE (log)')
ax7.legend(fontsize=7)
ax7.grid(True, alpha=0.3)

# 8. Summary statistics
ax8 = plt.subplot(3, 4, 8)
categories = {
    'Excellent\n(<0.05)': sum(1 for r in results.values() if r['rmse'] < 0.05),
    'Good\n(<0.1)': sum(1 for r in results.values() if 0.05 <= r['rmse'] < 0.1),
    'OK\n(<0.5)': sum(1 for r in results.values() if 0.1 <= r['rmse'] < 0.5),
    'Bad\n(≥0.5)': sum(1 for r in results.values() if r['rmse'] >= 0.5)
}
colors = ['darkgreen', 'lightgreen', 'orange', 'red']
bars = ax8.bar(categories.keys(), categories.values(), color=colors)
ax8.set_title('Performance Distribution')
ax8.set_ylabel('Number of Methods')
for bar, value in zip(bars, categories.values()):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(value), ha='center')

# 9-12. Individual method showcases
showcases = sorted_rmse[:4] if len(sorted_rmse) >= 4 else sorted_rmse
for idx, (name, data) in enumerate(showcases):
    ax = plt.subplot(3, 4, 9 + idx)

    # Show both smoothed and derivative
    ax2 = ax.twinx()
    ax.plot(t, data['x_smooth'], 'b-', linewidth=1.5, label='Smoothed', alpha=0.6)
    ax.plot(t, y_true, 'b--', linewidth=1, label='True signal', alpha=0.4)
    ax2.plot(t, data['dx_dt'], 'r-', linewidth=1.5, label='Derivative')
    ax2.plot(t, dy_true, 'r--', linewidth=1, label='True deriv', alpha=0.6)

    ax.set_title(f"{name}\nRMSE={data['rmse']:.4f}, Boundary={data['ratio']:.1f}x")
    ax.set_xlabel('t')
    ax.set_ylabel('Signal', color='b')
    ax2.set_ylabel('Derivative', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)

plt.suptitle('PyNumDiff: Complete Performance Analysis (Correct API)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('pynumdiff_correct_api.png', dpi=150, bbox_inches='tight')
print("Saved: pynumdiff_correct_api.png")

# Final verdict
print("\n" + "=" * 80)
print("THE VERDICT ON PYNUMDIFF")
print("=" * 80)

excellent = [n for n, r in results.items() if r['rmse'] < 0.05]
good = [n for n, r in results.items() if 0.05 <= r['rmse'] < 0.1]
ok = [n for n, r in results.items() if 0.1 <= r['rmse'] < 0.5]
bad = [n for n, r in results.items() if r['rmse'] >= 0.5]

if excellent:
    print(f"\n✓✓ EXCELLENT (<0.05 RMSE): {', '.join(excellent)}")
if good:
    print(f"✓ GOOD (0.05-0.1 RMSE): {', '.join(good)}")
if ok:
    print(f"~ OK (0.1-0.5 RMSE): {', '.join(ok)}")
if bad:
    print(f"✗ BAD (≥0.5 RMSE): {', '.join(bad)}")

# Boundary analysis
good_boundary = [n for n, r in results.items() if r['ratio'] < 2]
moderate_boundary = [n for n, r in results.items() if 2 <= r['ratio'] < 10]
bad_boundary = [n for n, r in results.items() if r['ratio'] >= 10]

print(f"\nBoundary handling:")
print(f"  Good (<2x worse): {', '.join(good_boundary) if good_boundary else 'None'}")
print(f"  Moderate (2-10x): {', '.join(moderate_boundary) if moderate_boundary else 'None'}")
print(f"  Terrible (>10x): {', '.join(bad_boundary) if bad_boundary else 'None'}")

if sorted_rmse:
    print(f"\n🏆 BEST: {sorted_rmse[0][0]} (RMSE={sorted_rmse[0][1]['rmse']:.4f})")
    if len(sorted_rmse) > 1:
        print(f"🥈 2nd: {sorted_rmse[1][0]} (RMSE={sorted_rmse[1][1]['rmse']:.4f})")
    print(f"👎 WORST: {sorted_rmse[-1][0]} (RMSE={sorted_rmse[-1][1]['rmse']:.4f})")

# Overall assessment
total = len(results)
excellent_good = len(excellent) + len(good)
print(f"\nOVERALL: {excellent_good}/{total} methods work well for first derivatives")
if excellent_good > 0:
    print(f"PyNumDiff IS NOT CRAP! It has {excellent_good} good methods for derivative estimation.")
    print("The key is using the right method with proper parameters.")
else:
    print("PyNumDiff struggles with this particular test case.")