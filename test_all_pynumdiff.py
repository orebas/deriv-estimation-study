"""
Test ALL PyNumDiff methods comprehensively
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
d2y_true = -(2 * np.pi)**2 * np.sin(2 * np.pi * t)

# Add noise
np.random.seed(42)
noise_level = 1e-3
y_noisy = y_true + noise_level * np.random.randn(n)

print("=" * 80)
print("COMPLETE PYNUMDIFF ALGORITHM COLLECTION TEST")
print("=" * 80)

# Define ALL methods with proper API
methods = {
    # ========== FINITE DIFFERENCES ==========
    'first_order': {
        'func': lambda: (y_noisy, pynumdiff.first_order(y_noisy, dt)),
        'category': 'Finite Difference',
        'description': 'Basic first-order finite difference'
    },
    'second_order': {
        'func': lambda: (y_noisy, pynumdiff.second_order(y_noisy, dt)),
        'category': 'Finite Difference',
        'description': 'Second-order finite difference'
    },
    'fourth_order': {
        'func': lambda: (y_noisy, pynumdiff.fourth_order(y_noisy, dt)),
        'category': 'Finite Difference',
        'description': 'Fourth-order finite difference'
    },

    # ========== SMOOTHING + FINITE DIFFERENCE ==========
    'butterdiff': {
        'func': lambda: sfd.butterdiff(y_noisy, dt, filter_order=2, cutoff_freq=0.2),
        'category': 'Smooth FD',
        'description': 'Butterworth filter + finite difference'
    },
    'gaussiandiff': {
        'func': lambda: sfd.gaussiandiff(y_noisy, dt, sigma=2.0, order=1),
        'category': 'Smooth FD',
        'description': 'Gaussian smoothing + finite difference'
    },
    'friedrichsdiff': {
        'func': lambda: sfd.friedrichsdiff(y_noisy, dt, k=5, p=1),
        'category': 'Smooth FD',
        'description': 'Friedrichs mollification'
    },
    'meandiff': {
        'func': lambda: sfd.meandiff(y_noisy, dt, window_size=7),
        'category': 'Smooth FD',
        'description': 'Moving average + finite difference'
    },
    'mediandiff': {
        'func': lambda: sfd.mediandiff(y_noisy, dt, window_size=7),
        'category': 'Smooth FD',
        'description': 'Median filter + finite difference'
    },

    # ========== POLYNOMIAL METHODS ==========
    'polydiff': {
        'func': lambda: (y_noisy, polydiff(y_noisy, t, deg=3, diff_order=1)),
        'category': 'Polynomial',
        'description': 'Global polynomial fit'
    },
    'savgoldiff': {
        'func': lambda: savgoldiff(y_noisy, n=3, degree=3, diff_order=1),
        'category': 'Polynomial',
        'description': 'Savitzky-Golay (local polynomial)'
    },
    'splinediff': {
        'func': lambda: splinediff(y_noisy, dt, degree=3, s=1e-3),
        'category': 'Polynomial',
        'description': 'Cubic spline smoothing'
    },

    # ========== BASIS FUNCTION METHODS ==========
    'spectraldiff': {
        'func': lambda: spectraldiff(y_noisy, dt, cutoff_freq=0.2, tau=0),
        'category': 'Basis',
        'description': 'Spectral (Fourier) differentiation'
    },
    'rbfdiff': {
        'func': lambda: rbfdiff(y_noisy, dt, sigma=0.1, diff_order=1),
        'category': 'Basis',
        'description': 'Radial Basis Function'
    },
    'lineardiff': {
        'func': lambda: (y_noisy, pynumdiff.lineardiff(y_noisy, dt, diff_order=1)),
        'category': 'Basis',
        'description': 'Linear basis functions'
    },

    # ========== TOTAL VARIATION REGULARIZATION ==========
    'tv_velocity': {
        'func': lambda: tvr.velocity(y_noisy, dt, alpha=0.01, iterations=20),
        'category': 'TV Regularization',
        'description': 'TV regularization for 1st derivative'
    },
    'tv_acceleration': {
        'func': lambda: tvr.acceleration(y_noisy, dt, alpha=0.01, iterations=20),
        'category': 'TV Regularization',
        'description': 'TV regularization for 2nd derivative',
        'order': 2
    },
    'tv_jerk': {
        'func': lambda: tvr.jerk(y_noisy, dt, alpha=0.01, iterations=20),
        'category': 'TV Regularization',
        'description': 'TV regularization for 3rd derivative',
        'order': 3
    },
    'tv_smooth_accel': {
        'func': lambda: tvr.smooth_acceleration(y_noisy, dt, alpha=0.01, iterations=20),
        'category': 'TV Regularization',
        'description': 'Smooth TV for 2nd derivative',
        'order': 2
    },
    'tv_iterative': {
        'func': lambda: tvr.iterative_velocity(y_noisy, t, alpha=1e-2, gamma=1e-3),
        'category': 'TV Regularization',
        'description': 'Iterative TV optimization'
    },

    # ========== KALMAN FILTERING ==========
    'kalman_const_vel': {
        'func': lambda: kalman_smooth.constant_velocity(y_noisy, dt, alpha=1.0),
        'category': 'Kalman',
        'description': 'Kalman with constant velocity model'
    },
    'kalman_const_accel': {
        'func': lambda: kalman_smooth.constant_acceleration(y_noisy, dt, alpha=1.0),
        'category': 'Kalman',
        'description': 'Kalman with constant acceleration model'
    },
    'kalman_const_jerk': {
        'func': lambda: kalman_smooth.constant_jerk(y_noisy, dt, alpha=1.0),
        'category': 'Kalman',
        'description': 'Kalman with constant jerk model'
    },
}

results = {}
categories_summary = {}

print("\nTesting all methods (noise=1e-3):")
print("-" * 80)
print("Category        | Method               | RMSE    | Max Err | Boundary | Status")
print("-" * 80)

for name, method_info in methods.items():
    try:
        # Get result
        result = method_info['func']()

        # Extract derivative (handle different return formats)
        if isinstance(result, tuple) and len(result) >= 2:
            x_smooth = result[0]
            dx_dt = result[1]
        else:
            dx_dt = result
            x_smooth = None

        # Handle special cases
        if name == 'savgoldiff' and hasattr(dx_dt, '__len__') and len(dx_dt) > 0:
            if hasattr(dx_dt[0], '__len__'):
                dx_dt = dx_dt[0]

        dx_dt = np.asarray(dx_dt).flatten()

        # Check which derivative order to compare
        target_order = method_info.get('order', 1)
        if target_order == 1:
            true_deriv = dy_true
        elif target_order == 2:
            true_deriv = d2y_true
        else:
            true_deriv = dy_true  # Default to 1st

        if len(dx_dt) != len(t):
            status = "SIZE_ERR"
            rmse = np.nan
        else:
            # Calculate metrics
            rmse = np.sqrt(np.mean((dx_dt - true_deriv)**2))
            max_err = np.max(np.abs(dx_dt - true_deriv))

            # Boundary analysis
            inner = slice(20, 181)
            boundary = np.concatenate([np.arange(20), np.arange(181, 201)])
            rmse_inner = np.sqrt(np.mean((dx_dt[inner] - true_deriv[inner])**2))
            rmse_boundary = np.sqrt(np.mean((dx_dt[boundary] - true_deriv[boundary])**2))
            ratio = rmse_boundary / rmse_inner if rmse_inner > 0 else np.inf

            results[name] = {
                'dx_dt': dx_dt,
                'x_smooth': x_smooth,
                'rmse': rmse,
                'max_err': max_err,
                'ratio': ratio,
                'category': method_info['category'],
                'description': method_info['description'],
                'order': target_order
            }

            # Status indicator
            if rmse < 0.05:
                status = "✓✓ Excellent"
            elif rmse < 0.1:
                status = "✓ Good"
            elif rmse < 0.5:
                status = "~ OK"
            else:
                status = "✗ Poor"

            # Update category summary
            cat = method_info['category']
            if cat not in categories_summary:
                categories_summary[cat] = []
            categories_summary[cat].append((name, rmse))

            print(f"{cat:15} | {name:20} | {rmse:7.4f} | {max_err:7.4f} | {ratio:8.1f}x | {status}")

    except Exception as e:
        print(f"{method_info['category']:15} | {name:20} | FAILED: {str(e)[:30]}")

# Create comprehensive visualization
print("\nGenerating comprehensive visualization...")

# Sort by performance
sorted_rmse = sorted([(k,v) for k,v in results.items()], key=lambda x: x[1]['rmse'])

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 12))

# 1. Best methods by category
ax1 = plt.subplot(3, 4, 1)
for cat, methods_list in categories_summary.items():
    if methods_list:
        best = min(methods_list, key=lambda x: x[1])
        ax1.bar(cat[:8], best[1], label=best[0])
ax1.set_title('Best Method per Category')
ax1.set_xlabel('Category')
ax1.set_ylabel('RMSE')
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim([0, min(0.5, max([min(m, key=lambda x: x[1])[1] for m in categories_summary.values()] + [0.1]) * 1.2)])
ax1.grid(True, alpha=0.3)

# 2. Top 5 overall
ax2 = plt.subplot(3, 4, 2)
ax2.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.7)
for i, (name, data) in enumerate(sorted_rmse[:5]):
    ax2.plot(t, data['dx_dt'], linewidth=1.5,
             label=f"{name} ({data['rmse']:.3f})", alpha=0.8)
ax2.set_title('Top 5 Methods Overall')
ax2.set_xlabel('t')
ax2.set_ylabel('dy/dt')
ax2.legend(fontsize=6)
ax2.grid(True, alpha=0.3)

# 3. Category performance distribution
ax3 = plt.subplot(3, 4, 3)
categories_data = []
categories_names = []
for cat in categories_summary:
    rmses = [r[1] for r in categories_summary[cat]]
    if rmses:
        categories_data.append(rmses)
        categories_names.append(cat[:8])
if categories_data:
    bp = ax3.boxplot(categories_data, labels=categories_names)
    ax3.set_title('Performance Distribution by Category')
    ax3.set_ylabel('RMSE')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim([0, 0.5])
    ax3.grid(True, alpha=0.3)

# 4. Boundary effect analysis
ax4 = plt.subplot(3, 4, 4)
boundary_data = [(name, data['ratio']) for name, data in results.items()]
boundary_data.sort(key=lambda x: x[1])
best_boundary = boundary_data[:5]
worst_boundary = boundary_data[-5:]

names = [b[0][:10] for b in best_boundary] + ['...'] + [b[0][:10] for b in worst_boundary]
ratios = [b[1] for b in best_boundary] + [0] + [b[1] for b in worst_boundary]
colors = ['green']*5 + ['white'] + ['red']*5

bars = ax4.bar(range(len(names)), ratios, color=colors)
ax4.set_xticks(range(len(names)))
ax4.set_xticklabels(names, rotation=45, ha='right')
ax4.set_title('Boundary Effects (Lower is Better)')
ax4.set_ylabel('Boundary/Interior Ratio')
ax4.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Good (<2x)')
ax4.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Poor (>10x)')
ax4.set_ylim([0, min(20, max(ratios)*1.1)])
ax4.legend(fontsize=6)
ax4.grid(True, alpha=0.3)

# 5-8. Category showcases
categories_to_show = list(categories_summary.keys())[:4]
for idx, cat in enumerate(categories_to_show):
    ax = plt.subplot(3, 4, 5 + idx)
    ax.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.5)

    # Get best 2 methods from this category
    cat_methods = categories_summary[cat]
    cat_methods.sort(key=lambda x: x[1])

    for method_name, rmse in cat_methods[:2]:
        if method_name in results:
            ax.plot(t, results[method_name]['dx_dt'], linewidth=1.5,
                   label=f"{method_name} ({rmse:.3f})", alpha=0.8)

    ax.set_title(f'{cat} Methods')
    ax.set_xlabel('t')
    ax.set_ylabel('Derivative')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

# 9-12. Individual showcases of best methods
for idx, (name, data) in enumerate(sorted_rmse[:4]):
    ax = plt.subplot(3, 4, 9 + idx)
    ax.plot(t, dy_true if data['order']==1 else d2y_true, 'k-', linewidth=2,
            label=f'True (order {data["order"]})', alpha=0.7)
    ax.plot(t, data['dx_dt'], 'b-', linewidth=1.5, label=name)
    ax.fill_between(t, dy_true if data['order']==1 else d2y_true,
                     data['dx_dt'], alpha=0.2, color='red')
    ax.set_title(f'{name}\nRMSE={data["rmse"]:.4f}, {data["category"]}')
    ax.set_xlabel('t')
    ax.set_ylabel(f'd{"" if data["order"]==1 else "²"}y/dt{"" if data["order"]==1 else "²"}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Complete PyNumDiff Algorithm Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('pynumdiff_complete_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: pynumdiff_complete_analysis.png")

# Final summary
print("\n" + "=" * 80)
print("SUMMARY BY CATEGORY")
print("=" * 80)

for cat in sorted(categories_summary.keys()):
    methods = categories_summary[cat]
    methods.sort(key=lambda x: x[1])

    print(f"\n{cat}:")
    print("-" * 40)

    excellent = [m for m in methods if m[1] < 0.05]
    good = [m for m in methods if 0.05 <= m[1] < 0.1]
    ok = [m for m in methods if 0.1 <= m[1] < 0.5]
    poor = [m for m in methods if m[1] >= 0.5]

    if excellent:
        print(f"  ✓✓ Excellent: {', '.join([m[0] for m in excellent])}")
    if good:
        print(f"  ✓ Good: {', '.join([m[0] for m in good])}")
    if ok:
        print(f"  ~ OK: {', '.join([m[0] for m in ok])}")
    if poor:
        print(f"  ✗ Poor: {', '.join([m[0] for m in poor])}")

    if methods:
        print(f"  Best: {methods[0][0]} (RMSE={methods[0][1]:.4f})")

# Overall assessment
total_tested = len(results)
excellent_count = sum(1 for r in results.values() if r['rmse'] < 0.05)
good_count = sum(1 for r in results.values() if 0.05 <= r['rmse'] < 0.1)

print("\n" + "=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)
print(f"\nTotal methods tested: {total_tested}")
print(f"Excellent (<0.05 RMSE): {excellent_count}")
print(f"Good (0.05-0.1 RMSE): {good_count}")
print(f"Success rate: {100*(excellent_count+good_count)/total_tested:.1f}%")

if sorted_rmse:
    print(f"\n🏆 TOP 3 METHODS:")
    for i, (name, data) in enumerate(sorted_rmse[:3]):
        print(f"  {i+1}. {name}: RMSE={data['rmse']:.4f} ({data['category']})")
        print(f"     {data['description']}")