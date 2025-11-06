"""
Comprehensive test of ALL PyNumDiff methods with correct signatures
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
print("COMPLETE PYNUMDIFF COLLECTION - ALL ALGORITHMS")
print("=" * 80)

# Define ALL methods with their ACTUAL signatures
methods = {
    # ========== BASIC FINITE DIFFERENCES ==========
    # These return just the derivative array (no smoothing)
    'first_order': lambda: pynumdiff.first_order(y_noisy, dt),
    'second_order': lambda: pynumdiff.second_order(y_noisy, dt),
    'fourth_order': lambda: pynumdiff.fourth_order(y_noisy, dt),

    # ========== SMOOTH FINITE DIFFERENCE ==========
    # These return (x_smooth, dx_dt) tuples
    'butterdiff': lambda: sfd.butterdiff(y_noisy, dt, filter_order=2, cutoff_freq=0.2),
    'gaussiandiff': lambda: sfd.gaussiandiff(y_noisy, dt, window_size=7, num_iterations=1),
    'friedrichsdiff': lambda: sfd.friedrichsdiff(y_noisy, dt, window_size=7, num_iterations=1),
    'meandiff': lambda: sfd.meandiff(y_noisy, dt, window_size=7, num_iterations=1),
    'mediandiff': lambda: sfd.mediandiff(y_noisy, dt, window_size=7, num_iterations=1),

    # ========== POLYNOMIAL ==========
    'savgoldiff': lambda: savgoldiff(y_noisy, dt, degree=3, window_size=7, smoothing_win=None),
    'splinediff': lambda: splinediff(y_noisy, dt, s=1e-3, d=3),
    'polydiff': lambda: polydiff(y_noisy, dt, degree=5, window_size=7, step_size=1, kernel='linear'),

    # ========== BASIS FUNCTIONS ==========
    'spectraldiff': lambda: spectraldiff(y_noisy, dt, cutoff_freq=0.2, tau=0),
    'rbfdiff': lambda: rbfdiff(t, y_noisy, sigma=0.1, lmbd=1e-3),
    'lineardiff': lambda: pynumdiff.lineardiff(y_noisy, dt, params={'diff_order': 1}),

    # ========== TOTAL VARIATION ==========
    'tvrdiff': lambda: tvr.tvrdiff(y_noisy, dt, alph=0.01, itern=20),
    'tv_velocity': lambda: tvr.velocity(y_noisy, dt, alpha=1e-2, gamma=1e-3),
    'tv_acceleration': lambda: tvr.acceleration(y_noisy, dt, alpha=1e-2, gamma=1e-3),
    'tv_iterative': lambda: tvr.iterative_velocity(y_noisy, t, alpha=1e-2, gamma=1e-3, tol=1e-6),

    # ========== KALMAN ==========
    'kalman_const_vel': lambda: kalman_smooth.constant_velocity(y_noisy, dt, r=1e-3, q=1e-5, forwardbackward='forward-backward'),
    'kalman_const_accel': lambda: kalman_smooth.constant_acceleration(y_noisy, dt, r=1e-3, q=1e-5, forwardbackward='forward-backward'),
    'kalman_rts': lambda: kalman_smooth.rts_smooth(y_noisy, dt, r=1e-3, q=1e-5),
}

results = {}
categories = {
    'Basic FD': ['first_order', 'second_order', 'fourth_order'],
    'Smooth FD': ['butterdiff', 'gaussiandiff', 'friedrichsdiff', 'meandiff', 'mediandiff'],
    'Polynomial': ['savgoldiff', 'splinediff', 'polydiff'],
    'Basis': ['spectraldiff', 'rbfdiff', 'lineardiff'],
    'Total Var': ['tvrdiff', 'tv_velocity', 'tv_acceleration', 'tv_iterative'],
    'Kalman': ['kalman_const_vel', 'kalman_const_accel', 'kalman_rts']
}

print("\nTesting all methods:")
print("-" * 80)
print("Method               | Type    | RMSE    | Max Err | Boundary | Status")
print("-" * 80)

for name, method in methods.items():
    try:
        # Get result
        result = method()

        # Determine what type of result we got
        if isinstance(result, tuple):
            if len(result) == 2:
                x_smooth, dx_dt = result
                result_type = "tuple"
            elif len(result) == 3:
                # Some methods return (x_smooth, dx_dt, d2x_dt2)
                x_smooth, dx_dt, d2x_dt2 = result
                result_type = "triple"
            else:
                dx_dt = result[0]
                result_type = f"tuple{len(result)}"
        else:
            # Direct derivative array
            dx_dt = result
            x_smooth = None
            result_type = "array"

        # Handle special formats
        if hasattr(dx_dt, 'flatten'):
            dx_dt = dx_dt.flatten()

        # Skip 2nd derivative methods for now
        if name in ['tv_acceleration', 'kalman_const_accel'] and result_type == "tuple":
            # These methods can return velocity too
            if len(result) >= 2 and result[1] is not None:
                dx_dt = result[1]
            else:
                print(f"{name:20} | {result_type:7} | Returns 2nd derivative - skipping")
                continue

        # Check size
        if len(dx_dt) != len(t):
            # For finite differences that return shorter arrays
            if len(dx_dt) == len(t) - 1:
                # Forward/backward difference - pad with last value
                dx_dt = np.append(dx_dt, dx_dt[-1])
            elif len(dx_dt) == len(t) - 2:
                # Central difference - pad both ends
                dx_dt = np.concatenate([[dx_dt[0]], dx_dt, [dx_dt[-1]]])
            else:
                print(f"{name:20} | {result_type:7} | Size mismatch: {len(dx_dt)} vs {len(t)}")
                continue

        # Calculate metrics
        rmse = np.sqrt(np.mean((dx_dt - dy_true)**2))
        max_err = np.max(np.abs(dx_dt - dy_true))

        # Boundary analysis (10% on each side)
        inner = slice(20, 181)
        boundary = np.concatenate([np.arange(20), np.arange(181, 201)])
        rmse_inner = np.sqrt(np.mean((dx_dt[inner] - dy_true[inner])**2))
        rmse_boundary = np.sqrt(np.mean((dx_dt[boundary] - dy_true[boundary])**2))
        ratio = rmse_boundary / rmse_inner if rmse_inner > 0 else np.inf

        results[name] = {
            'dx_dt': dx_dt,
            'x_smooth': x_smooth,
            'rmse': rmse,
            'max_err': max_err,
            'ratio': ratio,
            'type': result_type
        }

        # Status
        if rmse < 0.05:
            status = "✓✓ Excel"
        elif rmse < 0.1:
            status = "✓ Good"
        elif rmse < 0.5:
            status = "~ OK"
        else:
            status = "✗ Poor"

        print(f"{name:20} | {result_type:7} | {rmse:7.4f} | {max_err:7.4f} | {ratio:8.1f}x | {status}")

    except Exception as e:
        error_msg = str(e)[:35]
        print(f"{name:20} | failed  | ERROR: {error_msg}")

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

        for method_name, rmse in cat_results:
            if rmse < 0.05:
                symbol = "✓✓"
            elif rmse < 0.1:
                symbol = "✓"
            elif rmse < 0.5:
                symbol = "~"
            else:
                symbol = "✗"
            print(f"  {symbol} {method_name}: {rmse:.4f}")

        print(f"  Best: {cat_results[0][0]} (RMSE={cat_results[0][1]:.4f})")

# Overall statistics
total_tested = len(results)
total_attempted = len(methods)
excellent = sum(1 for r in results.values() if r['rmse'] < 0.05)
good = sum(1 for r in results.values() if 0.05 <= r['rmse'] < 0.1)
ok = sum(1 for r in results.values() if 0.1 <= r['rmse'] < 0.5)
poor = sum(1 for r in results.values() if r['rmse'] >= 0.5)

print("\n" + "=" * 80)
print("PYNUMDIFF COMPLETE COLLECTION STATISTICS")
print("=" * 80)
print(f"Total algorithms in PyNumDiff: {total_attempted}")
print(f"Successfully tested: {total_tested}")
print(f"Failed to run: {total_attempted - total_tested}")
print(f"\nPerformance breakdown:")
print(f"  Excellent (<0.05 RMSE): {excellent} methods")
print(f"  Good (0.05-0.1 RMSE): {good} methods")
print(f"  OK (0.1-0.5 RMSE): {ok} methods")
print(f"  Poor (>0.5 RMSE): {poor} methods")

# Top methods overall
sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
print(f"\n🏆 TOP 10 METHODS:")
for i, (name, data) in enumerate(sorted_results[:10], 1):
    cat = [c for c, m in categories.items() if name in m][0]
    print(f"  {i:2}. {name:20} RMSE={data['rmse']:.4f} ({cat})")

# Best boundary handling
best_boundary = sorted(results.items(), key=lambda x: x[1]['ratio'])[:5]
print(f"\n🎯 BEST BOUNDARY HANDLING (lowest ratio = best):")
for name, data in best_boundary:
    print(f"  {name}: {data['ratio']:.1f}x ratio")

# Success rate
success_rate = 100 * (excellent + good) / total_tested if total_tested > 0 else 0
print(f"\n📊 SUCCESS RATE: {success_rate:.1f}% of working methods are good")
print(f"PyNumDiff has {excellent+good}/{total_tested} effective methods for first derivatives")

# Method types analysis
print(f"\n📈 RESULT TYPES:")
type_counts = {}
for data in results.values():
    rt = data['type']
    type_counts[rt] = type_counts.get(rt, 0) + 1
for rt, count in sorted(type_counts.items()):
    print(f"  {rt}: {count} methods")

# Final verdict
print("\n" + "=" * 80)
print("FINAL VERDICT ON PYNUMDIFF")
print("=" * 80)

if excellent + good > 5:
    print("✅ PyNumDiff is GOOD! It provides multiple effective methods for numerical differentiation.")
    print(f"   - {excellent} excellent methods (RMSE < 0.05)")
    print(f"   - {good} good methods (RMSE < 0.1)")
    print("   Recommended: " + ", ".join([n for n, d in sorted_results[:3]]))
elif excellent + good > 0:
    print("⚠️ PyNumDiff has SOME useful methods, but many don't work well.")
    print(f"   Only {excellent + good} out of {total_tested} methods are effective.")
    if sorted_results:
        print("   Best option: " + sorted_results[0][0])
else:
    print("❌ PyNumDiff struggles with this test case - no methods achieved good performance.")

# Visualization
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Plot 1: Top 3 overall
ax = axes[0, 0]
ax.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.7)
for name, data in sorted_results[:3]:
    ax.plot(t, data['dx_dt'], linewidth=1.5, label=f"{name} ({data['rmse']:.3f})", alpha=0.8)
ax.set_title('Top 3 Methods Overall')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2-7: Best from each category
for idx, (cat_name, method_list) in enumerate(list(categories.items())[:6]):
    ax = axes.flatten()[idx+1]
    ax.plot(t, dy_true, 'k-', linewidth=2, label='True', alpha=0.5)

    cat_results = [(m, results[m]) for m in method_list if m in results]
    if cat_results:
        cat_results.sort(key=lambda x: x[1]['rmse'])
        name, data = cat_results[0]
        ax.plot(t, data['dx_dt'], 'b-', linewidth=1.5,
                label=f"{name} ({data['rmse']:.3f})")
        ax.set_title(f'Best {cat_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

# Plot 8: Performance distribution bar chart
ax = axes[2, 0]
performance_data = [excellent, good, ok, poor]
colors = ['darkgreen', 'lightgreen', 'orange', 'red']
labels = ['Excellent\n(<0.05)', 'Good\n(0.05-0.1)', 'OK\n(0.1-0.5)', 'Poor\n(>0.5)']
bars = ax.bar(labels, performance_data, color=colors)
ax.set_title('Performance Distribution')
ax.set_ylabel('Number of Methods')
for bar, val in zip(bars, performance_data):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha='center')

# Plot 9: Boundary effects comparison
ax = axes[2, 1]
boundary_data = sorted([(n, d['ratio']) for n, d in results.items() if d['ratio'] < 20])[:10]
if boundary_data:
    names, ratios = zip(*boundary_data)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, ratios)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Boundary/Interior Error Ratio')
    ax.set_title('Boundary Handling (lower is better)')
    ax.set_xlim(0, max(ratios) * 1.1)

# Plot 10: Error heatmap for top methods
ax = axes[2, 2]
if len(sorted_results) >= 5:
    top5_errors = []
    top5_names = []
    for name, data in sorted_results[:5]:
        errors = np.abs(data['dx_dt'] - dy_true)
        top5_errors.append(errors)
        top5_names.append(name)

    error_matrix = np.array(top5_errors)
    im = ax.imshow(error_matrix[:, ::4], aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_yticks(range(5))
    ax.set_yticklabels(top5_names, fontsize=8)
    ax.set_xlabel('Position (downsampled)')
    ax.set_title('Error Patterns (Top 5)')
    plt.colorbar(im, ax=ax)

# Plot 11: RMSE comparison across categories
ax = axes[2, 3]
cat_best = []
cat_labels = []
for cat_name, method_list in categories.items():
    cat_results = [(m, results[m]['rmse']) for m in method_list if m in results]
    if cat_results:
        best_rmse = min(r for _, r in cat_results)
        cat_best.append(best_rmse)
        cat_labels.append(cat_name)

if cat_best:
    colors_cat = ['green' if r < 0.1 else 'orange' if r < 0.5 else 'red' for r in cat_best]
    bars = ax.bar(range(len(cat_best)), cat_best, color=colors_cat)
    ax.set_xticks(range(len(cat_best)))
    ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Best RMSE in Category')
    ax.set_title('Best Performance by Category')
    ax.axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Excellent')
    ax.axhline(0.1, color='orange', linestyle='--', alpha=0.5, label='Good')
    ax.legend(fontsize=7)

plt.suptitle('PyNumDiff Complete Collection Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('pynumdiff_complete_collection.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved: pynumdiff_complete_collection.png")