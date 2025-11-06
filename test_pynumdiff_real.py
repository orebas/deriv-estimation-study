"""
Test PyNumDiff with correct API - show what it ACTUALLY does
"""

import numpy as np
import matplotlib.pyplot as plt
import pynumdiff

# Generate test signals
n = 201
t = np.linspace(0, 1, n)
dt = t[1] - t[0]

# Smooth sine wave
y_true = np.sin(2 * np.pi * t)
dy_true = 2 * np.pi * np.cos(2 * np.pi * t)

# Add noise
np.random.seed(42)
noise_levels = {'low': 1e-4, 'medium': 1e-3, 'high': 1e-2}
y_noisy = {level: y_true + noise * np.random.randn(n)
           for level, noise in noise_levels.items()}

print("=" * 80)
print("PYNUMDIFF: WHAT ACTUALLY WORKS")
print("=" * 80)

# Test all available methods with medium noise
test_y = y_noisy['medium']

methods = {
    # Basic finite differences
    'first_order': lambda: pynumdiff.first_order(test_y, dt),

    # Smoothing methods
    'savgoldiff': lambda: pynumdiff.savgoldiff(test_y, order=3, left=5, right=5, iwindow=1)[0],
    'gaussiandiff': lambda: pynumdiff.gaussiandiff(test_y, dt, sigma=2, order=1),
    'friedrichsdiff': lambda: pynumdiff.friedrichsdiff(test_y, dt, k=5, p=1),
    'butterdiff': lambda: pynumdiff.butterdiff(test_y, dt, n=2, cutoff=10),
    'splinediff': lambda: pynumdiff.splinediff(test_y, t, s=1e-4)[0],

    # Regularization methods
    'tvrdiff': lambda: pynumdiff.tvrdiff(test_y, dt, alpha=0.01, plot=False),

    # Kalman filter
    'kalman_smooth': lambda: pynumdiff.kalman_smooth(test_y, t, alpha=1, return_data='dxdt'),

    # Linear models
    'polydiff': lambda: pynumdiff.polydiff(test_y, t, order=3, diff_order=1),
    'lineardiff': lambda: pynumdiff.lineardiff(test_y, t, basis='polynomial', n_basis=10, diff_order=1),

    # Spectral
    'spectraldiff': lambda: pynumdiff.spectraldiff(test_y, t, cutoff_freq=10)[0],

    # Mean/Median filters
    'meandiff': lambda: pynumdiff.meandiff(test_y, dt, window=7),
    'mediandiff': lambda: pynumdiff.mediandiff(test_y, dt, window=7),
}

results = {}

print("\nTesting all PyNumDiff methods (noise=1e-3):")
print("-" * 70)
print("Method               | RMSE    | Max Error | Interior | Boundary | Ratio")
print("-" * 70)

for name, method in methods.items():
    try:
        # Get derivative
        dy = method()

        # Handle different return types
        if isinstance(dy, tuple):
            dy = dy[0]
        if hasattr(dy, 'flatten'):
            dy = dy.flatten()

        # Skip if wrong size
        if len(dy) != len(t):
            print(f"{name:20} | Wrong size: {len(dy)} vs {len(t)}")
            continue

        # Calculate metrics
        rmse = np.sqrt(np.mean((dy - dy_true)**2))
        max_err = np.max(np.abs(dy - dy_true))

        # Interior vs boundary
        inner = slice(20, 181)
        boundary = np.concatenate([np.arange(20), np.arange(181, 201)])

        rmse_inner = np.sqrt(np.mean((dy[inner] - dy_true[inner])**2))
        rmse_boundary = np.sqrt(np.mean((dy[boundary] - dy_true[boundary])**2))
        ratio = rmse_boundary / rmse_inner if rmse_inner > 0 else np.inf

        results[name] = {
            'dy': dy,
            'rmse': rmse,
            'max_err': max_err,
            'rmse_inner': rmse_inner,
            'rmse_boundary': rmse_boundary,
            'ratio': ratio
        }

        status = "✓" if rmse < 0.1 else ("~" if rmse < 0.5 else "✗")
        print(f"{name:20} | {rmse:7.4f} | {max_err:9.4f} | {rmse_inner:8.4f} | {rmse_boundary:8.4f} | {ratio:5.1f}x {status}")

    except Exception as e:
        print(f"{name:20} | FAILED: {str(e)[:40]}")

# Create visualization
print("\nGenerating visualization...")

fig = plt.figure(figsize=(16, 12))

# Sort by performance
sorted_rmse = sorted(results.items(), key=lambda x: x[1]['rmse'])
sorted_boundary = sorted(results.items(), key=lambda x: x[1]['ratio'], reverse=True)

# 1. Overall comparison
ax1 = plt.subplot(3, 3, 1)
ax1.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.7)
ax1.plot(t, test_y*20-10, 'gray', alpha=0.2, label='Noisy (scaled)')
colors = plt.cm.tab10(np.linspace(0, 1, min(4, len(sorted_rmse))))
for i, (name, data) in enumerate(sorted_rmse[:4]):
    ax1.plot(t, data['dy'], linewidth=1.5, label=f"{name} ({data['rmse']:.3f})",
             color=colors[i], alpha=0.8)
ax1.set_title('Best Methods Overall')
ax1.set_xlabel('t')
ax1.set_ylabel('dy/dt')
ax1.legend(fontsize=7, loc='best')
ax1.grid(True, alpha=0.3)

# 2. Zoomed comparison
ax2 = plt.subplot(3, 3, 2)
zoom = slice(80, 120)
ax2.plot(t[zoom], dy_true[zoom], 'k-', linewidth=3, label='True', marker='o', markersize=3)
for i, (name, data) in enumerate(sorted_rmse[:3]):
    ax2.plot(t[zoom], data['dy'][zoom], linewidth=1.5,
             label=name, color=colors[i], marker='s', markersize=3, alpha=0.8)
ax2.set_title('Zoomed View (Best 3)')
ax2.set_xlabel('t')
ax2.set_ylabel('dy/dt')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# 3. Error distribution
ax3 = plt.subplot(3, 3, 3)
for name, data in sorted_rmse[:5]:
    errors = np.abs(data['dy'] - dy_true)
    ax3.semilogy(t, errors + 1e-10, linewidth=1.5, label=name, alpha=0.7)
ax3.axvline(0.1, color='gray', linestyle='--', alpha=0.5, label='10% boundary')
ax3.axvline(0.9, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('Error Distribution')
ax3.set_xlabel('t')
ax3.set_ylabel('|Error| (log scale)')
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)

# 4. Boundary problems
ax4 = plt.subplot(3, 3, 4)
ax4.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.7)
for name, data in sorted_boundary[:3]:
    ax4.plot(t, data['dy'], linewidth=1.5,
             label=f"{name} ({data['ratio']:.1f}x)", alpha=0.8)
ax4.axvspan(0, 0.1, color='red', alpha=0.1, label='Boundary')
ax4.axvspan(0.9, 1, color='red', alpha=0.1)
ax4.set_title('Worst Boundary Effects')
ax4.set_xlabel('t')
ax4.set_ylabel('dy/dt')
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# 5. Performance vs noise
ax5 = plt.subplot(3, 3, 5)
selected = ['savgoldiff', 'tvrdiff', 'spectraldiff', 'first_order']
for method_name in selected:
    if method_name in methods:
        rmses = []
        for level, y_test in y_noisy.items():
            test_y = y_test  # Update test_y for lambda
            try:
                dy = methods[method_name]()
                if isinstance(dy, tuple):
                    dy = dy[0]
                rmse = np.sqrt(np.mean((dy - dy_true)**2))
                rmses.append(rmse)
            except:
                rmses.append(np.nan)
        ax5.semilogy(['low', 'medium', 'high'], rmses, 'o-',
                    label=method_name, linewidth=2, markersize=8)
ax5.set_title('Performance vs Noise Level')
ax5.set_xlabel('Noise Level')
ax5.set_ylabel('RMSE (log scale)')
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)

# 6. Method categories bar chart
ax6 = plt.subplot(3, 3, 6)
categories = {
    'Excellent\n(<0.05)': sum(1 for r in results.values() if r['rmse'] < 0.05),
    'Good\n(<0.1)': sum(1 for r in results.values() if 0.05 <= r['rmse'] < 0.1),
    'OK\n(<0.5)': sum(1 for r in results.values() if 0.1 <= r['rmse'] < 0.5),
    'Bad\n(≥0.5)': sum(1 for r in results.values() if r['rmse'] >= 0.5)
}
bars = ax6.bar(categories.keys(), categories.values(),
               color=['green', 'lightgreen', 'orange', 'red'])
ax6.set_title('Method Performance Distribution')
ax6.set_ylabel('Number of Methods')
for bar, value in zip(bars, categories.values()):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(value), ha='center')
ax6.set_ylim(0, max(categories.values()) + 1)

# 7-9. Individual showcases
showcases = sorted_rmse[:3] if len(sorted_rmse) >= 3 else sorted_rmse
for idx, (name, data) in enumerate(showcases):
    ax = plt.subplot(3, 3, 7 + idx)
    ax.plot(t, dy_true, 'k-', linewidth=3, label='True', alpha=0.5)
    ax.plot(t, data['dy'], 'b-', linewidth=1.5, label=name)
    ax.fill_between(t, dy_true, data['dy'], alpha=0.2, color='red')
    ax.set_title(f"{name}\nRMSE={data['rmse']:.4f}, Boundary={data['ratio']:.1f}x")
    ax.set_xlabel('t')
    ax.set_ylabel('dy/dt')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('PyNumDiff: Complete Performance Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('pynumdiff_real_performance.png', dpi=150, bbox_inches='tight')
print("Saved: pynumdiff_real_performance.png")

# Final verdict
print("\n" + "=" * 80)
print("THE VERDICT")
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
bad_boundary = [n for n, r in results.items() if r['ratio'] > 10]

print(f"\nBoundary handling:")
print(f"  Good (<2x worse): {', '.join(good_boundary) if good_boundary else 'None'}")
print(f"  Terrible (>10x worse): {', '.join(bad_boundary) if bad_boundary else 'None'}")

if sorted_rmse:
    print(f"\n🏆 BEST: {sorted_rmse[0][0]} (RMSE={sorted_rmse[0][1]['rmse']:.4f})")
    print(f"👎 WORST: {sorted_rmse[-1][0]} (RMSE={sorted_rmse[-1][1]['rmse']:.4f})")

# Overall assessment
total = len(results)
good_count = len(excellent) + len(good)
print(f"\nOVERALL: {good_count}/{total} methods work well for first derivatives")
print(f"PyNumDiff is {'GOOD' if good_count > total/2 else 'MIXED' if good_count > 0 else 'POOR'} for this task")