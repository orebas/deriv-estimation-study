"""
Comprehensive visualization of ALL PyNumDiff methods approximating derivatives
Shows each method's performance on a mild test function
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pynumdiff
from pynumdiff import smooth_finite_difference as sfd
from pynumdiff import total_variation_regularization as tvr
from pynumdiff.polynomial_fit import polydiff, savgoldiff, splinediff
from pynumdiff.basis_fit import rbfdiff, spectraldiff
from pynumdiff import kalman_smooth

# Generate test signal - a nice mild function
n = 301
t = np.linspace(0.1, 3, n)  # Start at 0.1 to avoid x^(5/2) issues at 0
dt = np.mean(np.diff(t))

# Function: x^(5/2) + sin(2x)
y_true = t**(5/2) + np.sin(2*t)
# True derivative: (5/2)*x^(3/2) + 2*cos(2x)
dy_true = (5/2) * t**(3/2) + 2*np.cos(2*t)

# Add mild noise
np.random.seed(42)
noise_level = 1e-3
y_noisy = y_true + noise_level * np.random.randn(n)

print("=" * 80)
print("MEGA VISUALIZATION: All PyNumDiff Methods on f(x) = x^(5/2) + sin(2x)")
print("=" * 80)

# Define all working methods
methods = {
    # Basic FD
    'First Order FD': lambda: pynumdiff.first_order(y_noisy, dt),
    'Second Order FD': lambda: pynumdiff.second_order(y_noisy, dt),
    'Fourth Order FD': lambda: pynumdiff.fourth_order(y_noisy, dt),

    # Smooth FD
    'Butterworth': lambda: sfd.butterdiff(y_noisy, dt, filter_order=2, cutoff_freq=0.1),
    'Gaussian': lambda: sfd.gaussiandiff(y_noisy, dt, window_size=11, num_iterations=2),
    'Friedrichs': lambda: sfd.friedrichsdiff(y_noisy, dt, window_size=11, num_iterations=2),
    'Mean Diff': lambda: sfd.meandiff(y_noisy, dt, window_size=11, num_iterations=2),
    'Median Diff': lambda: sfd.mediandiff(y_noisy, dt, window_size=11, num_iterations=2),

    # Polynomial
    'Spline': lambda: splinediff(y_noisy, t, s=1e-4, degree=3),
    'Polynomial': lambda: polydiff(y_noisy, dt, degree=5, window_size=11, kernel='friedrichs'),

    # Basis functions
    'Spectral': lambda: spectraldiff(y_noisy, dt, high_freq_cutoff=0.15),
    'RBF': lambda: rbfdiff(t, y_noisy, sigma=0.2, lmbd=1e-4),

    # Total variation
    'TV Regularized': lambda: tvr.tvrdiff(y_noisy, dt, order=1, gamma=5e-3),
    'TV Velocity': lambda: tvr.velocity(y_noisy, dt, gamma=5e-3),

    # Kalman
    'Kalman Const Vel': lambda: kalman_smooth.constant_velocity(
        y_noisy, dt, r=1e-3, q=1e-6, forwardbackward='forward-backward'
    ),
    'Kalman Const Accel': lambda: kalman_smooth.constant_acceleration(
        y_noisy, dt, r=1e-3, q=1e-6, forwardbackward='forward-backward'
    ),
}

# Process all methods
results = {}
for name, method in methods.items():
    try:
        result = method()

        # Extract derivative
        if isinstance(result, tuple):
            if len(result) >= 2:
                x_smooth, dx_dt = result[0], result[1]
            else:
                dx_dt = result[0]
        else:
            dx_dt = result

        # Handle special cases
        if hasattr(dx_dt, 'flatten'):
            dx_dt = dx_dt.flatten()

        # For Kalman const accel, use velocity not acceleration
        if name == 'Kalman Const Accel' and isinstance(result, tuple) and len(result) >= 3:
            dx_dt = result[1]  # Use velocity

        # Size adjustment for finite differences
        if len(dx_dt) != len(t):
            if len(dx_dt) == len(t) - 1:
                dx_dt = np.append(dx_dt, dx_dt[-1])
            elif len(dx_dt) == len(t) - 2:
                dx_dt = np.concatenate([[dx_dt[0]], dx_dt, [dx_dt[-1]]])
            else:
                print(f"  {name}: Size mismatch {len(dx_dt)} vs {len(t)}")
                continue

        # Calculate metrics
        error = dx_dt - dy_true
        rmse = np.sqrt(np.mean(error**2))
        max_err = np.max(np.abs(error))

        results[name] = {
            'dx_dt': dx_dt,
            'error': error,
            'rmse': rmse,
            'max_err': max_err
        }
        print(f"  {name:20} RMSE={rmse:.4f}, Max Error={max_err:.4f}")

    except Exception as e:
        print(f"  {name:20} FAILED: {str(e)[:40]}")

# Sort by performance
sorted_methods = sorted(results.items(), key=lambda x: x[1]['rmse'])

print(f"\nSuccessfully processed {len(results)} methods")
print(f"Best: {sorted_methods[0][0]} (RMSE={sorted_methods[0][1]['rmse']:.4f})")
print(f"Worst: {sorted_methods[-1][0]} (RMSE={sorted_methods[-1][1]['rmse']:.4f})")

# Create MEGA visualization
n_methods = len(results)
n_cols = 4
n_rows = int(np.ceil(n_methods / n_cols))

fig = plt.figure(figsize=(20, 5 * n_rows))
gs = gridspec.GridSpec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.5],
                       hspace=0.3, wspace=0.25)

# Color map for errors
cmap = plt.cm.RdBu_r
norm = plt.Normalize(vmin=-0.5, vmax=0.5)

# Main title
fig.suptitle(f'All PyNumDiff Methods: Derivative of $f(x) = x^{{5/2}} + \\sin(2x)$\n' +
             f'Noise level = {noise_level}, n = {n} points',
             fontsize=16, fontweight='bold', y=1.00)

# Plot each method
for idx, (name, data) in enumerate(sorted_methods):
    row = idx // n_cols
    col = idx % n_cols

    ax = fig.add_subplot(gs[row, col])

    # Plot true derivative
    ax.plot(t, dy_true, 'k-', linewidth=2, alpha=0.3, label='True')

    # Plot estimated derivative
    ax.plot(t, data['dx_dt'], 'b-', linewidth=1.5, alpha=0.8, label='Estimated')

    # Add error as colored background
    error_colors = cmap(norm(data['error']))
    for i in range(len(t)-1):
        ax.fill_between([t[i], t[i+1]],
                       [dy_true[i], dy_true[i+1]],
                       [data['dx_dt'][i], data['dx_dt'][i+1]],
                       color=error_colors[i], alpha=0.3)

    # Add metrics to title
    title = f'{name}\nRMSE={data["rmse"]:.4f}, Max={data["max_err"]:.3f}'

    # Color-code title by performance
    if data['rmse'] < 0.1:
        title_color = 'darkgreen'
        title += ' ✓✓'
    elif data['rmse'] < 0.3:
        title_color = 'green'
        title += ' ✓'
    elif data['rmse'] < 1.0:
        title_color = 'orange'
        title += ' ~'
    else:
        title_color = 'red'
        title += ' ✗'

    ax.set_title(title, fontsize=10, fontweight='bold', color=title_color)
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel("f'(x)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # Add legend only for first plot
    if idx == 0:
        ax.legend(fontsize=8, loc='upper left')

# Add error scale colorbar at bottom
ax_cb = fig.add_subplot(gs[-1, :])
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=ax_cb, orientation='horizontal')
cb.set_label('Error (blue = negative, red = positive)', fontsize=10)

plt.savefig('pynumdiff_mega_visualization.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: pynumdiff_mega_visualization.png")

# Create a second figure with error analysis
fig2, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for idx, (name, data) in enumerate(sorted_methods[:16]):
    ax = axes[idx]

    # Top subplot - derivative comparison
    ax2 = ax.twinx()

    # Plot derivative on left axis
    line1 = ax.plot(t, dy_true, 'k-', linewidth=2, alpha=0.4, label='True deriv')
    line2 = ax.plot(t, data['dx_dt'], 'b-', linewidth=1, alpha=0.8, label='Estimated')

    # Plot error on right axis
    line3 = ax2.plot(t, data['error'], 'r-', linewidth=1, alpha=0.6, label='Error')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # Formatting
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('Derivative', fontsize=8, color='b')
    ax2.set_ylabel('Error', fontsize=8, color='r')
    ax.tick_params(axis='y', labelcolor='b', labelsize=7)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)

    # Title with metrics
    title = f'{name}\nRMSE={data["rmse"]:.4f}'
    if data['rmse'] < 0.1:
        title_color = 'darkgreen'
    elif data['rmse'] < 0.3:
        title_color = 'green'
    elif data['rmse'] < 1.0:
        title_color = 'orange'
    else:
        title_color = 'red'
    ax.set_title(title, fontsize=9, fontweight='bold', color=title_color)

    ax.grid(True, alpha=0.3)

    # Add small inset showing error distribution
    ax_inset = ax.inset_axes([0.65, 0.7, 0.3, 0.25])
    ax_inset.hist(data['error'], bins=20, alpha=0.7, color='red', edgecolor='darkred')
    ax_inset.set_xlabel('Error', fontsize=6)
    ax_inset.set_ylabel('Count', fontsize=6)
    ax_inset.tick_params(labelsize=5)
    ax_inset.grid(True, alpha=0.3)

plt.suptitle('PyNumDiff Methods: Detailed Error Analysis\n' +
             f'$f(x) = x^{{5/2}} + \\sin(2x)$, $f\'(x) = \\frac{{5}}{{2}}x^{{3/2}} + 2\\cos(2x)$',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pynumdiff_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: pynumdiff_error_analysis.png")

# Create summary statistics figure
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. RMSE comparison bar chart
ax = axes[0, 0]
names = [name for name, _ in sorted_methods]
rmses = [data['rmse'] for _, data in sorted_methods]
colors = ['darkgreen' if r < 0.1 else 'green' if r < 0.3 else 'orange' if r < 1 else 'red' for r in rmses]

y_pos = np.arange(len(names))
bars = ax.barh(y_pos, rmses, color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('RMSE')
ax.set_title('Performance Ranking (RMSE)', fontweight='bold')
ax.axvline(0.1, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
ax.axvline(0.3, color='gray', linestyle=':', alpha=0.5, label='OK threshold')
ax.legend(fontsize=8)

# 2. Error distributions boxplot
ax = axes[0, 1]
error_data = [data['error'] for _, data in sorted_methods[:10]]  # Top 10
labels = [name[:15] for name, _ in sorted_methods[:10]]
bp = ax.boxplot(error_data, labels=labels, patch_artist=True, vert=False)
for patch, color in zip(bp['boxes'], colors[:10]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_xlabel('Error Distribution')
ax.set_title('Error Distribution (Top 10 Methods)', fontweight='bold')
ax.tick_params(axis='y', labelsize=8)
ax.grid(True, alpha=0.3)

# 3. Performance over signal
ax = axes[1, 0]
# Show how error changes along the signal for top 5
for i, (name, data) in enumerate(sorted_methods[:5]):
    window = 21
    local_rmse = np.convolve(data['error']**2, np.ones(window)/window, mode='same')
    local_rmse = np.sqrt(local_rmse)
    ax.plot(t, local_rmse, linewidth=1.5, alpha=0.7, label=name)
ax.set_xlabel('x')
ax.set_ylabel('Local RMSE (window=21)')
ax.set_title('Local Performance Along Signal (Top 5)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Summary table
ax = axes[1, 1]
ax.axis('tight')
ax.axis('off')

# Create summary statistics
categories = {
    'Excellent (<0.1)': sum(1 for r in rmses if r < 0.1),
    'Good (0.1-0.3)': sum(1 for r in rmses if 0.1 <= r < 0.3),
    'OK (0.3-1.0)': sum(1 for r in rmses if 0.3 <= r < 1.0),
    'Poor (>1.0)': sum(1 for r in rmses if r >= 1.0),
}

# Create table data
table_data = []
table_data.append(['Category', 'Count', 'Best Method'])
table_data.append(['Excellent', str(categories['Excellent (<0.1)']),
                  sorted_methods[0][0] if rmses[0] < 0.1 else 'None'])
table_data.append(['Good', str(categories['Good (0.1-0.3)']),
                  next((n for n, d in sorted_methods if 0.1 <= d['rmse'] < 0.3), 'None')])
table_data.append(['OK', str(categories['OK (0.3-1.0)']),
                  next((n for n, d in sorted_methods if 0.3 <= d['rmse'] < 1.0), 'None')])
table_data.append(['Poor', str(categories['Poor (>1.0)']),
                  next((n for n, d in sorted_methods if d['rmse'] >= 1.0), 'None')])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.3, 0.2, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code rows
colors_table = ['#90EE90', '#98FB98', '#FFE4B5', '#FFB6C1']
for i, color in enumerate(colors_table, 1):
    for j in range(3):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_alpha(0.3)

ax.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=20)

plt.suptitle('PyNumDiff Performance Analysis Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pynumdiff_summary_stats.png', dpi=150, bbox_inches='tight')
print(f"Saved: pynumdiff_summary_stats.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Generated 3 comprehensive visualizations:")
print(f"1. pynumdiff_mega_visualization.png - All methods side by side")
print(f"2. pynumdiff_error_analysis.png - Detailed error analysis")
print(f"3. pynumdiff_summary_stats.png - Performance summary")
print(f"\nFunction tested: f(x) = x^(5/2) + sin(2x)")
print(f"True derivative: f'(x) = (5/2)x^(3/2) + 2cos(2x)")
print(f"Best performer: {sorted_methods[0][0]} (RMSE={sorted_methods[0][1]['rmse']:.4f})")
print(f"Worst performer: {sorted_methods[-1][0]} (RMSE={sorted_methods[-1][1]['rmse']:.4f})")