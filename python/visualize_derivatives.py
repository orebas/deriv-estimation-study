#!/usr/bin/env python3
"""
Visualize derivative estimation results for top methods.
Shows ground truth data, noisy data, and derivative estimates from best methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results" / "comprehensive"
DATA_DIR = Path(__file__).parent.parent / "data"
REPORT_DIR = Path(__file__).parent.parent / "report"
REPORT_DIR.mkdir(exist_ok=True)

# Choose a representative noise level and trial
NOISE_LEVEL = 1e-2  # 1% noise
TRIAL = 1

# Number of top methods to show
N_TOP_METHODS = 4

print("="*80)
print("DERIVATIVE ESTIMATION VISUALIZATION")
print("="*80)
print(f"\nNoise level: {NOISE_LEVEL*100}%")
print(f"Trial: {TRIAL}")
print(f"Showing top {N_TOP_METHODS} methods")

# Load summary data to identify top methods
print("\nLoading summary data...")
summary = pd.read_csv(RESULTS_DIR / "comprehensive_summary.csv")

# Find top methods at this noise level for derivative order 3
top_methods_df = summary[
    (summary['noise_level'] == NOISE_LEVEL) &
    (summary['deriv_order'] == 3)
].nsmallest(N_TOP_METHODS, 'mean_rmse')

top_methods = top_methods_df['method'].tolist()
print(f"\nTop {N_TOP_METHODS} methods (by RMSE at order 3):")
for i, method in enumerate(top_methods, 1):
    rmse = top_methods_df[top_methods_df['method'] == method]['mean_rmse'].values[0]
    print(f"  {i}. {method}: RMSE = {rmse:.3f}")

# Load the input JSON file for this noise/trial
trial_id = f"noise{int(NOISE_LEVEL*1e8)}e-8_trial{TRIAL}"
input_json = DATA_DIR / "input" / f"{trial_id}.json"
output_json = DATA_DIR / "output" / f"{trial_id}_results.json"

print(f"\nLoading data from {trial_id}...")
with open(input_json, 'r') as f:
    input_data = json.load(f)

times = np.array(input_data['times'])
y_noisy = np.array(input_data['y_noisy'])
y_true = np.array(input_data['y_true'])

# Ground truth derivatives
ground_truth_derivs = {}
for order in range(8):
    ground_truth_derivs[order] = np.array(input_data['ground_truth_derivatives'][str(order)])

# Load method predictions from output JSON
with open(output_json, 'r') as f:
    output_data = json.load(f)

# Extract predictions for top methods
method_predictions = {}
for method in top_methods:
    if method in output_data['methods']:
        method_data = output_data['methods'][method]
        if method_data['success']:
            preds = {}
            for order in range(8):
                if str(order) in method_data['predictions']:
                    preds[order] = np.array(method_data['predictions'][str(order)])
            method_predictions[method] = preds

print(f"Loaded predictions for {len(method_predictions)} methods")

# Create comprehensive figure
print("\nGenerating visualization...")
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Color scheme
colors = plt.cm.tab10(np.linspace(0, 1, N_TOP_METHODS))
method_colors = {method: colors[i] for i, method in enumerate(top_methods)}

# Panel 1: Original data (order 0)
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(times, y_true, 'k-', linewidth=2, label='Ground Truth', zorder=10)
ax0.scatter(times, y_noisy, s=20, alpha=0.5, color='red', label=f'Noisy Data ({NOISE_LEVEL*100}%)', zorder=5)
for method in top_methods:
    if method in method_predictions and 0 in method_predictions[method]:
        ax0.plot(times, method_predictions[method][0], '--',
                linewidth=1.5, alpha=0.7, color=method_colors[method],
                label=method)
ax0.set_xlabel('Time (t)', fontsize=10)
ax0.set_ylabel('x(t)', fontsize=10)
ax0.set_title('Observable: x(t)', fontsize=11, fontweight='bold')
ax0.legend(fontsize=7, loc='best')
ax0.grid(True, alpha=0.3)

# Panels 2-9: Derivatives 1-7 and a summary panel
derivative_axes = []
for idx, order in enumerate(range(1, 8), start=1):
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    derivative_axes.append(ax)

    # Plot ground truth
    ax.plot(times, ground_truth_derivs[order], 'k-', linewidth=2,
           label='Ground Truth', zorder=10)

    # Plot predictions from top methods
    for method in top_methods:
        if method in method_predictions and order in method_predictions[method]:
            pred = method_predictions[method][order]
            # Filter out non-finite values
            valid = np.isfinite(pred)
            t_valid = times[valid]
            pred_valid = pred[valid]
            if len(pred_valid) > 0:
                ax.plot(t_valid, pred_valid, '--', linewidth=1.5, alpha=0.7,
                       color=method_colors[method], label=method)

    ax.set_xlabel('Time (t)', fontsize=10)
    ax.set_ylabel(f'd^{order}x/dt^{order}', fontsize=10)
    ax.set_title(f'{order}-th Derivative', fontsize=11, fontweight='bold')
    if order == 1:
        ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

# Panel 9: RMSE summary across all orders
ax_summary = fig.add_subplot(gs[2, 2])
orders = np.arange(8)
for method in top_methods:
    rmse_values = []
    for order in orders:
        method_summary = summary[
            (summary['method'] == method) &
            (summary['deriv_order'] == order) &
            (summary['noise_level'] == NOISE_LEVEL)
        ]
        if len(method_summary) > 0:
            rmse_values.append(method_summary['mean_rmse'].values[0])
        else:
            rmse_values.append(np.nan)

    ax_summary.semilogy(orders, rmse_values, 'o-', linewidth=2, markersize=6,
                       color=method_colors[method], label=method)

ax_summary.set_xlabel('Derivative Order', fontsize=10)
ax_summary.set_ylabel('RMSE (log scale)', fontsize=10)
ax_summary.set_title('RMSE vs Derivative Order', fontsize=11, fontweight='bold')
ax_summary.set_xticks(orders)
ax_summary.legend(fontsize=8, loc='best')
ax_summary.grid(True, alpha=0.3, which='both')

# Overall title
fig.suptitle(f'Derivative Estimation: Lotka-Volterra System\n'
            f'Noise Level: {NOISE_LEVEL*100}%, Trial {TRIAL}',
            fontsize=14, fontweight='bold', y=0.995)

# Save figure
output_file = REPORT_DIR / f"derivative_visualization_noise{int(NOISE_LEVEL*100)}pct.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")

# Also save as PNG for easy viewing
output_png = REPORT_DIR / f"derivative_visualization_noise{int(NOISE_LEVEL*100)}pct.png"
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"PNG version saved to: {output_png}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
