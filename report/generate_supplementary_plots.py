#!/usr/bin/env python3
"""
Generate filtered plots for supplementary material.
Creates heatmaps and line plots with sensible nRMSE filtering (cap at 10.0).
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Create output directory
os.makedirs('paper_figures/supplementary', exist_ok=True)

# Read CSV data
data = defaultdict(dict)  # data[(method, order, noise)] = nrmse
methods_set = set()

csv_path = '../results/comprehensive/comprehensive_summary.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
        order = int(row['deriv_order'])
        noise = float(row['noise_level'])
        nrmse = float(row['mean_nrmse'])

        data[(method, order, noise)] = nrmse
        methods_set.add(method)

# Define noise levels and orders
noise_levels = [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.02, 0.05]
noise_labels = ['1e-8', '1e-6', '1e-4', '1e-3', '1e-2', '2e-2', '5e-2']
orders = list(range(8))

# Identify full-coverage methods
full_coverage_methods = set()
for method in methods_set:
    coverage = sum(1 for o in orders for n in noise_levels if (method, o, n) in data)
    if coverage == 56:
        full_coverage_methods.add(method)

full_coverage_sorted = sorted(full_coverage_methods)

# Select top methods by overall performance
method_avg_nrmse = {}
for method in full_coverage_methods:
    nrmse_vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    # Filter out extreme values for ranking
    filtered_vals = [v for v in nrmse_vals if not np.isnan(v) and v < 10]
    if filtered_vals:
        method_avg_nrmse[method] = np.mean(filtered_vals)
    else:
        method_avg_nrmse[method] = np.inf

top_methods = sorted(method_avg_nrmse.keys(), key=lambda m: method_avg_nrmse[m])[:10]

print(f"Top 10 methods by filtered average nRMSE: {top_methods}")

# ============================================================================
# PLOT 1: Heatmap for top methods across orders (filtered, capped at 10)
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, order in enumerate(orders):
    ax = axes[idx]

    # Build matrix: methods × noise levels
    matrix = []
    for method in top_methods:
        row = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            # Cap at 10 for visualization
            if not np.isnan(val):
                val = min(val, 10.0)
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix)

    # Create heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=2.0, interpolation='nearest')

    ax.set_title(f'Order {order}', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(top_methods)))
    ax.set_yticklabels([m.replace('_', ' ') for m in top_methods], fontsize=8)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=8, rotation=45, ha='right')

    if idx == 7:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('nRMSE (capped at 10)', fontsize=10)

plt.suptitle('Performance Heatmap: Top 10 Methods Across Orders\n(nRMSE capped at 10 for visualization)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_figures/supplementary/heatmap_top_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: heatmap_top_methods.png")

# ============================================================================
# PLOT 2: Noise sensitivity curves for top 5 methods at each order
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

top5_methods = top_methods[:5]
colors = plt.cm.tab10(range(5))

for idx, order in enumerate(orders):
    ax = axes[idx]

    for midx, method in enumerate(top5_methods):
        nrmse_vals = [data.get((method, order, n), np.nan) for n in noise_levels]
        # Cap at 10 for plotting
        nrmse_vals = [min(v, 10.0) if not np.isnan(v) else np.nan for v in nrmse_vals]

        ax.plot(range(len(noise_levels)), nrmse_vals, 'o-',
                label=method.replace('_', ' '), color=colors[midx], linewidth=2, markersize=6)

    ax.set_title(f'Order {order}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('nRMSE', fontsize=10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    if idx == 7:
        ax.legend(fontsize=8, loc='upper left')

plt.suptitle('Noise Sensitivity: Top 5 Methods\n(nRMSE capped at 10)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_figures/supplementary/noise_sensitivity_top5.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: noise_sensitivity_top5.png")

# ============================================================================
# PLOT 3: Order progression for top 5 methods at moderate noise (1e-3)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

moderate_noise = 1e-3

for midx, method in enumerate(top5_methods):
    nrmse_vals = [data.get((method, o, moderate_noise), np.nan) for o in orders]
    # Cap at 10 for plotting
    nrmse_vals = [min(v, 10.0) if not np.isnan(v) else np.nan for v in nrmse_vals]

    ax.plot(orders, nrmse_vals, 'o-', label=method.replace('_', ' '),
            color=colors[midx], linewidth=2, markersize=8)

ax.set_xlabel('Derivative Order', fontsize=14)
ax.set_ylabel('nRMSE (capped at 10)', fontsize=14)
ax.set_title(f'Order Progression at Noise Level {moderate_noise}\nTop 5 Methods',
             fontsize=16, fontweight='bold')
ax.set_xticks(orders)
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='nRMSE = 1.0')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('paper_figures/supplementary/order_progression_moderate_noise.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: order_progression_moderate_noise.png")

# ============================================================================
# PLOT 4: Per-method heatmaps (noise × order) for top 6 methods
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, method in enumerate(top_methods[:6]):
    ax = axes[idx]

    # Build matrix: noise levels × orders
    matrix = []
    for noise in noise_levels:
        row = []
        for order in orders:
            val = data.get((method, order, noise), np.nan)
            # Cap at 5 for better visualization of non-catastrophic behavior
            if not np.isnan(val):
                val = min(val, 5.0)
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1.5, interpolation='nearest')

    ax.set_title(method.replace('_', ' '), fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(noise_labels)))
    ax.set_yticklabels(noise_labels, fontsize=9)
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([f'O{o}' for o in orders], fontsize=9)
    ax.set_ylabel('Noise Level', fontsize=10)
    ax.set_xlabel('Derivative Order', fontsize=10)

    if idx == 5:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('nRMSE (capped at 5)', fontsize=9)

plt.suptitle('Per-Method Performance: Noise × Order Heatmaps\n(Top 6 Methods)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_figures/supplementary/per_method_heatmaps.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: per_method_heatmaps.png")

print("\n=== Summary ===")
print(f"Generated 4 supplementary plots in paper_figures/supplementary/")
print(f"All plots use nRMSE filtering/capping to avoid extreme outliers")
