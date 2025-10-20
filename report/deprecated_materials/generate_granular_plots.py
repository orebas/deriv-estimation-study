#!/usr/bin/env python3
"""
Generate GRANULAR supplementary plots - NO AVERAGING over noise or order.
Every (order, noise) combination shown explicitly.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import os

os.makedirs('paper_figures/supplementary_granular', exist_ok=True)

# EXCLUDED METHODS
EXCLUDED = {'AAA-HighPrec', 'AAA-LowPrec', 'SavitzkyGolay_Python', 'GP-Julia-SE'}

# Read CSV data
data = defaultdict(dict)
methods_set = set()
categories = {}

csv_path = '../results/comprehensive/comprehensive_summary.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
        if method in EXCLUDED:
            continue

        order = int(row['deriv_order'])
        noise = float(row['noise_level'])
        nrmse = float(row['mean_nrmse'])
        category = row['category']

        data[(method, order, noise)] = nrmse
        methods_set.add(method)
        categories[method] = category

noise_levels = [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.02, 0.05]
noise_labels = ['1e-8', '1e-6', '1e-4', '1e-3', '1e-2', '2e-2', '5e-2']
orders = list(range(8))

# Identify full-coverage methods
full_coverage_methods = set()
for method in methods_set:
    coverage = sum(1 for o in orders for n in noise_levels if (method, o, n) in data)
    if coverage == 56:
        full_coverage_methods.add(method)

all_methods_sorted = sorted(methods_set)
full_coverage_sorted = sorted(full_coverage_methods)

# Get category info
unique_cats = sorted(set(categories.values()))
cat_colors = dict(zip(unique_cats, plt.cm.tab10(range(len(unique_cats)))))

print(f"Generating GRANULAR plots (no averaging over noise or order)")
print(f"Methods: {len(all_methods_sorted)} (filtered)")
print(f"Full-coverage: {len(full_coverage_sorted)}")
print(f"Total conditions: {len(orders)} orders × {len(noise_levels)} noise = 56")

# ===========================================================================
# PLOT SET 1: Per-Order Heatmaps (Method × Noise, one per order)
# ===========================================================================
print("\n[SET 1] Generating 8 per-order heatmaps (method × noise)...")

for order in orders:
    fig, ax = plt.subplots(figsize=(10, 12))

    # Build matrix: methods × noise levels
    matrix = []
    for method in all_methods_sorted:
        row = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                row.append(np.log10(val))
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=-2, vmax=2, interpolation='nearest')

    ax.set_yticks(range(len(all_methods_sorted)))
    ax.set_yticklabels([m.replace('_', ' ') for m in all_methods_sorted], fontsize=7)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=9)
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_title(f'Derivative Order {order}: Method Performance vs Noise\n(log10(nRMSE), {len(all_methods_sorted)} methods)',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(nRMSE)', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'paper_figures/supplementary_granular/order{order}_method_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 8 heatmaps: order0_method_vs_noise.png ... order7_method_vs_noise.png")

# ===========================================================================
# PLOT SET 2: Per-Noise Heatmaps (Method × Order, one per noise level)
# ===========================================================================
print("\n[SET 2] Generating 7 per-noise heatmaps (method × order)...")

for idx, noise in enumerate(noise_levels):
    fig, ax = plt.subplots(figsize=(10, 12))

    # Build matrix: methods × orders
    matrix = []
    for method in all_methods_sorted:
        row = []
        for order in orders:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                row.append(np.log10(val))
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=-2, vmax=2, interpolation='nearest')

    ax.set_yticks(range(len(all_methods_sorted)))
    ax.set_yticklabels([m.replace('_', ' ') for m in all_methods_sorted], fontsize=7)
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([f'O{o}' for o in orders], fontsize=9)
    ax.set_xlabel('Derivative Order', fontsize=11)
    ax.set_title(f'Noise Level {noise_labels[idx]}: Method Performance vs Order\n(log10(nRMSE), {len(all_methods_sorted)} methods)',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(nRMSE)', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'paper_figures/supplementary_granular/noise{noise_labels[idx]}_method_vs_order.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 7 heatmaps: noise1e-8_method_vs_order.png ... noise5e-2_method_vs_order.png")

# ===========================================================================
# PLOT SET 3: Line plots - each noise level separately (method vs order)
# ===========================================================================
print("\n[SET 3] Generating 7 line plots (one per noise level, all methods vs order)...")

colors_methods = plt.cm.tab20(np.linspace(0, 1, len(full_coverage_sorted)))

for idx, noise in enumerate(noise_levels):
    fig, ax = plt.subplots(figsize=(14, 8))

    for midx, method in enumerate(full_coverage_sorted):
        nrmse_by_order = []
        for order in orders:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                nrmse_by_order.append(np.log10(val))
            else:
                nrmse_by_order.append(np.nan)

        ax.plot(orders, nrmse_by_order, 'o-', label=method.replace('_', ' '),
                color=colors_methods[midx], linewidth=2, markersize=5, alpha=0.8)

    ax.set_xlabel('Derivative Order', fontsize=12)
    ax.set_ylabel('log10(nRMSE)', fontsize=12)
    ax.set_title(f'Performance vs Order at Noise Level {noise_labels[idx]}\n({len(full_coverage_sorted)} full-coverage methods)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='nRMSE = 1.0')
    ax.legend(fontsize=7, ncol=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'paper_figures/supplementary_granular/line_noise{noise_labels[idx]}_vs_order.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 7 line plots: line_noise1e-8_vs_order.png ... line_noise5e-2_vs_order.png")

# ===========================================================================
# PLOT SET 4: Line plots - each order separately (method vs noise)
# ===========================================================================
print("\n[SET 4] Generating 8 line plots (one per order, all methods vs noise)...")

for order in orders:
    fig, ax = plt.subplots(figsize=(14, 8))

    for midx, method in enumerate(full_coverage_sorted):
        nrmse_by_noise = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                nrmse_by_noise.append(np.log10(val))
            else:
                nrmse_by_noise.append(np.nan)

        ax.plot(range(len(noise_levels)), nrmse_by_noise, 'o-', label=method.replace('_', ' '),
                color=colors_methods[midx], linewidth=2, markersize=5, alpha=0.8)

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('log10(nRMSE)', fontsize=12)
    ax.set_title(f'Performance vs Noise at Derivative Order {order}\n({len(full_coverage_sorted)} full-coverage methods)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=9, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='nRMSE = 1.0')
    ax.legend(fontsize=7, ncol=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'paper_figures/supplementary_granular/line_order{order}_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 8 line plots: line_order0_vs_noise.png ... line_order7_vs_noise.png")

# ===========================================================================
# PLOT SET 5: Small multiples - top 10 methods, all (order, noise) combos
# ===========================================================================
print("\n[SET 5] Generating small multiples for top 10 methods...")

# Get top 10 by median performance
method_medians = {}
for method in full_coverage_sorted:
    vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    vals = [v for v in vals if not np.isnan(v) and v > 0]
    if vals:
        method_medians[method] = np.median(np.log10(vals))
    else:
        method_medians[method] = np.inf

top10_methods = sorted(method_medians.items(), key=lambda x: x[1])[:10]
top10_method_names = [m[0] for m in top10_methods]

# Create 10 subplots (2x5 grid)
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for idx, method in enumerate(top10_method_names):
    ax = axes[idx]

    # Build matrix: orders × noise
    matrix = []
    for order in orders:
        row = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                row.append(np.log10(val))
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=-2, vmax=2, interpolation='nearest')

    ax.set_title(method.replace('_', ' '), fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(orders)))
    ax.set_yticklabels([f'O{o}' for o in orders], fontsize=8)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels([n[:4] for n in noise_labels], fontsize=7, rotation=45, ha='right')

    if idx == 9:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log10(nRMSE)', fontsize=9)

plt.suptitle('Top 10 Methods: Performance Across All (Order × Noise) Conditions\n(Each subplot shows complete performance map for one method)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_figures/supplementary_granular/small_multiples_top10_full_grid.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  → Saved: small_multiples_top10_full_grid.png")

# ===========================================================================
# PLOT SET 6: Per-category performance at each (order, noise) combination
# ===========================================================================
print("\n[SET 6] Generating category comparison heatmaps (8 orders × 7 noise = 56 panels)...")

# Create mega-grid: 8 rows (orders) × 7 cols (noise) = 56 subplots
fig = plt.figure(figsize=(28, 32))
gs = GridSpec(8, 7, figure=fig, hspace=0.4, wspace=0.3)

for oidx, order in enumerate(orders):
    for nidx, noise in enumerate(noise_levels):
        ax = fig.add_subplot(gs[oidx, nidx])

        # Collect performance by category for this condition
        category_vals = defaultdict(list)
        for method in all_methods_sorted:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                category_vals[categories[method]].append(np.log10(val))

        # Box plot
        cats = sorted(category_vals.keys())
        data_to_plot = [category_vals[c] for c in cats]

        bp = ax.boxplot(data_to_plot, labels=[c[:4] for c in cats], patch_artist=True, vert=True)

        for patch, cat in zip(bp['boxes'], cats):
            patch.set_facecolor(cat_colors.get(cat, 'lightblue'))
            patch.set_alpha(0.6)

        ax.set_ylim(-3, 3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.tick_params(axis='x', labelsize=6, labelrotation=90)
        ax.tick_params(axis='y', labelsize=6)

        # Title only on top row
        if oidx == 0:
            ax.set_title(f'{noise_labels[nidx]}', fontsize=8)

        # Y-label only on left column
        if nidx == 0:
            ax.set_ylabel(f'Order {order}', fontsize=8)

fig.suptitle('Category Performance Across All (Order × Noise) Conditions\nlog10(nRMSE) by Method Category',
             fontsize=18, fontweight='bold')
plt.savefig('paper_figures/supplementary_granular/category_grid_all_conditions.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  → Saved: category_grid_all_conditions.png (56 subplots!)")

# ===========================================================================
# PLOT SET 7: Per-method full grids for all full-coverage methods
# ===========================================================================
print("\n[SET 7] Generating individual method grids (13 full-coverage methods)...")

for method in full_coverage_sorted:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Build matrix: orders × noise
    matrix = []
    for order in orders:
        row = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                row.append(np.log10(val))
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=-2, vmax=2, interpolation='nearest')

    ax.set_yticks(range(len(orders)))
    ax.set_yticklabels([f'Order {o}' for o in orders], fontsize=10)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=10)
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Derivative Order', fontsize=12)
    ax.set_title(f'{method.replace("_", " ")}: Complete Performance Map\nlog10(nRMSE) across all conditions',
                 fontsize=14, fontweight='bold')

    # Add text annotations for exact values
    for i in range(len(orders)):
        for j in range(len(noise_levels)):
            if not np.isnan(matrix[i, j]):
                text_val = f'{matrix[i, j]:.2f}'
                color = 'white' if abs(matrix[i, j]) > 1 else 'black'
                ax.text(j, i, text_val, ha="center", va="center", fontsize=6,
                       color=color, weight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(nRMSE)', fontsize=11)

    plt.tight_layout()
    safe_name = method.replace('_', '-').replace('/', '-')
    plt.savefig(f'paper_figures/supplementary_granular/method_{safe_name}_full_grid.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 13 method grids: method_*.png")

# ===========================================================================
# PLOT SET 8: Scatter plot - ALL data points (no aggregation!)
# ===========================================================================
print("\n[SET 8] Generating scatter plot with ALL individual (method, order, noise) points...")

fig, ax = plt.subplots(figsize=(16, 10))

# Plot every single data point
for method in full_coverage_sorted:
    x_vals = []  # derivative order
    y_vals = []  # log10(nRMSE)
    colors_pts = []  # noise level

    for order in orders:
        for nidx, noise in enumerate(noise_levels):
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                x_vals.append(order + np.random.uniform(-0.15, 0.15))  # jitter for visibility
                y_vals.append(np.log10(val))
                colors_pts.append(nidx)

    # Plot with color based on noise level
    scatter = ax.scatter(x_vals, y_vals, c=colors_pts, cmap='viridis',
                        s=20, alpha=0.6, edgecolors='none')

ax.set_xlabel('Derivative Order', fontsize=12)
ax.set_ylabel('log10(nRMSE)', fontsize=12)
ax.set_title(f'All Individual Data Points: {len(full_coverage_sorted)} Methods × 8 Orders × 7 Noise Levels\n(Color = noise level, jittered for visibility)',
             fontsize=14, fontweight='bold')
ax.set_xticks(orders)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(noise_labels)))
cbar.set_label('Noise Level', fontsize=11)
cbar.ax.set_yticklabels(noise_labels, fontsize=8)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_granular/scatter_all_datapoints.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"  → Saved: scatter_all_datapoints.png")

print("\n" + "="*70)
print("GRANULAR PLOTS COMPLETE!")
print("="*70)
print(f"\nGenerated plot sets:")
print(f"  SET 1: 8 heatmaps (one per order)")
print(f"  SET 2: 7 heatmaps (one per noise level)")
print(f"  SET 3: 7 line plots (method vs order, one per noise)")
print(f"  SET 4: 8 line plots (method vs noise, one per order)")
print(f"  SET 5: 1 small multiples plot (top 10 methods)")
print(f"  SET 6: 1 mega-grid (category performance, 56 conditions)")
print(f"  SET 7: 13 individual method grids")
print(f"  SET 8: 1 scatter plot (all data points)")
print(f"\nTotal: 45 plots, NO AVERAGING anywhere!")
print(f"All saved in: paper_figures/supplementary_granular/")
print("="*70)
