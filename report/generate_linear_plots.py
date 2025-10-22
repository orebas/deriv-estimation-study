#!/usr/bin/env python3
"""
Generate granular plots with LINEAR scale and ADAPTIVE capping.
Easy conditions (low order/noise): cap at 2.0
Hard conditions (high order/noise): cap at 5.0
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import os

os.makedirs('paper_figures/supplementary_linear', exist_ok=True)

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
orders = list(range(7))  # Data has orders 0-6 (max_order=6)

full_coverage_methods = set()
for method in methods_set:
    coverage = sum(1 for o in orders for n in noise_levels if (method, o, n) in data)
    if coverage == 49:  # 7 orders × 7 noise levels
        full_coverage_methods.add(method)

all_methods_sorted = sorted(methods_set)
full_coverage_sorted = sorted(full_coverage_methods)

unique_cats = sorted(set(categories.values()))
cat_colors = dict(zip(unique_cats, plt.cm.tab10(range(len(unique_cats)))))

print(f"Generating LINEAR scale plots with ADAPTIVE capping")
print(f"Methods: {len(all_methods_sorted)} (filtered)")
print(f"Full-coverage: {len(full_coverage_sorted)}")

def get_cap_for_condition(order, noise):
    """Determine appropriate cap based on difficulty."""
    # Easy conditions: low order AND low noise
    if order <= 3 and noise <= 1e-4:
        return 2.0
    # Hard conditions: high order OR high noise
    else:
        return 5.0

def cap_value(val, cap):
    """Cap value at maximum, return NaN if invalid."""
    if np.isnan(val) or val <= 0:
        return np.nan
    return min(val, cap)

# ===========================================================================
# PLOT SET 1: Per-Order Heatmaps (Method × Noise, LINEAR scale)
# ===========================================================================
print("\n[SET 1] Generating 8 per-order heatmaps (LINEAR scale)...")

for order in orders:
    fig, ax = plt.subplots(figsize=(10, 12))

    # Determine cap for this order
    # Use most conservative cap across noise levels for this order
    caps = [get_cap_for_condition(order, n) for n in noise_levels]
    max_cap = max(caps)

    # Build matrix
    matrix = []
    for method in all_methods_sorted:
        row = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            cap = get_cap_for_condition(order, noise)
            row.append(cap_value(val, cap))
        matrix.append(row)

    matrix = np.array(matrix)

    # Use linear colormap
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=max_cap, interpolation='nearest')

    ax.set_yticks(range(len(all_methods_sorted)))
    ax.set_yticklabels([m.replace('_', ' ') for m in all_methods_sorted], fontsize=7)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=9)
    ax.set_xlabel('Noise Level', fontsize=11)

    cap_info = f" (cap={max_cap:.1f})" if max_cap < 10 else ""
    ax.set_title(f'Order {order}: Method Performance vs Noise{cap_info}\n(nRMSE, LINEAR scale, {len(all_methods_sorted)} methods)',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'nRMSE (capped at {max_cap:.1f})', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'paper_figures/supplementary_linear/order{order}_method_vs_noise_linear.png',
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 8 heatmaps with adaptive capping")

# ===========================================================================
# PLOT SET 2: Per-Noise Heatmaps (Method × Order, LINEAR scale)
# ===========================================================================
print("\n[SET 2] Generating 7 per-noise heatmaps (LINEAR scale)...")

for idx, noise in enumerate(noise_levels):
    fig, ax = plt.subplots(figsize=(10, 12))

    # Determine cap for this noise level
    caps = [get_cap_for_condition(o, noise) for o in orders]
    max_cap = max(caps)

    matrix = []
    for method in all_methods_sorted:
        row = []
        for order in orders:
            val = data.get((method, order, noise), np.nan)
            cap = get_cap_for_condition(order, noise)
            row.append(cap_value(val, cap))
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=max_cap, interpolation='nearest')

    ax.set_yticks(range(len(all_methods_sorted)))
    ax.set_yticklabels([m.replace('_', ' ') for m in all_methods_sorted], fontsize=7)
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([f'O{o}' for o in orders], fontsize=9)
    ax.set_xlabel('Derivative Order', fontsize=11)

    cap_info = f" (cap={max_cap:.1f})" if max_cap < 10 else ""
    ax.set_title(f'Noise {noise_labels[idx]}: Method Performance vs Order{cap_info}\n(nRMSE, LINEAR scale, {len(all_methods_sorted)} methods)',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'nRMSE (capped at {max_cap:.1f})', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'paper_figures/supplementary_linear/noise{noise_labels[idx]}_method_vs_order_linear.png',
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 7 heatmaps with adaptive capping")

# ===========================================================================
# PLOT SET 3: Line plots - Per Noise Level (LINEAR scale)
# ===========================================================================
print("\n[SET 3] Generating 7 line plots per noise level (LINEAR scale)...")

colors_methods = plt.cm.tab20(np.linspace(0, 1, len(full_coverage_sorted)))

for idx, noise in enumerate(noise_levels):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Determine max cap for this noise level
    caps = [get_cap_for_condition(o, noise) for o in orders]
    max_cap = max(caps)

    for midx, method in enumerate(full_coverage_sorted):
        nrmse_by_order = []
        for order in orders:
            val = data.get((method, order, noise), np.nan)
            cap = get_cap_for_condition(order, noise)
            nrmse_by_order.append(cap_value(val, cap))

        ax.plot(orders, nrmse_by_order, 'o-', label=method.replace('_', ' '),
                color=colors_methods[midx], linewidth=2, markersize=5, alpha=0.8)

    ax.set_xlabel('Derivative Order', fontsize=12)
    ax.set_ylabel('nRMSE (LINEAR)', fontsize=12)
    ax.set_title(f'Performance vs Order at Noise {noise_labels[idx]}\n({len(full_coverage_sorted)} full-coverage methods, capped at {max_cap:.1f})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(orders)
    ax.set_ylim(0, max_cap * 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='nRMSE = 1.0')
    ax.legend(fontsize=7, ncol=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'paper_figures/supplementary_linear/line_noise{noise_labels[idx]}_vs_order_linear.png',
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 7 line plots")

# ===========================================================================
# PLOT SET 4: Line plots - Per Order (LINEAR scale)
# ===========================================================================
print("\n[SET 4] Generating 8 line plots per order (LINEAR scale)...")

for order in orders:
    fig, ax = plt.subplots(figsize=(14, 8))

    caps = [get_cap_for_condition(order, n) for n in noise_levels]
    max_cap = max(caps)

    for midx, method in enumerate(full_coverage_sorted):
        nrmse_by_noise = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            cap = get_cap_for_condition(order, noise)
            nrmse_by_noise.append(cap_value(val, cap))

        ax.plot(range(len(noise_levels)), nrmse_by_noise, 'o-', label=method.replace('_', ' '),
                color=colors_methods[midx], linewidth=2, markersize=5, alpha=0.8)

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('nRMSE (LINEAR)', fontsize=12)
    ax.set_title(f'Performance vs Noise at Order {order}\n({len(full_coverage_sorted)} full-coverage methods, capped at {max_cap:.1f})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=9, rotation=45, ha='right')
    ax.set_ylim(0, max_cap * 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='nRMSE = 1.0')
    ax.legend(fontsize=7, ncol=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'paper_figures/supplementary_linear/line_order{order}_vs_noise_linear.png',
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 8 line plots")

# ===========================================================================
# PLOT SET 5: Individual Method Grids with VALUES (LINEAR scale)
# ===========================================================================
print("\n[SET 5] Generating individual method grids with annotated values...")

for method in full_coverage_sorted:
    fig, ax = plt.subplots(figsize=(10, 8))

    matrix = []
    for order in orders:
        row = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            # Use condition-specific cap
            cap = get_cap_for_condition(order, noise)
            row.append(cap_value(val, cap))
        matrix.append(row)

    matrix = np.array(matrix)

    # Use vmax=5 for all methods to allow comparison
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=5.0, interpolation='nearest')

    ax.set_yticks(range(len(orders)))
    ax.set_yticklabels([f'Order {o}' for o in orders], fontsize=10)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=10)
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Derivative Order', fontsize=12)
    ax.set_title(f'{method.replace("_", " ")}: Complete Performance Map\nnRMSE (LINEAR scale, adaptive capping)',
                 fontsize=14, fontweight='bold')

    # Add text annotations with actual values
    for i in range(len(orders)):
        for j in range(len(noise_levels)):
            if not np.isnan(matrix[i, j]):
                # Show actual value before capping
                actual_val = data.get((method, orders[i], noise_levels[j]), np.nan)
                if actual_val < 10:
                    text_val = f'{actual_val:.2f}'
                else:
                    text_val = f'{actual_val:.1f}'

                # Color text based on brightness
                brightness = matrix[i, j] / 5.0  # normalize to 0-1
                color = 'white' if brightness > 0.5 else 'black'

                ax.text(j, i, text_val, ha="center", va="center", fontsize=7,
                       color=color, weight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('nRMSE (LINEAR)', fontsize=11)

    plt.tight_layout()
    safe_name = method.replace('_', '-').replace('/', '-')
    plt.savefig(f'paper_figures/supplementary_linear/method_{safe_name}_grid_linear.png',
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"  → Saved 13 method grids")

# ===========================================================================
# PLOT SET 6: Small Multiples - Top 10 Methods (LINEAR scale)
# ===========================================================================
print("\n[SET 6] Generating small multiples for top 10 methods...")

# Get top 10 by median
method_medians = {}
for method in full_coverage_sorted:
    vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    vals = [v for v in vals if not np.isnan(v) and v > 0 and v < 100]  # exclude extreme outliers
    if vals:
        method_medians[method] = np.median(vals)
    else:
        method_medians[method] = np.inf

top10_methods = sorted(method_medians.items(), key=lambda x: x[1])[:10]
top10_method_names = [m[0] for m in top10_methods]

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for idx, method in enumerate(top10_method_names):
    ax = axes[idx]

    matrix = []
    for order in orders:
        row = []
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            cap = get_cap_for_condition(order, noise)
            row.append(cap_value(val, cap))
        matrix.append(row)

    matrix = np.array(matrix)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=5.0, interpolation='nearest')

    ax.set_title(method.replace('_', ' '), fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(orders)))
    ax.set_yticklabels([f'O{o}' for o in orders], fontsize=8)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels([n[:4] for n in noise_labels], fontsize=7, rotation=45, ha='right')

    if idx == 9:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('nRMSE', fontsize=9)

plt.suptitle('Top 10 Methods: Performance Map (LINEAR scale, adaptive capping)\nEach subplot shows Order × Noise for one method',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_figures/supplementary_linear/small_multiples_top10_linear.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"  → Saved: small_multiples_top10_linear.png")

print("\n" + "="*70)
print("LINEAR SCALE PLOTS COMPLETE!")
print("="*70)
print(f"\nGenerated plot sets with ADAPTIVE capping:")
print(f"  - Easy conditions (order ≤3, noise ≤1e-4): cap at 2.0")
print(f"  - Hard conditions (order >3 or noise >1e-4): cap at 5.0")
print(f"\nSET 1: 8 per-order heatmaps")
print(f"SET 2: 7 per-noise heatmaps")
print(f"SET 3: 7 line plots (per noise)")
print(f"SET 4: 8 line plots (per order)")
print(f"SET 5: 13 individual method grids (with values!)")
print(f"SET 6: 1 small multiples (top 10)")
print(f"\nTotal: 44 plots in paper_figures/supplementary_linear/")
print(f"All using LINEAR nRMSE scale (NO log transform)")
print("="*70)
