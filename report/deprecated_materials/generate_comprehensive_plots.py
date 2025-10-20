#!/usr/bin/env python3
"""
Generate comprehensive supplementary plots with better scaling and more methods.
Implements 20 different plot ideas to choose from.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import os

# Create output directory
os.makedirs('paper_figures/supplementary_v2', exist_ok=True)

# Read CSV data
data = defaultdict(dict)  # data[(method, order, noise)] = nrmse
methods_set = set()
categories = {}  # method -> category

csv_path = '../results/comprehensive/comprehensive_summary.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
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

print(f"Total methods: {len(all_methods_sorted)}")
print(f"Full-coverage methods: {len(full_coverage_sorted)}")

# ===========================================================================
# PLOT 1: Log-scaled box plot by method (all methods, overall performance)
# ===========================================================================
print("\n[1/20] Generating log-scaled box plots by method...")

fig, ax = plt.subplots(figsize=(16, 10))

# Collect all nRMSE values per method
method_data = []
method_labels = []
for method in all_methods_sorted:
    values = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    values = [v for v in values if not np.isnan(v) and v > 0]
    if values:
        method_data.append(np.log10(values))
        method_labels.append(method.replace('_', ' '))

# Create box plot
bp = ax.boxplot(method_data, vert=False, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')

ax.set_yticks(range(1, len(method_labels) + 1))
ax.set_yticklabels(method_labels, fontsize=7)
ax.set_xlabel('log10(nRMSE)', fontsize=12)
ax.set_title('Overall Performance Distribution by Method (All 27 Methods)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='nRMSE = 1.0')
ax.legend()

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot01_boxplot_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot01_boxplot_all_methods.png")

# ===========================================================================
# PLOT 2: Line plot - log10(nRMSE) vs derivative order (all full-coverage methods)
# ===========================================================================
print("[2/20] Generating line plot: log(nRMSE) vs order for all full-coverage methods...")

fig, ax = plt.subplots(figsize=(14, 8))

colors = plt.cm.tab20(np.linspace(0, 1, len(full_coverage_sorted)))

for idx, method in enumerate(full_coverage_sorted):
    # Average across noise levels
    nrmse_by_order = []
    for order in orders:
        vals = [data.get((method, order, n), np.nan) for n in noise_levels]
        vals = [v for v in vals if not np.isnan(v) and v > 0]
        if vals:
            nrmse_by_order.append(np.mean(vals))
        else:
            nrmse_by_order.append(np.nan)

    log_vals = [np.log10(v) if not np.isnan(v) else np.nan for v in nrmse_by_order]
    ax.plot(orders, log_vals, 'o-', label=method.replace('_', ' '),
            color=colors[idx], linewidth=1.5, markersize=4, alpha=0.8)

ax.set_xlabel('Derivative Order', fontsize=12)
ax.set_ylabel('log10(nRMSE) [averaged over noise levels]', fontsize=12)
ax.set_title('Performance Degradation with Derivative Order\n(16 Full-Coverage Methods)',
             fontsize=14, fontweight='bold')
ax.set_xticks(orders)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='nRMSE = 1.0')
ax.legend(fontsize=7, ncol=2, loc='upper left')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot02_line_order_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot02_line_order_all_methods.png")

# ===========================================================================
# PLOT 3: Heatmap - log10(nRMSE) by method vs derivative order
# ===========================================================================
print("[3/20] Generating heatmap: method vs order...")

fig, ax = plt.subplots(figsize=(10, 14))

# Build matrix: methods × orders
matrix = []
for method in all_methods_sorted:
    row = []
    for order in orders:
        # Average across noise levels
        vals = [data.get((method, order, n), np.nan) for n in noise_levels]
        vals = [v for v in vals if not np.isnan(v) and v > 0]
        if vals:
            row.append(np.log10(np.mean(vals)))
        else:
            row.append(np.nan)
    matrix.append(row)

matrix = np.array(matrix)

im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=-2, vmax=2, interpolation='nearest')

ax.set_yticks(range(len(all_methods_sorted)))
ax.set_yticklabels([m.replace('_', ' ') for m in all_methods_sorted], fontsize=7)
ax.set_xticks(range(len(orders)))
ax.set_xticklabels([f'Order {o}' for o in orders], fontsize=10)
ax.set_title('log10(nRMSE) Heatmap: All Methods vs Derivative Order\n(Averaged over noise levels)',
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('log10(nRMSE)', fontsize=11)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot03_heatmap_method_order.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot03_heatmap_method_order.png")

# ===========================================================================
# PLOT 4: Small multiples - log10(nRMSE) vs noise for each order (all methods)
# ===========================================================================
print("[4/20] Generating small multiples: noise sensitivity at each order...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

colors = plt.cm.tab20(np.linspace(0, 1, len(full_coverage_sorted)))

for idx, order in enumerate(orders):
    ax = axes[idx]

    for midx, method in enumerate(full_coverage_sorted):
        nrmse_vals = [data.get((method, order, n), np.nan) for n in noise_levels]
        log_vals = [np.log10(v) if not np.isnan(v) and v > 0 else np.nan for v in nrmse_vals]

        ax.plot(range(len(noise_levels)), log_vals, 'o-',
                label=method.replace('_', ' ') if idx == 0 else '',
                color=colors[midx], linewidth=1.5, markersize=4, alpha=0.7)

    ax.set_title(f'Order {order}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('log10(nRMSE)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    if idx == 0:
        ax.legend(fontsize=6, ncol=2, loc='upper left')

plt.suptitle('Noise Sensitivity Across Derivative Orders (16 Full-Coverage Methods)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot04_small_multiples_noise.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot04_small_multiples_noise.png")

# ===========================================================================
# PLOT 5: Rank-ordered bar chart of mean log10(nRMSE) by method
# ===========================================================================
print("[5/20] Generating rank-ordered bar chart...")

fig, ax = plt.subplots(figsize=(10, 14))

# Calculate mean log10(nRMSE) for each method
method_means = {}
for method in all_methods_sorted:
    vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    vals = [v for v in vals if not np.isnan(v) and v > 0]
    if vals:
        method_means[method] = np.mean(np.log10(vals))
    else:
        method_means[method] = np.nan

# Sort by mean
sorted_methods = sorted(method_means.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 999)

method_names = [m[0].replace('_', ' ') for m in sorted_methods]
mean_vals = [m[1] for m in sorted_methods]

colors_bar = ['green' if v < 0 else 'orange' if v < 1 else 'red' for v in mean_vals]

ax.barh(range(len(method_names)), mean_vals, color=colors_bar, alpha=0.7)
ax.set_yticks(range(len(method_names)))
ax.set_yticklabels(method_names, fontsize=8)
ax.set_xlabel('Mean log10(nRMSE) Across All Conditions', fontsize=12)
ax.set_title('Method Ranking by Overall Performance\n(Green: nRMSE < 1, Orange: 1-10, Red: >10)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot05_ranking_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot05_ranking_bar_chart.png")

# ===========================================================================
# PLOT 6: Scatter - mean vs std of log10(nRMSE)
# ===========================================================================
print("[6/20] Generating scatter: mean vs std...")

fig, ax = plt.subplots(figsize=(12, 10))

means = []
stds = []
names = []
cats = []

for method in all_methods_sorted:
    vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    vals = [v for v in vals if not np.isnan(v) and v > 0]
    if vals:
        log_vals = np.log10(vals)
        means.append(np.mean(log_vals))
        stds.append(np.std(log_vals))
        names.append(method)
        cats.append(categories[method])

# Color by category
unique_cats = sorted(set(cats))
cat_colors = dict(zip(unique_cats, plt.cm.tab10(range(len(unique_cats)))))

for i, (m, s, n, c) in enumerate(zip(means, stds, names, cats)):
    ax.scatter(m, s, s=100, alpha=0.7, color=cat_colors[c], edgecolors='black', linewidths=0.5)
    ax.annotate(n.replace('_', ' '), (m, s), fontsize=6, alpha=0.8,
                xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('Mean log10(nRMSE)', fontsize=12)
ax.set_ylabel('Std Dev of log10(nRMSE)', fontsize=12)
ax.set_title('Performance Consistency: Mean vs Variability\n(Lower left = accurate & consistent)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Legend for categories
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors[c],
           markersize=8, label=c) for c in unique_cats]
ax.legend(handles=handles, fontsize=9, title='Category')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot06_scatter_mean_std.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot06_scatter_mean_std.png")

# ===========================================================================
# PLOT 7: Failure rate bar chart
# ===========================================================================
print("[7/20] Generating failure rate analysis...")

fig, ax = plt.subplots(figsize=(10, 14))

failure_threshold = 10  # nRMSE > 10 is "failure"
failure_counts = {}

for method in all_methods_sorted:
    count = 0
    total = 0
    for o in orders:
        for n in noise_levels:
            val = data.get((method, o, n), np.nan)
            if not np.isnan(val):
                total += 1
                if val > failure_threshold:
                    count += 1
    failure_counts[method] = (count / total * 100) if total > 0 else 0

sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)

method_names_f = [m[0].replace('_', ' ') for m in sorted_failures]
failure_rates = [m[1] for m in sorted_failures]

colors_f = ['red' if f > 50 else 'orange' if f > 10 else 'green' for f in failure_rates]

ax.barh(range(len(method_names_f)), failure_rates, color=colors_f, alpha=0.7)
ax.set_yticks(range(len(method_names_f)))
ax.set_yticklabels(method_names_f, fontsize=8)
ax.set_xlabel('Failure Rate (% of configs with nRMSE > 10)', fontsize=12)
ax.set_title('Method Reliability: Failure Rate Across All Conditions\n(Green: <10%, Orange: 10-50%, Red: >50%)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot07_failure_rate.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot07_failure_rate.png")

# ===========================================================================
# PLOT 8: Category-level box plots
# ===========================================================================
print("[8/20] Generating category comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

category_data = defaultdict(list)
for method in all_methods_sorted:
    cat = categories[method]
    vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    vals = [v for v in vals if not np.isnan(v) and v > 0]
    if vals:
        category_data[cat].extend(np.log10(vals))

cat_names = sorted(category_data.keys())
data_to_plot = [category_data[c] for c in cat_names]

bp = ax.boxplot(data_to_plot, labels=[c.replace('_', ' ') for c in cat_names],
                patch_artist=True, vert=True)

for patch, cat in zip(bp['boxes'], cat_names):
    patch.set_facecolor(cat_colors.get(cat, 'lightblue'))

ax.set_ylabel('log10(nRMSE)', fontsize=12)
ax.set_title('Performance Distribution by Method Category', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot08_category_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot08_category_boxplot.png")

# ===========================================================================
# PLOT 9: Line plot - category average vs derivative order
# ===========================================================================
print("[9/20] Generating category trends vs order...")

fig, ax = plt.subplots(figsize=(12, 8))

for cat in cat_names:
    methods_in_cat = [m for m in all_methods_sorted if categories[m] == cat]
    avg_by_order = []

    for order in orders:
        vals = []
        for method in methods_in_cat:
            method_vals = [data.get((method, order, n), np.nan) for n in noise_levels]
            method_vals = [v for v in method_vals if not np.isnan(v) and v > 0]
            if method_vals:
                vals.extend(method_vals)

        if vals:
            avg_by_order.append(np.mean(np.log10(vals)))
        else:
            avg_by_order.append(np.nan)

    ax.plot(orders, avg_by_order, 'o-', label=cat, linewidth=2.5, markersize=7,
            color=cat_colors.get(cat, 'gray'))

ax.set_xlabel('Derivative Order', fontsize=12)
ax.set_ylabel('Mean log10(nRMSE) [category average]', fontsize=12)
ax.set_title('Category Performance vs Derivative Order', fontsize=14, fontweight='bold')
ax.set_xticks(orders)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot09_category_vs_order.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot09_category_vs_order.png")

# ===========================================================================
# PLOT 10: Best method per condition heatmap
# ===========================================================================
print("[10/20] Generating best method per condition heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

# Find best method for each (order, noise) combination
best_method_grid = []
for order in orders:
    row = []
    for noise in noise_levels:
        best_nrmse = np.inf
        best_method = "None"

        for method in all_methods_sorted:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val < best_nrmse:
                best_nrmse = val
                best_method = method

        row.append(best_method)
    best_method_grid.append(row)

# Create numerical encoding for visualization
unique_methods = sorted(set([m for row in best_method_grid for m in row]))
method_to_num = {m: i for i, m in enumerate(unique_methods)}

matrix_num = [[method_to_num[m] for m in row] for row in best_method_grid]

im = ax.imshow(matrix_num, aspect='auto', cmap='tab20', interpolation='nearest')

ax.set_yticks(range(len(orders)))
ax.set_yticklabels([f'Order {o}' for o in orders], fontsize=10)
ax.set_xticks(range(len(noise_labels)))
ax.set_xticklabels(noise_labels, fontsize=10)
ax.set_xlabel('Noise Level', fontsize=12)
ax.set_ylabel('Derivative Order', fontsize=12)
ax.set_title('Best Performing Method Per Condition', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(orders)):
    for j in range(len(noise_levels)):
        text = best_method_grid[i][j].replace('_', '\n')
        ax.text(j, i, text, ha="center", va="center", fontsize=5,
                color="white", weight='bold')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot10_best_method_grid.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot10_best_method_grid.png")

print("\n[Continuing with plots 11-20...]")

# ===========================================================================
# PLOT 11: Stacked bar chart - performance categories
# ===========================================================================
print("[11/20] Generating stacked bar chart of performance categories...")

fig, ax = plt.subplots(figsize=(14, 10))

# Define performance bins
bins = [(0, 0.1, 'Excellent'), (0.1, 1.0, 'Good'), (1.0, 10, 'Acceptable'), (10, np.inf, 'Failed')]

method_bins = {}
for method in all_methods_sorted:
    counts = [0, 0, 0, 0]  # excellent, good, acceptable, failed

    for o in orders:
        for n in noise_levels:
            val = data.get((method, o, n), np.nan)
            if not np.isnan(val):
                for idx, (low, high, _) in enumerate(bins):
                    if low <= val < high:
                        counts[idx] += 1
                        break

    total = sum(counts)
    method_bins[method] = [c / total * 100 if total > 0 else 0 for c in counts]

# Sort by "excellent" percentage
sorted_methods_bins = sorted(method_bins.items(), key=lambda x: x[1][0], reverse=True)

method_names_bins = [m[0].replace('_', ' ') for m in sorted_methods_bins]
excellent = [m[1][0] for m in sorted_methods_bins]
good = [m[1][1] for m in sorted_methods_bins]
acceptable = [m[1][2] for m in sorted_methods_bins]
failed = [m[1][3] for m in sorted_methods_bins]

y_pos = range(len(method_names_bins))

p1 = ax.barh(y_pos, excellent, color='darkgreen', alpha=0.8, label='Excellent (nRMSE < 0.1)')
p2 = ax.barh(y_pos, good, left=excellent, color='yellowgreen', alpha=0.8, label='Good (0.1-1.0)')
p3 = ax.barh(y_pos, acceptable, left=np.array(excellent) + np.array(good),
             color='orange', alpha=0.8, label='Acceptable (1.0-10)')
p4 = ax.barh(y_pos, failed, left=np.array(excellent) + np.array(good) + np.array(acceptable),
             color='red', alpha=0.8, label='Failed (>10)')

ax.set_yticks(y_pos)
ax.set_yticklabels(method_names_bins, fontsize=7)
ax.set_xlabel('Percentage of Conditions', fontsize=12)
ax.set_title('Performance Category Distribution by Method\n(All Tested Conditions)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot11_stacked_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot11_stacked_performance.png")

# ===========================================================================
# PLOT 12: Performance against baseline (GP-Julia-AD)
# ===========================================================================
print("[12/20] Generating performance vs baseline...")

baseline = 'GP-Julia-AD'

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Select representative methods to compare
comparison_methods = ['AAA-HighPrec', 'AAA-LowPrec', 'Fourier-Interp',
                      'GP-Julia-SE', 'Savitzky-Golay', 'TrendFilter-k7',
                      'fourier', 'chebyshev']

for idx, order in enumerate(orders):
    ax = axes[idx]

    for method in comparison_methods:
        if method not in data:
            continue

        ratios = []
        for noise in noise_levels:
            baseline_val = data.get((baseline, order, noise), np.nan)
            method_val = data.get((method, order, noise), np.nan)

            if not np.isnan(baseline_val) and not np.isnan(method_val) and baseline_val > 0 and method_val > 0:
                ratio = np.log10(method_val / baseline_val)
                ratios.append(ratio)
            else:
                ratios.append(np.nan)

        ax.plot(range(len(noise_levels)), ratios, 'o-', label=method.replace('_', ' '),
                linewidth=1.5, markersize=5, alpha=0.7)

    ax.set_title(f'Order {order}', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('log10(nRMSE / nRMSE_GP-Julia-AD)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, label='Same as baseline')

    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')

plt.suptitle(f'Performance Relative to {baseline} (Baseline)\n(>0 = worse, <0 = better)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot12_vs_baseline.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot12_vs_baseline.png")

# ===========================================================================
# PLOT 13: Violin plots by method (top 12)
# ===========================================================================
print("[13/20] Generating violin plots for top methods...")

# Get top 12 by median performance
method_medians = {}
for method in all_methods_sorted:
    vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    vals = [v for v in vals if not np.isnan(v) and v > 0]
    if vals:
        method_medians[method] = np.median(np.log10(vals))
    else:
        method_medians[method] = np.inf

top12_methods = sorted(method_medians.items(), key=lambda x: x[1])[:12]

fig, ax = plt.subplots(figsize=(14, 8))

violin_data = []
violin_labels = []

for method, _ in top12_methods:
    vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    vals = [v for v in vals if not np.isnan(v) and v > 0]
    if vals:
        violin_data.append(np.log10(vals))
        violin_labels.append(method.replace('_', ' '))

parts = ax.violinplot(violin_data, vert=True, showmedians=True, showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

ax.set_xticks(range(1, len(violin_labels) + 1))
ax.set_xticklabels(violin_labels, fontsize=9, rotation=45, ha='right')
ax.set_ylabel('log10(nRMSE)', fontsize=12)
ax.set_title('Performance Distribution: Top 12 Methods\n(Violin plots showing full distribution)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot13_violin_top12.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot13_violin_top12.png")

# ===========================================================================
# PLOT 14: Heatmap with small multiples - method vs noise (one per order)
# ===========================================================================
print("[14/20] Generating heatmap small multiples (method vs noise by order)...")

fig, axes = plt.subplots(2, 4, figsize=(20, 14))
axes = axes.flatten()

for idx, order in enumerate(orders):
    ax = axes[idx]

    # Build matrix: full-coverage methods × noise levels
    matrix = []
    for method in full_coverage_sorted:
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

    ax.set_title(f'Order {order}', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(full_coverage_sorted)))
    ax.set_yticklabels([m.replace('_', ' ') for m in full_coverage_sorted], fontsize=6)
    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(noise_labels, fontsize=8, rotation=45, ha='right')

    if idx == 7:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log10(nRMSE)', fontsize=9)

plt.suptitle('Method × Noise Level Heatmaps (One per Derivative Order)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot14_heatmap_multiples.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot14_heatmap_multiples.png")

# ===========================================================================
# PLOT 15: Accuracy vs Robustness scatter
# ===========================================================================
print("[15/20] Generating accuracy vs robustness scatter...")

fig, ax = plt.subplots(figsize=(12, 10))

# Define "easy" and "hard" conditions
easy_orders = [0, 1, 2]
easy_noise = [1e-8, 1e-6, 1e-4]
hard_orders = [5, 6, 7]
hard_noise = [0.01, 0.02, 0.05]

accuracy_scores = []
robustness_scores = []
method_names_scatter = []
method_cats = []

for method in all_methods_sorted:
    # Accuracy: performance on easy conditions
    easy_vals = []
    for o in easy_orders:
        for n in easy_noise:
            val = data.get((method, o, n), np.nan)
            if not np.isnan(val) and val > 0:
                easy_vals.append(val)

    # Robustness: performance on hard conditions
    hard_vals = []
    for o in hard_orders:
        for n in hard_noise:
            val = data.get((method, o, n), np.nan)
            if not np.isnan(val) and val > 0:
                hard_vals.append(val)

    if easy_vals and hard_vals:
        accuracy_scores.append(np.mean(np.log10(easy_vals)))
        robustness_scores.append(np.mean(np.log10(hard_vals)))
        method_names_scatter.append(method)
        method_cats.append(categories[method])

# Plot
for i, (acc, rob, name, cat) in enumerate(zip(accuracy_scores, robustness_scores,
                                                method_names_scatter, method_cats)):
    ax.scatter(acc, rob, s=120, alpha=0.7, color=cat_colors.get(cat, 'gray'),
               edgecolors='black', linewidths=0.5)
    ax.annotate(name.replace('_', ' '), (acc, rob), fontsize=7, alpha=0.8,
                xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('Accuracy Score [mean log10(nRMSE) on easy conditions]', fontsize=12)
ax.set_ylabel('Robustness Score [mean log10(nRMSE) on hard conditions]', fontsize=12)
ax.set_title('Method Positioning: Accuracy vs Robustness\n(Lower left = accurate & robust)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add quadrant lines at 0
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors[c],
           markersize=8, label=c) for c in unique_cats]
ax.legend(handles=handles, fontsize=9, title='Category', loc='upper left')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot15_accuracy_robustness.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot15_accuracy_robustness.png")

# ===========================================================================
# PLOT 16: Performance at high-order derivatives only (orders 5-7)
# ===========================================================================
print("[16/20] Generating high-order performance analysis...")

fig, ax = plt.subplots(figsize=(14, 8))

high_orders = [5, 6, 7]

# Calculate average performance at high orders
method_high_order_perf = {}
for method in all_methods_sorted:
    vals = []
    for o in high_orders:
        for n in noise_levels:
            val = data.get((method, o, n), np.nan)
            if not np.isnan(val) and val > 0:
                vals.append(val)

    if vals:
        method_high_order_perf[method] = np.mean(np.log10(vals))
    else:
        method_high_order_perf[method] = np.nan

sorted_high = sorted([(m, v) for m, v in method_high_order_perf.items() if not np.isnan(v)],
                     key=lambda x: x[1])

method_names_high = [m[0].replace('_', ' ') for m in sorted_high]
perf_high = [m[1] for m in sorted_high]

colors_high = ['green' if p < 0 else 'orange' if p < 1 else 'red' for p in perf_high]

ax.barh(range(len(method_names_high)), perf_high, color=colors_high, alpha=0.7)
ax.set_yticks(range(len(method_names_high)))
ax.set_yticklabels(method_names_high, fontsize=8)
ax.set_xlabel('Mean log10(nRMSE) at High Derivative Orders (5-7)', fontsize=12)
ax.set_title('Method Performance at Extreme Challenge\n(Orders 5-7 only)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot16_high_order_only.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot16_high_order_only.png")

# ===========================================================================
# PLOT 17: Symlog scale heatmap (handles extreme range better)
# ===========================================================================
print("[17/20] Generating symlog heatmap...")

from matplotlib.colors import SymLogNorm

fig, ax = plt.subplots(figsize=(10, 14))

# Build matrix: methods × orders (mean across noise)
matrix = []
for method in all_methods_sorted:
    row = []
    for order in orders:
        vals = [data.get((method, order, n), np.nan) for n in noise_levels]
        vals = [v for v in vals if not np.isnan(v) and v > 0]
        if vals:
            row.append(np.mean(vals))
        else:
            row.append(np.nan)
    matrix.append(row)

matrix = np.array(matrix)

# Use symlog normalization
im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r',
               norm=SymLogNorm(linthresh=1.0, vmin=0.001, vmax=100),
               interpolation='nearest')

ax.set_yticks(range(len(all_methods_sorted)))
ax.set_yticklabels([m.replace('_', ' ') for m in all_methods_sorted], fontsize=7)
ax.set_xticks(range(len(orders)))
ax.set_xticklabels([f'Order {o}' for o in orders], fontsize=10)
ax.set_title('nRMSE Heatmap with SymLog Scale\n(Better handling of extreme range)',
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('nRMSE (symlog scale)', fontsize=11)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot17_symlog_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot17_symlog_heatmap.png")

# ===========================================================================
# PLOT 18: Order × Noise grid showing count of viable methods (nRMSE < 1)
# ===========================================================================
print("[18/20] Generating viable method count grid...")

fig, ax = plt.subplots(figsize=(10, 8))

# Count viable methods per condition
viable_counts = []
for order in orders:
    row = []
    for noise in noise_levels:
        count = 0
        for method in all_methods_sorted:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val < 1.0:
                count += 1
        row.append(count)
    viable_counts.append(row)

viable_counts = np.array(viable_counts)

im = ax.imshow(viable_counts, aspect='auto', cmap='YlGnBu', interpolation='nearest')

ax.set_yticks(range(len(orders)))
ax.set_yticklabels([f'Order {o}' for o in orders], fontsize=10)
ax.set_xticks(range(len(noise_labels)))
ax.set_xticklabels(noise_labels, fontsize=10)
ax.set_xlabel('Noise Level', fontsize=12)
ax.set_ylabel('Derivative Order', fontsize=12)
ax.set_title('Number of Viable Methods Per Condition\n(nRMSE < 1.0)',
             fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(orders)):
    for j in range(len(noise_levels)):
        text = ax.text(j, i, f'{viable_counts[i, j]}',
                      ha="center", va="center", fontsize=10,
                      color="white" if viable_counts[i, j] > 13 else "black",
                      weight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Count of viable methods', fontsize=11)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot18_viable_count_grid.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot18_viable_count_grid.png")

# ===========================================================================
# PLOT 19: Ridge plot of performance distributions by order
# ===========================================================================
print("[19/20] Generating ridge plot (performance distributions by order)...")

from matplotlib.collections import PolyCollection

fig, ax = plt.subplots(figsize=(14, 10))

# Collect log10(nRMSE) distributions for each order (all methods, all noise)
order_distributions = []
for order in orders:
    vals = []
    for method in all_methods_sorted:
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            if not np.isnan(val) and val > 0:
                vals.append(np.log10(val))
    order_distributions.append(vals)

# Create ridge plot (stacked density plots)
positions = []
for i, vals in enumerate(order_distributions):
    if vals:
        # Create histogram
        hist, bins = np.histogram(vals, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Scale for visibility
        hist_scaled = hist * 0.8 + i

        # Plot
        ax.fill_between(bin_centers, i, hist_scaled, alpha=0.6,
                        color=plt.cm.viridis(i / len(orders)))
        ax.plot(bin_centers, hist_scaled, color='black', linewidth=0.5)

ax.set_ylim(-0.5, len(orders))
ax.set_yticks(range(len(orders)))
ax.set_yticklabels([f'Order {o}' for o in orders], fontsize=11)
ax.set_xlabel('log10(nRMSE)', fontsize=12)
ax.set_title('Performance Distribution Evolution Across Derivative Orders\n(Ridge plot: all methods, all noise levels)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot19_ridge_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot19_ridge_plot.png")

# ===========================================================================
# PLOT 20: Coverage visualization - which methods work where
# ===========================================================================
print("[20/20] Generating coverage visualization...")

fig, ax = plt.subplots(figsize=(12, 14))

# Build binary matrix: method has data for (order, noise)?
coverage_matrix = []
for method in all_methods_sorted:
    row = []
    for order in orders:
        for noise in noise_levels:
            val = data.get((method, order, noise), np.nan)
            # 1 if tested, 0 if not, -1 if catastrophic failure
            if np.isnan(val):
                row.append(0)
            elif val > 1000:
                row.append(-1)
            else:
                row.append(1)
    coverage_matrix.append(row)

coverage_matrix = np.array(coverage_matrix)

# Custom colormap: gray for missing, red for catastrophic, green gradient for present
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['lightgray', 'darkgreen', 'red'])

im = ax.imshow(coverage_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

ax.set_yticks(range(len(all_methods_sorted)))
ax.set_yticklabels([m.replace('_', ' ') for m in all_methods_sorted], fontsize=7)

# X-axis: order-noise combinations
combo_labels = [f'O{o}\n{noise_labels[n_idx][:4]}'
                for o in orders for n_idx in range(len(noise_levels))]
ax.set_xticks(range(len(combo_labels)))
ax.set_xticklabels(combo_labels, fontsize=5, rotation=90)
ax.set_xlabel('Order-Noise Combinations', fontsize=12)
ax.set_title('Method Coverage Map\n(Green: tested, Gray: not tested, Red: catastrophic failure)',
             fontsize=14, fontweight='bold')

# Add vertical lines to separate orders
for i in range(1, len(orders)):
    ax.axvline(x=i * len(noise_levels) - 0.5, color='black', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v2/plot20_coverage_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot20_coverage_map.png")

print("\n" + "="*70)
print("COMPLETE! Generated 20 comprehensive plots in paper_figures/supplementary_v2/")
print("="*70)
print("\nPlot Summary:")
print("  1. Box plots (all methods)")
print("  2. Line plots (order progression, all full-coverage methods)")
print("  3. Heatmap (method × order)")
print("  4. Small multiples (noise sensitivity)")
print("  5. Rank bar chart")
print("  6. Mean vs Std scatter")
print("  7. Failure rate analysis")
print("  8. Category box plots")
print("  9. Category trends")
print(" 10. Best method grid")
print(" 11. Stacked performance bars")
print(" 12. Performance vs baseline")
print(" 13. Violin plots (top 12)")
print(" 14. Heatmap small multiples")
print(" 15. Accuracy vs robustness")
print(" 16. High-order performance")
print(" 17. Symlog heatmap")
print(" 18. Viable method count")
print(" 19. Ridge plot")
print(" 20. Coverage map")
