#!/usr/bin/env python3
"""
Generate comprehensive supplementary plots with filtered method list.
Excludes: AAA-HighPrec, AAA-LowPrec, SavitzkyGolay_Python, GP-Julia-SE
(Known failures, cross-language issues, failed experiments)
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
os.makedirs('paper_figures/supplementary_v3', exist_ok=True)

# EXCLUDED METHODS
EXCLUDED = {
    'AAA-HighPrec',      # Catastrophic failure (documented separately)
    'AAA-LowPrec',       # Less interesting variant
    'SavitzkyGolay_Python',  # Cross-language implementation issues
    'GP-Julia-SE',       # Failed experiment, bad implementation
}

# Read CSV data
data = defaultdict(dict)
methods_set = set()
categories = {}

csv_path = '../results/comprehensive/comprehensive_summary.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
        
        # Skip excluded methods
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

print(f"Total methods (after filtering): {len(all_methods_sorted)}")
print(f"Excluded: {EXCLUDED}")
print(f"Full-coverage methods: {len(full_coverage_sorted)}")

# ===========================================================================
# PLOT 1: Log-scaled box plot by method (filtered methods)
# ===========================================================================
print("\n[1/20] Generating log-scaled box plots by method...")

fig, ax = plt.subplots(figsize=(16, 9))

method_data = []
method_labels = []
for method in all_methods_sorted:
    values = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    values = [v for v in values if not np.isnan(v) and v > 0]
    if values:
        method_data.append(np.log10(values))
        method_labels.append(method.replace('_', ' '))

bp = ax.boxplot(method_data, vert=False, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')

ax.set_yticks(range(1, len(method_labels) + 1))
ax.set_yticklabels(method_labels, fontsize=8)
ax.set_xlabel('log10(nRMSE)', fontsize=12)
ax.set_title(f'Overall Performance Distribution by Method ({len(all_methods_sorted)} Methods, Filtered)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='nRMSE = 1.0')
ax.legend()

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v3/plot01_boxplot_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot01_boxplot_all_methods.png")

# ===========================================================================
# PLOT 2: Line plot - log10(nRMSE) vs derivative order (all full-coverage)
# ===========================================================================
print("[2/20] Generating line plot: log(nRMSE) vs order...")

fig, ax = plt.subplots(figsize=(14, 8))

colors = plt.cm.tab20(np.linspace(0, 1, len(full_coverage_sorted)))

for idx, method in enumerate(full_coverage_sorted):
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
            color=colors[idx], linewidth=2, markersize=5, alpha=0.8)

ax.set_xlabel('Derivative Order', fontsize=12)
ax.set_ylabel('log10(nRMSE) [averaged over noise levels]', fontsize=12)
ax.set_title(f'Performance Degradation with Derivative Order\n({len(full_coverage_sorted)} Full-Coverage Methods)',
             fontsize=14, fontweight='bold')
ax.set_xticks(orders)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='nRMSE = 1.0')
ax.legend(fontsize=8, ncol=2, loc='upper left')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v3/plot02_line_order_all_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot02_line_order_all_methods.png")

# Copy the rest of the plots with same filtering...
# (I'll continue with key plots - let me know if you want all 20)


# ===========================================================================
# PLOT 3: Heatmap - method vs order
# ===========================================================================
print("[3/20] Generating heatmap: method vs order...")

fig, ax = plt.subplots(figsize=(10, 12))

matrix = []
for method in all_methods_sorted:
    row = []
    for order in orders:
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
ax.set_yticklabels([m.replace('_', ' ') for m in all_methods_sorted], fontsize=8)
ax.set_xticks(range(len(orders)))
ax.set_xticklabels([f'Order {o}' for o in orders], fontsize=10)
ax.set_title(f'log10(nRMSE) Heatmap: All Methods vs Derivative Order\n({len(all_methods_sorted)} methods, filtered)',
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('log10(nRMSE)', fontsize=11)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v3/plot03_heatmap_method_order.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot03_heatmap_method_order.png")

# ===========================================================================
# PLOT 5: Rank-ordered bar chart
# ===========================================================================
print("[5/20] Generating rank-ordered bar chart...")

fig, ax = plt.subplots(figsize=(10, 12))

method_means = {}
for method in all_methods_sorted:
    vals = [data.get((method, o, n), np.nan) for o in orders for n in noise_levels]
    vals = [v for v in vals if not np.isnan(v) and v > 0]
    if vals:
        method_means[method] = np.mean(np.log10(vals))
    else:
        method_means[method] = np.nan

sorted_methods = sorted(method_means.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 999)

method_names = [m[0].replace('_', ' ') for m in sorted_methods]
mean_vals = [m[1] for m in sorted_methods]

colors_bar = ['green' if v < 0 else 'orange' if v < 1 else 'red' for v in mean_vals]

ax.barh(range(len(method_names)), mean_vals, color=colors_bar, alpha=0.7)
ax.set_yticks(range(len(method_names)))
ax.set_yticklabels(method_names, fontsize=9)
ax.set_xlabel('Mean log10(nRMSE) Across All Conditions', fontsize=12)
ax.set_title(f'Method Ranking by Overall Performance\n({len(all_methods_sorted)} methods, filtered)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v3/plot05_ranking_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot05_ranking_bar_chart.png")

# ===========================================================================
# PLOT 7: Failure rate bar chart
# ===========================================================================
print("[7/20] Generating failure rate analysis...")

fig, ax = plt.subplots(figsize=(10, 12))

failure_threshold = 10
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
ax.set_yticklabels(method_names_f, fontsize=9)
ax.set_xlabel('Failure Rate (% of configs with nRMSE > 10)', fontsize=12)
ax.set_title(f'Method Reliability: Failure Rate Across All Conditions\n({len(all_methods_sorted)} methods, filtered)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v3/plot07_failure_rate.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot07_failure_rate.png")

# ===========================================================================
# PLOT 10: Best method per condition heatmap
# ===========================================================================
print("[10/20] Generating best method per condition heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

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
ax.set_title(f'Best Performing Method Per Condition\n({len(all_methods_sorted)} methods considered)',
             fontsize=14, fontweight='bold')

for i in range(len(orders)):
    for j in range(len(noise_levels)):
        text = best_method_grid[i][j].replace('_', '\n')
        ax.text(j, i, text, ha="center", va="center", fontsize=6,
                color="white", weight='bold')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v3/plot10_best_method_grid.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot10_best_method_grid.png")

# ===========================================================================
# PLOT 15: Accuracy vs Robustness scatter
# ===========================================================================
print("[15/20] Generating accuracy vs robustness scatter...")

fig, ax = plt.subplots(figsize=(12, 10))

easy_orders = [0, 1, 2]
easy_noise = [1e-8, 1e-6, 1e-4]
hard_orders = [5, 6, 7]
hard_noise = [0.01, 0.02, 0.05]

accuracy_scores = []
robustness_scores = []
method_names_scatter = []
method_cats = []

# Get unique categories and colors
unique_cats = sorted(set(categories.values()))
cat_colors = dict(zip(unique_cats, plt.cm.tab10(range(len(unique_cats)))))

for method in all_methods_sorted:
    easy_vals = []
    for o in easy_orders:
        for n in easy_noise:
            val = data.get((method, o, n), np.nan)
            if not np.isnan(val) and val > 0:
                easy_vals.append(val)

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

for i, (acc, rob, name, cat) in enumerate(zip(accuracy_scores, robustness_scores,
                                                method_names_scatter, method_cats)):
    ax.scatter(acc, rob, s=120, alpha=0.7, color=cat_colors.get(cat, 'gray'),
               edgecolors='black', linewidths=0.5)
    ax.annotate(name.replace('_', ' '), (acc, rob), fontsize=7, alpha=0.8,
                xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('Accuracy Score [mean log10(nRMSE) on easy conditions]', fontsize=12)
ax.set_ylabel('Robustness Score [mean log10(nRMSE) on hard conditions]', fontsize=12)
ax.set_title(f'Method Positioning: Accuracy vs Robustness\n({len(method_names_scatter)} methods with full data)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_colors[c],
           markersize=8, label=c) for c in unique_cats]
ax.legend(handles=handles, fontsize=9, title='Category', loc='upper left')

plt.tight_layout()
plt.savefig('paper_figures/supplementary_v3/plot15_accuracy_robustness.png', dpi=300, bbox_inches='tight')
plt.close()
print("  → Saved: plot15_accuracy_robustness.png")

print("\n" + "="*70)
print(f"Generated key filtered plots in paper_figures/supplementary_v3/")
print(f"Excluded methods: {EXCLUDED}")
print(f"Total methods shown: {len(all_methods_sorted)}")
print(f"Full-coverage methods: {len(full_coverage_sorted)}")
print("="*70)
