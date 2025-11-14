#!/usr/bin/env python3
"""
Generate supplemental heatmaps for specific derivative order ranges.

Outputs:
1. Heatmap for methods covering orders 1-5 (ranked by avg over 1-5)
2. Heatmap for methods covering orders 6-7 (ranked by avg over 6-7)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'text.usetex': False
})

print("=" * 80)
print("GENERATING SUPPLEMENTAL HEATMAPS")
print("=" * 80)

# Load data
results_dir = Path(__file__).parent.parent / "build" / "results" / "comprehensive"
summary = pd.read_csv(results_dir / "comprehensive_summary.csv")

print(f"\nLoaded {len(summary)} rows from summary")
print(f"Unique methods: {summary['method'].nunique()}")

# Create output directory
output_dir = Path(__file__).parent.parent / "build" / "figures" / "supplemental"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HEATMAP 1: Methods with orders 1-5 (ranked by average over 1-5)
# ============================================================================

print("\n" + "=" * 80)
print("HEATMAP 1: Orders 1-5 Coverage")
print("=" * 80)

# Filter to orders 1-5 (excluding order 0 function approximation)
low_order_data = summary[summary['deriv_order'].isin([1, 2, 3, 4, 5])].copy()

# Find methods that have ALL of orders 1-5
method_order_counts = low_order_data.groupby('method')['deriv_order'].nunique()
methods_with_1to5 = method_order_counts[method_order_counts == 5].index.tolist()

print(f"Methods with full coverage of orders 1-5: {len(methods_with_1to5)}")

# Filter to those methods and rank by average nRMSE over orders 1-5
filtered_data = low_order_data[low_order_data['method'].isin(methods_with_1to5)]
avg_performance = filtered_data.groupby('method')['mean_nrmse'].mean().sort_values()

# Take top 20 methods (or fewer if not enough)
top_n = min(20, len(avg_performance))
top_methods_1to5 = avg_performance.head(top_n).index.tolist()

print(f"Top {top_n} methods by avg nRMSE over orders 1-5:")
for i, method in enumerate(top_methods_1to5[:5], 1):
    print(f"  {i}. {method}: {avg_performance[method]:.4f}")
print("  ...")

# Create pivot table for heatmap
heatmap_data_1to5 = filtered_data[filtered_data['method'].isin(top_methods_1to5)].pivot_table(
    index='method',
    columns='deriv_order',
    values='mean_nrmse',
    aggfunc='mean'
)

# Sort by average across orders
heatmap_data_1to5['avg'] = heatmap_data_1to5.mean(axis=1)
heatmap_data_1to5 = heatmap_data_1to5.sort_values('avg')
heatmap_data_1to5 = heatmap_data_1to5.drop('avg', axis=1)

# Plot heatmap
fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(
    heatmap_data_1to5,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn_r',  # Red = bad, Green = good
    vmin=0,
    vmax=1.0,
    cbar_kws={'label': 'nRMSE'},
    linewidths=0.5,
    ax=ax
)
ax.set_xlabel('Derivative Order', fontweight='bold')
ax.set_ylabel('Method', fontweight='bold')
ax.set_title(f'Top {top_n} Methods: Orders 1-5\n(Ranked by Avg nRMSE over Orders 1-5)',
             fontweight='bold')
plt.tight_layout()

output_file_1to5 = output_dir / "heatmap_orders_1to5.png"
plt.savefig(output_file_1to5, bbox_inches='tight', dpi=300)
plt.close()

print(f"\nSaved: {output_file_1to5}")

# ============================================================================
# HEATMAP 2: Methods with orders 6-7 (ranked by average over 6-7)
# ============================================================================

print("\n" + "=" * 80)
print("HEATMAP 2: High Orders 6-7 Coverage")
print("=" * 80)

# Filter to orders 6-7
high_order_data = summary[summary['deriv_order'].isin([6, 7])].copy()

# Find methods that have BOTH orders 6 and 7
method_order_counts_high = high_order_data.groupby('method')['deriv_order'].nunique()
methods_with_6and7 = method_order_counts_high[method_order_counts_high == 2].index.tolist()

print(f"Methods with coverage of both orders 6 and 7: {len(methods_with_6and7)}")

# Filter to those methods and rank by average nRMSE over orders 6-7
filtered_data_high = high_order_data[high_order_data['method'].isin(methods_with_6and7)]
avg_performance_high = filtered_data_high.groupby('method')['mean_nrmse'].mean().sort_values()

# Take top 20 methods (or fewer if not enough)
top_n_high = min(20, len(avg_performance_high))
top_methods_6to7 = avg_performance_high.head(top_n_high).index.tolist()

print(f"Top {top_n_high} methods by avg nRMSE over orders 6-7:")
for i, method in enumerate(top_methods_6to7[:5], 1):
    print(f"  {i}. {method}: {avg_performance_high[method]:.4f}")
print("  ...")

# Create pivot table for heatmap
heatmap_data_6to7 = filtered_data_high[filtered_data_high['method'].isin(top_methods_6to7)].pivot_table(
    index='method',
    columns='deriv_order',
    values='mean_nrmse',
    aggfunc='mean'
)

# Sort by average across orders
heatmap_data_6to7['avg'] = heatmap_data_6to7.mean(axis=1)
heatmap_data_6to7 = heatmap_data_6to7.sort_values('avg')
heatmap_data_6to7 = heatmap_data_6to7.drop('avg', axis=1)

# Plot heatmap
fig, ax = plt.subplots(figsize=(5, 10))
sns.heatmap(
    heatmap_data_6to7,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn_r',  # Red = bad, Green = good
    vmin=0,
    vmax=2.0,  # Higher range for high-order derivatives
    cbar_kws={'label': 'nRMSE'},
    linewidths=0.5,
    ax=ax
)
ax.set_xlabel('Derivative Order', fontweight='bold')
ax.set_ylabel('Method', fontweight='bold')
ax.set_title(f'Top {top_n_high} Methods: Orders 6-7\n(Ranked by Avg nRMSE over Orders 6-7)',
             fontweight='bold')
plt.tight_layout()

output_file_6to7 = output_dir / "heatmap_orders_6to7.png"
plt.savefig(output_file_6to7, bbox_inches='tight', dpi=300)
plt.close()

print(f"\nSaved: {output_file_6to7}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUPPLEMENTAL HEATMAPS COMPLETE")
print("=" * 80)
print(f"\nGenerated 2 heatmaps:")
print(f"  1. Orders 1-5: {output_file_1to5.name}")
print(f"  2. Orders 6-7: {output_file_6to7.name}")
print(f"\nOutput directory: {output_dir}")
print("=" * 80)
