#!/usr/bin/env python3
"""
Generate publication-quality tables and plots for the derivative estimation paper.
Creates individual tables for each derivative order and filtered plots.
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
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'text.usetex': False
})

print("="*80)
print("GENERATING PAPER TABLES AND PLOTS")
print("="*80)

# Load data
results_dir = Path(__file__).parent.parent / "results" / "comprehensive"
summary = pd.read_csv(results_dir / "comprehensive_summary.csv")

print(f"\nLoaded {len(summary)} rows from summary data")

# Noise levels to analyze
noise_levels = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2]
orders = list(range(8))  # 0-7

# Output directory
output_dir = Path(__file__).parent.parent / "report" / "paper_figures"
output_dir.mkdir(exist_ok=True)

print(f"Output directory: {output_dir}")

# ============================================================================
# Generate tables for each derivative order
# ============================================================================

print("\n" + "="*80)
print("GENERATING TABLES")
print("="*80)

tables_dir = output_dir / "tables"
tables_dir.mkdir(exist_ok=True)

for order in orders:
    print(f"\nProcessing order {order}...")

    # Filter data for this order
    order_data = summary[summary['deriv_order'] == order].copy()

    # Pivot table: methods x noise levels
    pivot = order_data.pivot_table(
        index='method',
        columns='noise_level',
        values='mean_nrmse',
        aggfunc='mean'
    )

    # Sort by average nRMSE across all noise levels
    pivot['average'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('average')

    # Format values: cap at 1000, format to 2 decimals
    def format_value(x):
        if pd.isna(x):
            return "—"
        elif x > 1000:
            return ">1e3"
        elif x > 10:
            return f"{x:.0f}"
        else:
            return f"{x:.2f}"

    # Create formatted table
    formatted = pivot.map(format_value)

    # Rename columns to scientific notation
    col_rename = {nl: f"{nl:.0e}" for nl in noise_levels}
    col_rename['average'] = 'Mean'
    formatted.rename(columns=col_rename, inplace=True)

    # Save as LaTeX
    latex_file = tables_dir / f"order_{order}_nrmse.tex"
    with open(latex_file, 'w') as f:
        f.write("% nRMSE values for derivative order {}\n".format(order))
        f.write(formatted.to_latex(
            caption=f"Normalized RMSE for derivative order {order}",
            label=f"tab:nrmse_order_{order}",
            escape=False
        ))

    # Save as CSV for reference
    csv_file = tables_dir / f"order_{order}_nrmse.csv"
    pivot.to_csv(csv_file)

    print(f"  Saved: {latex_file.name} and {csv_file.name}")
    print(f"  Methods: {len(pivot)}, Best: {pivot.index[0]} (nRMSE={pivot.iloc[0]['average']:.3f})")


# ============================================================================
# Generate plots for each derivative order (filtered by threshold)
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

plots_dir = output_dir / "plots"
plots_dir.mkdir(exist_ok=True)

THRESHOLD = 1.0  # nRMSE threshold for plotting

for order in orders:
    print(f"\nPlotting order {order}...")

    # Filter data for this order
    order_data = summary[summary['deriv_order'] == order].copy()

    # Calculate average nRMSE per method
    method_avg = order_data.groupby('method')['mean_nrmse'].mean()

    # Filter methods below threshold
    good_methods = method_avg[method_avg <= THRESHOLD].index.tolist()

    if len(good_methods) == 0:
        print(f"  No methods below threshold {THRESHOLD} for order {order}")
        continue

    print(f"  Methods below threshold: {len(good_methods)}")

    # Filter data
    plot_data = order_data[order_data['method'].isin(good_methods)]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in good_methods:
        method_data = plot_data[plot_data['method'] == method].sort_values('noise_level')
        ax.plot(
            method_data['noise_level'],
            method_data['mean_nrmse'],
            marker='o',
            label=method,
            linewidth=1.5,
            markersize=5
        )

    ax.set_xscale('log')
    ax.set_xlabel('Noise Level (fraction of signal std)')
    ax.set_ylabel('Normalized RMSE (nRMSE)')
    ax.set_title(f'Derivative Order {order}: Methods with nRMSE < {THRESHOLD}')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.set_ylim(0, THRESHOLD * 1.1)

    # Add interpretation bands
    ax.axhspan(0, 0.1, alpha=0.1, color='green', label='Excellent (<0.1)')
    ax.axhspan(0.1, 0.3, alpha=0.1, color='yellow', label='Moderate (0.1-0.3)')
    ax.axhspan(0.3, THRESHOLD, alpha=0.1, color='orange', label='Acceptable (0.3-1.0)')

    plt.tight_layout()

    # Save
    plot_file = plots_dir / f"order_{order}_nrmse_filtered.pdf"
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Saved: {plot_file.name}")

# ============================================================================
# Generate summary heatmap across all orders
# ============================================================================

print("\n" + "="*80)
print("GENERATING SUMMARY HEATMAP")
print("="*80)

# Get top 15 methods overall
overall_avg = summary.groupby('method')['mean_nrmse'].mean().sort_values()
top_methods = overall_avg.head(15).index.tolist()

# Create pivot for heatmap
heatmap_data = summary[summary['method'].isin(top_methods)].pivot_table(
    index='method',
    columns='deriv_order',
    values='mean_nrmse',
    aggfunc='mean'
)

# Sort by average across orders
heatmap_data['avg'] = heatmap_data.mean(axis=1)
heatmap_data = heatmap_data.sort_values('avg')
heatmap_data = heatmap_data.drop('avg', axis=1)

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn_r',
    vmin=0,
    vmax=1.0,
    cbar_kws={'label': 'nRMSE'},
    linewidths=0.5,
    ax=ax
)
ax.set_xlabel('Derivative Order')
ax.set_ylabel('Method')
ax.set_title('Top 15 Methods: nRMSE Across Derivative Orders')
plt.tight_layout()

heatmap_file = plots_dir / "top_methods_heatmap.pdf"
plt.savefig(heatmap_file, bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved: {heatmap_file.name}")

print("\n" + "="*80)
print("TABLE AND PLOT GENERATION COMPLETE")
print("="*80)
print(f"\nTables: {tables_dir}")
print(f"Plots: {plots_dir}")
print("="*80)
