#!/usr/bin/env python3
"""
Generate comprehensive per-method and per-order visualizations.

Outputs:
1. For each METHOD: 2D heatmap (noise × order) showing nRMSE
2. For each METHOD: Line plot (noise vs order)
3. For each ORDER: Bar/line plot comparing all methods at different noise levels
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
print("GENERATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Load data
results_dir = Path(__file__).parent.parent / "build" / "results" / "comprehensive"
summary = pd.read_csv(results_dir / "comprehensive_summary.csv")

print(f"\nLoaded {len(summary)} rows")
print(f"Unique methods: {summary['method'].nunique()}")
print(f"Noise levels: {sorted(summary['noise_level'].unique())}")
print(f"Derivative orders: {sorted(summary['deriv_order'].unique())}")

# Extract info
noise_levels = sorted(summary['noise_level'].unique())
orders = sorted(summary['deriv_order'].unique())
methods = sorted(summary['method'].unique())

# Create output directories
per_method_dir = Path(__file__).parent.parent / "build" / "figures" / "supplemental" / "per_method"
per_order_dir = Path(__file__).parent.parent / "build" / "figures" / "supplemental" / "per_order"
per_method_dir.mkdir(parents=True, exist_ok=True)
per_order_dir.mkdir(parents=True, exist_ok=True)

print(f"\nOutput directories:")
print(f"  Per-method: {per_method_dir}")
print(f"  Per-order:  {per_order_dir}")

# =============================================================================
# PART 1: Per-Method Heatmaps (Noise × Order → nRMSE)
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: GENERATING PER-METHOD HEATMAPS")
print("=" * 80)

for i, method in enumerate(methods, 1):
    print(f"[{i}/{len(methods)}] {method}...")

    # Filter data for this method
    method_data = summary[summary['method'] == method].copy()

    if len(method_data) == 0:
        print(f"  WARNING: No data for {method}, skipping")
        continue

    # Pivot to create noise × order grid
    pivot = method_data.pivot_table(
        index='noise_level',
        columns='deriv_order',
        values='mean_nrmse',
        aggfunc='mean'
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for color, cap at reasonable values
    pivot_capped = pivot.clip(upper=10.0)  # Cap extreme values for better visualization

    # Create heatmap with diverging colormap
    sns.heatmap(
        pivot_capped,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Red = bad (high nRMSE), Green = good (low nRMSE)
        vmin=0,
        vmax=2.0,  # Most methods should be < 2.0 nRMSE
        cbar_kws={'label': 'nRMSE'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    # Format labels
    ax.set_xlabel('Derivative Order', fontweight='bold')
    ax.set_ylabel('Noise Level', fontweight='bold')
    ax.set_title(f'{method}\nPerformance Heatmap (nRMSE)', fontweight='bold')

    # Format y-axis to show noise levels in scientific notation
    y_labels = [f"{nl:.0e}" for nl in pivot.index]
    ax.set_yticklabels(y_labels, rotation=0)

    plt.tight_layout()

    # Save with sanitized filename
    safe_name = method.replace('/', '_').replace(' ', '_').replace('.', '_')
    output_file = per_method_dir / f"{safe_name}_heatmap.pdf"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

print(f"\n✓ Generated {len(methods)} per-method heatmaps")

# =============================================================================
# PART 2: Per-Method Line Plots (Noise vs Order, showing curves)
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: GENERATING PER-METHOD LINE PLOTS")
print("=" * 80)

for i, method in enumerate(methods, 1):
    print(f"[{i}/{len(methods)}] {method}...")

    method_data = summary[summary['method'] == method].copy()

    if len(method_data) == 0:
        continue

    # Create figure with line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot one line per derivative order
    colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))

    for order, color in zip(orders, colors):
        order_subset = method_data[method_data['deriv_order'] == order].sort_values('noise_level')

        if len(order_subset) > 0:
            ax.plot(
                order_subset['noise_level'],
                order_subset['mean_nrmse'],
                marker='o',
                linewidth=2,
                markersize=6,
                label=f'Order {order}',
                color=color,
                alpha=0.8
            )

            # Add error bars if available
            if 'std_nrmse' in order_subset.columns:
                ax.fill_between(
                    order_subset['noise_level'],
                    order_subset['mean_nrmse'] - order_subset['std_nrmse'],
                    order_subset['mean_nrmse'] + order_subset['std_nrmse'],
                    alpha=0.2,
                    color=color
                )

    ax.set_xscale('log')
    ax.set_xlabel('Noise Level (log scale)', fontweight='bold')
    ax.set_ylabel('nRMSE', fontweight='bold')
    ax.set_title(f'{method}\nNoise Sensitivity by Derivative Order', fontweight='bold')
    ax.legend(loc='best', ncol=2, frameon=True)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0, min(5.0, method_data['mean_nrmse'].max() * 1.1))  # Cap at reasonable range

    # Add horizontal reference lines
    ax.axhline(y=0.1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (0.1)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Acceptable (0.5)')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor (1.0)')

    plt.tight_layout()

    safe_name = method.replace('/', '_').replace(' ', '_').replace('.', '_')
    output_file = per_method_dir / f"{safe_name}_noise_sensitivity.pdf"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

print(f"\n✓ Generated {len(methods)} per-method line plots")

# =============================================================================
# PART 3: Per-Order Method Comparison Plots
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: GENERATING PER-ORDER METHOD COMPARISONS")
print("=" * 80)

for order in orders:
    print(f"Order {order}...")

    order_data = summary[summary['deriv_order'] == order].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top 15 methods by average nRMSE at this order
    method_avg = order_data.groupby('method')['mean_nrmse'].mean().sort_values()
    top_methods = method_avg.head(15).index.tolist()

    # Plot lines for each method
    for method in top_methods:
        method_subset = order_data[order_data['method'] == method].sort_values('noise_level')

        if len(method_subset) > 0:
            ax.plot(
                method_subset['noise_level'],
                method_subset['mean_nrmse'],
                marker='o',
                linewidth=1.5,
                markersize=5,
                label=method,
                alpha=0.7
            )

    ax.set_xscale('log')
    ax.set_xlabel('Noise Level (log scale)', fontweight='bold')
    ax.set_ylabel('nRMSE', fontweight='bold')
    ax.set_title(f'Derivative Order {order}: Method Comparison\n(Top 15 methods by average nRMSE)', fontweight='bold')
    ax.legend(loc='best', ncol=2, frameon=True, fontsize=7)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0, min(3.0, order_data['mean_nrmse'].quantile(0.95)))

    # Add reference bands
    ax.axhspan(0, 0.1, alpha=0.05, color='green', label='Excellent')
    ax.axhspan(0.1, 0.5, alpha=0.05, color='yellow', label='Good')
    ax.axhspan(0.5, 1.0, alpha=0.05, color='orange', label='Acceptable')

    plt.tight_layout()

    output_file = per_order_dir / f"order_{order}_method_comparison.pdf"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

print(f"\n✓ Generated {len(orders)} per-order comparison plots")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE VISUALIZATION COMPLETE")
print("=" * 80)
print(f"\nGenerated:")
print(f"  {len(methods)} per-method heatmaps → {per_method_dir}")
print(f"  {len(methods)} per-method line plots → {per_method_dir}")
print(f"  {len(orders)} per-order comparisons → {per_order_dir}")
print(f"\nTotal: {2*len(methods) + len(orders)} plots")
print("=" * 80)
