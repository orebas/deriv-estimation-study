#!/usr/bin/env python3
"""
Generate 3-panel heatmap for high-order derivatives (6-7).

Panels:
1. Overall (all noise levels)
2. Low-Noise regime (≤0.1%: 1e-8, 1e-6, 1e-4, 1e-3)
3. High-Noise regime (≥1%: 0.01, 0.02)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'text.usetex': False
})

print("=" * 80)
print("GENERATING HIGH-ORDER DERIVATIVE HEATMAPS (Orders 6-7)")
print("=" * 80)

# Load data
results_dir = Path(__file__).parent.parent / "build" / "results" / "comprehensive"
summary = pd.read_csv(results_dir / "comprehensive_summary.csv")

# Define noise regimes
LO_NOISE = [1e-8, 1e-6, 1e-4, 1e-3]
HI_NOISE = [0.01, 0.02]
ALL_NOISE = LO_NOISE + HI_NOISE

print(f"\nNoise regimes:")
print(f"  Low-Noise (≤0.1%): {LO_NOISE}")
print(f"  High-Noise (≥1%): {HI_NOISE}")

# Filter to orders 6-7
high_order_data = summary[summary['deriv_order'].isin([6, 7])].copy()

# Find methods that have BOTH orders 6 and 7
method_order_counts = high_order_data.groupby('method')['deriv_order'].nunique()
methods_with_both = method_order_counts[method_order_counts == 2].index.tolist()

print(f"\nMethods with both orders 6 and 7: {len(methods_with_both)}")

# Create output directory
output_dir = Path(__file__).parent.parent / "build" / "figures" / "publication"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Create 3-panel figure
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 8))

# Panel configurations
panels = [
    {
        'title': 'Overall\n(All Noise Levels)',
        'noise_levels': ALL_NOISE,
        'ax': axes[0]
    },
    {
        'title': 'Low-Noise Regime\n(≤0.1%)',
        'noise_levels': LO_NOISE,
        'ax': axes[1]
    },
    {
        'title': 'High-Noise Regime\n(≥1%)',
        'noise_levels': HI_NOISE,
        'ax': axes[2]
    }
]

for panel_idx, panel in enumerate(panels):
    print(f"\n{'='*60}")
    print(f"Panel {panel_idx + 1}: {panel['title'].replace(chr(10), ' ')}")
    print('='*60)

    # Filter to this noise regime
    panel_data = high_order_data[
        high_order_data['noise_level'].isin(panel['noise_levels']) &
        high_order_data['method'].isin(methods_with_both)
    ].copy()

    # Rank methods by average nRMSE in this regime
    avg_performance = panel_data.groupby('method')['mean_nrmse'].mean().sort_values()

    # Take top 15 methods (or fewer if not enough)
    top_n = min(15, len(avg_performance))
    top_methods = avg_performance.head(top_n).index.tolist()

    print(f"Top {top_n} methods:")
    for i, method in enumerate(top_methods[:5], 1):
        print(f"  {i}. {method}: {avg_performance[method]:.4f}")
    if top_n > 5:
        print("  ...")

    # Create pivot: method × (noise_level, deriv_order) → mean_nrmse
    # We want to show: method × order with average across noise levels
    pivot_data = panel_data[panel_data['method'].isin(top_methods)].pivot_table(
        index='method',
        columns='deriv_order',
        values='mean_nrmse',
        aggfunc='mean'
    )

    # Sort by average performance
    pivot_data['avg'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('avg')
    pivot_data = pivot_data.drop('avg', axis=1)

    # Plot heatmap
    ax = panel['ax']

    # Determine vmax based on data range
    if panel_idx == 2:  # High-noise regime
        vmax = 3.0
    else:
        vmax = 2.0

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Red = bad, Green = good
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'nRMSE'},
        linewidths=0.5,
        ax=ax
    )

    ax.set_xlabel('Derivative Order', fontweight='bold', fontsize=10)
    ax.set_ylabel('Method', fontweight='bold', fontsize=10)
    ax.set_title(panel['title'], fontweight='bold', fontsize=11)

    # Rotate y-labels for readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# Overall title
fig.suptitle('High-Order Derivatives (Orders 6-7): Performance Across Noise Regimes',
             fontweight='bold', fontsize=13, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_file = output_dir / "heatmap_orders_6to7_by_noise_regime.png"
plt.savefig(output_file, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n{'='*80}")
print(f"3-panel heatmap saved: {output_file}")
print(f"File size: {output_file.stat().st_size / 1024:.0f} KB")
print('='*80)

# Also save as PDF for paper
output_pdf = output_dir / "heatmap_orders_6to7_by_noise_regime.pdf"
fig, axes = plt.subplots(1, 3, figsize=(15, 8))

for panel_idx, panel in enumerate(panels):
    # Filter to this noise regime
    panel_data = high_order_data[
        high_order_data['noise_level'].isin(panel['noise_levels']) &
        high_order_data['method'].isin(methods_with_both)
    ].copy()

    # Rank methods by average nRMSE in this regime
    avg_performance = panel_data.groupby('method')['mean_nrmse'].mean().sort_values()
    top_n = min(15, len(avg_performance))
    top_methods = avg_performance.head(top_n).index.tolist()

    # Create pivot
    pivot_data = panel_data[panel_data['method'].isin(top_methods)].pivot_table(
        index='method',
        columns='deriv_order',
        values='mean_nrmse',
        aggfunc='mean'
    )

    # Sort by average performance
    pivot_data['avg'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('avg')
    pivot_data = pivot_data.drop('avg', axis=1)

    # Plot heatmap
    ax = axes[panel_idx]

    vmax = 3.0 if panel_idx == 2 else 2.0

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'nRMSE'},
        linewidths=0.5,
        ax=ax
    )

    ax.set_xlabel('Derivative Order', fontweight='bold', fontsize=10)
    ax.set_ylabel('Method', fontweight='bold', fontsize=10)
    ax.set_title(panel['title'], fontweight='bold', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

fig.suptitle('High-Order Derivatives (Orders 6-7): Performance Across Noise Regimes',
             fontweight='bold', fontsize=13, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_pdf, bbox_inches='tight')
plt.close()

print(f"PDF version saved: {output_pdf}")
print('='*80)
