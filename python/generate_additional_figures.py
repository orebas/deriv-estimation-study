#!/usr/bin/env python3
"""
Generate additional publication figures based on expert feedback:
1. Pareto frontier plot (accuracy vs computational cost)
2. Small multiples grid (8 orders in 4x2 layout)
3. Qualitative comparison (actual derivatives for order 4, noise=2%)
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
print("GENERATING ADDITIONAL PUBLICATION FIGURES")
print("=" * 80)

# Load data
results_dir = Path(__file__).parent.parent / "results" / "comprehensive"
summary = pd.read_csv(results_dir / "comprehensive_summary.csv")
raw_results = pd.read_csv(results_dir / "comprehensive_results.csv")

print(f"\nLoaded {len(summary)} summary rows, {len(raw_results)} raw result rows")

# Output directory
output_dir = Path(__file__).parent.parent / "report" / "paper_figures" / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Figure 1: Pareto Frontier Plot (Accuracy vs Computational Cost)
# ============================================================================
print("\n" + "=" * 80)
print("FIGURE 1: PARETO FRONTIER (nRMSE vs Time)")
print("=" * 80)

# Compute overall average nRMSE and timing per method
method_summary = summary.groupby(['method', 'category']).agg({
    'mean_nrmse': 'mean',
    'mean_timing': 'mean'
}).reset_index()

print(f"Methods: {len(method_summary)}")

# Create Pareto plot
fig, ax = plt.subplots(figsize=(10, 7))

# Define category colors
category_colors = {
    'Gaussian Process': '#1f77b4',
    'Rational Approximation': '#ff7f0e',
    'Spectral': '#2ca02c',
    'Spline': '#d62728',
    'Finite Difference': '#9467bd',
    'Total Variation': '#8c564b',
    'RBF': '#e377c2',
    'Other': '#7f7f7f'
}

# Plot each method as a point
for category in method_summary['category'].unique():
    cat_data = method_summary[method_summary['category'] == category]
    ax.scatter(
        cat_data['mean_timing'],
        cat_data['mean_nrmse'],
        label=category,
        alpha=0.7,
        s=100,
        color=category_colors.get(category, '#7f7f7f'),
        edgecolors='black',
        linewidth=0.5
    )

# Annotate top methods
top_methods = ['GP-Julia-AD', 'AAA-HighPrec', 'Fourier-Interp', 'Central-FD-5pt']
for method in top_methods:
    if method in method_summary['method'].values:
        row = method_summary[method_summary['method'] == method].iloc[0]
        ax.annotate(
            method,
            xy=(row['mean_timing'], row['mean_nrmse']),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=0.5)
        )

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mean Computation Time (seconds, log scale)')
ax.set_ylabel('Mean nRMSE (averaged over all orders and noise levels, log scale)')
ax.set_title('Accuracy vs Computational Cost: Pareto Frontier')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', frameon=True, ncol=2)

# Add reference lines
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='nRMSE=1.0 (acceptable limit)')

plt.tight_layout()
pareto_file = output_dir / "pareto_frontier.pdf"
plt.savefig(pareto_file, bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved: {pareto_file.name}")

# ============================================================================
# Figure 2: Small Multiples Grid (8 orders in 4x2 layout)
# ============================================================================
print("\n" + "=" * 80)
print("FIGURE 2: SMALL MULTIPLES GRID (nRMSE vs Noise by Order)")
print("=" * 80)

noise_levels = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2]
orders = list(range(8))

# Get top 7 methods overall
overall_avg = summary.groupby('method')['mean_nrmse'].mean().sort_values()
top_7_methods = overall_avg.head(7).index.tolist()

print(f"Top 7 methods: {top_7_methods}")

# Create 4x2 grid
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.flatten()

for idx, order in enumerate(orders):
    ax = axes[idx]

    # Filter data for this order
    order_data = summary[summary['deriv_order'] == order]

    # Plot top 7 methods
    for method in top_7_methods:
        method_data = order_data[order_data['method'] == method].sort_values('noise_level')

        if len(method_data) > 0:
            # Plot mean nRMSE
            ax.plot(
                method_data['noise_level'],
                method_data['mean_nrmse'],
                marker='o',
                label=method,
                linewidth=1.5,
                markersize=4,
                alpha=0.8
            )

            # Add error bars (std_nrmse as shaded region)
            if 'std_nrmse' in method_data.columns:
                ax.fill_between(
                    method_data['noise_level'],
                    method_data['mean_nrmse'] - method_data['std_nrmse'],
                    method_data['mean_nrmse'] + method_data['std_nrmse'],
                    alpha=0.15
                )

    ax.set_xscale('log')
    ax.set_xlabel('Noise Level' if idx >= 6 else '')
    ax.set_ylabel('nRMSE' if idx % 2 == 0 else '')
    ax.set_title(f'Order {order}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)

    # Add interpretation bands
    ax.axhspan(0, 0.1, alpha=0.05, color='green')
    ax.axhspan(0.1, 0.3, alpha=0.05, color='yellow')
    ax.axhspan(0.3, 1.0, alpha=0.05, color='orange')

    # Legend only on first subplot
    if idx == 0:
        ax.legend(loc='upper left', fontsize=7, frameon=True, ncol=1)

# Overall title
fig.suptitle('Performance Across Derivative Orders: Top 7 Methods', fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
small_multiples_file = output_dir / "small_multiples_grid.pdf"
plt.savefig(small_multiples_file, bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved: {small_multiples_file.name}")

# ============================================================================
# Figure 3: Qualitative Comparison (Order 4, Noise=2%)
# ============================================================================
print("\n" + "=" * 80)
print("FIGURE 3: QUALITATIVE COMPARISON (Actual Derivatives)")
print("=" * 80)

# We need to load raw data for a specific configuration
# Let's try to find order 4, noise=2e-2, trial 1
target_order = 4
target_noise = 2e-2

# Check if we have input/output data for this configuration
trial_id = f"noise{int(target_noise*1e8)}e-8_trial1"
input_json = Path(__file__).parent.parent / "data" / "input" / f"{trial_id}.json"
output_json = Path(__file__).parent.parent / "data" / "output" / f"{trial_id}_results.json"

if input_json.exists() and output_json.exists():
    import json

    # Load ground truth
    with open(input_json, 'r') as f:
        input_data = json.load(f)

    times = np.array(input_data['times'])
    ground_truth = np.array(input_data['ground_truth_derivatives'][str(target_order)])

    # Load predictions
    with open(output_json, 'r') as f:
        output_data = json.load(f)

    # Also need Julia results - try to reconstruct from raw_results
    trial_results = raw_results[
        (raw_results['deriv_order'] == target_order) &
        (raw_results['noise_level'] == target_noise) &
        (raw_results['trial'] == 1)
    ]

    print(f"Found {len(trial_results)} method results for order={target_order}, noise={target_noise}")

    # Create 3-panel plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Panel A: GP-Julia-AD with confidence interval
    # Note: We don't have CI data saved, so we'll just show the estimate
    ax1.plot(times, ground_truth, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)

    # We need to get GP-AD predictions from somewhere
    # For now, create placeholder
    ax1.set_xlabel('Time')
    ax1.set_ylabel(f'd⁴y/dt⁴')
    ax1.set_title('Panel A: GP-Julia-AD (Best Method)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel B: AAA-HighPrec vs Fourier-Interp
    ax2.plot(times, ground_truth, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(f'd⁴y/dt⁴')
    ax2.set_title('Panel B: AAA-HighPrec vs Fourier-Interp', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel C: Central-FD-7pt (catastrophic failure)
    ax3.plot(times, ground_truth, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax3.set_xlabel('Time')
    ax3.set_ylabel(f'd⁴y/dt⁴')
    ax3.set_title('Panel C: Central-FD-7pt (Catastrophic Failure)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    qualitative_file = output_dir / "qualitative_comparison.pdf"
    plt.savefig(qualitative_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: {qualitative_file.name}")
    print("NOTE: Qualitative plot needs actual prediction data - currently showing ground truth only")
else:
    print(f"WARNING: Could not find data for {trial_id} - skipping qualitative comparison")
    print(f"  Input: {input_json.exists()}")
    print(f"  Output: {output_json.exists()}")

print("\n" + "=" * 80)
print("ADDITIONAL FIGURE GENERATION COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print("=" * 80)
