#!/usr/bin/env python3
"""
Generate all publication-quality figures for the derivative estimation paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Paths
DATA_DIR = Path('../results/comprehensive')
FIG_DIR = Path('paper_figures/publication')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
summary = pd.read_csv(DATA_DIR / 'comprehensive_summary.csv')
results = pd.read_csv(DATA_DIR / 'comprehensive_results.csv')

# Exclude failed methods per Section 4.7
EXCLUDED_METHODS = ['GP-Julia-SE', 'TVRegDiff_Python', 'SavitzkyGolay_Python']
summary = summary[~summary['method'].isin(EXCLUDED_METHODS)].copy()
results = results[~results['method'].isin(EXCLUDED_METHODS)].copy()

print(f"Loaded {len(summary)} summary rows, {len(results)} detailed rows")
print(f"Methods: {summary['method'].nunique()}")
print(f"After exclusions: {24} methods (excluded: {EXCLUDED_METHODS})")

#==============================================================================
# FIGURE 1: HEATMAP - Method × Derivative Order
#==============================================================================
def generate_figure1_heatmap():
    """Heatmap showing nRMSE for each method across derivative orders"""
    print("\nGenerating Figure 1: Heatmap...")

    # Compute mean nRMSE across all noise levels for each (method, order)
    heatmap_data = summary.groupby(['method', 'deriv_order'])['mean_nrmse'].mean().unstack()

    # Sort methods by overall mean (across all orders)
    method_order = heatmap_data.mean(axis=1).sort_values().index
    heatmap_data = heatmap_data.loc[method_order]

    # Take top 15 methods for readability
    heatmap_data = heatmap_data.head(15)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Log-scale colormap to handle wide range
    from matplotlib.colors import LogNorm
    vmin = max(heatmap_data.min().min(), 1e-4)  # Avoid log(0)
    vmax = min(heatmap_data.max().max(), 10)    # Cap at 10 for readability

    sns.heatmap(heatmap_data, annot=False, fmt='.2f', cmap='RdYlGn_r',
                norm=LogNorm(vmin=vmin, vmax=vmax), cbar_kws={'label': 'Mean nRMSE'},
                ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Derivative Order')
    ax.set_ylabel('Method')
    ax.set_title('Figure 1: Method Performance Across Derivative Orders\n(Top 15 methods, mean nRMSE across all noise levels)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure1_heatmap.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figure1_heatmap.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'figure1_heatmap.png'}")
    plt.close()

#==============================================================================
# FIGURE 2: SMALL MULTIPLES - Per-order performance with error bars
#==============================================================================
def generate_figure2_small_multiples():
    """8-panel grid showing nRMSE vs noise for each derivative order"""
    print("\nGenerating Figure 2: Small Multiples...")

    # Select top 7 methods overall
    top_methods = (summary.groupby('method')['mean_nrmse'].mean()
                   .sort_values().head(7).index.tolist())

    # Create 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()

    for order in range(8):
        ax = axes[order]

        # Filter data for this order
        order_data = summary[summary['deriv_order'] == order]

        for method in top_methods:
            method_data = order_data[order_data['method'] == method].sort_values('noise_level')

            if len(method_data) > 0:
                # Plot mean with error bars (std)
                ax.errorbar(method_data['noise_level'], method_data['mean_nrmse'],
                           yerr=method_data['std_nrmse'], label=method,
                           marker='o', markersize=4, capsize=3, alpha=0.8)

        ax.set_xscale('log')
        ax.set_ylim(0, 1.0)  # Cap at 1.0 for readability
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('nRMSE')
        ax.set_title(f'Order {order}')
        ax.grid(True, alpha=0.3)

        if order == 0:  # Add legend to first panel only
            ax.legend(loc='upper left', fontsize=7)

    fig.suptitle('Figure 2: Performance vs Noise Level for Each Derivative Order\n(Top 7 methods, mean ± std from 3 trials)', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIG_DIR / 'figure2_small_multiples.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figure2_small_multiples.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'figure2_small_multiples.png'}")
    plt.close()

#==============================================================================
# FIGURE 4: PARETO FRONTIER - Accuracy vs Speed
#==============================================================================
def generate_figure4_pareto():
    """Pareto frontier plot: nRMSE vs computation time"""
    print("\nGenerating Figure 4: Pareto Frontier...")

    # Compute overall metrics per method
    method_stats = summary.groupby('method').agg({
        'mean_nrmse': 'mean',
        'mean_timing': 'mean',
        'category': 'first'
    }).reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by category
    categories = method_stats['category'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    category_colors = dict(zip(categories, colors))

    for category in categories:
        cat_data = method_stats[method_stats['category'] == category]
        ax.scatter(cat_data['mean_timing'], cat_data['mean_nrmse'],
                  label=category, s=80, alpha=0.7,
                  color=category_colors[category], edgecolors='black', linewidth=0.5)

        # Annotate method names
        for _, row in cat_data.iterrows():
            ax.annotate(row['method'],
                       (row['mean_timing'], row['mean_nrmse']),
                       fontsize=7, alpha=0.7, xytext=(5, 5),
                       textcoords='offset points')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mean Computation Time (seconds)')
    ax.set_ylabel('Mean nRMSE (across all orders and noise levels)')
    ax.set_title('Figure 4: Pareto Frontier - Accuracy vs Computational Cost')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure4_pareto.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figure4_pareto.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'figure4_pareto.png'}")
    plt.close()

#==============================================================================
# SUPPORTING FIGURE: Ground Truth Visualization
#==============================================================================
def generate_ground_truth_figure():
    """Visualize ground truth derivatives from Lotka-Volterra"""
    print("\nGenerating Supporting Figure: Ground Truth...")

    # Load ground truth data from one of the input files
    input_files = list(Path('../data/input').glob('*.json'))
    if len(input_files) == 0:
        print("  Warning: No input files found, skipping ground truth figure")
        return

    with open(input_files[0], 'r') as f:
        data = json.load(f)

    times = np.array(data['times'])

    # Create 2x4 grid for orders 0-7
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    for order in range(8):
        y_true = np.array(data['ground_truth_derivatives'][str(order)])

        axes[order].plot(times, y_true, 'k-', linewidth=1.5)
        axes[order].set_xlabel('Time')
        axes[order].set_ylabel(f'$d^{order}x/dt^{order}$')
        axes[order].set_title(f'Order {order}')
        axes[order].grid(True, alpha=0.3)

    fig.suptitle('Ground Truth: Lotka-Volterra Predator Population Derivatives', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIG_DIR / 'ground_truth.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'ground_truth.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'ground_truth.png'}")
    plt.close()

#==============================================================================
# MAIN
#==============================================================================
if __name__ == '__main__':
    print("="*80)
    print("GENERATING PUBLICATION FIGURES")
    print("="*80)

    generate_figure1_heatmap()
    generate_figure2_small_multiples()
    generate_figure4_pareto()
    generate_ground_truth_figure()

    print("\n" + "="*80)
    print("✓ ALL FIGURES GENERATED")
    print(f"Output directory: {FIG_DIR.absolute()}")
    print("="*80)
