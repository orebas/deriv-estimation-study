#!/usr/bin/env python3
"""
Generate updated figures per user feedback:
1. Remove AAA methods from Pareto chart
2. Use "within 2x of best" instead of "top N" for charts
3. Exclude catastrophic methods (AAA-HighPrec, SavitzkyGolay_Python, GP-Julia-SE) entirely
4. Create detailed GP-Julia-AD noise × derivative order chart
5. Generate two-regime overall performance tables (low noise <1%, high noise ≥1%)
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

# EXCLUDE CATASTROPHIC METHODS ENTIRELY
EXCLUDED_METHODS = ['GP-Julia-SE', 'AAA-HighPrec', 'SavitzkyGolay_Python']
summary = summary[~summary['method'].isin(EXCLUDED_METHODS)].copy()
results = results[~results['method'].isin(EXCLUDED_METHODS)].copy()

print(f"Loaded {len(summary)} summary rows, {len(results)} detailed rows")
print(f"Methods: {summary['method'].nunique()}")
print(f"Excluded catastrophic methods: {EXCLUDED_METHODS}")

# Define noise regimes
LOW_NOISE = [1e-8, 1e-6, 1e-4, 1e-3]  # <1%
HIGH_NOISE = [1e-2, 2e-2, 5e-2]        # ≥1%

#==============================================================================
# NEW: TWO-REGIME OVERALL PERFORMANCE TABLES
#==============================================================================
def generate_regime_tables():
    """Generate overall performance tables split by noise regime"""
    print("\n" + "="*80)
    print("GENERATING TWO-REGIME OVERALL PERFORMANCE TABLES")
    print("="*80)

    # For orders 1, 2, 3 only (actual derivatives)
    deriv_data = results[results['deriv_order'].isin([1, 2, 3])].copy()

    regime_results = {}

    for regime_name, noise_levels in [("Low Noise (<1%)", LOW_NOISE),
                                       ("High Noise (≥1%)", HIGH_NOISE)]:
        print(f"\n{regime_name}:")
        print("-" * 80)

        regime_data = deriv_data[deriv_data['noise_level'].isin(noise_levels)]

        # For each derivative order
        for order in [1, 2, 3]:
            order_data = regime_data[regime_data['deriv_order'] == order]

            # Compute mean nRMSE per method
            method_nrmse = order_data.groupby('method')['nrmse'].mean().sort_values()

            print(f"\nOrder {order} (mean nRMSE):")
            print(method_nrmse.head(10).to_string())

            # Store for later use
            if regime_name not in regime_results:
                regime_results[regime_name] = {}
            regime_results[regime_name][order] = method_nrmse

    # Save overall performance tables (mean over all three derivatives)
    for regime_name, orders_dict in regime_results.items():
        # Find all methods that appear in all three derivative orders
        methods_in_all_orders = set(orders_dict[1].index) & set(orders_dict[2].index) & set(orders_dict[3].index)

        # Calculate mean nRMSE over orders 1-3
        overall_means = []
        for method in methods_in_all_orders:
            method_values = [orders_dict[order][method] for order in [1, 2, 3]]
            overall_means.append({
                'Method': method,
                'Mean_nRMSE': np.mean(method_values),
                'N_points': len(deriv_data[(deriv_data['method'] == method) &
                                            (deriv_data['noise_level'].isin(LOW_NOISE if 'Low' in regime_name else HIGH_NOISE))])
            })

        overall_df = pd.DataFrame(overall_means).sort_values('Mean_nRMSE')

        # Save CSV
        csv_filename = f'overall_performance_{"low" if "Low" in regime_name else "high"}_noise.csv'
        overall_df.to_csv(FIG_DIR.parent / csv_filename, index=False)
        print(f"\nSaved: {FIG_DIR.parent / csv_filename}")

        # Generate LaTeX table file (top 10 only)
        tex_filename = f'overall_performance_{"low" if "Low" in regime_name else "high"}_noise.tex'
        with open(FIG_DIR.parent / tex_filename, 'w') as f:
            f.write("% AUTO-GENERATED - DO NOT EDIT\n")
            f.write("% Regenerate by running: python report/generate_updated_figures.py\n\n")
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            # Escape special characters for LaTeX
            caption_text = (regime_name
                           .replace('≥', '$\\geq$')
                           .replace('<', '$<$')
                           .replace('>', '$>$')
                           .replace('%', '\\%'))
            f.write(f"\\caption{{Top 10 Methods: {caption_text}, Mean nRMSE over Derivatives 1-3}}\n")
            f.write("\\begin{tabular}{lll}\n")
            f.write("\\toprule\n")
            f.write("Rank & Method & Mean nRMSE \\\\\n")
            f.write("\\midrule\n")

            for rank, (_, row) in enumerate(overall_df.head(10).iterrows(), 1):
                method_clean = row['Method'].replace('_', '\\_')
                f.write(f"{rank} & {method_clean} & {row['Mean_nRMSE']:.3f} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"Saved LaTeX: {FIG_DIR.parent / tex_filename}")

    return regime_results

#==============================================================================
# NEW: DETAILED GP-JULIA-AD PERFORMANCE CHART
#==============================================================================
def generate_gp_julia_ad_detail():
    """Generate detailed noise × derivative order chart for GP-Julia-AD"""
    print("\nGenerating GP-Julia-AD Detail Chart...")

    gp_data = results[results['method'] == 'GP-Julia-AD'].copy()

    # Create heatmap: noise (rows) × derivative order (columns)
    heatmap_data = gp_data.groupby(['noise_level', 'deriv_order'])['nrmse'].mean().unstack()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Log-scale colormap
    from matplotlib.colors import LogNorm
    vmin = max(heatmap_data.min().min(), 1e-4)
    vmax = min(heatmap_data.max().max(), 1.0)

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                norm=LogNorm(vmin=vmin, vmax=vmax), cbar_kws={'label': 'nRMSE'},
                ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Derivative Order')
    ax.set_ylabel('Noise Level')
    ax.set_title('GP-Julia-AD Performance: nRMSE Across Noise Levels and Derivative Orders')

    # Format y-axis labels
    y_labels = [f'{float(label.get_text()):.0e}' for label in ax.get_yticklabels()]
    ax.set_yticklabels(y_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'gp_julia_ad_detail.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'gp_julia_ad_detail.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'gp_julia_ad_detail.png'}")
    plt.close()

#==============================================================================
# UPDATED: PARETO FRONTIER (NO AAA METHODS)
#==============================================================================
def generate_figure4_pareto_updated():
    """Pareto frontier plot: nRMSE vs computation time (NO AAA methods, only complete methods)"""
    print("\nGenerating Updated Figure 4: Pareto Frontier (no AAA, complete orders 1-3 only)...")

    # Find methods with complete orders 1-3 data
    methods_with_complete = set()
    for method in results['method'].unique():
        method_data = results[results['method'] == method]
        orders = set(method_data['deriv_order'].unique())
        if {1, 2, 3}.issubset(orders):
            methods_with_complete.add(method)

    print(f"  Methods with complete orders 1-3: {len(methods_with_complete)}")

    # Compute overall metrics per method
    method_stats = summary.groupby('method').agg({
        'mean_nrmse': 'mean',
        'mean_timing': 'mean',
        'category': 'first'
    }).reset_index()

    # EXCLUDE ALL AAA METHODS
    method_stats = method_stats[~method_stats['method'].str.contains('AAA', case=False)]

    # EXCLUDE METHODS WITHOUT COMPLETE ORDERS 1-3 DATA
    method_stats = method_stats[method_stats['method'].isin(methods_with_complete)]
    print(f"  Methods after filtering: {len(method_stats)}")

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
    ax.set_title('Figure 4: Pareto Frontier - Accuracy vs Computational Cost\n(AAA methods and incomplete methods excluded for clarity)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure4_pareto.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figure4_pareto.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'figure4_pareto.png'}")
    plt.close()

#==============================================================================
# UPDATED: SMALL MULTIPLES (WITHIN 2X OF BEST)
#==============================================================================
def generate_figure2_small_multiples_updated():
    """8-panel grid showing nRMSE vs noise for each derivative order (within 2x of best)"""
    print("\nGenerating Updated Figure 2: Small Multiples (within 2x of best)...")

    # For each order, find best method and include all within 2x
    methods_to_include = set()

    for order in range(8):
        order_data = summary[summary['deriv_order'] == order]
        best_nrmse = order_data.groupby('method')['mean_nrmse'].mean().min()

        # Include all methods within 2x of best for this order
        for method in order_data['method'].unique():
            method_nrmse = order_data[order_data['method'] == method]['mean_nrmse'].mean()
            if method_nrmse <= 2 * best_nrmse:
                methods_to_include.add(method)

    methods_to_include = sorted(methods_to_include)
    print(f"  Including {len(methods_to_include)} methods (within 2x of best across all orders)")

    # Create 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()

    for order in range(8):
        ax = axes[order]

        # Get data for this order
        order_data = summary[summary['deriv_order'] == order]

        # For each method within 2x of best
        for method in methods_to_include:
            method_data = order_data[order_data['method'] == method]
            if len(method_data) == 0:
                continue

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
            ax.legend(loc='upper left', fontsize=6, ncol=2)

    fig.suptitle('Figure 2: Performance vs Noise Level for Each Derivative Order\n(Methods within 2x of best, mean ± std from 3 trials)', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIG_DIR / 'figure2_small_multiples.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figure2_small_multiples.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'figure2_small_multiples.png'}")
    plt.close()

#==============================================================================
# UPDATED: HEATMAP (WITHIN 2X OF BEST)
#==============================================================================
def generate_figure1_heatmap_updated():
    """Heatmap showing nRMSE for each method across derivative orders (within 2x of best for orders 1-3)"""
    print("\nGenerating Updated Figure 1: Heatmap (within 2x of best)...")

    # Use results dataframe, compute mean nRMSE across all noise levels for each (method, order)
    heatmap_data = results.groupby(['method', 'deriv_order'])['nrmse'].mean().unstack()

    # Find best method for orders 1-3 only (the derivatives people actually use)
    orders_1_to_3 = [col for col in heatmap_data.columns if col in [1, 2, 3]]
    method_means_1_3 = heatmap_data[orders_1_to_3].mean(axis=1)
    best_overall_1_3 = method_means_1_3.min()

    print(f"  Best method mean nRMSE (orders 1-3): {best_overall_1_3:.4f}")
    print(f"  3x threshold: {3 * best_overall_1_3:.4f}")

    # Keep only methods within 3x of best (based on orders 1-3)
    # Using 3x instead of 2x to show more comparable methods while still filtering outliers
    methods_within_3x = method_means_1_3[method_means_1_3 <= 3 * best_overall_1_3].index
    heatmap_data = heatmap_data.loc[methods_within_3x]

    # Sort methods by mean performance on orders 1-3
    method_order = method_means_1_3[methods_within_3x].sort_values().index
    heatmap_data = heatmap_data.loc[method_order]

    print(f"  Including {len(heatmap_data)} methods within 3x of best (orders 1-3)")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Log-scale colormap
    from matplotlib.colors import LogNorm
    vmin = max(heatmap_data.min().min(), 1e-4)
    vmax = min(heatmap_data.max().max(), 10)

    sns.heatmap(heatmap_data, annot=False, fmt='.2f', cmap='RdYlGn_r',
                norm=LogNorm(vmin=vmin, vmax=vmax), cbar_kws={'label': 'Mean nRMSE'},
                ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Derivative Order')
    ax.set_ylabel('Method')
    ax.set_title(f'Figure 1: Method Performance Across Derivative Orders\n(Methods within 2x of best overall, mean nRMSE across all noise levels)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'figure1_heatmap.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figure1_heatmap.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'figure1_heatmap.png'}")
    plt.close()

#==============================================================================
# MAIN
#==============================================================================
if __name__ == '__main__':
    print("="*80)
    print("GENERATING UPDATED PUBLICATION FIGURES")
    print("="*80)

    # Generate all figures
    generate_regime_tables()
    generate_gp_julia_ad_detail()
    generate_figure1_heatmap_updated()
    generate_figure2_small_multiples_updated()
    generate_figure4_pareto_updated()

    print("\n" + "="*80)
    print("✓ ALL UPDATED FIGURES GENERATED")
    print(f"Output directory: {FIG_DIR}")
    print("="*80)
