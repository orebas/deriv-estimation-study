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
results_dir = Path(__file__).parent.parent / "build" / "results" / "comprehensive"
summary = pd.read_csv(results_dir / "comprehensive_summary.csv")

print(f"\nLoaded {len(summary)} rows from summary data")

# Consolidate functionally equivalent GP-Python methods
# These three methods have identical performance (confirmed by analysis)
gp_python_variants = ['GP-RBF-Iso-Python', 'GP-RBF-MeanSub-Python']
summary['method'] = summary['method'].replace(gp_python_variants, 'GP-RBF-Python')

# Deduplicate after consolidation
summary = summary.drop_duplicates(subset=['ode_system', 'method', 'deriv_order', 'noise_level'], keep='first')
print(f"Consolidated GP-Python variants into single method and deduplicated")

# Noise levels to analyze
noise_levels = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2]
orders = list(range(8))  # 0-7

# Output directory
output_dir = Path(__file__).parent.parent / "build" / "tables" / "publication"
output_dir.mkdir(parents=True, exist_ok=True)

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

    # Limit to top 50 methods for orders 0 and 1 (they have too many to fit on page)
    if order in [0, 1] and len(pivot) > 50:
        original_count = len(pivot)
        pivot = pivot.head(50)
        print(f"  Limiting to top 50 methods (out of {original_count} total)")

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

        # Escape underscores in the index (method names)
        formatted.index = formatted.index.str.replace('_', '\\_', regex=False)

        # Remove column and index names
        formatted.columns.name = None
        formatted.index.name = None

        latex_str = formatted.to_latex(
            caption=None,  # Remove caption from fragment
            label=None,    # Remove label from fragment
            escape=False
        )

        # Add "Method" as the first column header (replace leading " &" with "Method &")
        latex_str = latex_str.replace('\\toprule\n &', '\\toprule\nMethod &')

        f.write(latex_str)

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

plots_dir = Path(__file__).parent.parent / "build" / "figures" / "publication"
plots_dir.mkdir(parents=True, exist_ok=True)

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
    plot_file = plots_dir / f"order_{order}_nrmse_filtered.png"
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Saved: {plot_file.name}")

# ============================================================================
# Generate summary heatmap across all orders
# ============================================================================

print("\n" + "="*80)
print("GENERATING SUMMARY HEATMAP")
print("="*80)

# Filter to methods that have ALL derivative orders (0-7)
# Count how many derivative orders each method has
method_order_counts = summary.groupby('method')['deriv_order'].nunique()
full_order_methods = method_order_counts[method_order_counts == 8].index.tolist()

print(f"Methods with full derivative order coverage (0-7): {len(full_order_methods)}")

# Get top 15 methods from those with full coverage
full_coverage_summary = summary[summary['method'].isin(full_order_methods)]
overall_avg = full_coverage_summary.groupby('method')['mean_nrmse'].mean().sort_values()
top_methods = overall_avg.head(15).index.tolist()

print(f"Top 15 methods with full coverage: {top_methods[:5]}... (showing first 5)")

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

heatmap_file = plots_dir / "top_methods_heatmap.png"
plt.savefig(heatmap_file, bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved: {heatmap_file.name}")

# ============================================================================
# Generate key LaTeX table includes for paper
# ============================================================================

print("\n" + "="*80)
print("GENERATING KEY LATEX TABLE INCLUDES")
print("="*80)

# Also load raw results for some tables
raw_results = pd.read_csv(results_dir / "comprehensive_results.csv")

# Table 1: Full Coverage Ranking
print("\nGenerating tab:full_coverage_ranking...")
coverage = summary.groupby('method').size()
full_coverage_methods = coverage[coverage == 56].index.tolist()  # 8 orders × 7 noise levels = 56

# Get method categories (from summary data)
method_category = summary.groupby('method')['category'].first()

# Compute overall ranking for full-coverage methods
full_coverage_data = summary[summary['method'].isin(full_coverage_methods)].copy()
overall_ranking = full_coverage_data.groupby('method').agg({
    'mean_nrmse': 'mean'
}).reset_index()
overall_ranking['category'] = overall_ranking['method'].map(method_category)
overall_ranking['coverage'] = '56/56'
overall_ranking = overall_ranking.sort_values('mean_nrmse')
overall_ranking['rank'] = range(1, len(overall_ranking) + 1)

# Format nRMSE for display
def format_nrmse_sci(x):
    if np.isnan(x) or np.isinf(x):
        return "---"  # Method failed for this order
    if x < 1:
        return f"{x:.3f}"
    elif x < 1000:
        return f"{x:.1f}"
    else:
        exp = int(np.floor(np.log10(x)))
        mantissa = x / 10**exp
        return f"{mantissa:.1f}$\\times$10$^{{{exp}}}$"

overall_ranking['mean_nrmse_fmt'] = overall_ranking['mean_nrmse'].apply(format_nrmse_sci)

# Generate LaTeX table
latex_ranking = output_dir / "tab_full_coverage_ranking.tex"
with open(latex_ranking, 'w') as f:
    f.write("% AUTO-GENERATED: Full-Coverage Methods Overall Performance\n")
    f.write("% Data source: build/results/comprehensive/comprehensive_summary.csv\n")
    f.write("\\begin{tabular}{clccc}\n")
    f.write("\\toprule\n")
    f.write("Rank & Method & Category & Mean nRMSE & Coverage \\\\\n")
    f.write("\\midrule\n")
    for _, row in overall_ranking.iterrows():
        f.write(f"{row['rank']} & {row['method']} & {row['category']} & {row['mean_nrmse_fmt']} & {row['coverage']} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print(f"  Saved: {latex_ranking.name}")

# Table 2: Performance By Order
print("\nGenerating tab:performance_by_order...")
# Select representative methods (dynamically check availability)
candidate_methods = ['GP-TaylorAD-Julia', 'Spline-Dierckx-5', 'SavitzkyGolay-Fixed', 'Fourier-GCV', 'GP-RBF-Python']
available_methods = summary['method'].unique()
representative_methods = [m for m in candidate_methods if m in available_methods]

perf_by_order = []
for method in representative_methods:
    method_data = summary[summary['method'] == method].copy()
    order_means = method_data.groupby('deriv_order')['mean_nrmse'].mean()
    perf_by_order.append({
        'method': method,
        **{f'order_{i}': order_means.get(i, np.nan) for i in range(8)}
    })

perf_df = pd.DataFrame(perf_by_order)

latex_perf = output_dir / "tab_performance_by_order.tex"
with open(latex_perf, 'w') as f:
    f.write("% AUTO-GENERATED: Performance Degradation Across Derivative Orders\n")
    f.write("% Data source: build/results/comprehensive/comprehensive_summary.csv\n")
    f.write("\\begin{tabular}{l" + "c" * 8 + "}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Method} & \\textbf{Order 0} & \\textbf{Order 1} & \\textbf{Order 2} & \\textbf{Order 3} & \\textbf{Order 4} & \\textbf{Order 5} & \\textbf{Order 6} & \\textbf{Order 7} \\\\\n")
    f.write("\\midrule\n")
    for _, row in perf_df.iterrows():
        f.write(f"{row['method']}")
        for i in range(8):
            val = row[f'order_{i}']
            f.write(f" & {format_nrmse_sci(val)}")
        f.write(" \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print(f"  Saved: {latex_perf.name}")

# Table 3: Timing Comparison
print("\nGenerating tab:timing_comparison...")
# Need timing data from raw_results
timing_data = raw_results.groupby('method').agg({
    'timing': 'mean',
    'nrmse': 'mean'
}).reset_index()
timing_data = timing_data.sort_values('timing')

# Select representative methods for comparison (dynamically check availability)
# Includes 4 Pareto-optimal methods + representative methods across speed ranges
candidate_timing_methods = ['SavitzkyGolay-Fixed', 'ButterworthSpline_Python', 'Spline-Dierckx-5',
                            'Fourier-Continuation-Python', 'Fourier-GCV', 'Fourier-Adaptive-Julia',
                            'GP-RBF-Python', 'GP-TaylorAD-Julia']
available_timing_methods = timing_data['method'].unique()
timing_methods = [m for m in candidate_timing_methods if m in available_timing_methods]
timing_subset = timing_data[timing_data['method'].isin(timing_methods)].copy()

# Calculate speedup vs GP-Julia-AD
gp_time = timing_subset[timing_subset['method'] == 'GP-TaylorAD-Julia']['timing'].values[0]
timing_subset['speedup'] = gp_time / timing_subset['timing']

latex_timing = output_dir / "tab_timing_comparison.tex"
with open(latex_timing, 'w') as f:
    f.write("% AUTO-GENERATED: Computational Cost vs Accuracy Trade-Off\n")
    f.write("% Data source: build/results/comprehensive/comprehensive_results.csv\n")
    f.write("\\begin{tabular}{lrrc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Method} & \\textbf{Mean Time (s)} & \\textbf{Mean nRMSE} & \\textbf{Speedup vs GP} \\\\\n")
    f.write("\\midrule\n")
    for _, row in timing_subset.iterrows():
        speedup_str = f"{row['speedup']:.1f}$\\times$" if row['speedup'] != 1.0 else "1.0$\\times$ (baseline)"
        method_escaped = str(row['method']).replace('_', '\\_')
        f.write(f"{method_escaped} & {row['timing']:.4f} & {format_nrmse_sci(row['nrmse'])} & {speedup_str} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print(f"  Saved: {latex_timing.name}")

# Table 4: Noise Sensitivity at Order 4
print("\nGenerating tab:noise_sensitivity_order4...")
order4_data = summary[summary['deriv_order'] == 4].copy()
order4_pivot = order4_data.pivot_table(
    index='method',
    columns='noise_level',
    values='mean_nrmse'
)

# Select representative methods (dynamically check availability)
candidate_noise_methods = ['GP-TaylorAD-Julia', 'Spline-Dierckx-5', 'SavitzkyGolay-Fixed', 'Fourier-GCV', 'GP-RBF-Python']
available_noise_methods = order4_pivot.index.tolist()
noise_methods = [m for m in candidate_noise_methods if m in available_noise_methods]
if len(noise_methods) > 0:
    order4_subset = order4_pivot.loc[noise_methods]
else:
    # Fallback: use top 4 methods by median performance
    print("  Warning: No candidate methods found, using top 4 by performance")
    order4_means = order4_pivot.median(axis=1).sort_values()
    noise_methods = order4_means.head(4).index.tolist()
    order4_subset = order4_pivot.loc[noise_methods]

latex_noise = output_dir / "tab_noise_sensitivity_order4.tex"
with open(latex_noise, 'w') as f:
    f.write("% AUTO-GENERATED: Noise Sensitivity at Derivative Order 4\n")
    f.write("% Data source: build/results/comprehensive/comprehensive_summary.csv\n")
    f.write("\\begin{tabular}{l" + "r" * len(noise_levels) + "}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Method}")
    for nl in noise_levels:
        f.write(f" & \\textbf{{{nl:.0e}}}")
    f.write(" \\\\\n\\midrule\n")
    for method, row in order4_subset.iterrows():
        method_escaped = str(method).replace('_', '\\_')
        f.write(f"{method_escaped}")
        for nl in noise_levels:
            val = row[nl]
            f.write(f" & {format_nrmse_sci(val)}")
        f.write(" \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")

print(f"  Saved: {latex_noise.name}")

print("\n" + "="*80)
print("TABLE AND PLOT GENERATION COMPLETE")
print("="*80)
print(f"\nTables: {tables_dir}")
print(f"Plots: {plots_dir}")
print(f"LaTeX includes: {output_dir}")
print("="*80)
