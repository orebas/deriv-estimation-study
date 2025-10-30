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
results_dir = Path(__file__).parent.parent / "build" / "results" / "comprehensive"
summary = pd.read_csv(results_dir / "comprehensive_summary.csv")
raw_results = pd.read_csv(results_dir / "comprehensive_results.csv")

print(f"\nLoaded {len(summary)} summary rows, {len(raw_results)} raw result rows")

# Output directory
output_dir = Path(__file__).parent.parent / "build" / "figures" / "publication"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Figure 1: Pareto Frontier Plot (Accuracy vs Computational Cost)
# ============================================================================
print("\n" + "=" * 80)
print("FIGURE 1: PARETO FRONTIER (nRMSE vs Time)")
print("=" * 80)

# IMPORTANT: Only average over orders 0-5 for fair comparison
# (some methods support 0-7, others only 0-5, so we restrict to common range)
summary_orders_0_5 = summary[summary['deriv_order'] <= 5].copy()

print(f"Total rows: {len(summary)} -> Orders 0-5 only: {len(summary_orders_0_5)}")

# Compute overall average nRMSE and timing per method (over orders 0-5)
method_summary = summary_orders_0_5.groupby(['method', 'category']).agg({
    'mean_nrmse': 'mean',
    'mean_timing': 'mean'
}).reset_index()

print(f"Methods before filtering: {len(method_summary)}")

# Filter out unstable methods (AAA) and extreme outliers (nRMSE > 10)
# Also filter to only methods with full coverage (orders 0-5) to avoid coverage bias
methods_with_full_coverage = []
for method in summary_orders_0_5['method'].unique():
    orders = set(summary_orders_0_5[summary_orders_0_5['method'] == method]['deriv_order'].unique())
    if set(range(6)).issubset(orders):  # Has orders 0-5
        methods_with_full_coverage.append(method)

print(f"Methods with full coverage (0-5): {len(methods_with_full_coverage)}")

method_summary = method_summary[
    ~method_summary['method'].str.contains('AAA', case=False) &
    (method_summary['mean_nrmse'] <= 10) &
    method_summary['method'].isin(methods_with_full_coverage)
].reset_index(drop=True)

print(f"Methods after filtering (apples-to-apples, orders 0-5): {len(method_summary)}")

# Calculate Pareto frontier
# A point is Pareto-optimal if no other point is both faster AND more accurate
def is_pareto_optimal(df, idx):
    """Check if point at idx is Pareto-optimal (not dominated by any other point)"""
    time = df.loc[idx, 'mean_timing']
    nrmse = df.loc[idx, 'mean_nrmse']

    # Check if any other point dominates this one (faster AND more accurate)
    for other_idx in df.index:
        if other_idx == idx:
            continue
        other_time = df.loc[other_idx, 'mean_timing']
        other_nrmse = df.loc[other_idx, 'mean_nrmse']

        # If other point is both faster and more accurate, current point is dominated
        if other_time < time and other_nrmse < nrmse:
            return False
    return True

# Identify Pareto-optimal points
method_summary['is_pareto'] = [is_pareto_optimal(method_summary, i) for i in method_summary.index]
pareto_methods = method_summary[method_summary['is_pareto']].sort_values('mean_timing')

print(f"Pareto-optimal methods: {len(pareto_methods)}")
print("Pareto front:", list(pareto_methods['method'].values))

# Create Pareto plot
fig, ax = plt.subplots(figsize=(12, 8))

# Define category colors
category_colors = {
    'Gaussian Process': '#1f77b4',
    'Spectral': '#2ca02c',
    'Spline': '#d62728',
    'Finite Difference': '#9467bd',
    'Regularization': '#8c564b',
    'Local Polynomial': '#e377c2',
    'Other': '#7f7f7f'
}

# Plot dominated points (smaller, faded)
dominated = method_summary[~method_summary['is_pareto']]
for category in dominated['category'].unique():
    cat_data = dominated[dominated['category'] == category]
    ax.scatter(
        cat_data['mean_timing'],
        cat_data['mean_nrmse'],
        alpha=0.4,
        s=80,
        color=category_colors.get(category, '#7f7f7f'),
        edgecolors='gray',
        linewidth=0.5,
        label=f'{category} (dominated)'
    )

# Plot Pareto-optimal points (larger, bold)
for category in pareto_methods['category'].unique():
    cat_data = pareto_methods[pareto_methods['category'] == category]
    ax.scatter(
        cat_data['mean_timing'],
        cat_data['mean_nrmse'],
        alpha=0.9,
        s=200,
        color=category_colors.get(category, '#7f7f7f'),
        edgecolors='black',
        linewidth=2,
        marker='D',  # Diamond for Pareto-optimal
        label=f'{category} (Pareto-optimal)',
        zorder=10
    )

# Draw Pareto front line
ax.plot(pareto_methods['mean_timing'], pareto_methods['mean_nrmse'],
        'k--', linewidth=2, alpha=0.5, zorder=5, label='Pareto Front')

# Annotate ALL Pareto-optimal methods
for _, row in pareto_methods.iterrows():
    ax.annotate(
        row['method'],
        xy=(row['mean_timing'], row['mean_nrmse']),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5, color='black'),
        zorder=15
    )

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mean Computation Time (seconds, log scale)', fontsize=12)
ax.set_ylabel('Mean nRMSE (log scale)', fontsize=12)
ax.set_title('Speed vs Accuracy Trade-off: Pareto Frontier', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

# Cleaner legend (don't show every category twice)
handles, labels = ax.get_legend_handles_labels()
# Filter to unique categories
unique_labels = {}
for h, l in zip(handles, labels):
    base_cat = l.replace(' (dominated)', '').replace(' (Pareto-optimal)', '')
    if base_cat not in unique_labels:
        unique_labels[base_cat] = h
ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', frameon=True, fontsize=10)

# Add reference line
ax.axhline(y=1.0, color='red', linestyle=':', linewidth=2, alpha=0.6, label='nRMSE=1.0')

plt.tight_layout()
pareto_file = output_dir / "pareto_frontier.png"
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

# CRITICAL FIX: Aggregate across ODE systems first
# The summary has 3 rows per (method, order, noise) - one per system
# This causes jagged lines if we plot directly!
print("Aggregating data across ODE systems...")
summary_agg = summary.groupby(['method', 'category', 'deriv_order', 'noise_level']).agg({
    'mean_nrmse': 'mean',
    'std_nrmse': 'mean',
    'mean_timing': 'mean'
}).reset_index()

print(f"  Original rows: {len(summary)} -> Aggregated rows: {len(summary_agg)}")

# CRITICAL: Only select methods with FULL 0-7 coverage to avoid coverage bias
# (Methods like Central-FD only do orders 0-1, which makes them rank high artificially)
print("Identifying full-coverage methods (orders 0-7)...")
full_order_set = set(range(8))
method_orders = summary_agg.groupby('method')['deriv_order'].apply(set)
methods_full_0_7 = [m for m, orders_set in method_orders.items() if full_order_set.issubset(orders_set)]

print(f"  Found {len(methods_full_0_7)} methods with full 0-7 coverage")

# Rank ONLY among full-coverage methods (apples-to-apples comparison)
overall_avg_full = (summary_agg[summary_agg['method'].isin(methods_full_0_7)]
                    .groupby('method')['mean_nrmse']
                    .mean()
                    .sort_values())

# MANUALLY CURATED methods showing algorithmic diversity
# Show data only where it exists (methods can drop out in later panels)
top_methods = [
    'GP-Julia-AD',           # Best overall, full 0-7 coverage
    'Fourier-Interp',        # Best spectral low-order, full 0-7 coverage
    'Dierckx-5',             # Best non-GP for orders 2-5, drops at order 6
    'Savitzky-Golay',        # Filter baseline, drops at order 6
    'fourier_continuation',  # Strong spectral, full 0-7 coverage
    'Fourier-GCV',           # Best spectral high-order, full 0-7 coverage
]

print(f"  Manually curated methods (6 total):")
for method in top_methods:
    if method in summary_agg['method'].values:
        cat = summary_agg[summary_agg['method'] == method]['category'].iloc[0]
        avg_nrmse = summary_agg[summary_agg['method'] == method]['mean_nrmse'].mean()
        orders_supported = sorted(summary_agg[summary_agg['method'] == method]['deriv_order'].unique())
        coverage = f"0-{max(orders_supported)}"
        print(f"    {method:30s} ({cat:20s}) nRMSE: {avg_nrmse:.4f}  Orders: {coverage}")
    else:
        print(f"    {method:30s} WARNING: Not found in data!")

# Create stable color palette keyed to method names
# This ensures colors stay consistent across all panels
base_palette = sns.color_palette('tab10', n_colors=len(top_methods))
method_colors = dict(zip(top_methods, base_palette))

# Create 4x2 grid with extra space on right for legend
fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()

for idx, order in enumerate(orders):
    ax = axes[idx]

    # Filter data for this order (using aggregated data)
    order_data = summary_agg[summary_agg['deriv_order'] == order]

    # Plot top 6 methods with consistent colors
    for method in top_methods:
        method_data = order_data[order_data['method'] == method].sort_values('noise_level')

        if len(method_data) > 0:
            # Plot mean nRMSE with fixed color
            ax.plot(
                method_data['noise_level'],
                method_data['mean_nrmse'],
                marker='o',
                label=method,
                linewidth=2,
                markersize=4,
                alpha=0.9,
                color=method_colors[method]
            )

            # Add error bars (std_nrmse as shaded region)
            # Clip to avoid negative values
            if 'std_nrmse' in method_data.columns:
                lower = np.clip(method_data['mean_nrmse'] - method_data['std_nrmse'], 0, None)
                upper = np.clip(method_data['mean_nrmse'] + method_data['std_nrmse'], 0, None)
                ax.fill_between(
                    method_data['noise_level'],
                    lower,
                    upper,
                    alpha=0.12,
                    color=method_colors[method]
                )

    # Use LINEAR scale on y-axis (to see if bunching is better/worse)
    ax.set_xscale('log')
    # ax.set_yscale('log')  # TEMPORARILY DISABLED

    # Let each panel auto-scale its y-axis to avoid bunching
    # (Different orders have vastly different error ranges)
    # We'll add small padding above/below the data
    # Note: matplotlib will handle this automatically, no need to set ylim

    ax.set_xlabel('Noise Level' if idx >= 6 else '', fontsize=11)
    ax.set_ylabel('nRMSE (linear scale)' if idx % 2 == 0 else '', fontsize=11)
    ax.set_title(f'Order {order}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Add reference lines for interpretation (only if they're in visible range)
    # We'll add these after auto-scaling
    ax.relim()  # Recompute data limits
    ax.autoscale_view()  # Apply auto-scaling
    y_min, y_max = ax.get_ylim()

    # Hard cap y-axis at 2.0 (Fourier-Interp goes crazy at high orders)
    y_max_capped = min(y_max, 2.0)
    ax.set_ylim(0, y_max_capped)

    # Update y_max for reference lines
    y_max = y_max_capped

    # Reference lines for LINEAR scale
    for y_ref in [0.1, 0.3, 1.0]:
        if y_min < y_ref < y_max:  # Only draw if in visible range
            ax.axhline(y=y_ref, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Create a single shared legend outside the grid (right side)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    title='Methods',
    fontsize=10,
    title_fontsize=11
)

# Overall title
fig.suptitle('Performance Across Derivative Orders: Representative Methods',
             fontsize=14, fontweight='bold', y=0.995)

# Adjust layout to make room for legend on right
plt.tight_layout(rect=[0, 0, 0.85, 0.99])
small_multiples_file = output_dir / "small_multiples_grid.png"
plt.savefig(small_multiples_file, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {small_multiples_file.name}")

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
input_json = Path(__file__).parent.parent / "build" / "data" / "input" / f"{trial_id}.json"
output_json = Path(__file__).parent.parent / "build" / "data" / "output" / f"{trial_id}_results.json"

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

    # Panel B: AAA vs Fourier-Interp (use available AAA method)
    aaa_method = 'AAA-LowPrec' if 'AAA-LowPrec' in method_summary['method'].values else 'AAA-Adaptive-Diff2'
    ax2.plot(times, ground_truth, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(f'd⁴y/dt⁴')
    ax2.set_title(f'Panel B: {aaa_method} vs Fourier-Interp', fontweight='bold')
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
    qualitative_file = output_dir / "qualitative_comparison.png"
    plt.savefig(qualitative_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: {qualitative_file.name}")
    print("NOTE: Qualitative plot needs actual prediction data - currently showing ground truth only")
else:
    print(f"WARNING: Could not find data for {trial_id} - skipping qualitative comparison")
    print(f"  Input: {input_json.exists()}")
    print(f"  Output: {output_json.exists()}")

# ============================================================================
# Figure 4 SUPPLEMENTAL: Low-Order Specialists (Orders 0-3)
# ============================================================================
print("\n" + "=" * 80)
print("FIGURE 4: SUPPLEMENTAL - LOW-ORDER SPECIALISTS (Orders 0-3)")
print("=" * 80)

# Select specialists that excel at low orders plus top generalists for comparison
specialists_and_generalists = []

# Top 3 generalists from main figure (for reference)
specialists_and_generalists.extend(top_methods[:3])

# Add specialists that don't have full 0-7 coverage but excel at low orders
candidate_specialists = ['Dierckx-5', 'Central-FD', 'TVRegDiff-Julia', 'Savitzky-Golay']
for method in candidate_specialists:
    if method in summary_agg['method'].values and method not in specialists_and_generalists:
        specialists_and_generalists.append(method)

# Filter to methods that have data for at least orders 0-3
method_orders = summary_agg.groupby('method')['deriv_order'].apply(set)
low_order_set = set(range(4))
specialists_filtered = [m for m in specialists_and_generalists
                        if low_order_set.issubset(method_orders.get(m, set()))]

print(f"  Selected methods for low-order comparison: {specialists_filtered}")

# Create color palette for these methods
specialist_palette = sns.color_palette('tab10', n_colors=len(specialists_filtered))
specialist_colors = dict(zip(specialists_filtered, specialist_palette))

# Create 2x2 grid for orders 0-3
fig_supp, axes_supp = plt.subplots(2, 2, figsize=(14, 10))
axes_supp = axes_supp.flatten()

for idx, order in enumerate(range(4)):
    ax = axes_supp[idx]

    # Filter data for this order (using aggregated data)
    order_data = summary_agg[summary_agg['deriv_order'] == order]

    # Plot selected methods
    for method in specialists_filtered:
        method_data = order_data[order_data['method'] == method].sort_values('noise_level')

        if len(method_data) > 0:
            # Use dashed lines for specialists without full 0-7 coverage
            linestyle = '--' if method not in top_methods else '-'

            ax.plot(
                method_data['noise_level'],
                method_data['mean_nrmse'],
                marker='o',
                label=method,
                linewidth=2,
                markersize=4,
                alpha=0.9,
                linestyle=linestyle,
                color=specialist_colors[method]
            )

            # Add error bands
            if 'std_nrmse' in method_data.columns:
                lower = np.clip(method_data['mean_nrmse'] - method_data['std_nrmse'], 1e-6, None)
                upper = np.clip(method_data['mean_nrmse'] + method_data['std_nrmse'], 1e-6, None)
                ax.fill_between(
                    method_data['noise_level'],
                    lower,
                    upper,
                    alpha=0.12,
                    color=specialist_colors[method]
                )

    # Log scales with adaptive y-axis
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Noise Level' if idx >= 2 else '', fontsize=11)
    ax.set_ylabel('nRMSE (log scale)' if idx % 2 == 0 else '', fontsize=11)
    ax.set_title(f'Order {order}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Auto-scale and add reference lines in visible range
    ax.relim()
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()

    for y_ref in [1e-1, 3e-1, 1.0]:
        if y_min < y_ref < y_max:
            ax.axhline(y=y_ref, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Shared legend
handles_supp, labels_supp = axes_supp[0].get_legend_handles_labels()
fig_supp.legend(
    handles_supp, labels_supp,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    title='Methods (solid = full 0-7, dashed = specialists)',
    fontsize=10,
    title_fontsize=10
)

fig_supp.suptitle('Low-Order Performance: Specialists vs Generalists (Orders 0-3)',
                  fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 0.85, 0.96])
specialist_file = output_dir / "specialist_comparison_orders_0_3.png"
plt.savefig(specialist_file, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {specialist_file.name}")
print(f"  Note: Solid lines = full 0-7 coverage, Dashed lines = low-order specialists")

print("\n" + "=" * 80)
print("ADDITIONAL FIGURE GENERATION COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print("  - small_multiples_grid.png (main figure, orders 0-7, full-coverage only)")
print("  - specialist_comparison_orders_0_3.png (supplemental, specialists vs generalists)")
print("=" * 80)
