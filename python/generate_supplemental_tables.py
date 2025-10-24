#!/usr/bin/env python3
"""
Generate supplemental tables for the derivative estimation study.

Outputs:
1. NRMSE pivot tables (one per derivative order, sorted by average performance)
2. Average timing table for all methods
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("GENERATING SUPPLEMENTAL TABLES")
print("=" * 80)

# Load data
results_dir = Path(__file__).parent.parent / "build" / "results" / "comprehensive"
summary = pd.read_csv(results_dir / "comprehensive_summary.csv")

print(f"\nLoaded {len(summary)} rows from summary data")
print(f"Columns: {', '.join(summary.columns)}")

# Output directories
tables_dir = Path(__file__).parent.parent / "build" / "tables" / "supplemental"
tables_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {tables_dir}")

# Noise levels and derivative orders
noise_levels = sorted(summary['noise_level'].unique())
orders = sorted(summary['deriv_order'].unique())

print(f"\nNoise levels: {noise_levels}")
print(f"Derivative orders: {orders}")

# ============================================================================
# TABLE 1: NRMSE Pivot Tables (one per derivative order)
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING NRMSE PIVOT TABLES BY DERIVATIVE ORDER")
print("=" * 80)

for order in orders:
    print(f"\nOrder {order}...")

    # Filter data for this order
    order_data = summary[summary['deriv_order'] == order].copy()

    # Pivot table: methods × noise levels
    pivot = order_data.pivot_table(
        index='method',
        columns='noise_level',
        values='mean_nrmse',
        aggfunc='mean'
    )

    # Add grand total (average across all noise levels)
    pivot['Grand Total'] = pivot.mean(axis=1)

    # Sort by grand total (ascending = best methods first)
    pivot = pivot.sort_values('Grand Total')

    # Format column names
    col_names = {nl: f'{nl:.8f}' for nl in noise_levels}
    col_names['Grand Total'] = 'Grand Total'
    pivot.rename(columns=col_names, inplace=True)

    # Save as CSV
    csv_file = tables_dir / f"nrmse_order_{order}.csv"
    pivot.to_csv(csv_file, float_format='%.5f')
    print(f"  Saved CSV: {csv_file.name}")

    # Save as LaTeX table
    latex_file = tables_dir / f"nrmse_order_{order}.tex"

    # Format values for LaTeX (5 decimal places)
    pivot_formatted = pivot.copy()
    for col in pivot_formatted.columns:
        pivot_formatted[col] = pivot_formatted[col].apply(lambda x: f'{x:.5f}' if pd.notna(x) else '—')

    with open(latex_file, 'w') as f:
        f.write(f"% NRMSE values for derivative order {order}\n")
        f.write(f"% Sorted by average performance (best to worst)\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Normalized RMSE by noise level for derivative order " + str(order) + "}\n")
        f.write(f"\\label{{tab:nrmse_order_{order}}}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(pivot_formatted.to_latex(
            escape=False,
            column_format='l' + 'r' * len(pivot_formatted.columns),
            index=True
        ))
        f.write("}\n")
        f.write("\\end{table}\n")

    print(f"  Saved LaTeX: {latex_file.name}")
    print(f"  Methods: {len(pivot)}")

print(f"\n✓ Generated {len(orders)} NRMSE pivot tables")

# ============================================================================
# TABLE 2: Average Timing Table
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING AVERAGE TIMING TABLE")
print("=" * 80)

# Calculate average timing per method across all conditions
timing_data = summary.groupby('method').agg({
    'mean_timing': 'mean',
    'deriv_order': 'count'  # Number of data points
}).reset_index()

timing_data.rename(columns={
    'mean_timing': 'Average Time (s)',
    'deriv_order': 'Data Points'
}, inplace=True)

# Sort by average time (fastest first)
timing_data = timing_data.sort_values('Average Time (s)')

# Save as CSV
csv_file = tables_dir / "average_timing.csv"
timing_data.to_csv(csv_file, index=False, float_format='%.6f')
print(f"\nSaved CSV: {csv_file.name}")

# Save as LaTeX table
latex_file = tables_dir / "average_timing.tex"

# Format timing values for LaTeX
timing_formatted = timing_data.copy()
timing_formatted['Average Time (s)'] = timing_formatted['Average Time (s)'].apply(
    lambda x: f'{x:.6f}' if x < 1.0 else f'{x:.3f}'
)

with open(latex_file, 'w') as f:
    f.write("% Average timing for all methods\n")
    f.write("% Sorted by average execution time (fastest to slowest)\n")
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Average execution time per method across all conditions}\n")
    f.write("\\label{tab:average_timing}\n")
    f.write(timing_formatted.to_latex(
        escape=False,
        column_format='lrr',
        index=False
    ))
    f.write("\\end{table}\n")

print(f"Saved LaTeX: {latex_file.name}")
print(f"Methods: {len(timing_data)}")

# Print summary statistics
print("\n" + "=" * 80)
print("TIMING SUMMARY STATISTICS")
print("=" * 80)
print(f"Fastest method: {timing_data.iloc[0]['method']} ({timing_data.iloc[0]['Average Time (s)']:.6f} s)")
print(f"Slowest method: {timing_data.iloc[-1]['method']} ({timing_data.iloc[-1]['Average Time (s)']:.6f} s)")
print(f"Median time: {timing_data['Average Time (s)'].median():.6f} s")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUPPLEMENTAL TABLES GENERATION COMPLETE")
print("=" * 80)
print(f"\nGenerated:")
print(f"  {len(orders)} NRMSE pivot tables (CSV + LaTeX)")
print(f"  1 average timing table (CSV + LaTeX)")
print(f"\nOutput directory: {tables_dir}")
print("=" * 80)
