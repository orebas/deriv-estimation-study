#!/usr/bin/env python3
"""
Create Excel workbook from comprehensive study results.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "comprehensive"
OUTPUT_FILE = RESULTS_DIR / "comprehensive_results.xlsx"

print("Loading CSV files...")
raw_data = pd.read_csv(RESULTS_DIR / "comprehensive_results.csv")
summary_data = pd.read_csv(RESULTS_DIR / "comprehensive_summary.csv")

print(f"Raw data: {len(raw_data)} rows")
print(f"Summary data: {len(summary_data)} rows")

# Add helper columns for better pivot table experience
print("\nAdding helper columns...")

# For raw data
raw_data['noise_pct'] = raw_data['noise_level'] * 100
raw_data['log10_rmse'] = np.log10(raw_data['rmse'].replace(0, np.nan))
raw_data['log10_mae'] = np.log10(raw_data['mae'].replace(0, np.nan))

# For summary data
summary_data['noise_pct'] = summary_data['noise_level'] * 100
summary_data['log10_mean_rmse'] = np.log10(summary_data['mean_rmse'].replace(0, np.nan))
summary_data['log10_mean_mae'] = np.log10(summary_data['mean_mae'].replace(0, np.nan))

# Create Excel file with multiple sheets
print(f"\nWriting Excel file: {OUTPUT_FILE}")
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    # Sheet 1: Raw trial data
    raw_data.to_excel(writer, sheet_name='Raw Data', index=False)

    # Sheet 2: Summary statistics
    summary_data.to_excel(writer, sheet_name='Summary', index=False)

    # Sheet 3: Pivot-ready view (wide format for noise levels)
    print("Creating pivot-ready view...")
    pivot_ready = summary_data.pivot_table(
        index=['method', 'category', 'language', 'deriv_order'],
        columns='noise_level',
        values='mean_rmse',
        aggfunc='first'
    ).reset_index()

    # Rename noise level columns to be more readable
    pivot_ready.columns.name = None
    col_rename = {col: f'RMSE_noise_{col:.0e}' if isinstance(col, float) else col
                  for col in pivot_ready.columns}
    pivot_ready = pivot_ready.rename(columns=col_rename)

    pivot_ready.to_excel(writer, sheet_name='Pivot Ready', index=False)

    # Sheet 4: Method comparison at 1% noise
    print("Creating 1% noise comparison...")
    noise_1pct = summary_data[summary_data['noise_level'] == 0.01].copy()
    noise_1pct = noise_1pct.sort_values(['deriv_order', 'mean_rmse'])
    noise_1pct.to_excel(writer, sheet_name='1pct Noise', index=False)

    # Format headers
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True)

print("\n" + "="*60)
print(f"Excel file created: {OUTPUT_FILE}")
print("="*60)
print("\nSheets included:")
print("  1. Raw Data - All individual trial results (3633 rows)")
print("  2. Summary - Aggregated statistics (1211 rows)")
print("  3. Pivot Ready - Wide format for easy analysis")
print("  4. 1pct Noise - Filtered view at 1% noise level")
print("\nColumns in Raw Data:")
for col in raw_data.columns:
    print(f"  - {col}")
print("\nReady for pivot tables in Excel!")
print("="*60)
