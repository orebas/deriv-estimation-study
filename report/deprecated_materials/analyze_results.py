#!/usr/bin/env python3
"""
Automated Data Analysis Pipeline - Truth-Only Mode

This script:
1. Reads ACTUAL experimental data from comprehensive_summary.csv
2. Calculates TRUE overall rankings
3. Generates LaTeX tables DIRECTLY from data
4. Outputs JSON summary for verification

NO MANUAL DATA ENTRY. NO FABRICATION. ONLY TRUTH.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths - use ACTUAL data files only
DATA_DIR = Path("/home/orebas/derivative_estimation_study")
RESULTS_FILE = DATA_DIR / "results/comprehensive/comprehensive_summary.csv"
OUTPUT_DIR = DATA_DIR / "report/paper_figures/automated"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_real_data():
    """Load actual experimental results - NO FABRICATION"""
    print("Loading REAL experimental data...")
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} rows of ACTUAL experimental results")
    print(f"Methods: {df['method'].nunique()}")
    print(f"Orders: {sorted(df['deriv_order'].unique())}")
    print(f"Noise levels: {sorted(df['noise_level'].unique())}")
    return df

def calculate_true_overall_rankings(df):
    """Calculate TRUE overall rankings from actual data"""
    print("\nCalculating TRUE overall rankings...")

    # Group by method and calculate mean nRMSE across ALL configurations
    overall = df.groupby('method').agg({
        'mean_nrmse': 'mean',
        'mean_timing': 'mean',
        'category': 'first'
    }).reset_index()

    # Sort by mean_nrmse (lower is better)
    overall = overall.sort_values('mean_nrmse')
    overall['rank'] = range(1, len(overall) + 1)

    print(f"\nTOP 10 METHODS (by actual mean nRMSE):")
    for _, row in overall.head(10).iterrows():
        print(f"  {row['rank']:2d}. {row['method']:25s} nRMSE={row['mean_nrmse']:12.4f} time={row['mean_timing']:8.3f}s")

    print(f"\nBOTTOM 5 METHODS (worst performers):")
    for _, row in overall.tail(5).iterrows():
        print(f"  {row['rank']:2d}. {row['method']:25s} nRMSE={row['mean_nrmse']:12.4e} time={row['mean_timing']:8.3f}s")

    return overall

def generate_order_tables(df):
    """Generate per-order tables from ACTUAL data"""
    print("\nGenerating per-order tables from REAL data...")

    tables = {}
    for order in sorted(df['deriv_order'].unique()):
        order_df = df[df['deriv_order'] == order].copy()

        # Pivot: methods as rows, noise levels as columns
        pivot = order_df.pivot_table(
            index='method',
            columns='noise_level',
            values='mean_nrmse',
            aggfunc='mean'
        )

        # Calculate row means
        pivot['Mean'] = pivot.mean(axis=1)

        # Sort by mean
        pivot = pivot.sort_values('Mean')

        tables[f'order_{order}'] = pivot

        print(f"  Order {order}: Top method = {pivot.index[0]} (nRMSE={pivot['Mean'].iloc[0]:.4f})")

    return tables

def save_latex_tables(tables, output_dir):
    """Save tables as LaTeX - FROM ACTUAL DATA ONLY"""
    print("\nSaving LaTeX tables (from REAL data)...")

    for name, df in tables.items():
        output_file = output_dir / f"{name}_nrmse.tex"

        # Format numbers properly
        with open(output_file, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{nRMSE for " + name.replace('_', ' ').title() + " (ACTUAL DATA)}\n")
            f.write("\\label{tab:" + name + "}\n")
            f.write("\\tiny\n")
            f.write("\\begin{tabular}{l" + "c" * len(df.columns) + "}\n")
            f.write("\\toprule\n")

            # Header
            header = "\\textbf{Method} & " + " & ".join([f"\\textbf{{{col}}}" for col in df.columns]) + " \\\\\n"
            f.write(header)
            f.write("\\midrule\n")

            # Rows - TOP 15 methods only
            for method in df.index[:15]:
                row_data = []
                for col in df.columns:
                    val = df.loc[method, col]
                    if pd.isna(val):
                        row_data.append("---")
                    elif val < 0.01:
                        row_data.append(f"{val:.4f}")
                    elif val < 10:
                        row_data.append(f"{val:.2f}")
                    elif val < 1000:
                        row_data.append(f"{val:.1f}")
                    else:
                        row_data.append(f"${val:.2e}$")

                f.write(method + " & " + " & ".join(row_data) + " \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"  Saved: {output_file}")

def save_summary_json(overall, tables, output_dir):
    """Save machine-readable summary for verification"""
    print("\nSaving JSON summary for verification...")

    summary = {
        "metadata": {
            "source": str(RESULTS_FILE),
            "generated_by": "automated_pipeline",
            "warning": "ALL DATA FROM ACTUAL EXPERIMENTS - NO FABRICATION"
        },
        "overall_rankings": overall.to_dict('records'),
        "per_order_top_methods": {}
    }

    for name, df in tables.items():
        summary["per_order_top_methods"][name] = {
            "best_method": df.index[0],
            "best_nrmse": float(df['Mean'].iloc[0]),
            "top_5": df.index[:5].tolist()
        }

    output_file = output_dir / "true_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {output_file}")
    return summary

def identify_aaa_failures(df):
    """Document AAA method failures from ACTUAL data"""
    print("\nAnalyzing AAA method performance (ACTUAL DATA)...")

    aaa_df = df[df['method'].str.contains('AAA')].copy()

    print("\nAAA-HighPrec performance by order:")
    for order in sorted(aaa_df['deriv_order'].unique()):
        order_data = aaa_df[(aaa_df['deriv_order'] == order) & (aaa_df['method'] == 'AAA-HighPrec')]
        if len(order_data) > 0:
            mean_nrmse = order_data['mean_nrmse'].mean()
            max_nrmse = order_data['mean_nrmse'].max()
            print(f"  Order {order}: mean={mean_nrmse:.2e}, max={max_nrmse:.2e}")

    print("\nAAA-LowPrec performance by order:")
    for order in sorted(aaa_df['deriv_order'].unique()):
        order_data = aaa_df[(aaa_df['deriv_order'] == order) & (aaa_df['method'] == 'AAA-LowPrec')]
        if len(order_data) > 0:
            mean_nrmse = order_data['mean_nrmse'].mean()
            max_nrmse = order_data['mean_nrmse'].max()
            print(f"  Order {order}: mean={mean_nrmse:.2e}, max={max_nrmse:.2e}")

def main():
    print("="*80)
    print("AUTOMATED DATA ANALYSIS PIPELINE - TRUTH-ONLY MODE")
    print("="*80)

    # Load REAL data
    df = load_real_data()

    # Calculate TRUE rankings
    overall = calculate_true_overall_rankings(df)

    # Generate per-order tables
    tables = generate_order_tables(df)

    # Save LaTeX tables FROM DATA
    save_latex_tables(tables, OUTPUT_DIR)

    # Save JSON summary
    summary = save_summary_json(overall, tables, OUTPUT_DIR)

    # Document AAA failures
    identify_aaa_failures(df)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE - ALL OUTPUT FROM ACTUAL DATA")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - LaTeX tables: {OUTPUT_DIR}/order_*_nrmse.tex")
    print(f"  - JSON summary: {OUTPUT_DIR}/true_summary.json")
    print(f"\nIMPORTANT: All numbers are from REAL experimental data.")
    print(f"           NO fabrication. NO placeholders. ONLY TRUTH.")

    return summary

if __name__ == "__main__":
    summary = main()
