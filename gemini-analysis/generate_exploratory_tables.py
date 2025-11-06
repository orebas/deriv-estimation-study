#!/usr/bin/env python3
"""
Generate exploratory summary tables for different maximum derivative orders.
Outputs both markdown and LaTeX formats for paper integration.
"""
import pandas as pd
from pathlib import Path
import sys

# Config
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SUMMARY_CSV = REPO_ROOT / 'build/results/comprehensive/comprehensive_summary.csv'
OUT_DIR_MD = SCRIPT_DIR  # Markdown goes to gemini-analysis/
OUT_DIR_TEX = REPO_ROOT / 'build/tables/publication'  # LaTeX goes to publication tables

# Method name mapping for pretty printing
METHOD_MAP = {
    'Savitzky-Golay': 'Savitzky-Golay',
    'GP-Julia-AD': 'GP-Julia-AD',
    'Dierckx-5': 'Dierckx-5',
    'Fourier-GCV': 'Fourier-GCV',
    'fourier_continuation': 'Fourier-Continuation',
    'GP_RBF_Python': 'GP-RBF-Python',
    'GP_RBF_Iso_Python': 'GP-RBF-Iso-Python',
    'Fourier-FFT-Adaptive-Python': 'FFT-Adaptive-Py',
    'gp_rbf_mean': 'GP-RBF-Mean-Py',
    'Fourier-Continuation-Adaptive': 'Fourier-Cont-Adaptive',
    'fourier': 'Fourier',
    'GSS': 'GSS',
    'chebyshev': 'Chebyshev',
    'Chebyshev-AICc': 'Chebyshev-AICc',
    'Fourier-FFT-Adaptive': 'FFT-Adaptive-Julia',
    'GP-Julia-SE': 'GP-Julia-SE',
    'Central-FD': 'Central-FD',
    'TVRegDiff-Julia': 'TVRegDiff-Julia',
}


def generate_summary_table(df: pd.DataFrame, max_order: int, low_noise_levels: list, high_noise_levels: list) -> pd.DataFrame:
    """
    Generates a summary table for methods with full coverage up to a max derivative order.
    """
    # 1. Filter data to the specified order range
    df_filtered_order = df[df.deriv_order <= max_order].copy()

    # 2. Determine contenders based on coverage within this order range
    union_df = df_filtered_order[df_filtered_order.noise_level.isin(low_noise_levels + high_noise_levels)].copy()
    odes = sorted(union_df['ode_system'].unique())
    orders = sorted(union_df['deriv_order'].unique())
    noises = sorted(set(low_noise_levels + high_noise_levels))
    expected_cells = len(odes) * len(orders) * len(noises)

    cells_per_method = union_df.groupby('method').size()
    contenders = sorted(cells_per_method[cells_per_method == expected_cells].index.tolist())

    # 3. Split into noise regimes
    low_noise_df = df_filtered_order[df_filtered_order.noise_level.isin(low_noise_levels)].copy()
    high_noise_df = df_filtered_order[df_filtered_order.noise_level.isin(high_noise_levels)].copy()

    # 4. Helper to summarize each regime for the identified contenders
    def summarize_regime(df_regime):
        df_c = df_regime[df_regime['method'].isin(contenders)].copy()
        df_c['rank'] = df_c.groupby(['ode_system', 'deriv_order', 'noise_level'])['mean_nrmse'].rank()
        summary = df_c.groupby('method').agg(
            avg_rank=('rank', 'mean'),
            median_nrmse=('mean_nrmse', 'median'),
            q1_nrmse=('mean_nrmse', lambda x: x.quantile(0.25)),
            q3_nrmse=('mean_nrmse', lambda x: x.quantile(0.75)),
            success_rate=('mean_nrmse', lambda x: (x < 1.0).mean() * 100),
            catastrophic_rate=('mean_nrmse', lambda x: (x > 10.0).mean() * 100)
        ).reset_index()
        return summary.sort_values('avg_rank')

    low_summary = summarize_regime(low_noise_df)
    high_summary = summarize_regime(high_noise_df)

    # 5. Merge and format the final table
    merged = pd.merge(
        low_summary, high_summary,
        on='method',
        suffixes=('_low', '_high'),
        how='inner'
    )
    merged['avg_rank_overall'] = (merged['avg_rank_low'] + merged['avg_rank_high']) / 2
    merged['success_rate_overall'] = (merged['success_rate_low'] + merged['success_rate_high']) / 2
    merged = merged.sort_values('avg_rank_overall').reset_index(drop=True)

    final = pd.DataFrame({
        'Method': merged['method'].map(lambda m: METHOD_MAP.get(m, m)),
        'Avg. Rank': merged['avg_rank_overall'],
        'Median nRMSE (Overall)': (merged['median_nrmse_low'] + merged['median_nrmse_high']) / 2,
        'Success Rate (%)': merged['success_rate_overall'],
        'Low Noise Median': merged['median_nrmse_low'],
        'Low Noise IQR': merged.apply(lambda r: f"[{r['q1_nrmse_low']:.3f}, {r['q3_nrmse_low']:.3f}]", axis=1),
        'High Noise Median': merged['median_nrmse_high'],
        'High Noise IQR': merged.apply(lambda r: f"[{r['q1_nrmse_high']:.3f}, {r['q3_nrmse_high']:.3f}]", axis=1),
    })

    # 6. Format for markdown output
    def cap(x):
        try:
            val = float(x)
            if val > 10: return '>10'
            return f"{val:.3f}"
        except (ValueError, TypeError):
            return x

    final_fmt = final.copy()
    final_fmt['Low Noise Median'] = final_fmt['Low Noise Median'].map(cap)
    final_fmt['High Noise Median'] = final_fmt['High Noise Median'].map(cap)
    final_fmt['Median nRMSE (Overall)'] = final_fmt['Median nRMSE (Overall)'].map(cap)
    final_fmt['Avg. Rank'] = final_fmt['Avg. Rank'].map(lambda v: f"{v:.1f}")
    final_fmt['Success Rate (%)'] = final_fmt['Success Rate (%)'].map(lambda v: f"{v:.1f}")

    return final_fmt


def table_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to LaTeX table format."""
    latex = "% AUTO-GENERATED by gemini-analysis/generate_exploratory_tables.py\n"
    latex += "% Data source: build/results/comprehensive/comprehensive_summary.csv\n"
    latex += "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{lrrrllll}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Method} & \\textbf{Avg.} & \\textbf{Median} & \\textbf{Success} & \\textbf{Low Noise} & \\textbf{Low Noise} & \\textbf{High Noise} & \\textbf{High Noise} \\\\\n"
    latex += "& \\textbf{Rank} & \\textbf{nRMSE} & \\textbf{Rate (\\%)} & \\textbf{Median} & \\textbf{IQR} & \\textbf{Median} & \\textbf{IQR} \\\\\n"
    latex += "\\midrule\n"

    for _, row in df.iterrows():
        # Escape underscores in method names for LaTeX
        method_name = str(row['Method']).replace('_', '\\_')
        latex += f"{method_name} & {row['Avg. Rank']} & {row['Median nRMSE (Overall)']} & "
        latex += f"{row['Success Rate (%)']} & {row['Low Noise Median']} & "
        latex += f"{str(row['Low Noise IQR']).replace('[', '').replace(']', '')} & "
        latex += f"{row['High Noise Median']} & "
        latex += f"{str(row['High Noise IQR']).replace('[', '').replace(']', '')} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    return latex


def main():
    """Main execution function."""
    if not SUMMARY_CSV.exists():
        print(f"ERROR: Summary CSV not found: {SUMMARY_CSV}")
        print("Run the comprehensive study first: ./scripts/02_run_comprehensive.sh")
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} rows from {SUMMARY_CSV}")

    # Regime noise levels
    # Low-noise: everything below 0.01 (1e-8, 1e-6, 1e-4, 1e-3)
    # High-noise: 0.01 and above (0.01, 0.02)
    LOW_NOISE = [1e-8, 1e-6, 1e-4, 1e-3]
    HIGH_NOISE = [0.01, 0.02]

    # Generate tables for orders 3, 5, and 7
    tables = {
        3: generate_summary_table(df, 3, LOW_NOISE, HIGH_NOISE),
        5: generate_summary_table(df, 5, LOW_NOISE, HIGH_NOISE),
        7: generate_summary_table(df, 7, LOW_NOISE, HIGH_NOISE),
    }

    # Create output directories
    OUT_DIR_TEX.mkdir(parents=True, exist_ok=True)

    # Write markdown file
    md_file = OUT_DIR_MD / 'exploratory_order_analysis.md'
    with open(md_file, 'w') as f:
        f.write("# Exploratory Analysis by Derivative Order\n\n")
        f.write("This file contains alternative summary tables based on different maximum derivative orders for method inclusion.\n\n")

        for order, table in tables.items():
            f.write(f"## Table: Contenders with Full Coverage up to Order {order}\n\n")
            f.write(f"Methods included here have complete data for all noise levels and ODE systems for derivative orders 0 through {order}. ")
            f.write("Averages and ranks are computed over this range.\n\n")
            f.write(table.to_markdown(index=False))
            f.write("\n\n")

    print(f"✓ Wrote markdown to: {md_file}")

    # Write LaTeX files
    captions = {
        3: "Contender Method Performance Summary (Orders 0-3)",
        5: "Contender Method Performance Summary (Orders 0-5)",
        7: "Contender Method Performance Summary (Orders 0-7)",
    }

    for order, table in tables.items():
        tex_file = OUT_DIR_TEX / f"tab_summary_order{order}.tex"
        latex_content = table_to_latex(
            table,
            caption=captions[order],
            label=f"tab:summary_order{order}"
        )
        with open(tex_file, 'w') as f:
            f.write(latex_content)
        print(f"✓ Wrote LaTeX to: {tex_file}")

    print(f"\n{'='*80}")
    print("EXPLORATORY TABLE GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Generated {len(tables)} tables (orders 3, 5, 7) in both markdown and LaTeX formats")
    print(f"  Markdown: {OUT_DIR_MD}/exploratory_order_analysis.md")
    print(f"  LaTeX:    {OUT_DIR_TEX}/tab_summary_order{{3,5,7}}.tex")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
