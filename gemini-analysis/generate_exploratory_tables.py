#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Config
ROOT = Path('/home/orebas/tmp/deriv-estimation-study')
SUMMARY_CSV = ROOT / 'build/results/comprehensive/comprehensive_summary.csv'
OUT_FILE = ROOT / 'gemini-analysis/exploratory_order_analysis.md'

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
            avg_nrmse=('mean_nrmse', 'mean')
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
    merged = merged.sort_values('avg_rank_overall').reset_index(drop=True)

    final = pd.DataFrame({
        'Method': merged['method'].map(lambda m: METHOD_MAP.get(m, m)),
        'Avg. Rank (Overall)': merged['avg_rank_overall'],
        'Avg. Rank (Low Noise)': merged['avg_rank_low'],
        'Avg. nRMSE (Low Noise)': merged['avg_nrmse_low'],
        'Avg. Rank (High Noise)': merged['avg_rank_high'],
        'Avg. nRMSE (High Noise)': merged['avg_nrmse_high'],
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
    final_fmt['Avg. nRMSE (Low Noise)'] = final_fmt['Avg. nRMSE (Low Noise)'].map(cap)
    final_fmt['Avg. nRMSE (High Noise)'] = final_fmt['Avg. nRMSE (High Noise)'].map(cap)
    final_fmt['Avg. Rank (Overall)'] = final_fmt['Avg. Rank (Overall)'].map(lambda v: f"{v:.1f}")
    final_fmt['Avg. Rank (Low Noise)'] = final_fmt['Avg. Rank (Low Noise)'].map(lambda v: f"{v:.1f}")
    final_fmt['Avg. Rank (High Noise)'] = final_fmt['Avg. Rank (High Noise)'].map(lambda v: f"{v:.1f}")

    return final_fmt


def main():
    """Main execution function."""
    df = pd.read_csv(SUMMARY_CSV)

    # Regime noise levels (exclude 5%)
    LOW_NOISE = [1e-8, 1e-6]
    HIGH_NOISE = [0.01, 0.02]

    # Generate the two requested tables
    table_ord5 = generate_summary_table(df, 5, LOW_NOISE, HIGH_NOISE)
    table_ord3 = generate_summary_table(df, 3, LOW_NOISE, HIGH_NOISE)

    # Write to a single markdown file
    with open(OUT_FILE, 'w') as f:
        f.write("# Exploratory Analysis by Derivative Order\n\n")
        f.write("This file contains alternative summary tables based on different maximum derivative orders for method inclusion.\n\n")
        f.write("## Table 1: Contenders with Full Coverage up to Order 5\n\n")
        f.write("Methods included here have complete data for all noise levels and ODE systems for derivative orders 0 through 5. Averages and ranks are computed over this range.\n\n")
        f.write(table_ord5.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Table 2: Contenders with Full Coverage up to Order 3\n\n")
        f.write("Methods included here have complete data for all noise levels and ODE systems for derivative orders 0 through 3. Averages and ranks are computed over this range.\n\n")
        f.write(table_ord3.to_markdown(index=False))
        f.write("\n")

    print(f"Wrote exploratory tables to:\n  {OUT_FILE}")


if __name__ == '__main__':
    main()
