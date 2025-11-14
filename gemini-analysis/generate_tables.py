#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Config
ROOT = Path('/home/orebas/tmp/deriv-estimation-study')
SUMMARY_CSV = ROOT / 'build/results/comprehensive/comprehensive_summary.csv'
OUT_DIR = ROOT / 'gemini-analysis/paper/sections'

# Method name mapping for pretty printing
METHOD_MAP = {
    'Savitzky-Golay': 'Savitzky-Golay',
    'GP-TaylorAD-Julia': 'GP-TaylorAD-Julia',
    'Spline-Dierckx-5': 'Spline-Dierckx-5',
    'Fourier-GCV': 'Fourier-GCV',
    'Fourier-Continuation-Python': 'Fourier-Continuation',
    'GP-RBF-Python': 'GP-RBF-Python',
    'GP-RBF-Iso-Python': 'GP-RBF-Iso-Python',
    'Fourier-Adaptive-Python': 'FFT-Adaptive-Py',
    'GP-RBF-MeanSub-Python': 'GP-RBF-Mean-Py',
    'Fourier-Continuation-Adaptive': 'Fourier-Cont-Adaptive',
    'Fourier-Basic-Python': 'Fourier',
    'Spline-GSS': 'Spline-GSS',
    'Chebyshev-Basic-Python': 'Chebyshev',
    'Chebyshev-AICc': 'Chebyshev-AICc',
    'Fourier-Adaptive-Julia': 'FFT-Adaptive-Julia',
    'GP-Julia-SE': 'GP-Julia-SE',
    'FiniteDiff-Central': 'FiniteDiff-Central',
    'TVRegDiff-Julia': 'TVRegDiff-Julia',
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)

    # Regime noise levels (exclude 5%)
    LOW_NOISE = [1e-8, 1e-6]
    HIGH_NOISE = [0.01, 0.02]

    # Determine contenders: methods with full coverage across union of noise levels
    union_df = df[df.noise_level.isin(LOW_NOISE + HIGH_NOISE)].copy()
    odes = sorted(union_df['ode_system'].unique())
    orders = sorted(union_df['deriv_order'].unique())
    noises = sorted(set(LOW_NOISE + HIGH_NOISE))
    expected_cells = len(odes) * len(orders) * len(noises)

    # Count coverage cells per method and dynamically find contenders
    cells_per_method = (
        union_df.groupby('method').size()
    )
    full_coverage_methods = sorted(cells_per_method[cells_per_method == expected_cells].index.tolist())

    low_noise = df[df.noise_level.isin(LOW_NOISE)].copy()
    high_noise = df[df.noise_level.isin(HIGH_NOISE)].copy()

    # Helper: summarize within contenders only
    def summarize_regime(df_regime):
        df_c = df_regime[df_regime['method'].isin(full_coverage_methods)].copy()
        # Rank among contenders only, within each cell
        df_c['rank'] = df_c.groupby(['ode_system', 'deriv_order', 'noise_level'])['mean_nrmse'].rank()
        summary = df_c.groupby('method').agg(
            avg_rank=('rank', 'mean'),
            avg_nrmse=('mean_nrmse', 'mean')
        ).reset_index()
        return summary.sort_values('avg_rank')

    low_summary = summarize_regime(low_noise)
    high_summary = summarize_regime(high_noise)

    # Merge low/high and map display names
    merged = pd.merge(
        low_summary, high_summary,
        on='method',
        suffixes=('_low', '_high'),
        how='inner'  # contenders must appear in both regimes
    ).sort_values('avg_rank_high').reset_index(drop=True)

    final = pd.DataFrame({
        'Method': merged['method'].map(lambda m: METHOD_MAP.get(m, m)),
        'Avg. Rank (Low Noise)': merged['avg_rank_low'],
        'Avg. nRMSE (Low Noise)': merged['avg_nrmse_low'],
        'Avg. Rank (High Noise)': merged['avg_rank_high'],
        'Avg. nRMSE (High Noise)': merged['avg_nrmse_high'],
    })

    # Save master summary (CSV and MD) with nRMSE capping in MD
    md_path = OUT_DIR / 'master_summary_table.md'
    csv_path = OUT_DIR / 'master_summary_table.csv'

    # CSV: raw numeric
    final.to_csv(csv_path, index=False)

    # MD: cap nRMSE > 10
    fmt = final.copy()
    def cap(x):
        try:
            val = float(x)
            if val > 10: return '>10'
            return f"{val:.3f}"
        except (ValueError, TypeError):
            return x
    fmt['Avg. nRMSE (Low Noise)'] = fmt['Avg. nRMSE (Low Noise)'].map(cap)
    fmt['Avg. nRMSE (High Noise)'] = fmt['Avg. nRMSE (High Noise)'].map(cap)
    # Ranks to one decimal
    fmt['Avg. Rank (Low Noise)'] = fmt['Avg. Rank (Low Noise)'].map(lambda v: f"{v:.1f}")
    fmt['Avg. Rank (High Noise)'] = fmt['Avg. Rank (High Noise)'].map(lambda v: f"{v:.1f}")
    fmt.to_markdown(md_path, index=False)

    # Also write the contenders table for summary-of-findings
    sof_csv_path = ROOT / 'gemini-analysis' / 'paper' / '02_summary_of_findings.csv'
    final.to_csv(sof_csv_path, index=False)

    print("Wrote:")
    print("  ", csv_path)
    print("  ", md_path)
    print("  ", sof_csv_path)


if __name__ == '__main__':
    main()
