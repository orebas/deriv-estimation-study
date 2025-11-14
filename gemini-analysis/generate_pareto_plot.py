#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Config
ROOT = Path('/home/orebas/tmp/deriv-estimation-study')
SUMMARY_CSV = ROOT / 'build/results/comprehensive/comprehensive_summary.csv'
OUT_DIR = ROOT / 'gemini-analysis/paper/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    'TVRegDiff-Python': 'TVRegDiff-Python',
    'Kalman-Gradient': 'Kalman-Gradient',
    'AAA-Adaptive-Diff2': 'AAA-Adaptive-Diff2',
    'AAA-Adaptive-Wavelet': 'AAA-Adaptive-Wavelet',
    'AAA-LowTol': 'AAA-LowTol',
}

def compute_pareto_front(points: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Compute Pareto front for minimization of both x and y."""
    pts = points.sort_values([x_col, y_col], ascending=[True, True]).reset_index(drop=True)
    front_idx = []
    best_y = float('inf')
    for idx, row in pts.iterrows():
        y = row[y_col]
        if y < best_y:
            front_idx.append(idx)
            best_y = y
    return pts.loc[front_idx].copy()

def main():
    df = pd.read_csv(SUMMARY_CSV)

    # Calculate overall stats for each method
    method_stats = df.groupby('method').agg(
        avg_nrmse=('mean_nrmse', 'mean'),
        avg_time=('mean_timing', 'mean'),
        category=('category', 'first') # Assumes one category per method
    ).reset_index()

    # Determine full coverage methods (orders 0-5)
    df_ord5 = df[df.deriv_order <= 5]
    LOW_NOISE = [1e-8, 1e-6]
    HIGH_NOISE = [0.01, 0.02]
    union_df = df_ord5[df_ord5.noise_level.isin(LOW_NOISE + HIGH_NOISE)]
    odes = sorted(union_df['ode_system'].unique())
    orders = sorted(union_df['deriv_order'].unique())
    noises = sorted(set(LOW_NOISE + HIGH_NOISE))
    expected_cells = len(odes) * len(orders) * len(noises)
    cells_per_method = union_df.groupby('method').size()
    full_coverage_methods = sorted(cells_per_method[cells_per_method == expected_cells].index.tolist())

    # Filter stats to only include our contenders
    contender_stats = method_stats[method_stats['method'].isin(full_coverage_methods)].copy()
    contender_stats['Method'] = contender_stats['method'].map(lambda m: METHOD_MAP.get(m, m))

    # Compute Pareto front
    pareto_front = compute_pareto_front(contender_stats, x_col='avg_time', y_col='avg_nrmse')

    # Generate Plot
    sns.set_context('talk')
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 8))

    ax = sns.scatterplot(
        data=contender_stats,
        x='avg_time', y='avg_nrmse',
        hue='category', style='category',
        s=120, alpha=0.9
    )

    ax.set_title('Speed vs. Accuracy Trade-off (Contenders, Orders 0-5)')
    ax.set_xlabel('Average Time (s) [log scale, lower is better]')
    ax.set_ylabel('Average nRMSE [lower is better]')
    ax.set_xscale('log')

    # Annotate points
    for i, row in contender_stats.iterrows():
        ax.text(row['avg_time'] * 1.1, row['avg_nrmse'], row['Method'], fontsize=9)


    # Overlay Pareto front
    front_sorted = pareto_front.sort_values('avg_time')
    ax.plot(front_sorted['avg_time'], front_sorted['avg_nrmse'], '-o', color='black', linewidth=2, label='Pareto Front')

    ax.legend(title='Method Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    out_path = OUT_DIR / 'speed_accuracy_pareto.png'
    plt.savefig(out_path, dpi=300)
    print(f"Wrote Pareto plot to: {out_path}")


if __name__ == '__main__':
    main()
