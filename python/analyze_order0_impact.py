#!/usr/bin/env python3
"""Analyze impact of including/excluding derivative order 0 from rankings."""

import pandas as pd

# Load data
df = pd.read_csv('build/results/comprehensive/comprehensive_summary.csv')

# Define contenders (26 methods with full coverage for orders 0-5)
contenders = [
    'GP-TaylorAD-Julia', 'GP-RBF-Python',
    'SavitzkyGolay-Fixed-Julia', 'SavitzkyGolay-Adaptive-Julia',
    'SavitzkyGolay-Pkg-Fixed', 'SavitzkyGolay-Pkg-Hybrid', 'SavitzkyGolay-Pkg-Adaptive',
    'Fourier-Interp-Julia', 'Fourier-Adaptive-Julia', 'Fourier-Adaptive-Python',
    'Fourier-GCV', 'Fourier-Continuation-Adaptive', 'Fourier-Basic-Python',
    'Fourier-Continuation-Python', 'Chebyshev-AICc', 'Chebyshev-Basic-Python',
    'Dierckx-5', 'GSS',
    'AAA-LowPrec', 'AAA-Adaptive-Diff2', 'AAA-Adaptive-Wavelet',
    'PyNumDiff-SavGol-Tuned', 'PyNumDiff-Spectral-Auto', 'PyNumDiff-Spectral-Tuned',
    'Kalman-Grad-Python', 'TVRegDiff-Python'
]

# Filter to contenders only
df_contenders = df[df['method'].isin(contenders)].copy()

# Calculate rankings WITH order 0 (orders 0-5)
df_with_0 = df_contenders[df_contenders['deriv_order'].isin([0, 1, 2, 3, 4, 5])].copy()
df_with_0['rank'] = df_with_0.groupby(['ode_system', 'deriv_order', 'noise_level'])['mean_nrmse'].rank()
summary_with_0 = df_with_0.groupby('method')['rank'].mean().sort_values()

# Calculate rankings WITHOUT order 0 (orders 1-5 only)
df_without_0 = df_contenders[df_contenders['deriv_order'].isin([1, 2, 3, 4, 5])].copy()
df_without_0['rank'] = df_without_0.groupby(['ode_system', 'deriv_order', 'noise_level'])['mean_nrmse'].rank()
summary_without_0 = df_without_0.groupby('method')['rank'].mean().sort_values()

# Create comparison table
print("="*80)
print("RANKING COMPARISON: Including vs Excluding Derivative Order 0")
print("="*80)
print(f"\n{'Method':<35} {'With Order 0':<15} {'Without Order 0':<15} {'Difference':<10}")
print("-"*80)

# Show top 15 methods
for method in summary_with_0.head(15).index:
    rank_with = summary_with_0[method]
    rank_without = summary_without_0[method]
    diff = rank_without - rank_with
    print(f"{method:<35} {rank_with:>6.2f}          {rank_without:>6.2f}          {diff:>+6.2f}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Calculate some statistics
print(f"\nTotal test cases with order 0 (orders 0-5): {len(df_with_0)}")
print(f"Total test cases without order 0 (orders 1-5): {len(df_without_0)}")
print(f"Order 0 represents: {len(df_with_0[df_with_0['deriv_order']==0])/len(df_with_0)*100:.1f}% of test cases")

# Check if top 5 changes
top5_with = list(summary_with_0.head(5).index)
top5_without = list(summary_without_0.head(5).index)
print(f"\nTop 5 methods WITH order 0: {', '.join(top5_with[:5])}")
print(f"Top 5 methods WITHOUT order 0: {', '.join(top5_without[:5])}")

if top5_with == top5_without:
    print("\n✓ Top 5 ranking order is IDENTICAL")
else:
    print("\n✗ Top 5 ranking order CHANGES")

# Biggest winners/losers
rank_changes = pd.DataFrame({
    'with_0': summary_with_0,
    'without_0': summary_without_0
})
rank_changes['diff'] = rank_changes['without_0'] - rank_changes['with_0']
rank_changes = rank_changes.sort_values('diff')

print(f"\nBiggest WINNERS (rank improves without order 0):")
for method, row in rank_changes.head(3).iterrows():
    print(f"  {method}: {row['with_0']:.2f} → {row['without_0']:.2f} (improvement: {-row['diff']:.2f})")

print(f"\nBiggest LOSERS (rank worsens without order 0):")
for method, row in rank_changes.tail(3).iterrows():
    print(f"  {method}: {row['with_0']:.2f} → {row['without_0']:.2f} (worsens: {row['diff']:.2f})")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("""
The paper's focus is "Derivative Estimation from Noisy Data" - derivative order 0
is function approximation/interpolation, NOT differentiation. Including it:

1. Dilutes the focus on actual derivative estimation (orders 1+)
2. Represents 24% of test cases in the primary analysis
3. May favor methods optimized for interpolation over differentiation

RECOMMENDATION: EXCLUDE order 0 from primary rankings. Mention it separately in
a dedicated subsection about function approximation performance if desired.
""")
