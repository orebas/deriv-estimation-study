#!/usr/bin/env python3
"""
Deep dive into AAA performance at LOW noise
User hypothesis: AAA-HighPrec should excel at near-zero noise (it's an interpolator)
and degrade rapidly with noise.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("/home/orebas/derivative_estimation_study/results/comprehensive/comprehensive_summary.csv")

print("="*80)
print("AAA PERFORMANCE AT LOW NOISE - CRITICAL EVALUATION")
print("="*80)

aaa_hp = df[df['method'] == 'AAA-HighPrec'].copy()
aaa_lp = df[df['method'] == 'AAA-LowPrec'].copy()
gp_ad = df[df['method'] == 'GP-Julia-AD'].copy()

# Focus on lowest noise levels
low_noise_levels = [1e-8, 1e-6, 1e-4]

print("\n" + "="*80)
print("AAA-HighPrec at LOWEST noise levels")
print("="*80)

for noise in low_noise_levels:
    noise_data = aaa_hp[aaa_hp['noise_level'] == noise]
    print(f"\nNoise level: {noise:.0e}")
    print(f"  All orders mean: {noise_data['mean_nrmse'].mean():.6f}")
    print(f"  All orders median: {noise_data['mean_nrmse'].median():.6f}")
    print("\n  Per-order breakdown:")
    for order in sorted(noise_data['deriv_order'].unique()):
        order_val = noise_data[noise_data['deriv_order'] == order]['mean_nrmse'].values[0]
        print(f"    Order {order}: {order_val:.6e}")

print("\n" + "="*80)
print("COMPARISON: AAA-HighPrec vs GP-Julia-AD at LOWEST noise")
print("="*80)

for noise in low_noise_levels:
    aaa_noise = aaa_hp[aaa_hp['noise_level'] == noise]
    gp_noise = gp_ad[gp_ad['noise_level'] == noise]

    print(f"\nNoise level: {noise:.0e}")
    print(f"{'Order':<8} {'AAA-HighPrec':<15} {'GP-Julia-AD':<15} {'Winner':<15} {'AAA/GP Ratio':<15}")
    print("-" * 75)

    for order in sorted(aaa_noise['deriv_order'].unique()):
        aaa_val = aaa_noise[aaa_noise['deriv_order'] == order]['mean_nrmse'].values[0]
        gp_val = gp_noise[gp_noise['deriv_order'] == order]['mean_nrmse'].values[0]

        winner = "AAA" if aaa_val < gp_val else "GP-AD"
        ratio = aaa_val / gp_val if gp_val > 0 else np.inf

        print(f"{order:<8} {aaa_val:<15.6e} {gp_val:<15.6e} {winner:<15} {ratio:<15.2f}")

print("\n" + "="*80)
print("NOISE DEGRADATION PATTERN - AAA-HighPrec")
print("="*80)

# For each order, show how performance degrades with noise
for order in sorted(aaa_hp['deriv_order'].unique()):
    order_data = aaa_hp[aaa_hp['deriv_order'] == order].sort_values('noise_level')
    print(f"\nOrder {order}:")
    print(f"{'Noise':<12} {'nRMSE':<15} {'Degradation Factor':<20}")
    print("-" * 50)

    baseline = None
    for _, row in order_data.iterrows():
        nrmse = row['mean_nrmse']
        if baseline is None:
            baseline = nrmse
            deg_factor = 1.0
        else:
            deg_factor = nrmse / baseline if baseline > 0 else np.inf

        print(f"{row['noise_level']:<12.0e} {nrmse:<15.6e} {deg_factor:<20.2f}x")

print("\n" + "="*80)
print("RECOMMENDATION ANALYSIS")
print("="*80)

# At which noise levels and orders is AAA-HighPrec competitive?
print("\nWhere AAA-HighPrec beats GP-Julia-AD:")
wins = []
for _, aaa_row in aaa_hp.iterrows():
    order = aaa_row['deriv_order']
    noise = aaa_row['noise_level']
    aaa_nrmse = aaa_row['mean_nrmse']

    gp_row = gp_ad[(gp_ad['deriv_order'] == order) & (gp_ad['noise_level'] == noise)]
    if len(gp_row) > 0:
        gp_nrmse = gp_row['mean_nrmse'].values[0]
        if aaa_nrmse < gp_nrmse:
            wins.append({
                'order': order,
                'noise': noise,
                'aaa_nrmse': aaa_nrmse,
                'gp_nrmse': gp_nrmse,
                'advantage': gp_nrmse / aaa_nrmse
            })

if wins:
    wins_df = pd.DataFrame(wins)
    print(f"\nAAA-HighPrec wins in {len(wins)} configurations:")
    print(wins_df.to_string(index=False))
else:
    print("\nAAA-HighPrec does NOT beat GP-Julia-AD at any configuration.")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

# Calculate at what noise threshold AAA becomes problematic
print("\nFor each order, at what noise level does AAA-HighPrec cross nRMSE > 1.0?")
for order in sorted(aaa_hp['deriv_order'].unique()):
    order_data = aaa_hp[aaa_hp['deriv_order'] == order].sort_values('noise_level')

    # Find first noise level where nRMSE > 1.0
    bad_rows = order_data[order_data['mean_nrmse'] > 1.0]
    if len(bad_rows) > 0:
        threshold = bad_rows.iloc[0]['noise_level']
        threshold_nrmse = bad_rows.iloc[0]['mean_nrmse']
        print(f"  Order {order}: noise > {threshold:.0e} (nRMSE = {threshold_nrmse:.2f})")
    else:
        print(f"  Order {order}: NEVER (always good)")

print("\n" + "="*80)
print("USER HYPOTHESIS EVALUATION")
print("="*80)
print("User expected:")
print("  1. AAA-HighPrec should do well with extremely small noise")
print("  2. Should degrade very fast with noise (as it's an interpolator)")
print("  3. Should be useful for near-zero noise")
print("  4. Should recommend for noiseless case")
print("\nLet's evaluate each claim...")
