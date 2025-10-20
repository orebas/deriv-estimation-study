#!/usr/bin/env python3
"""
Investigate the Central-FD contradiction:
- How does it rank #1 overall (nRMSE=0.034)?
- Why is GP-AD best at every individual order?

This script analyzes the noise-level breakdown to understand the paradox.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("/home/orebas/derivative_estimation_study/results/comprehensive/comprehensive_summary.csv")

print("="*80)
print("INVESTIGATING CENTRAL-FD PARADOX")
print("="*80)

# Get Central-FD and GP-Julia-AD data
central_fd = df[df['method'] == 'Central-FD'].copy()
gp_ad = df[df['method'] == 'GP-Julia-AD'].copy()

print("\nCentral-FD performance by order:")
print("-" * 60)
for order in sorted(central_fd['deriv_order'].unique()):
    order_data = central_fd[central_fd['deriv_order'] == order]
    mean = order_data['mean_nrmse'].mean()
    min_val = order_data['mean_nrmse'].min()
    max_val = order_data['mean_nrmse'].max()
    print(f"Order {order}: mean={mean:8.4f} min={min_val:8.4f} max={max_val:10.2f}")

print("\nGP-Julia-AD performance by order:")
print("-" * 60)
for order in sorted(gp_ad['deriv_order'].unique()):
    order_data = gp_ad[gp_ad['deriv_order'] == order]
    mean = order_data['mean_nrmse'].mean()
    min_val = order_data['mean_nrmse'].min()
    max_val = order_data['mean_nrmse'].max()
    print(f"Order {order}: mean={mean:8.4f} min={min_val:8.4f} max={max_val:8.4f}")

print("\nDirect comparison (per-order mean nRMSE):")
print("-" * 60)
print(f"{'Order':<8} {'Central-FD':<12} {'GP-AD':<12} {'Winner':<15}")
print("-" * 60)
for order in sorted(central_fd['deriv_order'].unique()):
    cf_mean = central_fd[central_fd['deriv_order'] == order]['mean_nrmse'].mean()
    gp_mean = gp_ad[gp_ad['deriv_order'] == order]['mean_nrmse'].mean()
    winner = "Central-FD" if cf_mean < gp_mean else "GP-AD"
    print(f"{order:<8} {cf_mean:<12.4f} {gp_mean:<12.4f} {winner:<15}")

print("\nNoise-level breakdown for Central-FD:")
print("-" * 80)
for order in sorted(central_fd['deriv_order'].unique()):
    print(f"\nOrder {order}:")
    order_data = central_fd[central_fd['deriv_order'] == order].sort_values('noise_level')
    for _, row in order_data.iterrows():
        print(f"  Noise {row['noise_level']:.0e}: nRMSE={row['mean_nrmse']:10.4f}")

print("\n" + "="*80)
print("RESOLUTION:")
print("="*80)

# Calculate global means
cf_global = central_fd['mean_nrmse'].mean()
gp_global = gp_ad['mean_nrmse'].mean()

print(f"Central-FD global mean: {cf_global:.4f}")
print(f"GP-AD global mean: {gp_global:.4f}")
print(f"\nExplanation: The 'overall ranking' uses mean across ALL (method,order,noise)")
print(f"             The 'per-order best' uses mean across noise for each order.")
print(f"             Central-FD likely excels at low noise but fails at high noise.")
print(f"             GP-AD is more robust across the noise spectrum.")
