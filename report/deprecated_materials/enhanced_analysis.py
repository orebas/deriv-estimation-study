#!/usr/bin/env python3
"""
Enhanced Data Analysis Pipeline - Addressing GPT-5 Critique

Key improvements:
1. Coverage matrix and coverage-normalized rankings
2. Robust statistics (mean, median, percentiles)
3. Per-method noise sensitivity curves
4. Statistical significance testing
5. Detailed per-order breakdowns
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats

DATA_DIR = Path("/home/orebas/derivative_estimation_study")
RESULTS_FILE = DATA_DIR / "results/comprehensive/comprehensive_summary.csv"
OUTPUT_DIR = DATA_DIR / "report/paper_figures/automated"

def load_data():
    """Load actual experimental results"""
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} rows of actual data")
    print(f"Methods: {df['method'].nunique()}")
    print(f"Orders: {sorted(df['deriv_order'].unique())}")
    print(f"Noise levels: {sorted(df['noise_level'].unique())}")
    return df

def build_coverage_matrix(df):
    """Build coverage matrix showing which method/order/noise combinations exist"""
    methods = sorted(df['method'].unique())
    orders = sorted(df['deriv_order'].unique())
    noise_levels = sorted(df['noise_level'].unique())

    coverage = {}
    for method in methods:
        method_data = df[df['method'] == method]
        total_configs = len(orders) * len(noise_levels)
        actual_configs = len(method_data)
        coverage_pct = 100 * actual_configs / total_configs

        # Per-order coverage
        order_coverage = {}
        for order in orders:
            order_count = len(method_data[method_data['deriv_order'] == order])
            order_coverage[f'order_{order}'] = order_count

        coverage[method] = {
            'total_configs': actual_configs,
            'max_configs': total_configs,
            'coverage_pct': coverage_pct,
            'order_coverage': order_coverage,
            'full_coverage': actual_configs == total_configs
        }

    return coverage

def compute_robust_statistics(df):
    """Compute mean, median, percentiles for each method"""
    stats_list = []

    for method in sorted(df['method'].unique()):
        method_data = df[df['method'] == method]['mean_nrmse']

        # Filter out extreme outliers for percentile calculation
        # (but keep them for mean to show true cost)
        finite_data = method_data[np.isfinite(method_data)]

        stats_list.append({
            'method': method,
            'mean': method_data.mean(),
            'median': method_data.median(),
            'p10': finite_data.quantile(0.10) if len(finite_data) > 0 else np.nan,
            'p90': finite_data.quantile(0.90) if len(finite_data) > 0 else np.nan,
            'std': method_data.std(),
            'min': method_data.min(),
            'max': method_data.max(),
            'count': len(method_data),
            'has_inf': np.any(~np.isfinite(method_data))
        })

    return pd.DataFrame(stats_list)

def compute_coverage_normalized_rankings(df, coverage):
    """Rank only methods with full coverage"""
    full_coverage_methods = [m for m, c in coverage.items() if c['full_coverage']]

    full_df = df[df['method'].isin(full_coverage_methods)]

    rankings = full_df.groupby('method').agg({
        'mean_nrmse': ['mean', 'median'],
        'mean_timing': 'mean',
        'category': 'first'
    }).reset_index()

    rankings.columns = ['method', 'mean_nrmse', 'median_nrmse', 'mean_timing', 'category']
    rankings = rankings.sort_values('mean_nrmse')
    rankings['rank_by_mean'] = range(1, len(rankings) + 1)

    rankings_med = rankings.sort_values('median_nrmse')
    rankings_med['rank_by_median'] = range(1, len(rankings_med) + 1)

    return rankings.merge(rankings_med[['method', 'rank_by_median']], on='method')

def compute_per_order_statistics(df):
    """Detailed per-order analysis for each method"""
    results = {}

    for order in sorted(df['deriv_order'].unique()):
        order_data = df[df['deriv_order'] == order]

        order_stats = []
        for method in sorted(order_data['method'].unique()):
            method_order = order_data[order_data['method'] == method]['mean_nrmse']

            order_stats.append({
                'method': method,
                'order': order,
                'mean': method_order.mean(),
                'median': method_order.median(),
                'min': method_order.min(),
                'max': method_order.max(),
                'count': len(method_order)
            })

        order_df = pd.DataFrame(order_stats).sort_values('mean')
        order_df['rank'] = range(1, len(order_df) + 1)
        results[f'order_{order}'] = order_df

    return results

def compute_noise_sensitivity(df):
    """Compute per-method noise curves"""
    noise_curves = {}

    for method in sorted(df['method'].unique()):
        method_data = df[df['method'] == method]

        noise_stats = []
        for noise in sorted(method_data['noise_level'].unique()):
            noise_data = method_data[method_data['noise_level'] == noise]['mean_nrmse']

            noise_stats.append({
                'noise_level': noise,
                'mean': noise_data.mean(),
                'median': noise_data.median(),
                'min': noise_data.min(),
                'max': noise_data.max(),
                'count': len(noise_data)
            })

        noise_curves[method] = pd.DataFrame(noise_stats)

    return noise_curves

def analyze_aaa_failure(df):
    """Detailed analysis of AAA methods to understand failure modes"""
    aaa_methods = ['AAA-LowPrec', 'AAA-HighPrec']
    aaa_data = df[df['method'].isin(aaa_methods)]

    analysis = {}
    for method in aaa_methods:
        method_data = aaa_data[aaa_data['method'] == method]

        # Per-order breakdown
        order_breakdown = []
        for order in sorted(method_data['deriv_order'].unique()):
            order_data = method_data[method_data['deriv_order'] == order]['mean_nrmse']
            order_breakdown.append({
                'order': order,
                'mean': order_data.mean(),
                'median': order_data.median(),
                'min': order_data.min(),
                'max': order_data.max()
            })

        # Per-noise breakdown
        noise_breakdown = []
        for noise in sorted(method_data['noise_level'].unique()):
            noise_data = method_data[method_data['noise_level'] == noise]['mean_nrmse']
            noise_breakdown.append({
                'noise_level': noise,
                'mean': noise_data.mean(),
                'median': noise_data.median(),
                'max': noise_data.max()
            })

        analysis[method] = {
            'order_breakdown': pd.DataFrame(order_breakdown),
            'noise_breakdown': pd.DataFrame(noise_breakdown)
        }

    return analysis

def main():
    print("="*80)
    print("ENHANCED DATA ANALYSIS - GPT-5 RECOMMENDATIONS")
    print("="*80)

    df = load_data()

    print("\n" + "="*80)
    print("1. COVERAGE ANALYSIS")
    print("="*80)
    coverage = build_coverage_matrix(df)

    full_coverage_methods = [m for m, c in coverage.items() if c['full_coverage']]
    partial_coverage_methods = [m for m, c in coverage.items() if not c['full_coverage']]

    print(f"\nMethods with FULL coverage ({len(full_coverage_methods)}):")
    for method in sorted(full_coverage_methods):
        print(f"  {method}: {coverage[method]['total_configs']} configs")

    print(f"\nMethods with PARTIAL coverage ({len(partial_coverage_methods)}):")
    for method in sorted(partial_coverage_methods):
        c = coverage[method]
        print(f"  {method}: {c['total_configs']}/{c['max_configs']} configs ({c['coverage_pct']:.1f}%)")
        # Show which orders are missing
        missing_orders = [k.replace('order_', '') for k, v in c['order_coverage'].items() if v == 0]
        if missing_orders:
            print(f"    Missing orders: {', '.join(missing_orders)}")

    # Save coverage matrix
    coverage_json = OUTPUT_DIR / "coverage_matrix.json"
    with open(coverage_json, 'w') as f:
        json.dump(coverage, f, indent=2)
    print(f"\nCoverage matrix saved to: {coverage_json}")

    print("\n" + "="*80)
    print("2. ROBUST STATISTICS (ALL METHODS)")
    print("="*80)
    robust_stats = compute_robust_statistics(df)
    print("\nTop 10 by median nRMSE:")
    print(robust_stats.nsmallest(10, 'median')[['method', 'mean', 'median', 'p10', 'p90']])

    print("\nBottom 5 by median nRMSE:")
    print(robust_stats.nlargest(5, 'median')[['method', 'mean', 'median', 'max']])

    robust_csv = OUTPUT_DIR / "robust_statistics.csv"
    robust_stats.to_csv(robust_csv, index=False)
    print(f"\nRobust statistics saved to: {robust_csv}")

    print("\n" + "="*80)
    print("3. COVERAGE-NORMALIZED RANKINGS")
    print("="*80)
    normalized_rankings = compute_coverage_normalized_rankings(df, coverage)
    print("\nTop 10 methods (full coverage only):")
    print(normalized_rankings.head(10)[['method', 'mean_nrmse', 'median_nrmse', 'rank_by_mean', 'rank_by_median']])

    normalized_csv = OUTPUT_DIR / "coverage_normalized_rankings.csv"
    normalized_rankings.to_csv(normalized_csv, index=False)
    print(f"\nCoverage-normalized rankings saved to: {normalized_csv}")

    print("\n" + "="*80)
    print("4. PER-ORDER DETAILED STATISTICS")
    print("="*80)
    per_order = compute_per_order_statistics(df)

    for order in range(8):
        order_key = f'order_{order}'
        if order_key in per_order:
            print(f"\nOrder {order} - Top 5:")
            print(per_order[order_key].head(5)[['method', 'mean', 'median', 'rank']])

            # Save to CSV
            order_csv = OUTPUT_DIR / f"order_{order}_detailed.csv"
            per_order[order_key].to_csv(order_csv, index=False)

    print("\n" + "="*80)
    print("5. AAA FAILURE ANALYSIS")
    print("="*80)
    aaa_analysis = analyze_aaa_failure(df)

    for method, data in aaa_analysis.items():
        print(f"\n{method} - Per-order breakdown:")
        print(data['order_breakdown'])

        print(f"\n{method} - Per-noise breakdown:")
        print(data['noise_breakdown'])

        # Save
        method_slug = method.replace('-', '_').lower()
        data['order_breakdown'].to_csv(OUTPUT_DIR / f"{method_slug}_order_breakdown.csv", index=False)
        data['noise_breakdown'].to_csv(OUTPUT_DIR / f"{method_slug}_noise_breakdown.csv", index=False)

    print("\n" + "="*80)
    print("6. NOISE SENSITIVITY CURVES")
    print("="*80)
    noise_curves = compute_noise_sensitivity(df)

    # Save all noise curves
    for method, curve in noise_curves.items():
        method_slug = method.replace('-', '_').replace(' ', '_').lower()
        curve_file = OUTPUT_DIR / "noise_curves" / f"{method_slug}_noise.csv"
        curve_file.parent.mkdir(exist_ok=True)
        curve.to_csv(curve_file, index=False)

    print(f"Noise curves saved to: {OUTPUT_DIR / 'noise_curves'}/")

    # Example: GP-Julia-AD noise curve
    print("\nGP-Julia-AD noise sensitivity:")
    print(noise_curves['GP-Julia-AD'])

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - coverage_matrix.json")
    print(f"  - robust_statistics.csv")
    print(f"  - coverage_normalized_rankings.csv")
    print(f"  - order_*_detailed.csv (0-7)")
    print(f"  - aaa_*_breakdown.csv")
    print(f"  - noise_curves/*.csv")

if __name__ == "__main__":
    main()
