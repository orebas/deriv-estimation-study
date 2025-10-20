#!/usr/bin/env python3
"""
Generate remaining figures: Qualitative comparison and Noise sensitivity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configure matplotlib
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Paths
DATA_DIR = Path('../data')
FIG_DIR = Path('paper_figures/publication')
FIG_DIR.mkdir(parents=True, exist_ok=True)

#==============================================================================
# FIGURE 3: QUALITATIVE COMPARISON - Actual derivative estimates
#==============================================================================
def generate_figure3_qualitative():
    """Show actual derivative estimates for Order 4, Noise=2%"""
    print("\nGenerating Figure 3: Qualitative Comparison...")

    # Find a trial with noise = 0.02
    input_files = list((DATA_DIR / 'input').glob('noise2000000e-8*.json'))

    if len(input_files) == 0:
        print("  Warning: No files found for noise=2%, trying noise=5e-2...")
        input_files = list((DATA_DIR / 'input').glob('noise5000000*.json'))

    if len(input_files) == 0:
        print("  Error: No suitable input files found")
        return

    # Load first matching file
    with open(input_files[0], 'r') as f:
        input_data = json.load(f)

    # Load corresponding output
    output_file = input_files[0].name.replace('.json', '_results.json')
    output_path = DATA_DIR / 'output' / output_file

    if not output_path.exists():
        print(f"  Warning: Output file not found: {output_path}")
        return

    with open(output_path, 'r') as f:
        output_data = json.load(f)

    times = np.array(input_data['times'])
    y_true_order4 = np.array(input_data['ground_truth_derivatives']['4'])

    # Select methods to compare
    methods_to_plot = ['GP-Julia-AD', 'AAA-HighPrec', 'Fourier-Interp', 'Central-FD']

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, method_name in enumerate(methods_to_plot):
        ax = axes[idx]

        # Find method in output
        method_results = output_data.get('results', {}).get(method_name, {})
        if not method_results:
            ax.text(0.5, 0.5, f'{method_name}\nNo data', ha='center', va='center')
            ax.set_title(method_name)
            continue

        # Get order 4 prediction
        predictions = method_results.get('predictions', {})
        if '4' in predictions:
            y_pred = np.array(predictions['4'])

            # Compute nRMSE (excluding endpoints)
            valid_idx = slice(1, -1)
            rmse = np.sqrt(np.mean((y_pred[valid_idx] - y_true_order4[valid_idx])**2))
            nrmse = rmse / np.std(y_true_order4[valid_idx])

            # Plot
            ax.plot(times, y_true_order4, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
            ax.plot(times, y_pred, '--', linewidth=1.5, label=f'{method_name}\nnRMSE={nrmse:.3f}')
            ax.set_xlabel('Time')
            ax.set_ylabel('$d^4x/dt^4$')
            ax.set_title(f'{method_name}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{method_name}\nOrder 4 not available', ha='center', va='center')
            ax.set_title(method_name)

    noise_level = input_data['config']['noise_level']
    fig.suptitle(f'Figure 3: Qualitative Comparison - 4th Derivative at {noise_level:.1%} Noise', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIG_DIR / 'figure3_qualitative.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figure3_qualitative.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'figure3_qualitative.png'}")
    plt.close()

#==============================================================================
# FIGURE 5: NOISE SENSITIVITY - Selected orders
#==============================================================================
def generate_figure5_noise_sensitivity():
    """Noise sensitivity curves for selected derivative orders"""
    print("\nGenerating Figure 5: Noise Sensitivity...")

    # Load summary data
    summary = pd.read_csv(Path('../results/comprehensive/comprehensive_summary.csv'))

    # Exclude failed methods
    EXCLUDED = ['GP-Julia-SE', 'TVRegDiff_Python', 'SavitzkyGolay_Python']
    summary = summary[~summary['method'].isin(EXCLUDED)]

    # Select top 5 methods overall
    top_methods = (summary.groupby('method')['mean_nrmse'].mean()
                   .sort_values().head(5).index.tolist())

    # Select representative orders: 0, 2, 4, 7
    orders_to_plot = [0, 2, 4, 7]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, order in enumerate(orders_to_plot):
        ax = axes[idx]

        order_data = summary[summary['deriv_order'] == order]

        for method in top_methods:
            method_data = order_data[order_data['method'] == method].sort_values('noise_level')

            if len(method_data) > 0:
                ax.errorbar(method_data['noise_level'], method_data['mean_nrmse'],
                           yerr=method_data['std_nrmse'], label=method,
                           marker='o', markersize=5, capsize=3, alpha=0.8, linewidth=1.5)

        ax.set_xscale('log')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('nRMSE')
        ax.set_title(f'Derivative Order {order}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)  # Cap for readability

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Figure 5: Noise Sensitivity Across Derivative Orders\n(Top 5 methods, mean ± std)', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIG_DIR / 'figure5_noise_sensitivity.png', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'figure5_noise_sensitivity.pdf', bbox_inches='tight')
    print(f"  Saved: {FIG_DIR / 'figure5_noise_sensitivity.png'}")
    plt.close()

#==============================================================================
# MAIN
#==============================================================================
if __name__ == '__main__':
    print("="*80)
    print("GENERATING REMAINING FIGURES")
    print("="*80)

    generate_figure3_qualitative()
    generate_figure5_noise_sensitivity()

    print("\n" + "="*80)
    print("✓ REMAINING FIGURES GENERATED")
    print(f"Output directory: {FIG_DIR.absolute()}")
    print("="*80)
