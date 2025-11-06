#!/usr/bin/env python3
"""
Generate qualitative comparison figures showing actual derivative estimates vs ground truth.
These figures provide visual evidence of method performance in different noise regimes.

Outputs:
- high_noise_fit_comparison.png: Shows GP-Julia-AD, Savitzky-Golay-Fixed, Dierckx-5, Fourier-GCV at 2% noise
- low_noise_fit_comparison.png: Shows GP-Julia-AD, Dierckx-5, Fourier-GCV at 1e-6 noise
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

# High-noise comparison
HIGH_NOISE_CONFIG = {
    'trial_id': 'lotka_volterra_noise2000000e-8_trial1',
    'derivative_order': 4,
    'methods': ['GP-Julia-AD', 'Savitzky-Golay-Fixed', 'Dierckx-5', 'Fourier-GCV'],
    'output_filename': 'high_noise_fit_comparison.png',
    'title_suffix': 'High Noise (2%)',
    'noise_label': '2% noise',
    'colors': {
        'GP-Julia-AD': '#2ca02c',      # Green
        'Savitzky-Golay-Fixed': '#ff7f0e',   # Orange
        'Dierckx-5': '#1f77b4',        # Blue
        'Fourier-GCV': '#d62728',      # Red
    },
    'linestyles': {
        'GP-Julia-AD': '-',
        'Savitzky-Golay-Fixed': '--',
        'Dierckx-5': '-.',
        'Fourier-GCV': ':',
    }
}

# Low-noise comparison
LOW_NOISE_CONFIG = {
    'trial_id': 'lorenz_noise100e-8_trial1',
    'derivative_order': 5,
    'methods': ['GP-Julia-AD', 'Fourier-GCV', 'Dierckx-5'],
    'output_filename': 'low_noise_fit_comparison.png',
    'title_suffix': 'Low Noise (1e-6)',
    'noise_label': '1e-6 noise',
    'colors': {
        'GP-Julia-AD': '#2ca02c',     # Green
        'Fourier-GCV': '#d62728',     # Red
        'Dierckx-5': '#1f77b4',       # Blue
    },
    'linestyles': {
        'GP-Julia-AD': '-',
        'Fourier-GCV': '--',
        'Dierckx-5': '-.',
    }
}

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')


def generate_comparison_figure(config, output_dir):
    """Generate a qualitative comparison figure for the given configuration."""

    trial_id = config['trial_id']
    derivative_order = config['derivative_order']
    methods = config['methods']
    output_filename = config['output_filename']
    title_suffix = config['title_suffix']
    noise_label = config['noise_label']
    colors = config['colors']
    linestyles = config['linestyles']

    # Load prediction data
    predictions_file = REPO_ROOT / 'build/results/comprehensive/predictions' / f'{trial_id}.json'

    try:
        with open(predictions_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Predictions file not found: {predictions_file}")
        print(f"Skipping {output_filename}")
        return False

    # Extract data
    times = data['times']
    ground_truth = data['ground_truth_derivatives']

    # Load the ACTUAL noisy input data from input files (single source of truth)
    input_file = REPO_ROOT / 'build/data/input' / f'{trial_id}.json'
    try:
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        noisy_y = input_data['y_noisy']
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_file}")
        print(f"Skipping {output_filename}")
        return False
    except KeyError:
        print(f"ERROR: 'y_noisy' not found in input file")
        return False

    # Get predictions for all methods
    predictions = {}
    for method in methods:
        if method not in data['methods']:
            print(f"WARNING: Method {method} not found, skipping")
            continue
        predictions[method] = data['methods'][method]['predictions']

    if not predictions:
        print(f"ERROR: No valid methods found for {trial_id}")
        return False

    # Create figure with 3 panels
    fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Extract system name for titles
    system_name = trial_id.split('_')[0].replace('_', ' ').title()

    # --- Panel 1: Function Fit (0th Derivative) ---
    ax = axs[0]
    ax.plot(times, ground_truth['0'], '-', color='#808080', label='Ground Truth',
            linewidth=2.5, zorder=5, alpha=0.7)

    # Show noisy data points as black dots for both high and low noise
    ax.plot(times, noisy_y, 'o', color='black', alpha=0.6, markersize=4,
            label=f'Noisy Input Data ({noise_label})', zorder=10)

    ax.set_title(f'Function Fit (0th Derivative) - {system_name} ({title_suffix})', fontsize=14)
    ax.set_ylabel('Function Value', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Derivative Estimate ---
    ax = axs[1]
    d_ord_str = str(derivative_order)
    ax.plot(times, ground_truth[d_ord_str], '-', color='#808080',
            label='Ground Truth Derivative',
            linewidth=2.5, zorder=10, alpha=0.7)

    for method in predictions.keys():
        ax.plot(times, predictions[method][d_ord_str],
                label=f'{method} Estimate',
                color=colors.get(method, 'blue'),
                linestyle=linestyles.get(method, '-'),
                linewidth=2,
                alpha=0.85)

    ax.set_title(f'Derivative Order {derivative_order} Estimate Comparison ({title_suffix})',
                 fontsize=14)
    ax.set_ylabel('Derivative Value', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.autoscale(enable=True, axis='y', tight=False)

    # --- Panel 3: Estimation Errors ---
    ax = axs[2]

    # Plot zero line for reference
    ax.axhline(y=0, color='#808080', linestyle='-', linewidth=2, alpha=0.7, zorder=5, label='Zero Error')

    for method in predictions.keys():
        error = np.array(predictions[method][d_ord_str]) - np.array(ground_truth[d_ord_str])
        ax.plot(times, error,
                label=f'{method} Error',
                color=colors.get(method, 'blue'),
                linestyle=linestyles.get(method, '-'),
                linewidth=2,
                alpha=0.85)

    ax.set_title(f'Derivative Order {derivative_order} Estimation Error ({title_suffix})',
                 fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Error (Estimate - Ground Truth)', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.autoscale(enable=True, axis='y', tight=False)

    # Save figure
    fig.tight_layout(pad=2.0)
    output_path = output_dir / output_filename
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Generated: {output_filename}")
    return True


def main():
    """Generate both qualitative comparison figures."""

    # Output directory
    output_dir = REPO_ROOT / 'build/figures/publication'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("GENERATING QUALITATIVE COMPARISON FIGURES")
    print("="*80)
    print()

    # Check if predictions exist
    predictions_dir = REPO_ROOT / 'build/results/comprehensive/predictions'
    if not predictions_dir.exists():
        print(f"ERROR: Predictions directory not found: {predictions_dir}")
        print("Run the comprehensive study first: ./scripts/02_run_comprehensive.sh")
        sys.exit(1)

    # Generate both figures
    success_count = 0

    print("Generating high-noise comparison...")
    if generate_comparison_figure(HIGH_NOISE_CONFIG, output_dir):
        success_count += 1

    print("\nGenerating low-noise comparison...")
    if generate_comparison_figure(LOW_NOISE_CONFIG, output_dir):
        success_count += 1

    print()
    print("="*80)
    print(f"QUALITATIVE COMPARISON GENERATION COMPLETE")
    print("="*80)
    print(f"Successfully generated {success_count}/2 figures")
    print(f"Output directory: {output_dir}")
    print("="*80)

    return 0 if success_count == 2 else 1


if __name__ == '__main__':
    sys.exit(main())
