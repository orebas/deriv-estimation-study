#!/usr/bin/env python3
"""
Generate method comparison plots showing true vs predicted values and errors.

For each derivative order at a specific noise level, creates:
1. True values + predictions from selected methods
2. Errors (predictions - true) for those methods

IMPORTANT: Reads ONLY from build/results/comprehensive/predictions/
(Single source of truth for visualization data)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configuration
NOISE_LEVEL = 0.01  # 1% noise
TRIAL = 1  # Use first trial for visualization

# User-requested methods
METHODS_TO_PLOT = [
    "GP-TaylorAD-Julia",
    "Fourier-GCV",
    "AAA-Adaptive-Wavelet",
    "Spline-Dierckx-5",
    "Spline-GSS"
]

# Derivative orders to process
ORDERS = list(range(8))  # 0-7

# Output directory
OUTPUT_DIR = Path("build/figures/method_comparisons")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Single source of truth for predictions
PREDICTIONS_DIR = Path("build/results/comprehensive/predictions")


def find_predictions_file(noise_level, trial):
    """Find the predictions JSON file for given noise level and trial."""
    # Convert noise level to trial_id format
    trial_id = f"noise{int(noise_level * 1e8)}e-8_trial{trial}"
    filepath = PREDICTIONS_DIR / f"{trial_id}.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {filepath}\n"
            f"Run ./scripts/02_run_comprehensive.sh first"
        )

    return filepath


def load_predictions(noise_level, trial):
    """Load predictions from JSON (single source of truth)."""
    filepath = find_predictions_file(noise_level, trial)

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def plot_comparison(order, noise_level, trial, methods, output_dir):
    """Generate comparison plots for a given derivative order."""

    # Load data from single source of truth
    data = load_predictions(noise_level, trial)

    # Extract common data
    x_eval = np.array(data['times'])
    # Handle null values in ground truth (shouldn't happen, but be safe)
    true_raw = data['ground_truth_derivatives'][str(order)]
    true_values = np.array([v if v is not None else np.nan for v in true_raw], dtype=float)

    # Check if we have valid true values
    if len(true_values) == 0 or np.all(np.isnan(true_values)):
        print(f"  ⚠️  Order {order}: No valid true values, skipping")
        return

    # Extract predictions for each method
    predictions = {}
    for method in methods:
        if method not in data['methods']:
            print(f"  ⚠️  Order {order}: Method '{method}' not found in results")
            continue

        method_data = data['methods'][method]

        if not method_data.get('success', False):
            print(f"  ⚠️  Order {order}: Method '{method}' failed")
            continue

        if str(order) not in method_data['predictions']:
            print(f"  ⚠️  Order {order}: No predictions for '{method}'")
            continue

        # Convert predictions, handling null values (None becomes nan)
        pred_raw = method_data['predictions'][str(order)]
        pred = np.array([v if v is not None else np.nan for v in pred_raw], dtype=float)

        # Check for valid predictions
        if len(pred) > 0 and not np.all(np.isnan(pred)):
            predictions[method] = pred
        else:
            print(f"  ⚠️  Order {order}: '{method}' has all NaN/null predictions")

    if not predictions:
        print(f"  ⚠️  Order {order}: No valid predictions from any method, skipping")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    # Plot 1: True values + predictions
    ax1.plot(x_eval, true_values, 'k-', linewidth=2.5, label='True', zorder=10)

    for i, (method, pred) in enumerate(predictions.items()):
        ax1.plot(x_eval, pred, '--', linewidth=1.5, color=colors[i],
                label=method, alpha=0.8)

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel(f'd^{order}y/dt^{order}', fontsize=12)
    ax1.set_title(f'Derivative Order {order} - Noise {noise_level*100:.1f}% - Predictions vs Truth',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Errors
    for i, (method, pred) in enumerate(predictions.items()):
        error = pred - true_values
        ax2.plot(x_eval, error, '-', linewidth=1.5, color=colors[i],
                label=method, alpha=0.8)

    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Error (predicted - true)', fontsize=12)
    ax2.set_title(f'Derivative Order {order} - Prediction Errors',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save both PNG and PDF
    base_filename = f"order{order}_noise{noise_level:.3f}_comparison"

    png_path = output_dir / f"{base_filename}.png"
    pdf_path = output_dir / f"{base_filename}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Order {order}: Saved {png_path.name} and {pdf_path.name}")


def main():
    print("=" * 80)
    print("GENERATING METHOD COMPARISON PLOTS")
    print("=" * 80)
    print(f"\nNoise level: {NOISE_LEVEL*100:.1f}%")
    print(f"Trial: {TRIAL}")
    print(f"Methods: {', '.join(METHODS_TO_PLOT)}")
    print(f"Orders: {min(ORDERS)}-{max(ORDERS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nData source: {PREDICTIONS_DIR}/ (single source of truth)")
    print()

    # Check if predictions directory exists
    if not PREDICTIONS_DIR.exists():
        print(f"ERROR: Predictions directory not found: {PREDICTIONS_DIR}")
        print("Run ./scripts/02_run_comprehensive.sh first to generate predictions.")
        sys.exit(1)

    for order in ORDERS:
        try:
            plot_comparison(order, NOISE_LEVEL, TRIAL, METHODS_TO_PLOT, OUTPUT_DIR)
        except FileNotFoundError as e:
            print(f"  ✗ Order {order}: {e}")
            continue
        except Exception as e:
            print(f"  ✗ Order {order}: Error - {e}")
            continue

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nGenerated plots are in: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
