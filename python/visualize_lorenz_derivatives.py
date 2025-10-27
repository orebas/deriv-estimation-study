#!/usr/bin/env python3
"""
Visualize Lorenz system derivatives with GP-Julia-AD predictions.

Creates a comprehensive visualization showing:
1. Noisy observation data (x(t) from Lorenz system)
2. True derivatives vs GP-Julia-AD predictions for orders 0-7
3. Prediction errors for each order
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Configuration
ODE_SYSTEM = "lorenz"
NOISE_LEVEL = 0.01  # 1% noise
TRIAL = 1
METHOD_TO_VISUALIZE = "GP-Julia-AD"

# Output directory
OUTPUT_DIR = Path("build/figures/lorenz_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Single source of truth for predictions
PREDICTIONS_DIR = Path("build/results/comprehensive/predictions")


def load_predictions(ode_system, noise_level, trial):
    """Load predictions from JSON."""
    trial_id = f"{ode_system}_noise{int(noise_level * 1e8)}e-8_trial{trial}"
    filepath = PREDICTIONS_DIR / f"{trial_id}.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {filepath}\n"
            f"Available files: {list(PREDICTIONS_DIR.glob('*.json'))[:5]}"
        )

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data, trial_id


def clean_array(arr_raw):
    """Convert array with potential None/null values to numpy array with NaNs."""
    return np.array([v if v is not None else np.nan for v in arr_raw], dtype=float)


def plot_all_derivatives(data, method_name, trial_id, output_dir):
    """Create comprehensive plot showing all derivative orders."""

    # Extract common data
    times = np.array(data['times'])

    # Figure with 8 subplots (one per derivative order)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    # Process each derivative order
    for order in range(8):
        ax = axes[order]

        # Get ground truth
        true_raw = data['ground_truth_derivatives'][str(order)]
        true_vals = clean_array(true_raw)

        # Plot ground truth
        ax.plot(times, true_vals, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)

        # Get method predictions if available
        if method_name in data['methods']:
            method_data = data['methods'][method_name]

            if method_data.get('success', False) and str(order) in method_data['predictions']:
                pred_raw = method_data['predictions'][str(order)]
                pred_vals = clean_array(pred_raw)

                # Plot predictions
                ax.plot(times, pred_vals, 'r--', linewidth=1.5,
                       label=f'{method_name} Prediction', alpha=0.8)

                # Calculate and display RMSE (excluding endpoints)
                mask = np.isfinite(pred_vals) & np.isfinite(true_vals)
                if np.sum(mask) > 2:
                    # Exclude first and last points
                    inner_mask = mask.copy()
                    inner_mask[0] = False
                    inner_mask[-1] = False

                    if np.sum(inner_mask) > 0:
                        rmse = np.sqrt(np.mean((pred_vals[inner_mask] - true_vals[inner_mask])**2))
                        ax.text(0.02, 0.98, f'RMSE: {rmse:.2e}',
                               transform=ax.transAxes, fontsize=10,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                # Method failed
                ax.text(0.5, 0.5, f'{method_name}\nFAILED',
                       transform=ax.transAxes, fontsize=14, color='red',
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

        # Formatting
        ax.set_xlabel('Time (t)', fontsize=10)
        ax.set_ylabel(f'd^{order}x/dt^{order}', fontsize=11)
        ax.set_title(f'Derivative Order {order}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Lorenz System: {method_name} Derivative Estimation\n' +
                 f'Noise: {NOISE_LEVEL*100:.1f}%, Trial: {TRIAL}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    base_filename = f"lorenz_all_derivatives_{trial_id}"
    png_path = output_dir / f"{base_filename}.png"
    pdf_path = output_dir / f"{base_filename}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved all derivatives plot: {png_path.name}")


def plot_errors(data, method_name, trial_id, output_dir):
    """Create error plots showing prediction - truth for each order."""

    # Extract common data
    times = np.array(data['times'])

    # Figure with 8 subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    # Process each derivative order
    for order in range(8):
        ax = axes[order]

        # Get ground truth
        true_raw = data['ground_truth_derivatives'][str(order)]
        true_vals = clean_array(true_raw)

        # Get method predictions if available
        if method_name in data['methods']:
            method_data = data['methods'][method_name]

            if method_data.get('success', False) and str(order) in method_data['predictions']:
                pred_raw = method_data['predictions'][str(order)]
                pred_vals = clean_array(pred_raw)

                # Calculate error
                error = pred_vals - true_vals

                # Plot error
                ax.plot(times, error, 'r-', linewidth=1.5, alpha=0.7)
                ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

                # Calculate and display statistics
                mask = np.isfinite(error)
                if np.sum(mask) > 0:
                    mae = np.mean(np.abs(error[mask]))
                    max_error = np.max(np.abs(error[mask]))
                    ax.text(0.02, 0.98, f'MAE: {mae:.2e}\nMax: {max_error:.2e}',
                           transform=ax.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            else:
                # Method failed
                ax.text(0.5, 0.5, f'{method_name}\nFAILED',
                       transform=ax.transAxes, fontsize=14, color='red',
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

        # Formatting
        ax.set_xlabel('Time (t)', fontsize=10)
        ax.set_ylabel(f'Error (order {order})', fontsize=11)
        ax.set_title(f'Prediction Error - Order {order}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Lorenz System: {method_name} Prediction Errors\n' +
                 f'Noise: {NOISE_LEVEL*100:.1f}%, Trial: {TRIAL}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    base_filename = f"lorenz_errors_{trial_id}"
    png_path = output_dir / f"{base_filename}.png"
    pdf_path = output_dir / f"{base_filename}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved error plot: {png_path.name}")


def main():
    print("=" * 80)
    print("LORENZ SYSTEM DERIVATIVE VISUALIZATION")
    print("=" * 80)
    print(f"\nODE System: {ODE_SYSTEM}")
    print(f"Noise level: {NOISE_LEVEL*100:.1f}%")
    print(f"Trial: {TRIAL}")
    print(f"Method: {METHOD_TO_VISUALIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load data
    try:
        data, trial_id = load_predictions(ODE_SYSTEM, NOISE_LEVEL, TRIAL)
        print(f"✓ Loaded predictions from: {trial_id}.json")
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        sys.exit(1)

    # Generate plots
    print("\nGenerating plots...")
    plot_all_derivatives(data, METHOD_TO_VISUALIZE, trial_id, OUTPUT_DIR)
    plot_errors(data, METHOD_TO_VISUALIZE, trial_id, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
