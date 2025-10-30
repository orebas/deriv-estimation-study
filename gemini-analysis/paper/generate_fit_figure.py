import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# --- Configuration ---
TRIAL_ID = 'lotka_volterra_noise2000000e-8_trial1'
DERIVATIVE_ORDER_TO_PLOT = 4
METHODS_TO_PLOT = [
    'GP-Julia-AD',
    'GP_RBF_Python',
    'Savitzky-Golay',
]
# Use absolute paths for robustness
ROOT_DIR = Path('/home/orebas/tmp/deriv-estimation-study')
PREDICTIONS_FILE = ROOT_DIR / 'build/results/comprehensive/predictions' / f'{TRIAL_ID}.json'
OUTPUT_DIR = ROOT_DIR / 'gemini-analysis/paper/figures'
OUTPUT_FILENAME = OUTPUT_DIR / 'high_noise_fit_comparison.png'

# --- Plot Styling ---
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'GP-Julia-AD': '#2ca02c',      # Green
    'GP-RBF-Python': '#ff7f0e',   # Orange
    'Savitzky-Golay': '#9467bd',  # Purple
}
LINESTYLES = {
    'GP-Julia-AD': '-',
    'GP-RBF-Python': '--',
    'Savitzky-Golay': ':',
}

# --- Data Loading ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
try:
    with open(PREDICTIONS_FILE, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Predictions file not found at {PREDICTIONS_FILE}")
    print("Please ensure the comprehensive study has been run.")
    exit(1)

# Extract data into a more convenient format
times = data['times']
ground_truth = data['ground_truth_derivatives']
# The noisy data is the 0-th order derivative from any of the methods.
# We'll grab it from the first method that is guaranteed to be in the file.
noisy_y = data['methods']['GP-Julia-AD']['predictions']['0']
predictions = {method: data['methods'][method]['predictions'] for method in METHODS_TO_PLOT}

# --- Plotting ---
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Panel 1: Function Fit (0th Derivative) ---
ax = axs[0]
ax.plot(times, ground_truth['0'], 'k-', label='Ground Truth', linewidth=2)
# We plot the noisy data itself for context, not a "fit" for it
ax.plot(times, noisy_y, 'o', color='gray', alpha=0.5, markersize=3, label='Noisy Input Data (2% noise)')
ax.set_title(f'Function and Noisy Data for {TRIAL_ID.split("_")[0]}', fontsize=16)
ax.set_ylabel('Function Value', fontsize=12)
ax.legend(fontsize=10)

# --- Panel 2: Derivative Estimate ---
ax = axs[1]
d_ord_str = str(DERIVATIVE_ORDER_TO_PLOT)
ax.plot(times, ground_truth[d_ord_str], 'k-', label='Ground Truth Derivative', linewidth=2)

for method in METHODS_TO_PLOT:
    ax.plot(times, predictions[method][d_ord_str],
            label=f'{method} Estimate',
            color=COLORS.get(method, 'blue'),
            linestyle=LINESTYLES.get(method, '-'),
            linewidth=2.5)

ax.set_title(f'Derivative Order {DERIVATIVE_ORDER_TO_PLOT} Estimate Comparison (High Noise)', fontsize=16)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Derivative Value', fontsize=12)
ax.legend(fontsize=10)
# Auto-scaling y-axis, as our contenders should be well-behaved
ax.autoscale(enable=True, axis='y', tight=True)

# --- Final Touches ---
fig.tight_layout(pad=2.0)
fig.savefig(OUTPUT_FILENAME, dpi=150)

print(f"Figure saved to {OUTPUT_FILENAME}")
