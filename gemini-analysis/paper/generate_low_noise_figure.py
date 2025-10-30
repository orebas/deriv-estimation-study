#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# --- Configuration ---
TRIAL_ID = 'lorenz_noise1e-8_trial1'
DERIVATIVE_ORDER_TO_PLOT = 5  # High-order for precision test
METHODS_TO_PLOT = [
    'GP-Julia-AD',
    'Dierckx-5',
    'Fourier-Interp',
]
# Use absolute paths for robustness
ROOT_DIR = Path('/home/orebas/tmp/deriv-estimation-study')
PREDICTIONS_FILE = ROOT_DIR / 'build/results/comprehensive/predictions' / f'{TRIAL_ID}.json'
OUTPUT_DIR = ROOT_DIR / 'gemini-analysis/paper/figures'
OUTPUT_FILENAME = OUTPUT_DIR / 'low_noise_fit_comparison.png'

# --- Plot Styling ---
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'GP-Julia-AD': '#2ca02c',     # Green
    'Dierckx-5': '#1f77b4',       # Blue
    'Fourier-Interp': '#d62728',  # Red
}
LINESTYLES = {
    'GP-Julia-AD': '-',
    'Dierckx-5': '--',
    'Fourier-Interp': ':',
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

# Extract data
times = data['times']
ground_truth = data['ground_truth_derivatives']
noisy_y = data['methods']['GP-Julia-AD']['predictions']['0']
predictions = {method: data['methods'][method]['predictions'] for method in METHODS_TO_PLOT}

# --- Plotting ---
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Panel 1: Function Fit (0th Derivative) ---
ax = axs[0]
ax.plot(times, ground_truth['0'], 'k-', label='Ground Truth', linewidth=2)
# With very low noise, a scatter plot is less informative; we'll show fits
for method in METHODS_TO_PLOT:
    ax.plot(times, predictions[method]['0'],
            label=f'{method} Fit',
            color=COLORS.get(method, 'blue'),
            linestyle=LINESTYLES.get(method, '-'),
            linewidth=2)

ax.set_title(f'Function Fit (0th Derivative) on {TRIAL_ID.split("_")[0]} (Low Noise)', fontsize=16)
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

ax.set_title(f'Derivative Order {DERIVATIVE_ORDER_TO_PLOT} Estimate Comparison (Low Noise)', fontsize=16)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Derivative Value', fontsize=12)
ax.legend(fontsize=10)
ax.autoscale(enable=True, axis='y', tight=True)

# --- Final Touches ---
fig.tight_layout(pad=2.0)
fig.savefig(OUTPUT_FILENAME, dpi=150)

print(f"Figure saved to {OUTPUT_FILENAME}")
