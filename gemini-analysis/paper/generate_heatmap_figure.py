#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
NOISE_LEVEL = 0.01  # Medium noise
CONTENDERS = [
    'GP-Julia-AD',
    'GP-RBF-Python',
    'Savitzky-Golay',
    'Dierckx-5',
    'Fourier-GCV',
    'Fourier-Interp',
    'FFT-Adaptive-Julia',
    'GSS',
]
# Use absolute paths for robustness
ROOT_DIR = Path('/home/orebas/tmp/deriv-estimation-study')
SUMMARY_CSV = ROOT_DIR / 'build/results/comprehensive/comprehensive_summary.csv'
OUTPUT_DIR = ROOT_DIR / 'gemini-analysis/paper/figures'
OUTPUT_FILENAME = OUTPUT_DIR / 'performance_heatmap.png'

# --- Plot Styling ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- Data Loading ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SUMMARY_CSV)

# --- Data Preparation ---
# Filter to the chosen noise level and contenders, average across ODE systems
heatmap_data = df[
    (df['noise_level'] == NOISE_LEVEL) &
    (df['method'].isin(CONTENDERS))
].copy()

# Pivot to create a matrix: methods vs. derivative orders
pivot = heatmap_data.groupby(['method', 'deriv_order'])['mean_nrmse'].mean().unstack()

# Sort methods by their average nRMSE across all orders for better readability
pivot['mean_nrmse_all_orders'] = pivot.mean(axis=1)
pivot = pivot.sort_values('mean_nrmse_all_orders').drop(columns='mean_nrmse_all_orders')

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="viridis_r",  # Viridis reversed: lower is better (brighter)
    linewidths=.5,
    cbar_kws={'label': 'Mean nRMSE'},
    ax=ax,
    vmin=0,
    vmax=2.0  # Cap the color scale at 2.0 for better visual contrast
)

ax.set_title(f'Method Performance (nRMSE) vs. Derivative Order (Noise Level: {NOISE_LEVEL*100:.0f}%)', fontsize=16)
ax.set_xlabel('Derivative Order', fontsize=12)
ax.set_ylabel('Method', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# --- Final Touches ---
fig.tight_layout(pad=2.0)
fig.savefig(OUTPUT_FILENAME, dpi=150)

print(f"Heatmap saved to {OUTPUT_FILENAME}")
