#!/usr/bin/env python3
"""
Generate speed-accuracy tradeoff plot for derivative estimation methods.
Creates a log-log plot of nRMSE vs computational time for a specific configuration.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Configuration
NOISE_LEVEL = 0.0001  # 1e-4
DERIV_ORDER = 3
OUTPUT_DIR = Path('../build/figures/publication')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv('../build/results/comprehensive/comprehensive_summary.csv')

# Filter for specific configuration and average across ODE systems
filtered = df[(df['noise_level'] == NOISE_LEVEL) & (df['deriv_order'] == DERIV_ORDER)]
summary = filtered.groupby('method').agg({
    'mean_nrmse': 'mean',
    'mean_timing': 'mean'
}).reset_index()

# Method categories for coloring
categories = {
    'GP': ['GP-TaylorAD-Julia', 'GP-RBF-Python', 'GP-RBF-Iso-Python', 'GP-RBF-MeanSub-Python'],
    'Spectral': ['Fourier-GCV', 'Fourier-Basic-Python', 'Fourier-Continuation-Python', 'Fourier-Adaptive-Julia',
                 'Fourier-Adaptive-Python', 'Fourier-Continuation-Adaptive',
                 'Chebyshev-Basic-Python', 'Chebyshev-AICc'],
    'Spline': ['Spline-Dierckx-5', 'Spline-GSS'],
    'Filter': ['Savitzky-Golay', 'SavitzkyGolay-Fixed', 'SavitzkyGolay-Adaptive'],
    'Other': []
}

# Create reverse mapping
method_to_category = {}
for cat, methods in categories.items():
    for method in methods:
        method_to_category[method] = cat

# Assign categories
summary['category'] = summary['method'].map(lambda m: method_to_category.get(m, 'Other'))

# Define colors for categories
colors = {
    'GP': '#e74c3c',      # Red
    'Spectral': '#3498db', # Blue
    'Spline': '#2ecc71',   # Green
    'Filter': '#f39c12',   # Orange
    'Other': '#95a5a6'     # Gray
}

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each category
for cat in ['Filter', 'Spline', 'Spectral', 'GP', 'Other']:
    cat_data = summary[summary['category'] == cat]
    if not cat_data.empty:
        ax.scatter(cat_data['mean_timing'], cat_data['mean_nrmse'],
                  label=cat, color=colors[cat], s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

# Annotate key methods
key_methods = {
    'GP-TaylorAD-Julia': (-0.15, 0.15),
    'Savitzky-Golay': (0.1, -0.15),
    'Spline-Dierckx-5': (0.1, -0.15),
    'Fourier-GCV': (-0.15, 0.15),
    'Spline-GSS': (0.1, -0.15),
    'GP-RBF-Python': (0.1, 0.15)
}

for method, (dx, dy) in key_methods.items():
    method_data = summary[summary['method'] == method]
    if not method_data.empty:
        x = method_data['mean_timing'].values[0]
        y = method_data['mean_nrmse'].values[0]
        # Simplify method names for display
        display_name = method.replace('_', '-').replace('Savitzky-Golay', 'Savitzky-G.')
        ax.annotate(display_name, (x, y),
                   xytext=(x * (1 + dx), y * (1 + dy)),
                   fontsize=9, ha='center')

# Set log scales
ax.set_xscale('log')
ax.set_yscale('log')

# Labels and title
ax.set_xlabel('Computation Time (seconds)', fontsize=12)
ax.set_ylabel('Normalized RMSE', fontsize=12)
ax.set_title(f'Speed-Accuracy Tradeoff (Noise={NOISE_LEVEL}, Order={DERIV_ORDER})', fontsize=14)

# Grid
ax.grid(True, which='both', alpha=0.3, linestyle='--')
ax.grid(True, which='minor', alpha=0.1, linestyle=':')

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Add reference lines for different regimes
ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

# Add text labels for timing regimes
ax.text(0.003, ax.get_ylim()[1]*0.7, 'Fast\n(<10ms)', fontsize=9, ha='center', alpha=0.5)
ax.text(0.04, ax.get_ylim()[1]*0.7, 'Moderate\n(10-100ms)', fontsize=9, ha='center', alpha=0.5)
ax.text(0.4, ax.get_ylim()[1]*0.7, 'Slow\n(0.1-1s)', fontsize=9, ha='center', alpha=0.5)
ax.text(10, ax.get_ylim()[1]*0.7, 'Very Slow\n(>1s)', fontsize=9, ha='center', alpha=0.5)

# Add Pareto frontier (approximate)
# Sort by timing and identify frontier
sorted_data = summary.sort_values('mean_timing')
frontier_x = []
frontier_y = []
best_error = float('inf')

for _, row in sorted_data.iterrows():
    if row['mean_nrmse'] < best_error:
        frontier_x.append(row['mean_timing'])
        frontier_y.append(row['mean_nrmse'])
        best_error = row['mean_nrmse']

# Plot Pareto frontier
if frontier_x:
    ax.plot(frontier_x, frontier_y, 'k--', alpha=0.3, linewidth=2, label='Pareto Frontier')

# Adjust layout
plt.tight_layout()

# Save
output_file = OUTPUT_DIR / 'speed_accuracy_tradeoff.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_file}")

# Also save as PDF for paper
output_pdf = OUTPUT_DIR / 'speed_accuracy_tradeoff.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Saved PDF to {output_pdf}")

# Print summary statistics
print("\nKey findings:")
print(f"Fastest method: {sorted_data.iloc[0]['method']} ({sorted_data.iloc[0]['mean_timing']:.3f}s)")
print(f"Most accurate: {summary.loc[summary['mean_nrmse'].idxmin(), 'method']} (nRMSE={summary['mean_nrmse'].min():.4f})")
print(f"Speed range: {summary['mean_timing'].min():.3f}s to {summary['mean_timing'].max():.1f}s ({summary['mean_timing'].max()/summary['mean_timing'].min():.0f}x)")
print(f"Accuracy range: {summary['mean_nrmse'].min():.4f} to {summary['mean_nrmse'].max():.2f}")

plt.show()