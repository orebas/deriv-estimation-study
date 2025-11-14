#!/usr/bin/env python3
"""
Create speed-accuracy tradeoff plot from prepared data.
"""
import matplotlib.pyplot as plt
import numpy as np

# Read data
methods = []
times = []
errors = []

with open('/tmp/speed_accuracy_data.csv', 'r') as f:
    next(f)  # Skip header
    for line in f:
        if line.startswith('#'):
            break
        parts = line.strip().split(',')
        methods.append(parts[0])
        times.append(float(parts[1]))
        errors.append(float(parts[2]))

# Define categories and colors
categories = {
    'GP': {
        'methods': ['GP-TaylorAD-Julia', 'GP-RBF-Python', 'GP-RBF-Iso-Python', 'GP-RBF-MeanSub-Python'],
        'color': '#e74c3c',
        'marker': 'o'
    },
    'Spectral': {
        'methods': ['Fourier-GCV', 'Fourier-Basic-Python', 'Fourier-Continuation-Python', 'Fourier-Adaptive-Julia',
                   'Fourier-Adaptive-Python', 'Fourier-Continuation-Adaptive',
                   'Chebyshev-Basic-Python', 'Chebyshev-AICc', 'Fourier-Interp',
                   'PyNumDiff-Spectral-Auto', 'PyNumDiff-Spectral-Tuned'],
        'color': '#3498db',
        'marker': 's'
    },
    'Spline': {
        'methods': ['Spline-Dierckx-5', 'Spline-GSS'],
        'color': '#2ecc71',
        'marker': '^'
    },
    'Filter': {
        'methods': ['SavitzkyGolay-Fixed', 'SavitzkyGolay-Adaptive',
                   'SavitzkyGolay-Julia-Fixed', 'SavitzkyGolay-Julia-Adaptive', 'SavitzkyGolay-Julia-Hybrid',
                   'PyNumDiff-SavitzkyGolay-Auto', 'PyNumDiff-SavitzkyGolay-Tuned'],
        'color': '#f39c12',
        'marker': 'd'
    }
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each category
for cat_name, cat_info in categories.items():
    cat_times = []
    cat_errors = []
    for i, method in enumerate(methods):
        if method in cat_info['methods']:
            cat_times.append(times[i])
            cat_errors.append(errors[i])

    if cat_times:
        ax.scatter(cat_times, cat_errors,
                  label=cat_name,
                  color=cat_info['color'],
                  marker=cat_info['marker'],
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

# Plot remaining as "Other"
other_times = []
other_errors = []
all_categorized = []
for cat_info in categories.values():
    all_categorized.extend(cat_info['methods'])

for i, method in enumerate(methods):
    if method not in all_categorized:
        other_times.append(times[i])
        other_errors.append(errors[i])

if other_times:
    ax.scatter(other_times, other_errors,
              label='Other', color='#95a5a6', marker='x',
              s=60, alpha=0.5)

# Annotate key methods
annotations = {
    'GP-TaylorAD-Julia': (0.8, 1.2),
    'SavitzkyGolay-Adaptive': (1.3, 1.0),
    'Spline-Dierckx-5': (1.3, 0.8),
    'Fourier-GCV': (0.7, 1.2),
    'GP-RBF-Python': (1.3, 1.0)
}

for method_name, (x_mult, y_mult) in annotations.items():
    if method_name in methods:
        idx = methods.index(method_name)
        ax.annotate(method_name.replace('_', '-'),
                   (times[idx], errors[idx]),
                   xytext=(times[idx] * x_mult, errors[idx] * y_mult),
                   fontsize=9, ha='center')

# Set log scales
ax.set_xscale('log')
ax.set_yscale('log')

# Labels
ax.set_xlabel('Computation Time (seconds)', fontsize=12)
ax.set_ylabel('Normalized RMSE', fontsize=12)
ax.set_title('Speed-Accuracy Tradeoff (Noise=0.0001, Derivative Order=3)', fontsize=14)

# Grid
ax.grid(True, which='both', alpha=0.3, linestyle='--')
ax.grid(True, which='minor', alpha=0.1, linestyle=':')

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Add timing regime lines
for x_val in [0.01, 0.1, 1.0]:
    ax.axvline(x=x_val, color='gray', linestyle='--', alpha=0.3, linewidth=1)

# Add timing labels
ylims = ax.get_ylim()
y_pos = ylims[0] * (ylims[1]/ylims[0])**0.85  # 85% up in log space

ax.text(0.003, y_pos, 'Fast\n(<10ms)', fontsize=8, ha='center', alpha=0.5)
ax.text(0.04, y_pos, 'Moderate\n(10-100ms)', fontsize=8, ha='center', alpha=0.5)
ax.text(0.4, y_pos, 'Slow\n(0.1-1s)', fontsize=8, ha='center', alpha=0.5)
ax.text(10, y_pos, 'Very Slow\n(>1s)', fontsize=8, ha='center', alpha=0.5)

# Highlight Pareto frontier methods
pareto = [
    ('SavitzkyGolay-Fixed', 0.000664, 0.137993),
    ('SavitzkyGolay-Adaptive', 0.000837, 0.059860),
    ('GP-TaylorAD-Julia', 1.369867, 0.034486)
]

pareto_times = [p[1] for p in pareto]
pareto_errors = [p[2] for p in pareto]
ax.plot(pareto_times, pareto_errors, 'k--', alpha=0.4, linewidth=2, zorder=1)

# Adjust layout
plt.tight_layout()

# Save
plt.savefig('build/figures/publication/speed_accuracy_tradeoff.png', dpi=150, bbox_inches='tight')
plt.savefig('build/figures/publication/speed_accuracy_tradeoff.pdf', bbox_inches='tight')
print("Saved speed-accuracy plot")

plt.show()