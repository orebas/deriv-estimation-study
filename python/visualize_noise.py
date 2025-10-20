#!/usr/bin/env python3
"""
Visualize the noise in the data more clearly with zoomed-in view and residual plot.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
REPORT_DIR = Path(__file__).parent.parent / "report"
NOISE_LEVEL = 1e-2
TRIAL = 1

# Load data
trial_id = f"noise{int(NOISE_LEVEL*1e8)}e-8_trial{TRIAL}"
input_json = DATA_DIR / "input" / f"{trial_id}.json"

print(f"Loading data from {trial_id}...")
with open(input_json, 'r') as f:
    data = json.load(f)

times = np.array(data['times'])
y_true = np.array(data['y_true'])
y_noisy = np.array(data['y_noisy'])
noise = y_noisy - y_true

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Panel 1: Full view
ax1 = axes[0]
ax1.plot(times, y_true, 'k-', linewidth=2, label='Ground Truth', zorder=10)
ax1.scatter(times, y_noisy, s=30, alpha=0.6, color='red', label=f'Noisy Data ({NOISE_LEVEL*100}%)', zorder=5)
ax1.set_xlabel('Time (t)', fontsize=11)
ax1.set_ylabel('x(t)', fontsize=11)
ax1.set_title('Full View: Lotka-Volterra Observable', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Zoomed-in view (middle section)
ax2 = axes[1]
# Focus on t in [4, 6] where signal is interesting
zoom_mask = (times >= 4.0) & (times <= 6.0)
t_zoom = times[zoom_mask]
y_true_zoom = y_true[zoom_mask]
y_noisy_zoom = y_noisy[zoom_mask]

ax2.plot(t_zoom, y_true_zoom, 'k-', linewidth=2, label='Ground Truth', zorder=10)
ax2.scatter(t_zoom, y_noisy_zoom, s=50, alpha=0.7, color='red', label=f'Noisy Data', zorder=5)
# Add error bars to show noise magnitude
ax2.errorbar(t_zoom[::2], y_true_zoom[::2], yerr=noise.std(), fmt='none',
             ecolor='gray', alpha=0.3, capsize=3, label=f'±1 std ({noise.std():.4f})')
ax2.set_xlabel('Time (t)', fontsize=11)
ax2.set_ylabel('x(t)', fontsize=11)
ax2.set_title(f'Zoomed View (t ∈ [4, 6]): Noise std = {noise.std():.4f}', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Panel 3: Residual plot (noise only)
ax3 = axes[2]
ax3.scatter(times, noise, s=30, alpha=0.6, color='red', zorder=5)
ax3.axhline(0, color='k', linestyle='-', linewidth=1, zorder=10)
ax3.axhline(noise.std(), color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'+1 std ({noise.std():.4f})')
ax3.axhline(-noise.std(), color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'-1 std ({-noise.std():.4f})')
ax3.fill_between(times, -noise.std(), noise.std(), alpha=0.2, color='gray')
ax3.set_xlabel('Time (t)', fontsize=11)
ax3.set_ylabel('Noise = y_noisy - y_true', fontsize=11)
ax3.set_title(f'Noise Residuals: std = {noise.std():.6f}, max = {np.abs(noise).max():.6f}',
             fontsize=12, fontweight='bold')
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Save
output_file = REPORT_DIR / f"noise_visualization_{int(NOISE_LEVEL*100)}pct.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

output_png = REPORT_DIR / f"noise_visualization_{int(NOISE_LEVEL*100)}pct.png"
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"Saved: {output_png}")

print("\nNoise Statistics:")
print(f"  Standard deviation: {noise.std():.6f}")
print(f"  Max absolute value: {np.abs(noise).max():.6f}")
print(f"  As % of signal range: {100 * noise.std() / (y_true.max() - y_true.min()):.2f}%")
print(f"  SNR (signal/noise): {y_true.std() / noise.std():.1f}")
