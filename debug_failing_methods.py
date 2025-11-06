"""
Deep investigation: Why are these PyNumDiff methods failing so badly?
Let's debug RBF, Kalman, Spectral, and window-based methods
"""

import numpy as np
import matplotlib.pyplot as plt
from pynumdiff.basis_fit import rbfdiff, spectraldiff
from pynumdiff import kalman_smooth
from pynumdiff import smooth_finite_difference as sfd

# Generate test signal
n = 301
t = np.linspace(0.1, 3, n)
dt = np.mean(np.diff(t))

# Function: x^(5/2) + sin(2x)
y_true = t**(5/2) + np.sin(2*t)
dy_true = (5/2) * t**(3/2) + 2*np.cos(2*t)

# Add noise
np.random.seed(42)
noise_level = 1e-3
y_noisy = y_true + noise_level * np.random.randn(n)

print("=" * 80)
print("DEBUGGING CATASTROPHICALLY FAILING METHODS")
print("=" * 80)

# Create figure for investigation
fig, axes = plt.subplots(4, 4, figsize=(20, 16))

# ========== 1. RBF Investigation ==========
print("\n1. RBF METHOD INVESTIGATION:")
print("-" * 40)

# Try different RBF parameters
rbf_params = [
    {'sigma': 0.01, 'lmbd': 1e-6},   # Very narrow, low regularization
    {'sigma': 0.1, 'lmbd': 1e-4},    # Original
    {'sigma': 0.5, 'lmbd': 1e-3},    # Wider, more regularization
    {'sigma': 1.0, 'lmbd': 1e-2},    # Very wide, high regularization
]

for idx, params in enumerate(rbf_params):
    ax = axes[0, idx]
    try:
        x_smooth, dx_dt = rbfdiff(t, y_noisy, **params)
        dx_dt = np.asarray(dx_dt).flatten()

        rmse = np.sqrt(np.mean((dx_dt - dy_true)**2))

        ax.plot(t, dy_true, 'k-', linewidth=2, alpha=0.5, label='True')
        ax.plot(t, dx_dt, 'b-', linewidth=1.5, label=f'RBF (RMSE={rmse:.3f})')
        ax.set_title(f'RBF: σ={params["sigma"]}, λ={params["lmbd"]}')
        ax.set_ylim([dy_true.min() - 2, dy_true.max() + 2])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"  σ={params['sigma']:4}, λ={params['lmbd']:8} → RMSE={rmse:.4f}")
    except Exception as e:
        print(f"  σ={params['sigma']:4}, λ={params['lmbd']:8} → FAILED: {str(e)[:30]}")

# ========== 2. KALMAN Investigation ==========
print("\n2. KALMAN FILTER INVESTIGATION:")
print("-" * 40)

# Try different Kalman parameters (r=measurement noise, q=process noise)
kalman_params = [
    {'r': 1e-5, 'q': 1e-8},   # Trust measurements more
    {'r': 1e-3, 'q': 1e-6},   # Original
    {'r': 1e-2, 'q': 1e-4},   # Balanced
    {'r': 1e-1, 'q': 1e-2},   # Trust process model more
]

for idx, params in enumerate(kalman_params):
    ax = axes[1, idx]
    try:
        x_smooth, dx_dt = kalman_smooth.constant_velocity(
            y_noisy, dt, **params, forwardbackward='forward-backward'
        )
        dx_dt = np.asarray(dx_dt).flatten()

        rmse = np.sqrt(np.mean((dx_dt - dy_true)**2))

        ax.plot(t, dy_true, 'k-', linewidth=2, alpha=0.5, label='True')
        ax.plot(t, dx_dt, 'r-', linewidth=1.5, label=f'Kalman (RMSE={rmse:.3f})')
        ax.set_title(f'Kalman CV: r={params["r"]}, q={params["q"]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"  r={params['r']:8}, q={params['q']:8} → RMSE={rmse:.4f}")

        # Also show what the Kalman thinks the signal looks like
        ax2 = ax.twinx()
        ax2.plot(t, x_smooth, 'g--', alpha=0.3, linewidth=0.8)
        ax2.set_ylabel('Smoothed signal', color='g', fontsize=8)

    except Exception as e:
        print(f"  r={params['r']:8}, q={params['q']:8} → FAILED: {str(e)[:30]}")

# ========== 3. SPECTRAL Investigation ==========
print("\n3. SPECTRAL METHOD INVESTIGATION:")
print("-" * 40)

# Try different cutoff frequencies
spectral_params = [
    {'high_freq_cutoff': 0.05},   # Very aggressive filtering
    {'high_freq_cutoff': 0.15},   # Original
    {'high_freq_cutoff': 0.3},    # Moderate filtering
    {'high_freq_cutoff': 0.5},    # Minimal filtering
]

for idx, params in enumerate(spectral_params):
    ax = axes[2, idx]
    try:
        x_smooth, dx_dt = spectraldiff(y_noisy, dt, **params)
        dx_dt = np.asarray(dx_dt).flatten()

        rmse = np.sqrt(np.mean((dx_dt - dy_true)**2))

        ax.plot(t, dy_true, 'k-', linewidth=2, alpha=0.5, label='True')
        ax.plot(t, dx_dt, 'g-', linewidth=1.5, label=f'Spectral (RMSE={rmse:.3f})')
        ax.set_title(f'Spectral: cutoff={params["high_freq_cutoff"]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"  cutoff={params['high_freq_cutoff']:4} → RMSE={rmse:.4f}")
    except Exception as e:
        print(f"  cutoff={params['high_freq_cutoff']:4} → FAILED: {str(e)[:30]}")

# ========== 4. MEDIAN/MEAN DIFF Investigation ==========
print("\n4. WINDOW-BASED METHODS (MEDIAN/MEAN):")
print("-" * 40)

# Try different window sizes
window_sizes = [3, 7, 15, 31]

for idx, window_size in enumerate(window_sizes):
    ax = axes[3, idx]

    # Test median diff
    try:
        x_smooth_med, dx_dt_med = sfd.mediandiff(y_noisy, dt,
                                                  window_size=window_size,
                                                  num_iterations=3)
        dx_dt_med = np.asarray(dx_dt_med).flatten()
        rmse_med = np.sqrt(np.mean((dx_dt_med - dy_true)**2))
    except:
        rmse_med = np.inf
        dx_dt_med = None

    # Test mean diff
    try:
        x_smooth_mean, dx_dt_mean = sfd.meandiff(y_noisy, dt,
                                                 window_size=window_size,
                                                 num_iterations=3)
        dx_dt_mean = np.asarray(dx_dt_mean).flatten()
        rmse_mean = np.sqrt(np.mean((dx_dt_mean - dy_true)**2))
    except:
        rmse_mean = np.inf
        dx_dt_mean = None

    ax.plot(t, dy_true, 'k-', linewidth=2, alpha=0.5, label='True')
    if dx_dt_med is not None:
        ax.plot(t, dx_dt_med, 'orange', linewidth=1.5,
                label=f'Median (RMSE={rmse_med:.3f})', alpha=0.7)
    if dx_dt_mean is not None:
        ax.plot(t, dx_dt_mean, 'purple', linewidth=1.5,
                label=f'Mean (RMSE={rmse_mean:.3f})', alpha=0.7)

    ax.set_title(f'Window size = {window_size}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    print(f"  Window={window_size:2} → Median RMSE={rmse_med:.4f}, Mean RMSE={rmse_mean:.4f}")

plt.suptitle('Debugging Failing Methods: Parameter Investigation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('debug_failing_methods.png', dpi=150, bbox_inches='tight')
print("\nSaved: debug_failing_methods.png")

# ========== DEEP DIVE: Why is RBF so bad? ==========
print("\n" + "=" * 80)
print("DEEP DIVE: RBF FAILURE ANALYSIS")
print("=" * 80)

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

# 1. Show RBF basis functions
ax = axes2[0, 0]
# Sample a few RBF centers
n_centers = 20
centers = np.linspace(t.min(), t.max(), n_centers)
sigma = 0.1

for i, center in enumerate(centers[::4]):  # Show every 4th for clarity
    rbf = np.exp(-(t - center)**2 / (2 * sigma**2))
    ax.plot(t, rbf, alpha=0.5, linewidth=1)

ax.set_title('RBF Basis Functions (σ=0.1)')
ax.set_xlabel('x')
ax.set_ylabel('RBF value')
ax.grid(True, alpha=0.3)

# 2. Show condition number vs parameters
ax = axes2[0, 1]
sigmas = np.logspace(-2, 0, 20)
condition_numbers = []

for sigma in sigmas:
    # Build RBF matrix
    n_sample = 50  # Smaller for speed
    t_sample = np.linspace(t.min(), t.max(), n_sample)
    dist_matrix = np.abs(t_sample[:, None] - t_sample[None, :])
    rbf_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))

    # Add small regularization to avoid singular matrix
    rbf_matrix += 1e-10 * np.eye(n_sample)
    cond = np.linalg.cond(rbf_matrix)
    condition_numbers.append(cond)

ax.semilogy(sigmas, condition_numbers, 'b-', linewidth=2)
ax.set_xlabel('σ (RBF width)')
ax.set_ylabel('Condition Number')
ax.set_title('RBF Matrix Conditioning')
ax.grid(True, alpha=0.3)
ax.axhline(1e10, color='r', linestyle='--', label='Ill-conditioned')
ax.legend()

# 3. Show what RBF thinks the derivative looks like
ax = axes2[0, 2]
# Use best parameters we found
x_smooth, dx_dt = rbfdiff(t, y_noisy, sigma=1.0, lmbd=1e-2)
ax.plot(t, y_noisy, '.', alpha=0.2, markersize=2, label='Noisy data')
ax.plot(t, x_smooth, 'b-', linewidth=2, label='RBF smoothed')
ax.plot(t, y_true, 'k--', linewidth=1, alpha=0.5, label='True signal')
ax.set_title('RBF Signal Reconstruction')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ========== Kalman Filter Analysis ==========
# 4. Show Kalman's model assumption
ax = axes2[1, 0]
ax.text(0.5, 0.8, 'Kalman Constant Velocity Model:', transform=ax.transAxes,
        fontsize=12, fontweight='bold', ha='center')
ax.text(0.1, 0.6, 'State: [position, velocity]', transform=ax.transAxes, fontsize=10)
ax.text(0.1, 0.5, 'Assumes: velocity = constant + noise', transform=ax.transAxes, fontsize=10)
ax.text(0.1, 0.4, 'Reality: velocity = (5/2)x^(3/2) + 2cos(2x)', transform=ax.transAxes, fontsize=10)
ax.text(0.1, 0.3, '→ Model mismatch!', transform=ax.transAxes, fontsize=10, color='red')
ax.text(0.1, 0.2, 'The polynomial growth violates constant velocity', transform=ax.transAxes, fontsize=10)
ax.axis('off')

# 5. Show Kalman prediction vs reality
ax = axes2[1, 1]
x_smooth, dx_dt = kalman_smooth.constant_velocity(y_noisy, dt, r=1e-2, q=1e-4,
                                                  forwardbackward='forward-backward')
ax.plot(t[:50], dy_true[:50], 'k-', linewidth=2, label='True derivative')
ax.plot(t[:50], dx_dt[:50], 'r-', linewidth=2, label='Kalman estimate')
ax.set_title('Kalman Failure (First 50 points)')
ax.set_xlabel('x')
ax.set_ylabel('Derivative')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Window method boundary effects
ax = axes2[1, 2]
window_size = 11
x_smooth, dx_dt = sfd.meandiff(y_noisy, dt, window_size=window_size, num_iterations=1)

# Calculate local errors
errors = np.abs(dx_dt - dy_true)
ax.semilogy(t, errors, 'b-', linewidth=1.5, label=f'Mean diff (w={window_size})')

# Mark boundary regions
boundary_size = window_size // 2
ax.axvspan(t[0], t[boundary_size], color='red', alpha=0.2, label='Boundary region')
ax.axvspan(t[-boundary_size], t[-1], color='red', alpha=0.2)

ax.set_xlabel('x')
ax.set_ylabel('Absolute Error (log)')
ax.set_title('Window Method Boundary Effects')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('Root Cause Analysis: Why Methods Fail', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('failure_root_cause.png', dpi=150, bbox_inches='tight')
print("Saved: failure_root_cause.png")

print("\n" + "=" * 80)
print("ROOT CAUSES OF FAILURE:")
print("=" * 80)

print("\n1. RBF METHOD:")
print("   - Extremely sensitive to σ (width) and λ (regularization) parameters")
print("   - Matrix becomes ill-conditioned for small σ")
print("   - Cannot handle polynomial growth well")
print("   - Best parameters found: σ=1.0, λ=0.01 (still not great)")

print("\n2. KALMAN FILTERS:")
print("   - FUNDAMENTAL MODEL MISMATCH!")
print("   - Assumes constant velocity/acceleration")
print("   - Our function has x^(3/2) growth - violates the model")
print("   - Cannot track non-linear growth patterns")

print("\n3. SPECTRAL METHOD:")
print("   - Assumes periodic/band-limited signal")
print("   - x^(5/2) has unbounded growth - not periodic!")
print("   - FFT-based methods fail on non-periodic signals")

print("\n4. MEDIAN/MEAN DIFF:")
print("   - Severe boundary effects (window_size/2 points on each end)")
print("   - Window-based smoothing over-smooths the derivative")
print("   - Larger windows = more smoothing but worse boundaries")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("✅ For polynomial + oscillatory signals, use:")
print("   - Total Variation methods (handle both components well)")
print("   - Polynomial fitting (designed for this)")
print("   - Simple finite differences (surprisingly robust)")
print("\n❌ Avoid:")
print("   - RBF (too sensitive, conditioning issues)")
print("   - Kalman (wrong model for polynomial growth)")
print("   - Spectral (needs periodic signals)")
print("   - Window methods with small data (boundary effects)")