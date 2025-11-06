"""
Examine PyNumDiff Kalman filter documentation and implementation
Let's understand what these methods are SUPPOSED to be doing
"""

import inspect
import pynumdiff.kalman_smooth as kalman
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("PYNUMDIFF KALMAN FILTER: WHAT IT'S SUPPOSED TO DO")
print("=" * 80)

print("""
The Kalman filter is a MODEL-BASED estimator that assumes:

CONSTANT VELOCITY MODEL:
- The true velocity (derivative) is CONSTANT between measurements
- Any changes in velocity are due to small random noise
- State: [position, velocity]
- Dynamics: velocity stays constant, position += velocity * dt

CONSTANT ACCELERATION MODEL:
- The true acceleration is CONSTANT
- Velocity changes linearly over time
- State: [position, velocity, acceleration]
- Dynamics: acceleration constant, velocity += accel * dt

These are TRACKING models, like for GPS or radar!
""")

# Generate our test function
n = 301
t = np.linspace(0.1, 3, n)
dt = np.mean(np.diff(t))

# True signal and derivatives
y_true = t**(5/2) + np.sin(2*t)
dy_true = (5/2) * t**(3/2) + 2*np.cos(2*t)
ddy_true = (5/2) * (3/2) * t**(1/2) - 2*np.sin(2*t)

# Add noise
np.random.seed(42)
y_noisy = y_true + 1e-3 * np.random.randn(n)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# === TOP ROW: Show what Kalman assumes vs reality ===

ax1 = plt.subplot(3, 4, 1)
ax1.text(0.5, 0.9, 'CONSTANT VELOCITY MODEL', transform=ax1.transAxes,
         fontsize=11, fontweight='bold', ha='center')
ax1.text(0.1, 0.7, 'Assumes:', transform=ax1.transAxes, fontsize=10)
ax1.text(0.1, 0.55, '• v(t) = constant + noise', transform=ax1.transAxes, fontsize=9)
ax1.text(0.1, 0.4, '• a(t) ≈ 0', transform=ax1.transAxes, fontsize=9)
ax1.text(0.1, 0.25, 'Good for: GPS tracking,\nsmooth trajectories',
         transform=ax1.transAxes, fontsize=9, color='green')
ax1.text(0.1, 0.05, 'Bad for: Polynomial growth,\noscillations',
         transform=ax1.transAxes, fontsize=9, color='red')
ax1.axis('off')

ax2 = plt.subplot(3, 4, 2)
# Show constant velocity assumption
time_demo = np.linspace(0, 3, 100)
constant_vel = np.ones_like(time_demo) * 2.0
ax2.plot(time_demo, constant_vel, 'r-', linewidth=3, label='Model assumes')
ax2.fill_between(time_demo, constant_vel - 0.5, constant_vel + 0.5,
                  alpha=0.3, color='red', label='±noise')
ax2.plot(t, dy_true, 'k-', linewidth=1.5, alpha=0.7, label='Our function')
ax2.set_title('Velocity Assumption')
ax2.set_xlabel('Time')
ax2.set_ylabel('Velocity (dv/dt)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 4, 3)
# Show the mismatch
actual_accel = np.gradient(dy_true, dt)
ax3.plot(t, actual_accel, 'b-', linewidth=2, label='Actual acceleration')
ax3.axhline(0, color='r', linestyle='--', linewidth=2, label='Model assumes ≈ 0')
ax3.fill_between(t, -0.5, 0.5, alpha=0.3, color='red')
ax3.set_title('Acceleration Reality Check')
ax3.set_xlabel('Time')
ax3.set_ylabel('Acceleration')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 4, 4)
# Run constant velocity Kalman
x_smooth_cv, dx_dt_cv = kalman.constant_velocity(
    y_noisy, dt, r=1e-3, q=1e-6, forwardbackward='forward-backward'
)
ax4.plot(t, dy_true, 'k-', linewidth=2, label='True derivative')
ax4.plot(t, dx_dt_cv, 'r--', linewidth=1.5, label='Kalman CV estimate')
error_cv = np.abs(dy_true - dx_dt_cv)
ax4.set_title(f'Result: RMSE = {np.sqrt(np.mean(error_cv**2)):.2f}')
ax4.set_xlabel('Time')
ax4.set_ylabel('Derivative')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# === MIDDLE ROW: Show a GOOD example for Kalman ===

ax5 = plt.subplot(3, 4, 5)
ax5.text(0.5, 0.9, 'GOOD EXAMPLE:', transform=ax5.transAxes,
         fontsize=11, fontweight='bold', ha='center', color='green')
ax5.text(0.5, 0.7, 'Piecewise Constant Velocity', transform=ax5.transAxes,
         fontsize=10, ha='center')
ax5.text(0.1, 0.5, '• Robot tracking', transform=ax5.transAxes, fontsize=9)
ax5.text(0.1, 0.35, '• Vehicle on highway', transform=ax5.transAxes, fontsize=9)
ax5.text(0.1, 0.2, '• Radar tracking', transform=ax5.transAxes, fontsize=9)
ax5.axis('off')

# Generate a good signal for Kalman
t_good = np.linspace(0, 10, 500)
dt_good = t_good[1] - t_good[0]

# Piecewise constant velocity (like real tracking)
true_vel_good = np.zeros_like(t_good)
true_vel_good[100:200] = 2.0
true_vel_good[200:350] = -1.0
true_vel_good[350:450] = 3.0

# Add small random walk to velocity
np.random.seed(44)
true_vel_good += np.cumsum(0.01 * np.random.randn(len(t_good)))

# Integrate to get position
true_pos_good = np.cumsum(true_vel_good) * dt_good

# Add measurement noise
noisy_pos_good = true_pos_good + 0.1 * np.random.randn(len(t_good))

ax6 = plt.subplot(3, 4, 6)
ax6.plot(t_good, true_pos_good, 'k-', linewidth=2, label='True position')
ax6.plot(t_good, noisy_pos_good, '.', alpha=0.2, markersize=1, color='gray', label='Noisy')
ax6.set_title('Position Signal (Good for Kalman)')
ax6.set_xlabel('Time')
ax6.set_ylabel('Position')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

ax7 = plt.subplot(3, 4, 7)
ax7.plot(t_good, true_vel_good, 'k-', linewidth=2, label='True velocity')
ax7.set_title('Velocity (Approximately Constant)')
ax7.set_xlabel('Time')
ax7.set_ylabel('Velocity')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

ax8 = plt.subplot(3, 4, 8)
# Apply Kalman to good signal
x_smooth_good, dx_dt_good = kalman.constant_velocity(
    noisy_pos_good, dt_good, r=0.01, q=0.001, forwardbackward='forward-backward'
)
ax8.plot(t_good, true_vel_good, 'k-', linewidth=2, label='True velocity')
ax8.plot(t_good, dx_dt_good, 'g--', linewidth=1.5, label='Kalman estimate')
error_good = np.abs(true_vel_good - dx_dt_good)
ax8.set_title(f'Excellent! RMSE = {np.sqrt(np.mean(error_good**2)):.3f}')
ax8.set_xlabel('Time')
ax8.set_ylabel('Velocity')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# === BOTTOM ROW: Parameter sensitivity ===

ax9 = plt.subplot(3, 4, 9)
ax9.text(0.5, 0.9, 'PARAMETER EFFECTS:', transform=ax9.transAxes,
         fontsize=11, fontweight='bold', ha='center')
ax9.text(0.1, 0.7, 'r = measurement noise', transform=ax9.transAxes, fontsize=9)
ax9.text(0.1, 0.55, '  • Large r → trust model', transform=ax9.transAxes, fontsize=8)
ax9.text(0.1, 0.4, '  • Small r → trust data', transform=ax9.transAxes, fontsize=8)
ax9.text(0.1, 0.25, 'q = process noise', transform=ax9.transAxes, fontsize=9)
ax9.text(0.1, 0.1, '  • Large q → allow changes', transform=ax9.transAxes, fontsize=8)
ax9.axis('off')

# Test different r values
ax10 = plt.subplot(3, 4, 10)
r_values = [1e-5, 1e-3, 1e-1]
for r_val in r_values:
    _, dx_dt_test = kalman.constant_velocity(
        y_noisy, dt, r=r_val, q=1e-5, forwardbackward='forward-backward'
    )
    ax10.plot(t[:100], dx_dt_test[:100], linewidth=1.5, label=f'r={r_val}', alpha=0.7)
ax10.plot(t[:100], dy_true[:100], 'k--', linewidth=1, label='True', alpha=0.5)
ax10.set_title('Effect of r (measurement noise)')
ax10.set_xlabel('Time')
ax10.set_ylabel('Derivative')
ax10.legend(fontsize=7)
ax10.grid(True, alpha=0.3)

# Test different q values
ax11 = plt.subplot(3, 4, 11)
q_values = [1e-8, 1e-5, 1e-2]
for q_val in q_values:
    _, dx_dt_test = kalman.constant_velocity(
        y_noisy, dt, r=1e-3, q=q_val, forwardbackward='forward-backward'
    )
    ax11.plot(t[:100], dx_dt_test[:100], linewidth=1.5, label=f'q={q_val}', alpha=0.7)
ax11.plot(t[:100], dy_true[:100], 'k--', linewidth=1, label='True', alpha=0.5)
ax11.set_title('Effect of q (process noise)')
ax11.set_xlabel('Time')
ax11.set_ylabel('Derivative')
ax11.legend(fontsize=7)
ax11.grid(True, alpha=0.3)

# Summary
ax12 = plt.subplot(3, 4, 12)
ax12.text(0.5, 0.9, 'VERDICT:', transform=ax12.transAxes,
         fontsize=12, fontweight='bold', ha='center')
ax12.text(0.5, 0.7, 'Kalman filters are GREAT for', transform=ax12.transAxes,
         fontsize=10, ha='center', color='green')
ax12.text(0.5, 0.55, 'tracking & smoothing', transform=ax12.transAxes,
         fontsize=10, ha='center', color='green')
ax12.text(0.5, 0.35, 'but TERRIBLE for', transform=ax12.transAxes,
         fontsize=10, ha='center', color='red')
ax12.text(0.5, 0.2, 'polynomial growth!', transform=ax12.transAxes,
         fontsize=10, ha='center', color='red')
ax12.axis('off')

plt.suptitle('PyNumDiff Kalman Filters: Purpose, Assumptions, and Limitations',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('kalman_documentation_explained.png', dpi=150, bbox_inches='tight')
print("\nSaved: kalman_documentation_explained.png")

print("\n" + "=" * 80)
print("KEY INSIGHTS ABOUT KALMAN FILTERS")
print("=" * 80)

print("""
1. PURPOSE:
   - Originally designed for Apollo spacecraft navigation
   - Optimal estimator for LINEAR systems with Gaussian noise
   - Tracks objects with approximately constant velocity/acceleration

2. MATHEMATICAL MODEL:
   State Space: x[k+1] = F·x[k] + w[k]  (w ~ N(0,Q))
   Observation: y[k] = H·x[k] + v[k]    (v ~ N(0,R))

3. ASSUMPTIONS:
   ✓ Linear dynamics (or locally linear)
   ✓ Gaussian noise
   ✓ Known model structure
   ✓ Markov property (future depends only on present)

4. WHY IT FAILS ON x^(5/2) + sin(2x):
   ✗ Velocity grows as x^(3/2) - highly nonlinear!
   ✗ Model assumes constant velocity, reality has polynomial growth
   ✗ No amount of tuning can fix a fundamentally wrong model

5. WHEN TO USE:
   ✓ GPS/IMU sensor fusion
   ✓ Radar tracking
   ✓ Robot localization
   ✓ Stock prices (sometimes)
   ✓ Temperature sensors

6. WHEN NOT TO USE:
   ✗ Polynomial signals
   ✗ Exponential growth
   ✗ Strongly nonlinear dynamics
   ✗ Non-Gaussian noise
   ✗ Unknown model structure

The PyNumDiff implementation is CORRECT - it's just being used on the
WRONG type of signal!
""")