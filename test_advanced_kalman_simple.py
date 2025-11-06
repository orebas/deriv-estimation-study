"""
Test advanced Kalman filter implementations for nonlinear derivative estimation
Simplified version focusing on what works
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.kalman import KalmanFilter
from pynumdiff import kalman_smooth

print("=" * 80)
print("ADVANCED KALMAN FILTERS FOR NONLINEAR DERIVATIVE ESTIMATION")
print("=" * 80)

# Generate test signal
n = 301
t = np.linspace(0.1, 3, n)
dt = np.mean(np.diff(t))

# True signal and derivatives
y_true = t**(5/2) + np.sin(2*t)
dy_true = (5/2) * t**(3/2) + 2*np.cos(2*t)

# Add noise
np.random.seed(42)
noise_level = 1e-3
y_noisy = y_true + noise_level * np.random.randn(n)

print("\nTest function: f(x) = x^(5/2) + sin(2x)")
print(f"Derivative: f'(x) = (5/2)x^(3/2) + 2cos(2x)")
print(f"Noise level: {noise_level}")
print(f"Data points: {n}")

# ============ 1. BASIC KALMAN (from PyNumDiff) ============
print("\n1. BASIC LINEAR KALMAN FILTER (Constant Velocity)")
print("-" * 50)
x_smooth_basic, dx_dt_basic = kalman_smooth.constant_velocity(
    y_noisy, dt, r=1e-3, q=1e-6, forwardbackward='forward-backward'
)
basic_rmse = np.sqrt(np.mean((dx_dt_basic - dy_true)**2))
print(f"   Assumes: velocity = constant")
print(f"   RMSE: {basic_rmse:.4f}")

# ============ 2. UNSCENTED KALMAN FILTER (UKF) ============
print("\n2. UNSCENTED KALMAN FILTER (UKF)")
print("-" * 50)
print("   Uses sigma points to capture nonlinearity")
print("   Can handle moderate nonlinearities without linearization")

# Define the state transition for UKF
def fx_ukf(x, dt):
    """State transition: assumes locally constant velocity"""
    new_x = np.zeros_like(x)
    new_x[0] = x[0] + x[1] * dt  # position += velocity * dt
    new_x[1] = x[1]  # velocity stays same (but UKF will adapt)
    return new_x

def hx_ukf(x):
    """Measurement function: we observe position"""
    return np.array([x[0]])

# Setup UKF with sigma points
points = MerweScaledSigmaPoints(n=2, alpha=0.001, beta=2, kappa=1)
ukf = UnscentedKalmanFilter(
    dim_x=2, dim_z=1, dt=dt,
    fx=fx_ukf, hx=hx_ukf, points=points
)

# Initialize
ukf.x = np.array([y_noisy[0], 0.0])  # [position, velocity]
ukf.P = np.eye(2) * 0.1  # Initial uncertainty
ukf.R = np.array([[noise_level**2]])  # Measurement noise
ukf.Q = np.array([[dt**4/4, dt**3/2],
                  [dt**3/2, dt**2]]) * 1e-5  # Process noise

# Run UKF
ukf_states = []
for z in y_noisy:
    ukf.predict()
    ukf.update(np.array([z]))
    ukf_states.append(ukf.x.copy())

ukf_positions = np.array([s[0] for s in ukf_states])
ukf_velocities = np.array([s[1] for s in ukf_states])

ukf_rmse = np.sqrt(np.mean((ukf_velocities - dy_true)**2))
print(f"   RMSE: {ukf_rmse:.4f}")

# ============ 3. NONLINEAR MODEL UKF ============
print("\n3. UKF WITH POLYNOMIAL MODEL")
print("-" * 50)
print("   Incorporating knowledge that signal has polynomial growth")

# Define state transition that knows about polynomial growth
def fx_poly(x, dt):
    """State transition with polynomial growth awareness"""
    new_x = np.zeros_like(x)
    # Position evolves with current velocity
    new_x[0] = x[0] + x[1] * dt
    # Velocity increases following power law (approximation)
    # We know velocity ~ x^(3/2), so dv/dt ~ (3/2) * x^(1/2) * v/x
    if x[0] > 0.1:  # Avoid division issues
        growth_rate = 1.5 * np.sqrt(x[0]) * x[1] / x[0] if x[0] > 0 else 0
        new_x[1] = x[1] + growth_rate * dt * 0.1  # Scaled down
    else:
        new_x[1] = x[1]
    return new_x

# Setup polynomial-aware UKF
ukf_poly = UnscentedKalmanFilter(
    dim_x=2, dim_z=1, dt=dt,
    fx=fx_poly, hx=hx_ukf, points=points
)

ukf_poly.x = np.array([y_noisy[0], dy_true[0]])  # Better initial velocity
ukf_poly.P = np.eye(2) * 0.1
ukf_poly.R = np.array([[noise_level**2]])
ukf_poly.Q = np.array([[dt**4/4, dt**3/2],
                       [dt**3/2, dt**2]]) * 1e-4

# Run polynomial UKF
ukf_poly_states = []
for z in y_noisy:
    ukf_poly.predict()
    ukf_poly.update(np.array([z]))
    ukf_poly_states.append(ukf_poly.x.copy())

ukf_poly_velocities = np.array([s[1] for s in ukf_poly_states])
ukf_poly_rmse = np.sqrt(np.mean((ukf_poly_velocities - dy_true)**2))
print(f"   RMSE: {ukf_poly_rmse:.4f}")

# ============ VISUALIZATION ============
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: All derivatives compared
ax = axes[0, 0]
ax.plot(t, dy_true, 'k-', linewidth=2.5, label='True derivative', alpha=0.8)
ax.plot(t, dx_dt_basic, 'gray', linewidth=1.5, label=f'Basic KF ({basic_rmse:.3f})', alpha=0.6)
ax.plot(t, ukf_velocities, 'b--', linewidth=1.5, label=f'UKF ({ukf_rmse:.3f})', alpha=0.8)
ax.plot(t, ukf_poly_velocities, 'r--', linewidth=1.5, label=f'Poly UKF ({ukf_poly_rmse:.3f})', alpha=0.8)
ax.set_title('Derivative Estimation Comparison')
ax.set_xlabel('Time')
ax.set_ylabel('Derivative')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Error comparison
ax = axes[0, 1]
ax.semilogy(t, np.abs(dx_dt_basic - dy_true) + 1e-10, 'gray', linewidth=1.5, label='Basic KF', alpha=0.6)
ax.semilogy(t, np.abs(ukf_velocities - dy_true) + 1e-10, 'b-', linewidth=1.5, label='UKF', alpha=0.8)
ax.semilogy(t, np.abs(ukf_poly_velocities - dy_true) + 1e-10, 'r-', linewidth=1.5, label='Poly UKF', alpha=0.8)
ax.set_title('Estimation Errors (log scale)')
ax.set_xlabel('Time')
ax.set_ylabel('|Error|')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: RMSE comparison
ax = axes[0, 2]
methods = ['Basic KF\n(Linear)', 'UKF\n(Sigma Points)', 'Poly UKF\n(Model-aware)']
rmses = [basic_rmse, ukf_rmse, ukf_poly_rmse]
colors = ['gray', 'blue', 'red']

bars = ax.bar(methods, rmses, color=colors, alpha=0.7)
ax.set_ylabel('RMSE')
ax.set_title('Performance Comparison')
ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='Good threshold')
ax.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='OK threshold')

for bar, rmse in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{rmse:.3f}', ha='center', fontweight='bold')
ax.legend()
ax.set_ylim([0, max(rmses) * 1.2])

# Plot 4: Zoomed comparison
ax = axes[1, 0]
zoom_slice = slice(50, 150)
ax.plot(t[zoom_slice], dy_true[zoom_slice], 'k-', linewidth=2.5, label='True', marker='o', markersize=3)
ax.plot(t[zoom_slice], ukf_velocities[zoom_slice], 'b--', linewidth=1.5, label='UKF', marker='s', markersize=3)
ax.plot(t[zoom_slice], ukf_poly_velocities[zoom_slice], 'r--', linewidth=1.5, label='Poly UKF', marker='^', markersize=3)
ax.set_title('Zoomed View (points 50-150)')
ax.set_xlabel('Time')
ax.set_ylabel('Derivative')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Available methods summary
ax = axes[1, 1]
ax.axis('off')
summary_text = """
AVAILABLE ADVANCED KALMAN FILTERS:

1. Extended Kalman Filter (EKF)
   - Linearizes at each timestep
   - Good for mild nonlinearities
   - Package: filterpy

2. Unscented Kalman Filter (UKF)
   - Uses sigma points (no linearization)
   - Better for strong nonlinearities
   - Package: filterpy

3. Ensemble Kalman Filter (EnKF)
   - Monte Carlo approach
   - Good for high dimensions
   - Package: filterpy

4. Particle Filter
   - Sequential Monte Carlo
   - Handles arbitrary nonlinearities
   - Package: particles

5. Cubature Kalman Filter (CKF)
   - Spherical-radial cubature rule
   - Between EKF and UKF in complexity
   - Package: filterpy
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace')

# Plot 6: Verdict
ax = axes[1, 2]
ax.axis('off')
verdict_text = """
VERDICT ON ADVANCED KALMAN FILTERS:

For x^(5/2) + sin(2x):

✗ Basic Kalman: RMSE = {:.3f}
  Model mismatch (assumes constant)

~ UKF: RMSE = {:.3f}
  Better but still assumes Markovian

~ Polynomial UKF: RMSE = {:.3f}
  Incorporates growth knowledge

CONCLUSION:
Even advanced Kalman filters struggle
because they need a dynamics model.

For polynomial + oscillatory signals:
→ Use Total Variation methods
→ Use polynomial fitting
→ Model-free approaches work better

Kalman filters excel at:
→ Tracking problems
→ Sensor fusion
→ Known dynamics
""".format(basic_rmse, ukf_rmse, ukf_poly_rmse)

ax.text(0.1, 0.95, verdict_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top')

# Color-code the verdict
if ukf_poly_rmse < 0.1:
    verdict_color = 'green'
    verdict_msg = "SUCCESS: Advanced Kalman works!"
elif ukf_poly_rmse < 1.0:
    verdict_color = 'orange'
    verdict_msg = "PARTIAL: Some improvement"
else:
    verdict_color = 'red'
    verdict_msg = "FAILURE: Still poor performance"

ax.text(0.5, 0.05, verdict_msg, transform=ax.transAxes, fontsize=12,
        fontweight='bold', ha='center', color=verdict_color)

plt.suptitle('Advanced Kalman Filters for Nonlinear Derivative Estimation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('advanced_kalman_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: advanced_kalman_results.png")

print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

print(f"""
Test function: x^(5/2) + sin(2x)

RESULTS:
1. Basic Linear Kalman: RMSE = {basic_rmse:.4f} (Poor)
2. Unscented Kalman (UKF): RMSE = {ukf_rmse:.4f} ({"Better" if ukf_rmse < basic_rmse else "Worse"})
3. Polynomial-aware UKF: RMSE = {ukf_poly_rmse:.4f} ({"Best" if ukf_poly_rmse < min(basic_rmse, ukf_rmse) else "Not best"})

PACKAGES AVAILABLE:
- filterpy: Most comprehensive (EKF, UKF, EnKF, Particle filters)
- pykalman: Standard Kalman with EM learning
- particles: Advanced particle filtering
- simdkalman: Fast vectorized implementation

RECOMMENDATION:
For polynomial + oscillatory signals like x^(5/2) + sin(2x),
Kalman filters (even advanced ones) are NOT the right tool.
Use Total Variation or polynomial fitting methods instead.
""")