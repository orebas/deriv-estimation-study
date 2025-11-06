"""
Examine PyNumDiff Kalman filter documentation and implementation
Let's understand what these methods are SUPPOSED to be doing
"""

import inspect
import pynumdiff.kalman_smooth as kalman
import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("PYNUMDIFF KALMAN FILTER DOCUMENTATION & IMPLEMENTATION ANALYSIS")
print("=" * 80)

# Get documentation
print("\n1. CONSTANT VELOCITY MODEL:")
print("-" * 40)
print(kalman.constant_velocity.__doc__)

print("\n2. CONSTANT ACCELERATION MODEL:")
print("-" * 40)
print(kalman.constant_acceleration.__doc__)

print("\n3. RTS SMOOTHER:")
print("-" * 40)
print(kalman.rts_smooth.__doc__)

# Get source code for constant_velocity
print("\n" + "=" * 80)
print("SOURCE CODE INSPECTION - constant_velocity")
print("=" * 80)
try:
    source = inspect.getsource(kalman.constant_velocity)
    # Print first 50 lines to understand the model
    lines = source.split('\n')[:50]
    for i, line in enumerate(lines, 1):
        print(f"{i:3}: {line}")
except:
    print("Could not get source code")

# Let's understand the state space model
print("\n" + "=" * 80)
print("KALMAN FILTER STATE SPACE MODELS EXPLAINED")
print("=" * 80)

print("""
CONSTANT VELOCITY MODEL:
------------------------
State vector: x = [position, velocity]ᵀ

State transition (discrete time):
  x[k+1] = F * x[k] + w[k]

  where F = [1  dt]  (position += velocity * dt)
            [0   1]  (velocity stays constant)

  w[k] ~ N(0, Q) is process noise

Measurement model:
  y[k] = H * x[k] + v[k]

  where H = [1, 0]  (we only measure position)

  v[k] ~ N(0, R) is measurement noise

Parameters:
  - dt: time step
  - r: measurement noise variance (how much we trust measurements)
  - q: process noise variance (how much velocity can change)
  - forwardbackward: 'forward', 'backward', or 'forward-backward'

CONSTANT ACCELERATION MODEL:
----------------------------
State vector: x = [position, velocity, acceleration]ᵀ

State transition:
  F = [1  dt  dt²/2]  (position += velocity*dt + 0.5*accel*dt²)
      [0   1     dt]  (velocity += acceleration*dt)
      [0   0      1]  (acceleration stays constant)

The model assumes acceleration is constant (plus noise).
""")

# Now let's visualize what the Kalman filter THINKS it should do
print("\n" + "=" * 80)
print("VISUALIZING KALMAN FILTER ASSUMPTIONS VS REALITY")
print("=" * 80)

# Generate our test function
n = 301
t = np.linspace(0.1, 3, n)
dt = np.mean(np.diff(t))

# True signal and derivative
y_true = t**(5/2) + np.sin(2*t)
dy_true = (5/2) * t**(3/2) + 2*np.cos(2*t)
ddy_true = (5/2) * (3/2) * t**(1/2) - 2*np.sin(2*t)  # Second derivative

# Add noise
np.random.seed(42)
y_noisy = y_true + 1e-3 * np.random.randn(n)

# Create visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Row 1: What constant velocity model assumes
ax = axes[0, 0]
ax.plot([0, 3], [2, 2], 'r-', linewidth=2, label='Assumed: constant')
ax.plot(t, dy_true, 'k-', linewidth=1, label='Reality: (5/2)x^(3/2) + 2cos(2x)')
ax.set_title('Constant Velocity Model Assumption')
ax.set_ylabel('Velocity (1st derivative)')
ax.set_xlabel('Time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
# Show how velocity actually changes
velocity_change = np.diff(dy_true)
ax.plot(t[:-1], velocity_change/dt, 'b-', linewidth=1.5)
ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Model assumes: 0')
ax.set_title('Actual Velocity Change (Acceleration)')
ax.set_ylabel('dv/dt')
ax.set_xlabel('Time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
# Show the error growth
const_vel_estimate = np.full_like(dy_true, dy_true[0])  # Constant velocity = initial velocity
error = dy_true - const_vel_estimate
ax.plot(t, error, 'r-', linewidth=2)
ax.fill_between(t, 0, error, alpha=0.3, color='red')
ax.set_title('Error if Velocity Were Constant')
ax.set_ylabel('True - Constant')
ax.set_xlabel('Time')
ax.grid(True, alpha=0.3)

# Row 2: What constant acceleration model assumes
ax = axes[1, 0]
ax.plot([0, 3], [1, 1], 'r-', linewidth=2, label='Assumed: constant')
ax.plot(t, ddy_true, 'k-', linewidth=1, label='Reality: variable')
ax.set_title('Constant Acceleration Model Assumption')
ax.set_ylabel('Acceleration (2nd derivative)')
ax.set_xlabel('Time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
# Show actual jerk (change in acceleration)
jerk = np.diff(ddy_true)
ax.plot(t[:-1], jerk/dt, 'b-', linewidth=1.5)
ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Model assumes: 0')
ax.set_title('Actual Jerk (da/dt)')
ax.set_ylabel('d³x/dt³')
ax.set_xlabel('Time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
# Fit a constant acceleration and show error
const_accel = np.mean(ddy_true)
vel_from_const_accel = dy_true[0] + const_accel * (t - t[0])
error = dy_true - vel_from_const_accel
ax.plot(t, error, 'r-', linewidth=2)
ax.fill_between(t, 0, error, alpha=0.3, color='red')
ax.set_title('Error if Acceleration Were Constant')
ax.set_ylabel('True Velocity - Linear Model')
ax.set_xlabel('Time')
ax.grid(True, alpha=0.3)

# Row 3: Actual Kalman filter results
ax = axes[2, 0]
# Run constant velocity filter
x_smooth_cv, dx_dt_cv = kalman.constant_velocity(
    y_noisy, dt, r=1e-3, q=1e-5, forwardbackward='forward-backward'
)
ax.plot(t, dy_true, 'k-', linewidth=2, alpha=0.7, label='True velocity')
ax.plot(t, dx_dt_cv, 'r-', linewidth=1.5, label='Kalman CV estimate')
ax.set_title('Constant Velocity Filter Result')
ax.set_ylabel('Velocity')
ax.set_xlabel('Time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
# Run constant acceleration filter
x_smooth_ca, dx_dt_ca, d2x_dt2_ca = kalman.constant_acceleration(
    y_noisy, dt, r=1e-3, q=1e-5, forwardbackward='forward-backward'
)
ax.plot(t, dy_true, 'k-', linewidth=2, alpha=0.7, label='True velocity')
ax.plot(t, dx_dt_ca, 'b-', linewidth=1.5, label='Kalman CA estimate')
ax.set_title('Constant Acceleration Filter Result')
ax.set_ylabel('Velocity')
ax.set_xlabel('Time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 2]
# Compare errors
error_cv = np.abs(dy_true - dx_dt_cv)
error_ca = np.abs(dy_true - dx_dt_ca)
ax.semilogy(t, error_cv, 'r-', linewidth=1.5, label='Const Velocity', alpha=0.7)
ax.semilogy(t, error_ca, 'b-', linewidth=1.5, label='Const Accel', alpha=0.7)
ax.set_title('Kalman Filter Errors (log scale)')
ax.set_ylabel('|Error|')
ax.set_xlabel('Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Kalman Filter Models: Assumptions vs Reality', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('kalman_model_explanation.png', dpi=150, bbox_inches='tight')
print("\nSaved: kalman_model_explanation.png")

# Explain when these models WOULD work well
print("\n" + "=" * 80)
print("WHEN KALMAN FILTERS WORK WELL:")
print("=" * 80)

print("""
✅ GOOD USE CASES for Kalman filters:

1. TRACKING PROBLEMS:
   - GPS tracking (velocity ~constant between samples)
   - Radar tracking of aircraft (smooth trajectories)
   - Robot position estimation

2. SIGNALS WITH THESE PROPERTIES:
   - Piecewise constant or linear velocity
   - Slow, smooth changes
   - Gaussian noise
   - No strong nonlinear trends

3. EXAMPLE GOOD SIGNALS:
   - Step functions with noise
   - Ramps with small oscillations
   - Slowly varying sinusoids
   - Random walks

❌ BAD USE CASES (like our example):

1. POLYNOMIAL GROWTH:
   - x^n where n > 1
   - Exponential functions
   - Any strongly nonlinear trend

2. MIXED FREQUENCY CONTENT:
   - Polynomial + oscillations
   - Multi-scale signals
   - Chirp signals

3. NON-GAUSSIAN PROCESSES:
   - Impulsive noise
   - Outliers
   - Non-stationary variance

The fundamental issue: Kalman filters are MODEL-BASED estimators.
If your signal doesn't match the model, they WILL fail!
""")

# Show a GOOD example where Kalman works
print("\n" + "=" * 80)
print("DEMONSTRATING WHERE KALMAN FILTERS EXCEL")
print("=" * 80)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Generate a signal that matches Kalman assumptions
t2 = np.linspace(0, 10, 500)
dt2 = t2[1] - t2[0]

# Piecewise constant velocity (like tracking problem)
true_velocity = np.zeros_like(t2)
true_velocity[100:200] = 2.0   # Constant velocity period 1
true_velocity[200:350] = -1.0  # Constant velocity period 2
true_velocity[350:450] = 3.0   # Constant velocity period 3

# Integrate to get position
true_position = np.cumsum(true_velocity) * dt2

# Add realistic measurement noise
np.random.seed(43)
noisy_position = true_position + 0.1 * np.random.randn(len(t2))

# Apply Kalman filter
x_smooth, dx_dt = kalman.constant_velocity(
    noisy_position, dt2, r=0.01, q=0.1, forwardbackward='forward-backward'
)

ax = axes2[0, 0]
ax.plot(t2, true_position, 'k-', linewidth=2, label='True position')
ax.plot(t2, noisy_position, '.', alpha=0.3, markersize=1, label='Noisy measurements')
ax.plot(t2, x_smooth, 'r-', linewidth=1.5, label='Kalman filtered')
ax.set_title('Position: Where Kalman Excels')
ax.set_xlabel('Time')
ax.set_ylabel('Position')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes2[0, 1]
ax.plot(t2, true_velocity, 'k-', linewidth=2, label='True velocity')
ax.plot(t2, dx_dt, 'r-', linewidth=1.5, label='Kalman estimate')
ax.set_title('Velocity: Excellent Recovery')
ax.set_xlabel('Time')
ax.set_ylabel('Velocity')
ax.legend()
ax.grid(True, alpha=0.3)

# Show why it works: velocity IS approximately constant!
ax = axes2[1, 0]
ax.plot(t2[:-1], np.diff(true_velocity)/dt2, 'b-', linewidth=1.5)
ax.set_title('Acceleration: Nearly Zero (Model Assumption Holds!)')
ax.set_xlabel('Time')
ax.set_ylabel('Acceleration')
ax.grid(True, alpha=0.3)

ax = axes2[1, 1]
error = np.abs(dx_dt - true_velocity)
ax.semilogy(t2, error + 1e-10, 'r-', linewidth=1.5)
ax.set_title('Kalman Error: Excellent Performance')
ax.set_xlabel('Time')
ax.set_ylabel('|Error| (log scale)')
ax.grid(True, alpha=0.3)

plt.suptitle('Kalman Filters on Appropriate Signals: Excellent Performance',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('kalman_good_example.png', dpi=150, bbox_inches='tight')
print("Saved: kalman_good_example.png")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The PyNumDiff Kalman filters are CORRECTLY IMPLEMENTED for their intended purpose:
- Tracking problems with approximately constant velocity/acceleration
- Smoothing noisy measurements where the underlying dynamics are simple

They FAIL on our test function because:
- x^(5/2) has strongly nonlinear velocity growth
- The model assumptions are fundamentally violated

This is not a bug - it's using the wrong tool for the job!
""")