"""
Test advanced Kalman filter implementations for nonlinear systems
Looking for: Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), etc.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("SEARCHING FOR ADVANCED KALMAN FILTER IMPLEMENTATIONS")
print("=" * 80)

# First, let's check what's available
packages_to_check = [
    ('filterpy', 'Advanced Kalman filters (EKF, UKF, etc.)'),
    ('pykalman', 'Kalman filtering and smoothing'),
    ('simdkalman', 'Fast Kalman filtering'),
    ('kalman', 'Simple Kalman filter'),
    ('ukfm', 'Unscented Kalman Filter on Manifolds'),
]

available_packages = []

print("\nChecking available packages:")
print("-" * 40)
for pkg_name, description in packages_to_check:
    try:
        module = __import__(pkg_name)
        available_packages.append((pkg_name, module))
        print(f"✓ {pkg_name:15} - {description}")

        # Check what's in the package
        if hasattr(module, '__version__'):
            print(f"  Version: {module.__version__}")
    except ImportError:
        print(f"✗ {pkg_name:15} - Not installed")

# Let's try to install and use filterpy - the most comprehensive package
if not any(pkg[0] == 'filterpy' for pkg in available_packages):
    print("\nInstalling filterpy for advanced Kalman filters...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'filterpy'])
        import filterpy
        print("✓ filterpy installed successfully")
    except:
        print("✗ Could not install filterpy")

# Test with filterpy if available
try:
    from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
    from filterpy.kalman import KalmanFilter, MerweScaledSigmaPoints

    print("\n" + "=" * 80)
    print("TESTING ADVANCED KALMAN FILTERS")
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

    print("\n1. EXTENDED KALMAN FILTER (EKF)")
    print("-" * 40)
    print("EKF linearizes the nonlinear dynamics at each step")
    print("Good for mild nonlinearities, but still assumes local linearity")

    # Setup EKF for derivative estimation
    # State: [position, velocity]
    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = np.array([[y_noisy[0]], [0.0]])  # Initial state
    ekf.F = np.eye(2)  # Will be updated
    ekf.H = np.array([[1, 0]])  # Measure position only
    ekf.R = noise_level**2  # Measurement noise
    ekf.Q = np.array([[dt**4/4, dt**3/2],
                      [dt**3/2, dt**2]]) * 1e-4  # Process noise
    ekf.P *= 1.0  # Initial covariance

    # Define the state transition and its Jacobian
    def fx(x, dt):
        # Nonlinear state transition
        # This is where we could incorporate knowledge about the system
        F = np.array([[x[0, 0] + x[1, 0] * dt],
                      [x[1, 0]]])  # Still assuming constant velocity for now
        return F

    def FJacobian(x, dt):
        # Jacobian of state transition
        return np.array([[1, dt],
                        [0, 1]])

    # Run EKF
    ekf_states = []
    for i, z in enumerate(y_noisy):
        ekf.F = FJacobian(ekf.x, dt)
        ekf.predict()
        ekf.update(z, HJacobian=lambda x: np.array([[1, 0]]), Hx=lambda x: x[0])
        ekf_states.append(ekf.x.copy())

    ekf_positions = np.array([s[0, 0] for s in ekf_states])
    ekf_velocities = np.array([s[1, 0] for s in ekf_states])

    ekf_rmse = np.sqrt(np.mean((ekf_velocities - dy_true)**2))
    print(f"EKF RMSE: {ekf_rmse:.4f}")

    print("\n2. UNSCENTED KALMAN FILTER (UKF)")
    print("-" * 40)
    print("UKF uses sigma points to capture nonlinearity")
    print("Better than EKF for highly nonlinear systems")

    # Setup UKF
    points = MerweScaledSigmaPoints(2, alpha=1e-3, beta=2, kappa=1)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=dt, fx=fx, hx=lambda x: x[0],
                                points=points)
    ukf.x = np.array([y_noisy[0], 0.0])
    ukf.P *= 0.1
    ukf.R = noise_level**2
    ukf.Q = np.array([[dt**4/4, dt**3/2],
                      [dt**3/2, dt**2]]) * 1e-4

    # Run UKF
    ukf_states = []
    ukf_covs = []
    for z in y_noisy:
        ukf.predict()
        ukf.update(z)
        ukf_states.append(ukf.x.copy())
        ukf_covs.append(ukf.P.copy())

    ukf_positions = np.array([s[0] for s in ukf_states])
    ukf_velocities = np.array([s[1] for s in ukf_states])

    ukf_rmse = np.sqrt(np.mean((ukf_velocities - dy_true)**2))
    print(f"UKF RMSE: {ukf_rmse:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Position tracking
    ax = axes[0, 0]
    ax.plot(t, y_true, 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(t, ekf_positions, 'b--', linewidth=1.5, label='EKF', alpha=0.8)
    ax.plot(t, ukf_positions, 'r--', linewidth=1.5, label='UKF', alpha=0.8)
    ax.set_title('Position Tracking')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Velocity estimation
    ax = axes[0, 1]
    ax.plot(t, dy_true, 'k-', linewidth=2, label='True', alpha=0.7)
    ax.plot(t, ekf_velocities, 'b--', linewidth=1.5, label=f'EKF (RMSE={ekf_rmse:.3f})', alpha=0.8)
    ax.plot(t, ukf_velocities, 'r--', linewidth=1.5, label=f'UKF (RMSE={ukf_rmse:.3f})', alpha=0.8)
    ax.set_title('Derivative (Velocity) Estimation')
    ax.set_xlabel('Time')
    ax.set_ylabel('Derivative')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Error comparison
    ax = axes[0, 2]
    ax.semilogy(t, np.abs(ekf_velocities - dy_true) + 1e-10, 'b-', linewidth=1.5, label='EKF error', alpha=0.7)
    ax.semilogy(t, np.abs(ukf_velocities - dy_true) + 1e-10, 'r-', linewidth=1.5, label='UKF error', alpha=0.7)
    ax.set_title('Estimation Errors (log scale)')
    ax.set_xlabel('Time')
    ax.set_ylabel('|Error|')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Uncertainty bounds (UKF)
    ax = axes[1, 0]
    ukf_std = np.sqrt([P[1, 1] for P in ukf_covs])
    ax.plot(t, ukf_velocities, 'r-', linewidth=1.5, label='UKF estimate')
    ax.fill_between(t, ukf_velocities - 2*ukf_std, ukf_velocities + 2*ukf_std,
                     alpha=0.3, color='red', label='±2σ bounds')
    ax.plot(t, dy_true, 'k--', linewidth=1, alpha=0.7, label='True')
    ax.set_title('UKF with Uncertainty Bounds')
    ax.set_xlabel('Time')
    ax.set_ylabel('Derivative')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Compare all Kalman variants
    ax = axes[1, 1]

    # Also get basic Kalman from PyNumDiff for comparison
    from pynumdiff import kalman_smooth
    x_smooth_basic, dx_dt_basic = kalman_smooth.constant_velocity(
        y_noisy, dt, r=1e-3, q=1e-6, forwardbackward='forward-backward'
    )
    basic_rmse = np.sqrt(np.mean((dx_dt_basic - dy_true)**2))

    methods = ['Basic KF', 'EKF', 'UKF']
    rmses = [basic_rmse, ekf_rmse, ukf_rmse]
    colors = ['gray', 'blue', 'red']

    bars = ax.bar(methods, rmses, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE')
    ax.set_title('Kalman Filter Variants Comparison')
    ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    for bar, rmse in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{rmse:.3f}', ha='center')
    ax.legend()

    # Plot 6: Explanation
    ax = axes[1, 2]
    ax.text(0.5, 0.9, 'ADVANCED KALMAN FILTERS', transform=ax.transAxes,
            fontsize=12, fontweight='bold', ha='center')
    ax.text(0.1, 0.75, '• EKF: Linearizes at each step', transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.65, '• UKF: Uses sigma points', transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.55, '• Particle: Monte Carlo sampling', transform=ax.transAxes, fontsize=10)

    ax.text(0.1, 0.35, 'Still assume Markovian dynamics!', transform=ax.transAxes,
            fontsize=10, color='red')
    ax.text(0.1, 0.25, 'Cannot handle x^(5/2) growth well', transform=ax.transAxes,
            fontsize=10, color='red')
    ax.text(0.1, 0.15, 'Need model of the nonlinearity', transform=ax.transAxes,
            fontsize=10, color='red')
    ax.axis('off')

    plt.suptitle('Advanced Kalman Filters for Nonlinear Systems', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('advanced_kalman_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: advanced_kalman_comparison.png")

except ImportError as e:
    print(f"\nCould not test advanced Kalman filters: {e}")
    print("Try: pip install filterpy")

# Check other packages
print("\n" + "=" * 80)
print("OTHER NONLINEAR FILTERING PACKAGES")
print("=" * 80)

print("""
1. FILTERPY (pip install filterpy)
   - Extended Kalman Filter (EKF)
   - Unscented Kalman Filter (UKF)
   - Particle filters
   - Ensemble Kalman Filter
   - Good documentation and examples

2. PYKALMAN (pip install pykalman)
   - Standard Kalman filter
   - EM algorithm for learning parameters
   - Missing data handling
   - More focused on standard KF

3. SIMDKALMAN (pip install simdkalman)
   - Optimized for speed (uses vectorization)
   - Good for multiple simultaneous filters
   - Standard Kalman only

4. PARTICLE FILTERS (pip install particles)
   - Sequential Monte Carlo methods
   - Can handle arbitrary nonlinearities
   - No Gaussian assumption needed
   - Computationally expensive

5. PYTORCH-BASED (torch-kalman)
   - Neural network integration
   - Learnable dynamics
   - GPU acceleration

For x^(5/2) + sin(2x), even advanced Kalman filters struggle because:
- They still need a MODEL of the dynamics
- x^(5/2) is strongly nonlinear with no simple model
- Better to use model-free methods (TV, polynomial fitting)
""")