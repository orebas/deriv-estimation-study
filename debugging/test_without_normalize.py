#!/usr/bin/env python3
"""
Test Matérn WITHOUT normalize_y to isolate the issue.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

# Simple sine test
x_train = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)
y_train = np.sin(x_train.ravel())
x_eval = np.array([np.pi/4, np.pi/2, 3*np.pi/4, np.pi]).reshape(-1, 1)

print("Testing Matérn 2.5 WITHOUT normalize_y")
print("=" * 60)

# Test WITHOUT normalize_y
kernel = ConstantKernel(0.5, (1e-4, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(3e-3, 3e2), nu=2.5) \
         + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=20,
    alpha=1e-10,
    normalize_y=False,  # DISABLED
)

gp.fit(x_train, y_train)

print(f"\nFitted kernel: {gp.kernel_}")
print(f"\nPredictions vs True:")
preds = gp.predict(x_eval)
true = np.sin(x_eval.ravel())
for i, (p, t) in enumerate(zip(preds, true)):
    print(f"  {p:.6f} vs {t:.6f} (error: {abs(p-t):.6f})")

print(f"\nRMSE: {np.sqrt(np.mean((preds - true)**2)):.6f}")

print("\n" + "=" * 60)
print("Testing Matérn 2.5 WITH normalize_y")
print("=" * 60)

# Test WITH normalize_y
kernel2 = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(3e-3, 3e2), nu=2.5) \
          + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))

gp2 = GaussianProcessRegressor(
    kernel=kernel2,
    n_restarts_optimizer=20,
    alpha=1e-10,
    normalize_y=True,  # ENABLED
)

gp2.fit(x_train, y_train)

print(f"\nFitted kernel: {gp2.kernel_}")
print(f"\nPredictions vs True:")
preds2 = gp2.predict(x_eval)
for i, (p, t) in enumerate(zip(preds2, true)):
    print(f"  {p:.6f} vs {t:.6f} (error: {abs(p-t):.6f})")

print(f"\nRMSE: {np.sqrt(np.mean((preds2 - true)**2)):.6f}")
