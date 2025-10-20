#!/usr/bin/env python3
"""
Quick test to verify optimized Matern implementation works in the benchmark framework.
"""
import json
import sys
import os

# Add path for imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from python_methods import MethodEvaluator

# Test setup: simple sine wave
x_train = np.linspace(0, 2*np.pi, 21)
y_train = np.sin(x_train) + 0.001 * np.random.randn(len(x_train))
x_eval = x_train.copy()
orders = [0, 1, 2, 3, 4, 5]

print("=" * 70)
print("MATERN OPTIMIZATION VERIFICATION TEST")
print("=" * 70)

evaluator = MethodEvaluator(x_train, y_train, x_eval, orders)

# Test Matern-1.5
print("\nTesting GP_Matern_1.5_Python...")
import time
t0 = time.time()
result_15 = evaluator.evaluate_method("GP_Matern_1.5_Python")
t1 = time.time()

print(f"  Success: {result_15['success']}")
print(f"  Timing: {t1-t0:.3f}s")
if not result_15['success']:
    print(f"  Error: {result_15.get('error', 'Unknown')}")
else:
    print(f"  Orders completed: {list(result_15['predictions'].keys())}")
    print(f"  Failures: {result_15.get('failures', {})}")

# Test Matern-2.5
print("\nTesting GP_Matern_2.5_Python...")
t0 = time.time()
result_25 = evaluator.evaluate_method("GP_Matern_2.5_Python")
t1 = time.time()

print(f"  Success: {result_25['success']}")
print(f"  Timing: {t1-t0:.3f}s")
if not result_25['success']:
    print(f"  Error: {result_25.get('error', 'Unknown')}")
else:
    print(f"  Orders completed: {list(result_25['predictions'].keys())}")
    print(f"  Failures: {result_25.get('failures', {})}")

print("\n" + "=" * 70)
if result_15['success'] and result_25['success']:
    print("✓ MATERN OPTIMIZATION VERIFIED - READY FOR BENCHMARK")
else:
    print("✗ MATERN TEST FAILED - NEEDS DEBUGGING")
print("=" * 70)
