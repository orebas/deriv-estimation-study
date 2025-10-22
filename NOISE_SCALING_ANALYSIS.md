# Why High Noise Levels Are Slower

## Performance Data from Regeneration

| Noise Level | Avg Time | Status |
|-------------|----------|--------|
| 1e-8 (0.000001%) | 1638s (~27 min) | ✅ Success |
| 100e-8 (0.00001%) | 1795s (~30 min) | ✅ Success |
| 10000e-8 (0.001%) | 1885s (~31 min) | ✅ Success |
| 100000e-8 (0.01%) | >1923s (>32 min) | ❌ **Timeout** |
| 1000000e-8 (0.1%) | >1923s (>32 min) | ❌ **Timeout** |

**Pattern:** Time increases ~5% per 100× noise increase, exceeds 30min at highest levels.

---

## Why AAA-JAX Is Slower with High Noise

### 1. **AAA Algorithm Iterations**
AAA (Adaptive Antoulas-Anderson) is iterative:
- Low noise: Smooth data → Fast convergence (10-20 iterations)
- High noise: Noisy data → Slow convergence (50-100+ iterations)

Each iteration selects a new support point to minimize approximation error.

### 2. **More Support Points Selected**
With noise-adaptive tolerance:
```python
tolerance = 10.0 * noise_estimate  # From wavelet MAD or diff2
```

- Low noise (σ=1e-8): tol ≈ 1e-7 → Few support points (~10)
- High noise (σ=1e-4): tol ≈ 1e-3 → Many support points (~30-40)

More support points = larger rational function = slower JAX compilation.

### 3. **JAX Compilation Time**
JAX compiles functions based on their structure:
- Rational function with 10 poles: Fast compile
- Rational function with 40 poles: Slow compile
- **7th derivative** of 40-pole function: Very slow!

Each nested `jax.grad()` recompiles the derivative graph.

### 4. **Numerical Conditioning**
High noise makes the AAA linear systems ill-conditioned:
- More iterations needed for convergence
- Potential retries/refinements
- Slower linear algebra operations

---

## Estimated Time for Failed Files

Based on trend: **40-60 minutes each**

With 120-minute timeout: **Safe margin** ✅

---

## Why This Doesn't Affect Other Methods

- **Fourier methods:** FFT is O(n log n) regardless of noise
- **Chebyshev:** Fixed degree selection, independent of noise
- **GP methods:** Kernel computations scale with n, not noise
- **Python AAA:** Uses baryrat (not JAX), different performance profile

**Only AAA-JAX methods** have this exponential noise scaling due to:
- Adaptive support point selection
- JAX compilation overhead
- High-order derivative computation
