# Hyperparameter Inventory
**Purpose:** Document all hard-coded hyperparameters across methods for potential optimization

---

## Methods with Hard-Coded Hyperparameters

### 1. **Chebyshev** (Python)
**File:** `python/python_methods.py:202-226`

```python
deg = max(3, min(20, n_train - 1))
poly = Chebyshev.fit(x_train, y_train, deg=deg, domain=[tmin, tmax])
```

**Hyperparameters:**
- `min_degree = 3`
- `max_degree = 20`

**Current Logic:**
- If n_train < 4: use degree 3 (extrapolation!)
- If 4 ≤ n_train ≤ 21: use degree n_train - 1 (interpolating polynomial)
- If n_train > 21: use degree 20 (regularized fit)

**Questions:**
- Is degree 20 optimal across all (order, noise) conditions?
- Should max_degree adapt to noise level?
  - High noise → lower degree (more regularization)
  - Low noise → higher degree (capture more detail)
- Should it adapt to derivative order?
  - High order → maybe lower degree to avoid amplifying polynomial oscillations?

**Typical n_train values:** UNKNOWN (need to check experiment configs)

---

### 2. **Fourier** (Python)
**File:** `python/python_methods.py:228-281`

```python
M = max(1, min((n_train - 1) // 4, 25))  # harmonics
```

**Hyperparameters:**
- `harmonics_fraction = 0.25` (using 25% of n_train)
- `max_harmonics = 25`

**Current Logic:**
- Number of harmonics = min(n_train/4, 25)
- Total basis functions = 2M + 1 (DC + M cosines + M sines)

**Questions:**
- Why 25%? Is this optimal?
- Should harmonics fraction adapt to noise?
  - High noise → fewer harmonics (more regularization)
  - Low noise → more harmonics (better fit)
- Is max=25 ever reached? (needs n_train > 100)

---

### 3. **Fourier Continuation** (Python)
**File:** `python/python_methods.py:282-360`

```python
deg = int(os.environ.get("FC_TREND_DEG", "3"))  # polynomial trend
M = max(1, min(((n_train - 1) // 4), 25))        # harmonics (same as fourier)
```

**Hyperparameters:**
- `trend_degree = 3` (cubic polynomial, **ENV CONFIGURABLE**)
- `harmonics_fraction = 0.25`
- `max_harmonics = 25`

**Questions:**
- Same questions as Fourier method
- Is cubic trend always appropriate?
- Could trend degree adapt to data characteristics?

**Note:** This method allows ENV override! `FC_TREND_DEG` environment variable

---

### 4. **Fourier-FFT** (Julia) - aka "Fourier-Interp"
**File:** `src/julia_methods.jl:561-605`

```julia
filter_frac::Float64 = 0.4  # Configurable parameter
k_cutoff = filter_frac * k_max_abs
# Zero out frequencies where |k| > k_cutoff
```

**Hyperparameters:**
- `filter_frac = 0.4` (keep lowest 40% of frequencies, **CONFIGURABLE**)

**Current Logic:**
- Low-pass filter: only keep frequencies up to 40% of max frequency
- This prevents high-frequency noise amplification during differentiation

**Questions:**
- Is 40% optimal across all orders and noise levels?
- **Hypothesis:** Lower noise → can use higher filter_frac (more frequencies)
- **Hypothesis:** Higher derivative order → need lower filter_frac (more aggressive filtering)

**Note:** This is passed as parameter: `fourier_fft_deriv(...; filter_frac=filter_frac)`

---

### 5. **SpectralTaper_Python**
**File:** `python/python_methods.py:480-538`

```python
alpha = float(os.environ.get("SPEC_TAPER_ALPHA", "0.25"))
cutoff = float(os.environ.get("SPEC_CUTOFF", "0.5"))
shrink = float(os.environ.get("SPEC_SHRINK", "0.0"))
```

**Hyperparameters (ALL ENV CONFIGURABLE):**
- `SPEC_TAPER_ALPHA = 0.25` (Tukey window parameter, 0=rectangular, 1=Hanning)
- `SPEC_CUTOFF = 0.5` (fraction of Nyquist frequency to keep)
- `SPEC_SHRINK = 0.0` (spectral shrinkage 0-1, currently disabled)

**Current Logic:**
- Tukey taper with 25% edges
- Keep frequencies up to 50% of Nyquist
- No shrinkage applied

**Questions:**
- Should cutoff adapt to order/noise like Fourier-FFT?
- Is shrink=0 always best, or could adaptive shrinkage help?

**Note:** Method performs poorly (proposed for exclusion), so optimization may not be worthwhile

---

### 6. **Whittaker m=2** (Python)
**File:** `python/python_methods.py:540-607`

```python
lam = float(os.environ.get("WHIT_LAMBDA", "100.0"))
```

**Hyperparameters (ENV CONFIGURABLE):**
- `WHIT_LAMBDA = 100.0` (smoothing parameter λ)

**Current Logic:**
- Solves: min_f Σ(y_i - f_i)² + λ Σ(Δ²f)²
- Higher λ → smoother fit
- Uses spline interpolation for derivative evaluation

**Questions:**
- Is λ=100 optimal?
- Should λ adapt to noise level?
  - High noise → higher λ (more smoothing)
  - Low noise → lower λ (follow data more closely)

---

### 7. **RKHS Spline m=2** (Python)
**File:** `python/python_methods.py:609-651`

```python
lam = float(os.environ.get("RKHS_LAMBDA", "1e-3"))
```

**Hyperparameters (ENV CONFIGURABLE):**
- `RKHS_LAMBDA = 1e-3` (regularization parameter)

**Current Logic:**
- RKHS smoothing spline with second-order penalty
- Similar to Whittaker but different formulation

**Questions:**
- Same as Whittaker - should λ adapt to noise?

---

### 8. **GP Methods** (Python)
**File:** `python/python_methods.py:130-200`

```python
# GP-RBF
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
         + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
n_restarts_optimizer=5

# GP-Matern
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5, ...) + ...
n_restarts_optimizer=5
```

**Hyperparameters:**
- `length_scale_init = 1.0`
- `length_scale_bounds = (1e-2, 1e2)`
- `amplitude_bounds = (1e-3, 1e3)`
- `noise_level_bounds = (1e-10, 1e-2)`
- `n_restarts_optimizer = 5`
- `nu = 1.5 or 2.5` (for Matern)

**Note:** These are **optimized via MLE** during fitting, not hard-coded!

**Questions:**
- Are bounds appropriate?
- Is n_restarts=5 enough for global optimum?
- Could better initialization help?

---

## Summary Table

| Method | Hyperparameter | Current Value | Adaptive? | ENV Override? |
|--------|---------------|---------------|-----------|---------------|
| Chebyshev | max_degree | 20 | No | No |
| Chebyshev | min_degree | 3 | No | No |
| Fourier | harmonics_frac | 0.25 | No | No |
| Fourier | max_harmonics | 25 | No | No |
| Fourier-Cont | trend_degree | 3 | No | **YES** (FC_TREND_DEG) |
| Fourier-Cont | harmonics_frac | 0.25 | No | No |
| Fourier-FFT | filter_frac | 0.4 | No | **YES** (param) |
| SpectralTaper | taper_alpha | 0.25 | No | **YES** (SPEC_TAPER_ALPHA) |
| SpectralTaper | cutoff | 0.5 | No | **YES** (SPEC_CUTOFF) |
| SpectralTaper | shrink | 0.0 | No | **YES** (SPEC_SHRINK) |
| Whittaker | lambda | 100.0 | No | **YES** (WHIT_LAMBDA) |
| RKHS | lambda | 1e-3 | No | **YES** (RKHS_LAMBDA) |
| GP-* | length_scale | 1.0 (init) | **YES** (MLE) | No |
| GP-* | n_restarts | 5 | No | No |

---

## Optimization Opportunities

### High Priority (Popular/Good Methods)

1. **Fourier-FFT filter_frac (0.4)**
   - Currently one of the best methods
   - Strong theoretical reason for adaptation:
     - Higher order → need more aggressive filtering (lower frac)
     - Higher noise → need more aggressive filtering (lower frac)
   - **Proposed:** `filter_frac = f(order, noise)`

2. **Chebyshev max_degree (20)**
   - Good performer, but degree cap is arbitrary
   - **Proposed:** Adapt to noise level
     - Low noise: maybe degree 30-40?
     - High noise: maybe degree 10-15?
   - Need to check typical n_train first

3. **Fourier harmonics_fraction (0.25)**
   - Same logic as Fourier-FFT filtering
   - **Proposed:** `harmonics_frac = f(order, noise)`

### Medium Priority

4. **Whittaker lambda (100.0)**
   - Should clearly adapt to noise
   - **Proposed:** `lambda = g(noise)`

5. **RKHS lambda (1e-3)**
   - Same as Whittaker

### Low Priority (Poor Performers or Already Optimized)

6. **SpectralTaper parameters** - Method performs poorly, may not be worth optimizing
7. **GP parameters** - Already using MLE optimization
8. **Fourier-Continuation trend_degree** - Already ENV configurable, less critical

---

## Next Steps

### Investigation Phase

1. **Check typical n_train values**
   - Determines if degree caps are active
   - Look at experiment configuration files

2. **Theoretical Analysis**
   - What's the theoretical optimal filter_frac as function of (order, noise)?
   - What's the theoretical optimal degree for Chebyshev?

3. **Empirical Analysis**
   - For top methods, plot performance vs hyperparameter value
   - Look for patterns across (order, noise) conditions

### Optimization Phase

**Option A: Grid Search (Simple)**
- For each method, try 5-10 values of hyperparameter
- Re-run benchmark on subset of conditions
- Pick best value(s)

**Option B: Adaptive Rules (Smarter)**
- Derive heuristic formulas based on noise level and order
- Example: `filter_frac = 0.6 / (1 + 0.1*order + 10*noise)`
- Validate on held-out conditions

**Option C: Cross-Validation (Most Rigorous)**
- For each (order, noise, method) combination:
  - Split data into train/validation
  - Optimize hyperparameter via CV
  - Report best value
- Most expensive computationally

---

## Questions to Answer

1. **What is typical n_train in our experiments?**
   - Determines if caps (degree 20, harmonics 25) are active

2. **Why does AAA-LowPrec underperform Chebyshev?**
   - AAA uses rational approximation (theoretically more powerful)
   - Is it a hyperparameter issue?
   - Is it numerical precision?
   - Check AAA implementation for hidden parameters

3. **Should we optimize before or after paper submission?**
   - Before: Better results, more scientifically rigorous
   - After: Faster to publication, optimization could be future work
   - Compromise: Optimize 1-2 key methods, note others as future work

4. **How much improvement can we expect?**
   - Best case: 20-50% reduction in nRMSE for some conditions
   - Realistic: 10-20% average improvement
   - Worth it if it changes method rankings or conclusions
