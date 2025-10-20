# Hyperparameter Analysis
**Finding:** Both Chebyshev and Fourier are at their regularization caps!

---

## Experimental Setup

**Training data size:** n_train = **101 points**
- Domain: [0, 10]
- Spacing: Δt = 0.1
- Source: Lotka-Volterra ODE solution

---

## Active Hyperparameters

### Chebyshev
```python
n_train = 101
deg = max(3, min(20, n_train - 1))
    = max(3, min(20, 100))
    = 20  ← CAP IS ACTIVE!
```

**Interpretation:**
- Without cap: would use degree-100 polynomial (exact interpolation)
- With cap: using degree-20 polynomial (heavily regularized)
- **This is 80% regularization** (reducing from 100 to 20 degrees of freedom)

**Questions:**
- Is degree-20 optimal for all (order, noise) conditions?
- Could higher degrees work better at low noise?
- Could lower degrees work better at high noise or high orders?

---

### Fourier (Python)
```python
n_train = 101
M = max(1, min((n_train - 1) // 4, 25))
  = max(1, min(25, 25))
  = 25 harmonics  ← CAP IS EXACTLY REACHED!
```

**Interpretation:**
- Using 25% rule: (101-1)//4 = 25
- Total basis functions: 2M + 1 = 51 (DC + 25 cos + 25 sin)
- This is **50% of available degrees of freedom** (51/101)

**Note:** Cap of 25 is exactly at the computed value, so **not** limiting (yet)

---

### Fourier-FFT (Julia) - "Fourier-Interp"
```julia
# Uses FFT with 101 frequencies
filter_frac = 0.4
# Keeps only lowest 40% of frequencies during differentiation
```

**Interpretation:**
- Has access to ~51 frequency components (half spectrum)
- filter_frac=0.4 means keeping ~20 frequencies
- This is **40% of available frequencies** (20/51)

---

## Why Does AAA-LowPrec Underperform Chebyshev?

**Theory:** AAA (Adaptive Antoulas-Anderson) uses rational approximation:
- Form: r(z) = p(z)/q(z) where p, q are polynomials
- Should be **more powerful** than polynomial approximation
- Can represent functions with singularities (poles)

**Possible Explanations:**

### 1. **Overfitting to Noise**
- AAA tries to fit data "too well"
- With noise, this creates spurious poles near the domain
- Derivatives near poles → catastrophic errors

### 2. **Numerical Precision Issues**
- AAA-LowPrec uses Float64 vs AAA-HighPrec uses BigFloat
- LowPrec may have:
  - Ill-conditioned Cauchy matrices
  - Numerical cancellation in barycentric formula
  - Pole-zero cancellation errors

### 3. **Poor Hyperparameter Choice**
Looking at the AAA implementation:

```julia
function fit_aaa(x, y; tol=1e-13, mmax=100)
```

**Hyperparameters:**
- `tol = 1e-13` : Tolerance for greedy approximation
- `mmax = 100` : Maximum number of support points

**Problem:** `tol=1e-13` is VERY aggressive for Float64!
- Float64 machine epsilon: ~2.2e-16
- Asking for 1e-13 accuracy with **noisy data** → overfitting!
- The algorithm will keep adding poles to reduce residual
- Each pole is a potential source of derivative instability

### 4. **Comparison to Chebyshev**
- Chebyshev: Degree-20 polynomial (20 parameters)
- AAA: Up to 100 poles + 100 zeros (potentially 200 parameters!)
- AAA is **much more flexible** → **easier to overfit**

### 5. **Lack of Noise-Adaptive Regularization**
- Chebyshev: Fixed degree cap (uniform regularization)
- Fourier: Fixed harmonics cap (uniform regularization)
- AAA: `tol` parameter is **NOT adapted to noise level**
  - Same tol=1e-13 for noise=1e-8 and noise=5e-2!
  - Should use `tol = f(noise_level)` to prevent overfitting

---

## Proposed Hyperparameter Improvements

### Priority 1: AAA Tolerance Adaptation
**Current:**
```julia
fit_aaa(x, y; tol=1e-13, mmax=100)
```

**Proposed:**
```julia
tol_adaptive = max(1e-13, 10 * noise_level)
fit_aaa(x, y; tol=tol_adaptive, mmax=100)
```

**Rationale:**
- Don't try to fit noise to machine precision!
- At noise=5e-2: use tol=5e-1 (loose fit, avoid overfitting)
- At noise=1e-8: use tol=1e-7 (tight fit, but not too tight)

**Expected Impact:** Could dramatically improve AAA performance at high noise

---

### Priority 2: Fourier-FFT Filter Adaptation
**Current:**
```julia
filter_frac = 0.4  # Fixed across all conditions
```

**Proposed:**
```julia
# More aggressive filtering for high orders and high noise
filter_frac = 0.6 / (1.0 + 0.15*order + 2.0*noise_level)
```

**Examples:**
- order=0, noise=1e-8: frac ≈ 0.60 (relaxed filtering)
- order=0, noise=5e-2: frac ≈ 0.55 (moderate)
- order=4, noise=1e-4: frac ≈ 0.40 (aggressive)
- order=7, noise=5e-2: frac ≈ 0.25 (very aggressive)

**Expected Impact:** Better noise handling at high orders

---

### Priority 3: Chebyshev Degree Adaptation
**Current:**
```python
deg = max(3, min(20, n_train - 1))
```

**Proposed:**
```python
# Adapt to noise level
if noise_level < 1e-6:
    max_deg = 30  # More detail for clean data
elif noise_level < 1e-3:
    max_deg = 20  # Current default
else:
    max_deg = 12  # More regularization for noisy data

deg = max(3, min(max_deg, n_train - 1))
```

**Expected Impact:** Better fit for low noise, more robustness for high noise

---

### Priority 4: Fourier Harmonics Adaptation
**Current:**
```python
M = max(1, min((n_train - 1) // 4, 25))
```

**Proposed:**
```python
# Similar logic to Fourier-FFT filter_frac
frac = 0.6 / (1.0 + 0.15*order + 2.0*noise_level)
M_max = int(frac * n_train / 2)
M = max(1, min(M_max, 25))
```

**Expected Impact:** Adaptive regularization like Fourier-FFT

---

## Testing Strategy

### Quick Test (1-2 hours)
Pick one method (recommend: **AAA tolerance adaptation**)
1. Modify code to use adaptive tolerance
2. Re-run on subset of conditions:
   - Orders: [0, 3, 5, 7]
   - Noise: [1e-8, 1e-4, 1e-2, 5e-2]
   - 3 trials each
   - Total: 4×4×3 = 48 runs
3. Compare nRMSE before/after
4. If improvement > 10%, proceed to full benchmark

### Full Test (4-8 hours)
1. Modify all 4 methods above
2. Re-run complete comprehensive benchmark (1,310 conditions × 3 trials)
3. Generate new tables and plots
4. Compare method rankings

### Analysis
- Plot nRMSE improvement vs (order, noise)
- Check if any conclusions/rankings change
- Document which adaptations helped most

---

## Expected Outcomes

### Best Case
- AAA-LowPrec becomes competitive (top 10)
- Fourier methods show 15-20% improvement at high orders
- Chebyshev shows improvement at noise extremes
- Paper conclusion: "Adaptive hyperparameters matter!"

### Realistic Case
- AAA-LowPrec improves but still worse than GP methods
- Fourier-FFT shows 5-10% improvement
- Chebyshev shows marginal gains
- Paper conclusion: "Some adaptation helpful, but GP still best"

### Worst Case
- No significant improvement
- Adaptive rules introduce new failure modes
- Paper conclusion: "Default hyperparameters are reasonable"

---

## Decision Point

**Should we optimize hyperparameters now or later?**

### Arguments for NOW
1. More scientifically rigorous
2. Could change conclusions (especially AAA)
3. Better story for paper: "We optimized fairly"
4. Only 4-8 hours of compute time

### Arguments for LATER (Future Work)
1. Faster to publication with current results
2. Hyperparameter optimization is a paper in itself
3. Current results already show clear GP dominance
4. Can cite as limitation and future work

### My Recommendation
**Compromise: Test AAA tolerance adaptation ONLY**
- Fastest to implement (one line change)
- Highest potential impact (AAA currently terrible)
- If it works: adds to paper, if not: no harm done
- 1-2 hours total (coding + running + analysis)

**Then decide:**
- If AAA improves dramatically → do full optimization
- If AAA still fails → proceed with paper as-is, note in limitations

---

## Next Steps

1. Review this analysis with user
2. Decide: optimize now vs later vs never
3. If optimize:
   - Start with AAA tolerance (quick win?)
   - Proceed to others if justified
4. If not optimize:
   - Document hyperparameters in paper
   - Note as limitation/future work
   - Proceed with current plots and tables
