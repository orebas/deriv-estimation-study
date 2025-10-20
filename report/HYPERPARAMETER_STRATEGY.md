# Automatic Hyperparameter Selection Strategy
**Synthesized from o3 + Gemini Pro recommendations**

---

## Key Insight: Two Classes of Methods

### Class A: Linear Smoothers (NO σ NEEDED!)
**Methods:** Chebyshev, Fourier least-squares
- Can use **GCV** or **AICc** directly
- No need to estimate noise level!
- Self-contained, optimal in expectation

### Class B: Non-linear or Spectral Methods (NEED σ)
**Methods:** AAA, Fourier-FFT, SpectralTaper
- Require noise level estimate for tolerance/threshold
- Use **Wavelet MAD** (gold standard) or **2nd-order differences** (lightweight)

---

## Recommended Implementations

### 1. Chebyshev Polynomial - AICc Selection ⭐

**Why AICc over GCV:**
- One-line formula, familiar to readers
- Small-sample correction (n=101 is moderate)
- Both give similar results, AICc easier to explain

**Algorithm:**
```python
def select_chebyshev_degree(x_train, y_train, max_degree=30):
    """Select optimal degree via AICc."""
    n = len(x_train)
    tmin, tmax = x_train.min(), x_train.max()

    best_aicc = np.inf
    best_deg = 3

    for deg in range(3, min(max_degree + 1, n - 2)):
        # Fit Chebyshev polynomial
        poly = Chebyshev.fit(x_train, y_train, deg=deg, domain=[tmin, tmax])
        y_pred = poly(x_train)
        rss = np.sum((y_train - y_pred) ** 2)

        # AICc formula
        p = deg + 1  # number of parameters
        if n - p - 2 > 0:
            aicc = n * np.log(rss / n) + 2*p + 2*p*(p+1)/(n - p - 2)
        else:
            aicc = np.inf

        if aicc < best_aicc:
            best_aicc = aicc
            best_deg = deg

    return best_deg

# Usage:
optimal_deg = select_chebyshev_degree(x_train, y_train)
poly = Chebyshev.fit(x_train, y_train, deg=optimal_deg, domain=[tmin, tmax])
```

**Expected behavior:**
- Low noise: picks deg=15-25 (more detail)
- High noise: picks deg=8-15 (more regularization)
- Automatic, no tuning needed!

---

### 2. Fourier Series - GCV Selection ⭐

**Why GCV:**
- Standard for periodic/trigonometric smoothers
- Easy trace formula for linear system
- No σ estimation needed

**Algorithm:**
```python
def select_fourier_harmonics(x_train, y_train, max_harmonics=25):
    """Select optimal number of harmonics via GCV."""
    n = len(x_train)
    tmin, tmax = x_train.min(), x_train.max()
    T = tmax - tmin
    omega = 2.0 * np.pi / T

    best_gcv = np.inf
    best_M = 1

    for M in range(1, max_harmonics + 1):
        # Build design matrix
        phi_cols = [np.ones(n)]
        ang = x_train - tmin
        for k in range(1, M + 1):
            kω = k * omega
            phi_cols.append(np.cos(kω * ang))
            phi_cols.append(np.sin(kω * ang))
        Phi = np.vstack(phi_cols).T  # shape (n, 2M+1)

        # Least squares fit
        coef, residuals, rank, s = np.linalg.lstsq(Phi, y_train, rcond=None)
        y_pred = Phi @ coef
        rss = np.sum((y_train - y_pred) ** 2)

        # GCV formula
        df = 2*M + 1  # degrees of freedom
        if n - df > 0:
            gcv = (rss / n) / ((1 - df/n) ** 2)
        else:
            gcv = np.inf

        if gcv < best_gcv:
            best_gcv = gcv
            best_M = M

    return best_M

# Usage:
optimal_M = select_fourier_harmonics(x_train, y_train)
# Then fit with optimal_M harmonics
```

---

### 3. AAA - Wavelet-Based Tolerance ⭐⭐⭐

**Critical fix:** AAA currently uses tol=1e-13 always → overfits to noise!

**Strategy:**
1. Estimate σ using Wavelet MAD (gold standard)
2. Set tol = max(1e-13, 10 * σ_hat)

**Algorithm:**
```python
import pywt

def estimate_noise_wavelet(y):
    """Estimate noise σ using wavelet MAD (Donoho-Johnstone)."""
    # Decompose using Daubechies 4 wavelet
    coeffs = pywt.wavedec(y, 'db4', level=1)
    cD1 = coeffs[-1]  # Detail coefficients at finest scale

    # MAD estimator
    sigma_hat = np.median(np.abs(cD1)) / 0.6745
    return sigma_hat

def select_aaa_tolerance(y_train):
    """Select AAA tolerance based on estimated noise."""
    sigma_hat = estimate_noise_wavelet(y_train)

    # Rule: tol should be ~10x noise level to avoid fitting noise
    # But don't go below machine precision for clean data
    tol = max(1e-13, 10.0 * sigma_hat)

    return tol

# Usage in Julia AAA:
# tol_adaptive = select_aaa_tolerance(y_train)
# fitted_func = fit_aaa(x, y; tol=tol_adaptive, mmax=100)
```

**Alternative (no wavelet dependency):**
```python
def estimate_noise_diff2(y):
    """Estimate σ using 2nd-order differences (Gemini Pro recommendation)."""
    # Second-order difference operator: yᵢ - 2yᵢ₋₁ + yᵢ₋₂
    d = y[2:] - 2*y[1:-1] + y[:-2]

    # Var(noise in d) = 6σ² for i.i.d. Gaussian
    # MAD estimator (robust to outliers)
    sigma_hat = np.median(np.abs(d)) / 0.6745 / np.sqrt(6)

    return sigma_hat
```

**Expected impact:**
- At noise=5e-2: tol ≈ 5e-1 (very loose, prevents overfitting)
- At noise=1e-4: tol ≈ 1e-3 (moderate)
- At noise=1e-8: tol ≈ 1e-7 (tight but not absurd)

**This single change could make AAA-LowPrec competitive!**

---

### 4. Fourier-FFT - SURE Optimal Threshold ⭐

**Strategy:** Use SURE to find optimal soft-threshold, convert to filter_frac

**Algorithm (o3 recommendation):**
```python
def select_fourier_fft_filter(y_train):
    """Select filter_frac via SURE (Donoho-Johnstone)."""
    from scipy.fftpack import dct, idct

    n = len(y_train)

    # DCT coefficients
    c = dct(y_train, norm='ortho')
    c_abs_sorted = np.sort(np.abs(c))[::-1]  # descending

    # SURE formula (simplified for hard thresholding)
    def sure_risk(threshold):
        m = np.sum(np.abs(c) >= threshold)  # kept coefficients
        risk = n - 2*m + np.sum(c[np.abs(c) < threshold]**2)
        return risk

    # Grid search over sorted coefficient magnitudes
    min_risk = np.inf
    best_m = n // 2

    for i in range(10, n-10):  # avoid extremes
        t = c_abs_sorted[i]
        risk = sure_risk(t)
        if risk < min_risk:
            min_risk = risk
            best_m = i

    filter_frac = best_m / n
    return filter_frac

# Alternative simpler approach (Gemini Pro):
def select_filter_frac_simple(y_train):
    """Estimate noise, keep coefficients above 3σ."""
    sigma_hat = estimate_noise_wavelet(y_train)

    from scipy.fftpack import dct
    c = dct(y_train, norm='ortho')

    # Keep coefficients with |c| > 3σ (99% confidence)
    threshold = 3.0 * sigma_hat * np.sqrt(len(y_train))
    m = np.sum(np.abs(c) >= threshold)

    filter_frac = m / len(y_train)
    return max(0.1, min(0.8, filter_frac))  # guard rails
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. **AAA tolerance adaptation** (highest impact expected)
   - Add wavelet MAD estimator
   - Modify Julia AAA call to use adaptive tolerance
   - Test on subset: orders [0,3,5,7] × noise [1e-8, 1e-4, 1e-2, 5e-2]

### Phase 2: Linear Smoothers (2-3 hours)
2. **Chebyshev AICc** (well-established, easy)
3. **Fourier GCV** (similar to Chebyshev)

### Phase 3: Advanced (if Phase 1-2 show improvement)
4. **Fourier-FFT SURE** (more complex, but elegant)

---

## Validation Strategy

**For each adapted method:**
1. Run on 10 random test cases (different orders/noise)
2. Compare selected hyperparameter vs fixed value
3. Check if nRMSE improves on average
4. Look for pathological cases (did auto-selection fail anywhere?)

**Success criteria:**
- Average nRMSE improvement > 5%
- No catastrophic failures (nRMSE > 10x worse)
- Selected hyperparameters are "reasonable" (can plot them vs noise level)

**Diagnostic plots:**
- Selected degree/harmonics/tolerance vs noise level
- Show that higher noise → more regularization (as expected)
- AICc/GCV curve showing minimum (should be clear, not flat)

---

## Code Organization

```
src/
  hyperparameter_selection.jl  # Julia implementations

python/
  hyperparameters.py           # Python implementations

report/
  hyperparameter_validation.py # Testing script
  HYPERPARAMETER_RESULTS.md    # Results documentation
```

---

## Paper Narrative

**Methods section (1 paragraph per method):**

> "For Chebyshev polynomials, we select the degree automatically via the corrected Akaike Information Criterion (AICc), scanning degrees from 3 to 30 and selecting the minimum. This approach balances fit quality against model complexity without requiring a priori knowledge of the noise level.
>
> For Fourier series, we employ Generalized Cross-Validation (GCV) to select the number of harmonics. The GCV score approximates leave-one-out cross-validation error efficiently via the trace formula.
>
> For AAA rational approximation, we estimate the noise standard deviation σ̂ using the Median Absolute Deviation of wavelet detail coefficients at the finest scale (Donoho & Johnstone, 1994), then set the tolerance parameter as tol = max(10⁻¹³, 10σ̂). This prevents the algorithm from fitting noise as signal while maintaining high precision for clean data.
>
> For FFT-based spectral methods, we apply Stein's Unbiased Risk Estimate (SURE) to determine the optimal frequency cutoff, automatically balancing spectral resolution against noise amplification."

**Results section:**
- Table comparing fixed vs adaptive hyperparameters
- Show that adaptive selection improves robustness across noise regimes
- Note: GP methods still optimize via MLE, so comparison is now "fairer"

---

## Expected Outcomes

### Conservative Estimate
- AAA: 20-30% improvement at high noise, ~5% overall
- Chebyshev: 5-10% improvement (already had some regularization)
- Fourier: 5-10% improvement
- Story: "Automatic selection matters, but GP still best"

### Optimistic Estimate
- AAA: 50%+ improvement, becomes top-10 method
- Chebyshev: 15-20% improvement
- Fourier: 15-20% improvement
- Story: "With proper hyperparameter selection, classical methods competitive with GP"

### Worst Case
- Minimal improvement or even slight degradation
- Auto-selection adds complexity without benefit
- Story: "Fixed hyperparameters are surprisingly robust; future work could explore per-condition tuning"

---

## Next Steps

**DECISION POINT:** Should we implement this now?

**Arguments FOR:**
1. Makes benchmark "fair" - all methods optimize their parameters
2. Likely significant improvement (especially AAA)
3. Well-established techniques (not novel, just proper application)
4. 4-8 hours total work (doable in one session)

**Arguments AGAINST:**
1. Current results already show clear trends
2. Could be separate paper on hyperparameter selection
3. Adds complexity to reproducibility
4. May not change top method rankings (GP likely still wins)

**My recommendation:**
- **DO Phase 1** (AAA tolerance) - 1-2 hours, highest potential impact
- **IF AAA improves significantly:** Continue with Phase 2-3
- **IF AAA shows minimal improvement:** Proceed with current results, document as future work

What do you think?
