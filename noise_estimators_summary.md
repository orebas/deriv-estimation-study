# Noise Estimators in the Repository

## Julia (`src/hyperparameter_selection.jl`)

### 1. `estimate_noise_diff2(y)`
- **Method**: 2nd-order finite differences + MAD
- **Formula**: d[i] = y[i] - 2y[i-1] + y[i-2], then σ̂ = MAD(d) / (0.6745 * √6)
- **Variance scaling**: Var(d) = 6σ² for i.i.d. Gaussian noise
- **Advantages**:
  - Robust to linear trends (constant and linear terms cancel out)
  - No external dependencies
  - Fast computation
- **Disadvantages**:
  - Still picks up signal curvature (2nd derivative of true signal)
  - Overestimates noise for smooth signals
- **Used by**: AAA-Adaptive-Diff2, Fourier-FFT-Adaptive
- **Reference**: Rice (1984) "Bandwidth choice for nonparametric regression"

### 2. `estimate_noise_wavelet(y; wt=wavelet(WT.db4))`
- **Method**: Wavelet decomposition + MAD on detail coefficients
- **Formula**: DWT(y) → extract finest-scale details → σ̂ = MAD(details) / 0.6745
- **Advantages**:
  - "Gold standard" method (Donoho-Johnstone 1994)
  - Better at separating noise from signal
  - Wavelet basis captures noise in high-frequency components
- **Disadvantages**:
  - Requires Wavelets.jl package
  - Requires padding to power of 2
  - More complex computation
- **Used by**: AAA-Adaptive-Wavelet, Dierckx-5, GSS
- **Reference**: Donoho & Johnstone (1994) "Ideal spatial adaptation by wavelet shrinkage"

## Python (`python/hyperparameters.py`)

### 3. `estimate_noise_wavelet(y, wavelet='db4')`
- **Method**: Same as Julia version, using PyWavelets
- **Formula**: pywt.wavedec(y, 'db4', level=1) → σ̂ = MAD(cD1) / 0.6745
- **Used by**: AAA-Python-Adaptive-Wavelet, AAA-JAX-Adaptive-Wavelet
- **Reference**: Same as Julia version

### 4. `estimate_noise_diff2(y)`
- **Method**: Same as Julia version
- **Formula**: Identical implementation
- **Used by**: AAA-Python-Adaptive-Diff2, AAA-JAX-Adaptive-Diff2

### 5. `estimate_noise_auto(y)`
- **Method**: Wrapper that chooses best available
- **Logic**: Uses wavelet if PyWavelets available, otherwise diff2
- **Used by**: Various adaptive methods as default

## Broken Estimators (Found in Old Code)

### 6. 1st-order differences (BROKEN - removed)
- **Method**: d[i] = y[i] - y[i-1], then σ̂ = MAD(d) / (0.6745 * √2)
- **Problem**: NOT robust to trends - captures signal derivative
- **Was used by**: Savitzky-Golay-Adaptive (now fixed), SavitzkyGolay_Adaptive_Python (now disabled)
- **Variance scaling**: Var(d) = 2σ² for i.i.d. Gaussian noise
- **Why broken**: For smooth signals, 1st differences contain large signal content (the actual derivative)

## Comparison on Test Data (sin(2πx) + 0.5sin(4πx), N=101)

| Noise Level | True σ | Diff1 (broken) | Diff2     | Wavelet   |
|-------------|--------|----------------|-----------|-----------|
| 1e-8        | 1e-8   | 0.0427 (BAD)   | 0.00298   | ?         |
| 1e-6        | 1e-6   | 0.0427 (BAD)   | 0.00298   | ?         |
| 1e-4        | 1e-4   | 0.0425 (BAD)   | 0.00279   | ?         |
| 1e-3        | 1e-3   | 0.0433 (BAD)   | 0.00210   | ?         |

**Notes**:
- Diff1 overestimates by 4000× - completely broken
- Diff2 overestimates by 3-300× - better but still problematic for very smooth signals
- Wavelet method untested but theoretically should be best

## Why All Estimators Struggle with Smooth ODE Signals

For smooth, low-noise signals like ODE solutions:
- **True noise**: σ = 1e-8 to 1e-3
- **Signal variation**: Much larger (derivatives are O(1))
- **Problem**: Any finite-difference based estimator captures signal + noise

Even 2nd-order differences pick up signal curvature:
```
d[i] = y[i] - 2y[i-1] + y[i-2]
     ≈ (h²/2)f''(x) + noise difference
```

For smooth f(x) with large f''(x), the signal term dominates.

## Recommendation

For the current test suite (smooth ODEs, N=101, low noise):
1. **Wavelet estimator** is theoretically best - should be tested
2. **Diff2 estimator** works reasonably well (14-20× better than diff1)
3. **Fixed parameters** may be more appropriate than adaptive for this specific problem class
