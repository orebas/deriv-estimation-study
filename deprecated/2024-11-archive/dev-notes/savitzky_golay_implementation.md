# Savitzky-Golay Implementation in Julia

## Package Information

**We do NOT use an external package** - this is a custom implementation in `methods/julia/filtering/filters.jl`.

### Dependencies:
- `LinearAlgebra` (Julia stdlib) - for matrix operations
- `HyperparameterSelection` (our module) - for noise estimation

## Core Functions

### 1. `savitzky_golay_coeffs(window, polyorder, deriv_order=0)`

Computes filter coefficients via least-squares polynomial fitting.

**Parameters:**
- `window::Int` - Window size (must be odd, e.g., 7, 15, 21)
- `polyorder::Int` - Polynomial degree (e.g., 3, 5, 7, 11)
- `deriv_order::Int` - Derivative order to compute (0=smoothing, 1=1st deriv, etc.)

**Constraints:**
- `window` must be odd
- `window > polyorder`
- `polyorder >= deriv_order`

**Returns:**
- Vector of filter coefficients (length = window)

**Algorithm:**
1. Build Vandermonde matrix A over positions [-h, ..., h] where h = window÷2
2. Solve least-squares: `(A'A)c = e_k` where e_k selects the k-th coefficient
3. Compute filter weights: `w = A * c * k!` (factorial for derivative scaling)

**Mathematical basis:**
```
Fits polynomial: p(x) = c₀ + c₁x + c₂x² + ... + cₚxᵖ
k-th derivative at x=0: p⁽ᵏ⁾(0) = k! * cₖ
```

### 2. `apply_savitzky_golay_filter(y, window, polyorder, deriv_order, dx)`

Applies S-G filter to data.

**Parameters:**
- `y::Vector{Float64}` - Input data
- `window::Int` - Window size
- `polyorder::Int` - Polynomial order
- `deriv_order::Int` - Derivative order
- `dx::Float64` - Grid spacing (for scaling derivatives)

**Returns:**
- Vector of derivative values (NaN at boundaries)

**Algorithm:**
1. Get filter coefficients
2. For each interior point i:
   - Extract window: `y[(i-h):(i+h)]`
   - Apply filter: `dot(weights, window_data)`
   - Scale by `dx^deriv_order` (converts from unit spacing to actual spacing)
3. Leave boundary points as NaN

**Boundary handling:**
- Points within `h` of edges are set to NaN
- These are filled via linear interpolation to evaluation points

## Two Variants

### 3. `evaluate_savitzky_golay_fixed(x, y, x_eval, orders; params=Dict())`

**Fixed-parameter variant** optimized for N≈101-251.

**Parameters:**
```julia
params = Dict(
    :sg_window => 15,      # Default: 15 (adjustable)
    :sg_polyorder => 7     # Default: 7 (adjustable)
)
```

**Default settings:**
- `window = 15` - Balanced for N=101-251
- `polyorder = 7` - Supports up to 7th derivative

**Design rationale:**
- Fixed parameters avoid overfitting to noise estimates
- w=15 empirically optimal for N≈101
- Needs adjustment for N=251 (see below)

### 4. `evaluate_savitzky_golay_adaptive(x, y, x_eval, orders; params=Dict())`

**Adaptive variant** with noise-dependent window sizing.

**Parameters:**
```julia
params = Dict()  # No user parameters - all automatic
```

**Automatic parameter selection:**

#### Window size formula:
```julia
w* = 2⌊h*⌋ + 1
h* = c_p,r × (σ̂²/ρ̂²)^(1/(2p+3))
```

where:
- `σ̂` = estimated noise (wavelet MAD)
- `ρ̂` = estimated roughness (4th-order differences)
- `c_p,r` = calibration constant (depends on poly order and deriv order)

#### Polynomial order tiers:
```julia
p = if order <= 3
        7      # Low orders: p=7
    elseif order <= 5
        9      # Mid orders: p=9
    else
        11     # High orders: p=11
    end
```

#### Calibration constants `c_p,r`:
```julia
c_pr = Dict(
    (7, 0) => 1.0,  (7, 1) => 1.1,  (7, 2) => 1.2,  (7, 3) => 1.3,
    (9, 0) => 1.0,  (9, 1) => 1.1,  (9, 2) => 1.2,  (9, 3) => 1.3,
    (9, 4) => 1.4,  (9, 5) => 1.5,
    (11, 0) => 1.0, (11, 1) => 1.1, (11, 2) => 1.2, (11, 3) => 1.3,
    (11, 4) => 1.4, (11, 5) => 1.5, (11, 6) => 1.6, (11, 7) => 1.7,
)
```

**Fallback:** `c = 1.0 + 0.1 × order` if (p, order) not in table

#### Noise estimation:
```julia
σ̂ = HyperparameterSelection.estimate_noise_wavelet(y)
```
Uses Daubechies-4 wavelet decomposition with MAD estimator.

#### Roughness estimation:
```julia
d4 = y
for _ in 1:4
    d4 = diff(d4)  # 4th-order differences
end
ρ̂ = √(mean(d4²)) / (dx⁴ + ε)
```

#### Window constraints:
```julia
max_window = max(5, n÷3)    # Cap at 1/3 of data size
w_ideal = Int(2⌊h*/dx⌋ + 1)  # Convert from continuous to discrete
w = max(w_ideal, p + 3)      # At least p+3 for numerical stability
w = min(w, max_window)       # Apply cap
w = min(w, n)                # Can't exceed data size
w = isodd(w) ? w : w-1       # Ensure odd
```

## Interpolation to Evaluation Points

Both methods compute derivatives at training points `x`, then interpolate to `x_eval`:

**Method:** Linear interpolation
```julia
for (i, xi) in enumerate(x_eval)
    if xi < x_valid[1] || xi > x_valid[end]
        # Flat extrapolation (use nearest value)
        deriv_eval[i] = xi < x_valid[1] ? deriv_valid[1] : deriv_valid[end]
    else
        # Linear interpolation
        idx = searchsortedfirst(x_valid, xi)
        t = (xi - x_valid[idx-1]) / (x_valid[idx] - x_valid[idx-1])
        deriv_eval[i] = (1-t) * deriv_valid[idx-1] + t * deriv_valid[idx]
    end
end
```

## Recommended Settings by Data Size

Based on empirical testing:

| N   | Method   | Window | Polyorder | Notes |
|-----|----------|--------|-----------|-------|
| 51  | Fixed    | 7-9    | 7         | ~15% of N |
| 101 | Fixed    | 15     | 7         | ~15% of N (current default) |
| 251 | Fixed    | 35-39  | 7         | ~15% of N |
| 251 | Adaptive | Auto   | 7/9/11    | Should work (w≈45-51 expected) |

## Key Implementation Details

1. **Unit spacing assumption:** Coefficients computed assuming integer positions, then scaled by `dx^deriv_order`

2. **Boundary treatment:** Interior-only filtering + linear interpolation (not polynomial extrapolation)

3. **Factorial scaling:** Derivatives scaled by `k!` to convert polynomial coefficients to actual derivatives

4. **Numerical stability:** Window must be > polyorder to ensure well-conditioned least-squares

5. **Variance growth:** Variance ∝ 1/w^(2r+1), so higher derivatives need larger windows (adaptive method handles this)

## Known Limitations

1. **Assumes uniform spacing:** `dx_std/dx_mean > 0.05` triggers warning

2. **Adaptive fails for smooth+low-noise at small N:**
   - N=101: Recommends w=33 vs optimal w=15 (oversmoothing)
   - N=251: Should work (recommends w=45-51 ≈ optimal)

3. **No weighting:** All points in window equally weighted (unlike LOESS)

4. **Global polynomial order:** Same `p` for all windows (not locally adaptive)

## Comparison to scipy.signal.savgol_filter

Our implementation matches scipy's algorithm but differs in:
- **No built-in mode parameter:** We handle boundaries via NaN + interpolation
- **Derivative scaling:** We scale by `dx^deriv_order` (scipy requires `delta` parameter)
- **No Cython optimization:** Pure Julia (still fast)

## References

1. **Original paper:** Savitzky & Golay (1964), Analytical Chemistry
2. **Least-squares derivation:** Schafer (2011), IEEE Signal Processing Magazine
3. **Asymptotic theory:** Rice (1984), Annals of Statistics
4. **Noise estimation:** Donoho & Johnstone (1994), Biometrika
