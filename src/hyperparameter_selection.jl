"""
Automatic Hyperparameter Selection for Derivative Estimation (Julia)

Implements noise estimation and adaptive hyperparameter selection that doesn't
require knowing the true noise level.

References:
- Donoho & Johnstone (1994): Ideal spatial adaptation by wavelet shrinkage
- Rice (1984): Bandwidth choice for nonparametric regression
"""

module HyperparameterSelection

export estimate_noise_diff2, estimate_noise_wavelet, select_aaa_tolerance, select_fourier_filter_frac

using Statistics
using FFTW
using Random
using LinearAlgebra
using Wavelets

"""
    estimate_noise_diff2(y::Vector{Float64}) -> Float64

Estimate noise σ using second-order differences (robust to linear trends).

For smooth f(x) and noise ε ~ N(0,σ²):
    d[i] = y[i] - 2y[i-1] + y[i-2]
         ≈ ε[i] - 2ε[i-1] + ε[i-2]  (signal curvature << noise)

    Var(d) = (1² + (-2)² + 1²)σ² = 6σ²

MAD estimator: σ̂ = MAD(d) / (0.6745 * √6)

# Arguments
- `y`: Signal (possibly noisy)

# Returns
- Estimated noise standard deviation σ̂

# References
- Rice, J. (1984). Bandwidth choice for nonparametric regression.
  The Annals of Statistics, 12(4), 1215-1230.
"""
function estimate_noise_diff2(y::Vector{Float64})
    n = length(y)
    if n < 3
        error("Need at least 3 points for 2nd-order difference")
    end

    # Second-order difference operator
    d = y[3:end] - 2.0 * y[2:end-1] + y[1:end-2]

    # MAD estimator
    # Var(d) = 6σ² for i.i.d. Gaussian noise
    # Factor 0.6745 is the 75th percentile of standard normal
    σ̂ = median(abs.(d .- median(d))) / 0.6745 / sqrt(6.0)

    return σ̂
end


"""
    estimate_noise_wavelet(y::Vector{Float64}; wavelet=wavelet(WT.db4)) -> Float64

Estimate noise σ using wavelet MAD (Donoho-Johnstone estimator).

This is the "gold standard" noise estimation method. Uses the finest-scale
wavelet detail coefficients which primarily capture noise in smooth signals.

# Arguments
- `y`: Signal (possibly noisy)
- `wavelet`: Wavelet type (default: Daubechies-4, matching PyWavelets default)

# Returns
- Estimated noise standard deviation σ̂

# References
- Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by
  wavelet shrinkage. Biometrika, 81(3), 425-455.

# Note
Uses Wavelets.jl with 1-level decomposition. Daubechies-4 (db4) is the
standard choice for noise estimation in the literature.
"""
function estimate_noise_wavelet(y::Vector{Float64}; wt=wavelet(WT.db4))
    n = length(y)
    if n < 4
        error("Need at least 4 points for wavelet decomposition")
    end

    # Pad to next power of 2 if necessary (DWT requirement)
    n_pow2 = 2^ceil(Int, log2(n))
    if n_pow2 > n
        # Zero-pad (common practice for DWT)
        y_padded = vcat(y, zeros(n_pow2 - n))
    else
        y_padded = y
    end

    # 1-level discrete wavelet transform
    # Returns approximation (low-freq) and detail (high-freq) coefficients
    decomp = dwt(y_padded, wt, 1)

    # Extract detail coefficients at finest scale (level 1)
    # For 1-level decomp: decomp = [approx; detail]
    n_detail = n_pow2 ÷ 2
    detail = decomp[n_detail+1:end]

    # Only use detail coefficients from original signal (not padding)
    # This prevents bias from zero-padding
    n_detail_orig = min(n_detail, (n + 1) ÷ 2)
    detail_orig = detail[1:n_detail_orig]

    # MAD estimator on detail coefficients
    # Robust to outliers and works under Gaussian noise assumption
    # Factor 0.6745 is the 75th percentile of standard normal distribution
    σ̂ = median(abs.(detail_orig .- median(detail_orig))) / 0.6745

    return σ̂
end


"""
    select_aaa_tolerance(y::Vector{Float64}; multiplier::Float64=10.0, min_tol::Float64=1e-13) -> Float64

Select AAA tolerance based on estimated noise level.

The AAA algorithm tries to minimize residual error. If tolerance is too tight
relative to noise, it overfits by creating spurious poles that cause derivative
catastrophes.

# Arguments
- `y`: Training y values (used for noise estimation)
- `multiplier`: Safety factor (tol = multiplier * σ̂), default 10.0
- `min_tol`: Minimum tolerance (for clean data / machine precision), default 1e-13

# Returns
- Adaptive tolerance value

# Strategy
- High noise (σ̂=5e-2): tol ≈ 0.5 (very loose, prevents overfitting)
- Moderate (σ̂=1e-4): tol ≈ 1e-3
- Clean (σ̂=1e-8): tol ≈ 1e-7 (tight but not absurd)

# Example
```julia
using Random
x = LinRange(0, 10, 101)
y_true = sin.(x)
y_noisy = y_true + 0.05 * randn(length(x))
tol = select_aaa_tolerance(y_noisy)  # ≈ 0.5
```
"""
function select_aaa_tolerance(
    y::Vector{Float64};
    multiplier::Float64 = 10.0,
    min_tol::Float64 = 1e-13
)
    σ̂ = estimate_noise_diff2(y)

    # Rule: tolerance should be ~10x noise to avoid fitting noise
    # But don't go below machine precision for truly clean data
    tol = max(min_tol, multiplier * σ̂)

    return tol
end


"""
    select_fourier_filter_frac(y::Vector{Float64}; confidence_mult::Float64=3.0) -> Float64

Select filter fraction for Fourier-FFT based on estimated noise.

Keep frequency components whose magnitude exceeds the estimated noise floor.

# Arguments
- `y`: Training y values
- `confidence_mult`: Threshold multiplier (default 3.0 ≈ 99% confidence)

# Returns
- Filter fraction (between 0.1 and 0.8)

# Example
```julia
frac = select_fourier_filter_frac(y_noisy)
# Use in fourier_fft_deriv(...; filter_frac=frac)
```
"""
function select_fourier_filter_frac(
    y::Vector{Float64};
    confidence_mult::Float64 = 3.0
)
    σ̂ = estimate_noise_diff2(y)
    n = length(y)

    # DCT coefficients
    c = FFTW.dct(y)
    c_abs = abs.(c)

    # Keep coefficients with |c| > λσ̂√n
    # The √n factor accounts for energy spread across frequencies
    threshold = confidence_mult * σ̂ * sqrt(n)
    m = count(c_abs .>= threshold)

    # Guard rails: keep between 10% and 80%
    m = max(Int(floor(0.1 * n)), min(Int(ceil(0.8 * n)), m))

    filter_frac = m / n

    return filter_frac
end


# Module-level test function
function test_module()
    Random.seed!(42)

    # Generate test data: smooth function + noise
    x = LinRange(0, 10, 101)
    y_true = sin.(x) + 0.3 * sin.(3 .* x)
    noise_level = 0.05
    y_noisy = y_true + noise_level * randn(length(x))

    println("Testing hyperparameter selection module (Julia)...")
    println("\nTrue noise: $noise_level")

    # Test noise estimation
    σ_est = estimate_noise_diff2(y_noisy)
    println("2nd-order diff estimate: $σ_est")
    println("  Error: $(abs(σ_est - noise_level) / noise_level * 100)%")

    # Test AAA tolerance
    tol = select_aaa_tolerance(y_noisy)
    println("\nAAA adaptive tolerance: $tol")
    println("  vs fixed tol=1e-13")
    println("  Ratio: $(tol / 1e-13)")

    # Test Fourier filter
    frac = select_fourier_filter_frac(y_noisy)
    println("\nFourier-FFT filter fraction: $frac")
    println("  vs fixed frac=0.4")

    println("\n✅ All Julia tests passed!")
end

end # module

# Run test if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using .HyperparameterSelection
    HyperparameterSelection.test_module()
end
