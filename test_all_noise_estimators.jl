"""
Test all noise estimators on smooth ODE-like signals.
"""

using Random
using Statistics
using Wavelets

# Implement all three estimators
function estimate_noise_diff1(y)
    dy = diff(y)
    return median(abs.(dy .- median(dy))) / (0.6745 * sqrt(2.0))
end

function estimate_noise_diff2(y)
    n = length(y)
    d = y[3:end] - 2.0 * y[2:end-1] + y[1:end-2]
    return median(abs.(d .- median(d))) / 0.6745 / sqrt(6.0)
end

function estimate_noise_wavelet(y)
    wt = wavelet(Wavelets.WT.db4)
    n = length(y)
    n_pow2 = 2^ceil(Int, log2(n))
    if n_pow2 > n
        y_padded = vcat(y, zeros(n_pow2 - n))
    else
        y_padded = y
    end

    decomp = dwt(y_padded, wt, 1)
    n_detail = n_pow2 ÷ 2
    detail = decomp[n_detail+1:end]
    n_detail_orig = min(n_detail, (n + 1) ÷ 2)
    detail_orig = detail[1:n_detail_orig]

    return median(abs.(detail_orig .- median(detail_orig))) / 0.6745
end

# Test functions
function test_sin(x)
    return sin.(2π * x) .+ 0.5 * sin.(4π * x)
end

function test_exp(x)
    return exp.(-x) .* cos.(4π * x)
end

function test_polynomial(x)
    return x.^3 - 2 * x.^2 .+ x
end

Random.seed!(12345)
n = 101
x = collect(range(0, 1, length=n))

test_cases = [
    ("Sin combo", test_sin(x)),
    ("Exp decay", test_exp(x)),
    ("Polynomial", test_polynomial(x)),
]

noise_levels = [1e-8, 1e-6, 1e-4, 1e-3]

println("Noise Estimator Comparison on Smooth Signals")
println("=" ^ 90)

for (name, y_true) in test_cases
    println("\nTest case: $name")
    println("-" ^ 90)

    for noise_std in noise_levels
        y_noisy = y_true .+ noise_std * randn(n)

        σ_diff1 = estimate_noise_diff1(y_noisy)
        σ_diff2 = estimate_noise_diff2(y_noisy)
        σ_wavelet = estimate_noise_wavelet(y_noisy)

        err1 = abs(σ_diff1 - noise_std)
        err2 = abs(σ_diff2 - noise_std)
        err_wav = abs(σ_wavelet - noise_std)

        println("\nNoise: σ = $noise_std")
        println("  Diff1:    σ̂ = $(round(σ_diff1, sigdigits=4))   error = $(round(err1, sigdigits=3))")
        println("  Diff2:    σ̂ = $(round(σ_diff2, sigdigits=4))   error = $(round(err2, sigdigits=3))")
        println("  Wavelet:  σ̂ = $(round(σ_wavelet, sigdigits=4))   error = $(round(err_wav, sigdigits=3))")

        # Find best
        errors = [err1, err2, err_wav]
        names = ["Diff1", "Diff2", "Wavelet"]
        best_idx = argmin(errors)

        println("  → BEST: $(names[best_idx]) ($(round(minimum(errors), sigdigits=3)) error)")
    end
end
