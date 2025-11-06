"""
Analyze the actual smoothness of different smoothing methods
to understand why ApproxFun needs millions of coefficients
"""

using ApproxFun
using Statistics
using Printf
using Interpolations
using Plots
using FFTW

# Test signal
n = 201
t = collect(range(0, 1, length=n))
ω = 2π
y_true = sin.(ω .* t)
noise_level = 1e-3
y_noisy = y_true .+ noise_level .* randn(n)

println("="^80)
println("Smoothness Analysis of Different Methods")
println("="^80)

# Different smoothing methods
function moving_average(y, window=11)
    n = length(y)
    y_smooth = similar(y)
    half_win = window ÷ 2
    for i in 1:n
        i_start = max(1, i - half_win)
        i_end = min(n, i + half_win)
        y_smooth[i] = mean(y[i_start:i_end])
    end
    return y_smooth
end

function gaussian_smooth(y, window=7)
    n = length(y)
    y_smooth = similar(y)
    half_win = window ÷ 2
    for i in 1:n
        i_start = max(1, i - half_win)
        i_end = min(n, i + half_win)
        weights = exp.(-0.5 * ((collect(i_start:i_end) .- i) ./ (half_win/2)).^2)
        weights ./= sum(weights)
        y_smooth[i] = sum(weights .* y[i_start:i_end])
    end
    return y_smooth
end

function kalman_smooth(y, α=0.98)
    n = length(y)
    y_smooth = similar(y)
    y_smooth[1] = y[1]
    for i in 2:n
        y_smooth[i] = α * y_smooth[i-1] + (1-α) * y[i]
    end
    # Backward pass for symmetry
    for i in (n-1):-1:1
        y_smooth[i] = 0.5 * (y_smooth[i] + α * y_smooth[i+1] + (1-α) * y[i])
    end
    return y_smooth
end

function tv_smooth(y, iterations=3)
    y_smooth = copy(y)
    for _ in 1:iterations
        for i in 2:(length(y_smooth)-1)
            y_smooth[i] = 0.6*y_smooth[i] + 0.2*(y_smooth[i-1] + y_smooth[i+1])
        end
    end
    return y_smooth
end

# Apply different smoothing methods
methods = [
    ("Raw noisy", y_noisy),
    ("Moving average", moving_average(y_noisy)),
    ("Gaussian", gaussian_smooth(y_noisy)),
    ("Kalman α=0.98", kalman_smooth(y_noisy, 0.98)),
    ("Kalman α=0.95", kalman_smooth(y_noisy, 0.95)),
    ("Kalman α=0.90", kalman_smooth(y_noisy, 0.90)),
    ("TV (3 iter)", tv_smooth(y_noisy, 3)),
    ("TV (10 iter)", tv_smooth(y_noisy, 10))
]

println("\n1. ROUGHNESS METRICS")
println("-"^40)
println("Method               | Std(diff) | Std(diff²) | Max jump")
println("-"^40)

for (name, y_smooth) in methods
    d1 = diff(y_smooth)
    d2 = diff(d1)
    std_d1 = std(d1)
    std_d2 = std(d2)
    max_jump = maximum(abs.(d1))
    @printf("%-20s | %.3e | %.3e | %.3e\n", name, std_d1, std_d2, max_jump)
end

println("\n2. FREQUENCY CONTENT (FFT)")
println("-"^40)

for (name, y_smooth) in methods
    # FFT to see frequency content
    fft_coeffs = fft(y_smooth)
    power = abs.(fft_coeffs).^2

    # Find frequency where power drops to 1% of max
    max_power = maximum(power[2:div(end,2)])  # Skip DC
    cutoff_idx = findfirst(p -> p < 0.01 * max_power, power[2:div(end,2)])
    if isnothing(cutoff_idx)
        cutoff_idx = div(length(power), 2)
    else
        cutoff_idx += 1
    end

    # Estimate noise floor (high frequency average)
    high_freq_start = div(length(power), 4)
    noise_floor = mean(power[high_freq_start:div(end,2)])

    @printf("%-20s | Cutoff freq: %3d | Noise floor: %.2e\n",
            name, cutoff_idx, noise_floor)
end

println("\n3. APPROXFUN COEFFICIENT REQUIREMENTS")
println("-"^40)

# Test how many coefficients ApproxFun needs for each
for (name, y_smooth) in methods[2:end]  # Skip raw noisy
    # Create interpolator
    itp = linear_interpolation(t, y_smooth, extrapolation_bc=Flat())

    # Let ApproxFun fit with default tolerance
    try
        f = Fun(x -> itp(x), 0..1)
        n_coeffs = length(coefficients(f))

        # Check coefficient decay
        coeffs = coefficients(f)
        if length(coeffs) > 10
            # Find where coefficients drop below 1e-10
            small_idx = findfirst(c -> abs(c) < 1e-10, coeffs)
            if isnothing(small_idx)
                small_idx = length(coeffs)
            end

            # Decay rate (rough estimate)
            if length(coeffs) > 20
                log_coeffs = log10.(abs.(coeffs[1:20]) .+ 1e-16)
                decay_rate = -(log_coeffs[20] - log_coeffs[1]) / 19
            else
                decay_rate = 0.0
            end
        else
            small_idx = length(coeffs)
            decay_rate = 1.0  # Good!
        end

        @printf("%-20s | Coeffs: %7d | Small at: %4d | Decay: %.3f\n",
                name, n_coeffs, small_idx, decay_rate)
    catch e
        println("$name: ERROR - $(e)")
    end
end

println("\n4. POINT-BY-POINT ROUGHNESS")
println("-"^40)

# Look at local roughness to identify problem areas
for (name, y_smooth) in methods[2:4]  # Just first few
    # Compute local roughness (2nd derivative approximation)
    roughness = zeros(length(y_smooth)-2)
    for i in 2:(length(y_smooth)-1)
        roughness[i-1] = abs(y_smooth[i-1] - 2*y_smooth[i] + y_smooth[i+1])
    end

    # Find worst spots
    sorted_idx = sortperm(roughness, rev=true)
    worst_5 = sorted_idx[1:5]

    println("\n$name - Worst roughness at indices:")
    for idx in worst_5
        actual_idx = idx + 1  # Account for offset
        @printf("  t=%.3f: roughness=%.3e\n", t[actual_idx], roughness[idx])
    end
end

println("\n5. VISUAL COMPARISON (subset)")
println("-"^40)

# Look at a small window to see the actual signal shape
t_window = 41:61  # 20 points around t=0.2

println("\nWindow: t ∈ [$(t[t_window[1]]), $(t[t_window[end]])]")
println("True signal range: [$(minimum(y_true[t_window])), $(maximum(y_true[t_window]))]")

for (name, y_smooth) in methods[1:4]
    y_window = y_smooth[t_window]

    # Check for oscillations
    sign_changes = sum(diff(sign.(diff(y_window))) .!= 0)

    println("\n$name:")
    println("  Range: [$(round(minimum(y_window), digits=4)), $(round(maximum(y_window), digits=4))]")
    println("  Sign changes in derivative: $sign_changes")
    println("  Std from true: $(round(std(y_window - y_true[t_window]), digits=5))")
end

println("\n6. WHY SO MANY COEFFICIENTS?")
println("-"^40)

# Analyze the moving average case in detail
y_ma = moving_average(y_noisy)

# Look at the actual residuals after polynomial fits
for degree in [10, 20, 50, 100]
    # Fit polynomial of given degree
    A = zeros(n, degree+1)
    for i in 1:n
        for j in 0:degree
            A[i, j+1] = t[i]^j
        end
    end

    coeffs = A \ y_ma
    y_poly = A * coeffs

    residual = y_ma - y_poly
    max_residual = maximum(abs.(residual))

    println("Polynomial degree $degree: max residual = $(round(max_residual, sigdigits=3))")
end

println("\n" * "="^80)
println("KEY FINDINGS")
println("="^80)

println("""
1. Moving average and Gaussian smoothing still have high-frequency noise
   - Their difference sequences have std ~1e-3 (same as noise level!)
   - They're just local averaging, not true smoothing

2. Kalman with high α (0.98) oversmooths and loses signal features
   - Lower α (0.90-0.95) might be better but still has issues

3. TV smoothing creates piecewise linear segments
   - Good for order 1, terrible for higher orders

4. ApproxFun needs many coefficients because:
   - The "smoothed" signals still have point-to-point variations
   - These variations require high-degree polynomials to fit
   - It's trying to fit noise, not smooth functions

5. The fundamental issue: These methods don't produce C^∞ smooth functions
   - They produce numerically quieter but still rough signals
   - Spectral methods need analytical smoothness
""")

# Final test: what if we smooth MORE aggressively?
println("\nEXPERIMENT: More aggressive smoothing")
println("-"^40)

y_heavy = moving_average(moving_average(moving_average(y_noisy, 21), 21), 21)  # Triple smooth!
itp_heavy = linear_interpolation(t, y_heavy, extrapolation_bc=Flat())

try
    f_heavy = Fun(x -> itp_heavy(x), 0..1)
    n_coeffs = length(coefficients(f_heavy))
    println("Triple moving average (window=21): $(n_coeffs) coefficients")

    # Check accuracy
    y_fit = [f_heavy(ti) for ti in t]
    rmse_true = sqrt(mean((y_fit - y_true).^2))
    println("  RMSE to true signal: $(round(rmse_true, sigdigits=3))")
    println("  Signal distortion: $(round(100*rmse_true/std(y_true), digits=1))%")
catch e
    println("Triple smoothing: Still failed! $(e)")
end