"""
Test ApproxFun extension for PyNumDiff methods
"""

using ApproxFun
using Statistics
using Printf

# Load the extension module
include("methods/julia/extensions/approxfun_extension.jl")

println("="^80)
println("Testing ApproxFun Extension for Limited-Order Methods")
println("="^80)

# Test signal: sin(2πt) with noise
n = 101
t = collect(range(0, 1, length=n))
ω = 2π
y_true = sin.(ω .* t)
noise_level = 1e-3
y_noisy = y_true .+ noise_level .* randn(n)

# Simulate PyNumDiff output (smoothed signal)
# For testing, we'll use a simple moving average as "PyNumDiff output"
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

y_smooth = moving_average(y_noisy, 11)

println("\nTest Data:")
println("  Points: $n")
println("  Domain: [0, 1]")
println("  Signal: sin(2πt)")
println("  Noise: $(noise_level)")
println("  Smoothing: 11-point moving average")

# Ground truth derivatives
function ground_truth(order, t_val)
    if order == 0
        return sin(ω * t_val)
    else
        phase = order % 4
        if phase == 1
            return ω^order * cos(ω * t_val)
        elseif phase == 2
            return -ω^order * sin(ω * t_val)
        elseif phase == 3
            return -ω^order * cos(ω * t_val)
        else  # phase == 0
            return ω^order * sin(ω * t_val)
        end
    end
end

# Test different tolerance values
tolerances = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

println("\n" * "-"^80)
println("Testing with different tolerances")
println("-"^80)

t_eval = t[26:76]  # Middle 50% to avoid boundaries
n_eval = length(t_eval)

for tol in tolerances
    println("\nTolerance: $tol")

    result = extend_with_approxfun(t, y_smooth, t_eval, 7; tol=tol)

    # Report fit quality
    meta = result["metadata"]
    println("  Coefficients: $(meta["n_coefficients"])")
    println("  Fit RMSE: $(round(meta["rmse_fit"], sigdigits=3))")

    if haskey(meta["decay_metrics"], "decay_rate")
        println("  Decay rate: $(round(meta["decay_metrics"]["decay_rate"], digits=3))")
    end

    # Check derivatives at middle point
    mid_idx = div(n_eval, 2)
    t_test = t_eval[mid_idx]

    print("  Errors at t=$(round(t_test, digits=2)): ")
    for order in [0, 1, 3, 5, 7]
        pred = result["predictions"][order][mid_idx]
        truth = ground_truth(order, t_test)
        error = abs(pred - truth)
        print("O$order=$(round(error, sigdigits=2)) ")
    end
    println()

    # Print warnings if any
    if length(result["warnings"]) > 0
        println("  ⚠️ Warnings: ", join(result["warnings"], "; "))
    end
end

# Test with actual PyNumDiff-like output
println("\n" * "="^80)
println("Test with simulated PyNumDiff output")
println("="^80)

# Simulate a PyNumDiff method result
pynumdiff_result = Dict(
    "predictions" => Dict(
        0 => y_smooth,
        1 => [ω * cos(ω * t_i) + 0.01*randn() for t_i in t]  # Noisy first derivative
    ),
    "metadata" => Dict("method" => "Butterworth")
)

# Test the wrapper function
result = wrap_pynumdiff_with_approxfun("PyNumDiff-Butter-Auto",
                                       t, y_noisy, t_eval,
                                       pynumdiff_result;
                                       max_order=7,
                                       adaptive_tol=true,
                                       prefer_original_low_orders=false)

println("\nUsing PyNumDiff wrapper:")
println("  Base method: $(result["metadata"]["base_method"])")
println("  Coefficients: $(result["metadata"]["n_coefficients"])")
println("  Noise estimate: $(round(result["metadata"]["noise_estimate"], sigdigits=3))")

# Compute RMSE for each order
println("\nRMSE by derivative order:")
for order in 0:7
    if haskey(result["predictions"], order)
        pred = result["predictions"][order]
        truth = [ground_truth(order, t_i) for t_i in t_eval]
        rmse = sqrt(mean((pred .- truth).^2))
        println("  Order $order: $(round(rmse, sigdigits=4))")
    end
end

# Visual check of coefficient decay
println("\n" * "="^80)
println("Coefficient Decay Analysis")
println("="^80)

# Fit with default tolerance
using Interpolations
itp_smooth = linear_interpolation(t, y_smooth, extrapolation_bc=Flat())
f = Fun(x -> itp_smooth(x), 0..1)
coeffs_all = coefficients(f)
# Truncate at tolerance
n_keep = findfirst(i -> abs(coeffs_all[i]) < 1e-8, 1:length(coeffs_all))
if isnothing(n_keep)
    n_keep = length(coeffs_all)
end
coeffs = coeffs_all[1:n_keep]

println("First 10 coefficients (log10 |c_n|):")
for i in 1:min(10, length(coeffs))
    log_val = log10(abs(coeffs[i]) + 1e-16)
    bar_length = max(0, round(Int, 40 + 2*log_val))  # Scale for display
    bar = "█" ^ bar_length
    @printf("  c_%2d: %8.2f %s\n", i-1, log_val, bar)
end

if length(coeffs) > 20
    println("...")
    println("Last 5 coefficients:")
    for i in (length(coeffs)-4):length(coeffs)
        log_val = log10(abs(coeffs[i]) + 1e-16)
        bar_length = max(0, round(Int, 40 + 2*log_val))
        bar = "█" ^ bar_length
        @printf("  c_%2d: %8.2f %s\n", i-1, log_val, bar)
    end
end

# Check smoothness validation
println("\n" * "="^80)
println("Smoothness Validation")
println("="^80)

for target_order in [3, 5, 7, 9]
    is_smooth, noise_est = validate_smoothness_for_order(y_smooth, target_order)
    status = is_smooth ? "✓ Smooth enough" : "✗ Too noisy"
    println("  Order $target_order: $status (noise ≈ $(round(noise_est, sigdigits=2)))")
end

println("\n" * "="^80)
println("Test Complete")
println("="^80)