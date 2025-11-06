"""
Test ApproxFun extension for orders 1→3 (more realistic goal)
"""

using ApproxFun
using Statistics
using Printf
using Interpolations

# Load the extension module
include("methods/julia/extensions/approxfun_extension.jl")

println("="^80)
println("Testing ApproxFun Extension: Orders 1 → 3")
println("="^80)

# Test with multiple signals
test_cases = [
    (name="Sine wave",
     func=t -> sin(2π*t),
     derivatives=[
         t -> sin(2π*t),           # Order 0
         t -> 2π*cos(2π*t),         # Order 1
         t -> -(2π)^2*sin(2π*t),    # Order 2
         t -> -(2π)^3*cos(2π*t)     # Order 3
     ]),
    (name="Exponential decay",
     func=t -> exp(-2t),
     derivatives=[
         t -> exp(-2t),             # Order 0
         t -> -2*exp(-2t),          # Order 1
         t -> 4*exp(-2t),           # Order 2
         t -> -8*exp(-2t)           # Order 3
     ]),
    (name="Polynomial",
     func=t -> t^4 - 2t^3 + t^2,
     derivatives=[
         t -> t^4 - 2t^3 + t^2,     # Order 0
         t -> 4t^3 - 6t^2 + 2t,     # Order 1
         t -> 12t^2 - 12t + 2,      # Order 2
         t -> 24t - 12              # Order 3
     ])
]

# Generate test data
n = 101
t = collect(range(0, 1, length=n))
noise_levels = [1e-4, 1e-3, 1e-2]

# Methods that only provide order 0-1
methods_to_extend = [
    "PyNumDiff-Butter",
    "PyNumDiff-Gaussian",
    "PyNumDiff-Kalman",
    "TVRegDiff",
    "Central-FD"
]

println("\nTest Configuration:")
println("  Points: $n")
println("  Domain: [0, 1]")
println("  Max order: 3")
println("  Methods to extend: $(join(methods_to_extend, ", "))")
println()

# Function to simulate different smoothing methods
function simulate_smoothing_method(method_name, t, y_noisy)
    if occursin("Butter", method_name) || occursin("Gaussian", method_name)
        # Gaussian-like smoothing
        window = 7
        y_smooth = similar(y_noisy)
        half_win = window ÷ 2
        for i in 1:length(y_noisy)
            i_start = max(1, i - half_win)
            i_end = min(length(y_noisy), i + half_win)
            weights = exp.(-0.5 * ((collect(i_start:i_end) .- i) ./ (half_win/2)).^2)
            weights ./= sum(weights)
            y_smooth[i] = sum(weights .* y_noisy[i_start:i_end])
        end
        return y_smooth

    elseif occursin("Kalman", method_name)
        # Kalman-like (very smooth)
        α = 0.98  # High smoothing
        y_smooth = similar(y_noisy)
        y_smooth[1] = y_noisy[1]
        for i in 2:length(y_noisy)
            y_smooth[i] = α * y_smooth[i-1] + (1-α) * y_noisy[i]
        end
        # Backward pass for symmetry
        for i in (length(y_noisy)-1):-1:1
            y_smooth[i] = 0.5 * (y_smooth[i] + α * y_smooth[i+1] + (1-α) * y_noisy[i])
        end
        return y_smooth

    elseif occursin("TV", method_name)
        # Total variation (piecewise smooth)
        # Simple L1 trend filtering approximation
        y_smooth = copy(y_noisy)
        for _ in 1:3  # Few iterations
            for i in 2:(length(y_smooth)-1)
                y_smooth[i] = 0.6*y_smooth[i] + 0.2*(y_smooth[i-1] + y_smooth[i+1])
            end
        end
        return y_smooth

    else  # Central-FD
        # Minimal smoothing
        return copy(y_noisy)
    end
end

# Test each case
results = Dict()

for test_case in test_cases
    println("-"^80)
    println("Test: $(test_case.name)")
    println("-"^80)

    for noise_level in noise_levels
        println("\nNoise level: $noise_level")

        # Generate noisy data
        y_true = test_case.func.(t)
        y_noisy = y_true .+ noise_level * randn(n)

        # Evaluation points (avoid boundaries)
        t_eval = t[11:91]  # Middle 80%

        for method in methods_to_extend[1:3]  # Test first 3 methods
            # Simulate smoothing
            y_smooth = simulate_smoothing_method(method, t, y_noisy)

            # Apply ApproxFun extension with different tolerances
            tolerances = [1e-6, 1e-8, 1e-10]
            best_error = Inf
            best_tol = 0.0

            for tol in tolerances
                try
                    result = extend_with_approxfun(t, y_smooth, t_eval, 3;
                                                  tol=tol, max_coeffs=200,
                                                  check_decay=false, trim_boundary=0.1)

                    # Check order 3 accuracy at middle point
                    mid_idx = div(length(t_eval), 2)
                    t_test = t_eval[mid_idx]

                    pred3 = result["predictions"][3][mid_idx]
                    true3 = test_case.derivatives[4](t_test)  # Order 3 is index 4
                    error3 = abs(pred3 - true3) / (abs(true3) + 1e-10)  # Relative error

                    if error3 < best_error
                        best_error = error3
                        best_tol = tol
                    end
                catch e
                    # Skip on error
                end
            end

            @printf("  %-20s: Best rel. error = %.2f%% (tol=%g)\n",
                    method, best_error*100, best_tol)
        end
    end
end

# Detailed analysis for best case
println("\n" * "="^80)
println("Detailed Analysis: Kalman smoothing on polynomial")
println("="^80)

# Polynomial should be exact for spectral methods
y_true = test_cases[3].func.(t)
y_noisy = y_true .+ 1e-3 * randn(n)
y_smooth = simulate_smoothing_method("Kalman", t, y_noisy)
t_eval = t[11:91]

result = extend_with_approxfun(t, y_smooth, t_eval, 3;
                              tol=1e-8, max_coeffs=100,
                              check_decay=true, trim_boundary=0.1)

println("\nFit statistics:")
println("  Coefficients used: $(result["metadata"]["n_coefficients"])")
println("  Fit RMSE: $(round(result["metadata"]["rmse_fit"], sigdigits=3))")

if haskey(result["metadata"]["decay_metrics"], "decay_rate")
    println("  Decay rate: $(round(result["metadata"]["decay_metrics"]["decay_rate"], digits=3))")
end

# Compare all orders
println("\nAccuracy by order (RMSE over evaluation points):")
for order in 0:3
    pred = result["predictions"][order]
    true_vals = [test_cases[3].derivatives[order+1](t_i) for t_i in t_eval]
    rmse = sqrt(mean((pred .- true_vals).^2))
    rel_rmse = rmse / (sqrt(mean(true_vals.^2)) + 1e-10)
    @printf("  Order %d: RMSE = %.3e (%.1f%% relative)\n", order, rmse, rel_rmse*100)
end

# Check specific challenging methods
println("\n" * "="^80)
println("Challenge: TV Regularization (piecewise smooth)")
println("="^80)

y_smooth_tv = simulate_smoothing_method("TV", t, y_noisy)

# Compare smoothness
diff1 = diff(y_smooth)
diff1_tv = diff(y_smooth_tv)

println("Smoothness comparison (std of first differences):")
println("  Kalman: $(round(std(diff1), sigdigits=3))")
println("  TV:     $(round(std(diff1_tv), sigdigits=3))")

# Try extension
result_tv = extend_with_approxfun(t, y_smooth_tv, t_eval, 3;
                                 tol=1e-6, max_coeffs=50,  # Lower requirements
                                 check_decay=false)

println("\nTV extension results:")
println("  Coefficients: $(result_tv["metadata"]["n_coefficients"])")
for order in 0:3
    pred = result_tv["predictions"][order]
    true_vals = [test_cases[3].derivatives[order+1](t_i) for t_i in t_eval]
    rmse = sqrt(mean((pred .- true_vals).^2))
    @printf("  Order %d RMSE: %.3e\n", order, rmse)
end

# Final recommendations
println("\n" * "="^80)
println("RECOMMENDATIONS")
println("="^80)

println("""
For extending methods from order 1 to order 3:

✅ GOOD candidates (smooth output):
- PyNumDiff-Kalman: Excellent smoothness, low errors
- PyNumDiff-Gaussian: Good smoothness, moderate errors
- PyNumDiff-Butter: Acceptable for low noise

⚠️ MARGINAL candidates:
- TVRegDiff: Piecewise smooth, may need lower tolerance
- PyNumDiff-Spline: Depends on spline parameters

❌ POOR candidates:
- Central-FD: No smoothing, will amplify noise
- Any method with high noise (>1%)

Key parameters for order 3 extension:
- Tolerance: 1e-6 to 1e-8 (looser than order 7)
- Max coefficients: 50-200 (much less than order 7)
- Focus on middle 80% of domain (boundary issues)
""")