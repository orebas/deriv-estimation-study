"""
Test Julia Method Extraction

Validates that extracted Julia methods in methods/julia/* produce identical results
to the original implementations in src/julia_methods.jl.
"""

using Random
using Statistics
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

# Load original implementation
include("../src/julia_methods.jl")

# Load extracted implementations
include("julia/common.jl")
include("julia/gp/gaussian_process.jl")
include("julia/rational/aaa.jl")
include("julia/spectral/fourier.jl")
include("julia/splines/splines.jl")
include("julia/filtering/filters.jl")
include("julia/regularization/regularized.jl")
include("julia/finite_diff/finite_diff.jl")

# Test data
N_TRAIN = 30
N_EVAL = 20
x_train = sort(rand(N_TRAIN) .* 2π)
y_train = sin.(x_train) .+ 0.05 .* randn(N_TRAIN)
x_eval = sort(rand(N_EVAL) .* 2π)
orders = [0, 1, 2, 3]

# Tolerance for numerical comparison
RTOL = 1e-6
ATOL = 1e-5

"""Compare two prediction dictionaries"""
function compare_predictions(name::String, original::Dict, extracted::Dict; rtol=RTOL, atol=ATOL)
    println("\n" * "="^60)
    println("Testing: $name")
    println("="^60)

    all_passed = true

    for order in keys(original)
        if !haskey(extracted, order)
            println("❌ FAIL Order $order: Missing in extracted predictions")
            all_passed = false
            continue
        end

        orig_vals = original[order]
        extr_vals = extracted[order]

        if length(orig_vals) != length(extr_vals)
            println("❌ FAIL Order $order: Length mismatch ($(length(orig_vals)) vs $(length(extr_vals)))")
            all_passed = false
            continue
        end

        # Check for NaN matches
        orig_nan = isnan.(orig_vals)
        extr_nan = isnan.(extr_vals)

        if orig_nan != extr_nan
            println("❌ FAIL Order $order: NaN pattern mismatch")
            all_passed = false
            continue
        end

        # Compare non-NaN values
        valid_idx = .!orig_nan
        if any(valid_idx)
            orig_valid = orig_vals[valid_idx]
            extr_valid = extr_vals[valid_idx]

            if !isapprox(orig_valid, extr_valid; rtol=rtol, atol=atol)
                max_diff = maximum(abs.(orig_valid .- extr_valid))
                println("❌ FAIL Order $order: Values differ (max diff: $max_diff)")
                all_passed = false
                continue
            end
        end

        println("✅ PASS Order $order: Predictions match (rtol=$rtol, atol=$atol)")
    end

    if all_passed
        println("✅ $name: ALL TESTS PASSED")
    else
        println("❌ $name: SOME TESTS FAILED")
    end

    return all_passed
end

"""Test GP methods"""
function test_gp_methods()
    println("\n" * "="^60)
    println("TESTING GP METHODS")
    println("="^60)

    gp_methods = [
        ("GP-Julia-SE", evaluate_gp_se),
        ("GP-Julia-AD", evaluate_gp_ad),
        ("GP-Julia-Matern-0.5", evaluate_gp_matern_05),
        ("GP-Julia-Matern-1.5", evaluate_gp_matern_15),
        ("GP-Julia-Matern-2.5", evaluate_gp_matern_25),
    ]

    all_success = true

    for (method_name, eval_func) in gp_methods
        try
            # Original implementation
            orig_result = evaluate_julia_method(method_name, x_train, y_train, x_eval, orders)

            # Extracted implementation
            extr_result = eval_func(x_train, y_train, x_eval, orders)

            success = compare_predictions(method_name, orig_result.predictions, extr_result.predictions)
            all_success = all_success && success
        catch e
            println("❌ ERROR testing $method_name: $e")
            all_success = false
        end
    end

    return all_success
end

"""Test AAA/Rational methods"""
function test_aaa_methods()
    println("\n" * "="^60)
    println("TESTING AAA/RATIONAL METHODS")
    println("="^60)

    aaa_methods = [
        ("AAA-HighPrec", evaluate_aaa_highprec),
        ("AAA-LowPrec", evaluate_aaa_lowprec),
        ("AAA-Adaptive-Diff2", evaluate_aaa_adaptive_diff2),
        ("AAA-Adaptive-Wavelet", evaluate_aaa_adaptive_wavelet),
    ]

    all_success = true

    for (method_name, eval_func) in aaa_methods
        try
            # Original implementation
            orig_result = evaluate_julia_method(method_name, x_train, y_train, x_eval, orders)

            # Extracted implementation
            extr_result = eval_func(x_train, y_train, x_eval, orders)

            success = compare_predictions(method_name, orig_result.predictions, extr_result.predictions)
            all_success = all_success && success
        catch e
            println("❌ ERROR testing $method_name: $e")
            all_success = false
        end
    end

    return all_success
end

"""Test Spectral methods"""
function test_spectral_methods()
    println("\n" * "="^60)
    println("TESTING SPECTRAL METHODS")
    println("="^60)

    spectral_methods = [
        ("Fourier-Interp", evaluate_fourier_interp),
        ("Fourier-FFT-Adaptive", evaluate_fourier_fft_adaptive),
    ]

    all_success = true

    for (method_name, eval_func) in spectral_methods
        try
            # Original implementation
            orig_result = evaluate_julia_method(method_name, x_train, y_train, x_eval, orders)

            # Extracted implementation
            extr_result = eval_func(x_train, y_train, x_eval, orders)

            success = compare_predictions(method_name, orig_result.predictions, extr_result.predictions)
            all_success = all_success && success
        catch e
            println("❌ ERROR testing $method_name: $e")
            all_success = false
        end
    end

    return all_success
end

"""Test Splines methods"""
function test_splines_methods()
    println("\n" * "="^60)
    println("TESTING SPLINES METHODS")
    println("="^60)

    all_success = true

    try
        # Original implementation
        orig_result = evaluate_julia_method("Dierckx-5", x_train, y_train, x_eval, orders)

        # Extracted implementation
        extr_result = evaluate_dierckx(x_train, y_train, x_eval, orders)

        success = compare_predictions("Dierckx-5", orig_result.predictions, extr_result.predictions)
        all_success = all_success && success
    catch e
        println("❌ ERROR testing Dierckx-5: $e")
        all_success = false
    end

    return all_success
end

"""Test Filtering methods"""
function test_filtering_methods()
    println("\n" * "="^60)
    println("TESTING FILTERING METHODS")
    println("="^60)

    all_success = true

    try
        # Original implementation
        orig_result = evaluate_julia_method("Savitzky-Golay", x_train, y_train, x_eval, orders)

        # Extracted implementation
        extr_result = evaluate_savitzky_golay(x_train, y_train, x_eval, orders)

        success = compare_predictions("Savitzky-Golay", orig_result.predictions, extr_result.predictions)
        all_success = all_success && success
    catch e
        println("❌ ERROR testing Savitzky-Golay: $e")
        all_success = false
    end

    return all_success
end

"""Test Regularization methods"""
function test_regularization_methods()
    println("\n" * "="^60)
    println("TESTING REGULARIZATION METHODS")
    println("="^60)

    reg_methods = [
        ("TrendFilter-k7", evaluate_trend_filter_k7),
        ("TrendFilter-k2", evaluate_trend_filter_k2),
        ("TVRegDiff-Julia", evaluate_tvregdiff),
    ]

    all_success = true

    for (method_name, eval_func) in reg_methods
        try
            # Original implementation
            orig_result = evaluate_julia_method(method_name, x_train, y_train, x_eval, orders)

            # Extracted implementation
            extr_result = eval_func(x_train, y_train, x_eval, orders)

            success = compare_predictions(method_name, orig_result.predictions, extr_result.predictions)
            all_success = all_success && success
        catch e
            println("❌ ERROR testing $method_name: $e")
            all_success = false
        end
    end

    return all_success
end

"""Test Finite Diff methods"""
function test_finite_diff_methods()
    println("\n" * "="^60)
    println("TESTING FINITE DIFF METHODS")
    println("="^60)

    all_success = true

    try
        # Use uniform grid for finite differences
        x_uniform = collect(range(0, 2π, length=N_TRAIN))
        y_uniform = sin.(x_uniform) .+ 0.05 .* randn(N_TRAIN)
        x_eval_uniform = collect(range(0, 2π, length=N_EVAL))

        # Original implementation
        orig_result = evaluate_julia_method("Central-FD", x_uniform, y_uniform, x_eval_uniform, orders)

        # Extracted implementation
        extr_result = evaluate_central_fd(x_uniform, y_uniform, x_eval_uniform, orders)

        success = compare_predictions("Central-FD", orig_result.predictions, extr_result.predictions)
        all_success = all_success && success
    catch e
        println("❌ ERROR testing Central-FD: $e")
        all_success = false
    end

    return all_success
end

"""Main test runner"""
function main()
    println("="^60)
    println("Julia Method Extraction Validation Tests")
    println("="^60)
    println("Test data: $N_TRAIN training points, $N_EVAL evaluation points")
    println("Orders: $orders")
    println("Tolerances: rtol=$RTOL, atol=$ATOL")

    # Run all tests
    gp_success = test_gp_methods()
    aaa_success = test_aaa_methods()
    spectral_success = test_spectral_methods()
    splines_success = test_splines_methods()
    filtering_success = test_filtering_methods()
    regularization_success = test_regularization_methods()
    finite_diff_success = test_finite_diff_methods()

    # Overall summary
    println("\n" * "="^60)
    println("OVERALL RESULTS:")
    println("  GP Methods: $(gp_success ? "✅ PASSED" : "❌ FAILED")")
    println("  AAA/Rational Methods: $(aaa_success ? "✅ PASSED" : "❌ FAILED")")
    println("  Spectral Methods: $(spectral_success ? "✅ PASSED" : "❌ FAILED")")
    println("  Splines Methods: $(splines_success ? "✅ PASSED" : "❌ FAILED")")
    println("  Filtering Methods: $(filtering_success ? "✅ PASSED" : "❌ FAILED")")
    println("  Regularization Methods: $(regularization_success ? "✅ PASSED" : "❌ FAILED")")
    println("  Finite Diff Methods: $(finite_diff_success ? "✅ PASSED" : "❌ FAILED")")
    println("="^60)

    overall_success = all([gp_success, aaa_success, spectral_success, splines_success,
                          filtering_success, regularization_success, finite_diff_success])

    if overall_success
        println("🎉 ALL JULIA EXTRACTION TESTS PASSED!")
        return 0
    else
        println("⚠️  SOME JULIA EXTRACTION TESTS FAILED")
        return 1
    end
end

# Run tests
exit(main())
