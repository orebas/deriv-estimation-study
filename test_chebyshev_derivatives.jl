"""
Test Chebyshev polynomial approximation for stable derivative computation

Chebyshev polynomials should be MUCH more stable than rational functions
because they don't have denominators.

We'll test:
1. ApproxFun.jl (if available)
2. Manual Chebyshev fit using FFTW
"""

using LinearAlgebra
using FFTW

println("=" ^ 80)
println("Chebyshev Approximation for Derivative Estimation")
println("=" ^ 80)

# Try ApproxFun first
println("\nTesting ApproxFun availability...")
try
    using ApproxFun
    println("✓ ApproxFun is available!")
    HAS_APPROXFUN = true
catch e
    println("✗ ApproxFun not installed")
    println("  Install with: using Pkg; Pkg.add(\"ApproxFun\")")
    HAS_APPROXFUN = false
end

# Manual Chebyshev implementation
"""
Fit Chebyshev series to data using least squares.

Returns coefficients c such that:
f(x) ≈ Σ cᵢ Tᵢ(x̃)  where x̃ ∈ [-1, 1]
"""
function fit_chebyshev(x::Vector{T}, y::Vector{T}, degree::Int) where {T}
    n = length(x)

    # Map x to [-1, 1]
    x_min, x_max = extrema(x)
    x_scaled = @. 2 * (x - x_min) / (x_max - x_min) - 1

    # Build Vandermonde matrix of Chebyshev polynomials
    # T₀(x) = 1, T₁(x) = x, Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x)
    V = zeros(T, n, degree + 1)

    for i in 1:n
        V[i, 1] = 1.0  # T₀
        if degree >= 1
            V[i, 2] = x_scaled[i]  # T₁
        end
        for k in 3:(degree + 1)
            V[i, k] = 2 * x_scaled[i] * V[i, k-1] - V[i, k-2]
        end
    end

    # Least squares fit
    coeffs = V \ y

    return coeffs, x_min, x_max
end

"""
Evaluate Chebyshev series at point x.
"""
function eval_chebyshev(coeffs::Vector{T}, x::Real, x_min, x_max) where {T}
    # Map x to [-1, 1]
    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1

    # Evaluate using Clenshaw's algorithm (numerically stable)
    degree = length(coeffs) - 1
    if degree == 0
        return coeffs[1]
    elseif degree == 1
        return coeffs[1] + coeffs[2] * x_scaled
    end

    b_kp2 = 0.0
    b_kp1 = 0.0

    for k in degree:-1:0
        b_k = coeffs[k+1] + 2 * x_scaled * b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k
    end

    return b_kp1 - x_scaled * b_kp2
end

"""
Compute nth derivative of Chebyshev series analytically.

Chebyshev derivative formulas:
T₀' = 0
T₁' = 1
Tₙ' = 2nTₙ₋₁ + Tₙ₋₂'
"""
function deriv_chebyshev_coeffs(coeffs::Vector{T}, x_min, x_max) where {T}
    degree = length(coeffs) - 1

    if degree == 0
        return [0.0]
    end

    # Chebyshev derivative coefficients
    deriv_coeffs = zeros(T, degree)

    # Special formula: if f = Σ cᵢTᵢ, then f' = Σ dᵢTᵢ where
    # dₙ₋₁ = 2n cₙ
    # dₖ = dₖ₊₂ + 2(k+1)cₖ₊₁  for k = n-2, ..., 0

    deriv_coeffs[degree] = 2 * degree * coeffs[degree + 1]

    if degree >= 2
        for k in (degree-1):-1:1
            if k == degree - 1
                deriv_coeffs[k] = 2 * k * coeffs[k + 1]
            else
                deriv_coeffs[k] = deriv_coeffs[k + 2] + 2 * k * coeffs[k + 1]
            end
        end
    end

    # Scale by derivative of coordinate transform: d/dx = (2/(x_max - x_min)) * d/dx̃
    scale = 2 / (x_max - x_min)
    deriv_coeffs .*= scale

    return deriv_coeffs
end

"""
Compute nth derivative coefficients recursively.
"""
function nth_deriv_chebyshev_coeffs(coeffs::Vector{T}, n::Int, x_min, x_max) where {T}
    if n == 0
        return coeffs
    end

    # Compute 1st derivative
    deriv_coeffs = deriv_chebyshev_coeffs(coeffs, x_min, x_max)

    # Recursively compute higher derivatives
    for i in 2:n
        deriv_coeffs = deriv_chebyshev_coeffs(deriv_coeffs, x_min, x_max)
    end

    return deriv_coeffs
end

# Test on e^x
println("\n" * "=" ^ 80)
println("Test: Chebyshev Approximation of e^x")
println("=" ^ 80)

n_points = 101
t = range(0, 1, length=n_points)
y = exp.(t)

println("\nSignal: y = e^x on [0, 1]")
println("All derivatives should equal e^x")

# Test different polynomial degrees
degrees = [10, 20, 30, 50]

for degree in degrees
    println("\n" * "-" ^ 80)
    println("Chebyshev degree: $degree")
    println("-" ^ 80)

    # Fit Chebyshev
    coeffs, x_min, x_max = fit_chebyshev(collect(t), y, degree)

    # Compute derivatives at test point
    test_point = 0.5
    truth = exp(test_point)

    println("Test point: t = $test_point, ground truth = $(round(truth, sigdigits=10))")

    for order in 0:7
        # Get derivative coefficients
        deriv_coeffs = nth_deriv_chebyshev_coeffs(coeffs, order, x_min, x_max)

        # Evaluate
        deriv_val = eval_chebyshev(deriv_coeffs, test_point, x_min, x_max)

        error = abs(deriv_val - truth)
        rel_error = error / abs(truth)

        println("  Order $order: value = $(round(deriv_val, sigdigits=6)), error = $(round(error, sigdigits=4)), rel = $(round(rel_error, sigdigits=4))")
    end
end

# Compare with AAA
println("\n" * "=" ^ 80)
println("Comparison: Chebyshev vs AAA")
println("=" ^ 80)

include("methods/julia/common.jl")

# AAA fit
fitted_aaa = fit_aaa(collect(t), y; tol=1e-13, mmax=100)

# Chebyshev fit (degree 30)
coeffs_cheby, x_min, x_max = fit_chebyshev(collect(t), y, 30)

test_point = 0.5
truth = exp(test_point)

println("\nTest point: t = $test_point")
println("Ground truth: $(round(truth, sigdigits=10))")
println("\n" * "-" ^ 80)
println("Order | Chebyshev Error | AAA Error | Winner")
println("-" ^ 80)

for order in 0:7
    # Chebyshev
    deriv_coeffs = nth_deriv_chebyshev_coeffs(coeffs_cheby, order, x_min, x_max)
    cheby_val = eval_chebyshev(deriv_coeffs, test_point, x_min, x_max)
    cheby_error = abs(cheby_val - truth)

    # AAA
    if order == 0
        aaa_val = fitted_aaa(test_point)
    else
        using TaylorDiff
        aaa_val = TaylorDiff.derivative(fitted_aaa, test_point, Val(order))
    end
    aaa_error = abs(aaa_val - truth)

    winner = cheby_error < aaa_error ? "Chebyshev ✓" : "AAA"

    println("  $order   | $(round(cheby_error, sigdigits=4))      | $(round(aaa_error, sigdigits=4))    | $winner")
end

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
println("""
If Chebyshev outperforms AAA for higher derivatives, we should switch
to Chebyshev-ROM instead of AAA-ROM!

Chebyshev advantages:
✓ No denominators → more stable differentiation
✓ Analytic derivative formulas
✓ Well-conditioned for high-order derivatives
✓ Standard in numerical analysis
""")
