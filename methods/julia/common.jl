"""
Common utilities for Julia derivative estimation methods

Provides:
- MethodResult struct for standardized return values
- Derivative computation via ForwardDiff and TaylorDiff
- AAA rational approximation utilities
"""

using BaryRational
using ForwardDiff
using TaylorDiff
using Statistics
using LinearAlgebra

# Load hyperparameter selection module (from parent src/)
include("../../src/hyperparameter_selection.jl")
using .HyperparameterSelection

# ============================================================================
# Method Result Structure
# ============================================================================

"""Result from a single method evaluation"""
struct MethodResult
	name::String
	category::String
	predictions::Dict{Int, Vector{Float64}}  # order => values
	failures::Dict{Int, String}  # order => error message
	timing::Float64
	success::Bool
end


# ============================================================================
# Derivative Computation Utilities
# ============================================================================

"""
	nth_deriv_at(f, n, t)

Compute nth derivative of function f at point t using ForwardDiff (recursive).
"""
function nth_deriv_at(f, n::Int, t)
	n == 0 && return f(t)
	n == 1 && return ForwardDiff.derivative(f, t)
	g(t) = nth_deriv_at(f, n - 1, t)
	return ForwardDiff.derivative(g, t)
end

"""
	nth_deriv_taylor(f, n, t)

Compute nth derivative using TaylorDiff (faster for n > 1).
"""
function nth_deriv_taylor(f, n::Int, t)
	n == 0 && return f(t)
	return TaylorDiff.derivative(f, t, Val(n))
end

"""
	compute_derivatives_at_points(f, x_eval, orders; method=:taylor)

Compute derivatives of fitted function f at evaluation points.
"""
function compute_derivatives_at_points(f, x_eval, orders; method = :taylor)
	results = Dict{Int, Vector{Float64}}()

	for order in orders
		try
			if method == :taylor && order > 0
				results[order] = [nth_deriv_taylor(f, order, x) for x in x_eval]
			else
				results[order] = [nth_deriv_at(f, order, x) for x in x_eval]
			end
		catch e
			# Return NaNs if differentiation fails
			results[order] = fill(NaN, length(x_eval))
		end
	end

	return results
end


# ============================================================================
# AAA Rational Approximation
# ============================================================================

"""Barycentric evaluation"""
function bary_eval(z, f::Vector{T}, x::Vector{T}, w::Vector{T}, tol = 1e-13) where {T}
	num, den = zero(T), zero(T)
	breakflag, breakindex = false, -1

	for j in eachindex(f)
		if abs(z - x[j]) < sqrt(tol)
			breakflag = true
			breakindex = j
			break
		end
		t = w[j] / (z - x[j])
		num += t * f[j]
		den += t
	end

	if breakflag
		# Near a support point, use special formula
		num, den = zero(T), zero(T)
		for j in eachindex(f)
			if j != breakindex
				t = w[j] / (z - x[j])
				num += t * f[j]
				den += t
			end
		end
		m = z - x[breakindex]
		return (w[breakindex] * f[breakindex] + m * num) / (w[breakindex] + m * den)
	end

	return num / den
end

"""Wrapper for AAA approximation"""
struct AAAApprox
	f::Vector{Float64}
	x::Vector{Float64}
	w::Vector{Float64}
end

(approx::AAAApprox)(z) = bary_eval(z, approx.f, approx.x, approx.w)

"""
	fit_aaa(x, y; tol=1e-13, mmax=100)

Fit AAA rational approximation.
"""
function fit_aaa(x, y; tol = 1e-13, mmax = 100)
	result = BaryRational.aaa(x, y; tol = tol, mmax = mmax, verbose = false)
	return AAAApprox(result.f, result.x, result.w)
end
