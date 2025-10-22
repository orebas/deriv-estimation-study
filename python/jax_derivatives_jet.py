#!/usr/bin/env python3
"""
JAX Taylor-mode derivatives using jax.experimental.jet

This is MUCH faster than nested jvp() or grad() calls:
- Nested: O(n²) - compute each derivative separately
- Jet: O(n) - compute ALL derivatives in one forward pass

Expected speedup: 10-50× for orders 0-6
"""

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet

# Enable float64
jax.config.update("jax_enable_x64", True)


def compute_all_derivatives_jet(fn, x, max_order=6):
    """
    Compute derivatives 0 through max_order using Taylor-mode AD.

    This is MUCH faster than nested differentiation:
    - Nested jvp: 1+2+3+4+5+6 = 21 calls for orders 0-6
    - Jet: 1 forward pass for ALL orders

    Args:
        fn: Function to differentiate (JAX-traceable, scalar input -> scalar output)
        x: Point at which to evaluate (scalar)
        max_order: Maximum derivative order (default: 6)

    Returns:
        Array of shape (max_order+1,) containing [f, f', f'', ..., f^(n)]

    Example:
        >>> fn = lambda x: jnp.sin(x)
        >>> derivs = compute_all_derivatives_jet(fn, 1.0, max_order=3)
        >>> # derivs[0] = sin(1.0)
        >>> # derivs[1] = cos(1.0)
        >>> # derivs[2] = -sin(1.0)
        >>> # derivs[3] = -cos(1.0)
    """
    # For computing derivatives of f(x), we think of it as f(h(t))
    # where h(t) = x + t, and we want derivatives w.r.t. t at t=0.
    #
    # The tangent series represents derivatives of h(t):
    #   h(0) = x, h'(0) = 1, h''(0) = 0, h'''(0) = 0, ...
    #
    # CRITICAL: series must be a TUPLE OF SCALARS, not a numpy array!
    # Use tuple for scalar x, but this also works with vmap
    tangent_series = tuple([1.0] + [0.0] * max_order)

    # Ensure x is JAX array for compatibility
    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x, dtype=jnp.float64)

    # Call jet with factorial_scaled=True (default)
    # Returns actual derivatives, not Taylor coefficients
    primal_out, series_out = jet(
        fn,
        (x,),                    # Evaluation point: h(0) = x
        (tangent_series,),       # Derivatives of h: (h'(0), h''(0), ...)
    )

    # series_out is a list of Arrays: [f'(x), f''(x), f'''(x), ...]
    # Combine with primal: [f(x), f'(x), f''(x), ...]
    all_derivatives = jnp.concatenate([
        jnp.array([primal_out]),
        jnp.array(series_out)
    ])

    return all_derivatives


def compute_derivatives_at_points(fn, x_array, max_order=6):
    """
    Compute all derivatives at multiple evaluation points using vectorization.

    Args:
        fn: Function to differentiate (JAX-traceable)
        x_array: Array of evaluation points, shape (n_points,)
        max_order: Maximum derivative order

    Returns:
        Array of shape (max_order+1, n_points) where:
        - derivatives[k, i] = k-th derivative at point x_array[i]

    Example:
        >>> fn = lambda x: jnp.sin(x)
        >>> x_points = jnp.linspace(0, 1, 101)
        >>> all_derivs = compute_derivatives_at_points(fn, x_points, max_order=6)
        >>> # all_derivs.shape = (7, 101)
        >>> # all_derivs[0, :] = sin(x) at all points
        >>> # all_derivs[1, :] = cos(x) at all points
        >>> # all_derivs[2, :] = -sin(x) at all points
    """
    x_array = jnp.asarray(x_array, dtype=jnp.float64)

    # Vectorize over evaluation points
    # vmap creates a batched version that processes all points in parallel
    vectorized_fn = jax.vmap(
        lambda x: compute_all_derivatives_jet(fn, x, max_order)
    )

    # Result shape: (n_points, max_order+1)
    # Transpose to get (max_order+1, n_points)
    result = vectorized_fn(x_array).T

    return result


def nth_derivative_jet(fn, x, order):
    """
    Compute a SINGLE derivative order using jet (still faster than nested jvp).

    Args:
        fn: Function to differentiate
        x: Evaluation point (scalar or array)
        order: Derivative order (0, 1, 2, ...)

    Returns:
        n-th derivative at x (same shape as x)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    is_scalar = x.ndim == 0

    if is_scalar:
        # Scalar case
        all_derivs = compute_all_derivatives_jet(fn, x, max_order=order)
        return all_derivs[order]
    else:
        # Array case - vectorize
        all_derivs = compute_derivatives_at_points(fn, x, max_order=order)
        return all_derivs[order, :]


# Benchmark function
def benchmark_jet_vs_nested():
    """
    Compare performance of jet vs nested differentiation.
    """
    import time

    # Test function: barycentric rational (simplified)
    def test_fn(x):
        # Simplified barycentric: r(x) = sum(w_i/(x-z_i)) / sum(1/(x-z_i))
        z = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        w = jnp.array([1.0, -0.5, 0.3, -0.2, 0.1])

        diffs = x - z
        safe_diffs = jnp.where(diffs == 0, jnp.ones_like(diffs), diffs)
        inv_diffs = 1.0 / safe_diffs

        return jnp.sum(w * inv_diffs) / jnp.sum(inv_diffs)

    x_points = jnp.linspace(0.0, 1.0, 101)
    max_order = 6

    print("Benchmarking jet vs nested differentiation:")
    print(f"  Function: Barycentric rational (5 support points)")
    print(f"  Evaluation points: {len(x_points)}")
    print(f"  Max derivative order: {max_order}")
    print()

    # Warm-up (JIT compilation)
    _ = compute_derivatives_at_points(test_fn, x_points[:5], max_order)

    # Benchmark jet
    start = time.time()
    result_jet = compute_derivatives_at_points(test_fn, x_points, max_order)
    jax.block_until_ready(result_jet)  # Wait for GPU/async completion
    time_jet = time.time() - start

    print(f"Jet (Taylor mode): {time_jet:.4f}s")
    print(f"  Result shape: {result_jet.shape}")
    print()

    # Nested approach (for comparison)
    def nested_nth_derivative(fn, x, n):
        """Nested jvp approach"""
        if n == 0:
            return fn(x)

        def compute_deriv(fn, n):
            if n == 0:
                return fn
            elif n == 1:
                def first_deriv(x):
                    _, df = jax.jvp(fn, (x,), (jnp.array(1.0),))
                    return df
                return first_deriv
            else:
                lower = compute_deriv(fn, n-1)
                def higher(x):
                    _, df = jax.jvp(lower, (x,), (jnp.array(1.0),))
                    return df
                return higher

        deriv_fn = compute_deriv(fn, n)
        return deriv_fn(x)

    # Warm-up
    _ = nested_nth_derivative(test_fn, x_points[0], 3)

    start = time.time()
    result_nested = []
    for order in range(max_order + 1):
        vals = jax.vmap(lambda x: nested_nth_derivative(test_fn, x, order))(x_points)
        result_nested.append(vals)
    result_nested = jnp.stack(result_nested)
    jax.block_until_ready(result_nested)
    time_nested = time.time() - start

    print(f"Nested jvp: {time_nested:.4f}s")
    print(f"  Result shape: {result_nested.shape}")
    print()

    # Check agreement
    max_diff = jnp.max(jnp.abs(result_jet - result_nested))
    print(f"Max difference between methods: {max_diff:.2e}")
    print(f"Speedup: {time_nested/time_jet:.1f}×")

    return result_jet, result_nested


if __name__ == "__main__":
    print("Testing JAX jet-based derivatives")
    print("=" * 60)
    print()

    # Simple test
    print("Test 1: sin(x) derivatives at x=1.0")
    fn = lambda x: jnp.sin(x)
    derivs = compute_all_derivatives_jet(fn, 1.0, max_order=4)

    print("Computed derivatives:")
    print(f"  f(1)   = {derivs[0]:.6f} (expected: {jnp.sin(1.0):.6f})")
    print(f"  f'(1)  = {derivs[1]:.6f} (expected: {jnp.cos(1.0):.6f})")
    print(f"  f''(1) = {derivs[2]:.6f} (expected: {-jnp.sin(1.0):.6f})")
    print(f"  f'''(1)= {derivs[3]:.6f} (expected: {-jnp.cos(1.0):.6f})")
    print()

    # Benchmark
    print("=" * 60)
    print()
    benchmark_jet_vs_nested()
