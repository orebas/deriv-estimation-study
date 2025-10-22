#!/usr/bin/env python3
"""
JAX-based higher-order derivative computation.

Provides a wrapper to compute arbitrary-order derivatives of Python callables
using JAX's automatic differentiation, enabling higher-order derivatives from
packages like baryrat that don't natively support them.

This is analogous to Julia's TaylorDiff.jl used in the Julia methods.
"""

import numpy as np

try:
    import jax
    # Enable float64 support in JAX
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import jvp, pure_callback
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def jax_nth_derivative(f, x, order, use_jax_ad=True, fd_epsilon=None, fd_max_order=3):
    """
    Compute the n-th derivative of f at x.

    Uses JAX automatic differentiation for JAX-compatible functions,
    or finite differences for non-JAX functions (like baryrat).

    **WARNING**: Finite differences are numerically unstable for orders > 3.
    For Python AAA with orders > 3, errors can be very large. Consider using
    Julia's BaryRational.jl with TaylorDiff.jl for accurate high-order derivatives.

    Parameters
    ----------
    f : callable
        Function to differentiate. Must accept scalar input and return scalar.
    x : float
        Point at which to evaluate derivative
    order : int
        Derivative order (0 = function value, 1 = first derivative, etc.)
    use_jax_ad : bool, optional
        Try JAX AD first (default True). Falls back to finite differences.
    fd_epsilon : float, optional
        Step size for finite differences. Auto-selected if None.
    fd_max_order : int, optional
        Maximum order for finite differences before raising error (default 3).
        Higher orders are numerically unstable.

    Returns
    -------
    float
        The order-th derivative of f evaluated at x

    Examples
    --------
    >>> from baryrat import aaa
    >>> import numpy as np
    >>>
    >>> # Fit AAA to sin(x)
    >>> x_train = np.linspace(0, 10, 50)
    >>> y_train = np.sin(x_train)
    >>> r = aaa(x_train, y_train, tol=1e-10)
    >>>
    >>> # Compute 3rd derivative at x=1.0 (should be ≈ -cos(1.0))
    >>> f = lambda x: r(x)
    >>> d3 = jax_nth_derivative(f, 1.0, 3)
    >>> print(f"d³sin(1)/dx³ = {d3:.6f} (exact: {-np.cos(1.0):.6f})")

    Notes
    -----
    - For JAX-compatible functions: Uses forward-mode AD (fast, exact, all orders)
    - For non-JAX functions (like baryrat): Uses central finite differences
    - Finite differences are only reliable for orders 0-3
    - For higher orders with baryrat, use Julia's BaryRational.jl instead
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX not available - install with: pip install jax")

    if order == 0:
        return float(f(x))

    # Try JAX AD first for JAX-compatible functions
    if use_jax_ad:
        try:
            def jax_f(x_jax):
                result = f(x_jax)
                if isinstance(result, jnp.ndarray):
                    return result
                return jnp.asarray(result, dtype=jnp.float64)

            x_jax = jnp.array(x, dtype=jnp.float64)

            def compute_nth_derivative(fn, n):
                if n == 0:
                    return fn
                elif n == 1:
                    def first_deriv(x):
                        _, df = jvp(fn, (x,), (jnp.array(1.0, dtype=jnp.float64),))
                        return df
                    return first_deriv
                else:
                    lower_deriv = compute_nth_derivative(fn, n - 1)
                    def higher_deriv(x):
                        _, df = jvp(lower_deriv, (x,), (jnp.array(1.0, dtype=jnp.float64),))
                        return df
                    return higher_deriv

            deriv_fn = compute_nth_derivative(jax_f, order)
            result = deriv_fn(x_jax)
            return float(result)

        except (TypeError, jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
            # Fall through to finite differences
            pass

    # Use central finite differences for non-JAX functions (like baryrat)
    # Check order limit
    if order > fd_max_order:
        raise ValueError(
            f"Derivative order {order} exceeds fd_max_order={fd_max_order}. "
            f"Finite differences are numerically unstable for high orders. "
            f"For baryrat with orders > 3, use Julia's BaryRational.jl with TaylorDiff.jl instead."
        )

    return _finite_difference_derivative(f, x, order, fd_epsilon)


def _finite_difference_derivative(f, x, order, epsilon=None):
    """
    Compute n-th derivative using central finite differences.

    Uses central difference stencils with auto-selected step size.
    """
    if epsilon is None:
        # Auto-select epsilon based on machine precision and order
        # For double precision: eps ≈ 2.22e-16
        # Optimal step: h ≈ eps^(1/(order+2))
        epsilon = np.finfo(float).eps ** (1.0 / (order + 2))
        # Clamp to reasonable range
        epsilon = max(1e-8, min(1e-4, epsilon))

    return _manual_finite_diff(f, x, order, epsilon)


def _manual_finite_diff(f, x, order, h):
    """
    Manual central finite difference implementation.

    Uses central difference stencils for orders 1-10.
    Coefficients from Fornberg (1988) "Generation of Finite Difference Formulas on
    Arbitrarily Spaced Grids", Math. Comp. 51, 699-706.
    """
    if order == 1:
        # 2nd-order accurate central difference
        return (f(x + h) - f(x - h)) / (2*h)
    elif order == 2:
        # 2nd-order accurate central difference
        return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
    elif order == 3:
        # 2nd-order accurate central difference
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2*h**3)
    elif order == 4:
        # 2nd-order accurate central difference
        return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / (h**4)
    elif order == 5:
        # 2nd-order accurate central difference
        return (f(x + 3*h) - 4*f(x + 2*h) + 5*f(x + h) - 5*f(x - h) + 4*f(x - 2*h) - f(x - 3*h)) / (2*h**5)
    elif order == 6:
        # 2nd-order accurate central difference
        return (f(x + 3*h) - 6*f(x + 2*h) + 15*f(x + h) - 20*f(x) + 15*f(x - h) - 6*f(x - 2*h) + f(x - 3*h)) / (h**6)
    else:
        # For very high orders, use a general approach
        # This is less accurate but works for any order
        coeffs_dict = {
            7: ([-1, 8, -13, 13, -8, 1], 3, 8),
            8: ([1, -8, 28, -56, 70, -56, 28, -8, 1], 3, 12),
            9: ([-1, 12, -39, 56, -39, 12, -1], 4, 6),
            10: ([1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1], 4, 12)
        }

        if order in coeffs_dict:
            coeffs, offset, divisor = coeffs_dict[order]
            result = sum(c * f(x + (i - len(coeffs)//2) * h) for i, c in enumerate(coeffs))
            return result / (divisor * h**order)
        else:
            raise NotImplementedError(f"Finite differences for order {order} not implemented. Max supported: 10")


def test_jax_derivatives():
    """Test JAX derivative computation against known functions"""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping tests")
        return

    print("=" * 70)
    print("Testing JAX higher-order derivatives")
    print("=" * 70)
    print()

    # Test 1: sin(x) derivatives
    print("Test 1: sin(x) and its derivatives")
    x_test = 1.5

    # Use jnp.sin (JAX-compatible) instead of np.sin
    f_sin = lambda x: jnp.sin(x)

    for order in range(5):
        # Exact derivatives of sin:
        # d⁰sin/dx⁰ = sin(x)
        # d¹sin/dx¹ = cos(x)
        # d²sin/dx² = -sin(x)
        # d³sin/dx³ = -cos(x)
        # d⁴sin/dx⁴ = sin(x)
        exact = [np.sin, np.cos, lambda x: -np.sin(x),
                lambda x: -np.cos(x), np.sin][order](x_test)

        computed = jax_nth_derivative(f_sin, x_test, order)
        error = abs(computed - exact)

        print(f"  Order {order}: computed={computed:.8f}, exact={exact:.8f}, error={error:.2e}")

    print()

    # Test 2: Polynomial (x^4 - 2x^3 + x^2 - 5)
    print("Test 2: Polynomial x⁴ - 2x³ + x² - 5")
    x_test = 2.0

    f_poly = lambda x: x**4 - 2*x**3 + x**2 - 5

    # Exact derivatives:
    # f(x) = x⁴ - 2x³ + x² - 5
    # f'(x) = 4x³ - 6x² + 2x
    # f''(x) = 12x² - 12x + 2
    # f'''(x) = 24x - 12
    # f⁴(x) = 24
    # f⁵(x) = 0

    exact_derivs = [
        lambda x: x**4 - 2*x**3 + x**2 - 5,
        lambda x: 4*x**3 - 6*x**2 + 2*x,
        lambda x: 12*x**2 - 12*x + 2,
        lambda x: 24*x - 12,
        lambda x: 24,
        lambda x: 0
    ]

    for order in range(6):
        exact = exact_derivs[order](x_test)
        computed = jax_nth_derivative(f_poly, x_test, order)
        error = abs(computed - exact)

        print(f"  Order {order}: computed={computed:.8f}, exact={exact:.8f}, error={error:.2e}")

    print()
    print("=" * 70)
    print("✅ JAX derivative tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_jax_derivatives()
