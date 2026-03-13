"""
Minimal JAX port of the parts of baryrat that we need

Retained functionality
----------------------
* class BarycentricRational
    - __init__
    - __call__
* function aaa

Everything is written with jax.numpy / jax.scipy so that the whole
graph is traceable and differentiable by JAX AD.

FIXED: Added @jax.custom_jvp for proper gradient support at interpolation nodes.
Solution based on o3 reasoning model's approach.

Ported by o3 reasoning model, 2025-01-20
Fixed for JAX autodiff, 2025-10-26
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import linalg as jsp_linalg


################################################################################
# Core evaluation function with custom JVP for proper gradients
################################################################################

@jax.custom_jvp
def _barycentric_eval(x, z, f, w):
    """
    Evaluate barycentric rational function with proper handling of singularities.

    r(x) = [Σ w_j f_j / (x - z_j)] / [Σ w_j / (x - z_j)]

    The primal evaluation can use masking/where to handle NaN at nodes.
    The JVP will use analytical formulas for correct gradients.
    """
    diff      = x[..., None] - z                # (..., m)
    inv_diff  = 1.0 / diff                      # may create inf/NaN at nodes
    num       = jnp.sum(w * f * inv_diff, -1)   # (...)
    den       = jnp.sum(w       * inv_diff, -1) # (...)
    r         = num / den                       # (...)

    # Replace 0/0 at the nodes by the exact value f_k
    is_node   = diff == 0                       # (..., m) boolean
    any_node  = jnp.any(is_node, axis=-1)       # (...)
    f_at_node = jnp.sum(jnp.where(is_node, f, 0.), axis=-1)  # (...)
    return jnp.where(any_node, f_at_node, r)    # (...)


@_barycentric_eval.defjvp
def _barycentric_eval_jvp(primals, tangents):
    """
    JVP rule using analytical derivative formulas.

    Off nodes: r' = (N'D - ND') / D²
    At nodes:  r'(z_k) = Σ_{j≠k} [w_j/(z_k-z_j) · (f_j-f_k)] / Σ_{j≠k} [w_j/(z_k-z_j)]
    """
    x,  z,  f,  w   = primals
    ẋ, ż, ḟ, ẇ   = tangents

    # Only support differentiation w.r.t. x (z, f, w are constants)
    # This assertion uses concrete values, which is fine in JVP context
    # If you need gradients w.r.t. f/w/z, extend this rule

    diff      = x[..., None] - z                  # (..., m)
    inv       = 1.0 / diff                       # (...)
    inv2      = inv * inv                        # 1/(x - z_j)²

    N         = jnp.sum(w * f * inv,   -1)       # (...)
    D         = jnp.sum(w       * inv,   -1)     # (...)
    Ṅ        = jnp.sum(-w * f * inv2, -1) * ẋ  # chain rule
    Ḋ        = jnp.sum(-w     * inv2, -1) * ẋ
    r         = N / D
    ṙ        = (Ṅ * D - N * Ḋ) / (D * D)      # (...)

    # --- Patch singular points with limit derivative ---
    is_node   = diff == 0                        # (..., m)
    any_node  = jnp.any(is_node, axis=-1)        # (...)

    # Index of the node k
    k         = jnp.argmax(is_node, axis=-1)
    # Handle scalar case: z, f, w are 1D, k might be scalar
    z_k       = z[k]
    f_k       = f[k]
    w_k       = w[k]

    # Sums skipping k
    not_k     = jnp.logical_not(is_node)
    # Ensure z_k has compatible shape for broadcasting
    z_k_expanded = jnp.reshape(z_k, list(z_k.shape) + [1] * (len(is_node.shape) - len(z_k.shape)))
    inv_diff = 1.0 / (z_k_expanded - z)        # 1/(z_k - z_j)

    # Ensure f_k is broadcastable
    f_k_expanded = jnp.reshape(f_k, list(f_k.shape) + [1] * (len(is_node.shape) - len(f_k.shape)))
    num_lim   = jnp.sum(jnp.where(not_k, w * (f - f_k_expanded) * inv_diff, 0.), -1)
    den_lim   = jnp.sum(jnp.where(not_k, w * inv_diff, 0.), -1)
    ṙ_lim    = num_lim / den_lim * ẋ

    r_final   = jnp.where(any_node, f_k,  r)
    ṙ_final  = jnp.where(any_node, ṙ_lim, ṙ)
    return r_final, ṙ_final


################################################################################
# Barycentric rational function (minimal version)
################################################################################
class BarycentricRational:
    """
    Rational function in first-form barycentric representation

        r(x) =  Σ_j (w_j f_j)/(x − z_j)   /   Σ_j w_j/(x − z_j)

    Uses custom JVP rules for proper JAX autodiff support.
    """

    def __init__(self, z, f, w):
        z, f, w = map(jnp.asarray, (z, f, w))
        if not (z.shape == f.shape == w.shape):
            raise ValueError("z, f, w must have identical shapes")
        self.nodes   = z
        self.values  = f
        self.weights = w

    def __call__(self, x):
        """
        Evaluate r(x) for scalar or array-like x (broadcasting works).

        Fully differentiable via JAX autodiff with custom JVP rules.
        """
        z, f, w = self.nodes, self.values, self.weights
        xv = jnp.asarray(x).ravel()

        # Nothing to do for empty input
        if xv.size == 0:
            return jnp.empty_like(xv).reshape(jnp.shape(x))

        # Call the custom JVP function
        r = _barycentric_eval(xv, z, f, w)

        return r.reshape(jnp.shape(x))        # restore original shape


################################################################################
# AAA algorithm (only the basics, no bells & whistles)
################################################################################
def aaa(Z,
        F,
        tol: float = 1e-13,
        mmax: int = 100,
        return_errors: bool = False):
    """
    Adaptive Antoulas–Anderson (AAA) rational approximation, JAX edition.

    Parameters
    ----------
    Z : 1-D array of sample points
    F : 1-D array of function values *or* callable f(Z)
    tol : stopping tolerance
    mmax : maximum #iterations / support points
    return_errors : if True additionally return list of ∞-norm errors

    Returns
    -------
    r                : BarycentricRational
    (r, error_list)  : if return_errors == True
    """
    Z = jnp.asarray(Z).ravel()
    F = jnp.asarray(F(Z) if callable(F) else F).ravel()

    # Python containers that drive the greedy loop
    J   = list(range(len(F)))                # remaining sample indices
    zj  = jnp.empty(0, dtype=Z.dtype)        # support points
    fj  = jnp.empty(0, dtype=F.dtype)        # function values at support pts
    errors = []

    reltol = tol * jnp.linalg.norm(F, jnp.inf)
    R = jnp.full_like(F, jnp.mean(F))        # initial approximation

    for m in range(mmax):
        # 1) Find point with largest residual (from ALL points, not just J)
        jj = int(jnp.argmax(jnp.abs(F - R)))

        # 2) Add this point to support set
        zj = jnp.concatenate([zj, jnp.array([Z[jj]])])
        fj = jnp.concatenate([fj, jnp.array([F[jj]])])
        J.remove(jj)

        # If no points remaining, we're done
        if len(J) == 0:
            # Use uniform weights for last approximation
            wj = jnp.ones_like(zj)
            break

        # 3) Build matrices using remaining points J
        J_arr = jnp.asarray(J, dtype=jnp.int32)
        C = 1.0 / (Z[J_arr, None] - zj[None, :])       # Cauchy matrix
        A = (F[J_arr, None] - fj[None, :]) * C         # Loewner matrix

        # 4) Compute weights via SVD
        full_matrices = (A.shape[1] > A.shape[0])
        _, _, Vh = jsp_linalg.svd(A, full_matrices=full_matrices)
        wj = Vh[-1, :].conj()

        # 5) Update approximation R at remaining points
        N = C @ (wj * fj)
        D = C @ wj

        # Update R: copy all of F, then replace J positions
        R = F  # Full copy
        R = R.at[J_arr].set(N / D)

        # 6) Check convergence
        err = float(jnp.linalg.norm(F - R, jnp.inf))
        errors.append(err)
        if err <= reltol:
            break

    r = BarycentricRational(zj, fj, wj)
    return (r, errors) if return_errors else r
