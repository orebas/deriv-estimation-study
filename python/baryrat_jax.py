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

Ported by o3 reasoning model, 2025-01-20
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.scipy import linalg as jsp_linalg


################################################################################
# Barycentric rational function (minimal version)
################################################################################
class BarycentricRational:
    """
    Rational function in first-form barycentric representation

        r(x) =  Σ_j (w_j f_j)/(x − z_j)   /   Σ_j w_j/(x − z_j)

    Only evaluation (__call__) is implemented because JAX handles
    all derivatives automatically.
    """

    def __init__(self, z, f, w):
        z, f, w = map(jnp.asarray, (z, f, w))
        if not (z.shape == f.shape == w.shape):
            raise ValueError("z, f, w must have identical shapes")
        self.nodes   = z
        self.values  = f
        self.weights = w

    # ----------------------------------------------------------
    # Evaluation that can be traced / JIT-compiled by JAX
    # ----------------------------------------------------------
    def __call__(self, x):
        """
        Evaluate r(x) for scalar or array-like x (broadcasting works).

        Implementation is branch-free (except for a final reshape) so
        that it stays inside the JAX trace.
        """
        z, f, w = self.nodes, self.values, self.weights
        xv = jnp.asarray(x).ravel()

        # Nothing to do for empty input
        if xv.size == 0:
            return jnp.empty_like(xv).reshape(jnp.shape(x))

        D = xv[:, None] - z[None, :]          # pairwise differences
        same_node_mask = D == 0               # True where x == z_j

        # Replace the zeros by 1 so that division is well defined
        D_safe = jnp.where(same_node_mask, jnp.ones_like(D), D)
        C = 1.0 / D_safe

        num = C @ (w * f)
        den = C @ w
        r   = num / den                       # size = xv.size

        # For x that exactly coincide with nodes, force r(x)=f_j
        # JAX immutable update:
        node_xi, node_zi = jnp.nonzero(same_node_mask, size=0, fill_value=0)
        r = r.at[node_xi].set(f[node_zi])

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
