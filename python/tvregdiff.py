r"""
Minimal vendor of TVRegDiff wrapper to avoid PyPI.

This module exposes TVRegDiff(data, iter, alph, u0, scale, ep, dx, plotflag,
diagflag, precondflag, diffkernel, cgtol, cgmaxit) compatible with
stur86/tvregdiff signature.

Implementation adapted from Simone Sturniolo's tvregdiff (public domain MIT).
For our usage we delegate to a simplified squared-kernel variant when 'sq' is
requested; otherwise use 'abs'. This is a light wrapper sufficient for our
pipeline experiments; it is not a full reimplementation.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm


def TVRegDiff(data,
              iter,
              alph,
              u0=None,
              scale="small",
              ep=1e-6,
              dx=None,
              plotflag=False,
              diagflag=False,
              precondflag=False,
              diffkernel="abs",
              cgtol=1e-6,
              cgmaxit=100):
    r"""Return first derivative estimate u of data via TV-regularized diff.

    This is a compact variant of the Chartrand algorithm. It solves
    (in least-squares sense) min_u (1/2)||A u - y||^2 + alpha * \int phi(u') dx
    where phi is |.| (abs) or square.

    Parameters follow stur86/tvregdiff for compatibility.
    """
    # Defensive casting; tolerate strings/None
    try:
        iters = int(iter)
    except Exception:
        iters = int(float(iter)) if iter is not None and str(iter) != "" else 100
    try:
        alpha = float(alph)
    except Exception:
        alpha = 1e-2
    dk = str(diffkernel).lower().strip() if diffkernel is not None else "abs"
    dk = "sq" if dk in ("sq", "square") else "abs"

    y = np.asarray(data, dtype=float).ravel()
    n = y.size
    if dx is None or (isinstance(dx, str) and dx.strip() == ""):
        dx_val = 1.0 / max(n, 1)
    else:
        try:
            dx_val = float(dx)
        except Exception:
            dx_val = 1.0 / max(n, 1)

    # Finite difference operators
    # D: first-difference (n-1)x n, Neumann-like at ends via padding
    def grad1(v):
        return np.diff(v, append=v[-1]) / dx_val

    def div1(w):
        # adjoint of grad1 (approximate)
        out = np.zeros_like(y)
        out[0] = -w[0]
        out[1:-1] = w[0:-2] - w[1:-1]
        out[-1] = w[-2]
        return out / dx_val

    # A ~ integration operator inverse; we approximate Au ≈ cumulative sum of u * dx
    # and match to y after centering; this is a simplified surrogate adequate for our
    # comparative benchmarking.
    def apply_A(u):
        return np.cumsum(u) * dx_val

    def apply_At(v):
        # adjoint of A: reverse cumulative sum
        return np.cumsum(v[::-1])[::-1] * dx_val

    # Initialization
    if u0 is None or (isinstance(u0, str) and u0.strip() == ""):
        # naive derivative
        u = grad1(y)
    else:
        u = np.asarray(u0, dtype=float).ravel()
        if u.size != n:
            u = np.resize(u, n)

    # Iterative scheme: gradient descent with step by Barzilai-Borwein
    # F(u) = 0.5||A u - y||^2 + alpha * R(u)
    def reg_and_grad(u_vec):
        g = grad1(u_vec)
        if diffkernel == "sq":
            R = 0.5 * np.sum(g * g)
            w = g
        else:
            R = np.sum(np.sqrt(g * g + ep * ep))
            w = g / np.sqrt(g * g + ep * ep)
        grad_R = div1(w)
        return R, grad_R

    Au = apply_A(u)
    r = Au - (y - y.mean())
    R, grad_R = reg_and_grad(u)
    grad = apply_At(r) + alph * grad_R

    t_prev = 1.0
    u_prev = u.copy()

    for k in range(iters):
        # Take a step
        u_new = u - t_prev * grad

        # Update residuals and gradient
        Au = apply_A(u_new)
        r = Au - (y - y.mean())
        R, grad_R = reg_and_grad(u_new)
        grad_new = apply_At(r) + alpha * grad_R

        s = u_new - u
        y_bb = grad_new - grad
        denom = float(np.dot(s, y_bb))
        t_prev = max(1e-12, min(1e12, float(np.dot(s, s) / denom)) if denom != 0.0 else t_prev)

        u_prev = u
        u = u_new
        grad = grad_new

        tol = float(cgtol) if not isinstance(cgtol, str) else 1e-6
        if norm(grad) / max(norm(u), 1e-8) < tol:
            break

    return u


