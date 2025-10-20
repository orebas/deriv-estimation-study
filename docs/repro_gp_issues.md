Gaussian Process Issues: AD-through-predictor and Positive Definiteness
========================================================================

Context
-------
We encountered two recurrent issues when differentiating Gaussian-process predictors at high orders:
1. Automatic differentiation (AD) through predictor closures (`predict_f`) in Julia.
2. Positive-definiteness (PD) failures in Cholesky factorization (small sample sizes, near-duplicate inputs, or extremely low noise assumptions).

AD-through-predictor
---------------------
- Attempting to propagate dual numbers (ForwardDiff/TaylorDiff) through `GaussianProcesses.jl` predictor closures can fail due to type instabilities or unsupported operations inside the prediction stack.
- Mitigation: for the SE kernel, we use an analytic derivative evaluator based on closed-form kernel derivatives (probabilists’ Hermite polynomials), avoiding AD through the predictor.

PD and Jitter
-------------
- Cholesky factorization can fail (PosDefException) when the kernel matrix is ill-conditioned.
- We escalate diagonal jitter (scaled to the median of the kernel diagonal) and enforce a noise floor during MLE.
- Additional stabilizers: y-centering (removes constant offset), x standardization (z-scoring), and a small diagonal added during likelihood optimization.

Observations
------------
- For 201-point grids with SE, analytic derivatives are stable with robust jitter escalation.
- For smaller grids (e.g., 51 points), factorization can remain sensitive; increasing base jitter improves stability with minor bias.

Recommendations
---------------
- Prefer analytic SE derivative evaluator for high-order queries.
- Apply data normalization and maintain a nonzero noise floor.
- Use jitter escalation schedules proportional to the median diagonal element of the kernel matrix.


