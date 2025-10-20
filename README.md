Derivative Estimation Study
===========================

Purpose
-------
This repository evaluates numerical methods for estimating derivatives (orders 1–7) from ODE trajectory data under varying noise levels and sampling densities. The primary objective is to assess which methods remain viable for high-order differentiation and to document objective, reproducible results to support a larger manuscript section.

Scope and Positioning
---------------------
- Signals: scalar observations from canonical ODEs (e.g., Lotka–Volterra), sampled uniformly.
- Orders: default up to 3 in pilots; framework supports higher (
  5–7) where method permits.
- Methods: Gaussian-process regression, rational and spectral approximants, splines and local smoothing, trend filtering, total-variation regularized differentiation (TVRegDiff), and selected ML smoothers. Automatic differentiation (AD) is included for targeted tests.
- Cross-language: Julia orchestrates generation, analysis, and native methods; Python provides a suite of additional techniques. Data interchange uses JSON and CSV.

Quickstart
----------
Prerequisites
- Julia (project-managed; packages are resolved on first run)
- Python 3.13 with `uv` to manage a local venv (`python/.venv`)

Setup
1) Python environment (managed by uv):
   - `uv venv python/.venv`
   - `uv pip install -r python/requirements.txt -p python/.venv/bin/python`
2) Julia environment:
   - `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`

Single-configuration pilot (smoke test)
Run a compact experiment end-to-end (ground truth, Python methods, Julia methods, metrics, CSV export):
```
DSIZE=201 MAX_DERIV=3 NOISE=0.01 PY_MAX_ORDER=3 PY_INCLUDE_MATERN=0 JULIA_NUM_THREADS=4 \
  julia --startup-file=no --project=. src/minimal_pilot.jl
```
Outputs:
- JSON I/O in `data/input/` and `data/output/`
- Aggregated CSV at `results/pilot/minimal_pilot_results.csv`

Re-running with different noise levels (non-blocking examples):
```
NOISE=0.0  DSIZE=201 MAX_DERIV=3 PY_MAX_ORDER=3 PY_INCLUDE_MATERN=0 julia --startup-file=no --project=. src/minimal_pilot.jl &
NOISE=0.01 DSIZE=201 MAX_DERIV=3 PY_MAX_ORDER=3 PY_INCLUDE_MATERN=0 julia --startup-file=no --project=. src/minimal_pilot.jl &
NOISE=0.05 DSIZE=201 MAX_DERIV=3 PY_MAX_ORDER=3 PY_INCLUDE_MATERN=0 julia --startup-file=no --project=. src/minimal_pilot.jl &
wait
```

Method Inventory (High-level)
-----------------------------
Julia-native (`src/julia_methods.jl`)
- AAA (rational): `AAA-HighPrec`, `AAA-LowPrec` (barycentric AA-A)
- GP-Julia-SE: analytic SE kernel GP (MLE, Hermite-based closed-form derivatives)
- Fourier-Interp: spectral interpolation baseline
- Dierckx-5: smoothing spline with derivative evaluation (k=5)
- Savitzky–Golay: local polynomial smoothing
- TrendFilter-k7: Lasso-based trend filtering (penalizes 8th derivative)
- Central-FD: finite-difference baseline
- TVRegDiff-Julia: total-variation-regularized differentiation via `NoiseRobustDifferentiation.tvdiff`

Python (`python/python_methods.py`)
- chebyshev: global Chebyshev fit with analytic derivatives
- fourier: trigonometric series with analytic n-th derivatives
- fourier_continuation: polynomial-trend-removed trigonometric fit
- gp_rbf_mean: GP posterior mean derivatives (SE kernel, closed-form)
- ad_trig: trigonometric model differentiated via autograd
- Butterworth_Python, FiniteDiff_Python, SavitzkyGolay_Python, SVR_Python, KalmanGrad_Python: legacy smoothing + quintic-spline differentiation family
- TVRegDiff_Python: Rick Chartrand’s TV-regularized differentiation (vendored wrapper)
- Matern kernels are supported but disabled by default for efficiency and smoothness-order limitations (enable via `PY_INCLUDE_MATERN=1`).

Data Formats and Configuration
------------------------------
- Input JSON (to Python): `data/input/<trial>.json`
  - `times`, `y_noisy`, `y_true`, `ground_truth_derivatives`
  - `config`: trial metadata
- Output JSON (from Python): `data/output/<trial>_results.json`
  - Per-method predictions keyed by derivative order (string keys)
  - Non-finite values are filtered prior to serialization
- Aggregation CSV: `results/pilot/minimal_pilot_results.csv`
  - Columns: `method, category, language, deriv_order, rmse, mae, timing, valid_points`

Endpoint Handling and Metrics
-----------------------------
- Errors reported as RMSE and MAE against analytic ground truth
- Endpoints are excluded from error computation; only interior points contribute
- Non-finite predictions are masked prior to aggregation

Performance and Numerical Stability
-----------------------------------
- GP (SE) in Julia: uses y-centering, x z-scoring, noise floor, and escalating jitter for Cholesky stability
- Python GPs: scikit-learn fitting with bounds; convergence warnings are expected on noiseless data; results are still recorded
- Matérn kernels: de-emphasized due to limited differentiability for high-order derivatives and runtime overhead

Citations and References
------------------------
- Total Variation Regularized Differentiation (TVRegDiff):
  - Julia implementation: `NoiseRobustDifferentiation.jl` [Documentation](https://adrianhill.de/NoiseRobustDifferentiation.jl/dev/)
  - Python port: `stur86/tvregdiff` [Repository](https://github.com/stur86/tvregdiff)

Reproducibility Notes
---------------------
- Julia is project-scoped (`--project=.`). Instantiate before running.
- Python is pinned via `python/requirements.txt`, with a uv-managed venv.
- Randomness: Unless otherwise noted, deterministic seeds are not set for smoothing-based methods; systematic comparisons rely on the same inputs.

Directory Guide
---------------
- `src/`: Julia orchestration and native methods (see `minimal_pilot.jl`, `julia_methods.jl`)
- `python/`: Python method implementations and environment
- `data/`: JSON inputs/outputs for cross-language exchange
- `results/`: Aggregated CSV and derived results
- `docs/`: Detailed method and protocol documentation (see below)

Next Steps
----------
- Extend high-order spline/P-spline penalties and RKHS derivative-penalty formulations
- Robust Fourier continuation for non-periodic signals
- Harden analytic GP for small-sample regimes (adaptive jitter heuristics)


