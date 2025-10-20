# Experimental Protocols and Metrics

## Signal Generation
- Systems: canonical ODEs (initial focus: Lotka–Volterra) defined in `src/ground_truth.jl`.
- Sampling: uniform grid on `[t0, t1]` with `DSIZE` points.
- Orders: `0..MAX_DERIV` (default 3 in pilots), higher on demand.
- Noise: additive measurement noise (`NOISE` as standard deviation fraction of signal scale), applied to `y_true` to produce `y_noisy`.

## Cross-language Workflow
1. Julia generates ground truth (values and analytic derivatives).
2. JSON input written to `data/input/<trial>.json`.
3. Python script `python/python_methods.py` reads input, executes configured methods, and writes `data/output/<trial>_results.json`.
4. Julia ingests Python results, evaluates Julia-native methods, computes metrics, and writes aggregated CSV to `results/pilot/minimal_pilot_results.csv`.

## Configuration (Environment Variables)
- `DSIZE` (Int): number of sample points (e.g., 51, 201).
- `MAX_DERIV` (Int): maximum derivative order to evaluate (default 3).
- `NOISE` (Float): noise level for `y_noisy` (e.g., 0.0, 0.01, 0.05).
- `PY_MAX_ORDER` (Int): cap Python-side derivative orders.
- `PY_METHODS` (CSV): restrict Python methods to a subset.
- `PY_EXCLUDE` (CSV): exclude specific Python methods.
- `PY_INCLUDE_MATERN` (0/1): include Matérn GP methods (default 0).
- Method-specific overrides are documented in `docs/methods.md`.

## Metrics and Aggregation
- For each method and derivative order:
  - Non-finite predictions are masked.
  - Endpoints are excluded (error computed on interior points only).
  - RMSE and MAE are computed against analytic ground truth at evaluation nodes.
- Results are appended to `results/pilot/minimal_pilot_results.csv` with columns:
  - `method, category, language, deriv_order, rmse, mae, timing, valid_points`.

## Interpretation Guidelines
- First-order performance is necessary but not sufficient; we report the highest order for which results remain qualitatively reasonable.
- Locally low-order polynomial methods are expected to degrade rapidly for higher orders.
- Gaussian-process SE kernels and specialized regularization (e.g., TVRegDiff) are expected to be more robust.
- Matérn kernels may underperform for derivative orders exceeding their smoothness (ν).

## Reproducibility
- The Julia environment is project-scoped; invoke with `--project=.`.
- The Python environment is managed via uv; install with `uv pip install -r python/requirements.txt -p python/.venv/bin/python`.
- Random seeds are not forced unless specified; repeated runs with identical inputs should produce comparable (but not bit-identical) results across smoothing routines.
