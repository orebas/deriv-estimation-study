# Data Formats, Schemas, and Environment Variables

## JSON Schema (Input to Python)
`data/input/<trial>.json`
- `times` (Float64[]): evaluation nodes (and training inputs).
- `y_noisy` (Float64[]): observed signal at `times`.
- `y_true` (Float64[]): noise-free signal (for reference only).
- `ground_truth_derivatives` (Dict{String, Float64[]}): orders mapped to arrays.
- `config` (Dict): trial metadata (e.g., `trial_id`, `data_size`, `noise_level`, `tspan`).

## JSON Schema (Output from Python)
`data/output/<trial>_results.json`
- `methods` (Dict): per-method results with fields:
  - `predictions` (Dict{String, Float64[]}): derivative order → predictions.
  - `failures` (Dict{String, String}): error messages for failed orders.
  - `meta` (Dict): method-specific metadata (hyperparameters, etc.).
  - `success` (Bool): overall success flag.
  - `timing` (Float64): wall time in seconds.
- Non-finite values are filtered prior to serialization; orders lacking finite predictions may be omitted.

## CSV Aggregation
`results/pilot/minimal_pilot_results.csv`
- Columns: `method, category, language, deriv_order, rmse, mae, timing, valid_points`.
- Endpoints excluded; interior indices only.

## Environment Variables (Summary)
- `DSIZE`, `MAX_DERIV`, `NOISE`, `PY_MAX_ORDER`, `PY_METHODS`, `PY_EXCLUDE`, `PY_INCLUDE_MATERN`.
- Method-specific toggles (examples):
  - `FC_TREND_DEG` (Fourier continuation trend degree).
  - `TVREG_ITERS`, `TVREG_ALPHA`, `TVREG_SCALE`, `TVREG_DIFFKERNEL`, `TVREG_CGTOL`, `TVREG_CGMAXIT` (TVRegDiff Python).

## Notes
- Data exchange uses float arrays without NaN/Inf; Python sanitizes outputs to avoid JSON parsers rejecting non-finite values.
- Julia side double-checks finiteness and masks invalid entries during metric computation.
