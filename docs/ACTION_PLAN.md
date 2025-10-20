# Action Plan: Code Review Recommendations
**Generated:** October 17, 2025
**Based on:** Multi-model code review (Gemini Pro, O3, Claude)

---

## Priority 1: Critical Issues (Complete This Week)

### ✅ COMPLETED
- [x] Fix TVRegDiff dx parameter bug (RMSE 10^92 → reasonable)
- [x] Refactor TVRegDiff to iterative self-contained approach
- [x] Add error logging to Python `_clean_predictions`
- [x] Fix tvregdiff.py docstring warning

### 🔲 REMAINING

#### Task 1.1: Validate Iterative TVRegDiff Implementation
**Urgency:** High
**Effort:** 4-6 hours
**Owner:** TBD

**Acceptance Criteria:**
- [ ] Run minimal_pilot.jl with NOISE=0.0, compare against AAA-HighPrec baseline
- [ ] Test with varying noise levels (0.0, 0.01, 0.05, 0.1)
- [ ] Document RMSE performance across all derivative orders (0-3)
- [ ] Verify no catastrophic failures (RMSE < 100 for order 3)
- [ ] Create comparison table: iterative vs. old spline approach

**Commands:**
```bash
# Zero noise test
NOISE=0.0 DSIZE=51 MAX_DERIV=3 julia src/minimal_pilot.jl

# Moderate noise test
NOISE=0.05 DSIZE=51 MAX_DERIV=3 julia src/minimal_pilot.jl

# High noise test
NOISE=0.1 DSIZE=51 MAX_DERIV=3 julia src/minimal_pilot.jl
```

**Files Affected:**
- `src/minimal_pilot.jl` (run tests)
- `results/pilot/minimal_pilot_results.csv` (verify output)

**Success Metrics:**
- Order 1 RMSE < 2.0 (zero noise)
- Order 2 RMSE < 20.0 (zero noise)
- Order 3 RMSE < 100.0 (zero noise)
- No Inf/NaN values

---

#### Task 1.2: Fix GP-Julia-SE Overflow for Higher Orders
**Urgency:** High
**Effort:** 3-4 hours
**Owner:** TBD

**Problem:** Order 2+ derivatives return Inf due to:
- Hermite polynomial overflow
- Scaling factor `(ℓ̂^(-n))` grows exponentially

**Solution Approach:**
1. Add overflow protection in `eval_nth_deriv`:
```julia
# In src/julia_methods.jl:252-269
scale = (σf̂^2) * (ℓ̂ ^ (-n))
# Add clamping:
scale = min(scale, 1e6)  # Prevent overflow
```

2. Add early return for unstable orders:
```julia
if n >= 2 && maximum(abs.(u)) > 10.0
    @warn "GP-Julia-SE unstable for order $n, large u values"
    return NaN
end
```

3. Document limitations in docstring

**Acceptance Criteria:**
- [ ] No Inf values for orders 2-3 in diagnostic tests
- [ ] RMSE finite (may be high, that's okay)
- [ ] Warning logged when stability threshold exceeded
- [ ] Docstring updated with order limitations

**Test Command:**
```bash
julia src/diagnostic_test.jl 2>&1 | grep "GP-Julia-SE"
```

**Files Affected:**
- `src/julia_methods.jl:147-272`

---

## Priority 2: High-Impact Improvements (Next 2 Weeks)

#### Task 2.1: Add Comprehensive Test Suite
**Urgency:** Medium
**Effort:** 3-4 days
**Owner:** TBD

**Scope:**
1. **Unit Tests** (`test/unit/`)
   - Test each method independently
   - Validate order 0 returns input (interpolation)
   - Test edge cases (n=3 data points, n=1000 data points)

2. **Regression Tests** (`test/regression/`)
   - Known analytic functions (polynomials, sin, exp)
   - Compare against expected derivatives
   - Store baseline results, detect regressions

3. **Integration Tests** (`test/integration/`)
   - End-to-end pilot runs
   - Python-Julia interop validation

**Structure:**
```
test/
├── runtests.jl          # Main test runner
├── unit/
│   ├── test_tvregdiff.jl
│   ├── test_gp.jl
│   ├── test_aaa.jl
│   └── ...
├── regression/
│   ├── test_polynomials.jl
│   ├── test_trig.jl
│   └── baselines.json
└── integration/
    └── test_full_pipeline.jl
```

**Acceptance Criteria:**
- [ ] 80%+ code coverage for Julia methods
- [ ] All methods tested against polynomial ground truth (degree 3)
- [ ] Regression baselines stored in version control
- [ ] CI/CD integration (optional but recommended)

**Resources:**
- Use Julia Test stdlib: `using Test`
- Consider Test.jl for advanced features

---

#### Task 2.2: Document Hyperparameter Sensitivity
**Urgency:** Medium
**Effort:** 2-3 days
**Owner:** TBD

**Deliverables:**
1. **Sensitivity Analysis Report** (`docs/HYPERPARAMETER_TUNING.md`)
   - For TVRegDiff: vary alpha (1e-4 to 1e-1), iters (10 to 500)
   - For GP methods: vary kernel lengthscale
   - For splines: vary smoothing parameter s

2. **Parameter Selection Guide**
   - Rule of thumb based on noise level
   - Example configurations for common scenarios

3. **Update Method Docstrings**
   - Add parameter guidance to each method

**Methodology:**
```julia
# Example sweep for TVRegDiff
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
iters_values = [10, 50, 100, 200, 500]
noise_levels = [0.0, 0.01, 0.05, 0.1]

for noise in noise_levels
    for alpha in alphas
        for iters in iters_values
            # Run method, record RMSE
        end
    end
end
```

**Acceptance Criteria:**
- [ ] Heatmaps showing RMSE vs. (alpha, iters) for each noise level
- [ ] Recommended parameter ranges documented
- [ ] Guidance added to README

---

## Priority 3: Code Quality (Next Month)

#### Task 3.1: Refactor Common Utilities
**Effort:** 1 day

**Targets:**
- Extract shared linear interpolation logic
- Common validation functions (grid uniformity check)
- Standardized error handling

**Example:**
```julia
# src/utils.jl (new file)
function validate_uniform_grid(x; tol=0.1)
    diffs = diff(x)
    @assert maximum(diffs) / minimum(diffs) < (1 + tol) "Non-uniform grid detected"
end

function linear_interp(x_data, y_data, x_eval)
    # Shared implementation
end
```

---

#### Task 3.2: Performance Optimization
**Effort:** 1 week

**Targets:**
1. Profile current implementation (`@profile, ProfileView.jl`)
2. Parallelize independent method evaluations
3. Cache expensive GP computations
4. Consider warm-start for iterative methods

**Acceptance Criteria:**
- [ ] 20%+ speedup on full pilot run
- [ ] No accuracy degradation

---

## Priority 4: Nice-to-Have (Future)

- [ ] Interactive visualization dashboard
- [ ] Automatic hyperparameter tuning (Bayesian optimization)
- [ ] Support for multivariate derivatives
- [ ] GPU acceleration for large-scale studies

---

## Success Metrics

### Week 1 (Priority 1)
- ✅ TVRegDiff validated (RMSE < 100 for order 3, zero noise)
- ✅ GP-Julia-SE no longer produces Inf

### Week 2-3 (Priority 2)
- ✅ Test suite passing (80%+ coverage)
- ✅ Hyperparameter guide published

### Month 1 (Priority 3)
- ✅ Code refactored (20% fewer lines, same functionality)
- ✅ Performance improved (20%+ faster)

---

## Getting Help

- **Technical Questions:** Check `docs/CODE_REVIEW_SUMMARY.md`
- **Implementation Details:** See inline comments in `src/julia_methods.jl`
- **Testing Examples:** Refer to `src/diagnostic_test.jl`

---

## Revision History

- **2025-10-17:** Initial action plan based on multi-model review
- **TBD:** Update after Task 1.1 validation
