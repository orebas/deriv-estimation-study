# AAA-ROM Prototype Results (Phase 0)

## Summary

✅ **Workflow is functional end-to-end**
✅ **Orders 0-1 work well**
⚠️  **Higher-order derivatives have catastrophic error growth**

## Test Setup

- Signal: `y = sin(2πt)`, N=101, noise=1e-3
- Dense grid: 1000 points (cubic interpolation)
- AAA tolerance: 1e-13
- Methods tested: PyNumDiff-SavGol-Tuned, PyNumDiff-Butter-Tuned

## Results

### PyNumDiff-SavGol-Tuned

| Order | PyNumDiff Native | AAA-ROM | Status |
|-------|------------------|---------|--------|
| 0     | N/A              | 5.2e-04 | ✓ Excellent |
| 1     | 4.2e-02          | 5.0e-02 | ✓ Comparable |
| 2     | N/A              | 5.0e+02 | ✗ Terrible |
| 3     | N/A              | 1.2e+07 | ✗ Catastrophic |
| 4     | N/A              | 3.9e+11 | ✗ Catastrophic |
| 7     | N/A              | 5.8e+25 | ✗ Catastrophic |

### PyNumDiff-Butter-Tuned

| Order | PyNumDiff Native | AAA-ROM | Status |
|-------|------------------|---------|--------|
| 0     | N/A              | 4.5e-04 | ✓ Excellent |
| 1     | 1.5e-02          | 1.6e-02 | ✓ Comparable |
| 2     | N/A              | 1.5e+01 | ✗ Poor |
| 3     | N/A              | 4.5e+04 | ✗ Terrible |
| 4     | N/A              | 1.8e+08 | ✗ Catastrophic |
| 7     | N/A              | 3.4e+19 | ✗ Catastrophic |

## Analysis

### What's Working

1. **Python densification**: Successfully evaluates smoothed signals on dense grid
2. **JSON data exchange**: Python→Julia handoff works perfectly
3. **AAA fitting**: BaryRational.aaa builds interpolants successfully
4. **Orders 0-1**: Excellent accuracy, comparable to native methods

### What's Failing

**Higher-order derivatives explode exponentially:**
- Order 2: ~500x worse than expected
- Order 7: ~1e25x worse than expected

### Root Causes

1. **Overfitting residual noise**
   - Smoothed signal still has ~1e-3 noise
   - AAA fits this noise precisely (tol=1e-13)
   - Derivatives amplify noise exponentially: ε × ω^n

2. **Dense grid overparameterization**
   - 1000 points for 101 original points
   - May be creating too many AAA support points
   - More parameters → overfitting

3. **No regularization**
   - AAA minimizes interpolation error, not derivative error
   - No smoothing constraint on higher derivatives

## Proposed Solutions

### Option 1: Relax AAA Tolerance
- Current: `tol=1e-13` (near machine precision)
- Try: `tol=1e-6` to `1e-8`
- Rationale: Allow AAA to smooth over residual noise

### Option 2: Reduce Dense Grid Size
- Current: 1000 points
- Try: 200-500 points
- Rationale: Fewer parameters → less overfitting

### Option 3: Pre-smooth Before Densification
- Apply additional smoothing to remove residual noise
- E.g., moving average, light Gaussian filter
- Rationale: Give AAA cleaner data to fit

### Option 4: Use Original Grid (No Densification)
- Fit AAA directly on original 101 points
- Rationale: Trust PyNumDiff's smoothing, avoid interpolation

### Option 5: Hybrid Approach
- Use AAA for orders 0-2 only
- For orders 3+, use finite differences on AAA-smoothed signal
- Rationale: Limit derivative depth where instability occurs

## Recommendations

**Immediate testing:**
1. Try Option 1 + Option 2: `tol=1e-7, n_dense=300`
2. Try Option 4: No densification, fit on original grid

**If those fail:**
- Revisit whether AAA-ROM is viable for higher derivatives
- Consider alternative: Fit global polynomial instead of rational
- Or: Limit AAA-ROM to orders 0-2, report NaN for higher orders

## Next Steps

1. Implement tolerance sweep: test tol ∈ [1e-6, 1e-8, 1e-10, 1e-13]
2. Implement grid size sweep: test n_dense ∈ [101, 200, 500, 1000]
3. Compare against native SavGol orders 2-7 (we have these!)
4. Document when AAA-ROM is appropriate vs. when to use native methods

## Files Created

```
methods/python/aaa_rom_utils.py          # Densification utilities
methods/julia/aaa_rom/aaa_rom_wrapper.jl # Julia AAA-ROM evaluator
test_aaa_rom_prototype.py                 # End-to-end test
test_aaa_rom_debug.jl                     # Julia debugging
build/data/aaa_rom_input/*.json          # Data exchange files
```

## Conclusion

**Phase 0 prototype is functional** but reveals a fundamental challenge:
- AAA interpolation is *too accurate* for noisy derivative estimation
- Works well for orders 0-1, fails catastrophically for orders 2+
- Need regularization or alternative approach for higher derivatives

This is actually a valuable finding for the paper: *"High-precision interpolants (AAA) can amplify residual noise in derivative estimation. Regularization or tolerance tuning is essential."*
