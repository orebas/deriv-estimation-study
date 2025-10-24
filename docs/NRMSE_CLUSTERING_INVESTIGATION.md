# NRMSE Clustering Investigation

## Problem Statement

**38.5% of all benchmark records have NRMSE > 0.95**, suggesting many methods are failing silently by returning near-constant or near-zero predictions instead of raising errors.

## Key Findings

### 1. The Magic Number: NRMSE ≈ 0.994937

**156 records** have NRMSE exactly equal to 0.994937 (within numerical precision).

This value appears suspiciously consistent across:
- 6 different methods
- All derivative orders 1-7 (most common at orders 5-7)
- All noise levels (increases at higher noise)

**Affected Methods (with exact NRMSE = 0.994937):**
- Savitzky-Golay: 49 occurrences
- TrendFilter-k2: 42 occurrences
- TrendFilter-k7: 42 occurrences
- Fourier-FFT-Adaptive: 9 occurrences
- AAA-Adaptive-Wavelet: 7 occurrences
- AAA-Adaptive-Diff2: 7 occurrences

### 2. Broader Pattern: NRMSE > 0.95

**574 records (38.5%)** have NRMSE > 0.95, indicating widespread failures.

**Top 10 Most Problematic Methods:**
1. Fourier-FFT-Adaptive: 53 records
2. Savitzky-Golay: 49 records
3. AAA-JAX-Adaptive-Wavelet: 44 records
4. AAA-JAX-Adaptive-Diff2: 44 records
5. TrendFilter-k7: 42 records
6. TrendFilter-k2: 42 records
7. AAA-HighPrec: 40 records
8. SavitzkyGolay_Python: 35 records
9. Chebyshev-AICc: 35 records
10. chebyshev: 35 records

### 3. Pattern by Derivative Order

Failures concentrate at **high-order derivatives**:

```
Order 0:   2 failures (1% of order-0 records)
Order 1:  22 failures (11%)
Order 2:  58 failures (29%)
Order 3:  88 failures (44%)
Order 4:  94 failures (47%)
Order 5: 110 failures (55%)
Order 6: 101 failures (51%)
Order 7:  99 failures (50%)
```

**Interpretation:** Methods increasingly fail as derivative order increases.

### 4. No NaN Masking Issues

- All records show `valid_points = 99 = total_points`
- No records with `valid_points = 0`
- Methods are returning **numeric values**, not NaNs

This means the issue is **not** about NaN handling or masking. The methods are producing actual numbers, but those numbers are wrong.

### 5. What NRMSE ≈ 0.995 Means

**Theory**: NRMSE = RMSE / std(y_true)

For constant predictions (predicting the mean):
- RMSE = std(y_true)
- NRMSE = 1.0 exactly

For near-constant predictions or predictions close to a single value:
- NRMSE ≈ 0.995-1.0

**The value 0.994937 suggests predictions are nearly constant but with small variations.**

## Investigation Next Steps

### Recommended Diagnostic Actions

1. **Check actual prediction values** for a few failing cases
   - Look at raw prediction arrays from a Savitzky-Golay order-2 run
   - Verify if they're zeros, constants, or near-constants
   - Check if there's a pattern (e.g., all values ≈ 0.1)

2. **Review method implementations** for silent failures
   - Check if methods have fallback behavior that returns default values
   - Look for try-catch blocks that might suppress errors
   - Verify error handling in method wrappers

3. **Check for method limitations** at high orders
   - Savitzky-Golay: Does it support high-order derivatives?
   - TrendFilter: What's the maximum k value vs derivative order?
   - Fourier-FFT-Adaptive: Are there Nyquist frequency limits?

4. **Examine test function characteristics**
   - What's the Lotka-Volterra trajectory like at orders 5-7?
   - Are high-order derivatives numerically stable in ground truth?
   - Could the true derivatives be near-zero for some portions?

### Specific Checks to Run

```bash
# 1. Extract actual predictions for a failing case
# Look at comprehensive_results.csv or original JSON outputs

# 2. Check method source code for these patterns:
grep -r "catch.*return.*zeros" methods/
grep -r "catch.*return.*fill" methods/
grep -r "fallback\|default" methods/

# 3. Verify Savitzky-Golay order support
# Check if it's even supposed to work for order > window_length/2
```

### Questions to Answer

1. **Are predictions literally zeros?** Or small constants? Or random noise?

2. **Is this documented behavior?** Do methods explicitly fail at high orders?

3. **Should these be errors instead?** Should methods raise exceptions rather than returning bad values?

4. **How should we handle this in results?**
   - Filter out NRMSE > threshold from tables/plots?
   - Mark as "Method not applicable" in paper?
   - Add warning flags to data?

## Preliminary Recommendations (Before Full Investigation)

### For the Paper

1. **Add footnotes/markers** for methods with NRMSE > 0.95
   - Note: "Method failed silently at this order"
   - Distinguish from methods that properly raise errors

2. **Filter comparison plots** to show only NRMSE < 0.95 or < 1.0
   - Current 0-1.0 scale is good, but may need notes

3. **Document method limitations**
   - "Savitzky-Golay: Not applicable for order > X"
   - "TrendFilter-kN: Limited to order ≤ N"

### For the Code

1. **Add validation** to detect near-constant predictions
   - Compute std(predictions) and warn if too low
   - Flag NRMSE > 0.9 as suspicious

2. **Improve error handling**
   - Methods should raise exceptions for unsupported orders
   - Don't silently return garbage values

3. **Add method capability metadata**
   - max_supported_order field
   - Skip methods that can't handle requested order

## Summary Statistics

```
Total benchmark records: 1,491
Records with NRMSE > 0.95: 574 (38.5%)
Records with NRMSE = 0.994937: 156 (10.5%)

Most affected methods:
- Fourier-FFT-Adaptive: 53/56 records suspicious (95%)
- TrendFilter-k7: 42/56 records (75%)
- TrendFilter-k2: 42/56 records (75%)
- Savitzky-Golay: 49/56 records (88%)

Least affected methods:
- GP-Julia-AD: 1/56 records (2%)
- GP_RBF_Iso_Python: 1/56 records (2%)
- gp_rbf_mean: 1/56 records (2%)
```

## Next Steps

**User Decision Required:**

Should we:
1. **Investigate further** to determine root cause?
2. **Filter these from results** immediately and re-generate tables/plots?
3. **Add error handling** to make methods fail explicitly?
4. **Document as-is** with warnings in the paper?

Please advise on priority and approach.
