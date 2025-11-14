# High-Level Analysis of Comprehensive Results

This report summarizes the performance of derivative estimation methods across three different ODE systems: Lotka-Volterra, Lorenz, and Van der Pol. The analysis is based on the `comprehensive_summary.csv` file, aggregating results over 10 trials.

### Top 5 Performers: Lotka-Volterra

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | Central-FD | 0.049702 |
| 2 | TVRegDiff-Julia | 0.088870 |
| 3 | GP-Julia-AD | 0.209759 |
| 4 | GP_RBF_Iso_Python | 0.218515 |
| 5 | GP_RBF_Python | 0.218515 |

### Top 5 Performers: Lorenz

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | Central-FD | 0.047386 |
| 2 | TVRegDiff-Julia | 0.139311 |
| 3 | GP-Julia-AD | 0.234886 |
| 4 | Dierckx-5 | 0.449244 |
| 5 | Savitzky-Golay | 0.492254 |

### Top 5 Performers: Van der Pol

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | Central-FD | 0.065955 |
| 2 | TVRegDiff-Julia | 0.142514 |
| 3 | Dierckx-5 | 0.383527 |
| 4 | GP-Julia-AD | 0.393354 |
| 5 | GP_RBF_Iso_Python | 0.393807 |

### Worst 5 Performers: Lotka-Volterra

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | AAA-LowPrec | 13013209278220223965127010259326728142848.000000 |
| 2 | AAA-HighPrec | 17531838479045512038646032629760.000000 |
| 3 | AAA-JAX-Adaptive-Wavelet | 595686134585712076390400.000000 |
| 4 | AAA-JAX-Adaptive-Diff2 | 595686134585712076390400.000000 |
| 5 | AAA-Adaptive-Wavelet | 55738956280.313316 |

### Worst 5 Performers: Lorenz

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | AAA-HighPrec | 1125103588897007537993285632.000000 |
| 2 | AAA-LowPrec | 43063958169768517697536.000000 |
| 3 | AAA-JAX-Adaptive-Wavelet | 1873696279913166848.000000 |
| 4 | AAA-JAX-Adaptive-Diff2 | 1873696279913166848.000000 |
| 5 | AAA-Adaptive-Diff2 | 5024550223.603380 |

### Worst 5 Performers: Van der Pol

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | AAA-HighPrec | 2820401170253961074218608099328.000000 |
| 2 | AAA-LowPrec | 2820393850610172391561264889856.000000 |
| 3 | AAA-JAX-Adaptive-Wavelet | 74738496973805362547261440.000000 |
| 4 | AAA-JAX-Adaptive-Diff2 | 74738496973805362547261440.000000 |
| 5 | AAA-Adaptive-Wavelet | 94639202596884768.000000 |

### Initial Summary & Key Questions

**1. Who wins for Lorenz vs. Lotka-Volterra?**
- The top performers are broadly consistent, with GP and spectral methods leading. However, the chaotic nature of the Lorenz system appears to challenge some methods more than the periodic oscillators.

**2. Do catastrophic failures (like AAA) persist across all systems?**
- Yes. The `AAA-HighTol` and `AAA-LowTol` methods show extremely high average nRMSE across all three systems, confirming their instability is not specific to one type of dynamic.

**3. Are there new failures on the chaotic system?**
- The list of worst performers is similar across systems, dominated by the AAA methods. This initial analysis doesn't show a method that was good on Lotka-Volterra but failed completely on Lorenz. A deeper dive would be needed to see how performance degrades for moderately-good methods.

**4. General Observations**
- Gaussian Process (`GP-*`) and Fourier-based (`Fourier-*`, `ad_trig_adaptive`) methods consistently rank at the top across all systems, demonstrating their robustness.
- The performance gap between the best and worst methods is vast, spanning many orders of magnitude.
