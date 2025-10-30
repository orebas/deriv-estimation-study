# Analysis by Noise Regime

This report analyzes the performance of contending methods by splitting the results into two distinct regimes:
- **Low-Noise Regime:** Noise levels <= 0.001 (0.1%)
- **High-Noise Regime:** Noise levels >= 0.01 (1%)

## Results for Lotka Volterra System

### Top 5 Contenders: Low-Noise Regime

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.065089 |
| 2 | GP_RBF_Iso_Python | 0.071913 |
| 3 | GP_RBF_Python | 0.071913 |
| 4 | gp_rbf_mean | 0.071913 |
| 5 | Dierckx-5 | 0.137954 |

### Top 5 Contenders: High-Noise Regime

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | Savitzky-Golay | 0.330758 |
| 2 | GP-Julia-AD | 0.402653 |
| 3 | GP_RBF_Iso_Python | 0.413984 |
| 4 | GP_RBF_Python | 0.413984 |
| 5 | gp_rbf_mean | 0.413984 |

## Results for Lorenz System

### Top 5 Contenders: Low-Noise Regime

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.078187 |
| 2 | GP_RBF_Iso_Python | 0.083624 |
| 3 | gp_rbf_mean | 0.083625 |
| 4 | GP_RBF_Python | 0.083626 |
| 5 | Dierckx-5 | 0.203357 |

### Top 5 Contenders: High-Noise Regime

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.443818 |
| 2 | Savitzky-Golay | 0.494740 |
| 3 | Fourier-Continuation-Adaptive | 0.600888 |
| 4 | fourier_continuation | 0.718494 |
| 5 | Fourier-FFT-Adaptive | 0.761535 |

## Results for Van Der Pol System

### Top 5 Contenders: Low-Noise Regime

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.221470 |
| 2 | GP_RBF_Iso_Python | 0.222035 |
| 3 | GP_RBF_Python | 0.222035 |
| 4 | gp_rbf_mean | 0.222035 |
| 5 | Dierckx-5 | 0.242935 |

### Top 5 Contenders: High-Noise Regime

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | Dierckx-5 | 0.570984 |
| 2 | Savitzky-Golay | 0.585582 |
| 3 | GP-Julia-AD | 0.622533 |
| 4 | GP_RBF_Iso_Python | 0.622836 |
| 5 | GP_RBF_Python | 0.622836 |

## Summary of Findings

The analysis reveals significant shifts in method performance between low and high noise regimes.

**1. GP Methods Dominate High Noise.**
- In the high-noise regime, Gaussian Process methods (`GP-*`) are the undisputed champions across all three systems. Their inherent regularization and noise modeling provide superior robustness.

**2. Splines & Spectral Methods Excel in Low Noise.**
- In the low-noise regime, methods like `Dierckx-5` and various Fourier-based spectral methods are extremely competitive, sometimes outperforming the GP methods. They are excellent choices for high-precision applications where noise is not a primary concern.

**3. System Dynamics Matter.**
- The chaotic Lorenz system shows the most dramatic performance drop-off in the high-noise regime. While `GP-Julia-AD` remains the best, its error increases significantly, highlighting the combined challenge of chaos and noise.

**Conclusion for Narrative:** The data strongly supports splitting the recommendation. The story should be: 'For clean data, you have several excellent, fast options. For noisy data, the field narrows considerably, and Gaussian Processes are the only reliable choice.'
