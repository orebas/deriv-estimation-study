# Appendix C: Supporting Data Tables

This appendix contains the detailed data tables that support the conclusions presented in the main body of the paper.

## C.1. Catastrophic Failure of AAA Methods

The following table presents the average nRMSE for all six tested AAA variants across all three ODE systems, demonstrating their universal failure for this task.

| ODE System     | Method                     | Average nRMSE |
|----------------|----------------------------|---------------|
| Lorenz         | AAA-Adaptive-Diff2         | 5.02e+09 |
| Lorenz         | AAA-Adaptive-Wavelet       | 5.02e+09 |
| Lorenz         | AAA-HighPrec               | 1.13e+27 |
| Lorenz         | AAA-JAX-Adaptive-Diff2     | 1.87e+18 |
| Lorenz         | AAA-JAX-Adaptive-Wavelet   | 1.87e+18 |
| Lorenz         | AAA-LowPrec                | 4.31e+22 |
| Lotka Volterra | AAA-Adaptive-Diff2         | 5.26e+10 |
| Lotka Volterra | AAA-Adaptive-Wavelet       | 5.57e+10 |
| Lotka Volterra | AAA-HighPrec               | 1.75e+31 |
| Lotka Volterra | AAA-JAX-Adaptive-Diff2     | 5.96e+23 |
| Lotka Volterra | AAA-JAX-Adaptive-Wavelet   | 5.96e+23 |
| Lotka Volterra | AAA-LowPrec                | 1.30e+40 |
| Van der Pol    | AAA-Adaptive-Diff2         | 2.46e+15 |
| Van der Pol    | AAA-Adaptive-Wavelet       | 9.46e+16 |
| Van der Pol    | AAA-HighPrec               | 2.82e+30 |
| Van der Pol    | AAA-JAX-Adaptive-Diff2     | 7.47e+25 |
| Van der Pol    | AAA-JAX-Adaptive-Wavelet   | 7.47e+25 |
| Van der Pol    | AAA-LowPrec                | 2.82e+30 |

## C.2. Detailed Rankings by Noise Regime

The following tables provide the detailed, per-system rankings for the top contending methods in both the low- and high-noise regimes.

### Low-Noise Regime (≤ 0.1% Noise)

**Lotka Volterra:**
| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.065089 |
| 2 | GP_RBF_Iso_Python | 0.071913 |
| 3 | GP_RBF_Python | 0.071913 |
| 4 | gp_rbf_mean | 0.071913 |
| 5 | Dierckx-5 | 0.137954 |

**Lorenz:**
| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.078187 |
| 2 | GP_RBF_Iso_Python | 0.083624 |
| 3 | gp_rbf_mean | 0.083625 |
| 4 | GP_RBF_Python | 0.083626 |
| 5 | Dierckx-5 | 0.203357 |

**Van der Pol:**
| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.221470 |
| 2 | GP_RBF_Iso_Python | 0.222035 |
| 3 | GP_RBF_Python | 0.222035 |
| 4 | gp_rbf_mean | 0.222035 |
| 5 | Dierckx-5 | 0.242935 |

### High-Noise Regime (≥ 1% Noise)

**Lotka Volterra:**
| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | Savitzky-Golay | 0.330758 |
| 2 | GP-Julia-AD | 0.402653 |
| 3 | GP_RBF_Iso_Python | 0.413984 |
| 4 | GP_RBF_Python | 0.413984 |
| 5 | gp_rbf_mean | 0.413984 |

**Lorenz:**
| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.443818 |
| 2 | Savitzky-Golay | 0.494740 |
| 3 | Fourier-Continuation-Adaptive | 0.600888 |
| 4 | fourier_continuation | 0.718494 |
| 5 | Fourier-FFT-Adaptive | 0.761535 |

**Van der Pol:**
| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | Dierckx-5 | 0.570984 |
| 2 | Savitzky-Golay | 0.585582 |
| 3 | GP-Julia-AD | 0.622533 |
| 4 | GP_RBF_Iso_Python | 0.622836 |
| 5 | GP_RBF_Python | 0.622836 |
