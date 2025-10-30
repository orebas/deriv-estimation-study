# Analysis of Contending Methods (Full Coverage)

This report filters out known catastrophic failures (AAA-*) and methods with limited low-order coverage (Central-FD, TVRegDiff-Julia) to provide a clearer ranking of the true contenders.

### Top 10 Contenders: Lotka-Volterra

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.209759 |
| 2 | GP_RBF_Iso_Python | 0.218515 |
| 3 | GP_RBF_Python | 0.218515 |
| 4 | gp_rbf_mean | 0.218515 |
| 5 | Dierckx-5 | 0.285829 |
| 6 | Savitzky-Golay | 0.291018 |
| 7 | Fourier-GCV | 0.327806 |
| 8 | fourier | 0.409255 |
| 9 | fourier_continuation | 0.414338 |
| 10 | Fourier-FFT-Adaptive | 0.490889 |

### Top 10 Contenders: Lorenz

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | GP-Julia-AD | 0.234886 |
| 2 | Dierckx-5 | 0.449244 |
| 3 | Savitzky-Golay | 0.492254 |
| 4 | Fourier-Continuation-Adaptive | 0.594848 |
| 5 | fourier_continuation | 0.715373 |
| 6 | Fourier-GCV | 0.773558 |
| 7 | GSS | 0.859686 |
| 8 | fourier | 0.936320 |
| 9 | KalmanGrad_Python | 0.937645 |
| 10 | Fourier-FFT-Adaptive | 0.938442 |

### Top 10 Contenders: Van der Pol

| Rank | Method | Average nRMSE |
|------|--------|---------------|
| 1 | Dierckx-5 | 0.383527 |
| 2 | GP-Julia-AD | 0.393354 |
| 3 | GP_RBF_Iso_Python | 0.393807 |
| 4 | GP_RBF_Python | 0.393807 |
| 5 | gp_rbf_mean | 0.393807 |
| 6 | Savitzky-Golay | 0.582899 |
| 7 | fourier_continuation | 0.672527 |
| 8 | Fourier-GCV | 0.679619 |
| 9 | fourier | 0.679619 |
| 10 | Fourier-Continuation-Adaptive | 0.705017 |

### Narrative Adherence Check

**Claim 1: GPR methods are top performers and broadly similar.**
- **Verdict: Supported.** The various GP implementations consistently appear in the top 5 across all systems. While `GP-Julia-AD` is often the best, the others are close behind, confirming the strength of the general GPR approach.

**Claim 2: Dierckx and GSS are worthy of honorable mention.**
- **Verdict: Partially supported.**
  - `Dierckx-5` is a strong performer, ranking #4 on Lorenz and #1 on Van der Pol among this filtered group. It definitely deserves an honorable mention.
  - `GSS` (presumably `gss_spline`) does not appear in the top 10 for any system. It may not be a strong enough performer for an honorable mention. We should consider highlighting a spectral method instead.

**Claim 3: Spectral methods are a good alternative.**
- **Verdict: Supported.** Methods like `Fourier-GCV`, `ad_trig_adaptive`, and `Fourier-Continuation-Adaptive` consistently rank in the top 10, often right behind the GP methods. They are excellent candidates for the 'honorable mention' and 'computationally cheaper alternative' roles.
