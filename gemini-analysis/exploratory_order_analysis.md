# Exploratory Analysis by Derivative Order

This file contains alternative summary tables based on different maximum derivative orders for method inclusion.

## Table: Contenders with Full Coverage up to Order 3

Methods included here have complete data for all noise levels and ODE systems for derivative orders 1 through 3. Averages and ranks are computed over this range (excluding order 0 function approximation).

| Method                        |   Avg. Rank |   Low Noise Rank |   Low Noise Median |   High Noise Rank | High Noise Median   |
|:------------------------------|------------:|-----------------:|-------------------:|------------------:|:--------------------|
| GP-TaylorAD-Julia             |         1.9 |              1.4 |              0.003 |               2.5 | 0.131               |
| GP-RBF-Python                 |         2.5 |              2.2 |              0.004 |               2.8 | 0.131               |
| PyNumDiff-SavitzkyGolay-Tuned |         7.7 |              6.6 |              0.022 |               8.7 | 0.242               |
| SavitzkyGolay-Fixed           |         9.6 |             11   |              0.088 |               8.2 | 0.238               |
| Spline-Dierckx-5              |        10.3 |              6.9 |              0.029 |              13.6 | 0.432               |
| ButterworthSpline_Python      |        11.1 |             15.4 |              0.141 |               6.9 | 0.213               |
| Fourier-Interp                |        11.6 |             11.9 |              0.1   |              11.4 | 0.357               |
| SavitzkyGolay-Julia-Fixed     |        11.8 |              8.7 |              0.049 |              14.8 | 0.356               |
| SavitzkyGolay-Julia-Hybrid    |        12.1 |              8   |              0.047 |              16.1 | 0.367               |
| SavitzkyGolay-Adaptive        |        13.1 |              8.2 |              0.042 |              18   | 0.395               |
| Fourier-Continuation          |        13.1 |             17   |              0.316 |               9.3 | 0.318               |
| SavitzkyGolay-Julia-Adaptive  |        13.6 |              6.4 |              0.015 |              20.9 | 0.684               |
| PyNumDiff-SavitzkyGolay-Auto  |        13.7 |              9.7 |              0.076 |              17.7 | 0.432               |
| Fourier-GCV                   |        14.4 |             17.7 |              0.467 |              11.1 | 0.467               |
| RKHS_Spline_m2_Python         |        14.7 |             14.9 |              0.153 |              14.4 | 0.252               |
| Fourier                       |        15.6 |             18.8 |              0.493 |              12.3 | 0.493               |
| PyNumDiff-Spectral-Auto       |        15.6 |             13.6 |              0.049 |              17.7 | 0.514               |
| Fourier-Cont-Adaptive         |        17.1 |             20.9 |              0.381 |              13.3 | 0.390               |
| FFT-Adaptive-Julia            |        18.3 |             19.3 |              0.354 |              17.3 | 0.575               |
| Whittaker_m2_Python           |        18.8 |             22.4 |              0.497 |              15.1 | 0.500               |
| AAA-Adaptive-Wavelet          |        19.7 |             14.4 |              0.035 |              24.9 | 2.928               |
| FFT-Adaptive-Py               |        19.7 |             21.2 |              0.445 |              18.1 | 0.575               |
| Butterworth_Python            |        19.9 |             23.4 |              0.55  |              16.3 | 0.550               |
| PyNumDiff-Spectral-Tuned      |        21.6 |             24.9 |              0.58  |              18.3 | 0.580               |
| TVRegDiff-Python              |        22.1 |             22.8 |              0.319 |              21.3 | 0.589               |
| AAA-Adaptive-Diff2            |        22.4 |             19.9 |              0.476 |              24.8 | 2.896               |
| AAA-LowTol                    |        23.7 |             16.9 |              0.142 |              30.5 | >10                 |
| Spline-GSS                    |        24.4 |             27.2 |              0.917 |              21.6 | 0.917               |
| Chebyshev-AICc                |        26.1 |             28.2 |              0.867 |              24   | 0.859               |
| Kalman-Gradient               |        26.4 |             28.5 |              0.988 |              24.3 | 0.988               |
| SVR_Python                    |        27.2 |             29.1 |              0.987 |              25.2 | 0.987               |
| Chebyshev                     |        28.4 |             30.3 |              1.433 |              26.4 | 1.436               |

## Table: Contenders with Full Coverage up to Order 5

Methods included here have complete data for all noise levels and ODE systems for derivative orders 1 through 5. Averages and ranks are computed over this range (excluding order 0 function approximation).

| Method                        |   Avg. Rank |   Low Noise Rank |   Low Noise Median |   High Noise Rank | High Noise Median   |
|:------------------------------|------------:|-----------------:|-------------------:|------------------:|:--------------------|
| GP-TaylorAD-Julia             |         2   |              1.6 |              0.014 |               2.5 | 0.331               |
| GP-RBF-Python                 |         2.6 |              2.4 |              0.02  |               2.9 | 0.331               |
| SavitzkyGolay-Fixed           |         9.6 |              9.8 |              0.11  |               9.3 | 0.591               |
| PyNumDiff-SavitzkyGolay-Tuned |        10.1 |              7.5 |              0.105 |              12.6 | 0.848               |
| Spline-Dierckx-5              |        10.3 |              8.3 |              0.1   |              12.2 | 0.673               |
| ButterworthSpline_Python      |        10.7 |             13.9 |              0.368 |               7.5 | 0.437               |
| SavitzkyGolay-Julia-Fixed     |        12.6 |              9.1 |              0.083 |              16.1 | 0.660               |
| Fourier-Continuation          |        12.9 |             16.7 |              0.502 |               9.1 | 0.554               |
| SavitzkyGolay-Adaptive        |        13   |              8.1 |              0.08  |              18   | 0.706               |
| SavitzkyGolay-Julia-Hybrid    |        13.3 |              8.6 |              0.092 |              18.1 | 0.972               |
| Fourier-GCV                   |        14.1 |             17.4 |              0.519 |              10.8 | 0.609               |
| Fourier-Interp                |        14.6 |             14.3 |              0.321 |              14.9 | 0.848               |
| Fourier                       |        14.9 |             18.1 |              0.519 |              11.7 | 0.660               |
| RKHS_Spline_m2_Python         |        15.2 |             14.8 |              0.335 |              15.7 | 0.772               |
| SavitzkyGolay-Julia-Adaptive  |        15.4 |              7.7 |              0.048 |              23.2 | 3.789               |
| PyNumDiff-SavitzkyGolay-Auto  |        16.5 |             12   |              0.185 |              21   | 1.917               |
| Fourier-Cont-Adaptive         |        16.5 |             20.4 |              0.734 |              12.6 | 0.749               |
| FFT-Adaptive-Julia            |        16.9 |             18.6 |              0.643 |              15.3 | 0.828               |
| Whittaker_m2_Python           |        17.4 |             20.7 |              0.744 |              14   | 0.751               |
| PyNumDiff-Spectral-Auto       |        17.5 |             14.6 |              0.284 |              20.4 | 2.022               |
| Butterworth_Python            |        18   |             22   |              0.805 |              13.9 | 0.806               |
| FFT-Adaptive-Py               |        18.9 |             21.4 |              0.697 |              16.3 | 0.837               |
| PyNumDiff-Spectral-Tuned      |        19.9 |             23.5 |              0.896 |              16.3 | 0.895               |
| AAA-Adaptive-Wavelet          |        20.3 |             16.6 |              0.048 |              24.1 | >10                 |
| Spline-GSS                    |        22   |             25.3 |              0.991 |              18.7 | 0.991               |
| AAA-Adaptive-Diff2            |        22.4 |             20.8 |              0.499 |              24   | >10                 |
| AAA-LowTol                    |        23.6 |             16.1 |              0.269 |              31.1 | >10                 |
| Kalman-Gradient               |        23.7 |             26.5 |              0.998 |              21   | 0.998               |
| SVR_Python                    |        24.4 |             27.1 |              0.998 |              21.7 | 0.998               |
| TVRegDiff-Python              |        24.8 |             25.6 |              1.196 |              24.1 | 3.760               |
| Chebyshev                     |        26.9 |             29.4 |              1.633 |              24.3 | 1.633               |
| Chebyshev-AICc                |        26.9 |             29.1 |              3.012 |              24.7 | 3.036               |

## Table: Contenders with Full Coverage up to Order 7

Methods included here have complete data for all noise levels and ODE systems for derivative orders 1 through 7. Averages and ranks are computed over this range (excluding order 0 function approximation).

| Method                        |   Avg. Rank |   Low Noise Rank |   Low Noise Median |   High Noise Rank | High Noise Median   |
|:------------------------------|------------:|-----------------:|-------------------:|------------------:|:--------------------|
| GP-TaylorAD-Julia             |         1.9 |              1.7 |              0.04  |               2.1 | 0.535               |
| GP-RBF-Python                 |         2.6 |              2.6 |              0.059 |               2.6 | 0.535               |
| SavitzkyGolay-Fixed           |         8.2 |              9.1 |              0.282 |               7.3 | 0.820               |
| PyNumDiff-SavitzkyGolay-Tuned |         8.5 |              7.4 |              0.3   |               9.6 | 0.986               |
| SavitzkyGolay-Adaptive        |        10.4 |              7.1 |              0.16  |              13.6 | 1.135               |
| Fourier-Continuation          |        10.4 |             13.6 |              0.741 |               7.2 | 0.777               |
| SavitzkyGolay-Julia-Fixed     |        10.7 |              8.2 |              0.277 |              13.2 | 1.027               |
| Fourier-GCV                   |        10.8 |             13.7 |              0.733 |               7.9 | 0.812               |
| Fourier                       |        11.5 |             14.3 |              0.783 |               8.6 | 0.839               |
| SavitzkyGolay-Julia-Hybrid    |        11.6 |              7.8 |              0.244 |              15.3 | 2.581               |
| FFT-Adaptive-Julia            |        12.4 |             14.5 |              0.834 |              10.2 | 0.962               |
| Fourier-Cont-Adaptive         |        12.7 |             16   |              0.878 |               9.4 | 0.881               |
| Fourier-Interp                |        13   |             13.4 |              0.626 |              12.7 | 1.202               |
| SavitzkyGolay-Julia-Adaptive  |        13   |              7.6 |              0.092 |              18.5 | 9.036               |
| PyNumDiff-SavitzkyGolay-Auto  |        13.9 |             11.2 |              0.382 |              16.6 | 5.829               |
| FFT-Adaptive-Py               |        14   |             16.7 |              0.919 |              11.3 | 0.972               |
| PyNumDiff-Spectral-Tuned      |        14.3 |             17.5 |              0.974 |              11   | 0.974               |
| PyNumDiff-Spectral-Auto       |        14.4 |             12.8 |              0.423 |              16.1 | 3.948               |
| Spline-GSS                    |        15.4 |             18.4 |              0.997 |              12.5 | 0.997               |
| AAA-Adaptive-Wavelet          |        15.9 |             13.5 |              0.113 |              18.2 | >10                 |
| AAA-Adaptive-Diff2            |        17.5 |             17   |              0.744 |              18   | >10                 |
| AAA-LowTol                    |        18.2 |             12.8 |              0.38  |              23.6 | >10                 |
| Chebyshev                     |        18.6 |             21   |              1.573 |              16.2 | 1.572               |
| Chebyshev-AICc                |        20.2 |             22.1 |              3.95  |              18.2 | 3.969               |

