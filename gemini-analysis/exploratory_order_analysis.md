# Exploratory Analysis by Derivative Order

This file contains alternative summary tables based on different maximum derivative orders for method inclusion.

## Table 1: Contenders with Full Coverage up to Order 5

Methods included here have complete data for all noise levels and ODE systems for derivative orders 0 through 5. Averages and ranks are computed over this range.

| Method                |   Avg. Rank (Overall) |   Avg. Rank (Low Noise) | Avg. nRMSE (Low Noise)   |   Avg. Rank (High Noise) | Avg. nRMSE (High Noise)   |
|:----------------------|----------------------:|------------------------:|:-------------------------|-------------------------:|:--------------------------|
| GP-Julia-AD           |                   2.6 |                     2.6 | 0.021                    |                      2.6 | 0.314                     |
| GP-RBF-Iso-Python     |                   2.8 |                     3.1 | 0.025                    |                      2.4 | 0.314                     |
| GP-RBF-Mean-Py        |                   3.5 |                     3.4 | 0.025                    |                      3.6 | 0.314                     |
| GP-RBF-Python         |                   3.7 |                     3.9 | 0.025                    |                      3.4 | 0.314                     |
| Dierckx-5             |                   8.4 |                     6.7 | 0.151                    |                     10.1 | 0.554                     |
| Fourier-Continuation  |                   9.4 |                    11.7 | 0.435                    |                      7.1 | 0.473                     |
| Fourier-Interp        |                   9.6 |                     8.8 | 0.415                    |                     10.4 | 2.461                     |
| FFT-Adaptive-Julia    |                  10.5 |                    10.2 | 0.468                    |                     10.8 | 0.603                     |
| Fourier-GCV           |                  10.6 |                    12.2 | 0.464                    |                      8.9 | 0.498                     |
| Fourier               |                  10.8 |                    12.6 | 0.520                    |                      8.9 | 0.559                     |
| Savitzky-Golay        |                  10.8 |                    12.8 | 0.444                    |                      8.8 | 0.452                     |
| Fourier-Cont-Adaptive |                  11.9 |                    13.6 | 0.569                    |                     10.1 | 0.575                     |
| FFT-Adaptive-Py       |                  12.2 |                    12.8 | 0.813                    |                     11.6 | 0.618                     |
| AAA-Adaptive-Wavelet  |                  12.5 |                     8   | >10                      |                     16.9 | >10                       |
| AAA-Adaptive-Diff2    |                  14.1 |                    11.2 | >10                      |                     16.9 | >10                       |
| TVRegDiff_Python      |                  14.4 |                    14.1 | 6.666                    |                     14.6 | >10                       |
| AAA-LowPrec           |                  14.9 |                    10.3 | 0.233                    |                     19.6 | >10                       |
| GSS                   |                  15.7 |                    16.9 | 0.742                    |                     14.4 | 0.742                     |
| KalmanGrad_Python     |                  16.9 |                    17.8 | 0.866                    |                     16.1 | 0.866                     |
| Chebyshev-AICc        |                  17.6 |                    18.8 | 7.499                    |                     16.4 | 7.636                     |
| Chebyshev             |                  18.2 |                    19.2 | 1.689                    |                     17.2 | 1.689                     |

## Table 2: Contenders with Full Coverage up to Order 3

Methods included here have complete data for all noise levels and ODE systems for derivative orders 0 through 3. Averages and ranks are computed over this range.

| Method                |   Avg. Rank (Overall) |   Avg. Rank (Low Noise) |   Avg. nRMSE (Low Noise) |   Avg. Rank (High Noise) | Avg. nRMSE (High Noise)   |
|:----------------------|----------------------:|------------------------:|-------------------------:|-------------------------:|:--------------------------|
| GP-Julia-AD           |                   2.8 |                     2.8 |                    0.004 |                      2.7 | 0.154                     |
| GP-RBF-Iso-Python     |                   2.8 |                     3.2 |                    0.004 |                      2.4 | 0.154                     |
| GP-RBF-Mean-Py        |                   3.6 |                     3.7 |                    0.004 |                      3.5 | 0.154                     |
| GP-RBF-Python         |                   3.8 |                     4.2 |                    0.004 |                      3.5 | 0.154                     |
| Fourier-Interp        |                   7.6 |                     7   |                    0.102 |                      8.2 | 0.452                     |
| Dierckx-5             |                   8.6 |                     6.4 |                    0.016 |                     10.8 | 0.370                     |
| Fourier-Continuation  |                   9.6 |                    12   |                    0.276 |                      7.1 | 0.295                     |
| FFT-Adaptive-Julia    |                  10.4 |                     9.8 |                    0.277 |                     11   | 0.433                     |
| Fourier-GCV           |                  10.8 |                    12.7 |                    0.316 |                      8.9 | 0.334                     |
| Fourier               |                  11.1 |                    13.1 |                    0.35  |                      9.1 | 0.369                     |
| Savitzky-Golay        |                  11.1 |                    13   |                    0.256 |                      9.2 | 0.265                     |
| FFT-Adaptive-Py       |                  11.5 |                    11.4 |                    0.446 |                     11.7 | 0.454                     |
| Fourier-Cont-Adaptive |                  12.1 |                    14   |                    0.336 |                     10.1 | 0.341                     |
| AAA-Adaptive-Wavelet  |                  12.2 |                     7.1 |                    0.055 |                     17.2 | >10                       |
| TVRegDiff_Python      |                  12.2 |                    11.6 |                    0.397 |                     12.8 | 1.650                     |
| AAA-Adaptive-Diff2    |                  14.6 |                    12   |                    0.471 |                     17.2 | >10                       |
| AAA-LowPrec           |                  15.2 |                    11.6 |                    0.142 |                     18.9 | >10                       |
| Chebyshev-AICc        |                  16.9 |                    18   |                    1.401 |                     15.7 | 1.428                     |
| GSS                   |                  16.9 |                    18.1 |                    0.642 |                     15.6 | 0.642                     |
| KalmanGrad_Python     |                  18.4 |                    19.2 |                    0.8   |                     17.7 | 0.800                     |
| Chebyshev             |                  18.8 |                    19.9 |                    1.175 |                     17.8 | 1.176                     |
