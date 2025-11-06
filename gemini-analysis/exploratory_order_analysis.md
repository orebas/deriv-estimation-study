# Exploratory Analysis by Derivative Order

This file contains alternative summary tables based on different maximum derivative orders for method inclusion.

## Table: Contenders with Full Coverage up to Order 3

Methods included here have complete data for all noise levels and ODE systems for derivative orders 0 through 3. Averages and ranks are computed over this range.

| Method                   |   Avg. Rank (Overall) |   Avg. Rank (Low Noise) | Avg. nRMSE (Low Noise)   |   Avg. Rank (High Noise) | Avg. nRMSE (High Noise)   |
|:-------------------------|----------------------:|------------------------:|:-------------------------|-------------------------:|:--------------------------|
| GP-Julia-AD              |                   3.2 |                     2.6 | 0.015                    |                      3.8 | 0.154                     |
| GP-RBF-Python            |                   3.4 |                     3.9 | 0.015                    |                      3   | 0.154                     |
| GP-RBF-Iso-Python        |                   3.6 |                     3.9 | 0.015                    |                      3.2 | 0.154                     |
| GP-RBF-Mean-Py           |                   3.7 |                     3.7 | 0.015                    |                      3.7 | 0.154                     |
| PyNumDiff-SavGol-Tuned   |                   8.4 |                     9   | 0.056                    |                      7.8 | 0.393                     |
| Fourier-Interp           |                  11.3 |                    11   | 0.104                    |                     11.7 | 0.452                     |
| Dierckx-5                |                  12.2 |                     9.6 | 0.039                    |                     14.8 | 0.370                     |
| SG-Package-Fixed         |                  12.3 |                    11.5 | 0.081                    |                     13   | 0.858                     |
| Savitzky-Golay-Fixed     |                  12.3 |                    14.4 | 0.081                    |                     10.2 | 0.341                     |
| SG-Package-Hybrid        |                  12.3 |                    10.9 | 0.061                    |                     13.8 | 1.648                     |
| SG-Package-Adaptive      |                  12.9 |                     9.5 | 0.048                    |                     16.2 | 1.994                     |
| PyNumDiff-Spectral-Auto  |                  13.6 |                    11.5 | 0.214                    |                     15.6 | 1.767                     |
| Fourier-Continuation     |                  13.6 |                    17.1 | 0.276                    |                     10.1 | 0.295                     |
| Savitzky-Golay-Adaptive  |                  13.9 |                    11.9 | 0.051                    |                     16   | 1.177                     |
| Fourier-GCV              |                  15   |                    17.8 | 0.316                    |                     12.2 | 0.334                     |
| FFT-Adaptive-Julia       |                  15   |                    15.1 | 0.299                    |                     15   | 0.433                     |
| Fourier                  |                  15.5 |                    18.4 | 0.350                    |                     12.6 | 0.369                     |
| FFT-Adaptive-Py          |                  15.9 |                    16.3 | 0.429                    |                     15.6 | 0.454                     |
| Fourier-Cont-Adaptive    |                  16.6 |                    19.7 | 0.336                    |                     13.5 | 0.341                     |
| PyNumDiff-Spectral-Tuned |                  16.9 |                    18   | 0.466                    |                     15.8 | 0.469                     |
| TVRegDiff_Python         |                  17.5 |                    17   | 0.457                    |                     18   | 1.650                     |
| AAA-Adaptive-Wavelet     |                  18.6 |                    14.4 | >10                      |                     22.9 | >10                       |
| AAA-Adaptive-Diff2       |                  21.2 |                    19.5 | >10                      |                     22.9 | >10                       |
| AAA-LowPrec              |                  21.7 |                    17.9 | 5.469                    |                     25.6 | >10                       |
| GSS                      |                  22.6 |                    24.6 | 0.642                    |                     20.7 | 0.642                     |
| Chebyshev-AICc           |                  23   |                    24.6 | 1.401                    |                     21.4 | 1.428                     |
| KalmanGrad_Python        |                  24.5 |                    25.9 | 0.800                    |                     23.2 | 0.800                     |
| Chebyshev                |                  25.1 |                    26.5 | 1.175                    |                     23.8 | 1.176                     |

## Table: Contenders with Full Coverage up to Order 5

Methods included here have complete data for all noise levels and ODE systems for derivative orders 0 through 5. Averages and ranks are computed over this range.

| Method                   |   Avg. Rank (Overall) |   Avg. Rank (Low Noise) | Avg. nRMSE (Low Noise)   |   Avg. Rank (High Noise) | Avg. nRMSE (High Noise)   |
|:-------------------------|----------------------:|------------------------:|:-------------------------|-------------------------:|:--------------------------|
| GP-Julia-AD              |                   3.2 |                     2.6 | 0.055                    |                      3.8 | 0.314                     |
| GP-RBF-Python            |                   3.4 |                     3.9 | 0.057                    |                      3   | 0.314                     |
| GP-RBF-Iso-Python        |                   3.5 |                     3.9 | 0.057                    |                      3.1 | 0.314                     |
| GP-RBF-Mean-Py           |                   3.7 |                     3.6 | 0.057                    |                      3.7 | 0.314                     |
| PyNumDiff-SavGol-Tuned   |                  10   |                     9.4 | 0.180                    |                     10.7 | 1.607                     |
| Savitzky-Golay-Fixed     |                  11.5 |                    12.8 | 0.192                    |                     10.1 | 1.162                     |
| Dierckx-5                |                  11.7 |                    10.3 | 0.196                    |                     13.2 | 0.554                     |
| SG-Package-Fixed         |                  12.7 |                    11.1 | 0.256                    |                     14.3 | 6.529                     |
| Fourier-Continuation     |                  13   |                    16.5 | 0.435                    |                      9.6 | 0.473                     |
| SG-Package-Hybrid        |                  13.1 |                    10.6 | 0.500                    |                     15.5 | >10                       |
| Fourier-Interp           |                  13.4 |                    13   | 0.426                    |                     13.9 | 2.461                     |
| Savitzky-Golay-Adaptive  |                  13.5 |                    10.8 | 0.168                    |                     16.1 | 3.468                     |
| SG-Package-Adaptive      |                  14.2 |                     9.7 | 0.515                    |                     18.6 | >10                       |
| Fourier-GCV              |                  14.4 |                    17.2 | 0.464                    |                     11.6 | 0.498                     |
| FFT-Adaptive-Julia       |                  14.7 |                    15.5 | 0.489                    |                     13.8 | 0.603                     |
| Fourier                  |                  14.7 |                    17.7 | 0.520                    |                     11.8 | 0.559                     |
| PyNumDiff-Spectral-Auto  |                  15.2 |                    12.8 | 0.396                    |                     17.6 | 8.323                     |
| FFT-Adaptive-Py          |                  16   |                    17.4 | 0.742                    |                     14.6 | 0.618                     |
| Fourier-Cont-Adaptive    |                  16   |                    19.2 | 0.569                    |                     12.9 | 0.575                     |
| PyNumDiff-Spectral-Tuned |                  16.5 |                    18.3 | 0.626                    |                     14.7 | 0.628                     |
| AAA-Adaptive-Wavelet     |                  18.9 |                    15.5 | >10                      |                     22.2 | >10                       |
| TVRegDiff_Python         |                  20.1 |                    19.9 | 8.565                    |                     20.3 | >10                       |
| GSS                      |                  20.6 |                    23   | 0.742                    |                     18.2 | 0.742                     |
| AAA-Adaptive-Diff2       |                  20.9 |                    19.7 | >10                      |                     22.1 | >10                       |
| AAA-LowPrec              |                  21.6 |                    16.8 | >10                      |                     26.4 | >10                       |
| KalmanGrad_Python        |                  22.2 |                    24.1 | 0.866                    |                     20.4 | 0.866                     |
| Chebyshev-AICc           |                  23.5 |                    25.2 | 7.501                    |                     21.8 | 7.636                     |
| Chebyshev                |                  23.9 |                    25.7 | 1.689                    |                     22.1 | 1.689                     |

## Table: Contenders with Full Coverage up to Order 7

Methods included here have complete data for all noise levels and ODE systems for derivative orders 0 through 7. Averages and ranks are computed over this range.

| Method                   |   Avg. Rank (Overall) |   Avg. Rank (Low Noise) | Avg. nRMSE (Low Noise)   |   Avg. Rank (High Noise) | Avg. nRMSE (High Noise)   |
|:-------------------------|----------------------:|------------------------:|:-------------------------|-------------------------:|:--------------------------|
| GP-Julia-AD              |                   3.1 |                     2.5 | 0.122                    |                      3.8 | 0.455                     |
| GP-RBF-Python            |                   3.4 |                     3.9 | 0.128                    |                      2.9 | 0.455                     |
| GP-RBF-Iso-Python        |                   3.5 |                     4   | 0.128                    |                      3   | 0.455                     |
| GP-RBF-Mean-Py           |                   3.6 |                     3.6 | 0.128                    |                      3.6 | 0.455                     |
| PyNumDiff-SavGol-Tuned   |                   9.8 |                     9.2 | 0.333                    |                     10.5 | 2.484                     |
| Savitzky-Golay-Fixed     |                  10.6 |                    11.6 | 0.340                    |                      9.6 | 1.949                     |
| Fourier-Continuation     |                  12.1 |                    15.1 | 0.549                    |                      9.1 | 0.597                     |
| SG-Package-Fixed         |                  12.2 |                    10.3 | 3.838                    |                     14.1 | >10                       |
| Savitzky-Golay-Adaptive  |                  12.3 |                     9.6 | 0.288                    |                     15   | 3.905                     |
| SG-Package-Hybrid        |                  12.8 |                     9.8 | 4.000                    |                     15.9 | >10                       |
| Fourier-GCV              |                  12.9 |                    15.5 | 0.569                    |                     10.4 | 0.611                     |
| FFT-Adaptive-Julia       |                  13.3 |                    14.4 | 0.611                    |                     12.1 | 0.701                     |
| Fourier                  |                  13.3 |                    15.9 | 0.622                    |                     10.6 | 0.671                     |
| Fourier-Interp           |                  13.9 |                    13.5 | 1.162                    |                     14.3 | 8.863                     |
| SG-Package-Adaptive      |                  13.9 |                     9.6 | 4.185                    |                     18.3 | >10                       |
| Fourier-Cont-Adaptive    |                  14.5 |                    17.3 | 0.735                    |                     11.6 | 0.741                     |
| FFT-Adaptive-Py          |                  14.7 |                    16.4 | 0.900                    |                     13   | 0.712                     |
| PyNumDiff-Spectral-Tuned |                  14.9 |                    17   | 0.718                    |                     12.9 | 0.719                     |
| PyNumDiff-Spectral-Auto  |                  15.1 |                    13.1 | 0.993                    |                     17.1 | >10                       |
| AAA-Adaptive-Wavelet     |                  17.2 |                    14.4 | >10                      |                     19.9 | >10                       |
| GSS                      |                  17.7 |                    20.1 | 0.803                    |                     15.3 | 0.804                     |
| AAA-Adaptive-Diff2       |                  19.1 |                    18.4 | >10                      |                     19.8 | >10                       |
| AAA-LowPrec              |                  19.4 |                    14.7 | >10                      |                     24   | >10                       |
| Chebyshev                |                  20.4 |                    22.3 | 1.637                    |                     18.5 | 1.636                     |
| Chebyshev-AICc           |                  21.3 |                    22.9 | >10                      |                     19.6 | >10                       |

