| Method                |   Avg. Rank (Low Noise) | Avg. nRMSE (Low Noise)   |   Avg. Rank (High Noise) | Avg. nRMSE (High Noise)   |
|:----------------------|------------------------:|:-------------------------|-------------------------:|:--------------------------|
| GP-RBF-Iso-Python     |                     3   | 0.070                    |                      2.3 | 0.455                     |
| GP-Julia-AD           |                     2.6 | 0.058                    |                      2.5 | 0.455                     |
| GP-RBF-Python         |                     3.9 | 0.070                    |                      3.3 | 0.455                     |
| GP-RBF-Mean-Py        |                     3.4 | 0.070                    |                      3.5 | 0.455                     |
| Fourier-Continuation  |                    10   | 0.549                    |                      6.4 | 0.597                     |
| Fourier               |                    10.6 | 0.622                    |                      7.5 | 0.671                     |
| Fourier-GCV           |                    10.2 | 0.569                    |                      7.6 | 0.611                     |
| Fourier-Cont-Adaptive |                    11.5 | 0.735                    |                      8.8 | 0.741                     |
| FFT-Adaptive-Julia    |                     8.8 | 0.594                    |                      8.9 | 0.701                     |
| FFT-Adaptive-Py       |                    11.2 | 0.990                    |                      9.7 | 0.712                     |
| Fourier-Interp        |                     8.8 | 1.114                    |                     10.2 | 8.863                     |
| GSS                   |                    13.4 | 0.803                    |                     11.2 | 0.804                     |
| Chebyshev             |                    15.1 | 1.637                    |                     13.5 | 1.636                     |
| AAA-Adaptive-Diff2    |                     9.7 | >10                      |                     13.7 | >10                       |
| Chebyshev-AICc        |                    15.6 | >10                      |                     13.8 | >10                       |
| AAA-Adaptive-Wavelet  |                     7   | >10                      |                     13.9 | >10                       |
| AAA-LowPrec           |                     8.2 | 0.312                    |                     16.2 | >10                       |