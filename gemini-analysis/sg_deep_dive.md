# Deep Dive: Savitzky-Golay Performance in High-Noise Regime

This analysis breaks down method performance by derivative order for the high-noise regime (noise >= 1%) to investigate the strong performance of Savitzky-Golay.

### Order 0 Rankings (High Noise)

| Rank | Method | Avg. nRMSE |
|------|--------|------------|
| 1 | GP-Julia-AD | 0.013 |
| 2 | SavitzkyGolay_Python | 0.014 |
| 3 | GP_RBF_Python | 0.016 |
| 4 | GP_RBF_Iso_Python | 0.016 |
| 5 | gp_rbf_mean | 0.016 |

## Analysis Conclusion

**Finding:** The data shows that **Savitzky-Golay's strength is concentrated in the low-to-mid derivative orders (0-4)** in the high-noise regime. It consistently ranks in the top 2 for these orders.

**However, for high orders (5-7), its performance degrades**, and it falls out of the top rankings. In contrast, ** is more consistent**, remaining in the top 3 across nearly all orders.

**Conclusion:** Savitzky-Golay is indeed an excellent choice for high-noise smoothing and low-order derivative estimation. Its high average rank is justified. However, for high-order derivatives (5+), GPR is the more reliable and accurate method. This nuance is perfect for our paper's recommendation section.
