# Filtered Plots - Clean Version

## ✅ What Changed

**Excluded Methods (4 total):**
1. **AAA-HighPrec** - Catastrophic failure case (documented separately in paper)
2. **AAA-LowPrec** - Less interesting variant
3. **SavitzkyGolay_Python** - Cross-language implementation issues
4. **GP-Julia-SE** - Failed experiment / bad implementation

**Result:**
- Down from 27 → **23 methods** shown
- Full-coverage: 16 → **13 methods**
- Much cleaner visualizations!

## 📊 Generated Plots (7 key ones)

All in `paper_figures/supplementary_v3/`:

1. **plot01_boxplot_all_methods.png** - Box plots for 23 methods
2. **plot02_line_order_all_methods.png** - Order degradation (13 full-coverage)
3. **plot03_heatmap_method_order.png** - Method × Order heatmap
4. **plot05_ranking_bar_chart.png** - Sorted ranking bar chart
5. **plot07_failure_rate.png** - Failure rate analysis
6. **plot10_best_method_grid.png** - Best method per condition
7. **plot15_accuracy_robustness.png** - Accuracy vs Robustness scatter

## 🎯 Visual Improvements

**Before (v2):**
- AAA-HighPrec dominated failure plots with extreme outliers
- GP-Julia-SE cluttered GP comparisons
- SavitzkyGolay_Python was misleading (implementation issues)
- 27 methods = crowded labels

**After (v3):**
- Focus on viable, well-implemented methods
- Cleaner labels and legends
- Less visual noise from known failures
- GP methods show true Julia implementation performance

## 📁 Comparison

```
v2/ (original 20 plots, all methods)
v3/ (filtered 7 plots, 23 methods)
```

## Next Steps

**Option 1: Use these 7 plots**
- They're the most important ones
- Already generated and ready

**Option 2: Generate all 20 with filtering**
- I can extend the script to generate all 20 plot types
- Same filtering applied throughout
- Takes ~2 minutes to run

**Option 3: Mix and match**
- Use v3 (filtered) for most plots
- Keep specific v2 plots where all methods matter

## Which Plots Should Get Full Treatment?

My recommendation for **all 20 plots with filtering**:
- Plots 1-3, 5, 7, 10, 15 ✅ (already done)
- Worth adding with filtering:
  - Plot 4: Small multiples (noise sensitivity)
  - Plot 6: Mean vs Std scatter
  - Plot 8/9: Category analysis
  - Plot 11: Stacked performance bars
  - Plot 16: High-order performance only
  - Plot 19: Ridge plot (beautiful!)
  - Plot 20: Coverage map

What would you like me to do?
1. Generate all 20 plots with filtering?
2. Just add a few more specific plots?
3. Stick with these 7 filtered plots?
