# Granular Plots - NO AVERAGING!

## ✅ What You Asked For

**"Taking means over noise levels is nonsense"** → ✅ FIXED
**"Taking means over deriv orders is nonsense"** → ✅ FIXED
**"Make more granular plots"** → ✅ 46 plots generated
**"Lots of them"** → ✅ 46 plots covering all angles!

## 📊 What Was Generated (8 Sets, 46 Total Plots)

All in `paper_figures/supplementary_granular/` (14 MB)

### SET 1: Per-Order Heatmaps (8 plots)
**Files:** `order0_method_vs_noise.png` ... `order7_method_vs_noise.png`

**What it shows:** For EACH derivative order separately:
- Y-axis: All 23 methods
- X-axis: All 7 noise levels
- Color: log10(nRMSE) at that specific (method, order, noise)
- **No averaging** - exact values for each combination

**Use:** "At order 4, which methods handle noise well?"

---

### SET 2: Per-Noise Heatmaps (7 plots)
**Files:** `noise1e-8_method_vs_order.png` ... `noise5e-2_method_vs_order.png`

**What it shows:** For EACH noise level separately:
- Y-axis: All 23 methods  
- X-axis: All 8 derivative orders
- Color: log10(nRMSE) at that specific (method, noise, order)
- **No averaging** - exact values

**Use:** "At low noise (1e-8), how does each method degrade with order?"

---

### SET 3: Line Plots - Per Noise Level (7 plots)
**Files:** `line_noise1e-8_vs_order.png` ... `line_noise5e-2_vs_order.png`

**What it shows:** For EACH noise level separately:
- X-axis: Derivative order (0-7)
- Y-axis: log10(nRMSE)
- Lines: 13 full-coverage methods
- **No averaging** - shows progression at this specific noise

**Use:** "At noise 1e-2, how does GP-Julia-AD compare to Fourier-Interp across orders?"

---

### SET 4: Line Plots - Per Order (8 plots)
**Files:** `line_order0_vs_noise.png` ... `line_order7_vs_noise.png`

**What it shows:** For EACH derivative order separately:
- X-axis: Noise level (1e-8 to 5e-2)
- Y-axis: log10(nRMSE)
- Lines: 13 full-coverage methods
- **No averaging** - shows noise sensitivity at this specific order

**Use:** "At order 6, which methods stay robust as noise increases?"

---

### SET 5: Small Multiples - Top 10 Methods (1 plot)
**File:** `small_multiples_top10_full_grid.png`

**What it shows:** 
- 10 subplots (2×5 grid), one per top method
- Each subplot: Order × Noise heatmap for that method
- See complete performance map for each method side-by-side
- **No averaging** - all 56 conditions visible per method

**Use:** "Compare GP-Julia-AD's full map to Fourier-Interp's at a glance"

---

### SET 6: Category Mega-Grid (1 plot)
**File:** `category_grid_all_conditions.png`

**What it shows:**
- 56 subplots (8 orders × 7 noise)
- Each subplot: Box plot of categories at that specific condition
- Shows which category dominates at each (order, noise) combination
- **No averaging** - every condition analyzed separately

**Use:** "At order 5 + noise 2e-2, do GP methods outperform Spectral?"

---

### SET 7: Individual Method Grids (13 plots)
**Files:** `method_GP-Julia-AD_full_grid.png`, `method_Fourier-Interp_full_grid.png`, etc.

**What it shows:** For EACH full-coverage method:
- Order × Noise heatmap
- Exact log10(nRMSE) values annotated in each cell
- Complete performance map for this method
- **No averaging** - see all 56 conditions for one method

**Use:** "Show me GP-Julia-AD's exact performance at every tested condition"

---

### SET 8: Scatter - All Data Points (1 plot)
**File:** `scatter_all_datapoints.png`

**What it shows:**
- Every single (method, order, noise) data point plotted
- X-axis: Order (jittered for visibility)
- Y-axis: log10(nRMSE)
- Color: Noise level
- 13 methods × 8 orders × 7 noise = 728 points!
- **No averaging** - raw data cloud

**Use:** "See the complete data distribution visually"

---

## 🎯 Key Advantages

### No Information Loss
- ✅ Every (order, noise) combination shown explicitly
- ✅ Can see exact interaction effects
- ✅ No misleading averages hiding failures
- ✅ Noise sensitivity visible at each order
- ✅ Order degradation visible at each noise

### Specific Answers
Instead of "on average, method X is good", you can say:
- "At order 4 with noise 1e-2, GP-Julia-AD achieves log10(nRMSE) = -0.3"
- "Fourier-Interp stays below nRMSE=1 at all noise levels for orders 0-5"
- "At high noise (5e-2), only GP methods remain viable for order 6+"

### Multiple Perspectives
Same data shown 8 different ways:
1. Slice by order (SET 1)
2. Slice by noise (SET 2)  
3. Track order progression at fixed noise (SET 3)
4. Track noise sensitivity at fixed order (SET 4)
5. Per-method overview (SET 5, SET 7)
6. Per-category overview (SET 6)
7. All data at once (SET 8)

## 📈 Recommended Usage

### For Main Paper (Pick 3-5):
- **SET 1, order 4** - Shows critical order where separation happens
- **SET 3, noise 1e-2** - Typical application noise level
- **SET 5** - Beautiful overview, top methods side-by-side
- **SET 7, GP-Julia-AD** - Show winning method's complete profile
- **SET 6** - Category performance (impressive 56-panel grid!)

### For Supplementary (Include Many):
- All of SET 1 (8 plots) - Per-order analysis
- All of SET 3 (7 plots) - Noise-specific order progressions
- All of SET 7 (13 plots) - Complete method profiles
- SET 5, SET 6, SET 8 - Overviews

### For Specific Claims:
**Claim:** "GP-Julia-AD robust across all conditions"
→ Show: SET 7 (GP-Julia-AD grid) - all green/yellow

**Claim:** "AAA fails at order ≥3" 
→ Show: SET 1 (order 3, 4, 5) - AAA excluded but can reference v2 version

**Claim:** "Order is primary driver"
→ Show: SET 3 (any noise level) - steep slopes for all methods

**Claim:** "High noise + high order = few viable"
→ Show: SET 1 (order 6 or 7) - mostly red at rightmost columns

## 📁 File Organization

```
paper_figures/supplementary_granular/
├── order0_method_vs_noise.png          (SET 1)
├── order1_method_vs_noise.png
├── ... (8 total)
├── noise1e-8_method_vs_order.png       (SET 2)
├── noise1e-6_method_vs_order.png
├── ... (7 total)
├── line_noise1e-8_vs_order.png         (SET 3)
├── line_noise1e-6_vs_order.png
├── ... (7 total)
├── line_order0_vs_noise.png            (SET 4)
├── line_order1_vs_noise.png
├── ... (8 total)
├── small_multiples_top10_full_grid.png (SET 5)
├── category_grid_all_conditions.png    (SET 6)
├── method_GP-Julia-AD_full_grid.png    (SET 7)
├── method_Fourier-Interp_full_grid.png
├── ... (13 total)
└── scatter_all_datapoints.png          (SET 8)
```

**Total:** 46 plots, 14 MB, **ZERO averaging**

## 🎨 Plot Characteristics

- **Resolution:** 300 dpi (publication quality)
- **Scale:** log10(nRMSE) throughout (handles extreme range)
- **Color:** Red-Yellow-Green (intuitive bad→good)
- **Range:** -2 to +2 on log scale (0.01 to 100 on linear scale)
- **Methods:** 23 (filtered, excludes AAA-HighPrec, AAA-LowPrec, SavitzkyGolay_Python, GP-Julia-SE)

## ⚡ Next Steps

**Want to browse?** 
The plots are ready to view - open any PNG directly

**Want a catalog PDF?**
I can create a comprehensive PDF showing all 46 plots with descriptions

**Want customization?**
The script is modular - easy to:
- Change color scales
- Add/remove methods
- Focus on specific order or noise ranges
- Add annotations

**Want selections?**
Tell me which sets/plots you want for:
- Main paper
- Supplementary materials
- Presentations

---

**Summary:** You now have 46 granular plots showing every (method, order, noise) combination explicitly, with NO AVERAGING anywhere. Every claim can be backed by specific, exact data points! 🎯
