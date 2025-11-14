# Repository Cleanup History

## November 13, 2024 - Pre-sharing Cleanup

Cleaned up repository for sharing with colleagues. All archived materials moved to `deprecated/2024-11-archive/`.

### Archived Directories
- `debugging/` - GP debugging scripts from Oct 28-29, 2024
- `experiments/` - Experimental PyNumDiff validation code
- `rom_exploration/` - ROM/ApproxFun method exploration from Nov 2, 2024
- `backups/` - Miscellaneous backup files
- `gemini-analysis/paper/` - Old paper drafts
- `gemini-analysis/tbd/` - Analysis items marked for deletion

### Archived Development Notes (MD files)
All implementation summaries, design documents, and method notes moved to preserve development history while keeping root directory clean:
- AAA/ROM/ApproxFun implementation summaries
- Design documents and optimization notes
- Method inventories and catalogs
- Paper update tracking documents
- Savitzky-Golay theory and implementation notes

### Archived Scripts
One-time utility scripts used for data transformations:
- Method categorization and renaming scripts
- CSV and prediction file update utilities
- Failed trial retry scripts

### Archived Test Files
Individual GP test files for debugging and reproducibility

### Removed
- `.venv/` (795MB) - Stale virtual environment at root level. Correct venv is at `python/.venv`

### Added to Version Control
**New Python Scripts:**
- `python/flatten_tex.py` - TeX flattening for LLM parsing
- `python/generate_high_order_heatmaps.py` - High-order derivative visualization
- `python/generate_supplemental_heatmaps.py` - Order-specific heatmaps
- `python/generate_speed_accuracy_plot.py` - Speed/accuracy trade-off plots
- `python/plot_speed_accuracy.py`, `python/prepare_speed_accuracy_data.py`
- `python/analyze_order0_impact.py` - Order 0 analysis

**New Paper Sections:**
- `report/sections/section2_related_work.tex` - Related work section
- `report/sections/appendixA_method_catalog_complete.tex` - Complete method catalog
- `report/sections/appendixB_high_order.tex` - High-order analysis
- `report/sections/appendixC_reproducibility.tex` - Reproducibility appendix

**New Scripts & Utilities:**
- `run_python.sh` - Wrapper for correct Python venv
- `scripts/06_flatten_tex.sh` - TeX flattening pipeline step
- `related-work-content/` - Related work research materials

### Repository Structure After Cleanup
```
deriv-estimation-study/
├── methods/          # Method implementations ✓
├── src/              # Study runners ✓
├── python/           # Python infrastructure (cleaned) ✓
├── report/           # LaTeX paper source ✓
├── build/            # Generated outputs ✓
├── scripts/          # Build pipeline (cleaned) ✓
├── docs/             # Documentation ✓
├── lib/              # Custom Julia packages ✓
├── gemini-analysis/  # Analysis scripts (cleaned) ✓
├── deprecated/       # Archived materials (git-ignored) ✓
├── README.md         # Main documentation ✓
└── README_ORGANIZATION.md  # Detailed structure ✓
```

All build pipeline scripts (`build_all.sh`, etc.) verified working after cleanup.
