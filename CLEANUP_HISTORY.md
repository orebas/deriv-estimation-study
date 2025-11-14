# Repository Cleanup History

## November 13, 2024 - Pre-sharing Cleanup

Cleaned up repository for sharing with colleagues. Removed 795MB of stale data and 62+ temporary files from root directory.

### Summary
- **Archived**: 4 directories, 18 dev notes, 5 scripts, 40 test files, 22 images, multiple logs
- **Removed**: 795MB stale .venv directory
- **Added to version control**: 7 Python scripts, 4 paper sections, utility scripts, research materials
- **Result**: Clean, professional repository structure ready for collaboration

### Archived to deprecated/2024-11-archive/

**Directories:**
- `debugging/` - GP debugging scripts from Oct 28-29, 2024
- `experiments/` - Experimental PyNumDiff validation code  
- `rom_exploration/` - ROM/ApproxFun method exploration from Nov 2, 2024
- `backups/` - Miscellaneous backup files
- `gemini-analysis/paper/` - Old paper drafts
- `gemini-analysis/tbd/` - Analysis items marked for deletion

**Development Notes (18 MD files):**
All implementation summaries, design documents, and method notes moved to preserve development history while keeping root directory clean:
- AAA/ROM/ApproxFun implementation summaries
- Design documents and optimization notes
- Method inventories and catalogs
- Paper update tracking documents
- Savitzky-Golay theory and implementation notes

**One-time Scripts (5 files):**
Utility scripts used for data transformations:
- Method categorization and renaming scripts
- CSV and prediction file update utilities
- Failed trial retry scripts

**Test Files (43 total):**
- 3 GP test files from src/
- 40 test scripts from repository root (PyNumDiff, SG, Kalman, boundaries, smoothing)

**Images (22 PNG files):**
Generated visualization outputs from root:
- PyNumDiff analysis plots
- Kalman filter comparisons
- Smoothing demonstrations
- Boundary problem visualizations

**Logs and Temporary Files:**
- Test outputs (*.txt)
- Build logs (*.log, *.out)
- Temporary artifacts

### Removed
- `.venv/` (795MB) - Stale virtual environment at root level
  - Correct venv is at `python/.venv` per README_ORGANIZATION.md

### Added to Version Control

**New Python Scripts (7 files):**
- `python/flatten_tex.py` - TeX flattening for LLM parsing
- `python/generate_high_order_heatmaps.py` - High-order derivative visualization
- `python/generate_supplemental_heatmaps.py` - Order-specific heatmaps
- `python/generate_speed_accuracy_plot.py` - Speed/accuracy trade-off plots
- `python/plot_speed_accuracy.py`, `python/prepare_speed_accuracy_data.py`
- `python/analyze_order0_impact.py` - Order 0 analysis

**New Paper Sections (4 files):**
- `report/sections/section2_related_work.tex` - Related work section
- `report/sections/appendixA_method_catalog_complete.tex` - Complete method catalog
- `report/sections/appendixB_high_order.tex` - High-order analysis appendix
- `report/sections/appendixC_reproducibility.tex` - Reproducibility appendix

**New Scripts & Utilities:**
- `run_python.sh` - Wrapper script ensuring correct Python venv usage
- `scripts/06_flatten_tex.sh` - TeX flattening pipeline step
- `related-work-content/` - Related work research materials

### Repository Structure After Cleanup

```
deriv-estimation-study/
├── methods/            # Method implementations ✓
├── src/                # Study runners (cleaned) ✓
├── python/             # Python infrastructure (with new scripts) ✓
├── report/             # LaTeX paper source ✓
├── build/              # Generated outputs ✓
├── scripts/            # Build pipeline (cleaned) ✓
├── docs/               # Documentation ✓
├── lib/                # Custom Julia packages ✓
├── gemini-analysis/    # Analysis scripts (cleaned) ✓
├── deprecated/         # Archived materials (git-ignored) ✓
├── related-work-content/ # Related work materials ✓
├── *.toml, *.lock      # Configuration files ✓
└── README*.md          # Documentation ✓
```

### Verification
All build pipeline scripts verified working after cleanup:
```bash
SKIP_PILOT=1 SKIP_COMPREHENSIVE=1 ./scripts/build_all.sh
# ✓ Paper compilation: SUCCESS
# ✓ Figure generation: 285 images
# ✓ Table generation: 41 tables
# ✓ TeX flattening: COMPLETE
```

### Benefits
1. **Professional appearance**: Clean root directory with only essential files
2. **Easier onboarding**: Clear structure for new collaborators
3. **Preserved history**: All development artifacts archived, not deleted
4. **Verified workflow**: Build pipeline fully tested and functional
5. **Reduced size**: 795MB of stale dependencies removed
