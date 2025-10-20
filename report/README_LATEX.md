# LaTeX Build Workflow for Derivative Estimation Benchmark Paper

## Quick Start

```bash
cd /home/orebas/derivative_estimation_study/report
make            # Build PDF
make view       # Build and open PDF
```

## Requirements

- **pdflatex** (from TeX Live or MiKTeX)
- **bibtex** (for bibliography processing)
- **inotifywait** (optional, for `make watch`)

### Installing LaTeX on Linux (WSL)

```bash
sudo apt-get update
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

## File Structure

```
report/
├── paper.tex                          # Main document
├── references.bib                     # Bibliography (TODO markers for citations)
├── Makefile                           # Build automation
├── sections/
│   ├── section4_methodology.tex       # Complete
│   ├── section5_methods.tex           # Complete
│   ├── section6_results.tex           # Complete (with TODOs)
│   └── section7_discussion.tex        # Complete
└── paper_figures/publication/
    ├── figure1_heatmap.pdf
    ├── figure2_small_multiples.pdf
    ├── figure3_qualitative.pdf
    ├── figure4_pareto.pdf
    └── figure5_noise_sensitivity.pdf
```

## Make Targets

| Command | Description |
|---------|-------------|
| `make` | Build PDF (runs pdflatex × 3 + bibtex) |
| `make clean` | Remove build artifacts (.aux, .log, .pdf, etc.) |
| `make view` | Build and open PDF in default viewer |
| `make watch` | Continuous compilation on file changes (requires inotifywait) |
| `make help` | Show help message |

## Current Status

### ✅ Completed Sections (LaTeX)
- **Section 4:** Methodology (experimental design, test system, metrics, validation)
- **Section 5:** Methods Evaluated (24 methods with mathematical formulations)
- **Section 6:** Results (with figure references and TODO markers for tables)
- **Section 7:** Discussion (interpretation, GP excellence, AAA failure, limitations)

### 📝 TODO Sections (Stubs in paper.tex)
- **Section 1:** Introduction
- **Section 2:** Related Work
- **Section 3:** Problem Formulation
- **Section 8:** Practical Recommendations
- **Section 9:** Limitations and Future Work
- **Section 10:** Conclusion
- **Abstract**
- **Appendices**

### 🎨 Figures

**Required format:** PDF (preferred for LaTeX) or PNG  
**Current location:** `paper_figures/publication/`

All figures referenced in Sections 6-7 should exist as PDF files. PNG versions can be converted:

```bash
cd paper_figures/publication
for f in *.png; do convert "$f" "${f%.png}.pdf"; done
```

(Requires ImageMagick: `sudo apt-get install imagemagick`)

### 📚 Bibliography

`references.bib` currently contains only TODO markers. Citations must be added as they are referenced in the paper text using standard BibTeX format.

**Priority citations needed:**
- GaussianProcesses.jl
- AAA algorithm (Nakatsukasa et al.)
- TVRegDiff (Rick Chartrand)
- FFTW (Frigo & Johnson)
- Dierckx splines
- Lotka-Volterra model

## TODO Markers

The paper uses two TODO commands defined in `paper.tex`:

- `\TODO{text}` → Red inline note: **[TODO: text]**
- `\TODOITEM` → Red checkbox: ◼

These compile cleanly (no LaTeX errors) and are highly visible in the PDF.

## Compilation Notes

1. **First build may show warnings** about missing references (normal until references.bib is populated)
2. **Three pdflatex passes** are required for proper cross-references and TOC
3. **Figures must exist** or LaTeX will error; check `paper_figures/publication/` for all figure*.pdf files
4. **TODO markers are intentional** and will remain until data/citations are added

## Workflow for Filling TODOs

1. Identify TODO in compiled PDF (red text)
2. Locate corresponding `\TODO{...}` in source `.tex` file
3. Replace with actual content (data from CSV, citation, etc.)
4. Recompile: `make`
5. Verify in PDF

## Next Steps

1. **Convert PNG figures to PDF** (if not already done)
2. **Populate references.bib** with actual citations
3. **Fill data-driven TODOs** in Section 6 (tables from comprehensive_summary.csv)
4. **Write remaining sections** (1-3, 8-10, Abstract, Appendices)
5. **Author/affiliation information** (currently placeholder)
6. **Final review** by Gemini Pro (style) and GPT-5 (scientific rigor)

## Troubleshooting

**Error: "File `figure1_heatmap.pdf' not found"**
- Check `paper_figures/publication/` directory
- Convert PNG to PDF if needed
- Ensure filename matches exactly (case-sensitive)

**Error: "Undefined control sequence"**
- Check for unescaped special characters: `% $ & # _ { } ~`
- Escape with backslash: `\%` `\$` `\&` etc.

**Bibliography not appearing:**
- Ensure `references.bib` has at least one valid BibTeX entry
- Run full compile: `make clean && make`

**TODO markers causing errors:**
- Verify `\TODO` and `\TODOITEM` commands defined in paper.tex preamble
- Check for unescaped braces in TODO text

## Contact

For questions about the LaTeX workflow or build issues, refer to this README or check the Makefile comments.
