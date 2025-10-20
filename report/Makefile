# Makefile for Derivative Estimation Benchmark Paper
# Usage:
#   make           - Build PDF
#   make clean     - Remove build artifacts
#   make view      - Build and open PDF (requires xdg-open/open)
#   make watch     - Continuous compilation (requires inotifywait)

PAPER = paper
LATEX = pdflatex
BIBTEX = bibtex

# Section dependencies
SECTIONS = sections/section4_methodology.tex \
           sections/section5_methods.tex \
           sections/section6_results.tex \
           sections/section7_discussion.tex

# Figure dependencies (publication versions)
FIGURES = paper_figures/publication/figure1_heatmap.pdf \
          paper_figures/publication/figure2_small_multiples.pdf \
          paper_figures/publication/figure3_qualitative.pdf \
          paper_figures/publication/figure4_pareto.pdf \
          paper_figures/publication/figure5_noise_sensitivity.pdf

.PHONY: all clean view watch help

all: $(PAPER).pdf

# Main build target
$(PAPER).pdf: $(PAPER).tex $(SECTIONS) references.bib $(FIGURES)
	@echo "Building PDF..."
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER) || true
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)
	@echo "PDF built successfully: $(PAPER).pdf"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).out $(PAPER).toc $(PAPER).pdf
	@echo "Clean complete"

# Build and open PDF
view: $(PAPER).pdf
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(PAPER).pdf; \
	elif command -v open >/dev/null 2>&1; then \
		open $(PAPER).pdf; \
	else \
		echo "No PDF viewer found (tried xdg-open, open)"; \
	fi

# Continuous compilation (requires inotifywait)
watch:
	@echo "Watching for changes (Ctrl+C to stop)..."
	@while true; do \
		inotifywait -e modify $(PAPER).tex $(SECTIONS) references.bib 2>/dev/null && \
		make all; \
	done

# Help target
help:
	@echo "Derivative Estimation Benchmark Paper - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make           - Build PDF (default)"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make view      - Build and open PDF"
	@echo "  make watch     - Continuous compilation on file changes"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex (TeX Live or MiKTeX)"
	@echo "  - bibtex"
	@echo "  - inotifywait (for 'make watch', optional)"
