#!/bin/bash
# Flatten the LaTeX paper into a single file for easier LLM parsing

echo "====================================="
echo "STEP 6: FLATTEN TEX FILE"
echo "====================================="
echo ""
echo "Creating flattened version of paper for LLM parsing..."
echo ""

# Run the flattening script
./run_python.sh python/flatten_tex.py

echo ""
echo "Flattened TeX files created in build/flattened/"
echo "  - paper_flattened.tex: Full TeX with all includes expanded"
echo "  - paper_text.txt: Simplified text version"
echo ""
echo "====================================="
echo "FLATTEN COMPLETE"
echo "====================================="