#!/bin/bash
# ALWAYS USE THIS TO RUN PYTHON SCRIPTS IN THIS PROJECT
# Uses the uv-managed virtual environment at python/.venv
# Usage: ./run_python.sh path/to/script.py [args...]

python/.venv/bin/python "$@"
