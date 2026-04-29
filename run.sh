#!/usr/bin/env bash
# Reproducible pipeline for the S&P 500 semantic map (Sprint M04).
#
# Usage on a fresh clone:
#     bash run.sh
#
# What it does:
#   1. Re-scrapes the current S&P 500 constituents from Wikipedia and
#      writes data/sp500.csv (existing CSV is overwritten so the figure
#      always reflects the latest list).
#   2. Computes two SemAxis directions, scores every firm, and renders
#      figures/semantic_map.{png,pdf}.
#
# Both Python scripts use PEP-723 inline metadata, so `uv` resolves and
# installs their exact dependencies into ephemeral environments. The
# grader needs `uv` (https://github.com/astral-sh/uv) and nothing else.

set -euo pipefail

mkdir -p data figures

echo "[1/2] Fetching S&P 500 constituents from Wikipedia..."
uv run --with "requests" --with "beautifulsoup4" \
    scripts/fetch_sp500.py

echo "[2/2] Building the semantic map (embeddings + axes + scatter)..."
uv run --script scripts/build_figure.py

echo
echo "Done."
echo "  Data   : data/sp500.csv"
echo "  Scored : data/sp500_scored.csv"
echo "  Figure : figures/semantic_map.png  figures/semantic_map.pdf"
