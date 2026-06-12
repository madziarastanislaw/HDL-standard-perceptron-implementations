#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v pdflatex &>/dev/null; then
    echo "ERROR: pdflatex nie znaleziony."
    echo "  sudo apt install texlive-full"
    exit 1
fi

echo "=== Kompilacja testy.tex (1/2) ==="
pdflatex -interaction=nonstopmode testy.tex

echo "=== Kompilacja testy.tex (2/2) ==="
pdflatex -interaction=nonstopmode testy.tex

echo "=== Gotowe: testy.pdf ==="
