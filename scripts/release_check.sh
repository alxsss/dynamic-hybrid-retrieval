#!/bin/sh
set -eu
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest --cov=dynamic_hybrid_retrieval --cov-report=term-missing
.venv/bin/dhr evaluate --dataset data/evaluation/v1 --output .local/release-evaluation
git diff --check
