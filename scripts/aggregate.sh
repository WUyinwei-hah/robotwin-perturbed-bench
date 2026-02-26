#!/bin/bash
# Aggregate benchmark results across all policies
#
# Usage:
#   bash scripts/aggregate.sh [RESULTS_DIR] [OUTPUT_DIR]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$BENCH_ROOT"

RESULTS_DIR="${1:-results}"
OUTPUT_DIR="${2:-summary}"

PYTHON="${PYTHON:-/gemini/code/envs/robotwin/bin/python}"

echo "Aggregating results from $RESULTS_DIR..."

$PYTHON -m benchmark.aggregate_results \
    --results-dir "$RESULTS_DIR" \
    --output "$OUTPUT_DIR"

echo "Summary saved to $OUTPUT_DIR/"
