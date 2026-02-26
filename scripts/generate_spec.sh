#!/bin/bash
# Generate the benchmark specification JSON
# This must be run ONCE before any evaluation.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$BENCH_ROOT"

PYTHON="${PYTHON:-/gemini/code/envs/robotwin/bin/python}"

echo "Generating benchmark spec..."
$PYTHON -m benchmark.spec_generator \
    --output benchmark/benchmark_spec.json \
    --tasks-file tasks_all.txt \
    --repeats 5 \
    --master-seed 42

echo "Done. Spec saved to benchmark/benchmark_spec.json"
