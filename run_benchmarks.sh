#!/usr/bin/env bash
set -euo pipefail

SIZES=(10 50 100 256 512)
RUNS=10
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT_DIR/results"

echo "Working from: $ROOT_DIR"

for SIZE in "${SIZES[@]}"; do
    echo "=== Running benchmarks for SIZE=$SIZE, RUNS=$RUNS ==="

    echo "Running Python benchmark..."
    python3 "$ROOT_DIR/src/main.py" --size "$SIZE" --repeats "$RUNS" --sparse-size "$SIZE"

done

echo "All benchmarks completed. Results are stored in the 'benchmarks' directory."