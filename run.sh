#!/bin/bash

N=${1:-10}

for i in $(seq 1 "$N"); do
    echo "Run $i/$N starting..."
    uv run experiment.py
    echo "Run $i/$N finished."
    echo "----------------------"
done

echo "All $N runs completed!"
