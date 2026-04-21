#!/bin/bash

for i in {1..10}; do
    echo "Run $i/10 starting..."
    uv run experiment.py
    echo "Run $i/10 finished."
    echo "----------------------"
done

echo "All 10 runs completed!"
