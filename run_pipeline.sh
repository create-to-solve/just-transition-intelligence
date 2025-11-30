#!/usr/bin/env bash
# Basic pipeline run script

set -e

# Activate environment (optional — if using conda)
# conda activate your_env_name

echo "Running full JTIS pipeline..."
python -m src.pipeline.run

echo "Pipeline complete."
