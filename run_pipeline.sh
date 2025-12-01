#!/usr/bin/env bash
set -e

export PYTHONPATH=$PWD:$PYTHONPATH

python -m src.pipeline.run
