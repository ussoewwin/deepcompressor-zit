#!/bin/bash
# Run ZIT calibration data collection on Vast.ai
# Usage: ./run_calib_zit.sh

set -e

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="$(pwd)"
export XFORMERS_DISABLED=1

echo "=== Cleaning up previous cache files ==="
rm -rf datasets
rm -rf jobs

echo "=== Starting ZIT Calibration Data Collection ==="

python -m deepcompressor.app.diffusion.dataset.collect.calib \
    examples/diffusion/configs/model/zit.yaml \
    examples/diffusion/configs/collect/zit.yaml

echo "=== Calibration complete! ==="
echo "Output saved to: datasets/"
