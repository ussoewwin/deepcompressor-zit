#!/bin/bash
# Run ZIT quantization (PTQ) on Vast.ai
# Usage: ./run_quantize_zit.sh

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

echo "=== Starting ZIT Quantization (PTQ) ==="

python -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/zit.yaml \
    examples/diffusion/configs/quant/svdq-fp4-r128.yaml

echo "=== Quantization complete! ==="
echo "Output saved to: quantized_models/"
