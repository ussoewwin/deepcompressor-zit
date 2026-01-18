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
# Official model path for copying Refiner weights
export ZIT_OFFICIAL_MODEL_PATH="/root/models/svdq-fp4_r128-z-image-turbo.safetensors"

echo "=== Running Pre-flight Verification ==="
python tools/verify_struct.py
if [ $? -ne 0 ]; then
    echo "CRITICAL FAILURE: Struct verification failed. Aborting."
    exit 1
fi

python tools/verify_config.py examples/diffusion/configs/svdquant/svdq-fp4-r128.yaml
if [ $? -ne 0 ]; then
    echo "CRITICAL FAILURE: Config verification failed. Aborting."
    exit 1
fi
echo "=== Verification Passed ==="

echo "=== Cleaning up previous cache files ==="

echo "=== Starting ZIT Quantization (PTQ) ==="

python -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/zit.yaml \
    examples/diffusion/configs/quant/svdq-fp4-r128.yaml

echo "=== Quantization complete! ==="
echo "Output saved to: quantized_models/"
