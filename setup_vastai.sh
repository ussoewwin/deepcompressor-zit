#!/bin/bash
# Vast.ai setup script for ZIT quantization
# Run this script after connecting to your Vast.ai instance

set -e

echo "=== ZIT Quantization Environment Setup ==="

# Clone the repository
echo "[1/5] Cloning deepcompressor-zit repository..."
git clone https://github.com/ussoewwin/deepcompressor-zit
cd deepcompressor-zit

# Create and activate virtual environment
echo "[2/5] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "[3/5] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers diffusers accelerate safetensors
pip install omegaconf omniconfig datasets tqdm pillow
pip install poetry
poetry install --no-root

# Download Z-Image Turbo model from HuggingFace
echo "[4/5] Downloading Z-Image Turbo model..."
mkdir -p models
python3 -c "
from huggingface_hub import hf_hub_download
import os

# Download the BF16 transformer model
print('Downloading Z-Image Turbo transformer...')
hf_hub_download(
    repo_id='Tongyi-MAI/Z-Image-Turbo',
    filename='transformer/diffusion_pytorch_model-00001-of-00002.safetensors',
    local_dir='models/z-image-turbo',
    local_dir_use_symlinks=False
)
hf_hub_download(
    repo_id='Tongyi-MAI/Z-Image-Turbo',
    filename='transformer/diffusion_pytorch_model-00002-of-00002.safetensors',
    local_dir='models/z-image-turbo',
    local_dir_use_symlinks=False
)
hf_hub_download(
    repo_id='Tongyi-MAI/Z-Image-Turbo',
    filename='transformer/diffusion_pytorch_model.safetensors.index.json',
    local_dir='models/z-image-turbo',
    local_dir_use_symlinks=False
)
print('Download complete!')
"

echo "[5/5] Setup complete!"
echo ""
echo "=== Next Steps ==="
echo "1. Update the transformer_path in examples/diffusion/configs/model/zit.yaml"
echo "2. Run: source venv/bin/activate"
echo "3. Run calibration: ./run_calib_zit.sh"
echo ""
