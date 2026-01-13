import os
import shutil
import subprocess
import sys
import torch
import glob

# Constants
BASE_DIR = os.path.abspath("/deepcompressor-zit")
CALIB_CACHE_DIR = os.path.join(BASE_DIR, "datasets/torch.bfloat16/zimage/euler4-g3.5/zit_calib")
CACHE_SUBDIR = os.path.join(CALIB_CACHE_DIR, "s128", "caches")
RUN_CACHE_DIR = os.path.join(BASE_DIR, "runs/diffusion/cache")
EXPECTED_CHANNELS = 16

def check_cache_validity(cache_dir):
    """Check if cache exists and has correct shape (16 channels)."""
    pt_files = glob.glob(os.path.join(cache_dir, "*.pt"))
    if not pt_files:
        print("Cache is empty or missing.")
        return False
    
    # Check first file
    try:
        data = torch.load(pt_files[0], weights_only=False)
        # Check input_args[0] or whichever contains the latents
        # Based on previous dump: input_args[0] is list of tensors
        if 'input_args' in data and len(data['input_args']) > 0:
            arg0 = data['input_args'][0]
            if isinstance(arg0, list) and len(arg0) > 0:
                 latent = arg0[0]
                 if latent.dim() == 4:
                     channels = latent.shape[1]
                     print(f"Checking cache file {os.path.basename(pt_files[0])}: detected {channels} channels.")
                     if channels == EXPECTED_CHANNELS:
                         return True
                     else:
                         print(f"Invalid channel count: {channels} (expected {EXPECTED_CHANNELS})")
                         return False
    except Exception as e:
        print(f"Error checking cache file: {e}")
        return False
    
    return False

print("=== Smart ZIT Pipeline ===")
print(f"Base Dir: {BASE_DIR}")

# Step 1: Cache Verification & cleanup
print("\n=== Step 1: Cache Verification ===")
need_calibration = True

if os.path.exists(CACHE_SUBDIR):
    if check_cache_validity(CACHE_SUBDIR):
        print("VALID CACHE FOUND. Skipping calibration.")
        need_calibration = False
    else:
        print("INVALID CACHE DETECTED. Deleting...")
        if os.path.exists(CALIB_CACHE_DIR):
            shutil.rmtree(CALIB_CACHE_DIR)
        print("Cache deleted.")
else:
    print("Cache not found.")

# Step 2: Calibration (only if needed)
if need_calibration:
    print("\n=== Step 2: Running Calibration ===")
    calib_cmd = [
        "python3", "-u", "-m", "deepcompressor.app.diffusion.dataset.collect.calib",
        "examples/diffusion/configs/model/zit.yaml",
        "examples/diffusion/configs/collect/zit.yaml"
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = BASE_DIR
    env['XFORMERS_DISABLED'] = '1'
    env['PYTHONUNBUFFERED'] = '1'

    try:
        subprocess.run(calib_cmd, env=env, check=True, cwd=BASE_DIR)
        print("Calibration finished.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Calibration failed with exit code {e.returncode}")
        sys.exit(1)

# Step 3: Quantization
print("\n=== Step 3: Running Quantization ===")
# Clear run cache to be safe
if os.path.exists(RUN_CACHE_DIR):
    shutil.rmtree(RUN_CACHE_DIR)

quant_cmd = [
    "python3", "-u", "run_quantize_zit.py"
]
env = os.environ.copy()
env['PYTHONPATH'] = BASE_DIR
env['XFORMERS_DISABLED'] = '1'
env['PYTHONUNBUFFERED'] = '1'

try:
    subprocess.run(quant_cmd, env=env, check=True, cwd=BASE_DIR)
    print("Quantization finished successfully.")
except subprocess.CalledProcessError as e:
    print(f"ERROR: Quantization failed with exit code {e.returncode}")
    sys.exit(1)
