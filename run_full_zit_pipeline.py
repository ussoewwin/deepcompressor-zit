import os
import shutil
import subprocess
import sys

# 絶対パス設定
BASE_DIR = os.path.abspath("/deepcompressor-zit")
CALIB_CACHE_DIR = os.path.join(BASE_DIR, "datasets/torch.bfloat16/zimage/euler4-g3.5/zit_calib")
RUN_CACHE_DIR = os.path.join(BASE_DIR, "runs/diffusion/cache")

print("=== Starting Full ZIT Pipeline ===")
print(f"Base Dir: {BASE_DIR}")

# Step 1: Update Code
print("\n=== Step 1: Verification ===")
# Git操作はコマンドラインで行うためここではスキップ、あるいは確認のみ

# Step 2: Clear Cache
print("\n=== Step 2: Clearing Old Cache ===")
if os.path.exists(CALIB_CACHE_DIR):
    print(f"Deleting calibration cache: {CALIB_CACHE_DIR}")
    shutil.rmtree(CALIB_CACHE_DIR)
    if os.path.exists(CALIB_CACHE_DIR):
        print("ERROR: Failed to delete cache directory.")
        sys.exit(1)
    print("Cache deleted successfully.")
else:
    print(f"Cache not found (already clean): {CALIB_CACHE_DIR}")

if os.path.exists(RUN_CACHE_DIR):
    print(f"Deleting run cache: {RUN_CACHE_DIR}")
    shutil.rmtree(RUN_CACHE_DIR)

# Step 3: Run Calibration
print("\n=== Step 3: Collecting Calibration Data ===")
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
    print(f"Executing: {' '.join(calib_cmd)}")
    subprocess.run(calib_cmd, env=env, check=True, cwd=BASE_DIR)
    print("Calibration command finished.")
except subprocess.CalledProcessError as e:
    print(f"ERROR: Calibration failed with exit code {e.returncode}")
    sys.exit(1)

# Verify Calibration Output
if not os.path.exists(CALIB_CACHE_DIR):
    print("ERROR: Calibration cache directory was not created!")
    sys.exit(1)

num_files = sum([len(files) for r, d, files in os.walk(CALIB_CACHE_DIR) if any(f.endswith('.pt') for f in files)])
print(f"Generated {num_files} .pt files in cache.")
if num_files == 0:
    print("ERROR: No calibration files generated!")
    sys.exit(1)

# Step 4: Run Quantization
print("\n=== Step 4: Running Quantization ===")
quant_cmd = [
    "python3", "-u", "run_quantize_zit.py"
]

try:
    print(f"Executing: {' '.join(quant_cmd)}")
    subprocess.run(quant_cmd, env=env, check=True, cwd=BASE_DIR)
    print("Quantization finished successfully.")
except subprocess.CalledProcessError as e:
    print(f"ERROR: Quantization failed with exit code {e.returncode}")
    sys.exit(1)
