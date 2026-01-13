import os
import sys
import subprocess
import shutil

deepcompressor_path = '/deepcompressor-zit'
datasets_root = os.path.join(deepcompressor_path, 'datasets/torch.bfloat16/zimage/euler4-g3.5')

# 1. git pull (最新のコード・設定を取得)
print("\n=== Step 1: Updating Code (git fetch + reset) ===")
subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', deepcompressor_path], check=False)
subprocess.run(['git', 'fetch', '--all'], cwd=deepcompressor_path, check=False)
result = subprocess.run(['git', 'reset', '--hard', 'origin/main'], cwd=deepcompressor_path, check=False)
if result.returncode != 0:
    print("WARNING: git reset returned non-zero. Please check connection.")

# 2. キャッシュクリア (古いキャリブレーションデータを削除)
print("\n=== Step 2: Clearing Old Cache ===")
zit_calib_path = os.path.join(datasets_root, 'zit_calib')
runs_cache_path = os.path.join(deepcompressor_path, 'runs/diffusion/cache')

# キャリブレーションキャッシュ削除
if os.path.exists(zit_calib_path):
    print(f"Deleting: {zit_calib_path}")
    shutil.rmtree(zit_calib_path)
    print("[OK] Calibration cache deleted")
else:
    print(f"[SKIP] No calibration cache at {zit_calib_path}")

# 量子化キャッシュ削除
if os.path.exists(runs_cache_path):
    print(f"Deleting: {runs_cache_path}")
    shutil.rmtree(runs_cache_path)
    print("[OK] Quantization cache deleted")
else:
    print(f"[SKIP] No quantization cache at {runs_cache_path}")

# 3. キャリブレーション再収集
print("\n=== Step 3: Collecting Calibration Data ===")
env = os.environ.copy()
env['PYTHONPATH'] = deepcompressor_path
env['XFORMERS_DISABLED'] = '1'
env['PYTHONUNBUFFERED'] = '1'

calib_cmd = [
    'python3', '-u', '-m', 'deepcompressor.app.diffusion.dataset.collect',
    'examples/diffusion/configs/model/zit.yaml',
    'examples/diffusion/configs/collect/zit.yaml',
]

result = subprocess.run(calib_cmd, cwd=deepcompressor_path, env=env, check=False)
if result.returncode != 0:
    print(f"\nERROR: Calibration failed with code {result.returncode}")
    sys.exit(result.returncode)

# 4. キャッシュ確認
print("\n=== Step 4: Verifying Cache Location ===")
target_cache = os.path.join(zit_calib_path, 's128')

if os.path.exists(target_cache):
    print(f"[OK] Cache found at: {target_cache}")
    cache_files = os.listdir(target_cache)
    print(f"Cache contains {len(cache_files)} files/folders")
else:
    print(f"ERROR: Cache not found at {target_cache}")
    print("Calibration may have failed.")
    sys.exit(1)

# 5. 量子化実行
print("\n=== Step 5: Running Quantization (r128) ===")
output_path = '/root/models/z_image_turbo-r128-svdq-fp4.safetensors'
quant_cmd = [
    'python3', '-u', '-m', 'deepcompressor.app.diffusion.ptq',
    'examples/diffusion/configs/model/zit.yaml',
    'examples/diffusion/configs/svdquant/fp4.yaml',
    '--skip-eval',
    '--skip-gen',
    '--export-nunchaku-zit', output_path,
    '--cleanup-run-cache-after-export'
]

result = subprocess.run(quant_cmd, cwd=deepcompressor_path, env=env, check=False)

if result.returncode != 0:
    print(f"\nERROR: Quantization failed with code {result.returncode}")
    sys.exit(result.returncode)
else:
    print(f"\nSUCCESS: Output saved to {output_path}")
