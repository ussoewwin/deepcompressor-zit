
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

# 2. キャッシュフォルダの確認 (元のスクリプト通り)
print("\n=== Step 2: Verifying Cache Location ===")
zit_calib_path = os.path.join(datasets_root, 'zit_calib')
target_cache = os.path.join(zit_calib_path, 's128')

if os.path.exists(target_cache):
    print(f"[OK] Cache found at: {target_cache}")
    # キャッシュ内のファイル数を確認
    try:
        cache_files = os.listdir(target_cache)
        print(f"Cache contains {len(cache_files)} files/folders")
    except Exception as e:
        print(f"WARNING: Could not list cache files: {e}")
else:
    print(f"WARNING: Cache not found at {target_cache}")
    print("Calibration data may need to be collected first.")

# 3. 量子化実行
print("\n=== Step 3: Running Quantization (r128) ===")
env = os.environ.copy()
env['PYTHONPATH'] = deepcompressor_path
env['XFORMERS_DISABLED'] = '1'
env['PYTHONUNBUFFERED'] = '1'

output_path = '/root/models/z_image_turbo-r128-svdq-fp4.safetensors'
cmd = [
    'python3', '-u', '-m', 'deepcompressor.app.diffusion.ptq',
    'examples/diffusion/configs/model/zit.yaml',
    'examples/diffusion/configs/svdquant/fp4.yaml',
    '--skip-eval',
    '--skip-gen',
    '--export-nunchaku-zit', output_path,
    '--cleanup-run-cache-after-export'
]

result = subprocess.run(cmd, cwd=deepcompressor_path, env=env, check=False)

if result.returncode != 0:
    print(f"\nERROR: Quantization failed with code {result.returncode}")
    sys.exit(result.returncode)
else:
    print(f"\nSUCCESS: Output saved to {output_path}")
