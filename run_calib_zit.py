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

# 2. キャッシュの再生成
print("\n=== Step 2: Regenerating Calibration Cache ===")
zit_calib_path = os.path.join(datasets_root, 'zit_calib')

# 既存の腐ったキャッシュを削除
if os.path.exists(zit_calib_path):
    print(f"Removing old cache at: {zit_calib_path}")
    shutil.rmtree(zit_calib_path)

# Also remove jobs directory to force fresh quantization cache
jobs_path = os.path.join(deepcompressor_path, 'jobs')
if os.path.exists(jobs_path):
    print(f"Removing old jobs cache at: {jobs_path}")
    shutil.rmtree(jobs_path)

# キャリブレーション実行
print("Running calibration...")
env = os.environ.copy()
env['PYTHONPATH'] = deepcompressor_path
env['XFORMERS_DISABLED'] = '1'
env['PYTHONUNBUFFERED'] = '1'

calib_cmd = [
    'python3', '-u', '-m', 'deepcompressor.app.diffusion.dataset.collect.calib',
    'examples/diffusion/configs/model/zit.yaml',
    'examples/diffusion/configs/collect/zit.yaml'
]

result = subprocess.run(calib_cmd, cwd=deepcompressor_path, env=env, check=False)
if result.returncode != 0:
    print(f"\nERROR: Calibration failed with code {result.returncode}")
    sys.exit(result.returncode)
else:
    print("\nSUCCESS: Calibration completed.")
