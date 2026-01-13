import os
import sys
import torch

# キャリブレーションキャッシュの実際の内容を確認
cache_path = '/deepcompressor-zit/datasets/torch.bfloat16/zimage/euler4-g3.5/zit_calib/s128'

print(f"=== キャリブレーションキャッシュ調査 ===")
print(f"キャッシュパス: {cache_path}")

if not os.path.exists(cache_path):
    print(f"ERROR: パスが存在しません: {cache_path}")
    sys.exit(1)

# キャッシュ内のファイル一覧
files = sorted([f for f in os.listdir(cache_path) if f.endswith('.pt')])
print(f"\n.ptファイル数: {len(files)}")
print(f"最初の5ファイル: {files[:5]}")

# 最初のファイルを詳細に調査
if files:
    sample_file = os.path.join(cache_path, files[0])
    print(f"\n=== サンプルファイル調査: {files[0]} ===")
    
    data = torch.load(sample_file, weights_only=False)
    
    print(f"\nトップレベルキー: {list(data.keys())}")
    
    # input_args の調査
    if 'input_args' in data:
        print(f"\n--- input_args ---")
        input_args = data['input_args']
        print(f"型: {type(input_args)}")
        print(f"長さ: {len(input_args) if hasattr(input_args, '__len__') else 'N/A'}")
        
        for i, arg in enumerate(input_args):
            if isinstance(arg, torch.Tensor):
                print(f"  arg[{i}]: Tensor shape={arg.shape}, dtype={arg.dtype}")
            elif isinstance(arg, list):
                print(f"  arg[{i}]: list, len={len(arg)}")
                for j, item in enumerate(arg[:3]):  # 最初の3つだけ
                    if isinstance(item, torch.Tensor):
                        print(f"    [{j}]: Tensor shape={item.shape}, dtype={item.dtype}")
                    else:
                        print(f"    [{j}]: {type(item)}")
            else:
                print(f"  arg[{i}]: {type(arg)} = {arg}")
    
    # input_kwargs の調査
    if 'input_kwargs' in data:
        print(f"\n--- input_kwargs ---")
        input_kwargs = data['input_kwargs']
        print(f"キー: {list(input_kwargs.keys())}")
        
        for key, value in input_kwargs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"  {key}: list, len={len(value)}")
                for j, item in enumerate(value[:2]):
                    if isinstance(item, torch.Tensor):
                        print(f"    [{j}]: Tensor shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  {key}: {type(value)} = {value}")
    
    # outputs の調査
    if 'outputs' in data:
        print(f"\n--- outputs ---")
        outputs = data['outputs']
        if isinstance(outputs, torch.Tensor):
            print(f"  Tensor shape={outputs.shape}, dtype={outputs.dtype}")
        elif isinstance(outputs, tuple):
            print(f"  tuple len={len(outputs)}")
            for i, out in enumerate(outputs[:3]):
                if isinstance(out, torch.Tensor):
                    print(f"    [{i}]: Tensor shape={out.shape}")
    
    # その他のキー
    other_keys = [k for k in data.keys() if k not in ['input_args', 'input_kwargs', 'outputs']]
    if other_keys:
        print(f"\n--- その他のキー ---")
        for key in other_keys:
            value = data[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor shape={value.shape}")
            else:
                print(f"  {key}: {type(value)} = {value}")

print("\n=== 調査完了 ===")
