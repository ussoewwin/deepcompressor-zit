import torch
import sys

f = '/deepcompressor-zit/datasets/torch.bfloat16/zimage/euler4-g3.5/zit_calib/s128/caches/img_001-0-00000-0.pt'
d = torch.load(f, weights_only=False)
print('Keys:', list(d.keys()))

if 'input_args' in d:
    print('\n--- input_args ---')
    for i, a in enumerate(d['input_args']):
        if hasattr(a, 'shape'):
            print(f'input_args[{i}]: Tensor shape={a.shape} dtype={a.dtype}')
        elif isinstance(a, list):
            print(f'input_args[{i}]: list len={len(a)}')
            for j, x in enumerate(a[:3]):
                if hasattr(x, 'shape'):
                    print(f'  [{j}]: Tensor shape={x.shape} dtype={x.dtype}')
                else:
                    print(f'  [{j}]: {type(x).__name__}')
        else:
            print(f'input_args[{i}]: {type(a).__name__} = {a}')

if 'input_kwargs' in d:
    print('\n--- input_kwargs ---')
    for k, v in d['input_kwargs'].items():
        if hasattr(v, 'shape'):
            print(f'{k}: Tensor shape={v.shape} dtype={v.dtype}')
        elif isinstance(v, list):
            print(f'{k}: list len={len(v)}')
            for j, x in enumerate(v[:2]):
                if hasattr(x, 'shape'):
                    print(f'  [{j}]: Tensor shape={x.shape}')
        else:
            print(f'{k}: {type(v).__name__} = {v}')
