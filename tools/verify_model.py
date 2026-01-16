
import os
import argparse
from safetensors.torch import safe_open
import torch

def verify_model(generated_path, official_path):
    print(f"Verifying Generated Model: {generated_path}")
    print(f"Against Official Model:  {official_path}")

    if not os.path.exists(generated_path):
        print(f"ERROR: Generated model not found at {generated_path}")
        return
    if not os.path.exists(official_path):
        print(f"ERROR: Official model not found at {official_path}")
        return

    gen_meta = {}
    off_meta = {}
    
    gen_tensors = {}
    off_tensors = {}

    print("Loading models metadata...")
    with safe_open(generated_path, framework='pt', device='cpu') as f_gen:
        gen_keys = set(f_gen.keys())
        for k in f_gen.keys():
            t = f_gen.get_tensor(k)
            gen_tensors[k] = {'shape': list(t.shape), 'dtype': str(t.dtype)}

    with safe_open(official_path, framework='pt', device='cpu') as f_off:
        off_keys = set(f_off.keys())
        for k in f_off.keys():
            t = f_off.get_tensor(k)
            off_tensors[k] = {'shape': list(t.shape), 'dtype': str(t.dtype)}

    print(f"Generated Keys: {len(gen_keys)}")
    print(f"Official Keys:  {len(off_keys)}")

    # 1. Missing Keys (Critical)
    missing = off_keys - gen_keys
    if missing:
        print(f"\n[CRITICAL] Missing Keys in Generated Model ({len(missing)}):")
        for k in sorted(list(missing))[:20]:
            print(f"  - {k}")
        if len(missing) > 20: print(f"  ... and {len(missing)-20} more")
    else:
        print("\n[OK] No missing keys.")

    # 2. Extra Keys (Suspicious)
    extra = gen_keys - off_keys
    if extra:
        print(f"\n[WARNING] Extra Keys in Generated Model ({len(extra)}):")
        for k in sorted(list(extra))[:20]:
            print(f"  + {k}")
        if len(extra) > 20: print(f"  ... and {len(extra)-20} more")
    else:
        print("\n[OK] No extra keys.")

    # 3. Shape/Type Mismatch
    common = gen_keys.intersection(off_keys)
    mismatches = []
    for k in common:
        g = gen_tensors[k]
        o = off_tensors[k]
        if g['shape'] != o['shape'] or g['dtype'] != o['dtype']:
            mismatches.append((k, g, o))
    
    if mismatches:
        print(f"\n[CRITICAL] Shape/Dtype Mismatches ({len(mismatches)}):")
        for k, g, o in sorted(mismatches)[:20]:
            print(f"  * {k}:")
            print(f"    Gen: {g['shape']} {g['dtype']}")
            print(f"    Off: {o['shape']} {o['dtype']}")
    else:
        print("\n[OK] All common tensors match shape and dtype.")
        
    # 4. Refiner Specific Check
    refiner_missing = [k for k in missing if 'refiner' in k]
    if refiner_missing:
        print(f"\n[ANALYSIS] Refiner layers are MISSING. Struct registration failed.")
    
    # 5. SVD Check
    lora_missing = [k for k in missing if 'proj_down' in k or 'proj_up' in k]
    if lora_missing:
         print(f"\n[ANALYSIS] SVD Low-Rank branches are MISSING. SVD decomposition failed.")
         
    # 6. Quantization Check
    scale_missing = [k for k in missing if 'wscales' in k or 'wcscales' in k or 'wtscale' in k]
    if scale_missing:
         print(f"\n[ANALYSIS] Quantization Scales are MISSING. Quantization logic failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", default=r"D:\nu\z_image_turbo-r128-svdq-fp4.safetensors")
    parser.add_argument("--off_path", default=r"D:\nu\svdq-fp4_r128-z-image-turbo.safetensors")
    args = parser.parse_args()
    
    verify_model(args.gen_path, args.off_path)
