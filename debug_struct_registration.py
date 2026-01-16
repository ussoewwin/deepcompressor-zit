import sys
import os
print(f"SYS PATH BEFORE: {sys.path}")
import torch
import torch.nn as nn

# Force local deepcompressor package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct, ZImageTransformerStruct
try:
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
except ImportError:
    # If explicit import fails (e.g. diffusers version mismatch), try to mock just the class for registration check
    # But since we are in venv, it SHOULD work if diffusers is installed.
    # Let's assume venv has it. If not, we fail and know why.
    print("Importing ZImageTransformer2DModel from diffusers...")
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

def log(msg):
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def test_registration():
    if os.path.exists("debug_log.txt"):
        os.remove("debug_log.txt")
        
    log("Testing DiffusionModelStruct registration...")
    log(f"struct file: {sys.modules['deepcompressor.app.diffusion.nn.struct'].__file__}")
    
    # 1. Check if construct finds the ZIT struct
    model = ZImageTransformer2DModel()
    
    log(f"Model attributes: {dir(model)}")
    log(f"Model type: {type(model)}")
    log(f"Model type ID: {id(type(model))}")
    
    try:
        struct = DiffusionModelStruct.construct(model)
        log(f"Constructed struct type: {type(struct)}")
        
        if isinstance(struct, ZImageTransformerStruct):
            log("SUCCESS: Constructed struct IS ZImageTransformerStruct")
        else:
            log(f"FAILURE: Constructed struct is {type(struct)}, expected ZImageTransformerStruct")
            log("Factories available:")
            for k, v in DiffusionModelStruct._factories.items():
                log(f"  Key: {k} (ID: {id(k)}) -> {v}")
                if k == type(model):
                    log("  *** MATCH FOUND BY EQUALITY ***")
                if id(k) == id(type(model)):
                    log("  *** MATCH FOUND BY ID ***")

            # Check if we can verify registration logic manually
            target_class = ZImageTransformer2DModel
            log(f"Target class for registration: {target_class} (ID: {id(target_class)})")
            
    except Exception as e:
        log(f"FAILURE: Exception during construction: {e}")
        import traceback
        traceback.print_exc(file=open("debug_log.txt", "a"))

if __name__ == "__main__":
    test_registration()
