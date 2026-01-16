import sys
import os
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct, ZImageTransformerStruct
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

def verify_struct_registration():
    """
    Verifies that ZImageTransformerStruct is correctly registered as a factory
    for ZImageTransformer2DModel in the DiffusionModelStruct class.
    """
    print("Verifying ZImageTransformerStruct registration...")
    
    # Check if ZImageTransformer2DModel is in the _factories dict of DiffusionModelStruct
    factory = DiffusionModelStruct._factories.get(ZImageTransformer2DModel, None)
    
    is_registered = False
    if factory:
        # Check if the factory function is one of our constructors
        if factory == ZImageTransformerStruct._default_construct or factory == ZImageTransformerStruct._construct_direct:
            is_registered = True
            
    if is_registered:
        print("SUCCESS: ZImageTransformerStruct is correctly registered.")
        return 0
    else:
        print("FAILURE: ZImageTransformerStruct is NOT registered in DiffusionModelStruct.")
        print("Debug Info:")
        print(f"  Registered factories for {ZImageTransformer2DModel.__name__}: {factories}")
        return 1

if __name__ == "__main__":
    sys.exit(verify_struct_registration())
