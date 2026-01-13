import inspect
import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

def print_source(obj, name):
    try:
        source = inspect.getsource(obj)
        print(f"\n=== Source of {name} ===")
        print(source)
    except Exception as e:
        print(f"Could not get source for {name}: {e}")

print_source(ZImagePipeline.prepare_latents, "ZImagePipeline.prepare_latents")
print_source(ZImageTransformer2DModel.forward, "ZImageTransformer2DModel.forward")

# Also check call method
print_source(ZImagePipeline.__call__, "ZImagePipeline.__call__")
