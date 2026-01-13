
import inspect
import torch
import sys

print("=== ZImagePipeline Source ===")
try:
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
    print(inspect.getsource(ZImagePipeline.prepare_latents))
except Exception as e:
    print(f"Could not dump pipeline: {e}")

print("\n=== ZImageTransformer2DModel Source ===")
try:
    from diffusers.models.transformers import transformer_z_image
    print(inspect.getsource(transformer_z_image.ZImageTransformer2DModel.forward))
except Exception as e:
    print(f"Could not dump transformer forward: {e}")

print("\n=== ZImageTransformer2DModel __init__ Source ===")
try:
    from diffusers.models.transformers import transformer_z_image
    print(inspect.getsource(transformer_z_image.ZImageTransformer2DModel.__init__))
except Exception as e:
    print(f"Could not dump transformer init: {e}")
