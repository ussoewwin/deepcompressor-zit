import torch
import inspect
from diffusers import DiffusionPipeline

model_id = "Tongyi-MAI/Z-Image-Turbo"
pipe = DiffusionPipeline.from_pretrained(model_id, trust_remote_code=True)

# 1. Inspect Pipeline Call Signature
print("--- Pipeline Call Signature ---")
print(inspect.signature(pipe.__call__))

# 2. Trace VAE Encoding
print("\n--- Tracing VAE Encoding ---")
image = torch.zeros((1, 3, 1024, 1024)) # Dummy input
try:
    with torch.no_grad():
        encoded = pipe.vae.encode(image).latent_dist.sample()
        print(f"VAE Encoded Shape: {encoded.shape}")
except Exception as e:
    print(f"VAE Encoding Failed: {e}")

# 3. Simulate Pipeline Latent Preparation
print("\n--- Simulating Latent Preparation ---")
try:
    # Try to find prepare_latents method
    if hasattr(pipe, "prepare_latents"):
        print("Existing prepare_latents found.")
        # We can't easily call it without partial args, but existence is key
    else:
        print("No prepare_latents method found.")
except Exception as e:
    print(f"Inspection failed: {e}")
