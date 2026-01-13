import torch
from diffusers import DiffusionPipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

model_id = "Tongyi-MAI/Z-Image-Turbo"

print(f"Loading pipeline from {model_id}...")
try:
    pipe = DiffusionPipeline.from_pretrained(model_id, trust_remote_code=True)
    print("Pipeline loaded successfully.")
    
    print(f"\nPipeline class: {type(pipe)}")
    
    if hasattr(pipe, "vae"):
        print("\n--- VAE Info ---")
        print(f"VAE class: {type(pipe.vae)}")
        if hasattr(pipe.vae, "config"):
            print(f"VAE config: {pipe.vae.config}")
        else:
            print("VAE has no config attribute")
            
    if hasattr(pipe, "transformer"):
        print("\n--- Transformer Info (from pipeline) ---")
        print(f"Transformer class: {type(pipe.transformer)}")
        if hasattr(pipe.transformer, "config"):
            print(f"Transformer config: {pipe.transformer.config}")
            
    print("\n--- Library Default Transformer Info ---")
    default_model = ZImageTransformer2DModel()
    print(f"Default ZImageTransformer2DModel config: {default_model.config}")

except Exception as e:
    print(f"Error loading pipeline: {e}")
