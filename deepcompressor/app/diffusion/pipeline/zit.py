# -*- coding: utf-8 -*-
"""ZIT (Z-Image Turbo) specific pipeline for deepcompressor quantization.

This module provides a complete ZIT pipeline that includes T5 text encoder
and proper scheduling for calibration data collection.
"""

import re
import torch
import torch.nn as nn
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

__all__ = ["load_zit_transformer", "build_zit_pipeline", "convert_diffsynth_to_diffusers"]


def convert_diffsynth_to_diffusers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert DiffSynth-format state dict to Diffusers format.
    
    DiffSynth uses fused QKV weights and different naming conventions.
    Diffusers uses separated Q, K, V weights.
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Key renaming patterns
        # attention.qkv -> split into to_q, to_k, to_v
        if ".attention.qkv.weight" in key:
            # Split fused QKV into separate Q, K, V
            # QKV is concatenated as [Q, K, V] where each has same size
            # Total out_features = 3 * hidden_size, so each part = total / 3
            total_out = value.shape[0]
            chunk_size = total_out // 3
            
            q = value[:chunk_size, :]
            k = value[chunk_size:chunk_size*2, :]
            v = value[chunk_size*2:, :]
            
            base_key = key.replace(".attention.qkv.weight", ".attention")
            new_state_dict[f"{base_key}.to_q.weight"] = q
            new_state_dict[f"{base_key}.to_k.weight"] = k
            new_state_dict[f"{base_key}.to_v.weight"] = v
            continue
        
        # attention.out -> attention.to_out.0
        if ".attention.out.weight" in key:
            new_key = key.replace(".attention.out.weight", ".attention.to_out.0.weight")
        
        # attention.q_norm -> attention.norm_q
        elif ".attention.q_norm.weight" in key:
            new_key = key.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
        
        # attention.k_norm -> attention.norm_k
        elif ".attention.k_norm.weight" in key:
            new_key = key.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
        
        # x_embedder -> all_x_embedder.2-1
        elif key.startswith("x_embedder."):
            new_key = key.replace("x_embedder.", "all_x_embedder.2-1.")
        
        # final_layer -> all_final_layer.2-1
        elif key.startswith("final_layer."):
            new_key = key.replace("final_layer.", "all_final_layer.2-1.")
        
        # x_pad_token stay as is
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def load_zit_transformer(
    transformer_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> ZImageTransformer2DModel:
    """Load ZImageTransformer2DModel from safetensors file.
    
    Args:
        transformer_path: Path to the safetensors file.
        dtype: Model dtype (default: bfloat16).
        device: Device to load the model on.
        
    Returns:
        Loaded ZImageTransformer2DModel.
    """
    print(f"Loading ZIT transformer from: {transformer_path}")
    
    # Load state dict from safetensors (DiffSynth format)
    state_dict = load_file(transformer_path, device="cpu")
    
    # Convert to Diffusers format
    print("Converting state dict from DiffSynth to Diffusers format...")
    state_dict = convert_diffsynth_to_diffusers(state_dict)
    
    # Create model with default config (on meta device first for efficiency)
    # Default config matches Z-Image Turbo: 30 layers, 24 heads, etc.
    with torch.device("meta"):
        model = ZImageTransformer2DModel(in_channels=16)
        print(f"DEBUG: ZImageTransformer2DModel initialized with in_channels={model.config.in_channels}")
    
    # Move model to real device and load weights
    model = model.to_empty(device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"Warning: Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Warning: Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    
    model = model.to(dtype=dtype)
    model.eval()
    
    print(f"ZIT transformer loaded successfully with {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
    
    return model


def build_zit_pipeline(
    transformer_path: str | None,
    hf_model_path: str = "Tongyi-MAI/Z-Image-Turbo",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> ZImagePipeline:
    """Build a complete ZImagePipeline with real T5 encoder and scheduler.
    
    This loads the transformer from a safetensors file and combines it with
    the T5 text encoder and other components from HuggingFace.
    
    Args:
        transformer_path: Path to the transformer safetensors file.
                          If None, loads directly from HuggingFace.
        hf_model_path: HuggingFace model path for T5 encoder and scheduler.
        dtype: Model dtype (default: bfloat16).
        device: Device to load the model on.
        
    Returns:
        Complete ZImagePipeline ready for inference.
    """
    print(f"Building ZIT pipeline from HuggingFace: {hf_model_path}")
    
    if transformer_path:
        # Load transformer from local safetensors file
        transformer = load_zit_transformer(transformer_path, dtype, device="cpu")
        pipe = ZImagePipeline.from_pretrained(
            hf_model_path,
            transformer=transformer,
            torch_dtype=dtype,
        )
        print(f"DEBUG: ZImagePipeline built from local. Transformer in_channels={pipe.transformer.config.in_channels}")
    else:
        # Load entire pipeline directly from HuggingFace
        print("No transformer_path provided, loading from HuggingFace directly...")
        pipe = ZImagePipeline.from_pretrained(
            hf_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
    
    # Move to device
    pipe = pipe.to(device)
    
    # Monkeypatch prepare_latents to ensure correct channels and debug
    import types
    from diffusers.utils.torch_utils import randn_tensor
    
    def custom_prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        print(f"DEBUG: custom_prepare_latents called with num_channels_latents={num_channels_latents}, height={height}, width={width}")
        
        # Force 16 channels if ZIT
        if num_channels_latents != 16:
            print(f"WARNING: Forcing num_channels_latents to 16 (was {num_channels_latents})")
            num_channels_latents = 16

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)
        
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                print(f"DEBUG: latents mismatch shape {latents.shape} vs {shape}")
                pass
            latents = latents.to(device)
            
        print(f"DEBUG: prepare_latents returning shape {latents.shape}")
        return latents

    pipe.prepare_latents = types.MethodType(custom_prepare_latents, pipe)
    
    # === CRITICAL FIX ===
    # Wrap embedder to fix shape mismatch (Batch*16, 1) -> (Batch, 16)
    class ZITPatchEmbedWrapper(torch.nn.Module):
        def __init__(self, original_embedder):
            super().__init__()
            self.original_embedder = original_embedder
            
        def forward(self, x):
            # Unconditional debug print
            print(f"DEBUG: ZITPatchEmbedWrapper forward input: {x.shape}")
            # Detect flattened input: [Batch*16, 1, H, W] -> needs [Batch, 16, H, W]
            if x.dim() == 4 and x.shape[1] == 1:
                if x.shape[0] % 16 == 0:
                     print(f"DEBUG: Reshaping 1-channel flattened input to 16-channel")
                     batch_size = x.shape[0] // 16
                     x = x.view(batch_size, 16, x.shape[2], x.shape[3])
            return self.original_embedder(x)
        
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.original_embedder, name)

    if hasattr(pipe.transformer, "all_x_embedder"):
        print("DEBUG: Applying ZITPatchEmbedWrapper to transformer.all_x_embedder")
        for key in pipe.transformer.all_x_embedder.keys():
            original = pipe.transformer.all_x_embedder[key]
            # Ensure we don't double-wrap
            if not isinstance(original, ZITPatchEmbedWrapper):
                pipe.transformer.all_x_embedder[key] = ZITPatchEmbedWrapper(original)

    print("ZImagePipeline built successfully with T5 encoder, monkeypatched prepare_latents, and PatchEmbedWrapper")
    return pipe
