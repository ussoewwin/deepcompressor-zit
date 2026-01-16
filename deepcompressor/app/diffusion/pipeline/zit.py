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
    """Convert DiffSynth-format state dict to Diffusers/Nunchaku format.
    
    Key mappings based on official svdq-fp4_r128-z-image-turbo.safetensors:
    - attention.qkv -> attention.to_qkv (keep fused, don't split)
    - attention.out -> attention.to_out.0
    - attention.q_norm -> attention.norm_q
    - attention.k_norm -> attention.norm_k
    - feed_forward.w1 + w3 -> feed_forward.net.0.proj (fused gate+up projection)
    - feed_forward.w2 -> feed_forward.net.2
    """
    new_state_dict = {}
    
    # First pass: collect w1 and w3 for fusion
    w1_tensors = {}  # layer_prefix -> tensor
    w3_tensors = {}  # layer_prefix -> tensor
    
    for key, value in state_dict.items():
        # Collect w1 and w3 for later fusion
        match = re.match(r"(.+)\.feed_forward\.w1\.weight", key)
        if match:
            w1_tensors[match.group(1)] = value
            continue
            
        match = re.match(r"(.+)\.feed_forward\.w3\.weight", key)
        if match:
            w3_tensors[match.group(1)] = value
            continue
        
        new_key = key
        
        # attention.qkv -> attention.to_qkv (keep fused for Nunchaku)
        if ".attention.qkv.weight" in key:
            new_key = key.replace(".attention.qkv.weight", ".attention.to_qkv.weight")
        
        # attention.out -> attention.to_out.0
        elif ".attention.out.weight" in key:
            new_key = key.replace(".attention.out.weight", ".attention.to_out.0.weight")
        
        # attention.q_norm -> attention.norm_q
        elif ".attention.q_norm.weight" in key:
            new_key = key.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
        
        # attention.k_norm -> attention.norm_k
        elif ".attention.k_norm.weight" in key:
            new_key = key.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
        
        # feed_forward.w2 -> feed_forward.net.2
        elif ".feed_forward.w2.weight" in key:
            new_key = key.replace(".feed_forward.w2.weight", ".feed_forward.net.2.weight")
        
        # x_embedder -> all_x_embedder.2-1
        elif key.startswith("x_embedder."):
            new_key = key.replace("x_embedder.", "all_x_embedder.2-1.")
        
        # final_layer -> all_final_layer.2-1
        elif key.startswith("final_layer."):
            new_key = key.replace("final_layer.", "all_final_layer.2-1.")
        
        new_state_dict[new_key] = value
    
    # Second pass: fuse w1 and w3 into net.0.proj
    # The official model concatenates them as [w1; w3] along dimension 0
    for layer_prefix in w1_tensors:
        w1 = w1_tensors[layer_prefix]
        w3 = w3_tensors.get(layer_prefix)
        if w3 is not None:
            # Fuse w1 and w3: concatenate along output dimension
            # w1: [10240, 3840], w3: [10240, 3840] -> fused: [20480, 3840]
            fused = torch.cat([w1, w3], dim=0)
            new_key = f"{layer_prefix}.feed_forward.net.0.proj.weight"
            new_state_dict[new_key] = fused
        else:
            # w3 not found, just use w1 with warning
            print(f"Warning: w3 not found for {layer_prefix}, using w1 only")
            new_key = f"{layer_prefix}.feed_forward.net.0.proj.weight"
            new_state_dict[new_key] = w1
    
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

    print("ZImagePipeline built successfully with T5 encoder and monkeypatched prepare_latents")
    return pipe
