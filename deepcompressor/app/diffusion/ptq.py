"""Z-Image Turbo Nunchaku Export for DeepCompressor.

This module provides export functionality for Z-Image Turbo models quantized with SVDQuant.
Based on the official svdq-fp4_r128-z-image-turbo.safetensors structure.
"""
import gc
import json
import os
import sys

# Force local deepcompressor package to be used (Fix for Environment/Import issues)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
# Explicitly import ZImageTransformerStruct to ensure factory registration
try:
    from .nn.struct import ZImageTransformerStruct
except ImportError:
    pass # Might fail if relative import issues, but path hack should fix global import

import pprint
import traceback

import safetensors
import safetensors.torch
import torch
from diffusers import DiffusionPipeline

from deepcompressor.backend.nunchaku.convert import convert_to_nunchaku_w4x4y16_linear_state_dict
from deepcompressor.utils import tools

from .config import DiffusionPtqCacheConfig, DiffusionPtqRunConfig, DiffusionQuantCacheConfig, DiffusionQuantConfig
from .nn.struct import DiffusionModelStruct
from .quant import (
    load_diffusion_weights_state_dict,
    quantize_diffusion_activations,
    quantize_diffusion_weights,
    rotate_diffusion,
    smooth_diffusion,
)

__all__ = ["ptq"]


def _load_safetensors_state_dict(path: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict


def _detect_precision(wgts_dtype) -> str:
    """Detect precision from weight quantization dtype."""
    if wgts_dtype is None:
        return "nvfp4"
    dtype_str = str(wgts_dtype).lower()
    if "sfp4" in dtype_str or "fp4" in dtype_str or "nvfp4" in dtype_str:
        return "nvfp4"
    elif "sint4" in dtype_str or "int4" in dtype_str:
        return "int4"
    return "nvfp4"


def _zit_export_build_metadata(*, transformer, rank: int, precision: str) -> dict[str, str]:
    """Build metadata for Z-Image Turbo Nunchaku export."""
    # Get transformer config
    transformer_cfg = getattr(transformer, "config", None)
    if hasattr(transformer_cfg, "to_dict"):
        transformer_cfg = transformer_cfg.to_dict()
    if not isinstance(transformer_cfg, dict):
        # Fallback default config for Z-Image Turbo
        transformer_cfg = {
            "_class_name": "ZImageTransformer2DModel",
            "_diffusers_version": "0.36.0.dev0",
            "all_f_patch_size": [1],
            "all_patch_size": [2],
            "axes_dims": [32, 48, 48],
            "axes_lens": [1536, 512, 512],
            "cap_feat_dim": 2560,
            "dim": 3840,
            "in_channels": 16,
            "n_heads": 30,
            "n_kv_heads": 30,
            "n_layers": 30,
            "n_refiner_layers": 2,
            "norm_eps": 1e-05,
            "qk_norm": True,
            "rope_theta": 256.0,
            "t_scale": 1000.0,
        }
    
    # Build quantization config matching official format
    quant_config = {
        "method": "svdquant",
        "weight": {"dtype": "fp4_e2m1_all", "scale_dtype": [None, "fp8_e4m3_nan"], "group_size": 16},
        "activation": {"dtype": "fp4_e2m1_all", "scale_dtype": "fp8_e4m3_nan", "group_size": 16},
        "rank": int(rank),
        "skip_refiners": False,
    }
    
    return {
        "model_class": "NunchakuZImageTransformer2DModel",
        "config": json.dumps(transformer_cfg),
        "quantization_config": json.dumps(quant_config),
        "comfy_config": "{}",
    }


def _process_zit_linear(
    *,
    module_name: str,
    orig_state: dict[str, torch.Tensor],
    dequant_state: dict[str, torch.Tensor],
    scale_state: dict[str, torch.Tensor],
    branch_state: dict[str, dict[str, torch.Tensor]] | None,
    smooth_cache: dict[str, torch.Tensor] | None,
    out_state: dict[str, torch.Tensor],
    config: object | None = None,
    rank: int,
    torch_dtype: torch.dtype,
    float_point: bool,
    is_qkv_fused: bool = False,
    qkv_modules: list[str] | None = None,
) -> None:
    """Process a single linear layer for Z-Image Turbo export."""
    
    def _get_scale(name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        def get_raw(n):
            s0 = scale_state.get(f"{n}.weight.scale.0", None)
            s1 = scale_state.get(f"{n}.weight.scale.1", None)
            if s0 is not None and not isinstance(s0, torch.Tensor):
                s0 = torch.tensor([float(s0)], dtype=torch_dtype, device="cpu")
            if s1 is not None and not isinstance(s1, torch.Tensor):
                s1 = torch.tensor([float(s1)], dtype=torch_dtype, device="cpu")
            # Normalize to 4D [C, 1, G, 1] or [C, 1, 1, 1]
            if s0 is not None:
                if s0.ndim == 0: s0 = s0.view(1, 1, 1, 1)
                elif s0.ndim == 1: s0 = s0.view(-1, 1, 1, 1)
            if s1 is not None:
                if s1.ndim == 0: s1 = s1.view(1, 1, 1, 1)
                elif s1.ndim == 1: s1 = s1.view(-1, 1, 1, 1)
            return s0, s1

        # Try exact match
        s0, s1 = get_raw(name)
        if s0 is not None:
            return s0, s1

        # Fallback 1: QKV Fusion (to_qkv -> to_q, to_k, to_v)
        if ".attention.to_qkv" in name:
            q_name = name.replace(".attention.to_qkv", ".attention.to_q")
            k_name = name.replace(".attention.to_qkv", ".attention.to_k")
            v_name = name.replace(".attention.to_qkv", ".attention.to_v")
            
            qs0, qs1 = get_raw(q_name)
            ks0, ks1 = get_raw(k_name)
            vs0, vs1 = get_raw(v_name)
            
            if qs0 is not None and ks0 is not None and vs0 is not None:
                # Need to expand scales to match output channels
                # Standard scalar scale is [1, 1, 1, 1]
                # We need [Q_OC, 1, 1, 1] etc.
                
                q_oc, k_oc, v_oc = 0, 0, 0
                if config:
                    head_dim = getattr(config, "attention_head_dim", 64)
                    q_heads = getattr(config, "num_attention_heads", 0)
                    kv_heads = getattr(config, "num_key_value_heads", q_heads)
                    if kv_heads is None: kv_heads = q_heads
                    
                    q_oc = head_dim * q_heads
                    k_oc = head_dim * kv_heads
                    v_oc = head_dim * kv_heads
                
                # If config failed or 0, try to infer from orig_state (assuming equal split)
                if q_oc == 0:
                     fused_w = orig_state.get(f"{name}.weight")
                     if fused_w is not None:
                         total_oc = fused_w.shape[0]
                         q_oc = k_oc = v_oc = total_oc // 3
                
                def expand(s, size):
                    if s is None: return None
                    if s.shape[0] == 1 and size > 1:
                        return s.expand(size, -1, -1, -1).clone()
                    return s

                qs0 = expand(qs0, q_oc)
                ks0 = expand(ks0, k_oc)
                vs0 = expand(vs0, v_oc)
                
                qs1 = expand(qs1, q_oc)
                ks1 = expand(ks1, k_oc)
                vs1 = expand(vs1, v_oc)

                s0 = torch.cat([qs0, ks0, vs0], dim=0)
                
                s1 = None
                if qs1 is not None and ks1 is not None and vs1 is not None:
                    s1 = torch.cat([qs1, ks1, vs1], dim=0)
                return s0, s1

        # Fallback 2: FFN Fusion (net.0.proj -> w1 + w3)
        if ".feed_forward.net.0.proj" in name:
            w1_name = name.replace(".feed_forward.net.0.proj", ".feed_forward.w1")
            w3_name = name.replace(".feed_forward.net.0.proj", ".feed_forward.w3")
            
            w1s0, w1s1 = get_raw(w1_name)
            w3s0, w3s1 = get_raw(w3_name)
            
            if w1s0 is not None:
                if w3s0 is None:
                    # Missing w3 scale, reuse w1
                    w3s0 = w1s0.clone()
                    if w1s1 is not None: w3s1 = w1s1.clone()
                
                # Expand
                w1_oc, w3_oc = 0, 0
                fused_w = orig_state.get(f"{name}.weight")
                if fused_w is not None:
                    total_oc = fused_w.shape[0]
                    w1_oc = w3_oc = total_oc // 2
                
                def expand(s, size):
                    if s is None: return None
                    if s.shape[0] == 1 and size > 1:
                        return s.expand(size, -1, -1, -1).clone()
                    return s
                
                w1s0 = expand(w1s0, w1_oc)
                w3s0 = expand(w3s0, w3_oc)
                w1s1 = expand(w1s1, w1_oc)
                w3s1 = expand(w3s1, w3_oc)

                s0 = torch.cat([w1s0, w3s0], dim=0)
                
                s1 = None
                if w1s1 is not None and w3s1 is not None:
                    s1 = torch.cat([w1s1, w3s1], dim=0)
                return s0, s1

        # Fallback 3: FFN Down (net.2 -> w2)
        if ".feed_forward.net.2" in name:
            w2_name = name.replace(".feed_forward.net.2", ".feed_forward.w2")
            s0, s1 = get_raw(w2_name)
            if s0 is not None:
                return s0, s1

        return None, None
    
    def _get_branch(name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not branch_state:
            return None
        # Try direct match
        b = branch_state.get(name, None)
        # Try aliases if needed (omitted for brevity unless critical)
        if not b or "a.weight" not in b or "b.weight" not in b:
            # Try mapping for LoRA as well if needed
             if ".attention.to_qkv" in name:
                 # Logic for fused LoRA? Complex. 
                 # Assuming LoRA branches are handled or irrelevant for base quantization fix.
                 return None
             return None
        return b["a.weight"], b["b.weight"]
    
    def _get_smooth(name: str) -> torch.Tensor | None:
        def get_raw_smooth(n):
            # Check smooth_cache first
            if smooth_cache and n in smooth_cache:
                return smooth_cache[n]
            # Fallback dequant
            s_key = f"{n}.smooth_factor"
            if s_key in dequant_state:
                return dequant_state[s_key]
            return None

        s = get_raw_smooth(name)
        if s is not None: return s

        # Mappings for Smooth Scale (Input Smoothness)
        # Since input is shared for fused layers, just find ONE valid scale.
        
        # QKV
        if ".attention.to_qkv" in name:
            q_name = name.replace(".attention.to_qkv", ".attention.to_q")
            s = get_raw_smooth(q_name)
            if s is not None: return s

        # FFN net.0.proj (w1)
        if ".feed_forward.net.0.proj" in name:
            w1_name = name.replace(".feed_forward.net.0.proj", ".feed_forward.w1")
            s = get_raw_smooth(w1_name)
            if s is not None: return s
            
        # FFN net.2 (w2)
        if ".feed_forward.net.2" in name:
            w2_name = name.replace(".feed_forward.net.2", ".feed_forward.w2")
            s = get_raw_smooth(w2_name)
            if s is not None: return s
            
        return None
    
    if is_qkv_fused and qkv_modules:
        # Fuse Q, K, V into single to_qkv
        q_name, k_name, v_name = qkv_modules
        q_w = orig_state.get(f"{q_name}.weight")
        k_w = orig_state.get(f"{k_name}.weight")
        v_w = orig_state.get(f"{v_name}.weight")
        
        if q_w is None or k_w is None or v_w is None:
            return
        
        fused_w = torch.cat([q_w, k_w, v_w], dim=0)
        
        # Get dequant weights for residual
        dq_q_w = dequant_state.get(f"{q_name}.weight")
        dq_k_w = dequant_state.get(f"{k_name}.weight")
        dq_v_w = dequant_state.get(f"{v_name}.weight")
        
        if dq_q_w is not None and dq_k_w is not None and dq_v_w is not None:
            dq_fused_w = torch.cat([dq_q_w, dq_k_w, dq_v_w], dim=0)
            residual = (fused_w.to(dtype=torch.float32) - dq_fused_w.to(dtype=torch.float32)).to(dtype=torch.float16)
            u, s, vh = torch.linalg.svd(residual.double())
            b_w = (u[:, :rank] * s[:rank]).to(dtype=torch_dtype, device="cpu")
            a_w = vh[:rank].to(dtype=torch_dtype, device="cpu")
            lora = (a_w, b_w)
        else:
            lora = _get_branch(q_name)
        
        # Fuse scales
        q_s0, q_s1 = _get_scale(q_name)
        k_s0, k_s1 = _get_scale(k_name)
        v_s0, v_s1 = _get_scale(v_name)
        
        if q_s0 is not None and k_s0 is not None and v_s0 is not None:
            if q_s0.numel() == 1:
                fused_s0 = torch.cat([
                    q_s0.view(-1).expand(q_w.shape[0]).reshape(q_w.shape[0], 1, 1, 1),
                    k_s0.view(-1).expand(k_w.shape[0]).reshape(k_w.shape[0], 1, 1, 1),
                    v_s0.view(-1).expand(v_w.shape[0]).reshape(v_w.shape[0], 1, 1, 1),
                ], dim=0)
            else:
                fused_s0 = torch.cat([q_s0, k_s0, v_s0], dim=0)
        else:
            return
        
        fused_s1 = None
        if q_s1 is not None and k_s1 is not None and v_s1 is not None:
            if q_s1.numel() == 1:
                fused_s1 = torch.cat([
                    q_s1.view(-1).expand(q_w.shape[0]).reshape(q_w.shape[0], 1, 1, 1),
                    k_s1.view(-1).expand(k_w.shape[0]).reshape(k_w.shape[0], 1, 1, 1),
                    v_s1.view(-1).expand(v_w.shape[0]).reshape(v_w.shape[0], 1, 1, 1),
                ], dim=0)
            else:
                fused_s1 = torch.cat([q_s1, k_s1, v_s1], dim=0)
        
        smooth = _get_smooth(q_name)
        
        converted = convert_to_nunchaku_w4x4y16_linear_state_dict(
            weight=fused_w.to(dtype=torch_dtype, device="cpu"),
            scale=fused_s0.to(device="cpu"),
            bias=None,
            smooth=smooth.to(device="cpu") if smooth is not None else None,
            lora=lora,
            float_point=float_point,
            subscale=fused_s1.to(device="cpu") if fused_s1 is not None else None,
        )
        
        # Rename to match Nunchaku naming: proj_down, proj_up instead of lora_down, lora_up
        if "lora_down" in converted:
            converted["proj_down"] = converted.pop("lora_down")
        if "lora_up" in converted:
            converted["proj_up"] = converted.pop("lora_up")
        if "smooth" in converted:
            converted["smooth_factor"] = converted.pop("smooth")
        if "smooth_orig" in converted:
            converted["smooth_factor_orig"] = converted.pop("smooth_orig")
        
        # Ensure wcscales for nvfp4
        if float_point and "wcscales" not in converted:
            converted["wcscales"] = torch.ones(fused_w.shape[0], dtype=torch_dtype, device="cpu")
        
        # Write to output
        for kk, vv in converted.items():
            out_state[f"{module_name}.{kk}"] = vv
        
        # Remove original keys
        for name in qkv_modules:
            if f"{name}.weight" in out_state:
                del out_state[f"{name}.weight"]
    else:
        # Single linear layer
        weight_key = f"{module_name}.weight"
        if weight_key not in orig_state:
            return
        
        weight = orig_state[weight_key]
        bias = orig_state.get(f"{module_name}.bias", None)
        
        s0, s1 = _get_scale(module_name)
        if s0 is None:
            return  # Not quantized
        
        branch = _get_branch(module_name)
        smooth = _get_smooth(module_name)
        
        converted = convert_to_nunchaku_w4x4y16_linear_state_dict(
            weight=weight.to(dtype=torch_dtype, device="cpu"),
            scale=s0.to(device="cpu"),
            bias=None, # Force None to match official model structure (no linear bias)
            smooth=smooth.to(device="cpu") if smooth is not None else None,
            lora=branch,
            float_point=float_point,
            subscale=s1.to(device="cpu") if s1 is not None else None,
        )
        
        # Rename wscales to wtscale for non-qkv (Official model convention)
        if "to_qkv" not in module_name and "wscales" in converted:
            converted["wtscale"] = converted.pop("wscales")

        # Rename to match Nunchaku naming
        if "lora_down" in converted:
            converted["proj_down"] = converted.pop("lora_down")
        if "lora_up" in converted:
            converted["proj_up"] = converted.pop("lora_up")
        if "smooth" in converted:
            converted["smooth_factor"] = converted.pop("smooth")
        if "smooth_orig" in converted:
            converted["smooth_factor_orig"] = converted.pop("smooth_orig")
        
        # wcscales handling: official model only has wcscales for to_qkv layers
        if float_point:
            if "to_qkv" in module_name:
                # Ensure wcscales exists for to_qkv
                if "wcscales" not in converted:
                    converted["wcscales"] = torch.ones(weight.shape[0], dtype=torch_dtype, device="cpu")
            else:
                # Remove wcscales for non-qkv layers (to match official structure)
                if "wcscales" in converted:
                    del converted["wcscales"]
        
        # Write to output
        if f"{module_name}.weight" in out_state:
            del out_state[f"{module_name}.weight"]
        if f"{module_name}.bias" in out_state:
            del out_state[f"{module_name}.bias"]
        
        for kk, vv in converted.items():
            out_state[f"{module_name}.{kk}"] = vv


def _zit_export_to_nunchaku_single_safetensors(
    *,
    output_path: str,
    orig_transformer_path: str,
    dequant_state: dict[str, torch.Tensor],
    scale_state: dict[str, torch.Tensor],
    branch_state: dict[str, dict[str, torch.Tensor]] | None,
    smooth_cache: dict[str, torch.Tensor] | None,
    rank: int,
    precision: str,
    torch_dtype: torch.dtype,
    transformer,
) -> None:
    """Export Z-Image Turbo Nunchaku safetensors file."""
    logger = tools.logging.getLogger(__name__)
    
    assert orig_transformer_path, "orig_transformer_path is required"
    assert os.path.exists(orig_transformer_path), f"File not found: {orig_transformer_path}"
    
    # Load original weights (DiffSynth format)
    orig_state_raw = _load_safetensors_state_dict(orig_transformer_path)
    
    # Convert to Diffusers format to match quantized model keys
    from .pipeline.zit import convert_diffsynth_to_diffusers
    orig_state = convert_diffsynth_to_diffusers(orig_state_raw)
    out_state: dict[str, torch.Tensor] = dict(orig_state)
    
    float_point = precision == "nvfp4"
    
    # Z-Image Turbo structure (from analysis):
    # - layers.0-29: main transformer layers
    # - context_refiner.0-1: context refiner layers
    # - noise_refiner.0-1: noise refiner layers
    # Each layer has:
    #   - attention.to_q/to_k/to_v -> fused to attention.to_qkv
    #   - attention.to_out.0
    #   - feed_forward.net.0.proj
    #   - feed_forward.net.2
    
    # Collect layer prefixes
    main_layers = [f"layers.{i}" for i in range(30)]
    context_refiner_layers = [f"context_refiner.{i}" for i in range(2)]
    noise_refiner_layers = [f"noise_refiner.{i}" for i in range(2)]
    all_layers = main_layers + context_refiner_layers + noise_refiner_layers
    
    logger.info(f"* Exporting Nunchaku Z-Image Turbo: processing {len(all_layers)} layers")
    
    for layer_prefix in all_layers:
        # QKV: attention.to_qkv (already fused from DiffSynth conversion)
        qkv_name = f"{layer_prefix}.attention.to_qkv"
        
        _process_zit_linear(
            module_name=qkv_name,
            orig_state=orig_state,
            dequant_state=dequant_state,
            scale_state=scale_state,
            branch_state=branch_state,
            smooth_cache=smooth_cache,
            out_state=out_state,
            config=transformer.config,
            rank=rank,
            torch_dtype=torch_dtype,
            float_point=float_point,
            is_qkv_fused=False,  # Already fused, process as single linear
            qkv_modules=None,
        )
        
        # attention.to_out.0
        _process_zit_linear(
            module_name=f"{layer_prefix}.attention.to_out.0",
            orig_state=orig_state,
            dequant_state=dequant_state,
            scale_state=scale_state,
            branch_state=branch_state,
            smooth_cache=smooth_cache,
            out_state=out_state,
            config=transformer.config,
            rank=rank,
            torch_dtype=torch_dtype,
            float_point=float_point,
        )
        
        # feed_forward.net.0.proj
        _process_zit_linear(
            module_name=f"{layer_prefix}.feed_forward.net.0.proj",
            orig_state=orig_state,
            dequant_state=dequant_state,
            scale_state=scale_state,
            branch_state=branch_state,
            smooth_cache=smooth_cache,
            out_state=out_state,
            config=transformer.config,
            rank=rank,
            torch_dtype=torch_dtype,
            float_point=float_point,
        )
        
        # feed_forward.net.2
        _process_zit_linear(
            module_name=f"{layer_prefix}.feed_forward.net.2",
            orig_state=orig_state,
            dequant_state=dequant_state,
            scale_state=scale_state,
            branch_state=branch_state,
            smooth_cache=smooth_cache,
            out_state=out_state,
            config=transformer.config,
            rank=rank,
            torch_dtype=torch_dtype,
            float_point=float_point,
        )
    
    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    metadata = _zit_export_build_metadata(transformer=transformer, rank=rank, precision=precision)
    logger.info(f"* Saving Nunchaku Z-Image Turbo to {output_path}")
    safetensors.torch.save_file(out_state, output_path, metadata=metadata)


def ptq(
    model: DiffusionModelStruct,
    config: DiffusionQuantConfig,
    cache: DiffusionPtqCacheConfig | None = None,
    load_dirpath: str = "",
    save_dirpath: str = "",
    copy_on_save: bool = False,
    save_model: bool = False,
    export_nunchaku_zit: dict | None = None,
) -> DiffusionModelStruct:
    """Post-training quantization of Z-Image Turbo model."""
    logger = tools.logging.getLogger(__name__)
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)

    quant_wgts = config.enabled_wgts
    quant_ipts = config.enabled_ipts
    quant_opts = config.enabled_opts
    quant_acts = quant_ipts or quant_opts
    quant = quant_wgts or quant_acts

    load_model_path, load_path, save_path = "", None, None
    if load_dirpath:
        load_path = DiffusionQuantCacheConfig(
            smooth=os.path.join(load_dirpath, "smooth.pt"),
            branch=os.path.join(load_dirpath, "branch.pt"),
            wgts=os.path.join(load_dirpath, "wgts.pt"),
            acts=os.path.join(load_dirpath, "acts.pt"),
        )
        load_model_path = os.path.join(load_dirpath, "model.pt")
        if os.path.exists(load_model_path):
            if config.enabled_wgts and config.wgts.enabled_low_rank:
                load_model = os.path.exists(load_path.branch)
            else:
                load_model = True
            if load_model:
                logger.info(f"* Loading model from {load_model_path}")
                save_dirpath = ""
        else:
            load_model = False
    else:
        load_model = False
    
    if save_dirpath:
        os.makedirs(save_dirpath, exist_ok=True)
        save_path = DiffusionQuantCacheConfig(
            smooth=os.path.join(save_dirpath, "smooth.pt"),
            branch=os.path.join(save_dirpath, "branch.pt"),
            wgts=os.path.join(save_dirpath, "wgts.pt"),
            acts=os.path.join(save_dirpath, "acts.pt"),
        )
    else:
        save_model = False

    # Smooth quantization
    smooth_cache = {}  # Initialize for later use in export
    if quant and config.enabled_smooth:
        logger.info("* Smoothing model for quantization")
        tools.logging.Formatter.indent_inc()
        load_from = ""
        if load_path and os.path.exists(load_path.smooth):
            load_from = load_path.smooth
        elif cache and cache.path.smooth and os.path.exists(cache.path.smooth):
            load_from = cache.path.smooth
        if load_from:
            logger.info(f"- Loading smooth scales from {load_from}")
            smooth_cache = torch.load(load_from)
            smooth_diffusion(model, config, smooth_cache=smooth_cache)
        else:
            logger.info("- Generating smooth scales")
            smooth_cache = smooth_diffusion(model, config)
            if cache and cache.path.smooth:
                logger.info(f"- Saving smooth scales to {cache.path.smooth}")
                os.makedirs(cache.dirpath.smooth, exist_ok=True)
                torch.save(smooth_cache, cache.path.smooth)
        # NOTE: Do NOT delete smooth_cache here - needed for export
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()

    # Weight quantization
    if load_model:
        logger.info(f"* Loading model checkpoint from {load_model_path}")
        load_diffusion_weights_state_dict(
            model,
            config,
            state_dict=torch.load(load_model_path),
            branch_state_dict=torch.load(load_path.branch) if os.path.exists(load_path.branch) else None,
        )
        gc.collect()
        torch.cuda.empty_cache()
    elif quant_wgts:
        logger.info("* Quantizing weights")
        tools.logging.Formatter.indent_inc()
        quantizer_state_dict, quantizer_load_from = None, ""
        if load_path and os.path.exists(load_path.wgts):
            quantizer_load_from = load_path.wgts
        elif cache and cache.path.wgts and os.path.exists(cache.path.wgts):
            quantizer_load_from = cache.path.wgts
        if quantizer_load_from:
            logger.info(f"- Loading weight settings from {quantizer_load_from}")
            quantizer_state_dict = torch.load(quantizer_load_from)
        branch_state_dict, branch_load_from = None, ""
        if load_path and os.path.exists(load_path.branch):
            branch_load_from = load_path.branch
        elif cache and cache.path.branch and os.path.exists(cache.path.branch):
            branch_load_from = cache.path.branch
        if branch_load_from:
            logger.info(f"- Loading branch settings from {branch_load_from}")
            branch_state_dict = torch.load(branch_load_from)
        
        quantizer_state_dict, branch_state_dict, scale_state_dict = quantize_diffusion_weights(
            model,
            config,
            quantizer_state_dict=quantizer_state_dict,
            branch_state_dict=branch_state_dict,
            return_with_scale_state_dict=bool(save_dirpath) or bool(save_model) or bool(export_nunchaku_zit),
        )
        
        if not quantizer_load_from and cache and cache.dirpath.wgts:
            logger.info(f"- Saving weight settings to {cache.path.wgts}")
            os.makedirs(cache.dirpath.wgts, exist_ok=True)
            torch.save(quantizer_state_dict, cache.path.wgts)
        if not branch_load_from and cache and cache.dirpath.branch:
            logger.info(f"- Saving branch settings to {cache.path.branch}")
            os.makedirs(cache.dirpath.branch, exist_ok=True)
            torch.save(branch_state_dict, cache.path.branch)
        
        if export_nunchaku_zit:
            export_path = export_nunchaku_zit["output_path"]
            orig_path = export_nunchaku_zit["orig_transformer_path"]
            transformer = export_nunchaku_zit["transformer"]
            rank_val = int(export_nunchaku_zit["rank"])
            precision_val = export_nunchaku_zit["precision"]
            dtype_val = export_nunchaku_zit["torch_dtype"]
            
            dequant_state = {k: v.detach().cpu() for k, v in transformer.state_dict().items()}
            _zit_export_to_nunchaku_single_safetensors(
                output_path=export_path,
                orig_transformer_path=orig_path,
                dequant_state=dequant_state,
                scale_state=scale_state_dict,
                branch_state=branch_state_dict,
                smooth_cache=smooth_cache,
                rank=rank_val,
                precision=precision_val,
                torch_dtype=dtype_val,
                transformer=transformer,
            )
        
        del quantizer_state_dict, branch_state_dict, scale_state_dict
        tools.logging.Formatter.indent_dec()
        gc.collect()
        torch.cuda.empty_cache()
    
    return model


def main(config: DiffusionPtqRunConfig, logging_level: int = tools.logging.DEBUG) -> DiffusionPipeline:
    """Post-training quantization of Z-Image Turbo model."""
    config.output.lock()
    config.dump(path=config.output.get_running_job_path("config.yaml"))
    tools.logging.setup(path=config.output.get_running_job_path("run.log"), level=logging_level)
    logger = tools.logging.getLogger(__name__)

    logger.info("=== Z-Image Turbo Quantization ===")
    tools.logging.info(config.formatted_str(), logger=logger)
    logger.info("=== Output Directory ===")
    logger.info(config.output.job_dirpath)

    logger.info("=== Building Pipeline ===")
    tools.logging.Formatter.indent_inc()
    pipeline = config.pipeline.build()
    # NOTE: For ZIT models, transformer loading is already handled by build_zit_pipeline
    # with proper DiffSynth->Diffusers key conversion. Do NOT reload here as it would
    # overwrite with unconverted keys and cause shape mismatch errors.
    # The code below is only for non-ZIT models that need manual transformer loading.
    
    model = DiffusionModelStruct.construct(pipeline)
    tools.logging.Formatter.indent_dec()
    
    save_dirpath = ""
    if hasattr(config, "export_nunchaku_zit") and config.export_nunchaku_zit:
        save_dirpath = ""
        save_model = False
    elif hasattr(config, "save_model") and config.save_model:
        save_dirpath = os.path.join(config.output.running_job_dirpath, "model")
        save_model = True
    else:
        save_model = False
    
    export_ctx = None
    if hasattr(config, "export_nunchaku_zit") and config.export_nunchaku_zit:
        if not hasattr(config.pipeline, "transformer_path") or not config.pipeline.transformer_path:
            raise ValueError("export_nunchaku_zit requires pipeline.transformer_path")
        if not config.quant.enabled_wgts:
            raise ValueError("export_nunchaku_zit requires weight quantization enabled")
        
        transformer = getattr(pipeline, "transformer", None)
        export_ctx = {
            "output_path": config.export_nunchaku_zit,
            "orig_transformer_path": config.pipeline.transformer_path,
            "transformer": transformer,
            "rank": int(config.quant.wgts.low_rank.rank) if config.quant.wgts.low_rank else 128,
            "precision": _detect_precision(config.quant.wgts.dtype),
            "torch_dtype": config.pipeline.dtype,
        }
    
    model = ptq(
        model,
        config.quant,
        cache=config.cache,
        load_dirpath=getattr(config, "load_from", ""),
        save_dirpath=save_dirpath,
        copy_on_save=getattr(config, "copy_on_save", False),
        save_model=save_model,
        export_nunchaku_zit=export_ctx,
    )
    
    tools.logging.shutdown()
    try:
        config.output.unlock()
    except Exception:
        pass
    
    return pipeline


if __name__ == "__main__":
    config, _, unused_cfgs, unused_args, unknown_args = DiffusionPtqRunConfig.get_parser().parse_known_args()
    assert isinstance(config, DiffusionPtqRunConfig)
    try:
        main(config, logging_level=tools.logging.DEBUG)
    except Exception as e:
        tools.logging.Formatter.indent_reset()
        tools.logging.error("=== Error ===")
        tools.logging.error(traceback.format_exc())
        tools.logging.shutdown()
        traceback.print_exc()
        config.output.unlock(error=True)
        raise e
