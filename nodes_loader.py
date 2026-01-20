"""
FXTD STUDIO RADIANCE - LOADER NODES
-----------------------------------
Professional unified loader for Diffusion models, Dual CLIPs, and VAEs.
"""

import os
import torch
import folder_paths
import comfy.sd
import comfy.utils

class FXTDRadianceLoader:
    """
    Radiance Unified Loader
    
    A professional 10/10 quality loader that combines:
    - Diffusion Model Loading (UNET) with weight type control
    - Dual CLIP Loading (e.g., T5 + CLIP-L for Flux)
    - VAE Loading
    
    All in a single, robust node with comprehensive tooltips and error handling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ═══════════════════════════════════════════════════════════
                # DIFFUSION MODEL (UNET)
                # ═══════════════════════════════════════════════════════════
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Select the main diffusion model (UNET/Transformer). e.g. Flux, SD3, SDXL."
                }),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "fp32"], {
                    "default": "default",
                    "tooltip": "Precision for loading the UNET weights. 'fp8_e4m3fn' is recommended for Flux on <24GB VRAM cards."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # DUAL CLIP LOADER
                # ═══════════════════════════════════════════════════════════
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Primary Text Encoder (e.g., t5xxl_fp16.safetensors for Flux)."
                }),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Secondary Text Encoder (e.g., clip_l.safetensors for Flux/SDXL)."
                }),
                "type": (["sdxl", "sd3", "flux", "sd3.5"], {
                    "default": "flux",
                    "tooltip": "The model architecture type. Critical for correct CLIP tokenizer visualization and encoding."
                }),
                "device": (["default", "cpu"], {
                    "default": "default",
                    "tooltip": "Execution device. 'default' uses GPU. Use 'cpu' for massive T5 models if system RAM is plentiful but VRAM is tight."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # VAE LOADER
                # ═══════════════════════════════════════════════════════════
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "Variational Autoencoder (VAE) for decoding latents to images."
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_radiance_stack"
    CATEGORY = "FXTD Studios/Radiance/Loaders"
    DESCRIPTION = "Unified Professional Loader for Flux and modern Diffusion models (UNET + Dual CLIP + VAE)."

    def load_radiance_stack(self, unet_name, weight_dtype, clip_name1, clip_name2, type, device, vae_name):
        # 1. LOAD DIFFUSION MODEL (UNET)
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model_options = {}
        
        # Handle Weight Dtype (FP8 support)
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        elif weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "fp32":
            model_options["dtype"] = torch.float32
            
        try:
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load Diffusion Model '{unet_name}': {str(e)}")

        # 2. LOAD DUAL CLIP
        clip_path1 = folder_paths.get_full_path("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path("text_encoders", clip_name2)
        
        # Parse CLIP type
        clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        if type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "sd3.5":
            clip_type = comfy.sd.CLIPType.SD3
        
        # Handle Device
        clip_load_device = "cpu" if device == "cpu" else None
        
        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options={"load_device": clip_load_device} if clip_load_device else {}
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load CLIPs '{clip_name1}' & '{clip_name2}': {str(e)}")

        # 3. LOAD VAE
        vae_path = folder_paths.get_full_path("vae", vae_name)
        try:
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load VAE '{vae_name}': {str(e)}")
            
        return (model, clip, vae)

# Node Mappings
NODE_CLASS_MAPPINGS = {
    "FXTDRadianceLoader": FXTDRadianceLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTDRadianceLoader": "◆ Radiance Unified Loader"
}
