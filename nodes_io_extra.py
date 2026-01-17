"""
FXTD Radiance - Professional IO Extra
Additional I/O tools for HDRI, Channel Merging, and Bit Depth conversion.
"""

import torch
import numpy as np
import os
import struct

# Shared utils similar to nodes_exr
def to_numpy(image):
    return image.cpu().numpy()

class RadianceSaveHDRI:
    """
    Save image as .hdr (Radiance RGBE) format.
    Uses imageio or cv2.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Radiance_HDRI"}),
            },
            "optional": {
                "subfolder": ("STRING", {"default": "hdri"}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_hdri"
    CATEGORY = "FXTD Studios/Radiance/IO"

    def save_hdri(self, images, filename_prefix, subfolder):
        import cv2
        
        output_dir = os.path.join("output", subfolder)
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, image in enumerate(images):
            # Convert to numpy float32 BGR for OpenCV
            img_np = to_numpy(image)
            img_bgr = img_np[..., ::-1].copy() # RGB to BGR
            
            filename = f"{filename_prefix}_{i:04d}.hdr"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, img_bgr)
            results.append(filepath)
            
        return {"ui": {"images": results}}

class RadianceEXRChannelMerge:
    """
    Merge multiple separate images into a single multi-channel output tensor (for saving).
    This doesn't save *files*, it prepares the channel dictionary for the saver?
    Actually user request implies a node that combines them. The existing `FXTDSaveEXRMultiLayer` handles this logic directly in the saver.
    
    This node might be intended to create a 'pipe' of channels or a multi-tensor.
    For now, I will implement it as a pass-through that organizes them, or simply checks dimensions.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "R": ("IMAGE",),
                "G": ("IMAGE",),
                "B": ("IMAGE",),
                "A": ("IMAGE",),
                "Depth": ("IMAGE",),
                # Add unlimited inputs? ComfyUI dynamic inputs usually better.
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK") # Returns combined RGBA
    FUNCTION = "merge"
    CATEGORY = "FXTD Studios/Radiance/IO"
    
    def merge(self, **kwargs):
        # Extract inputs
        r_img = kwargs.get("R")
        g_img = kwargs.get("G")
        b_img = kwargs.get("B")
        a_img = kwargs.get("A")
        
        # Get dimensions from the first available input
        ref_img = next((img for img in [r_img, g_img, b_img, a_img] if img is not None), None)
        
        if ref_img is None:
            # No inputs provided, return generic black 512x512
            empty = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            mask = torch.ones((1, 512, 512), dtype=torch.float32)
            return (empty, mask)
        
        batch, h, w, _ = ref_img.shape
        device = ref_img.device
        
        # Helpers to extract single channel
        def get_channel(img, channel_idx=0):
            if img is None:
                return torch.zeros((batch, h, w), dtype=torch.float32, device=device)
            
            # Use first channel if multi-channel, or 0 if single
            if img.shape[-1] > channel_idx:
                return img[..., channel_idx]
            return img[..., 0]

        r = get_channel(r_img, 0)
        g = get_channel(g_img, 1) if g_img is not None else torch.zeros_like(r)
        b = get_channel(b_img, 2) if b_img is not None else torch.zeros_like(r)
        a = get_channel(a_img, 0) if a_img is not None else torch.ones_like(r)
        
        # Stack RGB
        merged_rgb = torch.stack([r, g, b], dim=-1)
        
        # A is mask
        mask = a
        
        # Handle Depth if needed for future (currently unused in return types)
        # depth = get_channel(kwargs.get("Depth"))
        
        return (merged_rgb, mask)


class RadianceBitDepthConvert:
    """
    Explict conversion between float32 (0-1) and int16/uint16/uint8 ranges.
    ComfyUI images are always float32 0-1. This is mostly for 'simulation' or preparation.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_depth": (["8-bit (0-255)", "10-bit (0-1023)", "12-bit (0-4095)", "16-bit (0-65535)"],),
                "mode": (["Scale", "Clip"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "FXTD Studios/Radiance/IO"
    
    def convert(self, image, target_depth, mode):
        # Simulates quantization
        max_val = 255.0
        if "10-bit" in target_depth: max_val = 1023.0
        elif "12-bit" in target_depth: max_val = 4095.0
        elif "16-bit" in target_depth: max_val = 65535.0
        
        # Scale 0-1 to 0-max_val
        scaled = image * max_val
        quantized = torch.round(scaled)
        
        # Convert back to 0-1 for ComfyUI pipe
        return (quantized / max_val,)

NODE_CLASS_MAPPINGS = {
    "RadianceSaveHDRI": RadianceSaveHDRI,
    "RadianceBitDepthConvert": RadianceBitDepthConvert
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceSaveHDRI": "Radiance Save HDRI",
    "RadianceBitDepthConvert": "Radiance Bit Depth Convert"
}
