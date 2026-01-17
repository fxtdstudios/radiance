"""
═══════════════════════════════════════════════════════════════════════════════
                    FXTD RADIANCE GRADE NODE
              Professional Color Grading for ComfyUI
                     FXTD Studios © 2024
═══════════════════════════════════════════════════════════════════════════════

Professional color grading node providing industry-standard controls:
- Lift (Shadows)
- Gamma (Midtones)
- Gain (Highlights)
- Offset (Global)
- Contrast (S-Curve)
- Saturation (Luminance preserving)

Uses 32-bit floating point precision for highest quality.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any

class FXTD_Grade:
    """
    Professional Color Grading Node
    
    Implements standard ASC CDL-like grading controls plus Contrast and Saturation.
    Operations are performed in 32-bit float space.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "lift_r": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "lift_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "lift_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "gamma_r": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001}),
                "gamma_g": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001}),
                "gamma_b": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001}),
                "gain_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.001}),
                "gain_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.001}),
                "gain_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.001}),
                "offset_r": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "offset_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "offset_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "pivot": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "grade"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Professional color grading with per-channel Lift/Gamma/Gain/Offset, plus Contrast and Saturation."
    
    def grade(self, image: torch.Tensor, 
              lift_r: float = 0.0, lift_g: float = 0.0, lift_b: float = 0.0,
              gamma_r: float = 1.0, gamma_g: float = 1.0, gamma_b: float = 1.0,
              gain_r: float = 1.0, gain_g: float = 1.0, gain_b: float = 1.0,
              offset_r: float = 0.0, offset_g: float = 0.0, offset_b: float = 0.0,
              contrast: float = 1.0, pivot: float = 0.5, saturation: float = 1.0) -> Tuple[torch.Tensor]:
        
        # Ensure input is float32
        img = image.float().clone()
        
        # 1. Lift (Shadows) - Additive in darks
        # Simplified ASC CDL slope/offset/power model treats Lift differently usually, 
        # but standardized "Lift" in Nuke/DaVinci is often (x * (gain - lift) + lift) or similar.
        # Here we implement standard Lift/Gamma/Gain formula:
        # Out = pow( (In * Gain + Lift) * (1-Lightness) + Lightness , 1/Gamma ) + Offset  <-- varying definitions
        
        # Let's use the most common ASC CDL + Lift extension logic:
        # Slope (Gain), Offset (Lift/Offset combined), Power (Gamma)
        # But users expect "Lift" to raise black point.
        
        # Lift: Black point adjustment -> x * (1 + lift) + lift ? No.
        # Standard simpler model:
        # Lift is offset, Gain is slope, Gamma is power.
        # But traditionally Lift is "Offset" but biased towards blacks?
        # Let's use standard DaVinci math approximation or equivalent:
        # out = (in + lift) * gain
        # then gamma.
        
        # Apply LIFT (Offsetting blacks)
        if lift_r != 0 or lift_g != 0 or lift_b != 0:
            img[..., 0] += lift_r
            img[..., 1] += lift_g
            img[..., 2] += lift_b
        
        # Apply GAIN (Slope)
        if gain_r != 1.0 or gain_g != 1.0 or gain_b != 1.0:
            img[..., 0] *= gain_r
            img[..., 1] *= gain_g
            img[..., 2] *= gain_b
            
        # Apply OFFSET (Overall brightness shift)
        if offset_r != 0 or offset_g != 0 or offset_b != 0:
            img[..., 0] += offset_r
            img[..., 1] += offset_g
            img[..., 2] += offset_b
            
        # Apply GAMMA (Power)
        # Avoid negative numbers for power power
        if gamma_r != 1.0 or gamma_g != 1.0 or gamma_b != 1.0:
            epsilon = 1e-8
            img = torch.max(img, torch.tensor(0.0, device=img.device)) # Clip negatives for gamma safety
            
            if gamma_r != 1.0:
                img[..., 0] = torch.pow(img[..., 0] + epsilon, 1.0 / gamma_r)
            if gamma_g != 1.0:
                img[..., 1] = torch.pow(img[..., 1] + epsilon, 1.0 / gamma_g)
            if gamma_b != 1.0:
                img[..., 2] = torch.pow(img[..., 2] + epsilon, 1.0 / gamma_b)
                
        # Apply CONTRAST
        if contrast != 1.0:
            img = (img - pivot) * contrast + pivot
            
        # Apply SATURATION
        if saturation != 1.0 and img.shape[-1] >= 3:
            luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            luma = luma.unsqueeze(-1)
            img = luma + saturation * (img - luma)
            
        return (img,)

NODE_CLASS_MAPPINGS = {
    "FXTD_Grade": FXTD_Grade
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTD_Grade": "🎨 Radiance Grade"
}
