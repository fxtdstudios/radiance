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


# =============================================================================
# GRADE PRESETS - Common cinematic looks
# =============================================================================

GRADE_PRESETS = {
    "None (Custom)": {
        "description": "No preset - use manual controls",
        "lift": (0.0, 0.0, 0.0),
        "gamma": (1.0, 1.0, 1.0),
        "gain": (1.0, 1.0, 1.0),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.0,
        "saturation": 1.0,
    },
    "Cinematic Teal & Orange": {
        "description": "Popular blockbuster look with teal shadows and orange highlights",
        "lift": (0.0, 0.02, 0.05),  # Blue/teal shadows
        "gamma": (1.0, 0.98, 0.95),
        "gain": (1.05, 0.98, 0.90),  # Orange highlights
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.15,
        "saturation": 1.1,
    },
    "Bleach Bypass": {
        "description": "Desaturated, high contrast - popular in war and thriller films",
        "lift": (0.02, 0.02, 0.02),
        "gamma": (0.95, 0.95, 0.95),
        "gain": (1.1, 1.1, 1.1),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.3,
        "saturation": 0.5,
    },
    "Cross Process": {
        "description": "Film cross-processing look with color shifts",
        "lift": (0.05, -0.02, 0.08),
        "gamma": (0.95, 1.05, 0.9),
        "gain": (1.1, 0.95, 1.15),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.2,
        "saturation": 1.2,
    },
    "Film Noir": {
        "description": "High contrast black and white with deep shadows",
        "lift": (-0.02, -0.02, -0.02),
        "gamma": (0.9, 0.9, 0.9),
        "gain": (1.2, 1.2, 1.2),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.4,
        "saturation": 0.0,
    },
    "Vintage Film": {
        "description": "Warm, faded look reminiscent of aged film prints",
        "lift": (0.03, 0.02, 0.0),
        "gamma": (1.05, 1.0, 0.92),
        "gain": (1.0, 0.98, 0.88),
        "offset": (0.02, 0.01, -0.02),
        "contrast": 0.9,
        "saturation": 0.85,
    },
    "Cool Blue Hour": {
        "description": "Blue-tinted look for dusk/dawn scenes",
        "lift": (0.0, 0.01, 0.04),
        "gamma": (0.98, 1.0, 1.05),
        "gain": (0.95, 1.0, 1.1),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.05,
        "saturation": 0.9,
    },
    "Golden Hour": {
        "description": "Warm golden tones for sunset scenes",
        "lift": (0.02, 0.01, -0.02),
        "gamma": (0.98, 1.0, 1.05),
        "gain": (1.1, 1.02, 0.88),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.1,
        "saturation": 1.15,
    },
    "Matrix Green": {
        "description": "Green-tinted cyberpunk look",
        "lift": (0.0, 0.02, 0.0),
        "gamma": (0.95, 1.05, 0.95),
        "gain": (0.9, 1.1, 0.9),
        "offset": (0.0, 0.01, 0.0),
        "contrast": 1.2,
        "saturation": 0.8,
    },
    "High Key Bright": {
        "description": "Bright, airy look for fashion and beauty",
        "lift": (0.03, 0.03, 0.03),
        "gamma": (0.9, 0.9, 0.9),
        "gain": (1.0, 1.0, 1.0),
        "offset": (0.05, 0.05, 0.05),
        "contrast": 0.85,
        "saturation": 0.95,
    },
    "Low Key Moody": {
        "description": "Dark, moody look with crushed blacks",
        "lift": (-0.03, -0.03, -0.03),
        "gamma": (1.1, 1.1, 1.1),
        "gain": (0.95, 0.95, 0.95),
        "offset": (-0.02, -0.02, -0.02),
        "contrast": 1.25,
        "saturation": 0.9,
    },
    "Sci-Fi Cold": {
        "description": "Cold, sterile look for sci-fi environments",
        "lift": (0.0, 0.01, 0.03),
        "gamma": (1.0, 1.0, 0.98),
        "gain": (0.95, 0.98, 1.05),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.15,
        "saturation": 0.75,
    },
    "Horror Desaturated": {
        "description": "Desaturated with green tint for horror",
        "lift": (0.0, 0.02, 0.0),
        "gamma": (1.0, 1.02, 1.0),
        "gain": (0.95, 1.0, 0.95),
        "offset": (0.0, 0.0, 0.0),
        "contrast": 1.3,
        "saturation": 0.4,
    },
}


class FXTD_Grade:
    """
    Professional Color Grading Node
    
    Implements standard ASC CDL-like grading controls plus Contrast and Saturation.
    Operations are performed in 32-bit float space.
    
    Includes professional presets for common cinematic looks.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        preset_names = list(GRADE_PRESETS.keys())
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to grade. Processed in 32-bit float precision for maximum quality."}),
                "preset": (preset_names, {
                    "default": "None (Custom)",
                    "tooltip": "Load a preset look. Select 'None (Custom)' for manual control. Presets set initial values which you can then fine-tune."
                }),
                "preset_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and preset (1). Use to dial back preset intensity."
                }),
            },
            "optional": {
                # Lift (Shadows)
                "lift_r": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Red channel lift (shadow offset). Shifts black point. Positive = brighter shadows, Negative = darker."
                }),
                "lift_g": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Green channel lift (shadow offset)."
                }),
                "lift_b": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Blue channel lift (shadow offset)."
                }),
                
                # Gamma (Midtones)
                "gamma_r": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001,
                    "tooltip": "Red channel gamma (midtone power). <1 = brighter midtones, >1 = darker midtones."
                }),
                "gamma_g": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001,
                    "tooltip": "Green channel gamma (midtone power)."
                }),
                "gamma_b": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 5.0, "step": 0.001,
                    "tooltip": "Blue channel gamma (midtone power)."
                }),
                
                # Gain (Highlights)
                "gain_r": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.001,
                    "tooltip": "Red channel gain (highlight multiplier). Scales entire channel. Most visible in highlights."
                }),
                "gain_g": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.001,
                    "tooltip": "Green channel gain (highlight multiplier)."
                }),
                "gain_b": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.001,
                    "tooltip": "Blue channel gain (highlight multiplier)."
                }),
                
                # Offset (Global)
                "offset_r": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Red channel offset (global shift). Applied after all other operations. Use for overall color cast."
                }),
                "offset_g": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Green channel offset (global shift)."
                }),
                "offset_b": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Blue channel offset (global shift)."
                }),
                
                # Contrast and Saturation
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Contrast multiplier around pivot point. >1 = more contrast, <1 = flatter image."
                }),
                "pivot": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Contrast pivot point. Values at this level stay fixed while others push away/toward it."
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Color saturation. 0 = grayscale, 1 = original, >1 = more vibrant. Uses Rec.709 luminance-preserving method."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "grade_info")
    OUTPUT_TOOLTIPS = ("Graded image in 32-bit float.", "Information about applied grade settings.")
    FUNCTION = "grade"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Professional color grading with per-channel Lift/Gamma/Gain/Offset, Contrast, Saturation, and cinematic presets."
    
    def grade(self, image: torch.Tensor, preset: str = "None (Custom)",
              preset_strength: float = 1.0,
              lift_r: float = 0.0, lift_g: float = 0.0, lift_b: float = 0.0,
              gamma_r: float = 1.0, gamma_g: float = 1.0, gamma_b: float = 1.0,
              gain_r: float = 1.0, gain_g: float = 1.0, gain_b: float = 1.0,
              offset_r: float = 0.0, offset_g: float = 0.0, offset_b: float = 0.0,
              contrast: float = 1.0, pivot: float = 0.5, 
              saturation: float = 1.0) -> Tuple[torch.Tensor, str]:
        
        # Apply preset if selected
        if preset != "None (Custom)" and preset in GRADE_PRESETS and preset_strength > 0:
            p = GRADE_PRESETS[preset]
            s = preset_strength
            
            # Blend preset values with manual values
            lift_r = lift_r * (1-s) + p["lift"][0] * s
            lift_g = lift_g * (1-s) + p["lift"][1] * s
            lift_b = lift_b * (1-s) + p["lift"][2] * s
            
            gamma_r = gamma_r * (1-s) + p["gamma"][0] * s
            gamma_g = gamma_g * (1-s) + p["gamma"][1] * s
            gamma_b = gamma_b * (1-s) + p["gamma"][2] * s
            
            gain_r = gain_r * (1-s) + p["gain"][0] * s
            gain_g = gain_g * (1-s) + p["gain"][1] * s
            gain_b = gain_b * (1-s) + p["gain"][2] * s
            
            offset_r = offset_r * (1-s) + p["offset"][0] * s
            offset_g = offset_g * (1-s) + p["offset"][1] * s
            offset_b = offset_b * (1-s) + p["offset"][2] * s
            
            contrast = contrast * (1-s) + p["contrast"] * s
            saturation = saturation * (1-s) + p["saturation"] * s
        
        # Ensure input is float32
        img = image.float().clone()
        
        # 1. Apply LIFT (Offsetting blacks)
        if lift_r != 0 or lift_g != 0 or lift_b != 0:
            img[..., 0] += lift_r
            img[..., 1] += lift_g
            img[..., 2] += lift_b
        
        # 2. Apply GAIN (Slope)
        if gain_r != 1.0 or gain_g != 1.0 or gain_b != 1.0:
            img[..., 0] *= gain_r
            img[..., 1] *= gain_g
            img[..., 2] *= gain_b
            
        # 3. Apply OFFSET (Overall brightness shift)
        if offset_r != 0 or offset_g != 0 or offset_b != 0:
            img[..., 0] += offset_r
            img[..., 1] += offset_g
            img[..., 2] += offset_b
            
        # 4. Apply GAMMA (Power)
        if gamma_r != 1.0 or gamma_g != 1.0 or gamma_b != 1.0:
            epsilon = 1e-8
            img = torch.max(img, torch.tensor(0.0, device=img.device))
            
            if gamma_r != 1.0:
                img[..., 0] = torch.pow(img[..., 0] + epsilon, 1.0 / gamma_r)
            if gamma_g != 1.0:
                img[..., 1] = torch.pow(img[..., 1] + epsilon, 1.0 / gamma_g)
            if gamma_b != 1.0:
                img[..., 2] = torch.pow(img[..., 2] + epsilon, 1.0 / gamma_b)
                
        # 5. Apply CONTRAST
        if contrast != 1.0:
            img = (img - pivot) * contrast + pivot
            
        # 6. Apply SATURATION
        if saturation != 1.0 and img.shape[-1] >= 3:
            luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            luma = luma.unsqueeze(-1)
            img = luma + saturation * (img - luma)
        
        # Build info string
        if preset != "None (Custom)":
            preset_info = GRADE_PRESETS.get(preset, {}).get("description", "")
            info = f"Preset: {preset} ({preset_strength*100:.0f}%)\n{preset_info}"
        else:
            info = f"Custom Grade | Contrast: {contrast:.2f} | Saturation: {saturation:.2f}"
            
        return (img, info)


NODE_CLASS_MAPPINGS = {
    "FXTD_Grade": FXTD_Grade
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTD_Grade": "◆ Radiance Grade"
}
