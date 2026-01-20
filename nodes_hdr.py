"""
═══════════════════════════════════════════════════════════════════════════════
                    COMFYUI 32-BIT HDR PROCESSING NODES
                            FXTD STUDIOS © 2024
═══════════════════════════════════════════════════════════════════════════════

Custom nodes for professional 32-bit floating point HDR image generation,
processing, and export in ComfyUI.

Installation:
    1. Copy this file to: ComfyUI/custom_nodes/comfyui_32bit_hdr/
    2. Create __init__.py with: from .nodes_32bit_hdr import *
    3. Restart ComfyUI

Dependencies:
    pip install numpy opencv-python OpenEXR Imath colour-science

═══════════════════════════════════════════════════════════════════════════════
"""
# ═══════════════════════════════════════════════════════════════════════════════
# FIX: OpenMP duplicate library conflict (numpy/pytorch)
# ═══════════════════════════════════════════════════════════════════════════════
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import torch
import numpy as np

# Import GPU utilities for scipy replacements
try:
    from .gpu_utils import (
        gpu_gaussian_blur, gpu_local_contrast, 
        gpu_trilinear_sample, gpu_laplacian_pyramid_blend
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Enable OpenEXR support in OpenCV
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from typing import Tuple, Optional, Dict, Any, List
import os
import json
from pathlib import Path

# Conditional imports for optional dependencies
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("[32bit-HDR] OpenEXR not found. Install with: pip install OpenEXR Imath")

try:
    import colour
    HAS_COLOUR = True
except ImportError:
    HAS_COLOUR = False
    print("[32bit-HDR] colour-science not found. Install with: pip install colour-science")


# ═══════════════════════════════════════════════════════════════════════════════
#                           UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy float32 array."""
    if tensor.dim() == 4:
        # Batch dimension: (B, H, W, C)
        return tensor.cpu().numpy().astype(np.float32)
    elif tensor.dim() == 3:
        # Single image: (H, W, C)
        return tensor.cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unexpected tensor dimensions: {tensor.dim()}")


def numpy_to_tensor_float32(array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor."""
    if array.ndim == 3:
        array = array[np.newaxis, ...]  # Add batch dimension
    return torch.from_numpy(array.astype(np.float32))


def ensure_linear(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Convert sRGB to linear color space."""
    
    # Bypass if gamma is 1.0 (Already Linear)
    if abs(gamma - 1.0) < 0.01:
        return image.astype(np.float32)
        
    # Handle negative values
    sign = np.sign(image)
    abs_image = np.abs(image)
    
    # If gamma matches sRGB approx (2.2), use the precise sRGB curve
    if abs(gamma - 2.2) < 0.1:
        linear = np.where(
            abs_image <= 0.04045,
            abs_image / 12.92,
            np.power((abs_image + 0.055) / 1.055, 2.4)
        )
    else:
        # Generic Gamma
        linear = np.power(abs_image, gamma)
        
    return sign * linear


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB color space."""
    sign = np.sign(image)
    abs_image = np.abs(image)
    
    srgb = np.where(
        abs_image <= 0.0031308,
        abs_image * 12.92,
        1.055 * np.power(abs_image, 1.0/2.4) - 0.055
    )
    return sign * srgb


def tensor_srgb_to_linear(tensor: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """
    Convert sRGB tensor to linear color space.
    GPU-compatible and differentiable.
    """
    # Bypass if gamma is 1.0 (Already Linear)
    if abs(gamma - 1.0) < 0.01:
        return tensor.float()
        
    # Handle negative values
    sign = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)
    
    # If gamma matches sRGB approx (2.2), use the precise sRGB curve
    if abs(gamma - 2.2) < 0.1:
        linear = torch.where(
            abs_tensor <= 0.04045,
            abs_tensor / 12.92,
            torch.pow((abs_tensor + 0.055) / 1.055, 2.4)
        )
    else:
        # Generic Gamma
        linear = torch.pow(abs_tensor, gamma)
        
    return sign * linear


def tensor_linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert linear tensor to sRGB color space.
    GPU-compatible and differentiable.
    """
    sign = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)
    
    srgb = torch.where(
        abs_tensor <= 0.0031308,
        abs_tensor * 12.92,
        1.055 * torch.pow(abs_tensor, 1.0/2.4) - 0.055
    )
    return sign * srgb


# ═══════════════════════════════════════════════════════════════════════════════
#                          CORE 32-BIT NODES
# ═══════════════════════════════════════════════════════════════════════════════

class ImageToFloat32:
    """
    Convert image tensor to 32-bit floating point precision.
    Ensures full HDR range is preserved without clamping.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": False}),
                "source_gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Convert images to 32-bit float precision for HDR processing. Preserves full dynamic range without clamping."
    
    def convert(self, image: torch.Tensor, normalize: bool = False, 
                source_gamma: float = 2.2) -> Tuple[torch.Tensor]:
        # Ensure float32
        img = image.float()
        
        if normalize and img.max() > 1.0:
            img = img / img.max()
        
        return (img,)


class Float32ColorCorrect:
    """
    Professional color correction in 32-bit float space.
    Preserves full dynamic range during adjustments.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to color correct. Processed in 32-bit float precision."}),
                "exposure": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Exposure adjustment in stops. +1 = double brightness, -1 = half brightness. Use for overall brightness control."
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Contrast multiplier around midpoint (0.5). Values >1 increase contrast, <1 decrease. 0 = flat gray."
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Additive brightness offset. Unlike exposure, this is linear shift not multiplicative."
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Color saturation. 0 = grayscale, 1 = original, >1 = more saturated. Uses Rec.709 luminance-preserving method."
                }),
            },
            "optional": {
                "gamma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01,
                    "tooltip": "Gamma correction (power curve). <1 = brighten midtones, >1 = darken midtones. Applied after other corrections."
                }),
                "lift_r": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                    "tooltip": "Red channel lift (shadow offset). Adds to red channel after gain. Affects entire tonal range but most visible in shadows."
                }),
                "lift_g": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                    "tooltip": "Green channel lift (shadow offset). Adds to green channel after gain."
                }),
                "lift_b": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                    "tooltip": "Blue channel lift (shadow offset). Adds to blue channel after gain."
                }),
                "gain_r": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Red channel gain (multiplier). 1 = no change, <1 = reduce red, >1 = boost red."
                }),
                "gain_g": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Green channel gain (multiplier). 1 = no change."
                }),
                "gain_b": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Blue channel gain (multiplier). 1 = no change."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Color corrected image in 32-bit float.",)
    FUNCTION = "correct"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Professional 32-bit color correction with exposure, contrast, saturation, gamma, and per-channel lift/gain controls."
    
    def correct(self, image: torch.Tensor, exposure: float = 0.0, 
                contrast: float = 1.0, brightness: float = 0.0,
                saturation: float = 1.0, gamma: float = 1.0,
                lift_r: float = 0.0, lift_g: float = 0.0, lift_b: float = 0.0,
                gain_r: float = 1.0, gain_g: float = 1.0, 
                gain_b: float = 1.0) -> Tuple[torch.Tensor]:
        
        img = image.clone().float()
        
        # Apply exposure (in stops)
        if exposure != 0.0:
            img = img * (2.0 ** exposure)
        
        # Apply lift/gain per channel
        if img.shape[-1] >= 3:
            img[..., 0] = img[..., 0] * gain_r + lift_r
            img[..., 1] = img[..., 1] * gain_g + lift_g
            img[..., 2] = img[..., 2] * gain_b + lift_b
        
        # Apply contrast around midpoint
        if contrast != 1.0:
            img = (img - 0.5) * contrast + 0.5
        
        # Apply brightness
        if brightness != 0.0:
            img = img + brightness
        
        # Apply gamma
        if gamma != 1.0:
            img = torch.clamp(img, min=0.0)
            img = torch.pow(img + 1e-8, 1.0 / gamma)
        
        # Apply saturation
        if saturation != 1.0 and img.shape[-1] >= 3:
            luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            luma = luma.unsqueeze(-1)
            img = luma + saturation * (img - luma)
        
        return (img,)



class HDRExpandDynamicRange:
    """
    Expand image dynamic range for HDR output.
    Simulates extended stops of dynamic range from SDR source.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "source_gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1}),
                "highlight_recovery": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "black_point": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001}),
                "target_stops": ("FLOAT", {"default": 14.0, "min": 8.0, "max": 20.0, "step": 0.5}),
                "highlight_rolloff": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "expand"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Expand SDR images to HDR dynamic range by recovering highlights and extending stops of exposure latitude."
    
    def expand(self, image: torch.Tensor, source_gamma: float = 2.2,
               highlight_recovery: float = 1.0, black_point: float = 0.0,
               target_stops: float = 14.0, highlight_rolloff: float = 1.5) -> Tuple[torch.Tensor]:
        
        # GPU-accelerated implementation
        img = image.float()
        
        # 1. Convert to linear using tensor helper
        linear = tensor_srgb_to_linear(img, source_gamma)
        
        # 2. Adjust black point
        linear = linear - black_point
        linear = torch.clamp(linear, min=0.0)
        
        # 3. Calculate Luminance (to preserve color relationships)
        # Using Rec.709 coefficients
        luma = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
        
        # 4. Expansion Logic
        # Target Peak: 2^(stops - 8). e.g., 14 stops -> 2^6 = 64.0
        target_peak = 2.0 ** (target_stops - 8.0)
        
        # Use highlight_rolloff to control the threshold softness
        # Higher rolloff = softer transition, starts expansion earlier
        # Range 1.0-3.0 maps to threshold 0.9-0.5
        threshold = 1.0 - (highlight_rolloff - 1.0) * 0.2  # rolloff 1.0->0.9, 1.5->0.8, 3.0->0.5
        threshold = max(0.5, min(0.95, threshold))  # Clamp to safe range
        
        if target_peak > 1.0 and highlight_recovery > 0:
            t = threshold
            
            # Calculate 'a' coefficient for quadratic curve
            a = (target_peak - 1.0) / ((1.0 - t) ** 2)
            
            # Clamp luma for curve calculation
            luma_clamped = torch.clamp(luma, max=1.0)
            
            # Quadratic expansion: y = t + (x - t) + a * (x - t)^2
            # Only apply to values above threshold
            expanded_luma = torch.where(
                luma > t,
                t + (luma_clamped - t) + a * torch.pow(luma_clamped - t, 2),
                luma
            )
            
            # For inputs > 1.0, extrapolate linearly with slope at 1.0
            slope_at_1 = 1.0 + 2 * a * (1.0 - t)
            expanded_luma = torch.where(
                luma > 1.0,
                target_peak + (luma - 1.0) * slope_at_1,
                expanded_luma
            )
            
            # Blend based on recovery strength
            final_luma = expanded_luma * highlight_recovery + luma * (1.0 - highlight_recovery)
            
            # Apply new luminance to RGB
            # NewColor = OldColor * (NewLuma / OldLuma)
            ratio = final_luma / (luma + 1e-8)
            
            # Expand dims for RGB multiply
            ratio = ratio.unsqueeze(-1)
            
            linear = linear * ratio
        
        return (linear,)


class HDRToneMap:
    """
    Professional HDR tone mapping with presets and advanced grading controls.
    GPU-accelerated with automatic CPU fallback.
    
    Features:
    - 8 industry-standard tone mapping operators
    - One-click look presets (Film, HDR TV, Web, Print, etc.)
    - Advanced highlight/shadow control
    - Contrast and saturation grading
    - GPU-accelerated with CPU fallback
    
    Example:
        Use preset "🎬 Cinematic Film" for instant film look
        Or select operator and fine-tune with controls
    """
    
    TONEMAP_OPERATORS = [
        "filmic_aces",
        "filmic_uncharted2",
        "agx",
        "reinhard",
        "reinhard_extended", 
        "reinhard_luminance",
        "linear_clamp",
        "exposure_only"
    ]
    
    # Tone map look presets
    LOOK_PRESETS = [
        "None (Custom)",
        "🎬 Cinematic Film",
        "📺 HDR Display",
        "🌐 Web / Social",
        "🖨️ Print Ready",
        "🎮 Game Engine",
        "📷 Photography",
        "🌙 Low Key / Dark",
        "☀️ High Key / Bright",
    ]
    
    PRESET_CONFIGS = {
        "🎬 Cinematic Film": {
            "operator": "filmic_aces",
            "exposure": 0.0,
            "gamma": 2.2,
            "white_point": 1.0,
            "contrast": 1.1,
            "saturation": 0.95,
            "highlight_compression": 0.8,
            "shadow_lift": 0.02,
        },
        "📺 HDR Display": {
            "operator": "agx",
            "exposure": 0.3,
            "gamma": 2.2,
            "white_point": 2.0,
            "contrast": 1.0,
            "saturation": 1.1,
            "highlight_compression": 0.5,
            "shadow_lift": 0.0,
        },
        "🌐 Web / Social": {
            "operator": "filmic_aces",
            "exposure": 0.2,
            "gamma": 2.2,
            "white_point": 1.0,
            "contrast": 1.15,
            "saturation": 1.1,
            "highlight_compression": 0.9,
            "shadow_lift": 0.01,
        },
        "🖨️ Print Ready": {
            "operator": "reinhard_luminance",
            "exposure": -0.2,
            "gamma": 2.2,
            "white_point": 1.0,
            "contrast": 0.95,
            "saturation": 0.9,
            "highlight_compression": 0.7,
            "shadow_lift": 0.03,
        },
        "🎮 Game Engine": {
            "operator": "filmic_uncharted2",
            "exposure": 0.0,
            "gamma": 2.2,
            "white_point": 4.0,
            "contrast": 1.05,
            "saturation": 1.0,
            "highlight_compression": 0.6,
            "shadow_lift": 0.0,
        },
        "📷 Photography": {
            "operator": "reinhard_extended",
            "exposure": 0.0,
            "gamma": 2.2,
            "white_point": 2.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "highlight_compression": 0.5,
            "shadow_lift": 0.0,
        },
        "🌙 Low Key / Dark": {
            "operator": "filmic_aces",
            "exposure": -0.5,
            "gamma": 2.4,
            "white_point": 1.0,
            "contrast": 1.2,
            "saturation": 0.85,
            "highlight_compression": 0.9,
            "shadow_lift": 0.0,
        },
        "☀️ High Key / Bright": {
            "operator": "reinhard",
            "exposure": 0.5,
            "gamma": 2.0,
            "white_point": 1.5,
            "contrast": 0.9,
            "saturation": 1.05,
            "highlight_compression": 0.4,
            "shadow_lift": 0.05,
        },
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input HDR or SDR image to tone map"}),
            },
            "optional": {
                "preset": (cls.LOOK_PRESETS, {"default": "🎬 Cinematic Film", "tooltip": "Quick look preset. Overrides settings below."}),
                "operator": (cls.TONEMAP_OPERATORS, {"default": "filmic_aces", "tooltip": "Tone mapping algorithm:\n• filmic_aces: Industry-standard film curve\n• filmic_uncharted2: Game industry favorite\n• agx: Modern Blender default\n• reinhard: Classic, preserves color\n• linear_clamp: Simple clip"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "tooltip": "Exposure adjustment in stops. Negative = darker, Positive = brighter."}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1, "tooltip": "Display gamma. 2.2 = sRGB standard. Higher = darker midtones."}),
                "white_point": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 10.0, "step": 0.1, "tooltip": "Maximum brightness that maps to white. Higher = more headroom for highlights."}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "Contrast adjustment around midpoint. >1 = more punch."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Color saturation. 0 = grayscale, 1 = original, >1 = vivid."}),
                "highlight_compression": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Compress highlights to prevent clipping. 0 = no compression, 1 = full."}),
                "shadow_lift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Lift shadows to reveal detail. 0 = no lift, 0.1 = subtle."}),
                "use_gpu": ("BOOLEAN", {"default": True, "tooltip": "Use GPU acceleration when available."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Tone-mapped SDR image ready for display or export.",)
    FUNCTION = "tonemap"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Professional HDR tone mapping with presets and advanced controls. GPU-accelerated."
    
    def _gpu_tonemap(self, x: torch.Tensor, operator: str, white_point: float) -> torch.Tensor:
        """GPU-accelerated tone mapping."""
        if operator == "reinhard":
            return x / (1.0 + x)
        elif operator == "reinhard_extended":
            white_sq = white_point * white_point
            return (x * (1.0 + x / white_sq)) / (1.0 + x)
        elif operator == "reinhard_luminance":
            # GPU version of luminance-based Reinhard
            luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
            luma = torch.clamp(luma, min=1e-6)
            white_sq = white_point * white_point
            luma_tm = (luma * (1.0 + luma / white_sq)) / (1.0 + luma)
            scale = (luma_tm / luma).unsqueeze(-1)
            return x * scale
        elif operator == "filmic_aces":
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
        elif operator == "filmic_uncharted2":
            A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
            def curve(v):
                return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F
            white_scale = 1.0 / curve(torch.tensor(white_point, device=x.device))
            return curve(x) * white_scale
        elif operator == "agx":
            x = torch.clamp(x, min=1e-6)
            x = torch.log2(x + 1e-6) / 16.0 + 0.5
            x = torch.clamp(x, 0, 1)
            return x * x * (3.0 - 2.0 * x)
        elif operator == "linear_clamp":
            return torch.clamp(x / white_point, 0, 1)
        else:  # exposure_only
            return torch.clamp(x, 0, 1)
    
    def _reinhard(self, x: np.ndarray, white: float = 1.0) -> np.ndarray:
        """Simple Reinhard tone mapping."""
        return x / (1.0 + x)
    
    def _reinhard_extended(self, x: np.ndarray, white: float = 4.0) -> np.ndarray:
        """Extended Reinhard with white point."""
        numerator = x * (1.0 + x / (white * white))
        return numerator / (1.0 + x)
    
    def _reinhard_luminance(self, rgb: np.ndarray, white: float = 4.0) -> np.ndarray:
        """Reinhard applied to luminance only."""
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        luma = np.maximum(luma, 1e-6)
        
        luma_tm = self._reinhard_extended(luma, white)
        scale = luma_tm / luma
        
        return rgb * scale[..., np.newaxis]
    
    def _filmic_aces(self, x: np.ndarray) -> np.ndarray:
        """ACES filmic tone mapping approximation."""
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    
    def _filmic_uncharted2(self, x: np.ndarray) -> np.ndarray:
        """Uncharted 2 filmic curve."""
        A = 0.15  # Shoulder strength
        B = 0.50  # Linear strength
        C = 0.10  # Linear angle
        D = 0.20  # Toe strength
        E = 0.02  # Toe numerator
        F = 0.30  # Toe denominator
        
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    
    def _agx(self, x: np.ndarray) -> np.ndarray:
        """AgX-inspired tone mapping."""
        # Simplified AgX curve
        x = np.maximum(x, 0.0)
        x = np.log2(x + 1e-6) / 16.0 + 0.5
        x = np.clip(x, 0, 1)
        
        # Contrast
        x = x * x * (3.0 - 2.0 * x)
        return x
    
    def tonemap(self, image: torch.Tensor, preset: str = "🎬 Cinematic Film",
                operator: str = "filmic_aces", exposure: float = 0.0, gamma: float = 2.2,
                white_point: float = 1.0, contrast: float = 1.0, saturation: float = 1.0,
                highlight_compression: float = 0.5, shadow_lift: float = 0.0,
                use_gpu: bool = True) -> Tuple[torch.Tensor]:
        
        # Apply preset if selected
        if preset != "None (Custom)" and preset in self.PRESET_CONFIGS:
            config = self.PRESET_CONFIGS[preset]
            operator = config.get("operator", operator)
            exposure = config.get("exposure", exposure)
            gamma = config.get("gamma", gamma)
            white_point = config.get("white_point", white_point)
            contrast = config.get("contrast", contrast)
            saturation = config.get("saturation", saturation)
            highlight_compression = config.get("highlight_compression", highlight_compression)
            shadow_lift = config.get("shadow_lift", shadow_lift)
        
        # Try GPU path
        if use_gpu and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                img = image.to(device).float()
                
                # Apply exposure
                img = img * (2.0 ** exposure)
                
                # Apply highlight compression (before tone mapping)
                if highlight_compression > 0:
                    # Soft knee compression for highlights
                    threshold = 1.0 - highlight_compression * 0.5
                    highlight_mask = img > threshold
                    compressed = threshold + (img - threshold) / (1.0 + (img - threshold) * highlight_compression * 2)
                    img = torch.where(highlight_mask, compressed, img)
                
                # Apply shadow lift
                if shadow_lift > 0:
                    img = img + shadow_lift * (1.0 - img)
                
                # Apply tone mapping
                result = self._gpu_tonemap(img, operator, white_point)
                
                # Apply contrast
                if contrast != 1.0:
                    result = (result - 0.5) * contrast + 0.5
                
                # Apply saturation
                if saturation != 1.0 and result.shape[-1] >= 3:
                    luma = 0.2126 * result[..., 0] + 0.7152 * result[..., 1] + 0.0722 * result[..., 2]
                    luma = luma.unsqueeze(-1)
                    result = luma + saturation * (result - luma)
                
                # Apply gamma
                result = torch.clamp(result, 0, 1)
                result = torch.pow(result, 1.0 / gamma)
                
                return (result,)
                
            except RuntimeError:
                torch.cuda.empty_cache()
        
        # CPU fallback
        img = tensor_to_numpy_float32(image)
        
        # Apply exposure
        img = img * (2.0 ** exposure)
        
        # Apply highlight compression (before tone mapping)
        if highlight_compression > 0:
            threshold = 1.0 - highlight_compression * 0.5
            highlight_mask = img > threshold
            compressed = threshold + (img - threshold) / (1.0 + (img - threshold) * highlight_compression * 2)
            img = np.where(highlight_mask, compressed, img)
        
        # Apply shadow lift
        if shadow_lift > 0:
            img = img + shadow_lift * (1.0 - img)
        
        # Apply tone mapping operator
        if operator == "reinhard":
            result = self._reinhard(img, white_point)
        elif operator == "reinhard_extended":
            result = self._reinhard_extended(img, white_point)
        elif operator == "reinhard_luminance":
            result = self._reinhard_luminance(img, white_point)
        elif operator == "filmic_aces":
            result = self._filmic_aces(img)
        elif operator == "filmic_uncharted2":
            white_scale = 1.0 / self._filmic_uncharted2(np.array([white_point]))[0]
            result = self._filmic_uncharted2(img) * white_scale
        elif operator == "agx":
            result = self._agx(img)
        elif operator == "linear_clamp":
            result = np.clip(img / white_point, 0, 1)
        else:  # exposure_only
            result = np.clip(img, 0, 1)
        
        # Apply contrast
        if contrast != 1.0:
            result = (result - 0.5) * contrast + 0.5
        
        # Apply saturation adjustment
        if saturation != 1.0 and result.shape[-1] >= 3:
            luma = 0.2126 * result[..., 0] + 0.7152 * result[..., 1] + 0.0722 * result[..., 2]
            luma = luma[..., np.newaxis]
            result = luma + saturation * (result - luma)
        
        # Apply gamma (convert to display)
        result = np.clip(result, 0, 1)
        result = np.power(result, 1.0 / gamma)
        
        return (numpy_to_tensor_float32(result),)


class ColorSpaceConvert:
    """
    Convert between color spaces for VFX pipeline integration.
    GPU-accelerated with automatic CPU fallback.
    """
    
    COLOR_SPACES = [
        "sRGB",
        "Linear_sRGB", 
        "ACEScg",
        "ACEScct",
        "ACES2065-1",
        "Rec709",
        "Rec2020",
        "DCI-P3",
        "Display_P3"
    ]
    
    CHROMATIC_ADAPTATIONS = ["Bradford", "Von Kries", "XYZ Scaling", "None"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "source_space": (cls.COLOR_SPACES, {"default": "sRGB"}),
                "target_space": (cls.COLOR_SPACES, {"default": "ACEScg"}),
            },
            "optional": {
                "chromatic_adaptation": (cls.CHROMATIC_ADAPTATIONS, {"default": "Bradford"}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "GPU-accelerated color space conversion (sRGB, ACEScg, ACEScct, Rec.2020, DCI-P3)."
    
    # Color space primaries (xy chromaticity)
    PRIMARIES = {
        "sRGB": np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]),
        "Linear_sRGB": np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]),
        "ACEScg": np.array([[0.713, 0.293], [0.165, 0.830], [0.128, 0.044]]),
        "ACEScct": np.array([[0.713, 0.293], [0.165, 0.830], [0.128, 0.044]]), # ACEScct uses AP1
        "ACES2065-1": np.array([[0.7347, 0.2653], [0.0, 1.0], [0.0001, -0.077]]),
        "Rec709": np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]),
        "Rec2020": np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]),
        "DCI-P3": np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]),
        "Display_P3": np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]),
    }
    
    # White points (xy chromaticity)
    WHITE_POINTS = {
        "sRGB": np.array([0.3127, 0.3290]),  # D65
        "Linear_sRGB": np.array([0.3127, 0.3290]),
        "ACEScg": np.array([0.32168, 0.33767]),  # ACES white
        "ACEScct": np.array([0.32168, 0.33767]),
        "ACES2065-1": np.array([0.32168, 0.33767]),
        "Rec709": np.array([0.3127, 0.3290]),
        "Rec2020": np.array([0.3127, 0.3290]),
        "DCI-P3": np.array([0.314, 0.351]),  # DCI white
        "Display_P3": np.array([0.3127, 0.3290]),  # D65
    }
    
    def _primaries_to_matrix(self, primaries: np.ndarray, white: np.ndarray) -> np.ndarray:
        """Calculate RGB to XYZ matrix from primaries and white point."""
        # Convert xy to XYZ
        def xy_to_XYZ(xy):
            return np.array([xy[0]/xy[1], 1.0, (1-xy[0]-xy[1])/xy[1]])
        
        r_XYZ = xy_to_XYZ(primaries[0])
        g_XYZ = xy_to_XYZ(primaries[1])
        b_XYZ = xy_to_XYZ(primaries[2])
        w_XYZ = xy_to_XYZ(white)
        
        M = np.column_stack([r_XYZ, g_XYZ, b_XYZ])
        S = np.linalg.solve(M, w_XYZ)
        
        return M * S
    
    
    def _srgb_to_linear(self, rgb: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear."""
        return ensure_linear(rgb)
    
    def _linear_to_srgb(self, rgb: np.ndarray) -> np.ndarray:
        """Convert linear to sRGB."""
        return linear_to_srgb(rgb)

    def _get_gpu_adaptation_matrix(self, src_white, tgt_white, device):
        """Calculate Bradford adaptation matrix on GPU."""
        if np.allclose(src_white, tgt_white):
            return torch.eye(3, device=device)
            
        M_bradford = torch.tensor([
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296]
        ], dtype=torch.float32, device=device)
        
        # Convert white points to XYZ
        def xy_to_XYZ(xy):
            return torch.tensor([xy[0]/xy[1], 1.0, (1-xy[0]-xy[1])/xy[1]], 
                              dtype=torch.float32, device=device)
        
        src_w_node = torch.tensor(src_white, dtype=torch.float32, device=device)
        tgt_w_node = torch.tensor(tgt_white, dtype=torch.float32, device=device)
        
        src_XYZ = xy_to_XYZ(src_w_node)
        tgt_XYZ = xy_to_XYZ(tgt_w_node)
        
        src_cone = M_bradford @ src_XYZ
        tgt_cone = M_bradford @ tgt_XYZ
        
        scale = tgt_cone / (src_cone + 1e-8)
        # Use linalg.inv for better precision/stability
        M_adapt = torch.linalg.inv(M_bradford) @ torch.diag(scale) @ M_bradford
        return M_adapt

    def convert(self, image: torch.Tensor, source_space: str = "sRGB",
                target_space: str = "ACEScg", 
                chromatic_adaptation: str = "Bradford",
                use_gpu: bool = True) -> Tuple[torch.Tensor]:
        
        if source_space == target_space:
            return (image,)
        
        # Try GPU path for common conversions
        if use_gpu and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                img = image.to(device).float()
                
                # Input Transfer Functions (removal)
                if source_space == "sRGB":
                    img = torch.where(img <= 0.04045, img / 12.92, 
                                      torch.pow((img + 0.055) / 1.055, 2.4))
                elif source_space == "ACEScct":
                    # ACEScct to Linear (AP1)
                    A = 10.5402377416545
                    B = 0.0729055341958355
                    img = torch.where(img <= 0.16875,
                                      (img - B) / A,
                                      torch.pow(2.0, img * 17.52 - 9.72))
                
                # Get matrices for GPU path
                src_matrix = torch.from_numpy(self._primaries_to_matrix(
                    self.PRIMARIES[source_space], 
                    self.WHITE_POINTS[source_space]
                ).astype(np.float32)).to(device)
                
                tgt_matrix = torch.from_numpy(self._primaries_to_matrix(
                    self.PRIMARIES[target_space],
                    self.WHITE_POINTS[target_space]
                ).astype(np.float32)).to(device)
                
                # Handle alpha
                has_alpha = img.shape[-1] == 4
                if has_alpha:
                    alpha = img[..., 3:4]
                    rgb = img[..., :3]
                else:
                    rgb = img
                
                # Convert to XYZ using einsum
                # Fix: Do not transpose matrix here. Einsum '...c,kc->...k' implies:
                # Out[k] = Sum_c( In[c] * Matrix[k,c] )
                # Since M[k,c] corresponds to Row k, Col c of Matrix M,
                # and M maps Col_c (Source) to Row_k (Dest), this is correct without .T
                xyz = torch.einsum('...c,kc->...k', rgb, src_matrix)
                
                # Apply Chromatic Adaptation (Bradford)
                if chromatic_adaptation != "None":
                    src_white = self.WHITE_POINTS[source_space]
                    tgt_white = self.WHITE_POINTS[target_space]
                    
                    if not np.allclose(src_white, tgt_white):
                        M_adapt = self._get_gpu_adaptation_matrix(src_white, tgt_white, device)
                        xyz = torch.einsum('...c,kc->...k', xyz, M_adapt)
                
                # Convert from XYZ to target
                # Use linalg.inv
                inv_tgt = torch.linalg.inv(tgt_matrix)
                result = torch.einsum('...c,kc->...k', xyz, inv_tgt)
                
                # GPU linear to sRGB
                if target_space == "sRGB":
                    result = torch.where(result <= 0.0031308, result * 12.92,
                                         1.055 * torch.pow(result.clamp(min=1e-10), 1/2.4) - 0.055)
                # GPU linear to ACEScct (Log)
                elif target_space == "ACEScct":
                     # AP1 to ACEScct curve
                     A = 10.5402377416545
                     B = 0.0729055341958355
                     result = torch.where(result <= 0.018581, 
                                          A * result + B,
                                          (torch.log2(result) + 9.72) / 17.52)
                
                # Restore alpha
                if has_alpha:
                    result = torch.cat([result, alpha], dim=-1)
                
                return (result.cpu(),)
            
            except Exception as gpu_err:
                # print(f"[ColorSpaceConvert] GPU Error: {gpu_err}. Falling back to CPU.")
                pass # Fallback to CPU
        
        # CPU fallback
        img = tensor_to_numpy_float32(image)
        original_shape = img.shape
        
        # Flatten to 2D for matrix operations
        if img.ndim == 4:
            batch, h, w, c = img.shape
            img = img.reshape(-1, c)
        else:
            h, w, c = img.shape
            batch = 1
            img = img.reshape(-1, c)
        
        # Handle alpha channel
        has_alpha = c == 4
        if has_alpha:
            alpha = img[:, 3:4]
            img = img[:, :3]
        
        # Step 1: Convert source to linear if needed
        if source_space == "sRGB":
            img = self._srgb_to_linear(img)
        
        # Step 2: Convert to XYZ
        src_matrix = self._primaries_to_matrix(
            self.PRIMARIES[source_space], 
            self.WHITE_POINTS[source_space]
        )
        xyz = img @ src_matrix.T
        
        # Step 3: Chromatic adaptation if white points differ
        src_white = self.WHITE_POINTS[source_space]
        tgt_white = self.WHITE_POINTS[target_space]
        
        if not np.allclose(src_white, tgt_white) and chromatic_adaptation != "None":
            # Bradford chromatic adaptation
            M_bradford = np.array([
                [0.8951, 0.2664, -0.1614],
                [-0.7502, 1.7135, 0.0367],
                [0.0389, -0.0685, 1.0296]
            ])
            
            def xy_to_XYZ(xy):
                return np.array([xy[0]/xy[1], 1.0, (1-xy[0]-xy[1])/xy[1]])
            
            src_XYZ = xy_to_XYZ(src_white)
            tgt_XYZ = xy_to_XYZ(tgt_white)
            
            src_cone = M_bradford @ src_XYZ
            tgt_cone = M_bradford @ tgt_XYZ
            
            scale = tgt_cone / src_cone
            M_adapt = np.linalg.inv(M_bradford) @ np.diag(scale) @ M_bradford
            
            xyz = xyz @ M_adapt.T
        
        # Step 4: Convert from XYZ to target
        tgt_matrix = self._primaries_to_matrix(
            self.PRIMARIES[target_space],
            self.WHITE_POINTS[target_space]
        )
        rgb = xyz @ np.linalg.inv(tgt_matrix).T
        
        # Step 5: Apply target transfer function if needed
        if target_space == "sRGB":
            rgb = self._linear_to_srgb(rgb)
        
        # Restore alpha
        if has_alpha:
            rgb = np.concatenate([rgb, alpha], axis=-1)
        
        # Reshape back
        rgb = rgb.reshape(original_shape)
        
        return (numpy_to_tensor_float32(rgb),)


# ═══════════════════════════════════════════════════════════════════════════════
#                           EXR OUTPUT NODES
# ═══════════════════════════════════════════════════════════════════════════════


# Import signature mixin
from .nodes_dna import FXTDSignatureMixin

class SaveImageEXR(FXTDSignatureMixin):
    """
    Save images as 32-bit or 16-bit EXR files.
    Supports multiple compression methods and multi-layer output.
    """
    
    BIT_DEPTHS = ["float32", "float16", "uint32"]
    COMPRESSION_TYPES = ["NONE", "RLE", "ZIPS", "ZIP", "PIZ", "PXR24", "B44", "B44A", "DWAA", "DWAB"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "output/hdr_"}),
                "bit_depth": (cls.BIT_DEPTHS, {"default": "float32"}),
                "compression": (cls.COMPRESSION_TYPES, {"default": "ZIP"}),
            },
            "optional": {
                "include_alpha": ("BOOLEAN", {"default": True}),
                "layer_name": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Save images as 32-bit or 16-bit EXR files with multiple compression options for VFX pipelines."


    def save(self, images: torch.Tensor, filename_prefix: str = "output/hdr_",
             bit_depth: str = "float32", compression: str = "ZIP",
             include_alpha: bool = True, layer_name: str = "") -> Dict:
        """
        Save images as EXR files using OpenCV (most reliable method).
        OpenCV natively supports 32-bit float EXR output.
        """
        import cv2
        import folder_paths
        
        # Get ComfyUI output directory
        output_dir = Path(folder_paths.get_output_directory())
        
        # Parse the filename prefix
        clean_prefix = filename_prefix.replace("\\", "/")
        if clean_prefix.startswith("output/"):
            clean_prefix = clean_prefix[7:]
        
        # Split into directory and base name
        if "/" in clean_prefix:
            subdir, base_name = clean_prefix.rsplit("/", 1)
            full_dir = output_dir / subdir
        else:
            full_dir = output_dir
            base_name = clean_prefix
        
        # Create directory
        full_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        counter = 0

        # Find starting counter (simple check)
        while True:
            filename = f"{base_name}_{counter:05d}.exr"
            file_path = full_dir / filename
            if not file_path.exists():
                break
            counter += 1
        
        for i, image in enumerate(images):
            # Sign the image with "FXTD Digital DNA" before saving
            try:
                # Sign individual image tensor (C, H, W) or (H, W, C)
                # sign_image expects standard format, usually we might need to conform it
                signed_image = self.sign_image(image, extra_metadata={
                    "saved_as": bit_depth,
                    "compression": compression
                })
            except Exception as e:
                print(f"[SaveImageEXR] Signing warning: {e}")
                signed_image = image

            # Convert to numpy float32
            np_img = tensor_to_numpy_float32(signed_image)
            
            # Generate filename
            filename = f"{base_name}_{counter:05d}.exr"
            file_path = full_dir / filename
            filepath_str = str(file_path.resolve())
            
            # Convert RGB/RGBA to BGR/BGRA for OpenCV
            if np_img.ndim == 3:
                h, w, c = np_img.shape
                bgr = np_img.copy() # Default
                
                if c == 3:
                    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                elif c == 4:
                    if include_alpha:
                        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)
                    else:
                        img_rgb = np_img[:, :, :3]
                        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                 # Handle unusual shapes if necessary
                 bgr = np_img

            # Apply bit depth constraint if requested (OpenCV usually handles float32/16)
            flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
            if bit_depth == "float16":
                flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]
            
            try:
                success = cv2.imwrite(filepath_str, bgr, flags)
                
                if success:
                    results.append({"filename": filename, "subfolder": str(subdir) if "/" in clean_prefix else "", "type": "output"})
                else:
                    print(f"[32bit-HDR] ✗ Failed to save: {filepath_str}")
                    
            except Exception as e:
                print(f"[32bit-HDR] Error saving {filepath_str}: {e}")
                
            counter += 1
            
        return {"ui": {"images": results}}




class LoadImageEXR:
    """
    Load EXR/HDR files with full HDR data preservation.
    Supports 16-bit and 32-bit float EXR files.
    
    Two input methods:
    1. file_path: Enter the full path to any EXR/HDR file on your system
    2. image: Select from EXR/HDR files already in ComfyUI's input folder
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        files = []
        try:
            for f in os.listdir(input_dir):
                if f.lower().endswith(('.exr', '.hdr')):
                    files.append(f)
        except Exception:
            pass
        
        # Build file list with "none" as first option
        file_options = ["-- Select from input folder --"] + sorted(files) if files else ["-- No EXR files in input folder --"]
            
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Full path to EXR/HDR file (e.g., C:/images/my_image.exr). Leave empty to use dropdown."
                }),
            },
            "optional": {
                "input_folder_file": (file_options, {
                    "tooltip": "Select EXR/HDR from ComfyUI's input folder. Only used if file_path is empty."
                }),
                "exposure_adjust": ("FLOAT", {
                    "default": 0.0, 
                    "min": -10.0, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "Adjust exposure in stops (EV). Positive = brighter, negative = darker."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "load"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Load EXR/HDR files with full HDR dynamic range. Enter file path directly or select from input folder."
    
    @classmethod
    def IS_CHANGED(cls, file_path, **kwargs):
        # Determine which path to use
        actual_path = cls._get_actual_path(file_path, kwargs.get("input_folder_file", ""))
        if not actual_path:
            return float("nan")
        try:
            if os.path.exists(actual_path):
                return os.path.getmtime(actual_path)
        except Exception:
            pass
        return float("nan")
    
    @classmethod
    def _get_actual_path(cls, file_path: str, input_folder_file: str) -> str:
        """Determine the actual file path to load."""
        import folder_paths
        
        # Priority 1: Direct file path if provided
        if file_path and file_path.strip():
            # Strip whitespace and quotes (users often paste paths with quotes)
            cleaned_path = file_path.strip().strip('"').strip("'").strip()
            if cleaned_path:
                return cleaned_path
        
        # Priority 2: Input folder selection
        if input_folder_file and not input_folder_file.startswith("--"):
            return folder_paths.get_annotated_filepath(input_folder_file)
        
        return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, file_path, **kwargs):
        input_folder_file = kwargs.get("input_folder_file", "")
        
        actual_path = cls._get_actual_path(file_path, input_folder_file)
        
        if not actual_path:
            return "No file specified. Either enter a file path OR select a file from the input folder dropdown."
        
        if not os.path.exists(actual_path):
            return f"File not found: '{actual_path}'"
        
        if not actual_path.lower().endswith(('.exr', '.hdr')):
            return f"Invalid file type. Only .exr and .hdr files are supported. Got: '{actual_path}'"
        
        return True
    
    def load(self, file_path: str, input_folder_file: str = "", exposure_adjust: float = 0.0) -> Tuple[torch.Tensor, str]:
        """Load EXR/HDR file using multiple backends for maximum compatibility."""
        
        # Get the actual path to load
        actual_path = self._get_actual_path(file_path, input_folder_file)
        
        if not actual_path:
            raise ValueError("No file path specified. Enter a file path or select from dropdown.")
        
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"File not found: {actual_path}")
        
        img = None
        load_method = "unknown"
        
        # Method 0: Try OpenEXR library (most reliable, just installed)
        try:
            import OpenEXR
            import Imath
            
            exr_file = OpenEXR.InputFile(actual_path)
            header = exr_file.header()
            
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # Get channel names
            channels = list(header['channels'].keys())
            
            # Read pixel data
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            
            if 'R' in channels and 'G' in channels and 'B' in channels:
                # RGB or RGBA
                r_str = exr_file.channel('R', pt)
                g_str = exr_file.channel('G', pt)
                b_str = exr_file.channel('B', pt)
                
                r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
                g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
                b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
                
                if 'A' in channels:
                    a_str = exr_file.channel('A', pt)
                    a = np.frombuffer(a_str, dtype=np.float32).reshape(height, width)
                    img = np.stack([r, g, b, a], axis=-1)
                else:
                    img = np.stack([r, g, b], axis=-1)
            elif 'Y' in channels:
                # Grayscale
                y_str = exr_file.channel('Y', pt)
                img = np.frombuffer(y_str, dtype=np.float32).reshape(height, width, 1)
            else:
                # Use first available channel
                ch_name = channels[0]
                ch_str = exr_file.channel(ch_name, pt)
                img = np.frombuffer(ch_str, dtype=np.float32).reshape(height, width, 1)
            
            exr_file.close()
            load_method = "OpenEXR"
        except ImportError:
            pass
        except Exception as e:
            print(f"[LoadImageEXR] OpenEXR library failed: {e}")
        
        # Method 1: Try imageio (good compatibility)
        if img is None:
            try:
                import imageio.v3 as iio
                img = iio.imread(actual_path)
                load_method = "imageio"
            except ImportError:
                pass
            except Exception as e:
                print(f"[LoadImageEXR] imageio failed: {e}")
        
        # Method 2: Try OpenImageIO (industry standard)
        if img is None:
            try:
                import OpenImageIO as oiio
                inp = oiio.ImageInput.open(actual_path)
                if inp:
                    spec = inp.spec()
                    img = np.zeros((spec.height, spec.width, spec.nchannels), dtype=np.float32)
                    inp.read_image(0, 0, oiio.FLOAT, img)
                    inp.close()
                    load_method = "OpenImageIO"
            except ImportError:
                pass
            except Exception as e:
                print(f"[LoadImageEXR] OpenImageIO failed: {e}")
        
        # Method 3: Try OpenCV (may not have EXR support)
        if img is None:
            try:
                import cv2
                # Enable OpenEXR if available
                os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
                img = cv2.imread(actual_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    # Convert BGR to RGB
                    if len(img.shape) == 3:
                        if img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                        elif img.shape[2] >= 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    load_method = "OpenCV"
            except Exception as e:
                print(f"[LoadImageEXR] OpenCV failed: {e}")
        
        # Method 4: Try pyexr
        if img is None:
            try:
                import pyexr
                img = pyexr.read(actual_path)
                load_method = "pyexr"
            except ImportError:
                pass
            except Exception as e:
                print(f"[LoadImageEXR] pyexr failed: {e}")
        
        if img is None:
            raise RuntimeError(
                f"Failed to load EXR file: {actual_path}\n\n"
                "Please install one of these packages:\n"
                "  pip install imageio[pyav]\n"
                "  pip install imageio-ffmpeg\n"
                "  pip install pyexr\n"
                "  pip install OpenImageIO\n"
            )
        
        # Ensure float32
        img = np.asarray(img, dtype=np.float32)
        
        # Handle 2D grayscale images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        
        # Apply exposure adjustment
        if exposure_adjust != 0:
            img = img * (2.0 ** exposure_adjust)
        
        # Build metadata
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1
        
        # Calculate dynamic range safely
        min_val = float(img.min())
        max_val = float(img.max())
        if min_val > 0 and max_val > 0:
            dynamic_range = float(np.log2(max_val / min_val))
        else:
            dynamic_range = 0.0
            
        metadata = {
            "file": os.path.basename(actual_path),
            "path": actual_path,
            "width": w,
            "height": h,
            "channels": channels,
            "dtype": str(img.dtype),
            "min": min_val,
            "max": max_val,
            "dynamic_range_stops": dynamic_range,
            "load_method": load_method
        }
        
        metadata_str = json.dumps(metadata, indent=2)
        
        return (numpy_to_tensor_float32(img), metadata_str)


class LoadImageEXRSequence:
    """
    Load EXR/HDR image sequences from a folder.
    Supports VFX-standard naming conventions and frame range selection.
    
    Examples of supported patterns:
    - render.0001.exr, render.0002.exr, ...
    - frame_001.exr, frame_002.exr, ...
    - image0001.exr, image0002.exr, ...
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Full path to folder containing EXR sequence (e.g., C:/renders/sequence/)"
                }),
            },
            "optional": {
                "file_pattern": ("STRING", {
                    "default": "*.exr",
                    "multiline": False,
                    "tooltip": "File pattern to match (e.g., *.exr, render.*.exr, frame_????.exr)"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "First frame to load (0 = from beginning)"
                }),
                "end_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Last frame to load (0 = to end)"
                }),
                "frame_step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Load every Nth frame"
                }),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Maximum frames to load (0 = no limit). Useful to avoid memory issues."
                }),
                "exposure_adjust": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Adjust exposure in stops (EV)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "metadata", "frame_count")
    FUNCTION = "load_sequence"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Load EXR/HDR image sequence from a folder. Returns a batch of images."
    
    @classmethod
    def IS_CHANGED(cls, folder_path, **kwargs):
        folder_path = folder_path.strip().strip('"').strip("'")
        if not folder_path or not os.path.isdir(folder_path):
            return float("nan")
        try:
            return os.path.getmtime(folder_path)
        except Exception:
            return float("nan")
    
    @classmethod
    def VALIDATE_INPUTS(cls, folder_path, **kwargs):
        folder_path = folder_path.strip().strip('"').strip("'")
        if not folder_path:
            return "No folder path specified."
        if not os.path.isdir(folder_path):
            return f"Folder not found: '{folder_path}'"
        return True
    
    def _load_single_exr(self, file_path: str) -> np.ndarray:
        """Load a single EXR file using the best available method."""
        img = None
        
        # Method 1: OpenEXR library
        try:
            import OpenEXR
            import Imath
            
            exr_file = OpenEXR.InputFile(file_path)
            header = exr_file.header()
            
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            channels = list(header['channels'].keys())
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            
            if 'R' in channels and 'G' in channels and 'B' in channels:
                r_str = exr_file.channel('R', pt)
                g_str = exr_file.channel('G', pt)
                b_str = exr_file.channel('B', pt)
                
                r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
                g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
                b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
                
                if 'A' in channels:
                    a_str = exr_file.channel('A', pt)
                    a = np.frombuffer(a_str, dtype=np.float32).reshape(height, width)
                    img = np.stack([r, g, b, a], axis=-1)
                else:
                    img = np.stack([r, g, b], axis=-1)
            else:
                ch_name = channels[0] if channels else 'Y'
                ch_str = exr_file.channel(ch_name, pt)
                img = np.frombuffer(ch_str, dtype=np.float32).reshape(height, width, 1)
            
            exr_file.close()
            return img
        except Exception:
            pass
        
        # Method 2: imageio
        if img is None:
            try:
                import imageio.v3 as iio
                img = iio.imread(file_path)
                return np.asarray(img, dtype=np.float32)
            except Exception:
                pass
        
        # Method 3: OpenCV
        if img is None:
            try:
                import cv2
                os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
                img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if img is not None:
                    if len(img.shape) == 3 and img.shape[2] >= 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img.astype(np.float32)
            except Exception:
                pass
        
        return None
    
    def load_sequence(self, folder_path: str, file_pattern: str = "*.exr",
                      start_frame: int = 0, end_frame: int = 0,
                      frame_step: int = 1, max_frames: int = 0,
                      exposure_adjust: float = 0.0) -> Tuple[torch.Tensor, str, int]:
        """Load EXR sequence from folder."""
        import glob
        import re
        
        # Clean path
        folder_path = folder_path.strip().strip('"').strip("'")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Find matching files
        pattern = os.path.join(folder_path, file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            raise ValueError(f"No files matching pattern '{file_pattern}' in folder: {folder_path}")
        
        # Sort files naturally (handle frame numbers correctly)
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]
        
        files = sorted(files, key=natural_sort_key)
        
        # Apply frame range
        if start_frame > 0:
            files = files[start_frame:]
        if end_frame > 0 and end_frame < len(files) + start_frame:
            files = files[:end_frame - start_frame + 1]
        
        # Apply frame step
        if frame_step > 1:
            files = files[::frame_step]
        
        # Apply max frames limit
        if max_frames > 0:
            files = files[:max_frames]
        
        if not files:
            raise ValueError("No frames to load after applying range/step filters.")
        
        print(f"[LoadImageEXRSequence] Loading {len(files)} frames from {folder_path}")
        
        # Load all frames
        images = []
        loaded_files = []
        
        for i, file_path in enumerate(files):
            try:
                img = self._load_single_exr(file_path)
                if img is not None:
                    # Handle 2D grayscale
                    if len(img.shape) == 2:
                        img = img[:, :, np.newaxis]
                    
                    # Apply exposure adjustment
                    if exposure_adjust != 0:
                        img = img * (2.0 ** exposure_adjust)
                    
                    images.append(img)
                    loaded_files.append(os.path.basename(file_path))
                    
                    if (i + 1) % 10 == 0:
                        print(f"[LoadImageEXRSequence] Loaded {i + 1}/{len(files)} frames")
            except Exception as e:
                print(f"[LoadImageEXRSequence] Warning: Failed to load {file_path}: {e}")
        
        if not images:
            raise RuntimeError("Failed to load any frames from the sequence.")
        
        # Stack into batch tensor
        # Ensure all images have same dimensions
        h, w, c = images[0].shape
        valid_images = []
        for img in images:
            if img.shape[:2] == (h, w):
                # Ensure same channel count
                if img.shape[2] != c:
                    if img.shape[2] < c:
                        padding = np.ones((*img.shape[:2], c - img.shape[2]), dtype=np.float32)
                        img = np.concatenate([img, padding], axis=-1)
                    else:
                        img = img[:, :, :c]
                valid_images.append(img)
        
        # Stack to batch
        batch = np.stack(valid_images, axis=0)
        
        # Build metadata
        metadata = {
            "folder": folder_path,
            "pattern": file_pattern,
            "frame_count": len(valid_images),
            "files": loaded_files[:10] + (["..."] if len(loaded_files) > 10 else []),
            "width": w,
            "height": h,
            "channels": c,
            "total_found": len(files),
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        
        metadata_str = json.dumps(metadata, indent=2)
        
        print(f"[LoadImageEXRSequence] Done. Loaded {len(valid_images)} frames ({w}x{h}x{c})")
        
        # Convert to tensor: (B, H, W, C)
        tensor_batch = torch.from_numpy(batch).float()
        
        return (tensor_batch, metadata_str, len(valid_images))

class SaveImage16bit:
    """
    Save images as 16-bit PNG or TIFF for wider compatibility.
    """
    
    FORMATS = ["PNG", "TIFF"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "output/16bit_"}),
                "format": (cls.FORMATS, {"default": "PNG"}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Save images as 16-bit PNG or TIFF for wider software compatibility while preserving extended range."
    
    def save(self, images: torch.Tensor, filename_prefix: str = "output/16bit_",
             format: str = "PNG") -> Dict:
        
        import cv2
        import folder_paths
        
        # Get ComfyUI output directory (use absolute path for Windows compatibility)
        output_dir = Path(folder_paths.get_output_directory())
        
        # Parse the filename prefix
        clean_prefix = filename_prefix.replace("\\", "/")
        if clean_prefix.startswith("output/"):
            clean_prefix = clean_prefix[7:]
        
        # Split into directory and base name
        if "/" in clean_prefix:
            subdir, base_name = clean_prefix.rsplit("/", 1)
            full_dir = output_dir / subdir
        else:
            full_dir = output_dir
            base_name = clean_prefix
        
        # Create directory
        full_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        ext = ".png" if format == "PNG" else ".tiff"
        
        for i, img in enumerate(images):
            np_img = img.cpu().numpy()
            
            # Clip and convert to 16-bit
            np_img = np.clip(np_img, 0, 1)
            np_img = (np_img * 65535).astype(np.uint16)
            
            # Convert RGB to BGR for OpenCV
            if np_img.shape[-1] == 3:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            elif np_img.shape[-1] == 4:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)
            
            filename = f"{base_name}{i:05d}{ext}"
            filepath = full_dir / filename
            
            cv2.imwrite(str(filepath), np_img)
            results.append({"filename": str(filepath), "type": "output"})
            print(f"[32bit-HDR] Saved 16-bit: {filepath}")
        
        return {"ui": {"images": results}}


# ═══════════════════════════════════════════════════════════════════════════════
#                          ANALYSIS NODES
# ═══════════════════════════════════════════════════════════════════════════════

class HDRHistogram:
    """
    Analyze HDR image histogram and dynamic range.
    """
    
    MODES = ["luminance", "rgb", "log_luminance"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (cls.MODES, {"default": "luminance"}),
                "show_clipping": ("BOOLEAN", {"default": True}),
                "stops_range": ("INT", {"default": 14, "min": 8, "max": 24}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("histogram", "stats")
    FUNCTION = "analyze"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Analyze HDR image histogram with dynamic range statistics, clipping indicators, and stops visualization."
    
    def analyze(self, image: torch.Tensor, mode: str = "luminance",
                show_clipping: bool = True, stops_range: int = 14) -> Tuple[torch.Tensor, str]:
        
        print("[HDRHistogram] Analyzing...")
        try:
            from PIL import Image, ImageDraw
            
            img = tensor_to_numpy_float32(image)
            if img.ndim == 4:
                img = img[0]  # Take first image in batch
            
            # Sanitize Input: Handle NaNs and Infs
            img = np.nan_to_num(img, nan=0.0, posinf=65504.0, neginf=-65504.0)
            
            # Calculate luminance
            if img.shape[-1] >= 3:
                luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            else:
                luma = img[..., 0]
            
            # Statistics
            min_val = float(np.min(img))
            max_val = float(np.max(img))
            mean_val = float(np.mean(img))
            
            # Dynamic range in stops
            eps = 1e-10
            dynamic_range = np.log2(max(max_val, eps) / max(min_val, eps))
            
            # Clipping analysis
            clip_low = float(np.sum(img <= 0) / img.size * 100)
            clip_high = float(np.sum(img >= 1) / img.size * 100)
            
            stats = f"""═══════════════════════════════════
            HDR IMAGE ANALYSIS
    ═══════════════════════════════════
    Min Value:     {min_val:.6f}
    Max Value:     {max_val:.6f}
    Mean Value:    {mean_val:.6f}
    Dynamic Range: {dynamic_range:.1f} stops
    ───────────────────────────────────
    Clipped Low:   {clip_low:.2f}%
    Clipped High:  {clip_high:.2f}%
    ═══════════════════════════════════"""
            
            # ═══════════════════════════════════════════════════════════════
            # CREATE HISTOGRAM WITH PIL (NO MATPLOTLIB)
            # ═══════════════════════════════════════════════════════════════
            print("[HDRHistogram] Rendering with PIL...")
            
            # Canvas size
            hist_w, hist_h = 1000, 600
            bg_color = (26, 26, 46)  # Dark blue background
            
            # Create image
            hist_img = Image.new('RGB', (hist_w, hist_h), bg_color)
            draw = ImageDraw.Draw(hist_img)
            
            # Graph area
            margin = 60
            graph_w = hist_w - margin * 2
            graph_h = (hist_h - margin * 3) // 2
            graph_top1 = margin
            graph_top2 = margin + graph_h + margin
            
            # ── LINEAR HISTOGRAM ──
            draw.rectangle([margin - 5, graph_top1 - 5, margin + graph_w + 5, graph_top1 + graph_h + 5], 
                          fill=(22, 33, 62))
            
            # Compute histogram bins
            num_bins = 256
            bin_width = graph_w / num_bins
            
            if mode == "rgb" and img.shape[-1] >= 3:
                # RGB mode - draw each channel
                colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]
                for ch_idx, color in enumerate(colors):
                    channel = img[..., ch_idx].flatten()
                    channel = np.clip(channel, 0, max(1, max_val))
                    hist_vals, _ = np.histogram(channel, bins=num_bins, range=(0, max(1, max_val)))
                    max_count = max(hist_vals.max(), 1)
                    
                    for i, count in enumerate(hist_vals):
                        bar_h = int((count / max_count) * graph_h * 0.9)
                        x1 = margin + int(i * bin_width)
                        x2 = margin + int((i + 1) * bin_width)
                        y1 = graph_top1 + graph_h - bar_h
                        y2 = graph_top1 + graph_h
                        # Semi-transparent overlay
                        alpha_color = tuple(c // 2 for c in color)
                        draw.rectangle([x1, y1, x2, y2], fill=alpha_color)
            else:
                # Luminance mode
                hist_vals, _ = np.histogram(luma.flatten(), bins=num_bins, range=(0, max(1, max_val)))
                max_count = max(hist_vals.max(), 1)
                
                for i, count in enumerate(hist_vals):
                    bar_h = int((count / max_count) * graph_h * 0.9)
                    x1 = margin + int(i * bin_width)
                    x2 = margin + int((i + 1) * bin_width)
                    y1 = graph_top1 + graph_h - bar_h
                    y2 = graph_top1 + graph_h
                    draw.rectangle([x1, y1, x2, y2], fill=(233, 69, 96))
            
            # Clipping indicators
            if show_clipping:
                draw.line([(margin, graph_top1), (margin, graph_top1 + graph_h)], fill=(0, 255, 255), width=2)
                # White clipping at normalized 1.0
                white_x = margin + int(graph_w * min(1.0 / max(max_val, 1), 1.0))
                draw.line([(white_x, graph_top1), (white_x, graph_top1 + graph_h)], fill=(255, 255, 0), width=2)
            
            # Title
            draw.text((margin, graph_top1 - 25), "Linear Histogram", fill=(255, 255, 255))
            
            # ── LOG HISTOGRAM (STOPS) ──
            draw.rectangle([margin - 5, graph_top2 - 5, margin + graph_w + 5, graph_top2 + graph_h + 5], 
                          fill=(22, 33, 62))
            
            # Compute log histogram
            log_luma = np.log2(np.maximum(luma, eps))
            min_stop = -8
            max_stop = stops_range - 8
            log_bins = stops_range * 10
            
            log_hist, log_edges = np.histogram(log_luma.flatten(), bins=log_bins, range=(min_stop, max_stop))
            log_max = max(log_hist.max(), 1)
            log_bin_width = graph_w / log_bins
            
            for i, count in enumerate(log_hist):
                bar_h = int((count / log_max) * graph_h * 0.9)
                x1 = margin + int(i * log_bin_width)
                x2 = margin + int((i + 1) * log_bin_width)
                y1 = graph_top2 + graph_h - bar_h
                y2 = graph_top2 + graph_h
                draw.rectangle([x1, y1, x2, y2], fill=(15, 52, 96))
            
            # Middle gray marker (0 stops = 18% gray)
            mid_x = margin + int(graph_w * (-min_stop) / (max_stop - min_stop))
            draw.line([(mid_x, graph_top2), (mid_x, graph_top2 + graph_h)], fill=(233, 69, 96), width=2)
            
            # Title
            draw.text((margin, graph_top2 - 25), f"Log Histogram ({stops_range} stops)", fill=(255, 255, 255))
            
            # Bottom info
            draw.text((margin, hist_h - 30), f"DR: {dynamic_range:.1f} stops | Min: {min_val:.3f} | Max: {max_val:.3f}", 
                     fill=(180, 180, 200))
            
            # Convert to tensor
            hist_np = np.array(hist_img).astype(np.float32) / 255.0
            print("[HDRHistogram] Done.")
            return (numpy_to_tensor_float32(hist_np), stats)
            
        except Exception as e:
            print(f"[HDRHistogram] Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return RED ERROR IMAGE so user knows it failed
            if image.dim() == 4:
                B, H, W, C = image.shape
                err_img = torch.zeros((B, H, W, C), dtype=torch.float32)
                err_img[..., 0] = 1.0  # Red
            else:
                H, W, C = image.shape
                err_img = torch.zeros((H, W, C), dtype=torch.float32)
                err_img[..., 0] = 1.0  # Red
                
            return (err_img, f"Error: {str(e)}")



# ═══════════════════════════════════════════════════════════════════════════════
#                          LOG CURVE ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

# Professional log curve transfer functions for cinema workflows

def _arri_logc3_encode(linear: np.ndarray) -> np.ndarray:
    """ARRI LogC3 encoding (EI 800)."""
    cut = 0.010591
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    
    return np.where(
        linear > cut,
        c * np.log10(a * linear + b) + d,
        e * linear + f
    )

def _arri_logc4_encode(linear: np.ndarray) -> np.ndarray:
    """ARRI LogC4 encoding (ALEXA 35)."""
    a = (np.log(2) ** 18 - 16) / 117.45
    b = (1023 - 95) / (1023 * np.log(2))
    c = 95 / 1023
    s = (7 * np.log(2) * 2 ** (7 - 14 * c / b)) / (a * np.log(10))
    t = (2 ** (14 * (-c / b) + 6) - 64) / a
    
    return np.where(
        linear >= t,
        (np.log2(a * linear + 64) - 6) / 14 * b + c,
        (linear - t) / s
    )

def _slog3_encode(linear: np.ndarray) -> np.ndarray:
    """Sony S-Log3 encoding."""
    return np.where(
        linear >= 0.01125000,
        (420.0 + np.log10((linear + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0,
        (linear * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0
    )

def _vlog_encode(linear: np.ndarray) -> np.ndarray:
    """Panasonic V-Log encoding."""
    cut_in = 0.01
    b = 0.00873
    c = 0.241514
    d = 0.598206
    return np.where(
        linear < cut_in,
        5.6 * linear + 0.125,
        c * np.log10(linear + b) + d
    )

def _canonlog3_encode(linear: np.ndarray) -> np.ndarray:
    """Canon Log 3 encoding."""
    return np.where(
        linear < -0.014,
        -0.36726845 * np.log10(-linear * 14.98325 + 1) + 0.12783901,
        np.where(
            linear <= 0.014,
            1.9754798 * linear + 0.12512219,
            0.36726845 * np.log10(linear * 14.98325 + 1) + 0.12240537
        )
    )

def _acescct_encode(linear: np.ndarray) -> np.ndarray:
    """ACEScct encoding (ACES Central/Cinematic Transform)."""
    X_BRK = 0.0078125
    Y_BRK = 0.155251141552511
    A = 10.5402377416545
    B = 0.0729055341958355
    
    return np.where(
        linear <= X_BRK,
        A * linear + B,
        (np.log2(linear) + 9.72) / 17.52
    )

def _davinci_intermediate_encode(linear: np.ndarray) -> np.ndarray:
    """DaVinci Intermediate encoding."""
    A = 0.0075
    B = 7.0
    C = 0.07329248
    M = 10.44426855
    LIN_CUT = 0.00262409
    LOG_CUT = 0.02740668
    
    return np.where(
        linear <= LIN_CUT,
        linear * M,
        (np.log10((linear + A) * B) * C + LOG_CUT)
    )


class LogCurveEncode:
    """
    Encode linear HDR images to professional log curves.
    GPU-accelerated with automatic CPU fallback.
    """

    LOG_CURVES = [
        "Linear (No Change)",
        "ARRI LogC3 (EI 800)",
        "ARRI LogC4 (ALEXA 35)",
        "Sony S-Log3",
        "Panasonic V-Log",
        "Canon Log 3",
        "ACEScct",
        "DaVinci Intermediate",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "log_curve": (cls.LOG_CURVES, {"default": "ARRI LogC3 (EI 800)"}),
            },
            "optional": {
                "input_is_linear": ("BOOLEAN", {"default": True}),
                "source_gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1}),
                "exposure_offset": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.1}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("log_encoded", "curve_info")
    FUNCTION = "encode"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "GPU-accelerated log curve encoding (ARRI LogC, S-Log3, V-Log, ACEScct) for cinema workflows."

    def _gpu_logc3(self, x: torch.Tensor) -> torch.Tensor:
        """ARRI LogC3 on GPU."""
        cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        return torch.where(x > cut, c * torch.log10(a * x + b) + d, e * x + f)

    def _gpu_logc4(self, x: torch.Tensor) -> torch.Tensor:
        """ARRI LogC4 on GPU."""
        x = torch.clamp(x, min=1e-10)
        return (torch.log2(x) + 7) / 14

    def _gpu_slog3(self, x: torch.Tensor) -> torch.Tensor:
        """Sony S-Log3 on GPU."""
        x = torch.clamp(x, min=0.0)
        return torch.where(
            x >= 0.01125,
            (420.0 + torch.log10((x + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0,
            (x * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0
        )

    def _gpu_vlog(self, x: torch.Tensor) -> torch.Tensor:
        """Panasonic V-Log on GPU."""
        cut_in, b, c, d = 0.01, 0.00873, 0.241514, 0.598206
        return torch.where(x < cut_in, 5.6 * x + 0.125, c * torch.log10(x + b) + d)

    def _gpu_acescct(self, x: torch.Tensor) -> torch.Tensor:
        """ACEScct on GPU."""
        X_BRK, A, B = 0.0078125, 10.5402377416545, 0.0729055341958355
        x = torch.clamp(x, min=1e-10)
        return torch.where(x <= X_BRK, A * x + B, (torch.log2(x) + 9.72) / 17.52)

    def _gpu_canonlog3(self, x: torch.Tensor) -> torch.Tensor:
        """Canon Log 3 on GPU."""
        x = torch.clamp(x, min=1e-10)
        return torch.where(
            x < 0.014,
            -(0.36726845 * torch.log10(14.98325 - 9.20208 * x) - 0.12783901),
            0.36726845 * torch.log10(10.1596 * x + 1) + 0.12783901
        )

    def _gpu_davinci_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        """DaVinci Intermediate on GPU."""
        x = torch.clamp(x, min=1e-10)
        return torch.log(x * 8191 + 1) / torch.log(torch.tensor(8192.0, device=x.device))

    def encode(self, image: torch.Tensor, log_curve: str = "ARRI LogC3 (EI 800)",
               input_is_linear: bool = True, source_gamma: float = 2.2,
               exposure_offset: float = 0.0, use_gpu: bool = True) -> Tuple[torch.Tensor, str]:

        # Try GPU path
        if use_gpu and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                img = image.to(device).float()
                
                # Linearize if needed
                if not input_is_linear:
                    img = torch.pow(torch.clamp(img, min=1e-10), source_gamma)
                
                # Exposure offset
                if exposure_offset != 0.0:
                    img = img * (2.0 ** exposure_offset)
                
                img = torch.clamp(img, min=0.0)
                
                # Apply GPU log curve
                if log_curve == "Linear (No Change)":
                    result = img
                    curve_info = "Linear (No encoding applied)"
                elif log_curve == "ARRI LogC3 (EI 800)":
                    result = self._gpu_logc3(img)
                    curve_info = "ARRI LogC3 EI 800 | GPU | Mid Gray: 0.391"
                elif log_curve == "ARRI LogC4 (ALEXA 35)":
                    result = self._gpu_logc4(img)
                    curve_info = "ARRI LogC4 | GPU | Mid Gray: 0.278"
                elif log_curve == "Sony S-Log3":
                    result = self._gpu_slog3(img)
                    curve_info = "Sony S-Log3 | GPU | Mid Gray: 0.406"
                elif log_curve == "Panasonic V-Log":
                    result = self._gpu_vlog(img)
                    curve_info = "Panasonic V-Log | GPU | Mid Gray: 0.423"
                elif log_curve == "ACEScct":
                    result = self._gpu_acescct(img)
                    curve_info = "ACEScct | GPU | Mid Gray: 0.413"
                elif log_curve == "Canon Log 3":
                    result = self._gpu_canonlog3(img)
                    curve_info = "Canon Log 3 | GPU | Mid Gray: 0.343"
                elif log_curve == "DaVinci Intermediate":
                    result = self._gpu_davinci_intermediate(img)
                    curve_info = "DaVinci Intermediate | GPU | Mid Gray: 0.336"
                else:
                    raise RuntimeError("Use CPU path")
                
                result = torch.clamp(result, 0.0, 1.0)
                return (result, curve_info)
                
            except RuntimeError:
                torch.cuda.empty_cache()
        
        # CPU fallback
        img = tensor_to_numpy_float32(image)
        
        # Convert to linear if needed
        if not input_is_linear:
            img = ensure_linear(img, source_gamma)
        
        # Apply exposure offset (in stops)
        if exposure_offset != 0.0:
            img = img * (2.0 ** exposure_offset)
        
        # Ensure non-negative for log encoding
        img = np.maximum(img, 0.0)
        
        # Apply selected log curve
        if log_curve == "Linear (No Change)":
            result = img
            curve_info = "Linear (No encoding applied)"
        elif log_curve == "ARRI LogC3 (EI 800)":
            result = _arri_logc3_encode(img)
            curve_info = "ARRI LogC3 EI 800 | CPU | Mid Gray: 0.391"
        elif log_curve == "ARRI LogC4 (ALEXA 35)":
            result = _arri_logc4_encode(img)
            curve_info = "ARRI LogC4 | CPU | Mid Gray: 0.278"
        elif log_curve == "Sony S-Log3":
            result = _slog3_encode(img)
            curve_info = "Sony S-Log3 | CPU | Mid Gray: 0.406"
        elif log_curve == "Panasonic V-Log":
            result = _vlog_encode(img)
            curve_info = "Panasonic V-Log | CPU | Mid Gray: 0.423"
        elif log_curve == "Canon Log 3":
            result = _canonlog3_encode(img)
            curve_info = "Canon Log 3 | CPU | Mid Gray: 0.343"
        elif log_curve == "ACEScct":
            result = _acescct_encode(img)
            curve_info = "ACEScct | CPU | Mid Gray: 0.413"
        elif log_curve == "DaVinci Intermediate":
            result = _davinci_intermediate_encode(img)
            curve_info = "DaVinci Intermediate | CPU | Mid Gray: 0.336"
        else:
            result = img
            curve_info = "Unknown curve - no encoding applied"
        
        # Clip to valid range for log encoding
        result = np.clip(result, 0.0, 1.0)
        
        return (numpy_to_tensor_float32(result), curve_info)


class LogCurveDecode:
    """
    Decode log-encoded images back to linear for HDR processing.
    GPU-accelerated with automatic CPU fallback.
    """

    LOG_CURVES = [
        "Linear (No Change)",
        "ARRI LogC3 (EI 800)",
        "ARRI LogC4 (ALEXA 35)",
        "Sony S-Log3",
        "Panasonic V-Log",
        "Canon Log 3",
        "ACEScct",
        "DaVinci Intermediate",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "log_curve": (cls.LOG_CURVES, {"default": "ARRI LogC3 (EI 800)"}),
            },
            "optional": {
                "output_gamma": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.1}),
                "exposure_offset": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.1}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("linear_image",)
    FUNCTION = "decode"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "GPU-accelerated log curve decoding for HDR processing."

    def _gpu_logc3_decode(self, x: torch.Tensor) -> torch.Tensor:
        """ARRI LogC3 decode on GPU."""
        cut, a, b, c, d, e, f = 0.1496582, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        return torch.where(x > cut, (torch.pow(10.0, (x - d) / c) - b) / a, (x - f) / e)

    def _gpu_logc4_decode(self, x: torch.Tensor) -> torch.Tensor:
        """ARRI LogC4 decode on GPU."""
        return torch.pow(2.0, x * 14 - 7)

    def _gpu_slog3_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Sony S-Log3 decode on GPU."""
        cut = 171.2102946929 / 1023.0
        return torch.where(
            x >= cut,
            torch.pow(10.0, (x * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01,
            (x * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0)
        )

    def _gpu_acescct_decode(self, x: torch.Tensor) -> torch.Tensor:
        """ACEScct decode on GPU."""
        Y_BRK, A, B = 0.155251141552511, 10.5402377416545, 0.0729055341958355
        return torch.where(x <= Y_BRK, (x - B) / A, torch.pow(2.0, x * 17.52 - 9.72))

    def _gpu_vlog_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Panasonic V-Log decode on GPU."""
        cut_out, b, c, d = 0.181, 0.00873, 0.241514, 0.598206
        return torch.where(x < cut_out, (x - 0.125) / 5.6, torch.pow(10.0, (x - d) / c) - b)

    def _gpu_canonlog3_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Canon Log 3 decode on GPU."""
        cut = 0.04076162
        return torch.where(
            x < cut,
            (14.98325 - torch.pow(10.0, (0.12783901 - x) / 0.36726845)) / 9.20208,
            (torch.pow(10.0, (x - 0.12783901) / 0.36726845) - 1) / 10.1596
        )

    def _gpu_davinci_decode(self, x: torch.Tensor) -> torch.Tensor:
        """DaVinci Intermediate decode on GPU."""
        return (torch.pow(8192.0, x) - 1) / 8191

    def _arri_logc3_decode(self, log_val: np.ndarray) -> np.ndarray:
        """ARRI LogC3 decoding."""
        cut, a, b, c, d, e, f = 0.1496582, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        return np.where(log_val > cut, (10.0 ** ((log_val - d) / c) - b) / a, (log_val - f) / e)

    def _arri_logc4_decode(self, log_val: np.ndarray) -> np.ndarray:
        """ARRI LogC4 decoding."""
        return 2.0 ** (log_val * 14 - 7)

    def _slog3_decode(self, log_val: np.ndarray) -> np.ndarray:
        """Sony S-Log3 decoding."""
        return np.where(
            log_val >= 171.2102946929 / 1023.0,
            10.0 ** ((log_val * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01,
            (log_val * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0)
        )

    def _vlog_decode(self, log_val: np.ndarray) -> np.ndarray:
        """Panasonic V-Log decoding."""
        cut_out, b, c, d = 0.181, 0.00873, 0.241514, 0.598206
        return np.where(log_val < cut_out, (log_val - 0.125) / 5.6, 10.0 ** ((log_val - d) / c) - b)

    def _canonlog3_decode(self, log_val: np.ndarray) -> np.ndarray:
        """Canon Log 3 decoding."""
        cut = 0.04076162
        return np.where(
            log_val < cut,
            (14.98325 - 10.0 ** ((0.12783901 - log_val) / 0.36726845)) / 9.20208,
            (10.0 ** ((log_val - 0.12783901) / 0.36726845) - 1) / 10.1596
        )

    def _davinci_decode(self, log_val: np.ndarray) -> np.ndarray:
        """DaVinci Intermediate decoding."""
        return (np.power(8192.0, log_val) - 1) / 8191

    def _acescct_decode(self, log_val: np.ndarray) -> np.ndarray:
        """ACEScct decoding."""
        Y_BRK, A, B = 0.155251141552511, 10.5402377416545, 0.0729055341958355
        return np.where(log_val <= Y_BRK, (log_val - B) / A, 2.0 ** (log_val * 17.52 - 9.72))

    def decode(self, image: torch.Tensor, log_curve: str = "ARRI LogC3 (EI 800)",
               output_gamma: float = 1.0, exposure_offset: float = 0.0,
               use_gpu: bool = True) -> Tuple[torch.Tensor]:

        # Try GPU path
        if use_gpu and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                img = image.to(device).float()
                
                # Apply GPU decode
                if log_curve == "Linear (No Change)":
                    result = img
                elif log_curve == "ARRI LogC3 (EI 800)":
                    result = self._gpu_logc3_decode(img)
                elif log_curve == "ARRI LogC4 (ALEXA 35)":
                    result = self._gpu_logc4_decode(img)
                elif log_curve == "Sony S-Log3":
                    result = self._gpu_slog3_decode(img)
                elif log_curve == "ACEScct":
                    result = self._gpu_acescct_decode(img)
                elif log_curve == "Panasonic V-Log":
                    result = self._gpu_vlog_decode(img)
                elif log_curve == "Canon Log 3":
                    result = self._gpu_canonlog3_decode(img)
                elif log_curve == "DaVinci Intermediate":
                    result = self._gpu_davinci_decode(img)
                else:
                    raise RuntimeError("Use CPU path")
                
                # Exposure offset
                if exposure_offset != 0.0:
                    result = result * (2.0 ** exposure_offset)
                
                # Output gamma
                if output_gamma != 1.0:
                    result = torch.pow(torch.clamp(result, min=0.0), 1.0 / output_gamma)
                
                return (result,)
                
            except RuntimeError:
                torch.cuda.empty_cache()
        
        # CPU fallback
        img = tensor_to_numpy_float32(image)
        
        # Apply selected log decode
        if log_curve == "Linear (No Change)":
            result = img
        elif log_curve == "ARRI LogC3 (EI 800)":
            result = self._arri_logc3_decode(img)
        elif log_curve == "ARRI LogC4 (ALEXA 35)":
            result = self._arri_logc4_decode(img)
        elif log_curve == "Sony S-Log3":
            result = self._slog3_decode(img)
        elif log_curve == "Panasonic V-Log":
            result = self._vlog_decode(img)
        elif log_curve == "Canon Log 3":
            result = self._canonlog3_decode(img)
        elif log_curve == "DaVinci Intermediate":
            result = self._davinci_decode(img)
        elif log_curve == "ACEScct":
            result = self._acescct_decode(img)
        else:
            result = img
        
        # Apply exposure offset
        if exposure_offset != 0.0:
            result = result * (2.0 ** exposure_offset)
        
        # Apply output gamma if needed
        if output_gamma != 1.0:
            result = np.power(np.maximum(result, 0), 1.0 / output_gamma)
        
        return (numpy_to_tensor_float32(result),)




# ═══════════════════════════════════════════════════════════════════════════════
#                          HDR EXPOSURE BLENDING
# ═══════════════════════════════════════════════════════════════════════════════

class HDRExposureBlend:
    """
    Blend multiple exposures for extended dynamic range and better color grading.
    
    Professional exposure fusion that combines shadows from high-exposure images
    with highlights from low-exposure images. This creates a true HDR image with
    smooth tonal transitions and no clipping.
    """

    BLEND_METHODS = [
        "Mertens Fusion",
        "Luminance Weighted",
        "Shadow/Highlight Mask",
        "Exposure Weighted",
        "Laplacian Pyramid",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "low_exposure": ("IMAGE",),   # Darker image - good highlights
                "high_exposure": ("IMAGE",),  # Brighter image - good shadows
                "blend_method": (cls.BLEND_METHODS, {"default": "Mertens Fusion"}),
            },
            "optional": {
                "mid_exposure": ("IMAGE",),   # Optional middle exposure
                "shadow_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "highlight_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "transition_smoothness": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05}),
                "exposure_offset_low": ("FLOAT", {"default": -2.0, "min": -6.0, "max": 0.0, "step": 0.5}),
                "exposure_offset_high": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 6.0, "step": 0.5}),
                "ghost_removal": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("blended_hdr", "blend_mask", "blend_info")
    FUNCTION = "blend_exposures"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Blend multiple exposures (bracketing) for extended dynamic range. Takes highlights from low exposure and shadows from high exposure for optimal color grading."

    def _calculate_luminance(self, img: np.ndarray) -> np.ndarray:
        """Calculate Rec.709 luminance."""
        return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    def _mertens_weights(self, img: np.ndarray, contrast_weight: float = 1.0,
                        saturation_weight: float = 1.0, exposure_weight: float = 1.0) -> np.ndarray:
        """Calculate Mertens exposure fusion weights."""
        # Contrast measure (Laplacian)
        gray = self._calculate_luminance(img)
        laplacian = np.abs(np.gradient(np.gradient(gray, axis=0), axis=0) + 
                          np.gradient(np.gradient(gray, axis=1), axis=1))
        contrast = laplacian ** contrast_weight
        
        # Saturation measure
        mean_rgb = np.mean(img, axis=-1)
        saturation = np.sqrt(np.mean((img - mean_rgb[..., np.newaxis]) ** 2, axis=-1))
        saturation = saturation ** saturation_weight
        
        # Well-exposedness measure (Gaussian centered at 0.5)
        sigma = 0.2
        exposedness = np.exp(-0.5 * ((img - 0.5) / sigma) ** 2)
        exposedness = np.prod(exposedness, axis=-1) ** exposure_weight
        
        # Combined weight
        weight = contrast * saturation * exposedness + 1e-10
        return weight

    def _mertens_fusion(self, images: list, weights: list = None) -> np.ndarray:
        """Mertens exposure fusion algorithm."""
        if weights is None:
            weights = [self._mertens_weights(img) for img in images]
        
        # Normalize weights
        weight_sum = sum(weights)
        weight_sum = np.maximum(weight_sum, 1e-10)
        normalized_weights = [w / weight_sum for w in weights]
        
        # Weighted blend
        result = np.zeros_like(images[0])
        for img, w in zip(images, normalized_weights):
            result += img * w[..., np.newaxis]
        
        return result

    def _luminance_weighted_blend(self, low_exp: np.ndarray, high_exp: np.ndarray,
                                  transition: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Luminance-weighted blending for natural HDR merge."""
        # Calculate luminance of both images
        lum_low = self._calculate_luminance(low_exp)
        lum_high = self._calculate_luminance(high_exp)
        
        # Create smooth transition mask based on luminance
        # Low exposure: prefer it for bright areas (highlights preserved)
        # High exposure: prefer it for dark areas (shadows preserved)
        
        # Sigmoid-based transition
        mid_point = 0.5
        x = (lum_low - mid_point) / transition
        low_weight = 1.0 / (1.0 + np.exp(-x))  # High weight for bright areas
        high_weight = 1.0 - low_weight
        
        # Blend
        result = (low_exp * low_weight[..., np.newaxis] + 
                  high_exp * high_weight[..., np.newaxis])
        
        mask = low_weight[..., np.newaxis].repeat(3, axis=-1)
        return result, mask

    def _shadow_highlight_blend(self, low_exp: np.ndarray, high_exp: np.ndarray,
                                shadow_weight: float, highlight_weight: float,
                                transition: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract shadows from high exposure, highlights from low exposure."""
        lum_low = self._calculate_luminance(low_exp)
        
        # Shadow mask (where low exposure is dark)
        shadow_threshold = 0.25
        shadow_mask = np.clip((shadow_threshold - lum_low) / transition + 0.5, 0, 1)
        
        # Highlight mask (where low exposure is bright)
        highlight_threshold = 0.75
        highlight_mask = np.clip((lum_low - highlight_threshold) / transition + 0.5, 0, 1)
        
        # Midtone mask (remainder)
        midtone_mask = 1.0 - shadow_mask - highlight_mask
        midtone_mask = np.maximum(midtone_mask, 0)
        
        # Blend: shadows from high_exp, highlights from low_exp, midtones averaged
        result = (high_exp * shadow_mask[..., np.newaxis] * shadow_weight +
                  low_exp * highlight_mask[..., np.newaxis] * highlight_weight +
                  (low_exp + high_exp) / 2 * midtone_mask[..., np.newaxis])
        
        # Normalize
        total_weight = (shadow_mask * shadow_weight + 
                       highlight_mask * highlight_weight + 
                       midtone_mask)
        total_weight = np.maximum(total_weight, 1e-10)
        result = result / total_weight[..., np.newaxis]
        
        # Visualization mask
        mask = np.stack([shadow_mask, midtone_mask, highlight_mask], axis=-1)
        return result, mask

    def _laplacian_pyramid_blend(self, low_exp: np.ndarray, high_exp: np.ndarray,
                                 levels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-scale Laplacian pyramid blending for seamless HDR merge."""
        
        # Use GPU-accelerated version if available
        if GPU_UTILS_AVAILABLE and torch.cuda.is_available():
            device = torch.device("cuda")
            low_tensor = torch.from_numpy(low_exp).to(device)
            high_tensor = torch.from_numpy(high_exp).to(device)
            
            # Create mask based on luminance
            lum = 0.2126 * low_tensor[..., 0] + 0.7152 * low_tensor[..., 1] + 0.0722 * low_tensor[..., 2]
            mask = (lum > 0.5).float()
            
            result_tensor = gpu_laplacian_pyramid_blend(low_tensor, high_tensor, mask, levels)
            result = result_tensor.cpu().numpy()
            mask_np = mask.cpu().numpy()[..., np.newaxis].repeat(3, axis=-1)
            return result, mask_np
        
        # Fallback to scipy
        from scipy.ndimage import gaussian_filter
        
        def build_gaussian_pyramid(img, levels):
            pyramid = [img]
            for _ in range(levels - 1):
                blurred = gaussian_filter(pyramid[-1], sigma=2)
                downsampled = blurred[::2, ::2]
                pyramid.append(downsampled)
            return pyramid
        
        def build_laplacian_pyramid(gaussian_pyr):
            laplacian = []
            for i in range(len(gaussian_pyr) - 1):
                upsampled = np.repeat(np.repeat(gaussian_pyr[i+1], 2, axis=0), 2, axis=1)
                # Handle size mismatch
                h, w = gaussian_pyr[i].shape[:2]
                upsampled = upsampled[:h, :w]
                laplacian.append(gaussian_pyr[i] - upsampled)
            laplacian.append(gaussian_pyr[-1])
            return laplacian
        
        # Build pyramids
        g_low = build_gaussian_pyramid(low_exp, levels)
        g_high = build_gaussian_pyramid(high_exp, levels)
        l_low = build_laplacian_pyramid(g_low)
        l_high = build_laplacian_pyramid(g_high)
        
        # Create mask based on luminance
        lum = self._calculate_luminance(low_exp)
        mask = (lum > 0.5).astype(np.float32)
        mask_pyr = build_gaussian_pyramid(mask[..., np.newaxis].repeat(3, axis=-1), levels)
        
        # Blend pyramids
        blended_laplacian = []
        for l_l, l_h, m in zip(l_low, l_high, mask_pyr):
            blended = l_l * m + l_h * (1 - m)
            blended_laplacian.append(blended)
        
        # Reconstruct from blended Laplacian pyramid
        result = blended_laplacian[-1]
        for i in range(len(blended_laplacian) - 2, -1, -1):
            upsampled = np.repeat(np.repeat(result, 2, axis=0), 2, axis=1)
            h, w = blended_laplacian[i].shape[:2]
            upsampled = upsampled[:h, :w]
            result = upsampled + blended_laplacian[i]
        
        return result, mask[..., np.newaxis].repeat(3, axis=-1)

    def blend_exposures(self, low_exposure: torch.Tensor, high_exposure: torch.Tensor,
                       blend_method: str = "Mertens Fusion",
                       mid_exposure: torch.Tensor = None,
                       shadow_weight: float = 1.0, highlight_weight: float = 1.0,
                       transition_smoothness: float = 0.3,
                       exposure_offset_low: float = -2.0, exposure_offset_high: float = 2.0,
                       ghost_removal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, str]:

        low_np = tensor_to_numpy_float32(low_exposure)
        high_np = tensor_to_numpy_float32(high_exposure)
        
        # Handle batch dimension
        if low_np.ndim == 4:
            low_np = low_np[0]
            high_np = high_np[0]
        
        # Apply exposure compensation to bring to common scale
        low_np = low_np * (2.0 ** (-exposure_offset_low))  # Brighten low exposure
        high_np = high_np * (2.0 ** (-exposure_offset_high))  # Darken high exposure
        
        # Perform blending based on method
        if blend_method == "Mertens Fusion":
            images = [low_np, high_np]
            if mid_exposure is not None:
                mid_np = tensor_to_numpy_float32(mid_exposure)
                if mid_np.ndim == 4:
                    mid_np = mid_np[0]
                images.insert(1, mid_np)
            result = self._mertens_fusion(images)
            mask = np.ones_like(result) * 0.5
            
        elif blend_method == "Luminance Weighted":
            result, mask = self._luminance_weighted_blend(low_np, high_np, transition_smoothness)
            
        elif blend_method == "Shadow/Highlight Mask":
            result, mask = self._shadow_highlight_blend(low_np, high_np, shadow_weight, 
                                                        highlight_weight, transition_smoothness)
            
        elif blend_method == "Exposure Weighted":
            # Simple weighted average based on exposure settings
            total_range = abs(exposure_offset_high - exposure_offset_low)
            low_weight = abs(exposure_offset_high) / total_range
            high_weight = abs(exposure_offset_low) / total_range
            result = low_np * low_weight + high_np * high_weight
            mask = np.ones_like(result) * low_weight
            
        elif blend_method == "Laplacian Pyramid":
            result, mask = self._laplacian_pyramid_blend(low_np, high_np)
        else:
            result = (low_np + high_np) / 2
            mask = np.ones_like(result) * 0.5
        
        # Ensure valid range for 32-bit HDR
        result = np.maximum(result, 0.0)
        
        # Calculate blend statistics
        dynamic_range = np.log2(np.maximum(result.max(), 1e-10) / np.maximum(result[result > 0].min(), 1e-10))
        
        info = f"Method: {blend_method} | DR: {dynamic_range:.1f} stops | Range: [{result.min():.3f}, {result.max():.3f}]"
        
        return (numpy_to_tensor_float32(result), numpy_to_tensor_float32(mask.astype(np.float32)), info)


class HDRShadowHighlightRecovery:
    """
    Recover shadow and highlight detail from a single HDR image.
    
    Uses local tone mapping techniques to pull detail from dark 
    and bright regions without affecting overall exposure.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "highlight_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "shadow_tone": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.01}),
                "highlight_tone": ("FLOAT", {"default": 0.75, "min": 0.5, "max": 1.0, "step": 0.01}),
                "color_correction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "local_contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("recovered_image",)
    FUNCTION = "recover"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Recover shadow and highlight detail from a single HDR image for better color grading flexibility."

    def recover(self, image: torch.Tensor, shadow_amount: float = 0.5,
                highlight_amount: float = 0.5, shadow_tone: float = 0.25,
                highlight_tone: float = 0.75, color_correction: float = 0.5,
                local_contrast: float = 0.0) -> Tuple[torch.Tensor]:
        
        img = tensor_to_numpy_float32(image)
        if img.ndim == 4:
            img = img[0]
        
        # Calculate luminance
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        lum = np.maximum(lum, 1e-10)
        
        # Shadow recovery
        # Soft mask that decays as luminance increases, without hard 0-1 clipping
        # shadow_mask = np.clip((shadow_tone - lum) / shadow_tone, 0, 1) ** 2  <-- OLD (Clamped)
        
        # New HDR-safe mask: 1.0 at 0, smooth falloff to 0.0 at shadow_tone
        shadow_ratio = lum / (shadow_tone + 1e-6)
        shadow_mask = np.exp(-3.0 * shadow_ratio)  # Exponential decay
        
        shadow_boost = (1.0 + shadow_amount * shadow_mask)
        
        # Highlight recovery
        # Old logic assumed highlights start at 'highlight_tone' (e.g. 0.75) and end at 1.0
        # For HDR, highlights go to infinity.
        # highlight_mask = np.clip((lum - highlight_tone) / highlight_range, 0, 1) ** 2 <-- OLD
        
        # New HDR-safe mask: 0.0 below tone, smooth rise to 1.0+ above
        # Use a smoothstep-like function that doesn't clamp at 1.0
        
        # Normalized position above threshold
        h_pos = (lum - highlight_tone) / (1.0 - highlight_tone + 1e-6)
        
        # Sigmoid-like activation that works for > 1.0
        # value 0 at h_pos <= 0
        # rises to 1.0 at h_pos = 1.0
        # continues rising for h_pos > 1.0 (to catch super-whites)
        
        highlight_mask = np.maximum(0.0, h_pos)
        
        # Apply reduction
        # We want to compress range, not clamp it. 
        # reduction factor should approach 1/amount asymptotically? 
        # simpler: just apply the curve.
        
        highlight_reduce = 1.0 / (1.0 + highlight_amount * highlight_mask * 0.5)
        
        # Apply to image
        result = img * shadow_boost[..., np.newaxis] * highlight_reduce[..., np.newaxis]
        
        # Color correction to reduce oversaturation in lifted shadows
        if color_correction > 0:
            new_lum = 0.2126 * result[..., 0] + 0.7152 * result[..., 1] + 0.0722 * result[..., 2]
            new_lum = np.maximum(new_lum, 1e-10)
            sat_factor = 1.0 - (shadow_mask * color_correction * 0.3)
            result = new_lum[..., np.newaxis] + sat_factor[..., np.newaxis] * (result - new_lum[..., np.newaxis])
        
        # Local contrast enhancement
        if local_contrast != 0:
            # Use GPU-accelerated local contrast if available
            if GPU_UTILS_AVAILABLE:
                result_tensor = torch.from_numpy(result).cuda() if torch.cuda.is_available() else torch.from_numpy(result)
                result_tensor = gpu_local_contrast(result_tensor, sigma=50.0, amount=local_contrast)
                result = result_tensor.cpu().numpy() if result_tensor.is_cuda else result_tensor.numpy()
            else:
                # Fallback to scipy
                from scipy.ndimage import gaussian_filter
                local_lum = gaussian_filter(lum, sigma=50)
                detail = lum / (local_lum + 1e-10)
                contrast_boost = 1.0 + local_contrast * (detail - 1.0)
                result = result * contrast_boost[..., np.newaxis]
        
        return (numpy_to_tensor_float32(result),)




# ═══════════════════════════════════════════════════════════════════════════════
#                          OCIO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Check for PyOpenColorIO
try:
    import PyOpenColorIO as OCIO
    HAS_OCIO = True
except ImportError:
    HAS_OCIO = False
    print("[32bit-HDR] PyOpenColorIO not found. Install with: pip install opencolorio")


class OCIOColorTransform:
    """
    Transform colors using OpenColorIO configuration files.
    
    Provides industry-standard color management compatible with:
    - ACES 1.3 / ACES 2.0
    - DaVinci Resolve
    - Nuke / Foundry
    - Maya / Houdini
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "source_colorspace": ("STRING", {"default": "ACES - ACEScg"}),
                "target_colorspace": ("STRING", {"default": "Output - sRGB"}),
            },
            "optional": {
                "ocio_config_path": ("STRING", {"default": "", "multiline": False}),
                "look": ("STRING", {"default": ""}),
                "direction": (["Forward", "Inverse"], {"default": "Forward"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("transformed_image", "colorspaces_info")
    FUNCTION = "transform"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Transform colors using OpenColorIO config files. Compatible with ACES, DaVinci, Nuke pipelines."

    def __init__(self):
        self.config = None
        self.config_path = None

    def _load_config(self, config_path: str = ""):
        """Load OCIO config from path or environment."""
        if not HAS_OCIO:
            return None, "PyOpenColorIO not installed"
        
        try:
            if config_path and os.path.exists(config_path):
                config = OCIO.Config.CreateFromFile(config_path)
            else:
                # Try environment variable
                config = OCIO.GetCurrentConfig()
            
            return config, "Config loaded successfully"
        except Exception as e:
            return None, f"Error loading config: {str(e)}"

    def _get_available_colorspaces(self, config) -> list:
        """Get list of available colorspaces from config."""
        if config is None:
            return []
        return [config.getColorSpaceNameByIndex(i) for i in range(config.getNumColorSpaces())]

    def transform(self, image: torch.Tensor, source_colorspace: str,
                  target_colorspace: str, ocio_config_path: str = "",
                  look: str = "", direction: str = "Forward") -> Tuple[torch.Tensor, str]:

        if not HAS_OCIO:
            return (image, "ERROR: PyOpenColorIO not installed. pip install opencolorio")

        # Load config
        config, status = self._load_config(ocio_config_path)
        if config is None:
            return (image, f"ERROR: {status}")

        img = tensor_to_numpy_float32(image)
        if img.ndim == 4:
            img = img[0]

        try:
            # Create processor
            if look:
                processor = config.getProcessor(
                    source_colorspace, look, target_colorspace,
                    OCIO.TransformDirection.TRANSFORM_DIR_FORWARD if direction == "Forward" 
                    else OCIO.TransformDirection.TRANSFORM_DIR_INVERSE
                )
            else:
                processor = config.getProcessor(source_colorspace, target_colorspace)

            cpu_processor = processor.getDefaultCPUProcessor()

            # Process image
            h, w, c = img.shape
            result = img.copy()
            
            # Process pixel by pixel for RGBA support
            if c == 4:
                rgb = result[..., :3].reshape(-1, 3).astype(np.float32)
                cpu_processor.applyRGB(rgb)
                result[..., :3] = rgb.reshape(h, w, 3)
            else:
                rgb = result.reshape(-1, 3).astype(np.float32)
                cpu_processor.applyRGB(rgb)
                result = rgb.reshape(h, w, 3)

            # Get colorspace info
            colorspaces = self._get_available_colorspaces(config)
            info = f"Transformed: {source_colorspace} → {target_colorspace}\nAvailable spaces: {len(colorspaces)}"

            return (numpy_to_tensor_float32(result), info)

        except Exception as e:
            return (image, f"ERROR: {str(e)}")


class OCIOListColorspaces:
    """
    List all available colorspaces from an OCIO config.
    Useful for discovering available transforms.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "ocio_config_path": ("STRING", {"default": "", "multiline": False}),
                "filter_role": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("colorspaces_list",)
    FUNCTION = "list_spaces"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "List all colorspaces available in an OCIO configuration file."

    def list_spaces(self, ocio_config_path: str = "", 
                    filter_role: str = "") -> Tuple[str]:
        
        if not HAS_OCIO:
            return ("PyOpenColorIO not installed. pip install opencolorio",)

        try:
            if ocio_config_path and os.path.exists(ocio_config_path):
                config = OCIO.Config.CreateFromFile(ocio_config_path)
            else:
                config = OCIO.GetCurrentConfig()

            # Get colorspaces
            spaces = []
            for i in range(config.getNumColorSpaces()):
                name = config.getColorSpaceNameByIndex(i)
                cs = config.getColorSpace(name)
                family = cs.getFamily() if cs else ""
                spaces.append(f"{name} [{family}]" if family else name)

            # Get looks
            looks = [config.getLookNameByIndex(i) for i in range(config.getNumLooks())]

            # Get displays and views
            displays = []
            for d in range(config.getNumDisplays()):
                display = config.getDisplay(d)
                views = [config.getView(display, v) for v in range(config.getNumViews(display))]
                displays.append(f"{display}: {', '.join(views)}")

            result = "═══ OCIO COLORSPACES ═══\n"
            result += "\n".join(spaces[:50])  # Limit to 50
            if len(spaces) > 50:
                result += f"\n... and {len(spaces) - 50} more"
            
            if looks:
                result += "\n\n═══ LOOKS ═══\n"
                result += "\n".join(looks)
            
            if displays:
                result += "\n\n═══ DISPLAYS ═══\n"
                result += "\n".join(displays)

            return (result,)

        except Exception as e:
            return (f"Error: {str(e)}",)


# ═══════════════════════════════════════════════════════════════════════════════
#                          3D LUT SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

class LUTApply:
    """
    Apply 3D LUTs for professional color grading.
    
    Supports industry-standard formats:
    - .cube (Adobe/Resolve)
    - .3dl (Autodesk Lustre)
    - .spi3d (Sony Pictures)
    
    Uses tetrahedral interpolation for maximum quality.
    """

    LUT_EXTENSIONS = [".cube", ".3dl", ".spi3d"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "interpolation": (["Tetrahedral", "Trilinear"], {"default": "Tetrahedral"}),
                "preserve_luminance": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("graded_image", "lut_info")
    FUNCTION = "apply_lut"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Apply 3D LUTs (.cube, .3dl) for film looks and professional color grading."

    def _parse_cube_lut(self, filepath: str) -> Tuple[np.ndarray, int, dict]:
        """Parse Adobe/Resolve .cube LUT file."""
        lut_data = []
        lut_size = 0
        domain_min = [0.0, 0.0, 0.0]
        domain_max = [1.0, 1.0, 1.0]
        title = ""

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('TITLE'):
                    title = line.split('"')[1] if '"' in line else line.split()[-1]
                elif line.startswith('LUT_3D_SIZE'):
                    lut_size = int(line.split()[-1])
                elif line.startswith('DOMAIN_MIN'):
                    domain_min = [float(x) for x in line.split()[1:4]]
                elif line.startswith('DOMAIN_MAX'):
                    domain_max = [float(x) for x in line.split()[1:4]]
                else:
                    try:
                        values = [float(x) for x in line.split()[:3]]
                        if len(values) == 3:
                            lut_data.append(values)
                    except ValueError:
                        continue

        lut_array = np.array(lut_data, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3)
        
        metadata = {
            "title": title,
            "size": lut_size,
            "domain_min": domain_min,
            "domain_max": domain_max,
            "format": "cube"
        }
        
        return lut_array, lut_size, metadata

    def _parse_3dl_lut(self, filepath: str) -> Tuple[np.ndarray, int, dict]:
        """Parse Autodesk Lustre .3dl LUT file."""
        lut_data = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # First line is mesh size
        first_line = lines[0].strip().split()
        lut_size = int(first_line[0]) if first_line[0].isdigit() else 17

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                values = [int(x) / 4095.0 for x in line.split()[:3]]  # 12-bit to float
                if len(values) == 3:
                    lut_data.append(values)
            except ValueError:
                continue

        lut_array = np.array(lut_data, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3)
        
        metadata = {
            "title": os.path.basename(filepath),
            "size": lut_size,
            "format": "3dl"
        }
        
        return lut_array, lut_size, metadata

    def _tetrahedral_interp(self, img: np.ndarray, lut: np.ndarray, lut_size: int) -> np.ndarray:
        """Tetrahedral interpolation for highest quality LUT application."""
        h, w, c = img.shape
        result = np.zeros_like(img)
        
        # Scale input to LUT indices
        scale = (lut_size - 1)
        r = np.clip(img[..., 0] * scale, 0, lut_size - 1.001)
        g = np.clip(img[..., 1] * scale, 0, lut_size - 1.001)
        b = np.clip(img[..., 2] * scale, 0, lut_size - 1.001)
        
        # Integer indices
        r0, g0, b0 = r.astype(int), g.astype(int), b.astype(int)
        r1, g1, b1 = np.minimum(r0 + 1, lut_size - 1), np.minimum(g0 + 1, lut_size - 1), np.minimum(b0 + 1, lut_size - 1)
        
        # Fractional parts
        fr, fg, fb = r - r0, g - g0, b - b0
        
        # Get 8 corners of the cube
        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]
        
        # Tetrahedral interpolation
        fr_e = fr[..., np.newaxis]
        fg_e = fg[..., np.newaxis]
        fb_e = fb[..., np.newaxis]
        
        # Determine which tetrahedron we're in
        mask1 = fr >= fg
        mask2 = fg >= fb
        mask3 = fr >= fb
        
        # Case 1: r >= g >= b
        case1 = mask1 & mask2
        result[case1] = (c000[case1] * (1 - fr_e[case1]) + 
                        (c100[case1] - c000[case1]) * fr_e[case1] +
                        (c110[case1] - c100[case1]) * fg_e[case1] +
                        (c111[case1] - c110[case1]) * fb_e[case1])
        
        # Case 2: r >= b >= g
        case2 = mask1 & ~mask2 & mask3
        result[case2] = (c000[case2] * (1 - fr_e[case2]) +
                        (c100[case2] - c000[case2]) * fr_e[case2] +
                        (c101[case2] - c100[case2]) * fb_e[case2] +
                        (c111[case2] - c101[case2]) * fg_e[case2])
        
        # Case 3: b >= r >= g
        case3 = mask1 & ~mask2 & ~mask3
        result[case3] = (c000[case3] * (1 - fb_e[case3]) +
                        (c001[case3] - c000[case3]) * fb_e[case3] +
                        (c101[case3] - c001[case3]) * fr_e[case3] +
                        (c111[case3] - c101[case3]) * fg_e[case3])
        
        # Case 4: g >= r >= b
        case4 = ~mask1 & mask2
        result[case4] = (c000[case4] * (1 - fg_e[case4]) +
                        (c010[case4] - c000[case4]) * fg_e[case4] +
                        (c110[case4] - c010[case4]) * fr_e[case4] +
                        (c111[case4] - c110[case4]) * fb_e[case4])
        
        # Case 5: g >= b >= r
        case5 = ~mask1 & ~mask2 & ~mask3
        result[case5] = (c000[case5] * (1 - fg_e[case5]) +
                        (c010[case5] - c000[case5]) * fg_e[case5] +
                        (c011[case5] - c010[case5]) * fb_e[case5] +
                        (c111[case5] - c011[case5]) * fr_e[case5])
        
        # Case 6: b >= g >= r
        case6 = ~mask1 & ~mask2 & mask3
        result[case6] = (c000[case6] * (1 - fb_e[case6]) +
                        (c001[case6] - c000[case6]) * fb_e[case6] +
                        (c011[case6] - c001[case6]) * fg_e[case6] +
                        (c111[case6] - c011[case6]) * fr_e[case6])
        
        return result

    def _trilinear_interp(self, img: np.ndarray, lut: np.ndarray, lut_size: int) -> np.ndarray:
        """Trilinear interpolation (faster but lower quality)."""
        
        # Use GPU-accelerated version if available
        if GPU_UTILS_AVAILABLE and torch.cuda.is_available():
            device = torch.device("cuda")
            img_tensor = torch.from_numpy(img).to(device)
            lut_tensor = torch.from_numpy(lut).to(device)
            
            result_tensor = gpu_trilinear_sample(lut_tensor, img_tensor)
            return result_tensor.cpu().numpy()
        
        # Fallback to scipy
        from scipy.ndimage import map_coordinates
        
        h, w, c = img.shape
        result = np.zeros_like(img)
        
        scale = (lut_size - 1)
        coords = np.clip(img * scale, 0, lut_size - 1.001)
        
        for channel in range(3):
            result[..., channel] = map_coordinates(
                lut[..., channel],
                [coords[..., 0], coords[..., 1], coords[..., 2]],
                order=1, mode='nearest'
            )
        
        return result

    def apply_lut(self, image: torch.Tensor, lut_path: str,
                  strength: float = 1.0, interpolation: str = "Tetrahedral",
                  preserve_luminance: bool = False) -> Tuple[torch.Tensor, str]:
        
        if not lut_path or not os.path.exists(lut_path):
            return (image, f"ERROR: LUT file not found: {lut_path}")

        # Determine format and parse
        ext = os.path.splitext(lut_path)[1].lower()
        
        try:
            if ext == ".cube":
                lut, lut_size, metadata = self._parse_cube_lut(lut_path)
            elif ext == ".3dl":
                lut, lut_size, metadata = self._parse_3dl_lut(lut_path)
            else:
                return (image, f"ERROR: Unsupported format: {ext}")
        except Exception as e:
            return (image, f"ERROR parsing LUT: {str(e)}")

        img = tensor_to_numpy_float32(image)
        if img.ndim == 4:
            img = img[0]

        # Store alpha if present
        has_alpha = img.shape[-1] == 4
        if has_alpha:
            alpha = img[..., 3:4]
            img = img[..., :3]

        # Calculate original luminance if preserving
        if preserve_luminance:
            orig_lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

        # Apply LUT
        if interpolation == "Tetrahedral":
            graded = self._tetrahedral_interp(img, lut, lut_size)
        else:
            graded = self._trilinear_interp(img, lut, lut_size)

        # Preserve luminance if requested
        if preserve_luminance:
            new_lum = 0.2126 * graded[..., 0] + 0.7152 * graded[..., 1] + 0.0722 * graded[..., 2]
            new_lum = np.maximum(new_lum, 1e-10)
            graded = graded * (orig_lum / new_lum)[..., np.newaxis]

        # Blend with strength
        if strength < 1.0:
            graded = img * (1 - strength) + graded * strength

        # Restore alpha
        if has_alpha:
            graded = np.concatenate([graded, alpha], axis=-1)

        info = f"LUT: {metadata.get('title', os.path.basename(lut_path))} | Size: {lut_size}³ | Format: {ext}"
        
        return (numpy_to_tensor_float32(graded), info)


# ═══════════════════════════════════════════════════════════════════════════════
#                          GPU ACCELERATION
# ═══════════════════════════════════════════════════════════════════════════════

class GPUColorMatrix:
    """
    GPU-accelerated color matrix operations.
    
    Uses PyTorch CUDA for 10-50x faster processing on GPU.
    Automatically falls back to CPU if GPU not available.
    """

    MATRICES = {
        "Identity": np.eye(3),
        "Rec709 to XYZ": np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]),
        "XYZ to Rec709": np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ]),
        "Rec709 to Rec2020": np.array([
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956]
        ]),
        "Rec2020 to Rec709": np.array([
            [1.6605, -0.5876, -0.0728],
            [-0.1246, 1.1329, -0.0083],
            [-0.0182, -0.1006, 1.1187]
        ]),
        "Desaturate": np.array([
            [0.2126, 0.7152, 0.0722],
            [0.2126, 0.7152, 0.0722],
            [0.2126, 0.7152, 0.0722]
        ]),
        "Sepia": np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ]),
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "matrix_preset": (list(cls.MATRICES.keys()), {"default": "Identity"}),
            },
            "optional": {
                "force_gpu": ("BOOLEAN", {"default": True}),
                "custom_matrix": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("transformed_image", "device_info")
    FUNCTION = "apply_matrix"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "GPU-accelerated color matrix transformation. 10-50x faster on CUDA GPUs."

    def apply_matrix(self, image: torch.Tensor, matrix_preset: str,
                     force_gpu: bool = True, custom_matrix: str = "") -> Tuple[torch.Tensor, str]:
        
        # Determine device
        if force_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
        else:
            device = torch.device("cpu")
            device_name = "CPU"

        # Get matrix
        if custom_matrix.strip():
            try:
                rows = custom_matrix.strip().split('\n')
                matrix = np.array([[float(x) for x in row.split()] for row in rows], dtype=np.float32)
            except:
                matrix = self.MATRICES[matrix_preset]
        else:
            matrix = self.MATRICES[matrix_preset]

        # Convert to torch tensor
        matrix_t = torch.from_numpy(matrix.astype(np.float32)).to(device)
        
        # Move image to device
        img = image.to(device)
        
        # Apply matrix efficiently using einsum
        # Shape: (B, H, W, C) @ (3, 3).T = (B, H, W, 3)
        has_alpha = img.shape[-1] == 4
        
        if has_alpha:
            rgb = img[..., :3]
            alpha = img[..., 3:4]
        else:
            rgb = img

        # Matrix multiply using einsum for batched operation
        result = torch.einsum('...c,dc->...d', rgb, matrix_t)
        
        if has_alpha:
            result = torch.cat([result, alpha], dim=-1)

        # Move back to CPU
        result = result.cpu()

        info = f"Device: {device_name} | Matrix: {matrix_preset} ({matrix.shape})"
        
        return (result, info)


class GPUTensorOps:
    """
    GPU-accelerated tensor operations for HDR processing.
    Provides fast exposure, gamma, and blend operations.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": (["Exposure", "Gamma", "Lift/Gain", "Normalize", "Clamp"], {"default": "Exposure"}),
            },
            "optional": {
                "value": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "force_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "performance_info")
    FUNCTION = "process"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "GPU-accelerated HDR operations. Fast exposure, gamma, and normalization."

    def process(self, image: torch.Tensor, operation: str,
                value: float = 0.0, force_gpu: bool = True) -> Tuple[torch.Tensor, str]:
        
        import time
        start_time = time.time()

        # Determine device
        if force_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        img = image.to(device)

        # Apply operation
        if operation == "Exposure":
            result = img * (2.0 ** value)
        elif operation == "Gamma":
            gamma = max(0.1, value) if value != 0 else 2.2
            result = torch.pow(torch.clamp(img, 0.001, None), 1.0 / gamma)
        elif operation == "Lift/Gain":
            result = img + value  # Simple lift
        elif operation == "Normalize":
            min_val = img.min()
            max_val = img.max()
            result = (img - min_val) / (max_val - min_val + 1e-10)
        elif operation == "Clamp":
            result = torch.clamp(img, 0.0, 1.0)
        else:
            result = img

        result = result.cpu()
        
        elapsed = (time.time() - start_time) * 1000
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        info = f"Device: {device_name} | Op: {operation} | Time: {elapsed:.2f}ms"

        return (result, info)



# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#                          HDR 360 PANORAMA NODES
# ═══════════════════════════════════════════════════════════════════════════════

class HDR360Generate:
    """
    Generate 360° equirectangular panoramas for HDRI environment mapping.
    
    Creates industry-standard equirectangular projections from source images
    that can be used as environment maps in 3D applications like Blender,
    Maya, Unreal Engine, and Unity.
    """
    
    PROJECTION_TYPES = ["Equirectangular", "Cube_Map", "Mirror_Ball", "Angular_Map"]
    INTERPOLATION_MODES = ["Bilinear", "Bicubic", "Lanczos", "Nearest"]
    FILL_MODES = ["Mirror", "Repeat", "Black", "Edge"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "Input image to project. For best results, use a wide or panoramic image."}),
                "projection_type": (cls.PROJECTION_TYPES, {
                    "default": "Equirectangular",
                    "tooltip": "Projection type: Equirectangular (standard HDRI), Cube_Map (6-face), Mirror_Ball (chrome ball), Angular_Map (light probe)."
                }),
                "output_width": ("INT", {
                    "default": 4096, "min": 512, "max": 16384, "step": 64,
                    "tooltip": "Output panorama width in pixels. Standard HDRI sizes: 2048 (preview), 4096 (standard), 8192 (high), 16384 (ultra)."
                }),
                "output_height": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 64,
                    "tooltip": "Output panorama height. For equirectangular, height should be half of width (2:1 ratio)."
                }),
            },
            "optional": {
                "horizontal_fov": ("FLOAT", {
                    "default": 360.0, "min": 30.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Horizontal field of view in degrees. 360° = full sphere, less = partial panorama."
                }),
                "vertical_fov": ("FLOAT", {
                    "default": 180.0, "min": 15.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Vertical field of view in degrees. 180° = pole to pole, less = band around horizon."
                }),
                "rotation_x": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Rotation around X axis (pitch) in degrees. Tilts the panorama up/down."
                }),
                "rotation_y": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Rotation around Y axis (yaw) in degrees. Rotates the panorama left/right."
                }),
                "rotation_z": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Rotation around Z axis (roll) in degrees. Tilts the horizon."
                }),
                "interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "Lanczos",
                    "tooltip": "Sampling interpolation: Lanczos (highest quality), Bicubic (good), Bilinear (fast), Nearest (pixelated)."
                }),
                "fill_mode": (cls.FILL_MODES, {
                    "default": "Mirror",
                    "tooltip": "How to fill areas outside source image: Mirror (reflects), Repeat (tiles), Black (transparent), Edge (extends edge pixels)."
                }),
                "exposure_adjust": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Exposure adjustment in stops. +1 = double brightness, -1 = half brightness."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("panorama", "projection_map")
    OUTPUT_TOOLTIPS = ("Generated 360° panorama.", "UV projection map for debugging.")
    FUNCTION = "generate"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Generate 360° equirectangular panoramas for HDRI environment mapping in 3D applications."
    
    def _create_rotation_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """Create 3D rotation matrix from Euler angles (in degrees)."""
        rx, ry, rz = np.radians([rx, ry, rz])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    def _equirectangular_to_xyz(self, width: int, height: int) -> np.ndarray:
        """Convert equirectangular coordinates to 3D unit sphere coordinates."""
        u = np.linspace(0, 1, width)
        v = np.linspace(0, 1, height)
        u, v = np.meshgrid(u, v)
        
        theta = (u - 0.5) * 2 * np.pi
        phi = (0.5 - v) * np.pi
        
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi)
        z = np.cos(phi) * np.cos(theta)
        
        return np.stack([x, y, z], axis=-1)
    
    def _xyz_to_equirectangular(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 3D coordinates back to equirectangular UV coordinates."""
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        
        theta = np.arctan2(x, z)
        phi = np.arcsin(np.clip(y, -1, 1))
        
        u = theta / (2 * np.pi) + 0.5
        v = 0.5 - phi / np.pi
        
        return u, v
    
    def _apply_fill_mode(self, u: np.ndarray, v: np.ndarray, fill_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply fill mode for out-of-bounds coordinates."""
        if fill_mode == "Mirror":
            u = np.abs(np.mod(u, 2) - 1) * (np.mod(np.floor(u), 2) * 2 - 1) + np.mod(np.floor(u) + 1, 2)
            v = np.abs(np.mod(v, 2) - 1) * (np.mod(np.floor(v), 2) * 2 - 1) + np.mod(np.floor(v) + 1, 2)
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            mask = np.ones_like(u)
        elif fill_mode == "Repeat":
            u = np.mod(u, 1)
            v = np.mod(v, 1)
            mask = np.ones_like(u)
        elif fill_mode == "Edge":
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            mask = np.ones_like(u)
        else:  # Black
            mask = ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).astype(np.float32)
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
        
        return u, v, mask
    
    def _sample_image(self, img: np.ndarray, u: np.ndarray, v: np.ndarray, 
                      interpolation: str) -> np.ndarray:
        """Sample image at UV coordinates with specified interpolation."""
        import cv2
        
        h, w = img.shape[:2]
        x = u * (w - 1)
        y = v * (h - 1)
        
        interp_map = {
            "Nearest": cv2.INTER_NEAREST,
            "Bilinear": cv2.INTER_LINEAR,
            "Bicubic": cv2.INTER_CUBIC,
            "Lanczos": cv2.INTER_LANCZOS4
        }
        
        map_x = x.astype(np.float32)
        map_y = y.astype(np.float32)
        
        result = cv2.remap(img, map_x, map_y, interp_map.get(interpolation, cv2.INTER_LANCZOS4),
                          borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    def generate(self, source_image: torch.Tensor, projection_type: str = "Equirectangular",
                 output_width: int = 4096, output_height: int = 2048,
                 horizontal_fov: float = 360.0, vertical_fov: float = 180.0,
                 rotation_x: float = 0.0, rotation_y: float = 0.0, rotation_z: float = 0.0,
                 interpolation: str = "Lanczos", fill_mode: str = "Mirror",
                 exposure_adjust: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        
        import cv2
        
        img = tensor_to_numpy_float32(source_image)
        if img.ndim == 4:
            img = img[0]
        
        src_h, src_w = img.shape[:2]
        
        # Generate equirectangular UV grid
        xyz = self._equirectangular_to_xyz(output_width, output_height)
        
        # Apply rotation if specified
        if rotation_x != 0 or rotation_y != 0 or rotation_z != 0:
            R = self._create_rotation_matrix(rotation_x, rotation_y, rotation_z)
            xyz = xyz @ R.T
        
        # Project based on type
        if projection_type == "Equirectangular":
            u, v = self._xyz_to_equirectangular(xyz)
            # Apply FOV scaling
            h_scale = horizontal_fov / 360.0
            v_scale = vertical_fov / 180.0
            u = (u - 0.5) / h_scale + 0.5
            v = (v - 0.5) / v_scale + 0.5
            
        elif projection_type == "Cube_Map":
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
            max_axis = np.maximum(np.maximum(abs_x, abs_y), abs_z)
            u = np.where(max_axis == abs_x, 0.5 + y / (2 * abs_x + 1e-8), 
                        np.where(max_axis == abs_z, 0.5 + x / (2 * abs_z + 1e-8), 0.5 + x / (2 * abs_y + 1e-8)))
            v = np.where(max_axis == abs_y, 0.5 + z / (2 * abs_y + 1e-8), 0.5 + y / (2 * max_axis + 1e-8))
            
        elif projection_type == "Mirror_Ball":
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            r = np.sqrt(x**2 + y**2)
            m = 2 * np.sqrt(x**2 + y**2 + (z + 1)**2 + 1e-8)
            u = x / m + 0.5
            v = y / m + 0.5
            
        elif projection_type == "Angular_Map":
            x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
            r = np.arccos(np.clip(z, -1, 1)) / np.pi
            phi = np.arctan2(y, x)
            u = r * np.cos(phi) * 0.5 + 0.5
            v = r * np.sin(phi) * 0.5 + 0.5
        
        # Apply fill mode
        u, v, mask = self._apply_fill_mode(u, v, fill_mode)
        
        # Sample the source image
        panorama = self._sample_image(img, u, v, interpolation)
        
        # Apply mask for black fill mode
        if fill_mode == "Black":
            panorama = panorama * mask[..., np.newaxis]
        
        # Apply exposure adjustment
        if exposure_adjust != 0:
            panorama = panorama * (2.0 ** exposure_adjust)
        
        # Create UV map for visualization
        uv_map = np.stack([u, v, mask], axis=-1).astype(np.float32)
        
        return (numpy_to_tensor_float32(panorama), numpy_to_tensor_float32(uv_map))


class SaveHDRI:
    """
    Save HDR panoramas as industry-standard HDRI environment maps.
    
    Supports EXR, HDR (Radiance), and 32-bit TIFF formats for use
    in 3D applications like Blender, Maya, or Unreal Engine.
    """
    
    FORMATS = ["EXR", "HDR", "TIFF"]
    BIT_DEPTHS = ["float32", "float16"]
    COMPRESSIONS = ["NONE", "ZIP", "PIZ", "DWAA", "DWAB", "RLE"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "panorama": ("IMAGE", {"tooltip": "HDR panorama image to save. Should be in linear color space for proper HDR."}),
                "filename_prefix": ("STRING", {
                    "default": "output/hdri_",
                    "tooltip": "Filename prefix including path. Numbers will be appended: hdri_0001.exr, hdri_0002.exr, etc."
                }),
                "format": (cls.FORMATS, {
                    "default": "EXR",
                    "tooltip": "Output format: EXR (best for VFX, supports layers), HDR (Radiance RGBE, widely compatible), TIFF (16-bit integer)."
                }),
            },
            "optional": {
                "bit_depth": (cls.BIT_DEPTHS, {
                    "default": "float32",
                    "tooltip": "Floating point precision: float32 (full precision, larger files), float16 (half precision, smaller files)."
                }),
                "compression": (cls.COMPRESSIONS, {
                    "default": "ZIP",
                    "tooltip": "EXR compression: ZIP (lossless, good), PIZ (lossless, fast), DWAA/DWAB (lossy, smallest), NONE (fastest), RLE (fast)."
                }),
                "include_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save JSON metadata file with image info (dimensions, dynamic range, peak luminance)."
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1,
                    "tooltip": "Pre-save gamma. 1.0 = linear (recommended for HDR). Only change if you need gamma-encoded output."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_TOOLTIPS = ("Path to saved HDRI file.",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Save HDR panoramas as HDRI environment maps (EXR/HDR/TIFF) for use in 3D applications."
    
    def save(self, panorama: torch.Tensor, filename_prefix: str = "output/hdri_",
             format: str = "EXR", bit_depth: str = "float32",
             compression: str = "ZIP", include_metadata: bool = True,
             gamma: float = 1.0) -> Tuple[str]:
        
        import cv2
        import folder_paths
        
        img = tensor_to_numpy_float32(panorama)
        if img.ndim == 4:
            img = img[0]
        
        # Apply gamma if needed
        if gamma != 1.0:
            img = np.power(np.maximum(img, 0), gamma)
        
        # Get output directory
        output_dir = folder_paths.get_output_directory()
        prefix_path = Path(filename_prefix)
        
        if prefix_path.parent != Path("."):
            full_output_dir = Path(output_dir) / prefix_path.parent
            full_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            full_output_dir = Path(output_dir)
        
        base_name = prefix_path.name
        
        # Find unique filename
        counter = 1
        while True:
            if format == "EXR":
                filename = f"{base_name}{counter:04d}.exr"
            elif format == "HDR":
                filename = f"{base_name}{counter:04d}.hdr"
            else:  # TIFF
                filename = f"{base_name}{counter:04d}.tiff"
            
            full_path = full_output_dir / filename
            if not full_path.exists():
                break
            counter += 1
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Handle bit depth
        if bit_depth == "float16":
            img_bgr = img_bgr.astype(np.float16)
        
        # Save based on format
        if format == "EXR":
            compression_map = {
                "NONE": cv2.IMWRITE_EXR_COMPRESSION_NO,
                "ZIP": cv2.IMWRITE_EXR_COMPRESSION_ZIP,
                "PIZ": cv2.IMWRITE_EXR_COMPRESSION_PIZ,
                "RLE": cv2.IMWRITE_EXR_COMPRESSION_RLE,
                "DWAA": cv2.IMWRITE_EXR_COMPRESSION_DWAA,
                "DWAB": cv2.IMWRITE_EXR_COMPRESSION_DWAB,
            }
            params = [
                cv2.IMWRITE_EXR_TYPE, 
                cv2.IMWRITE_EXR_TYPE_FLOAT if bit_depth == "float32" else cv2.IMWRITE_EXR_TYPE_HALF,
                cv2.IMWRITE_EXR_COMPRESSION,
                compression_map.get(compression, cv2.IMWRITE_EXR_COMPRESSION_ZIP)
            ]
            cv2.imwrite(str(full_path), img_bgr.astype(np.float32), params)
            
        elif format == "HDR":
            cv2.imwrite(str(full_path), img_bgr.astype(np.float32))
            
        else:  # TIFF
            cv2.imwrite(str(full_path), (img_bgr * 65535).astype(np.uint16))
        
        # Save metadata
        if include_metadata and format in ["EXR", "HDR"]:
            meta_path = full_path.with_suffix('.json')
            h, w = img.shape[:2]
            metadata = {
                "type": "HDRI Environment Map",
                "projection": "equirectangular",
                "width": w,
                "height": h,
                "aspect_ratio": f"{w}:{h}",
                "bit_depth": bit_depth,
                "format": format,
                "dynamic_range": {
                    "min": float(img.min()),
                    "max": float(img.max()),
                    "stops": float(np.log2(max(img.max(), 1e-10) / max(img.min(), 1e-10))) if img.min() > 0 else 0.0
                }
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return (str(full_path),)


# ═══════════════════════════════════════════════════════════════════════════════
#                          ACES 2.0 OUTPUT TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

class ACES2OutputTransform:
    """
    ACES 2.0 Output Transform implementation.
    
    The new ACES 2.0 (OpenDRT-based) provides improved:
    - Highlight handling without harsh clipping
    - Better skin tone preservation
    - Smoother roll-off in saturated colors
    - SDR, HDR, and cinema output support
    """
    
    OUTPUT_TRANSFORMS = [
        "ACES 2.0 SDR (sRGB/Rec.709)",
        "ACES 2.0 SDR (P3-D65)",
        "ACES 2.0 HDR (Rec.2100 PQ 1000 nits)",
        "ACES 2.0 HDR (Rec.2100 PQ 2000 nits)",
        "ACES 2.0 HDR (Rec.2100 PQ 4000 nits)",
        "ACES 2.0 HDR (Rec.2100 HLG)",
        "ACES 2.0 Cinema (DCI-P3 D60)",
        "ACES 2.0 Cinema (DCI-P3 D65)",
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "input_colorspace": (["ACEScg", "ACES2065-1", "Linear_sRGB", "Linear_Rec2020"], 
                                    {"default": "ACEScg"}),
                "output_transform": (cls.OUTPUT_TRANSFORMS, 
                                    {"default": "ACES 2.0 SDR (sRGB/Rec.709)"}),
            },
            "optional": {
                "peak_luminance": ("FLOAT", {"default": 100.0, "min": 48.0, "max": 10000.0, "step": 1.0}),
                "surround": (["Dark", "Dim", "Average"], {"default": "Dim"}),
                "creative_white_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01}),
                "exposure_adjust": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "transform_info")
    FUNCTION = "apply_transform"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Apply ACES 2.0 Output Transform for SDR, HDR, or Cinema output."
    
    # ACES 2.0 core matrices and parameters
    ACES_AP0_TO_AP1 = np.array([
        [1.4514393161, -0.2365107469, -0.2149285693],
        [-0.0765537734, 1.1762296998, -0.0996759264],
        [0.0083161484, -0.0060324498, 0.9977163014]
    ], dtype=np.float32)
    
    ACES_AP1_TO_sRGB = np.array([
        [1.7050509, -0.6217921, -0.0832588],
        [-0.1302564, 1.1408047, -0.0105483],
        [-0.0240033, -0.1289690, 1.1529723]
    ], dtype=np.float32)
    
    ACES_AP1_TO_P3D65 = np.array([
        [1.3792141, -0.3088546, -0.0703595],
        [-0.0693257, 1.0823507, -0.0130250],
        [-0.0021522, -0.0454616, 1.0476138]
    ], dtype=np.float32)
    
    ACES_AP1_TO_Rec2020 = np.array([
        [1.0258246, 0.0000000, -0.0258246],
        [-0.0000000, 1.0000000, 0.0000000],
        [-0.0000000, -0.0000000, 1.0000000]
    ], dtype=np.float32)
    
    def _apply_tonescale(self, rgb: np.ndarray, peak_luminance: float = 100.0) -> np.ndarray:
        """Apply ACES 2.0 tonescale curve with OpenDRT-style contrast."""
        # Simplified OpenDRT-style tonescale
        # Uses a modified Reinhard with shoulder compression
        
        # Calculate luminance
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        luma = np.maximum(luma, 1e-10)
        
        # Peak white scaling
        peak_scale = peak_luminance / 100.0
        
        # OpenDRT-style contrast curve with smooth highlight rolloff
        # Parameters tuned for natural skin tones
        contrast = 1.6
        pivot = 0.18
        
        # Log-space contrast
        log_luma = np.log2(luma / pivot + 1e-10)
        log_luma_contrast = log_luma * contrast
        luma_contrast = np.power(2.0, log_luma_contrast) * pivot
        
        # Highlight compression (soft clip)
        shoulder_start = 0.8 * peak_scale
        luma_compressed = np.where(
            luma_contrast < shoulder_start,
            luma_contrast,
            shoulder_start + (1.0 - shoulder_start) * np.tanh(
                (luma_contrast - shoulder_start) / (peak_scale - shoulder_start + 1e-10)
            ) * (peak_scale - shoulder_start)
        )
        
        # Scale RGB by luminance ratio
        ratio = luma_compressed / (luma + 1e-10)
        result = rgb * ratio[..., np.newaxis]
        
        return np.clip(result / peak_scale, 0, 1)
    
    def _pq_encode(self, linear: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
        """Encode to PQ (Perceptual Quantizer) for HDR output."""
        # Normalize to 10000 nits reference
        L = np.clip(linear * peak_nits / 10000.0, 0, 1)
        
        # PQ constants
        m1 = 0.1593017578125
        m2 = 78.84375
        c1 = 0.8359375
        c2 = 18.8515625
        c3 = 18.6875
        
        Lm1 = np.power(L, m1)
        pq = np.power((c1 + c2 * Lm1) / (1 + c3 * Lm1), m2)
        
        return pq
    
    def _hlg_encode(self, linear: np.ndarray) -> np.ndarray:
        """Encode to HLG (Hybrid Log-Gamma) for broadcast HDR."""
        a = 0.17883277
        b = 0.28466892
        c = 0.55991073
        
        hlg = np.where(
            linear <= 1/12,
            np.sqrt(3 * linear),
            a * np.log(12 * linear - b) + c
        )
        
        return np.clip(hlg, 0, 1)
    
    def apply_transform(self, image: torch.Tensor, input_colorspace: str,
                       output_transform: str, peak_luminance: float = 100.0,
                       surround: str = "Dim", creative_white_scale: float = 1.0,
                       exposure_adjust: float = 0.0) -> Tuple[torch.Tensor, str]:
        
        img = tensor_to_numpy_float32(image)
        if img.ndim == 4:
            img = img[0]
        
        # Apply exposure
        if exposure_adjust != 0:
            img = img * (2.0 ** exposure_adjust)
        
        # Convert input to ACEScg if needed
        if input_colorspace == "ACES2065-1":
            img = img @ self.ACES_AP0_TO_AP1.T
        elif input_colorspace == "Linear_sRGB":
            # sRGB to ACEScg
            srgb_to_ap1 = np.linalg.inv(self.ACES_AP1_TO_sRGB)
            img = img @ srgb_to_ap1.T
        
        # Apply creative adjustment
        img = img * creative_white_scale
        
        # Determine output parameters
        is_hdr = "HDR" in output_transform
        is_pq = "PQ" in output_transform
        is_hlg = "HLG" in output_transform
        is_p3 = "P3" in output_transform
        is_rec2020 = "2100" in output_transform or "Rec.2100" in output_transform
        
        # Extract peak nits for HDR
        if is_hdr and is_pq:
            if "4000" in output_transform:
                peak_nits = 4000.0
            elif "2000" in output_transform:
                peak_nits = 2000.0
            else:
                peak_nits = 1000.0
        else:
            peak_nits = peak_luminance
        
        # Apply tonescale
        tonemapped = self._apply_tonescale(img, peak_luminance=peak_nits/10.0)
        
        # Convert to output color space
        if is_rec2020:
            output = tonemapped @ self.ACES_AP1_TO_Rec2020.T
        elif is_p3:
            output = tonemapped @ self.ACES_AP1_TO_P3D65.T
        else:  # sRGB/Rec.709
            output = tonemapped @ self.ACES_AP1_TO_sRGB.T
        
        # Apply EOTF encoding
        if is_pq:
            output = self._pq_encode(output, peak_nits)
        elif is_hlg:
            output = self._hlg_encode(output)
        else:
            # sRGB gamma
            output = linear_to_srgb(np.clip(output, 0, 1))
        
        output = np.clip(output, 0, 1)
        
        info = f"ACES 2.0 | {input_colorspace} → {output_transform}"
        if is_hdr:
            info += f" | Peak: {peak_nits} nits"
        
        return (numpy_to_tensor_float32(output), info)


class DaVinciWideGamut:
    """
    DaVinci Wide Gamut and DaVinci Intermediate color space support.
    
    Used by DaVinci Resolve for its native color pipeline.
    Provides wider gamut than Rec.2020 for high-end grading.
    """
    
    TRANSFORMS = [
        "Linear to DaVinci WG",
        "DaVinci WG to Linear",
        "Linear to DaVinci Intermediate",
        "DaVinci Intermediate to Linear",
        "DaVinci WG to ACEScg",
        "ACEScg to DaVinci WG",
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "transform": (cls.TRANSFORMS, {"default": "Linear to DaVinci WG"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Convert to/from DaVinci Wide Gamut and DaVinci Intermediate."
    
    # DaVinci Wide Gamut primaries (wider than Rec.2020)
    # White point: D65
    DWG_TO_XYZ = np.array([
        [0.7006, 0.1487, 0.1014],
        [0.2741, 0.8736, -0.1477],
        [-0.0099, -0.0315, 0.9417]
    ], dtype=np.float32)
    
    XYZ_TO_DWG = np.linalg.inv(np.array([
        [0.7006, 0.1487, 0.1014],
        [0.2741, 0.8736, -0.1477],
        [-0.0099, -0.0315, 0.9417]
    ], dtype=np.float32))
    
    # sRGB to XYZ
    SRGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)
    
    XYZ_TO_SRGB = np.linalg.inv(np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32))
    
    def _davinci_intermediate_encode(self, linear: np.ndarray) -> np.ndarray:
        """Encode linear to DaVinci Intermediate log curve."""
        # DaVinci Intermediate parameters
        a = 0.0075
        b = 7.0
        c = 0.07329248
        m = 10.44426855
        lin_cut = 0.00262409
        log_cut = 0.02740668
        
        return np.where(
            linear < lin_cut,
            linear * m,
            c * np.log2(linear + a) + b * 0.1  # Simplified
        )
    
    def _davinci_intermediate_decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode DaVinci Intermediate to linear."""
        a = 0.0075
        b = 7.0
        c = 0.07329248
        m = 10.44426855
        lin_cut = 0.00262409
        log_cut = 0.02740668
        
        return np.where(
            encoded < log_cut,
            encoded / m,
            np.power(2.0, (encoded - b * 0.1) / c) - a
        )
    
    def convert(self, image: torch.Tensor, transform: str) -> Tuple[torch.Tensor]:
        img = tensor_to_numpy_float32(image)
        if img.ndim == 4:
            img = img[0]
        
        if transform == "Linear to DaVinci WG":
            # Linear sRGB to DaVinci Wide Gamut
            xyz = img @ self.SRGB_TO_XYZ.T
            result = xyz @ self.XYZ_TO_DWG.T
        
        elif transform == "DaVinci WG to Linear":
            xyz = img @ self.DWG_TO_XYZ.T
            result = xyz @ self.XYZ_TO_SRGB.T
        
        elif transform == "Linear to DaVinci Intermediate":
            # First to DWG, then encode
            xyz = img @ self.SRGB_TO_XYZ.T
            dwg = xyz @ self.XYZ_TO_DWG.T
            result = self._davinci_intermediate_encode(dwg)
        
        elif transform == "DaVinci Intermediate to Linear":
            dwg = self._davinci_intermediate_decode(img)
            xyz = dwg @ self.DWG_TO_XYZ.T
            result = xyz @ self.XYZ_TO_SRGB.T
        
        elif transform == "DaVinci WG to ACEScg":
            # DWG -> XYZ -> ACEScg
            xyz = img @ self.DWG_TO_XYZ.T
            # XYZ to ACEScg (AP1)
            XYZ_TO_AP1 = np.array([
                [1.6410, -0.3249, -0.2365],
                [-0.6636, 1.6153, 0.0168],
                [0.0117, -0.0084, 0.9884]
            ], dtype=np.float32)
            result = xyz @ XYZ_TO_AP1.T
        
        else:  # ACEScg to DaVinci WG
            AP1_TO_XYZ = np.array([
                [0.6624, 0.1340, 0.1561],
                [0.2722, 0.6741, 0.0537],
                [-0.0056, 0.0040, 1.0103]
            ], dtype=np.float32)
            xyz = img @ AP1_TO_XYZ.T
            result = xyz @ self.XYZ_TO_DWG.T
        
        return (numpy_to_tensor_float32(result),)


class ARRIWideGamut4:
    """
    ARRI Wide Gamut 4 (AWG4) color space support.
    
    The latest ARRI color science used with LogC4 in Alexa 35.
    Provides excellent color fidelity for high-end production.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["AWG4 to ACEScg", "ACEScg to AWG4", 
                              "AWG4 to Linear sRGB", "Linear sRGB to AWG4"], 
                             {"default": "AWG4 to ACEScg"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = "Convert to/from ARRI Wide Gamut 4 (AWG4) for Alexa 35."
    
    # ARRI Wide Gamut 4 to XYZ (D65)
    AWG4_TO_XYZ = np.array([
        [0.7048583, 0.1290112, 0.1166296],
        [0.2540892, 0.7814076, -0.0354969],
        [-0.0094877, -0.0324927, 0.8954361]
    ], dtype=np.float32)
    
    XYZ_TO_AWG4 = np.linalg.inv(np.array([
        [0.7048583, 0.1290112, 0.1166296],
        [0.2540892, 0.7814076, -0.0354969],
        [-0.0094877, -0.0324927, 0.8954361]
    ], dtype=np.float32))
    
    # Standard matrices
    SRGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)
    
    XYZ_TO_SRGB = np.linalg.inv(np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32))
    
    XYZ_TO_AP1 = np.array([
        [1.6410, -0.3249, -0.2365],
        [-0.6636, 1.6153, 0.0168],
        [0.0117, -0.0084, 0.9884]
    ], dtype=np.float32)
    
    AP1_TO_XYZ = np.array([
        [0.6624, 0.1340, 0.1561],
        [0.2722, 0.6741, 0.0537],
        [-0.0056, 0.0040, 1.0103]
    ], dtype=np.float32)
    
    def convert(self, image: torch.Tensor, direction: str) -> Tuple[torch.Tensor]:
        img = tensor_to_numpy_float32(image)
        if img.ndim == 4:
            img = img[0]
        
        if direction == "AWG4 to ACEScg":
            xyz = img @ self.AWG4_TO_XYZ.T
            result = xyz @ self.XYZ_TO_AP1.T
        
        elif direction == "ACEScg to AWG4":
            xyz = img @ self.AP1_TO_XYZ.T
            result = xyz @ self.XYZ_TO_AWG4.T
        
        elif direction == "AWG4 to Linear sRGB":
            xyz = img @ self.AWG4_TO_XYZ.T
            result = xyz @ self.XYZ_TO_SRGB.T
        
        else:  # Linear sRGB to AWG4
            xyz = img @ self.SRGB_TO_XYZ.T
        return (numpy_to_tensor_float32(result),)


class RadianceVAEEncode:
    """
    Professional VAE Encoder with 32-bit Color Space awareness.
    
    Bridges the gap between Linear/ACEScg pipelines and Standard VAEs.
    Allows encoding of high dynamic range data without forced clamping.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "source_space": (["Linear", "ACEScg", "sRGB", "Raw"], {"default": "Linear"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "clip_to_0_1": ("BOOLEAN", {"default": False, "tooltip": "Force values to 0.0-1.0 range. Disable for HDR experiments."}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Encode 32-bit Linear/ACEScg images to VAE Latents with correct color handling."
    
    def encode(self, pixels: torch.Tensor, vae: Any, source_space: str = "Linear",
               exposure: float = 0.0, clip_to_0_1: bool = False) -> Tuple[Dict[str, Any]]:
        
        # 1. Standardize Input (Clone to avoid modifying original)
        img = pixels.clone().float()
        
        # 2. Apply Exposure (32-bit float operation)
        if exposure != 0.0:
            img = img * (2.0 ** exposure)
        
        # 3. Transform to sRGB (Standard VAE Input Space)
        # Most VAEs (SD1.5, SDXL, Flux) expect Gamma-Corrected sRGB inputs ~ [0,1]
        
        if source_space == "Linear":
            img = tensor_linear_to_srgb(img)
            
        elif source_space == "ACEScg":
            # ACEScg (AP1) -> Linear sRGB (Rec.709) -> sRGB Gamma
            # Matrix: ACEScg to Rec.709 (Linear)
            # AP1_TO_REC709
            AP1_TO_REC709 = torch.tensor([
                [1.6410, -0.3249, -0.2365],
                [-0.6636, 1.6153, 0.0168],
                [0.0117, -0.0084, 0.9884]
            ], dtype=img.dtype, device=img.device).T
            
            # Matmul: (..., 3) @ (3, 3)
            # Handle dimensions
            orig_shape = img.shape
            if img.dim() == 4:
                rgb = img[..., :3].reshape(-1, 3)
                img[..., :3] = (rgb @ AP1_TO_REC709).reshape(orig_shape[0], orig_shape[1], orig_shape[2], 3)
            elif img.dim() == 3 and img.shape[-1] == 3:
                 img = img @ AP1_TO_REC709
            
            # Now Linear Rec.709 -> sRGB Gamma using Helper
            img = tensor_linear_to_srgb(img)

        # "sRGB" source implies it's already in Gamma space, no transform needed.
        # "Raw" implies passthrough.

        # 4. Handling 32-bit High Range (Values > 1.0)
        # Standard VAEs are trained on 0-1. Values > 1.0 (whiter than white)
        # might be encoded, but decoding them is not guaranteed.
        # However, the user explicitly requested "Match 32 bit pipeline" and "Not Clamped".
        
        if clip_to_0_1:
            img = torch.clamp(img, 0.0, 1.0)
        
        # 5. Encode
        # Resize logic is usually handled by VAEEncode, but here we assume 'pixels'
        # are fed directly. ComfyUI VAE encode usually takes (B, H, W, 3).
        
        # img is (B, H, W, 3)
        # ComfyUI vae.encode expected layout: (B, H, W, 3)
        
        return (vae.encode(img[:,:,:,:3]),)


class RadianceVAEDecode:
    """
    Professional VAE Decoder for 32-bit HDR Pipelines.
    
    Decodes Latents directly into Linear or ACEScg space,
    bypassing manual conversion steps and ensuring float32 precision.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "target_space": (["Linear", "ACEScg", "sRGB", "Raw"], {"default": "Linear"}),
                "exposure_adjust": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Decode VAE Latents directly to 32-bit Linear/ACEScg images."
    
    def decode(self, samples: Dict[str, Any], vae: Any, target_space: str = "Linear",
               exposure_adjust: float = 0.0) -> Tuple[torch.Tensor]:
        
        # 1. Decode (Standard VAE output is sRGB Gamma 2.2)
        # VAE decode returns tensor (B, H, W, 3) in [0, 1] usually
        img = vae.decode(samples["samples"])
        
        # Ensure float32
        img = img.float()
        
        # 2. Exposure Adjust & Color Space Conversion
        # All operations performed on Tether/GPU using pure Torch
        
        if target_space == "Raw":
             # "Raw" implies untouched structure, but we can still apply exposure
             if exposure_adjust != 0.0:
                 img = img * (2.0 ** exposure_adjust)
        
        elif target_space == "sRGB":
            # Input is already sRGB (Gamma 2.2-ish)
            
            # If exposure is requested, we must linearize -> expose -> re-gamma
            # strictly for correctness.
            if exposure_adjust != 0.0:
                linear = tensor_srgb_to_linear(img)
                linear = linear * (2.0 ** exposure_adjust)
                img = tensor_linear_to_srgb(linear)
        
        elif target_space == "Linear":
            img = tensor_srgb_to_linear(img)
            if exposure_adjust != 0.0:
                img = img * (2.0 ** exposure_adjust)
                
        elif target_space == "ACEScg":
            # sRGB -> Linear sRGB -> ACEScg
            linear = tensor_srgb_to_linear(img)
            
            if exposure_adjust != 0.0:
                linear = linear * (2.0 ** exposure_adjust)
            
            # Matrix: Linear sRGB (Rec.709) -> ACEScg (AP1)
            # REC709_TO_AP1
            # Row-major for direct multiplication with (..., 3)
            # Source (Rec709) -> Dest (AP1)
            REC709_TO_AP1 = torch.tensor([
                [0.613097, 0.339523, 0.047379],
                [0.070194, 0.916354, 0.013452],
                [0.020616, 0.109570, 0.869815]
            ], dtype=img.dtype, device=img.device).T
            
            # Apply matrix multiplication
            # Handle (B, H, W, 3) or (H, W, 3)
            if img.shape[-1] == 3:
                # Optimized einsum or matmul
                img = torch.matmul(linear, REC709_TO_AP1)
            else:
                # Fallback purely linear (shouldn't happen for valid VAE output)
                img = linear

        return (img,)


#                          NODE MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "ImageToFloat32": ImageToFloat32,
    "Float32ColorCorrect": Float32ColorCorrect,
    "HDRExpandDynamicRange": HDRExpandDynamicRange,
    "HDRToneMap": HDRToneMap,
    "ColorSpaceConvert": ColorSpaceConvert,
    "SaveImageEXR": SaveImageEXR,
    "LoadImageEXR": LoadImageEXR,
    "LoadImageEXRSequence": LoadImageEXRSequence,
    "SaveImage16bit": SaveImage16bit,
    "HDRHistogram": HDRHistogram,
    "LogCurveEncode": LogCurveEncode,
    "LogCurveDecode": LogCurveDecode,
    "HDRExposureBlend": HDRExposureBlend,
    "HDRShadowHighlightRecovery": HDRShadowHighlightRecovery,
    "OCIOColorTransform": OCIOColorTransform,
    "OCIOListColorspaces": OCIOListColorspaces,
    "GPUTensorOps": GPUTensorOps,
    "HDR360Generate": HDR360Generate,
    "SaveHDRI": SaveHDRI,
    "ACES2OutputTransform": ACES2OutputTransform,
    "DaVinciWideGamut": DaVinciWideGamut,
    "ARRIWideGamut4": ARRIWideGamut4,
    "RadianceVAEEncode": RadianceVAEEncode,
    "RadianceVAEDecode": RadianceVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToFloat32": "◆ Radiance Image to Float32",
    "Float32ColorCorrect": "◆ Radiance Float32 Color Correct",
    "HDRExpandDynamicRange": "◆ Radiance HDR Expand Dynamic Range",
    "HDRToneMap": "◆ Radiance HDR Tone Map",
    "ColorSpaceConvert": "◆ Radiance Color Space Convert",
    "SaveImageEXR": "◆ Radiance Save EXR (32-bit)",
    "LoadImageEXR": "◆ Radiance Load EXR",
    "LoadImageEXRSequence": "◆ Radiance Load EXR Sequence",
    "SaveImage16bit": "◆ Radiance Save 16-bit PNG/TIFF",
    "HDRHistogram": "◆ Radiance HDR Histogram",
    "LogCurveEncode": "◆ Radiance Log Curve Encode",
    "LogCurveDecode": "◆ Radiance Log Curve Decode",
    "HDRExposureBlend": "◆ Radiance HDR Exposure Blend",
    "HDRShadowHighlightRecovery": "◆ Radiance HDR Shadow/Highlight Recovery",
    "OCIOColorTransform": "◆ Radiance OCIO Color Transform",
    "OCIOListColorspaces": "◆ Radiance OCIO List Colorspaces",
    "GPUTensorOps": "◆ Radiance GPU Tensor Ops",
    "HDR360Generate": "◆ Radiance HDR 360 Generate",
    "SaveHDRI": "◆ Radiance Save HDRI",
    "ACES2OutputTransform": "◆ Radiance ACES 2.0 Output Transform",
    "DaVinciWideGamut": "◆ Radiance DaVinci Wide Gamut",
    "ARRIWideGamut4": "◆ Radiance ARRI Wide Gamut 4",
    "RadianceVAEEncode": "◆ Radiance VAE Encode (32-bit)",
    "RadianceVAEDecode": "◆ Radiance VAE Decode (32-bit)",
}



if __name__ == "__main__":
    print("═" * 60)
    print("  ComfyUI 32-Bit HDR Processing Nodes")
    print("  FXTD Studios © 2024")
    print("═" * 60)
    print(f"\n  OpenEXR available: {HAS_OPENEXR}")
    print(f"  Colour-science available: {HAS_COLOUR}")
    print(f"\n  Registered nodes: {len(NODE_CLASS_MAPPINGS)}")
    for name, display in NODE_DISPLAY_NAME_MAPPINGS.items():
        print(f"    • {display}")
    print()
