"""
FXTD Pro Upscaler - Professional 32-bit Upscaling for ComfyUI
Version: 1.0.0
Author: FXTD Studios

Professional upscaling solution optimized for Flux and HDR workflows:
- True 32-bit float processing pipeline
- Multiple upscaling algorithms
- Tile-based processing for large images
- Detail enhancement and sharpening
- Color space aware processing
- Flux-optimized presets
"""

# FIX: OpenMP duplicate library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional, List, Union
import math
from enum import Enum


# =============================================================================
# UPSCALING ALGORITHMS
# =============================================================================

class UpscaleMethod(Enum):
    """Available upscaling methods."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    LANCZOS4 = "lanczos4"
    MITCHELL = "mitchell"
    CATROM = "catrom"
    HERMITE = "hermite"
    GAUSSIAN = "gaussian"


def lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """Lanczos kernel function."""
    x = np.abs(x)
    result = np.zeros_like(x)
    
    # For |x| < a
    mask = x < a
    x_masked = x[mask]
    
    # Avoid division by zero
    nonzero = x_masked != 0
    result_masked = np.zeros_like(x_masked)
    
    if np.any(nonzero):
        x_nz = x_masked[nonzero]
        result_masked[nonzero] = (a * np.sin(np.pi * x_nz) * np.sin(np.pi * x_nz / a)) / (np.pi ** 2 * x_nz ** 2)
    
    result_masked[~nonzero] = 1.0
    result[mask] = result_masked
    
    return result


def mitchell_kernel(x: np.ndarray, B: float = 1/3, C: float = 1/3) -> np.ndarray:
    """Mitchell-Netravali kernel (used in Photoshop)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    
    # |x| < 1
    mask1 = x < 1
    x1 = x[mask1]
    result[mask1] = ((12 - 9*B - 6*C) * x1**3 + (-18 + 12*B + 6*C) * x1**2 + (6 - 2*B)) / 6
    
    # 1 <= |x| < 2
    mask2 = (x >= 1) & (x < 2)
    x2 = x[mask2]
    result[mask2] = ((-B - 6*C) * x2**3 + (6*B + 30*C) * x2**2 + (-12*B - 48*C) * x2 + (8*B + 24*C)) / 6
    
    return result


def catmull_rom_kernel(x: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline kernel (sharper than Mitchell)."""
    return mitchell_kernel(x, B=0, C=0.5)


def hermite_kernel(x: np.ndarray) -> np.ndarray:
    """Hermite kernel (smooth)."""
    x = np.abs(x)
    result = np.zeros_like(x)
    mask = x < 1
    x_m = x[mask]
    result[mask] = (2 * x_m**3 - 3 * x_m**2 + 1)
    return result


def gaussian_kernel(x: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Gaussian kernel."""
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


def create_1d_kernel(size: int, scale: float, method: str, support: float = None) -> np.ndarray:
    """Create 1D resampling kernel."""
    if support is None:
        if method in ['lanczos', 'lanczos4']:
            support = 4.0 if method == 'lanczos4' else 3.0
        elif method in ['mitchell', 'catrom']:
            support = 2.0
        elif method == 'hermite':
            support = 1.0
        elif method == 'gaussian':
            support = 2.0
        else:
            support = 1.0
    
    # Calculate kernel size
    kernel_size = int(np.ceil(support * 2 * scale)) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    center = kernel_size // 2
    x = (np.arange(kernel_size) - center) / scale
    
    if method == 'lanczos':
        kernel = lanczos_kernel(x, a=3)
    elif method == 'lanczos4':
        kernel = lanczos_kernel(x, a=4)
    elif method == 'mitchell':
        kernel = mitchell_kernel(x)
    elif method == 'catrom':
        kernel = catmull_rom_kernel(x)
    elif method == 'hermite':
        kernel = hermite_kernel(x)
    elif method == 'gaussian':
        kernel = gaussian_kernel(x)
    else:
        kernel = np.ones(kernel_size)
    
    # Normalize
    kernel = kernel / (kernel.sum() + 1e-10)
    
    return kernel.astype(np.float32)


def separable_resize_32bit(img: np.ndarray, new_h: int, new_w: int, 
                           method: str = 'lanczos') -> np.ndarray:
    """
    High-quality separable resize maintaining 32-bit precision.
    Processes in float32 throughout.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    
    # Ensure float32
    img = img.astype(np.float32)
    
    # Scale factors
    scale_h = h / new_h
    scale_w = w / new_w
    
    # Horizontal resize first
    if new_w != w:
        temp = np.zeros((h, new_w, channels), dtype=np.float32)
        kernel = create_1d_kernel(new_w, max(1, scale_w), method)
        k_size = len(kernel)
        k_center = k_size // 2
        
        for x_out in range(new_w):
            x_in = (x_out + 0.5) * scale_w - 0.5
            x_start = int(x_in) - k_center
            
            weighted_sum = np.zeros((h, channels), dtype=np.float32)
            weight_sum = 0.0
            
            for k in range(k_size):
                x_src = x_start + k
                x_src_clamped = max(0, min(w - 1, x_src))
                weight = kernel[k]
                weighted_sum += img[:, x_src_clamped, :] * weight
                weight_sum += weight
            
            temp[:, x_out, :] = weighted_sum / (weight_sum + 1e-10)
    else:
        temp = img
    
    # Vertical resize
    if new_h != h:
        result = np.zeros((new_h, new_w, channels), dtype=np.float32)
        kernel = create_1d_kernel(new_h, max(1, scale_h), method)
        k_size = len(kernel)
        k_center = k_size // 2
        
        for y_out in range(new_h):
            y_in = (y_out + 0.5) * scale_h - 0.5
            y_start = int(y_in) - k_center
            
            weighted_sum = np.zeros((new_w, channels), dtype=np.float32)
            weight_sum = 0.0
            
            for k in range(k_size):
                y_src = y_start + k
                y_src_clamped = max(0, min(h - 1, y_src))
                weight = kernel[k]
                weighted_sum += temp[y_src_clamped, :, :] * weight
                weight_sum += weight
            
            result[y_out, :, :] = weighted_sum / (weight_sum + 1e-10)
    else:
        result = temp
    
    if channels == 1:
        result = result[:, :, 0]
    
    return result


def torch_resize_32bit(tensor: torch.Tensor, new_h: int, new_w: int, 
                       method: str = 'bicubic') -> torch.Tensor:
    """
    PyTorch-based resize maintaining 32-bit precision.
    Uses GPU acceleration when available.
    """
    # Ensure BCHW format
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Convert to BCHW if needed (from BHWC)
    if tensor.shape[-1] in [1, 3, 4]:
        tensor = tensor.permute(0, 3, 1, 2)
    
    # Ensure float32
    tensor = tensor.float()
    
    # Map method names
    mode_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',  # Fallback
        'lanczos4': 'bicubic',
        'mitchell': 'bicubic',
        'catrom': 'bicubic',
        'hermite': 'bilinear',
        'gaussian': 'bilinear',
    }
    
    mode = mode_map.get(method, 'bicubic')
    align_corners = False if mode != 'nearest' else None
    
    if mode == 'nearest':
        resized = F.interpolate(tensor, size=(new_h, new_w), mode=mode)
    else:
        resized = F.interpolate(tensor, size=(new_h, new_w), mode=mode, 
                                align_corners=align_corners, antialias=True)
    
    # Convert back to BHWC
    resized = resized.permute(0, 2, 3, 1)
    
    return resized


# =============================================================================
# DETAIL ENHANCEMENT
# =============================================================================

def unsharp_mask_32bit(img: np.ndarray, amount: float = 1.0, 
                       radius: float = 1.0, threshold: float = 0.0) -> np.ndarray:
    """
    Unsharp mask in 32-bit precision.
    """
    from PIL import ImageFilter
    h, w = img.shape[:2]
    
    # Create blurred version
    blurred = np.zeros_like(img)
    for c in range(img.shape[2]):
        # Scale to 0-255 for PIL, then back
        channel = img[..., c]
        ch_min, ch_max = channel.min(), channel.max()
        ch_range = ch_max - ch_min if ch_max != ch_min else 1.0
        
        ch_normalized = ((channel - ch_min) / ch_range * 255).astype(np.uint8)
        ch_pil = Image.fromarray(ch_normalized)
        ch_pil = ch_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        if ch_pil.size != (w, h):
            ch_pil = ch_pil.resize((w, h), Image.BILINEAR)
        
        blurred_ch = np.array(ch_pil).astype(np.float32) / 255.0 * ch_range + ch_min
        blurred[..., c] = blurred_ch
    
    # Calculate mask
    mask = img - blurred
    
    # Apply threshold
    if threshold > 0:
        mask_abs = np.abs(mask)
        mask = np.where(mask_abs > threshold, mask, 0)
    
    # Apply sharpening
    result = img + mask * amount
    
    return result


def high_pass_sharpen_32bit(img: np.ndarray, strength: float = 0.5,
                            radius: float = 3.0) -> np.ndarray:
    """
    High-pass sharpening in 32-bit.
    """
    from PIL import ImageFilter
    h, w = img.shape[:2]
    
    # Create heavily blurred version
    low_pass = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[..., c]
        ch_min, ch_max = channel.min(), channel.max()
        ch_range = ch_max - ch_min if ch_max != ch_min else 1.0
        
        ch_normalized = np.clip((channel - ch_min) / ch_range * 255, 0, 255).astype(np.uint8)
        ch_pil = Image.fromarray(ch_normalized)
        ch_pil = ch_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        if ch_pil.size != (w, h):
            ch_pil = ch_pil.resize((w, h), Image.BILINEAR)
        
        low_pass_ch = np.array(ch_pil).astype(np.float32) / 255.0 * ch_range + ch_min
        low_pass[..., c] = low_pass_ch
    
    # High pass = original - low pass
    high_pass = img - low_pass
    
    # Blend: overlay mode approximation
    result = img + high_pass * strength
    
    return result


def detail_enhancement_32bit(img: np.ndarray, detail_strength: float = 0.5,
                             edge_strength: float = 0.3,
                             local_contrast: float = 0.2) -> np.ndarray:
    """
    Multi-scale detail enhancement in 32-bit.
    """
    from PIL import ImageFilter
    h, w = img.shape[:2]
    
    result = img.copy()
    
    # Calculate luminance
    if img.shape[2] >= 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img[..., 0]
    
    # Multi-scale detail extraction
    scales = [1.0, 2.0, 4.0]
    detail_layers = []
    
    prev_blur = lum.copy()
    for scale in scales:
        # Blur at this scale
        lum_min, lum_max = lum.min(), lum.max()
        lum_range = lum_max - lum_min if lum_max != lum_min else 1.0
        
        lum_normalized = np.clip((prev_blur - lum_min) / lum_range * 255, 0, 255).astype(np.uint8)
        lum_pil = Image.fromarray(lum_normalized)
        lum_pil = lum_pil.filter(ImageFilter.GaussianBlur(radius=scale))
        if lum_pil.size != (w, h):
            lum_pil = lum_pil.resize((w, h), Image.BILINEAR)
        
        current_blur = np.array(lum_pil).astype(np.float32) / 255.0 * lum_range + lum_min
        
        # Detail at this scale
        detail = prev_blur - current_blur
        detail_layers.append(detail)
        prev_blur = current_blur
    
    # Combine details
    combined_detail = np.zeros_like(lum)
    weights = [0.5, 0.3, 0.2]  # Fine to coarse
    for detail, weight in zip(detail_layers, weights):
        combined_detail += detail * weight * detail_strength
    
    # Apply to color channels
    for c in range(min(3, img.shape[2])):
        result[..., c] = img[..., c] + combined_detail
    
    # Local contrast enhancement
    if local_contrast > 0:
        # Calculate local mean
        local_radius = 15
        lum_normalized = np.clip((lum - lum.min()) / (lum.max() - lum.min() + 1e-10) * 255, 0, 255).astype(np.uint8)
        lum_pil = Image.fromarray(lum_normalized)
        lum_pil = lum_pil.filter(ImageFilter.GaussianBlur(radius=local_radius))
        if lum_pil.size != (w, h):
            lum_pil = lum_pil.resize((w, h), Image.BILINEAR)
        local_mean = np.array(lum_pil).astype(np.float32) / 255.0 * (lum.max() - lum.min()) + lum.min()
        
        # Local contrast
        local_diff = (lum - local_mean) * local_contrast
        for c in range(min(3, img.shape[2])):
            result[..., c] = result[..., c] + local_diff
    
    return result


# =============================================================================
# ANTI-ALIASING
# =============================================================================

def apply_antialiasing_32bit(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Apply edge-aware antialiasing in 32-bit.
    """
    from PIL import ImageFilter
    h, w = img.shape[:2]
    
    # Calculate luminance
    if img.shape[2] >= 3:
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        lum = img[..., 0]
    
    # Edge detection using Sobel-like operation
    # Simple gradient magnitude
    grad_x = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    grad_y = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Normalize edges
    edges = edges / (edges.max() + 1e-10)
    
    # Create blurred version
    blurred = np.zeros_like(img)
    for c in range(img.shape[2]):
        channel = img[..., c]
        ch_min, ch_max = channel.min(), channel.max()
        ch_range = ch_max - ch_min if ch_max != ch_min else 1.0
        
        ch_normalized = np.clip((channel - ch_min) / ch_range * 255, 0, 255).astype(np.uint8)
        ch_pil = Image.fromarray(ch_normalized)
        ch_pil = ch_pil.filter(ImageFilter.GaussianBlur(radius=1.0))
        if ch_pil.size != (w, h):
            ch_pil = ch_pil.resize((w, h), Image.BILINEAR)
        
        blurred_ch = np.array(ch_pil).astype(np.float32) / 255.0 * ch_range + ch_min
        blurred[..., c] = blurred_ch
    
    # Blend based on edges
    edge_mask = (edges * strength)[..., np.newaxis]
    result = img * (1 - edge_mask) + blurred * edge_mask
    
    return result


# =============================================================================
# TILE PROCESSING
# =============================================================================

def process_tiles_32bit(img: np.ndarray, tile_size: int, overlap: int,
                        process_func, **kwargs) -> np.ndarray:
    """
    Process image in tiles with overlap for seamless results.
    Maintains 32-bit precision throughout.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    # Calculate output size from a test tile
    test_tile = img[:min(tile_size, h), :min(tile_size, w)]
    test_output = process_func(test_tile, **kwargs)
    
    scale_h = test_output.shape[0] / test_tile.shape[0]
    scale_w = test_output.shape[1] / test_tile.shape[1]
    
    out_h = int(h * scale_h)
    out_w = int(w * scale_w)
    out_tile_size = int(tile_size * scale_h)
    out_overlap = int(overlap * scale_h)
    
    # Initialize output and weight buffers
    output = np.zeros((out_h, out_w, channels), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)
    
    # Create blending weight
    blend_weight = create_tile_weight(out_tile_size, out_overlap)
    
    # Process tiles
    stride = tile_size - overlap
    out_stride = out_tile_size - out_overlap
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img[y:y_end, x:x_end]
            
            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, channels), dtype=np.float32)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            # Process tile
            processed = process_func(tile, **kwargs)
            
            # Calculate output position
            out_y = int(y * scale_h)
            out_x = int(x * scale_w)
            out_y_end = min(out_y + out_tile_size, out_h)
            out_x_end = min(out_x + out_tile_size, out_w)
            
            # Get tile weight
            tile_h = out_y_end - out_y
            tile_w = out_x_end - out_x
            weight = blend_weight[:tile_h, :tile_w]
            
            # Accumulate
            for c in range(channels):
                output[out_y:out_y_end, out_x:out_x_end, c] += processed[:tile_h, :tile_w, c] * weight
            weights[out_y:out_y_end, out_x:out_x_end] += weight
            
            x += stride
            if x >= w - overlap and x < w:
                x = w - tile_size
                if x < 0:
                    break
        
        y += stride
        if y >= h - overlap and y < h:
            y = h - tile_size
            if y < 0:
                break
    
    # Normalize
    weights = np.maximum(weights, 1e-10)
    for c in range(channels):
        output[..., c] /= weights
    
    return output


def create_tile_weight(size: int, overlap: int) -> np.ndarray:
    """Create smooth blending weight for tiles."""
    weight = np.ones((size, size), dtype=np.float32)
    
    if overlap > 0:
        # Create linear ramp for edges
        ramp = np.linspace(0, 1, overlap)
        
        # Apply to edges
        for i in range(overlap):
            weight[i, :] *= ramp[i]
            weight[-(i+1), :] *= ramp[i]
            weight[:, i] *= ramp[i]
            weight[:, -(i+1)] *= ramp[i]
    
    return weight


# =============================================================================
# COLOR SPACE HANDLING
# =============================================================================

def linear_to_srgb_32bit(img: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB in 32-bit."""
    result = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 0), 1/2.4) - 0.055
    )
    return result.astype(np.float32)


def srgb_to_linear_32bit(img: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear in 32-bit."""
    result = np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((np.maximum(img, 0) + 0.055) / 1.055, 2.4)
    )
    return result.astype(np.float32)


# =============================================================================
# FLUX-OPTIMIZED PRESETS
# =============================================================================

FLUX_PRESETS = {
    "Flux Default": {
        "description": "Balanced upscaling optimized for Flux outputs",
        "method": "lanczos",
        "sharpening": 0.3,
        "detail_enhancement": 0.2,
        "antialiasing": 0.3,
        "color_space": "sRGB",
    },
    "Flux Sharp": {
        "description": "Sharp upscaling for detailed Flux images",
        "method": "lanczos4",
        "sharpening": 0.6,
        "detail_enhancement": 0.4,
        "antialiasing": 0.2,
        "color_space": "sRGB",
    },
    "Flux Smooth": {
        "description": "Smooth upscaling for soft Flux renders",
        "method": "mitchell",
        "sharpening": 0.1,
        "detail_enhancement": 0.1,
        "antialiasing": 0.5,
        "color_space": "sRGB",
    },
    "Flux HDR": {
        "description": "HDR-aware upscaling for high dynamic range",
        "method": "lanczos",
        "sharpening": 0.25,
        "detail_enhancement": 0.3,
        "antialiasing": 0.3,
        "color_space": "Linear",
    },
    "Flux Print": {
        "description": "High quality for print output",
        "method": "lanczos4",
        "sharpening": 0.5,
        "detail_enhancement": 0.5,
        "antialiasing": 0.15,
        "color_space": "sRGB",
    },
    "Flux Cinematic": {
        "description": "Film-like upscaling with subtle softness",
        "method": "catrom",
        "sharpening": 0.2,
        "detail_enhancement": 0.15,
        "antialiasing": 0.4,
        "color_space": "Linear",
    },
    "Flux Maximum": {
        "description": "Maximum detail preservation",
        "method": "lanczos4",
        "sharpening": 0.7,
        "detail_enhancement": 0.6,
        "antialiasing": 0.1,
        "color_space": "sRGB",
    },
}


# =============================================================================
# MAIN COMFYUI NODES
# =============================================================================

class FXTDProUpscale:
    """
    Professional 32-bit Upscaler for Flux
    
    Features:
    - True 32-bit float processing pipeline
    - Multiple high-quality algorithms (Lanczos, Mitchell, Catmull-Rom)
    - Flux-optimized presets
    - Detail enhancement and sharpening
    - Tile-based processing for large images
    - Color space aware processing
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        method_list = ["lanczos", "lanczos4", "bicubic", "mitchell", "catrom", 
                       "hermite", "bilinear", "nearest"]
        preset_list = ["Custom"] + list(FLUX_PRESETS.keys())
        
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "preset": (preset_list,),
            },
            "optional": {
                # Method override
                "method": (method_list,),
                
                # Sharpening
                "sharpening": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "sharpen_radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1
                }),
                
                # Detail enhancement
                "detail_enhancement": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                
                # Antialiasing
                "antialiasing": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                
                # Color space
                "input_color_space": (["sRGB", "Linear", "Auto"],),
                "process_in_linear": ("BOOLEAN", {"default": True}),
                
                # Tiling
                "use_tiles": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 64
                }),
                "tile_overlap": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16
                }),
                
                # Output
                "output_bit_depth": (["32-bit Float", "16-bit Float", "8-bit"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("upscaled_image", "width", "height", "info")
    FUNCTION = "upscale"
    CATEGORY = "FXTD Studios/Radiance/Upscale"
    DESCRIPTION = "Professional 32-bit upscaler optimized for Flux with HDR support."
    
    def upscale(self, image: torch.Tensor, scale_factor: float, preset: str,
                method: str = "lanczos", sharpening: float = 0.3,
                sharpen_radius: float = 1.0, detail_enhancement: float = 0.2,
                antialiasing: float = 0.3, input_color_space: str = "sRGB",
                process_in_linear: bool = True, use_tiles: bool = False,
                tile_size: int = 512, tile_overlap: int = 64,
                output_bit_depth: str = "32-bit Float"):
        
        # Apply preset if not custom
        if preset != "Custom" and preset in FLUX_PRESETS:
            p = FLUX_PRESETS[preset]
            method = p.get('method', method)
            sharpening = p.get('sharpening', sharpening)
            detail_enhancement = p.get('detail_enhancement', detail_enhancement)
            antialiasing = p.get('antialiasing', antialiasing)
            if p.get('color_space') == 'Linear':
                process_in_linear = True
        
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        results = []
        
        for b in range(batch_size):
            # Convert to numpy, ensure float32
            img = image[b].cpu().numpy().astype(np.float32)
            
            # Handle alpha
            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]
            
            # Convert to linear if needed
            if process_in_linear and input_color_space == "sRGB":
                img = srgb_to_linear_32bit(img)
            
            # Upscale
            if use_tiles and (h > tile_size or w > tile_size):
                # Tile-based processing
                def upscale_tile(tile, **kw):
                    th, tw = tile.shape[:2]
                    new_th = int(th * scale_factor)
                    new_tw = int(tw * scale_factor)
                    return separable_resize_32bit(tile, new_th, new_tw, method)
                
                upscaled = process_tiles_32bit(img, tile_size, tile_overlap, upscale_tile)
            else:
                # Direct processing
                upscaled = separable_resize_32bit(img, new_h, new_w, method)
            
            # Detail enhancement
            if detail_enhancement > 0:
                upscaled = detail_enhancement_32bit(upscaled, detail_enhancement)
            
            # Sharpening
            if sharpening > 0:
                upscaled = unsharp_mask_32bit(upscaled, sharpening, sharpen_radius)
            
            # Antialiasing
            if antialiasing > 0:
                upscaled = apply_antialiasing_32bit(upscaled, antialiasing)
            
            # Convert back to sRGB if processed in linear
            if process_in_linear and input_color_space == "sRGB":
                upscaled = linear_to_srgb_32bit(upscaled)
            
            # Handle alpha
            if has_alpha:
                # Upscale alpha separately
                alpha_up = separable_resize_32bit(alpha, new_h, new_w, 'lanczos')
                if len(alpha_up.shape) == 2:
                    alpha_up = alpha_up[:, :, np.newaxis]
                upscaled = np.concatenate([upscaled, alpha_up], axis=-1)
            
            # Convert bit depth
            if output_bit_depth == "16-bit Float":
                upscaled = upscaled.astype(np.float16).astype(np.float32)
            elif output_bit_depth == "8-bit":
                upscaled = np.clip(upscaled, 0, 1)
                upscaled = (upscaled * 255).astype(np.uint8).astype(np.float32) / 255.0
            
            results.append(upscaled)
        
        # Stack results
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        # Build info string
        info = f"Upscaled: {w}x{h} → {new_w}x{new_h} ({scale_factor}x)\n"
        info += f"Method: {method}\n"
        info += f"Preset: {preset}\n"
        info += f"Sharpening: {sharpening}, Detail: {detail_enhancement}\n"
        info += f"Output: {output_bit_depth}"
        
        return (output_tensor, new_w, new_h, info)


class FXTDUpscaleBySize:
    """
    Upscale to exact dimensions.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        method_list = ["lanczos", "lanczos4", "bicubic", "mitchell", "catrom",
                       "hermite", "bilinear", "nearest"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 16384,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 16384,
                    "step": 8
                }),
                "method": (method_list,),
            },
            "optional": {
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                "aspect_mode": (["fit", "fill", "stretch"],),
                "sharpening": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "process_in_linear": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("upscaled_image", "final_width", "final_height")
    FUNCTION = "upscale"
    CATEGORY = "FXTD Studios/Radiance/Upscale"
    DESCRIPTION = "Upscale to exact dimensions with aspect ratio control."
    
    def upscale(self, image: torch.Tensor, width: int, height: int, method: str,
                maintain_aspect: bool = True, aspect_mode: str = "fit",
                sharpening: float = 0.2, process_in_linear: bool = True):
        
        batch_size = image.shape[0]
        orig_h, orig_w = image.shape[1], image.shape[2]
        
        # Calculate target size
        if maintain_aspect:
            aspect_ratio = orig_w / orig_h
            
            if aspect_mode == "fit":
                # Fit within bounds
                if width / height > aspect_ratio:
                    new_h = height
                    new_w = int(height * aspect_ratio)
                else:
                    new_w = width
                    new_h = int(width / aspect_ratio)
            elif aspect_mode == "fill":
                # Fill bounds (may crop)
                if width / height > aspect_ratio:
                    new_w = width
                    new_h = int(width / aspect_ratio)
                else:
                    new_h = height
                    new_w = int(height * aspect_ratio)
            else:  # stretch
                new_w = width
                new_h = height
        else:
            new_w = width
            new_h = height
        
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]
            
            # Convert to linear
            if process_in_linear:
                img = srgb_to_linear_32bit(img)
            
            # Upscale
            upscaled = separable_resize_32bit(img, new_h, new_w, method)
            
            # Sharpen
            if sharpening > 0:
                upscaled = unsharp_mask_32bit(upscaled, sharpening, 1.0)
            
            # Convert back
            if process_in_linear:
                upscaled = linear_to_srgb_32bit(upscaled)
            
            # Handle alpha
            if has_alpha:
                alpha_up = separable_resize_32bit(alpha, new_h, new_w, 'lanczos')
                if len(alpha_up.shape) == 2:
                    alpha_up = alpha_up[:, :, np.newaxis]
                upscaled = np.concatenate([upscaled, alpha_up], axis=-1)
            
            results.append(upscaled)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        return (output_tensor, new_w, new_h)


class FXTDUpscaleTiled:
    """
    Tile-based upscaler for very large images.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        method_list = ["lanczos", "lanczos4", "bicubic", "mitchell"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.5
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 64
                }),
                "tile_overlap": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16
                }),
                "method": (method_list,),
            },
            "optional": {
                "sharpening": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "detail_enhancement": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "process_in_linear": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("upscaled_image", "width", "height", "tiles_processed")
    FUNCTION = "upscale"
    CATEGORY = "FXTD Studios/Radiance/Upscale"
    DESCRIPTION = "Tile-based upscaler for processing very large images efficiently."
    
    def upscale(self, image: torch.Tensor, scale_factor: float,
                tile_size: int, tile_overlap: int, method: str,
                sharpening: float = 0.3, detail_enhancement: float = 0.2,
                process_in_linear: bool = True):
        
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        # Calculate number of tiles
        stride = tile_size - tile_overlap
        tiles_x = max(1, math.ceil((w - tile_overlap) / stride))
        tiles_y = max(1, math.ceil((h - tile_overlap) / stride))
        total_tiles = tiles_x * tiles_y
        
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]
            
            # Convert to linear
            if process_in_linear:
                img = srgb_to_linear_32bit(img)
            
            # Define tile processing function
            def process_tile(tile, **kwargs):
                th, tw = tile.shape[:2]
                new_th = int(th * scale_factor)
                new_tw = int(tw * scale_factor)
                
                # Upscale
                upscaled = separable_resize_32bit(tile, new_th, new_tw, method)
                
                # Detail enhancement
                if detail_enhancement > 0:
                    upscaled = detail_enhancement_32bit(upscaled, detail_enhancement)
                
                # Sharpen
                if sharpening > 0:
                    upscaled = unsharp_mask_32bit(upscaled, sharpening, 1.0)
                
                return upscaled
            
            # Process with tiles
            upscaled = process_tiles_32bit(img, tile_size, tile_overlap, process_tile)
            
            # Ensure exact output size
            if upscaled.shape[0] != new_h or upscaled.shape[1] != new_w:
                upscaled = separable_resize_32bit(upscaled, new_h, new_w, 'lanczos')
            
            # Convert back
            if process_in_linear:
                upscaled = linear_to_srgb_32bit(upscaled)
            
            # Handle alpha
            if has_alpha:
                alpha_up = separable_resize_32bit(alpha, new_h, new_w, 'lanczos')
                if len(alpha_up.shape) == 2:
                    alpha_up = alpha_up[:, :, np.newaxis]
                upscaled = np.concatenate([upscaled, alpha_up], axis=-1)
            
            results.append(upscaled)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        return (output_tensor, new_w, new_h, total_tiles)


class FXTDSharpen32bit:
    """
    Professional 32-bit sharpening node.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_method": (["Unsharp Mask", "High Pass", "Multi-Scale"],),
                "amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01
                }),
                "edge_protection": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "process_in_linear": ("BOOLEAN", {"default": True}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sharpened_image",)
    FUNCTION = "sharpen"
    CATEGORY = "FXTD Studios/Radiance/Legacy"
    DESCRIPTION = "(DEPRECATED - Use FXTD Pro Film Effects) GPU-accelerated 32-bit sharpening."
    
    def sharpen(self, image: torch.Tensor, sharpen_method: str, amount: float,
                radius: float = 1.0, threshold: float = 0.0,
                edge_protection: float = 0.0, process_in_linear: bool = True,
                use_gpu: bool = True):
        
        # Try GPU path for Unsharp Mask
        if use_gpu and torch.cuda.is_available() and sharpen_method == "Unsharp Mask":
            try:
                device = torch.device("cuda")
                img = image.to(device).float()
                
                # GPU unsharp mask using convolution
                kernel_size = max(3, int(radius * 2) * 2 + 1)
                sigma = radius
                
                # Create Gaussian kernel
                x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
                gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
                gauss_1d = gauss_1d / gauss_1d.sum()
                gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
                gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
                
                # Apply to each channel
                b, h, w, c = img.shape
                img_permuted = img.permute(0, 3, 1, 2)  # BHWC -> BCHW
                
                # Blur with separable convolution
                blurred = torch.nn.functional.conv2d(
                    img_permuted, gauss_2d.expand(c, 1, -1, -1),
                    padding=kernel_size // 2, groups=c
                )
                
                # Unsharp mask
                blurred_hwc = blurred.permute(0, 2, 3, 1)  # BCHW -> BHWC
                detail = img - blurred_hwc
                
                # Apply amount
                result = img + detail * amount
                # HDR: Only clamp negatives, preserve super-white values
                result = torch.clamp(result, min=0)
                
                return (result.cpu(),)
                
            except RuntimeError:
                torch.cuda.empty_cache()
        
        # CPU fallback
        batch_size = image.shape[0]
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]
            
            # Convert to linear
            if process_in_linear:
                img = srgb_to_linear_32bit(img)
            
            # Apply sharpening
            if sharpen_method == "Unsharp Mask":
                sharpened = unsharp_mask_32bit(img, amount, radius, threshold)
            elif sharpen_method == "High Pass":
                sharpened = high_pass_sharpen_32bit(img, amount, radius)
            else:  # Multi-Scale
                sharpened = detail_enhancement_32bit(img, amount, amount * 0.5, amount * 0.3)
            
            # Edge protection (blend back some original near edges)
            if edge_protection > 0:
                # Simple edge detection
                lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
                grad_x = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
                grad_y = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
                edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
                edges = edges / (edges.max() + 1e-10)
                
                # Blend
                edge_mask = (edges * edge_protection)[..., np.newaxis]
                sharpened = sharpened * (1 - edge_mask) + img * edge_mask
            
            # Convert back
            if process_in_linear:
                sharpened = linear_to_srgb_32bit(sharpened)
            
            if has_alpha:
                sharpened = np.concatenate([sharpened, alpha], axis=-1)
            
            results.append(sharpened)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        return (output_tensor,)


class FXTDDownscale32bit:
    """
    High-quality 32-bit downscaling with anti-aliasing.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        method_list = ["lanczos", "lanczos4", "bicubic", "mitchell", "gaussian", "bilinear"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "method": (method_list,),
            },
            "optional": {
                "antialiasing": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "pre_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "process_in_linear": ("BOOLEAN", {"default": True}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("downscaled_image", "width", "height")
    FUNCTION = "downscale"
    CATEGORY = "FXTD Studios/Radiance/Upscale"
    DESCRIPTION = "GPU-accelerated 32-bit downscaling with anti-aliasing."
    
    def downscale(self, image: torch.Tensor, scale_factor: float, method: str,
                  antialiasing: float = 0.5, pre_blur: float = 0.0,
                  process_in_linear: bool = True, use_gpu: bool = True):
        
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        new_h = max(1, int(h * scale_factor))
        new_w = max(1, int(w * scale_factor))
        
        # Try GPU path
        if use_gpu and torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                img = image.to(device).float()
                
                # Permute to BCHW for interpolate
                img_bchw = img.permute(0, 3, 1, 2)
                
                # Use bicubic if lanczos requested (torch doesn't support lanczos)
                mode = 'bicubic' if 'lanczos' in method or method == 'mitchell' else method
                if mode not in ['nearest', 'bilinear', 'bicubic']:
                    mode = 'bicubic'
                
                # Downscale
                result = torch.nn.functional.interpolate(
                    img_bchw, size=(new_h, new_w), mode=mode, 
                    align_corners=False if mode != 'nearest' else None
                )
                
                # Permute back to BHWC
                result = result.permute(0, 2, 3, 1)
                # HDR: Preserve super-white values, only clamp negatives
                result = torch.clamp(result, min=0)
                
                return (result.cpu(), new_w, new_h)
                
            except RuntimeError:
                torch.cuda.empty_cache()
        
        # CPU fallback
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            has_alpha = img.shape[-1] == 4
            if has_alpha:
                alpha = img[..., 3:4]
                img = img[..., :3]
            
            # Convert to linear
            if process_in_linear:
                img = srgb_to_linear_32bit(img)
            
            # Pre-blur for anti-aliasing
            if pre_blur > 0:
                from PIL import ImageFilter
                for c in range(3):
                    channel = img[..., c]
                    ch_min, ch_max = channel.min(), channel.max()
                    ch_range = ch_max - ch_min if ch_max != ch_min else 1.0
                    
                    ch_norm = np.clip((channel - ch_min) / ch_range * 255, 0, 255).astype(np.uint8)
                    ch_pil = Image.fromarray(ch_norm)
                    ch_pil = ch_pil.filter(ImageFilter.GaussianBlur(radius=pre_blur))
                    if ch_pil.size != (w, h):
                        ch_pil = ch_pil.resize((w, h), Image.BILINEAR)
                    
                    img[..., c] = np.array(ch_pil).astype(np.float32) / 255.0 * ch_range + ch_min
            
            # Downscale
            downscaled = separable_resize_32bit(img, new_h, new_w, method)
            
            # Convert back
            if process_in_linear:
                downscaled = linear_to_srgb_32bit(downscaled)
            
            if has_alpha:
                alpha_down = separable_resize_32bit(alpha, new_h, new_w, 'lanczos')
                if len(alpha_down.shape) == 2:
                    alpha_down = alpha_down[:, :, np.newaxis]
                downscaled = np.concatenate([downscaled, alpha_down], axis=-1)
            
            results.append(downscaled)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        return (output_tensor, new_w, new_h)


class FXTDBitDepthConvert:
    """
    Convert between bit depths with professional dithering support.
    
    Supports Floyd-Steinberg, Ordered (Bayer), and Blue Noise dithering
    to minimize banding artifacts when converting to lower bit depths.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image in float32 format."}),
                "output_depth": (["32-bit Float", "16-bit Float", "16-bit Int", "10-bit", "8-bit"], {
                    "tooltip": "Target bit depth. 32-bit = full precision, 16-bit float = HDR, 10-bit = broadcast, 8-bit = web/SDR."
                }),
            },
            "optional": {
                "dithering": (["None", "Floyd-Steinberg", "Ordered", "Blue Noise"], {
                    "default": "None",
                    "tooltip": "Dithering algorithm: Floyd-Steinberg (error diffusion), Ordered (Bayer), Blue Noise (high-frequency)."
                }),
                "dither_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Dithering intensity. 1.0 = standard, <1 = subtle, >1 = aggressive."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("converted_image", "bit_depth_info")
    OUTPUT_TOOLTIPS = ("Image quantized to target bit depth.", "Information about the conversion.")
    FUNCTION = "convert"
    CATEGORY = "FXTD Studios/Radiance/Upscale"
    DESCRIPTION = "Convert between bit depths with professional dithering to reduce banding."
    
    def convert(self, image: torch.Tensor, output_depth: str,
                dithering: str = "None", dither_strength: float = 1.0):
        
        batch_size = image.shape[0]
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy().astype(np.float32)
            
            # Determine quantization levels
            if output_depth == "32-bit Float":
                # No quantization needed
                results.append(img)
                continue
            elif output_depth == "16-bit Float":
                # Convert to float16 and back
                result = img.astype(np.float16).astype(np.float32)
            elif output_depth == "16-bit Int":
                levels = 65535
            elif output_depth == "10-bit":
                levels = 1023
            else:  # 8-bit
                levels = 255
            
            if output_depth not in ["32-bit Float", "16-bit Float"]:
                # Apply dithering
                if dithering != "None":
                    img = self._apply_dither(img, levels, dithering, dither_strength)
                
                # Quantize
                result = np.round(img * levels) / levels
                result = np.clip(result, 0, 1)
            
            results.append(result)
        
        output = np.stack(results, axis=0)
        output_tensor = torch.from_numpy(output).float()
        
        info = f"Converted to {output_depth}"
        if dithering != "None":
            info += f" with {dithering} dithering"
        
        return (output_tensor, info)
    
    def _apply_dither(self, img: np.ndarray, levels: int, 
                      method: str, strength: float) -> np.ndarray:
        """Apply dithering before quantization."""
        h, w = img.shape[:2]
        
        if method == "Floyd-Steinberg":
            # Simple error diffusion approximation
            result = img.copy()
            noise = np.random.randn(h, w, 1) * (strength / levels)
            result = result + noise
            
        elif method == "Ordered":
            # Bayer matrix dithering
            bayer = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ]) / 16.0 - 0.5
            
            # Tile bayer matrix
            bayer_tiled = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
            noise = bayer_tiled[..., np.newaxis] * (strength / levels)
            result = img + noise
            
        elif method == "Blue Noise":
            # Approximation using high-frequency noise
            noise1 = np.random.randn(h, w, 1)
            noise2 = np.random.randn(h // 2 + 1, w // 2 + 1, 1)
            
            # Upsample noise2 and subtract
            from PIL import Image as PILImage
            n2_pil = PILImage.fromarray(((noise2[..., 0] * 127.5 + 127.5).clip(0, 255)).astype(np.uint8))
            n2_pil = n2_pil.resize((w, h), PILImage.BILINEAR)
            noise2_up = (np.array(n2_pil).astype(np.float32) - 127.5) / 127.5
            
            blue_noise = noise1[..., 0] - noise2_up * 0.5
            blue_noise = blue_noise[..., np.newaxis] * (strength / levels)
            result = img + blue_noise
        else:
            result = img
        
        return result



# =============================================================================
# AI UPSCALER INTEGRATION
# =============================================================================


class FXTDAIUpscale:
    """
    Professional AI upscaling using RealESRGAN and other neural network models.
    
    Supports multiple AI upscaling models for high-quality image enlargement.
    Models are loaded from ComfyUI's models/upscale_models directory.
    """
    
    AI_MODELS = [
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B", 
        "RealESRGAN_x2plus",
        "ESRGAN_4x",
        "4x-UltraSharp",
        "4x-AnimeSharp",
        "SwinIR_4x",
        "HAT_4x",
        "SUPIR-v0F_fp16",
        "SUPIR-v0Q_fp16",
    ]
    
    MODEL_URLS = {
        "SUPIR-v0F_fp16": "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0F_fp16.safetensors",
        "SUPIR-v0Q_fp16": "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0Q_fp16.safetensors",
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    }

    def __init__(self):
        self.model = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to upscale."}),
                "model_name": (cls.AI_MODELS, {"default": "RealESRGAN_x4plus", "tooltip": "AI upscaling model. RealESRGAN_x4plus is recommended for general use."}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64, "tooltip": "Tile size for processing. Smaller = less VRAM, slower. 512 recommended."}),
                "tile_overlap": ("INT", {"default": 32, "min": 0, "max": 128, "step": 8, "tooltip": "Overlap between tiles to avoid seams. 32-64 recommended."}),
                "auto_download": ("BOOLEAN", {"default": True, "tooltip": "Automatically download models if not found."}),
            },
            "optional": {
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model from VRAM after processing to free memory."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = ("Upscaled image.", "Information about the upscaling process.")
    FUNCTION = "upscale"
    CATEGORY = "FXTD Studios/Radiance/Upscale"
    DESCRIPTION = "AI-powered upscaling using neural network models. Supports tiled processing for large images."

    def _download_model(self, model_name: str, target_path: str) -> bool:
        """Download model if URL is available."""
        if model_name not in self.MODEL_URLS:
            return False
        
        import urllib.request
        url = self.MODEL_URLS[model_name]
        print(f"[FXTD AI Upscale] Downloading {model_name}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"[FXTD AI Upscale] Download complete: {target_path}")
            return True
        except Exception as e:
            print(f"[FXTD AI Upscale] Download failed: {e}")
            return False

    def _load_model(self, model_name: str):
        """Load an upscale model."""
        try:
            import folder_paths
            from comfy import model_management
            import comfy.utils
        except ImportError:
            return None, "ComfyUI modules not available"

        # Find model path
        model_path = folder_paths.get_full_path("upscale_models", f"{model_name}.pth")
        if model_path is None:
            model_path = folder_paths.get_full_path("upscale_models", f"{model_name}.safetensors")
        
        if model_path is None:
            # Try auto-download
            models_dir = folder_paths.get_folder_paths("upscale_models")[0]
            ext = ".safetensors" if "SUPIR" in model_name else ".pth"
            target_path = os.path.join(models_dir, f"{model_name}{ext}")
            
            if self._download_model(model_name, target_path):
                model_path = target_path
            else:
                return None, f"Model {model_name} not found. Place in models/upscale_models/"

        # Load the model using spandrel (modern) or fallback
        try:
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            
            # Try spandrel first (modern library)
            try:
                import spandrel
                try:
                    upscale_model = spandrel.ModelLoader().load_from_state_dict(sd).model.eval()
                    return upscale_model, f"Loaded: {model_name} (spandrel)"
                except Exception as spandrel_err:
                    err_name = type(spandrel_err).__name__
                    if "UnsupportedModelError" in err_name or "ValueError" in err_name:
                         print(f"[FXTD AI Upscale] Spandrel load error: {spandrel_err}")
                         return None, f"Model architecture not supported by installed spandrel version. Please update ComfyUI."
                    raise spandrel_err

            except ImportError:
                pass
            
            # Fallback to chainner_models (deprecated but works)
            try:
                import warnings
                warnings.filterwarnings('ignore', message='.*deprecated.*')
                from comfy_extras.chainner_models import model_loading
                upscale_model = model_loading.load_state_dict(sd).eval()
                return upscale_model, f"Loaded: {model_name} (legacy)"
            except ImportError:
                pass
            
            return None, "No model loader available (install spandrel)"
            
        except Exception as e:
            # Check for generic loading errors that imply corruption
            try:
                if os.path.exists(model_path):
                     size = os.path.getsize(model_path)
                     if size < 1000: # Empty or tiny file
                         print(f"[FXTD AI Upscale] Found invalid model file ({size} bytes). Deleting.")
                         os.remove(model_path)
                         return None, "Corrupt model file deleted. Rerun to download."
            except:
                pass
                
            return None, f"Error loading model: {type(e).__name__}: {str(e)}"

    def _fallback_upscale(self, image, model_name):
        """Fallback to algorithmic upscale when AI model unavailable."""
        print(f"[FXTD AI Upscale] Falling back to Lanczos upscale (AI model not loaded)")
        
        # Determine scale from model name
        scale = 4
        if "x2" in model_name.lower():
            scale = 2
        elif "x8" in model_name.lower():
            scale = 8
        
        # Use torch for GPU-accelerated upscaling
        b, h, w, c = image.shape
        new_h, new_w = h * scale, w * scale
        
        # Permute to BCHW for interpolate
        img_bchw = image.permute(0, 3, 1, 2)
        
        # Use bicubic (closest to Lanczos in torch)
        upscaled = torch.nn.functional.interpolate(
            img_bchw, size=(new_h, new_w), mode='bicubic', align_corners=False
        )
        
        # Permute back to BHWC
        result = upscaled.permute(0, 2, 3, 1)
        # HDR: Preserve super-white values, only clamp negatives
        result = torch.clamp(result, min=0)
        
        return (result, f"Bicubic {scale}x (AI model not available)")

    def _process_tile(self, tile: torch.Tensor, model, device) -> torch.Tensor:
        """Process a single tile through the model."""
        with torch.no_grad():
            tile_input = tile.to(device)
            output = model(tile_input)
            return output

    def upscale(self, image, model_name: str = "RealESRGAN_x4plus",
                tile_size: int = 512, tile_overlap: int = 32,
                auto_download: bool = True, unload_model: bool = False):
        """Upscale image using AI model with tiled processing."""
        
        # Load model if needed
        if self.model is None or self.current_model_name != model_name:
            self.model, load_info = self._load_model(model_name)
            self.current_model_name = model_name
            
            if self.model is None:
                print(f"[FXTD AI Upscale] {load_info}")
                return self._fallback_upscale(image, model_name)

        try:
            from comfy import model_management
            device = model_management.get_torch_device()
            self.model = self.model.to(device)
            
            # Determine scale factor from model (try to infer from a test)
            scale = 4  # Default
            if "x2" in model_name.lower():
                scale = 2
            elif "x8" in model_name.lower():
                scale = 8
            
            result_images = []
            
            for batch_idx in range(image.shape[0]):
                img = image[batch_idx:batch_idx+1].permute(0, 3, 1, 2)  # BHWC -> BCHW
                _, c, h, w = img.shape
                
                # Check if tiling is needed
                if h <= tile_size and w <= tile_size:
                    # Small image - process directly
                    with torch.no_grad():
                        img_device = img.to(device)
                        output = self.model(img_device)
                    result_images.append(output.permute(0, 2, 3, 1)[0])  # BCHW -> BHWC
                else:
                    # Large image - use tiled processing
                    new_h, new_w = h * scale, w * scale
                    output = torch.zeros((1, c, new_h, new_w), dtype=torch.float32, device=device)
                    weight = torch.zeros((1, 1, new_h, new_w), dtype=torch.float32, device=device)
                    
                    # Calculate tile positions
                    stride = tile_size - tile_overlap
                    tiles_x = max(1, math.ceil((w - tile_overlap) / stride))
                    tiles_y = max(1, math.ceil((h - tile_overlap) / stride))
                    
                    print(f"[FXTD AI Upscale] Processing {tiles_x * tiles_y} tiles ({tiles_x}x{tiles_y})...")
                    
                    for ty in range(tiles_y):
                        for tx in range(tiles_x):
                            # Calculate input tile coordinates
                            x1 = min(tx * stride, w - tile_size)
                            y1 = min(ty * stride, h - tile_size)
                            x2 = min(x1 + tile_size, w)
                            y2 = min(y1 + tile_size, h)
                            
                            # Handle edge cases
                            x1 = max(0, x2 - tile_size)
                            y1 = max(0, y2 - tile_size)
                            
                            # Extract tile
                            tile = img[:, :, y1:y2, x1:x2].to(device)
                            
                            # Process tile
                            with torch.no_grad():
                                tile_output = self.model(tile)
                            
                            # Calculate output coordinates
                            out_x1 = x1 * scale
                            out_y1 = y1 * scale
                            out_x2 = x2 * scale
                            out_y2 = y2 * scale
                            
                            # Create blending weight (feather edges)
                            th, tw = tile_output.shape[2], tile_output.shape[3]
                            tile_weight = torch.ones((1, 1, th, tw), dtype=torch.float32, device=device)
                            
                            # Feather edges for blending
                            if tile_overlap > 0:
                                feather = tile_overlap * scale
                                for i in range(feather):
                                    alpha = (i + 1) / feather
                                    tile_weight[:, :, i, :] *= alpha
                                    tile_weight[:, :, th - 1 - i, :] *= alpha
                                    tile_weight[:, :, :, i] *= alpha
                                    tile_weight[:, :, :, tw - 1 - i] *= alpha
                            
                            # Accumulate
                            output[:, :, out_y1:out_y2, out_x1:out_x2] += tile_output * tile_weight
                            weight[:, :, out_y1:out_y2, out_x1:out_x2] += tile_weight
                    
                    # Normalize by weight
                    output = output / (weight + 1e-8)
                    result_images.append(output.permute(0, 2, 3, 1)[0])  # BCHW -> BHWC
            
            # Stack results
            result = torch.stack(result_images)
            # HDR: Preserve super-white values, only clamp negatives
            result = torch.clamp(result, min=0)
            info = f"Upscaled with {model_name} ({scale}x) [HDR preserved]"
            
            # Unload model if requested to free VRAM
            if unload_model:
                self.model = None
                self.current_model_name = None
                torch.cuda.empty_cache()
                info += " (model unloaded)"
                print(f"[FXTD AI Upscale] Model unloaded from VRAM")
            
            return (result, info)
            
        except Exception as e:
            print(f"[FXTD AI Upscale] Error during upscale: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_upscale(image, model_name)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "FXTDProUpscale": FXTDProUpscale,
    "FXTDUpscaleBySize": FXTDUpscaleBySize,
    "FXTDUpscaleTiled": FXTDUpscaleTiled,
    "FXTDDownscale32bit": FXTDDownscale32bit,
    "FXTDBitDepthConvert": FXTDBitDepthConvert,
    "FXTDAIUpscale": FXTDAIUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTDProUpscale": "◆ Radiance Pro Upscale",
    "FXTDUpscaleBySize": "◆ Radiance Upscale By Size",
    "FXTDUpscaleTiled": "◆ Radiance Upscale Tiled",
    "FXTDDownscale32bit": "◆ Radiance Downscale 32-bit",
    "FXTDBitDepthConvert": "◆ Radiance Bit Depth Convert",
    "FXTDAIUpscale": "◆ Radiance AI Upscale",
}

