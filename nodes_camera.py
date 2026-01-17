"""
FXTD Camera Simulation - Professional Camera Effects for ComfyUI
Version: 1.0.0
Author: FXTD Studios

Realistic camera simulation nodes:
- White Balance (color temperature, tint)
- Depth of Field (bokeh blur with depth map)
- Motion Blur (directional, radial, zoom)
- Rolling Shutter (scanline warping)
- Compression Artifacts (JPEG, banding)
- Camera Shake (handheld jitter, Perlin noise)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Dict, Any, Optional
import math


# =============================================================================
# GPU UTILITY FUNCTIONS
# =============================================================================

def get_device(use_gpu: bool = True) -> torch.device:
    """Get the appropriate device."""
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_gaussian_blur(tensor: torch.Tensor, sigma: float, 
                      kernel_size: int = None) -> torch.Tensor:
    """GPU-accelerated Gaussian blur."""
    if sigma < 0.1:
        return tensor
    
    if kernel_size is None:
        kernel_size = int(sigma * 6) | 1
        kernel_size = max(3, min(kernel_size, 31))
    
    device = tensor.device
    dtype = tensor.dtype
    
    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    kernel_h = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1)
    
    was_bhwc = tensor.dim() == 4 and tensor.shape[-1] in [1, 3, 4]
    if was_bhwc:
        tensor = tensor.permute(0, 3, 1, 2)
    
    b, c, h, w = tensor.shape
    
    kernel_h = kernel_h.expand(c, 1, 1, kernel_size)
    kernel_v = kernel_v.expand(c, 1, kernel_size, 1)
    
    pad_h = kernel_size // 2
    pad_v = kernel_size // 2
    
    tensor_padded = torch.nn.functional.pad(tensor, (pad_h, pad_h, 0, 0), mode='reflect')
    blurred = torch.nn.functional.conv2d(tensor_padded, kernel_h, groups=c)
    
    blurred_padded = torch.nn.functional.pad(blurred, (0, 0, pad_v, pad_v), mode='reflect')
    blurred = torch.nn.functional.conv2d(blurred_padded, kernel_v, groups=c)
    
    if was_bhwc:
        blurred = blurred.permute(0, 2, 3, 1)
    
    return blurred


# =============================================================================
# COLOR TEMPERATURE UTILITIES
# =============================================================================

def kelvin_to_rgb(kelvin: float) -> Tuple[float, float, float]:
    """Convert color temperature in Kelvin to RGB multipliers."""
    kelvin = max(1000, min(40000, kelvin))
    temp = kelvin / 100.0
    
    # Red
    if temp <= 66:
        r = 1.0
    else:
        r = temp - 60
        r = 329.698727446 * (r ** -0.1332047592) / 255.0
        r = max(0, min(1, r))
    
    # Green
    if temp <= 66:
        g = temp
        g = 99.4708025861 * math.log(g) - 161.1195681661
        g = g / 255.0
    else:
        g = temp - 60
        g = 288.1221695283 * (g ** -0.0755148492) / 255.0
    g = max(0, min(1, g))
    
    # Blue
    if temp >= 66:
        b = 1.0
    elif temp <= 19:
        b = 0.0
    else:
        b = temp - 10
        b = 138.5177312231 * math.log(b) - 305.0447927307
        b = b / 255.0
        b = max(0, min(1, b))
    
    return (r, g, b)


# =============================================================================
# WHITE BALANCE NODE
# =============================================================================

class FXTDWhiteBalance:
    """
    Professional White Balance Control
    
    Adjust image color temperature and tint like a camera's white balance settings.
    Supports Kelvin temperature, tint shift, and preset modes.
    """
    
    WHITE_BALANCE_PRESETS = {
        "Custom": (5500, 0),
        "Daylight (5500K)": (5500, 0),
        "Cloudy (6500K)": (6500, 5),
        "Shade (7500K)": (7500, 5),
        "Tungsten (3200K)": (3200, 0),
        "Fluorescent (4000K)": (4000, 10),
        "Flash (5500K)": (5500, 0),
        "Candlelight (1850K)": (1850, 0),
        "Sunrise/Sunset (3000K)": (3000, 5),
        "Blue Hour (9000K)": (9000, -10),
        "Moonlight (4100K)": (4100, -5),
    }
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_list = list(cls.WHITE_BALANCE_PRESETS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (preset_list, {"default": "Daylight (5500K)"}),
                "temperature": ("INT", {
                    "default": 5500,
                    "min": 1000,
                    "max": 15000,
                    "step": 100,
                    "display": "slider"
                }),
                "tint": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "slider"
                }),
            },
            "optional": {
                "source_temperature": ("INT", {
                    "default": 5500,
                    "min": 1000,
                    "max": 15000,
                    "step": 100
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_white_balance"
    CATEGORY = "FXTD Studios/Radiance/Camera"
    DESCRIPTION = "Adjust white balance using color temperature (Kelvin) and tint."
    
    def apply_white_balance(self, image: torch.Tensor, preset: str,
                           temperature: int, tint: float,
                           source_temperature: int = 5500,
                           intensity: float = 1.0,
                           use_gpu: bool = True):
        
        # Use preset values if not Custom
        if preset != "Custom":
            temp_preset, tint_preset = self.WHITE_BALANCE_PRESETS[preset]
            temperature = temp_preset
            tint = tint_preset
        
        device = get_device(use_gpu)
        
        try:
            img = image.to(device).float()
            
            # Get RGB multipliers for source and target temperatures
            src_rgb = kelvin_to_rgb(source_temperature)
            tgt_rgb = kelvin_to_rgb(temperature)
            
            # Calculate correction multipliers
            r_mult = tgt_rgb[0] / (src_rgb[0] + 1e-6)
            g_mult = tgt_rgb[1] / (src_rgb[1] + 1e-6)
            b_mult = tgt_rgb[2] / (src_rgb[2] + 1e-6)
            
            # Normalize to maintain overall brightness
            avg_mult = (r_mult + g_mult + b_mult) / 3
            r_mult /= avg_mult
            g_mult /= avg_mult
            b_mult /= avg_mult
            
            # Apply tint (green-magenta shift)
            tint_factor = tint / 100.0
            g_mult *= (1.0 + tint_factor * 0.3)
            
            # Apply intensity
            r_mult = 1.0 + (r_mult - 1.0) * intensity
            g_mult = 1.0 + (g_mult - 1.0) * intensity
            b_mult = 1.0 + (b_mult - 1.0) * intensity
            
            # Apply multipliers
            output = img.clone()
            output[..., 0] *= r_mult
            output[..., 1] *= g_mult
            output[..., 2] *= b_mult
            
            output = torch.clamp(output, 0, 1)
            return (output.cpu(),)
            
        except RuntimeError:
            torch.cuda.empty_cache()
            return self.apply_white_balance(image, preset, temperature, tint,
                                           source_temperature, intensity, False)


# =============================================================================
# DEPTH OF FIELD NODE
# =============================================================================

class FXTDDepthOfField:
    """
    Cinematic Depth of Field
    
    Apply realistic depth of field blur using an optional depth map.
    Simulates camera bokeh with adjustable focus point and aperture.
    """
    
    BOKEH_SHAPES = ["Circle", "Hexagon", "Octagon", "Anamorphic Oval"]
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_amount": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider"
                }),
            },
            "optional": {
                "depth_map": ("IMAGE",),
                "focus_distance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "focus_range": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01
                }),
                "bokeh_shape": (cls.BOKEH_SHAPES, {"default": "Circle"}),
                "highlight_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1
                }),
                "foreground_blur": ("BOOLEAN", {"default": True}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_dof"
    CATEGORY = "FXTD Studios/Radiance/Camera"
    DESCRIPTION = "Apply cinematic depth of field blur with optional depth map input."
    
    def apply_dof(self, image: torch.Tensor, blur_amount: float,
                  depth_map: torch.Tensor = None,
                  focus_distance: float = 0.5,
                  focus_range: float = 0.1,
                  bokeh_shape: str = "Circle",
                  highlight_boost: float = 1.0,
                  foreground_blur: bool = True,
                  use_gpu: bool = True):
        
        if blur_amount < 0.1:
            return (image,)
        
        device = get_device(use_gpu)
        batch_size, h, w, c = image.shape
        
        try:
            img = image.to(device).float()
            
            # Create or use depth map
            if depth_map is not None:
                # Use provided depth map
                depth = depth_map.to(device).float()
                if depth.shape[-1] > 1:
                    depth = depth[..., 0:1]  # Use first channel
                depth = depth.mean(dim=-1, keepdim=False)  # (B, H, W)
                
                # Resize if needed
                if depth.shape[1:3] != (h, w):
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False
                    ).squeeze(1)
            else:
                # Create radial depth (center focused)
                y = torch.linspace(-1, 1, h, device=device)
                x = torch.linspace(-1, 1, w, device=device)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                depth = torch.sqrt(xx ** 2 + yy ** 2)
                depth = depth / depth.max()
                depth = depth.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Calculate blur strength based on depth
            depth_diff = torch.abs(depth - focus_distance)
            blur_mask = torch.clamp((depth_diff - focus_range) / (1 - focus_range + 1e-6), 0, 1)
            
            # Only blur foreground if enabled
            if not foreground_blur:
                foreground_mask = depth < focus_distance
                blur_mask = blur_mask * (~foreground_mask).float()
            
            # Apply multi-pass blur with varying strengths
            output = img.clone()
            
            # Create blur levels
            num_levels = 5
            for level in range(1, num_levels + 1):
                level_sigma = blur_amount * level / num_levels
                level_threshold = (level - 1) / num_levels
                
                # Blur the image
                blurred = gpu_gaussian_blur(img, level_sigma)
                
                # Blend based on blur mask
                level_mask = (blur_mask >= level_threshold).float().unsqueeze(-1)
                output = output * (1 - level_mask) + blurred * level_mask
            
            # Highlight boost (bokeh brightness)
            if highlight_boost > 1.0:
                luma = 0.2126 * output[..., 0] + 0.7152 * output[..., 1] + 0.0722 * output[..., 2]
                highlight_mask = (luma > 0.8) * blur_mask
                boost = 1.0 + (highlight_boost - 1.0) * highlight_mask.unsqueeze(-1)
                output = output * boost
            
            output = torch.clamp(output, 0, 1)
            return (output.cpu(),)
            
        except RuntimeError:
            torch.cuda.empty_cache()
            return self.apply_dof(image, blur_amount, depth_map, focus_distance,
                                 focus_range, bokeh_shape, highlight_boost, 
                                 foreground_blur, False)


# =============================================================================
# MOTION BLUR NODE
# =============================================================================

class FXTDMotionBlur:
    """
    Cinematic Motion Blur
    
    Apply directional, radial, or zoom motion blur to simulate camera/subject movement.
    """
    
    BLUR_TYPES = ["Directional", "Radial", "Zoom"]
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (cls.BLUR_TYPES, {"default": "Directional"}),
                "amount": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "slider"
                }),
            },
            "optional": {
                "angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "samples": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 4
                }),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_motion_blur"
    CATEGORY = "FXTD Studios/Radiance/Camera"
    DESCRIPTION = "Apply motion blur (directional, radial, or zoom) to simulate camera movement."
    
    def apply_motion_blur(self, image: torch.Tensor, blur_type: str, amount: float,
                          angle: float = 0.0, center_x: float = 0.5, center_y: float = 0.5,
                          samples: int = 16, use_gpu: bool = True):
        
        if amount < 0.5:
            return (image,)
        
        device = get_device(use_gpu)
        batch_size, h, w, c = image.shape
        
        try:
            img = image.to(device).float()
            output = torch.zeros_like(img)
            
            if blur_type == "Directional":
                # Directional motion blur
                angle_rad = math.radians(angle)
                dx = math.cos(angle_rad) * amount / w
                dy = math.sin(angle_rad) * amount / h
                
                for i in range(samples):
                    t = (i / (samples - 1)) - 0.5  # -0.5 to 0.5
                    offset_x = t * dx * 2
                    offset_y = t * dy * 2
                    
                    # Create translation grid
                    theta = torch.tensor([[[1, 0, offset_x], [0, 1, offset_y]]], 
                                        device=device, dtype=img.dtype)
                    theta = theta.expand(batch_size, -1, -1)
                    grid = torch.nn.functional.affine_grid(theta, img.permute(0, 3, 1, 2).shape, 
                                                           align_corners=False)
                    
                    sampled = torch.nn.functional.grid_sample(
                        img.permute(0, 3, 1, 2), grid,
                        mode='bilinear', padding_mode='border', align_corners=False
                    ).permute(0, 2, 3, 1)
                    
                    output += sampled
                
            elif blur_type == "Radial":
                # Radial/rotational blur around center
                cx = center_x * 2 - 1
                cy = center_y * 2 - 1
                
                for i in range(samples):
                    t = (i / (samples - 1)) - 0.5
                    rotation = t * amount * 0.02  # degrees to radians factor
                    
                    cos_r = math.cos(rotation)
                    sin_r = math.sin(rotation)
                    
                    # Rotation matrix around center point
                    theta = torch.tensor([
                        [[cos_r, -sin_r, cx * (1 - cos_r) + cy * sin_r],
                         [sin_r, cos_r, cy * (1 - cos_r) - cx * sin_r]]
                    ], device=device, dtype=img.dtype)
                    theta = theta.expand(batch_size, -1, -1)
                    
                    grid = torch.nn.functional.affine_grid(theta, img.permute(0, 3, 1, 2).shape,
                                                           align_corners=False)
                    
                    sampled = torch.nn.functional.grid_sample(
                        img.permute(0, 3, 1, 2), grid,
                        mode='bilinear', padding_mode='border', align_corners=False
                    ).permute(0, 2, 3, 1)
                    
                    output += sampled
                    
            else:  # Zoom
                # Zoom blur from center
                cx = center_x * 2 - 1
                cy = center_y * 2 - 1
                
                for i in range(samples):
                    t = (i / (samples - 1))
                    scale = 1.0 + (t - 0.5) * amount * 0.01
                    
                    theta = torch.tensor([
                        [[scale, 0, cx * (1 - scale)],
                         [0, scale, cy * (1 - scale)]]
                    ], device=device, dtype=img.dtype)
                    theta = theta.expand(batch_size, -1, -1)
                    
                    grid = torch.nn.functional.affine_grid(theta, img.permute(0, 3, 1, 2).shape,
                                                           align_corners=False)
                    
                    sampled = torch.nn.functional.grid_sample(
                        img.permute(0, 3, 1, 2), grid,
                        mode='bilinear', padding_mode='border', align_corners=False
                    ).permute(0, 2, 3, 1)
                    
                    output += sampled
            
            output = output / samples
            output = torch.clamp(output, 0, 1)
            return (output.cpu(),)
            
        except RuntimeError:
            torch.cuda.empty_cache()
            return self.apply_motion_blur(image, blur_type, amount, angle,
                                         center_x, center_y, samples, False)


# =============================================================================
# ROLLING SHUTTER NODE
# =============================================================================

class FXTDRollingShutter:
    """
    Rolling Shutter Simulation
    
    Simulate CMOS rolling shutter artifacts like skew, wobble, and flash banding.
    """
    
    SHUTTER_MODES = ["Horizontal", "Vertical", "Both"]
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "skew_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": -50.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "slider"
                }),
            },
            "optional": {
                "shutter_direction": (cls.SHUTTER_MODES, {"default": "Vertical"}),
                "wobble_frequency": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5
                }),
                "wobble_amplitude": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5
                }),
                "flash_band_position": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "flash_band_width": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01
                }),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_rolling_shutter"
    CATEGORY = "FXTD Studios/Radiance/Camera"
    DESCRIPTION = "Simulate rolling shutter artifacts (skew, wobble, flash banding)."
    
    def apply_rolling_shutter(self, image: torch.Tensor, skew_amount: float,
                              shutter_direction: str = "Vertical",
                              wobble_frequency: float = 0.0,
                              wobble_amplitude: float = 0.0,
                              flash_band_position: float = -1.0,
                              flash_band_width: float = 0.1,
                              use_gpu: bool = True):
        
        if abs(skew_amount) < 0.1 and wobble_amplitude < 0.1 and flash_band_position < -0.5:
            return (image,)
        
        device = get_device(use_gpu)
        batch_size, h, w, c = image.shape
        
        try:
            img = image.to(device).float()
            
            # Create coordinate grids
            y = torch.linspace(-1, 1, h, device=device, dtype=img.dtype)
            x = torch.linspace(-1, 1, w, device=device, dtype=img.dtype)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            
            if shutter_direction == "Vertical":
                # Vertical rolling shutter - each row shifts based on its y position
                offset_x = yy * skew_amount / w
                offset_y = torch.zeros_like(yy)
                
                # Add wobble
                if wobble_amplitude > 0:
                    offset_x += torch.sin(yy * wobble_frequency * math.pi * 2) * wobble_amplitude / w
                    
            elif shutter_direction == "Horizontal":
                # Horizontal rolling shutter
                offset_x = torch.zeros_like(xx)
                offset_y = xx * skew_amount / h
                
                if wobble_amplitude > 0:
                    offset_y += torch.sin(xx * wobble_frequency * math.pi * 2) * wobble_amplitude / h
            else:
                # Both
                offset_x = yy * skew_amount / w * 0.5
                offset_y = xx * skew_amount / h * 0.5
            
            # Apply transformation
            grid_x = xx + offset_x
            grid_y = yy + offset_y
            grid = torch.stack([grid_x, grid_y], dim=-1)
            grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            output = torch.nn.functional.grid_sample(
                img.permute(0, 3, 1, 2), grid,
                mode='bilinear', padding_mode='border', align_corners=False
            ).permute(0, 2, 3, 1)
            
            # Apply flash banding
            if flash_band_position >= -0.5:
                if shutter_direction == "Vertical":
                    scanline = yy
                else:
                    scanline = xx
                
                band_mask = torch.exp(-((scanline - flash_band_position) ** 2) / (flash_band_width ** 2))
                band_mask = band_mask.unsqueeze(0).unsqueeze(-1)
                output = output + band_mask * 0.3  # Flash brightness
            
            output = torch.clamp(output, 0, 1)
            return (output.cpu(),)
            
        except RuntimeError:
            torch.cuda.empty_cache()
            return self.apply_rolling_shutter(image, skew_amount, shutter_direction,
                                             wobble_frequency, wobble_amplitude,
                                             flash_band_position, flash_band_width, False)


# =============================================================================
# COMPRESSION ARTIFACTS NODE
# =============================================================================

class FXTDCompressionArtifacts:
    """
    Compression Artifacts Simulation
    
    Add realistic JPEG compression artifacts, color banding, and blocking.
    """
    
    ARTIFACT_TYPES = ["JPEG", "Banding", "Both"]
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "artifact_type": (cls.ARTIFACT_TYPES, {"default": "JPEG"}),
                "quality": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "block_size": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 32,
                    "step": 4
                }),
                "color_subsampling": ("BOOLEAN", {"default": True}),
                "banding_levels": ("INT", {
                    "default": 32,
                    "min": 4,
                    "max": 256,
                    "step": 4
                }),
                "noise_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.005
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_artifacts"
    CATEGORY = "FXTD Studios/Radiance/Camera"
    DESCRIPTION = "Add compression artifacts (JPEG blocking, color banding)."
    
    def apply_artifacts(self, image: torch.Tensor, artifact_type: str, quality: int,
                        block_size: int = 8, color_subsampling: bool = True,
                        banding_levels: int = 32, noise_amount: float = 0.0):
        
        batch_size = image.shape[0]
        results = []
        
        for b in range(batch_size):
            img = image[b].cpu().numpy()
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            
            if artifact_type in ["JPEG", "Both"]:
                # Apply JPEG compression
                import io
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=quality, 
                            subsampling=0 if not color_subsampling else 2)
                buffer.seek(0)
                pil_img = Image.open(buffer)
            
            result = np.array(pil_img).astype(np.float32) / 255.0
            
            if artifact_type in ["Banding", "Both"]:
                # Apply color banding (posterization)
                result = np.floor(result * banding_levels) / banding_levels
            
            # Add optional noise
            if noise_amount > 0:
                noise = np.random.randn(*result.shape) * noise_amount
                result = result + noise
            
            result = np.clip(result, 0, 1)
            results.append(result)
        
        output = torch.from_numpy(np.stack(results, axis=0)).float()
        return (output,)


# =============================================================================
# CAMERA SHAKE NODE
# =============================================================================

class FXTDCameraShake:
    """
    Camera Shake / Handheld Simulation
    
    Add realistic handheld camera shake using Perlin noise-based motion.
    """
    
    SHAKE_PRESETS = {
        "Custom": (0, 0, 0),
        "Subtle Handheld": (2.0, 0.5, 0.3),
        "Documentary": (4.0, 1.0, 0.5),
        "Action Cam": (8.0, 2.0, 1.0),
        "Earthquake": (20.0, 5.0, 3.0),
        "Vehicle Interior": (6.0, 3.0, 0.2),
        "Nervous Hold": (3.0, 2.0, 0.8),
    }
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_list = list(cls.SHAKE_PRESETS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (preset_list, {"default": "Subtle Handheld"}),
            },
            "optional": {
                "shake_x": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "shake_y": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1
                }),
                "frequency": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647
                }),
                "animate": ("BOOLEAN", {"default": True}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_shake"
    CATEGORY = "FXTD Studios/Radiance/Camera"
    DESCRIPTION = "Add handheld camera shake with translation and rotation."
    
    def apply_shake(self, image: torch.Tensor, preset: str,
                    shake_x: float = 2.0, shake_y: float = 2.0,
                    rotation: float = 0.5, frequency: float = 1.0,
                    seed: int = 0, animate: bool = True,
                    use_gpu: bool = True):
        
        # Apply preset
        if preset != "Custom":
            preset_values = self.SHAKE_PRESETS[preset]
            shake_x, shake_y, rotation = preset_values
        
        if shake_x < 0.1 and shake_y < 0.1 and rotation < 0.01:
            return (image,)
        
        device = get_device(use_gpu)
        batch_size, h, w, c = image.shape
        
        try:
            results = []
            
            for b in range(batch_size):
                img = image[b:b+1].to(device).float()
                
                # Generate smooth random offsets using Perlin-like noise
                frame_seed = seed + b if animate else seed
                np.random.seed(frame_seed)
                
                # Low-frequency noise components
                t = np.random.rand() * 1000
                offset_x = (np.sin(t * frequency) * 0.6 + 
                           np.sin(t * frequency * 2.3) * 0.3 +
                           np.sin(t * frequency * 5.1) * 0.1) * shake_x / w * 2
                offset_y = (np.sin(t * frequency * 1.1 + 0.5) * 0.6 +
                           np.sin(t * frequency * 2.7 + 0.3) * 0.3 +
                           np.sin(t * frequency * 4.9 + 0.7) * 0.1) * shake_y / h * 2
                rot_angle = (np.sin(t * frequency * 0.7) * 0.7 +
                            np.sin(t * frequency * 1.9) * 0.3) * rotation * 0.01
                
                # Create transformation matrix
                cos_r = math.cos(rot_angle)
                sin_r = math.sin(rot_angle)
                
                theta = torch.tensor([
                    [[cos_r, -sin_r, offset_x],
                     [sin_r, cos_r, offset_y]]
                ], device=device, dtype=img.dtype)
                
                grid = torch.nn.functional.affine_grid(theta, img.permute(0, 3, 1, 2).shape,
                                                       align_corners=False)
                
                output = torch.nn.functional.grid_sample(
                    img.permute(0, 3, 1, 2), grid,
                    mode='bilinear', padding_mode='border', align_corners=False
                ).permute(0, 2, 3, 1)
                
                results.append(output.cpu())
            
            output = torch.cat(results, dim=0)
            return (output,)
            
        except RuntimeError:
            torch.cuda.empty_cache()
            return self.apply_shake(image, preset, shake_x, shake_y,
                                   rotation, frequency, seed, animate, False)


# =============================================================================
# FXTD PHYSICAL CAMERA - UNIFIED PROFESSIONAL CAMERA SIMULATION
# =============================================================================

class FXTDPhysicalCamera:
    """
    Professional Physical Camera Simulation
    
    Industry-level camera node combining:
    - Exposure Triangle (Shutter Speed, Aperture, ISO)
    - Sensor Size (Full Frame, Super 35, APS-C, M4/3)
    - White Balance (Color Temperature, Tint)
    - Depth of Field (Bokeh, Focus)
    - Motion Blur (based on shutter angle)
    - Rolling Shutter artifacts
    - Camera Shake / Handheld
    
    Each effect can be individually enabled/disabled.
    """
    
    # Sensor sizes with crop factors (relative to Full Frame)
    SENSOR_SIZES = {
        "Full Frame (36x24mm)": 1.0,
        "Super 35 (24.89x18.66mm)": 1.4,
        "APS-C (23.6x15.6mm)": 1.5,
        "APS-C Canon (22.3x14.9mm)": 1.6,
        "Micro 4/3 (17.3x13mm)": 2.0,
        "1\" Sensor (13.2x8.8mm)": 2.7,
        "Super 16 (12.52x7.41mm)": 2.9,
    }
    
    # Common shutter speeds
    SHUTTER_SPEEDS = [
        "1/24", "1/25", "1/30", "1/48", "1/50", "1/60", "1/100", 
        "1/125", "1/250", "1/500", "1/1000", "1/2000", "1/4000", "1/8000"
    ]
    
    # Common apertures (f-stops)
    APERTURES = [
        "f/1.4", "f/2", "f/2.8", "f/4", "f/5.6", 
        "f/8", "f/11", "f/16", "f/22"
    ]
    
    # White balance presets
    WB_PRESETS = [
        "Daylight (5500K)", "Cloudy (6500K)", "Shade (7500K)",
        "Tungsten (3200K)", "Fluorescent (4000K)", "Custom"
    ]
    
    # Bokeh shapes
    BOKEH_SHAPES = ["Circle", "Hexagon", "Octagon", "Anamorphic"]
    
    # Motion blur types
    MOTION_TYPES = ["Directional", "Radial", "Zoom"]
    
    # Shake presets
    SHAKE_PRESETS = [
        "None", "Subtle Handheld", "Documentary", "Action Cam", "Vehicle Interior"
    ]
    
    # Camera body presets: (sensor, base_iso, color_science, default_shutter, default_aperture)
    CAMERA_BODY_PRESETS = {
        "Custom": None,
        "ARRI Alexa 35": {
            "sensor": "Super 35 (24.89x18.66mm)",
            "iso": 800,
            "shutter": "1/48",
            "aperture": "f/2.8",
            "color_temp": 5600,
            "description": "ARRI's flagship cinema camera with 4.6K sensor"
        },
        "ARRI Alexa Mini LF": {
            "sensor": "Full Frame (36x24mm)",
            "iso": 800,
            "shutter": "1/48",
            "aperture": "f/2",
            "color_temp": 5600,
            "description": "Large format ARRI with natural skin tones"
        },
        "RED V-Raptor 8K": {
            "sensor": "Full Frame (36x24mm)",
            "iso": 800,
            "shutter": "1/48",
            "aperture": "f/2.8",
            "color_temp": 5500,
            "description": "RED's 8K Vista Vision sensor camera"
        },
        "RED Komodo 6K": {
            "sensor": "Super 35 (24.89x18.66mm)",
            "iso": 800,
            "shutter": "1/48",
            "aperture": "f/4",
            "color_temp": 5600,
            "description": "Compact cinema camera with global shutter"
        },
        "Sony Venice 2": {
            "sensor": "Full Frame (36x24mm)",
            "iso": 500,
            "shutter": "1/48",
            "aperture": "f/2.8",
            "color_temp": 5500,
            "description": "Sony's flagship full-frame cinema camera"
        },
        "Sony FX6": {
            "sensor": "Full Frame (36x24mm)",
            "iso": 800,
            "shutter": "1/50",
            "aperture": "f/2.8",
            "color_temp": 5600,
            "description": "Compact full-frame cinema line camera"
        },
        "Blackmagic URSA Mini Pro 12K": {
            "sensor": "Super 35 (24.89x18.66mm)",
            "iso": 800,
            "shutter": "1/48",
            "aperture": "f/4",
            "color_temp": 5600,
            "description": "12K resolution with Blackmagic color science"
        },
        "Canon C500 Mark II": {
            "sensor": "Full Frame (36x24mm)",
            "iso": 800,
            "shutter": "1/48",
            "aperture": "f/2.8",
            "color_temp": 5600,
            "description": "Canon's full-frame cinema EOS camera"
        },
        "Panavision DXL2": {
            "sensor": "Full Frame (36x24mm)",
            "iso": 800,
            "shutter": "1/48",
            "aperture": "f/2.8",
            "color_temp": 5600,
            "description": "8K RED sensor with Panavision color science"
        },
    }
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # ═══════════════════════════════════════════════════════════
                # CAMERA BODY PRESET
                # ═══════════════════════════════════════════════════════════
                "camera_body": (list(cls.CAMERA_BODY_PRESETS.keys()), {
                    "default": "Custom",
                    "tooltip": "Select a professional cinema camera preset. Applies sensor size, ISO, shutter, aperture, and color temperature automatically."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # EXPOSURE TRIANGLE
                # ═══════════════════════════════════════════════════════════
                "shutter_speed": (cls.SHUTTER_SPEEDS, {
                    "default": "1/48",
                    "tooltip": "Shutter speed controls exposure time. 1/48 is cinema standard (180° shutter at 24fps). Faster speeds freeze motion, slower speeds add blur."
                }),
                "aperture": (cls.APERTURES, {
                    "default": "f/2.8",
                    "tooltip": "Lens aperture (f-stop). Lower values = wider aperture = more light + shallower depth of field. Higher values = sharper but darker."
                }),
                "iso": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 12800,
                    "step": 100,
                    "display": "slider",
                    "tooltip": "Sensor sensitivity. Higher ISO = brighter image but more noise. Base ISO for most cinema cameras is 800."
                }),
                "ev_compensation": ("FLOAT", {
                    "default": 0.0,
                    "min": -3.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Exposure Value compensation in stops. +1 = double brightness, -1 = half brightness."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # SENSOR
                # ═══════════════════════════════════════════════════════════
                "sensor_size": (list(cls.SENSOR_SIZES.keys()), {
                    "default": "Full Frame (36x24mm)",
                    "tooltip": "Camera sensor size affects crop factor and depth of field. Larger sensors = shallower DoF, wider field of view."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # WHITE BALANCE
                # ═══════════════════════════════════════════════════════════
                "enable_white_balance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable white balance color correction."
                }),
                "wb_preset": (cls.WB_PRESETS, {
                    "default": "Daylight (5500K)",
                    "tooltip": "White balance preset. Select lighting condition or use Custom for manual temperature control."
                }),
                "wb_temperature": ("INT", {
                    "default": 5500,
                    "min": 1000,
                    "max": 15000,
                    "step": 100,
                    "tooltip": "Color temperature in Kelvin. Lower = warmer/orange, Higher = cooler/blue. Daylight is ~5500K."
                }),
                "wb_tint": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Green-Magenta tint adjustment. Negative = more green, Positive = more magenta."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # DEPTH OF FIELD
                # ═══════════════════════════════════════════════════════════
                "enable_dof": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable depth of field blur simulation."
                }),
                "focus_distance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Focus distance (0=near, 1=far). Areas at this depth will be sharp."
                }),
                "dof_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Blur intensity for out-of-focus areas. Affected by aperture setting."
                }),
                "bokeh_shape": (cls.BOKEH_SHAPES, {
                    "default": "Circle",
                    "tooltip": "Shape of bokeh highlights. Circle is standard, Anamorphic creates oval bokeh."
                }),
                "depth_map": ("IMAGE",),
                
                # ═══════════════════════════════════════════════════════════
                # LENS EFFECTS (NEW)
                # ═══════════════════════════════════════════════════════════
                "enable_vignette": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable lens vignette (edge darkening)."
                }),
                "vignette_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Vignette intensity. Higher values create stronger edge darkening."
                }),
                "vignette_radius": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.3,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Vignette falloff radius. Lower = tighter vignette, Higher = extends further from center."
                }),
                "enable_chromatic_aberration": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable chromatic aberration (RGB color fringing at edges)."
                }),
                "ca_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Chromatic aberration intensity. Simulates color separation in cheaper lenses."
                }),
                "enable_lens_distortion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable barrel/pincushion lens distortion."
                }),
                "distortion_k1": ("FLOAT", {
                    "default": 0.0,
                    "min": -0.5,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Primary distortion coefficient. Negative = barrel (wide angle), Positive = pincushion (telephoto)."
                }),
                "distortion_k2": ("FLOAT", {
                    "default": 0.0,
                    "min": -0.3,
                    "max": 0.3,
                    "step": 0.01,
                    "tooltip": "Secondary distortion coefficient for complex lens profiles."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # MOTION BLUR
                # ═══════════════════════════════════════════════════════════
                "enable_motion_blur": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable motion blur based on shutter speed."
                }),
                "motion_type": (cls.MOTION_TYPES, {
                    "default": "Directional",
                    "tooltip": "Type of motion blur: Directional (linear), Radial (rotation), Zoom (dolly)."
                }),
                "motion_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 5.0,
                    "tooltip": "Direction of motion blur in degrees (for Directional type)."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # ROLLING SHUTTER
                # ═══════════════════════════════════════════════════════════
                "enable_rolling_shutter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable CMOS rolling shutter artifacts."
                }),
                "rs_skew": ("FLOAT", {
                    "default": 0.0,
                    "min": -30.0,
                    "max": 30.0,
                    "step": 1.0,
                    "tooltip": "Rolling shutter skew amount. Simulates the 'jello effect' from CMOS sensors."
                }),
                "rs_wobble": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Rolling shutter wobble frequency. Adds wavy distortion effect."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # CAMERA SHAKE
                # ═══════════════════════════════════════════════════════════
                "enable_shake": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable handheld camera shake simulation."
                }),
                "shake_preset": (cls.SHAKE_PRESETS, {
                    "default": "Subtle Handheld",
                    "tooltip": "Camera shake style preset."
                }),
                "shake_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Shake intensity multiplier."
                }),
                
                # ═══════════════════════════════════════════════════════════
                # PROCESSING
                # ═══════════════════════════════════════════════════════════
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU acceleration for processing. Disable if encountering VRAM issues."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed for shake animation. Same seed = reproducible results."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "camera_info")
    FUNCTION = "process"
    CATEGORY = "FXTD Studios/Radiance/Camera"
    DESCRIPTION = """Professional Physical Camera simulation with:
• Exposure Triangle (Shutter, Aperture, ISO)
• Sensor Size selection
• White Balance with temperature/tint
• Depth of Field with bokeh
• Motion Blur (based on shutter)
• Rolling Shutter artifacts
• Camera Shake simulation
Each effect can be enabled/disabled."""
    
    def _parse_shutter(self, shutter_str: str) -> float:
        """Parse shutter speed string to fraction value."""
        parts = shutter_str.replace("1/", "").strip()
        try:
            return 1.0 / float(parts)
        except:
            return 1/48
    
    def _parse_aperture(self, aperture_str: str) -> float:
        """Parse aperture string to f-number."""
        try:
            return float(aperture_str.replace("f/", "").strip())
        except:
            return 2.8
    
    def _apply_exposure(self, img: torch.Tensor, shutter: float, aperture: float, 
                        iso: int, ev_comp: float) -> torch.Tensor:
        """Apply exposure based on camera settings."""
        # Base exposure at 1/48, f/2.8, ISO 800 = 0 EV
        base_shutter = 1/48
        base_aperture = 2.8
        base_iso = 800
        
        # Calculate relative exposure in stops
        shutter_stops = math.log2(shutter / base_shutter)
        aperture_stops = 2 * math.log2(aperture / base_aperture)  # Aperture is f-stop squared
        iso_stops = math.log2(iso / base_iso)
        
        # Total exposure change
        ev_change = shutter_stops + aperture_stops + iso_stops + ev_comp
        
        # Apply exposure multiplier
        multiplier = 2.0 ** ev_change
        return img * multiplier
    
    def process(self, image: torch.Tensor,
                # Camera Body
                camera_body: str = "Custom",
                # Exposure
                shutter_speed: str = "1/48",
                aperture: str = "f/2.8",
                iso: int = 800,
                ev_compensation: float = 0.0,
                # Sensor
                sensor_size: str = "Full Frame (36x24mm)",
                # White Balance
                enable_white_balance: bool = True,
                wb_preset: str = "Daylight (5500K)",
                wb_temperature: int = 5500,
                wb_tint: float = 0.0,
                # DoF
                enable_dof: bool = False,
                focus_distance: float = 0.5,
                dof_strength: float = 5.0,
                bokeh_shape: str = "Circle",
                depth_map: torch.Tensor = None,
                # Lens Effects (NEW)
                enable_vignette: bool = False,
                vignette_strength: float = 0.3,
                vignette_radius: float = 0.8,
                enable_chromatic_aberration: bool = False,
                ca_strength: float = 1.0,
                enable_lens_distortion: bool = False,
                distortion_k1: float = 0.0,
                distortion_k2: float = 0.0,
                # Motion Blur
                enable_motion_blur: bool = False,
                motion_type: str = "Directional",
                motion_angle: float = 0.0,
                # Rolling Shutter
                enable_rolling_shutter: bool = False,
                rs_skew: float = 0.0,
                rs_wobble: float = 0.0,
                # Shake
                enable_shake: bool = False,
                shake_preset: str = "Subtle Handheld",
                shake_intensity: float = 1.0,
                # Processing
                use_gpu: bool = True,
                seed: int = 0):
        
        # Apply camera body preset if selected
        if camera_body != "Custom" and camera_body in self.CAMERA_BODY_PRESETS:
            preset = self.CAMERA_BODY_PRESETS[camera_body]
            if preset:
                sensor_size = preset.get("sensor", sensor_size)
                iso = preset.get("iso", iso)
                shutter_speed = preset.get("shutter", shutter_speed)
                aperture = preset.get("aperture", aperture)
                wb_temperature = preset.get("color_temp", wb_temperature)
        
        device = get_device(use_gpu)
        crop_factor = self.SENSOR_SIZES.get(sensor_size, 1.0)
        
        try:
            img = image.to(device).float()
            batch_size, h, w, c = img.shape
            
            # ═══════════════════════════════════════════════════════════
            # 1. EXPOSURE
            # ═══════════════════════════════════════════════════════════
            shutter_val = self._parse_shutter(shutter_speed)
            aperture_val = self._parse_aperture(aperture)
            img = self._apply_exposure(img, shutter_val, aperture_val, iso, ev_compensation)
            
            # ═══════════════════════════════════════════════════════════
            # 2. WHITE BALANCE
            # ═══════════════════════════════════════════════════════════
            if enable_white_balance:
                # Parse preset temperature
                if wb_preset != "Custom":
                    temp_match = wb_preset.split("(")[1].replace("K)", "").strip()
                    wb_temperature = int(temp_match)
                
                # Get RGB multipliers
                tgt_rgb = kelvin_to_rgb(wb_temperature)
                src_rgb = kelvin_to_rgb(5500)  # D55 reference
                
                r_mult = tgt_rgb[0] / (src_rgb[0] + 1e-6)
                g_mult = tgt_rgb[1] / (src_rgb[1] + 1e-6)
                b_mult = tgt_rgb[2] / (src_rgb[2] + 1e-6)
                
                # Normalize
                avg_mult = (r_mult + g_mult + b_mult) / 3
                r_mult /= avg_mult
                g_mult /= avg_mult
                b_mult /= avg_mult
                
                # Apply tint
                g_mult *= (1.0 + wb_tint / 100.0 * 0.3)
                
                img[..., 0] *= r_mult
                img[..., 1] *= g_mult
                img[..., 2] *= b_mult
            
            # ═══════════════════════════════════════════════════════════
            # 3. DEPTH OF FIELD
            # ═══════════════════════════════════════════════════════════
            if enable_dof and dof_strength > 0.1:
                # DoF strength affected by aperture and sensor size
                # Wider aperture (lower f-number) = more blur
                # Larger sensor = more blur
                effective_blur = dof_strength * (2.8 / aperture_val) / crop_factor
                
                if depth_map is not None:
                    depth = depth_map.to(device).float()
                    if depth.shape[-1] > 1:
                        depth = depth[..., 0:1]
                    depth = depth.mean(dim=-1, keepdim=False)
                    if depth.shape[1:3] != (h, w):
                        depth = torch.nn.functional.interpolate(
                            depth.unsqueeze(1), size=(h, w), mode='bilinear'
                        ).squeeze(1)
                else:
                    # Radial falloff
                    y = torch.linspace(-1, 1, h, device=device)
                    x = torch.linspace(-1, 1, w, device=device)
                    yy, xx = torch.meshgrid(y, x, indexing='ij')
                    depth = torch.sqrt(xx**2 + yy**2) / 1.414
                    depth = depth.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Create blur mask
                depth_diff = torch.abs(depth - focus_distance)
                blur_mask = torch.clamp(depth_diff * 3, 0, 1)
                
                # Apply blur
                blurred = gpu_gaussian_blur(img, effective_blur)
                blur_mask = blur_mask.unsqueeze(-1)
                img = img * (1 - blur_mask) + blurred * blur_mask
            
            # ═══════════════════════════════════════════════════════════
            # 4. LENS EFFECTS
            # ═══════════════════════════════════════════════════════════
            
            # A. LENS DISTORTION
            if enable_lens_distortion and (abs(distortion_k1) > 0.001 or abs(distortion_k2) > 0.001):
                y = torch.linspace(-1, 1, h, device=device, dtype=img.dtype)
                x = torch.linspace(-1, 1, w, device=device, dtype=img.dtype)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                
                # Convert to polar coordinates
                r2 = xx**2 + yy**2
                r = torch.sqrt(r2)
                
                # Apply polynomial distortion model: r_distorted = r * (1 + k1*r^2 + k2*r^4)
                # But for inverse mapping (grid_sample), we need to undistort or just apply as is depending on intent.
                # Standard lens distortion nodes usually warp output -> input.
                # To simulate barrel (k1 < 0), we essentially want to "zoom out" center, so we sample from further out?
                # Let's stick to standard formula for displacement.
                
                factor = 1.0 + distortion_k1 * r2 + distortion_k2 * (r2 ** 2)
                
                # Apply distortion to coordinates
                grid_x = xx * factor
                grid_y = yy * factor
                
                # Stack and expand
                grid = torch.stack([grid_x, grid_y], dim=-1)
                grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
                
                # Sample
                img = torch.nn.functional.grid_sample(
                    img.permute(0, 3, 1, 2), grid,
                    mode='bilinear', padding_mode='reflection', align_corners=False
                ).permute(0, 2, 3, 1)

            # B. CHROMATIC ABERRATION
            if enable_chromatic_aberration and ca_strength > 0:
                # Radial shift for R and B channels
                # R scales out, B scales in (or vice versa)
                scale_r = 1.0 + (0.002 * ca_strength)
                scale_b = 1.0 - (0.002 * ca_strength)
                
                # Base grid
                y = torch.linspace(-1, 1, h, device=device, dtype=img.dtype)
                x = torch.linspace(-1, 1, w, device=device, dtype=img.dtype)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
                
                # Red channel (Apply scale)
                grid_r = grid * scale_r
                r_chan = torch.nn.functional.grid_sample(
                    img[..., 0:1].permute(0, 3, 1, 2), grid_r,
                    mode='bilinear', padding_mode='border', align_corners=False
                ).permute(0, 2, 3, 1)
                
                # Blue channel (Apply scale)
                grid_b = grid * scale_b
                b_chan = torch.nn.functional.grid_sample(
                    img[..., 2:3].permute(0, 3, 1, 2), grid_b,
                    mode='bilinear', padding_mode='border', align_corners=False
                ).permute(0, 2, 3, 1)
                
                # Reassemble
                img = torch.cat([r_chan, img[..., 1:2], b_chan], dim=-1)

            # C. VIGNETTE
            if enable_vignette and vignette_strength > 0:
                y = torch.linspace(-1, 1, h, device=device)
                x = torch.linspace(-1, 1, w, device=device)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                
                # Distance from center
                dist = torch.sqrt(xx**2 + yy**2)
                
                # Smooth darkening at edges
                # vignette_radius controls where falloff begins
                # vignette_strength controls how dark it gets
                
                max_dist = 1.414  # Corner distance
                radius_scaled = vignette_radius * 1.0 # arbitrary scaling to feel right
                
                mask = 1.0 - torch.clamp((dist - radius_scaled) / (max_dist - radius_scaled + 0.001), 0, 1)
                mask = torch.pow(mask, vignette_strength * 2.0) # curved falloff
                
                # Apply strength
                # 0 strength = no change (mask=1), 1 strength = full effect
                # Actually standard formula: output = input * (1 - strength * (1-mask))
                # My mask creates 1 at center, 0 at edges.
                
                # Let's use cosine falloff for more natural look
                # cos(dist * intensity)
                
                falloff = dist * (1.0 / vignette_radius)
                falloff = torch.clamp(falloff, 0, 1)
                mask = 1.0 - (falloff * vignette_strength)
                mask = torch.clamp(mask, 0, 1)
                
                # Use a simpler polynomial falloff that looks good
                mask = 1.0 - (dist ** 2) * vignette_strength / (vignette_radius ** 2)
                mask = torch.clamp(mask, 0, 1)
                
                img = img * mask.unsqueeze(0).unsqueeze(-1)
            
            # ═══════════════════════════════════════════════════════════
            # 5. MOTION BLUR
            # ═══════════════════════════════════════════════════════════
            if enable_motion_blur:
                # Motion blur amount based on shutter angle (180° = standard)
                # Slower shutter = more blur
                base_motion = 10.0 * (shutter_val / (1/48))  # Relative to 1/48
                
                if motion_type == "Directional":
                    angle_rad = math.radians(motion_angle)
                    samples = 16
                    output = torch.zeros_like(img)
                    
                    for i in range(samples):
                        t = (i / (samples - 1)) - 0.5
                        dx = math.cos(angle_rad) * base_motion / w * t * 2
                        dy = math.sin(angle_rad) * base_motion / h * t * 2
                        
                        theta = torch.tensor([[[1, 0, dx], [0, 1, dy]]], 
                                            device=device, dtype=img.dtype)
                        theta = theta.expand(batch_size, -1, -1)
                        grid = torch.nn.functional.affine_grid(theta, img.permute(0,3,1,2).shape, 
                                                               align_corners=False)
                        sampled = torch.nn.functional.grid_sample(
                            img.permute(0,3,1,2), grid, mode='bilinear', 
                            padding_mode='border', align_corners=False
                        ).permute(0,2,3,1)
                        output += sampled
                    
                    img = output / samples
            
            # ═══════════════════════════════════════════════════════════
            # 5. ROLLING SHUTTER
            # ═══════════════════════════════════════════════════════════
            if enable_rolling_shutter and (abs(rs_skew) > 0.1 or rs_wobble > 0.1):
                y = torch.linspace(-1, 1, h, device=device, dtype=img.dtype)
                x = torch.linspace(-1, 1, w, device=device, dtype=img.dtype)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                
                # Vertical rolling shutter
                offset_x = yy * rs_skew / w
                if rs_wobble > 0:
                    offset_x += torch.sin(yy * 8 * math.pi) * rs_wobble / w
                
                grid_x = xx + offset_x
                grid_y = yy
                grid = torch.stack([grid_x, grid_y], dim=-1)
                grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
                
                img = torch.nn.functional.grid_sample(
                    img.permute(0,3,1,2), grid, mode='bilinear', 
                    padding_mode='border', align_corners=False
                ).permute(0,2,3,1)
            
            # ═══════════════════════════════════════════════════════════
            # 6. CAMERA SHAKE
            # ═══════════════════════════════════════════════════════════
            if enable_shake and shake_preset != "None":
                shake_values = {
                    "Subtle Handheld": (2.0, 0.5),
                    "Documentary": (4.0, 1.0),
                    "Action Cam": (8.0, 2.0),
                    "Vehicle Interior": (6.0, 0.2),
                }
                base_shake, base_rot = shake_values.get(shake_preset, (2.0, 0.5))
                shake_x = base_shake * shake_intensity
                shake_y = base_shake * shake_intensity
                rot = base_rot * shake_intensity
                
                results = []
                for b in range(batch_size):
                    frame_seed = seed + b
                    np.random.seed(frame_seed)
                    t = np.random.rand() * 1000
                    
                    offset_x = np.sin(t) * shake_x / w * 2
                    offset_y = np.sin(t * 1.1 + 0.5) * shake_y / h * 2
                    rot_angle = np.sin(t * 0.7) * rot * 0.01
                    
                    cos_r = math.cos(rot_angle)
                    sin_r = math.sin(rot_angle)
                    
                    theta = torch.tensor([
                        [[cos_r, -sin_r, offset_x],
                         [sin_r, cos_r, offset_y]]
                    ], device=device, dtype=img.dtype)
                    
                    grid = torch.nn.functional.affine_grid(
                        theta, img[b:b+1].permute(0,3,1,2).shape, align_corners=False
                    )
                    frame = torch.nn.functional.grid_sample(
                        img[b:b+1].permute(0,3,1,2), grid, mode='bilinear',
                        padding_mode='border', align_corners=False
                    ).permute(0,2,3,1)
                    results.append(frame)
                
                img = torch.cat(results, dim=0)
            
            # ═══════════════════════════════════════════════════════════
            # FINALIZE
            # ═══════════════════════════════════════════════════════════
            img = torch.clamp(img, 0, 1)
            
            # Build camera info string
            camera_info = f"""═══ PHYSICAL CAMERA SETTINGS ═══
Shutter: {shutter_speed}  Aperture: {aperture}  ISO: {iso}
EV Comp: {ev_compensation:+.1f}  Sensor: {sensor_size.split('(')[0].strip()}
Camera Body: {camera_body}

Effects Enabled:
  White Balance: {'✓' if enable_white_balance else '✗'} ({wb_temperature}K)
  Depth of Field: {'✓' if enable_dof else '✗'} (f/{aperture_val:.1f})
  Lens Effects: {'✓' if (enable_vignette or enable_chromatic_aberration or enable_lens_distortion) else '✗'}
  Motion Blur: {'✓' if enable_motion_blur else '✗'}
  Rolling Shutter: {'✓' if enable_rolling_shutter else '✗'}
  Camera Shake: {'✓' if enable_shake else '✗'} ({shake_preset})
═══════════════════════════════════"""
            
            return (img.cpu(), camera_info)
            
        except RuntimeError:
            torch.cuda.empty_cache()
            return self.process(image, camera_body, shutter_speed, aperture, iso, ev_compensation,
                              sensor_size, enable_white_balance, wb_preset, wb_temperature,
                              wb_tint, enable_dof, focus_distance, dof_strength, bokeh_shape,
                              depth_map, enable_vignette, vignette_strength, vignette_radius,
                              enable_chromatic_aberration, ca_strength, enable_lens_distortion,
                              distortion_k1, distortion_k2, enable_motion_blur, motion_type, 
                              motion_angle, enable_rolling_shutter, rs_skew, rs_wobble, 
                              enable_shake, shake_preset, shake_intensity, False, seed)


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "FXTDWhiteBalance": FXTDWhiteBalance,
    "FXTDDepthOfField": FXTDDepthOfField,
    "FXTDMotionBlur": FXTDMotionBlur,
    "FXTDRollingShutter": FXTDRollingShutter,
    "FXTDCompressionArtifacts": FXTDCompressionArtifacts,
    "FXTDCameraShake": FXTDCameraShake,
    "FXTDPhysicalCamera": FXTDPhysicalCamera,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTDWhiteBalance": "🎨 Radiance White Balance",
    "FXTDDepthOfField": "🔍 Radiance Depth of Field",
    "FXTDMotionBlur": "💨 Radiance Motion Blur",
    "FXTDRollingShutter": "📷 Radiance Rolling Shutter",
    "FXTDCompressionArtifacts": "📦 Radiance Compression Artifacts",
    "FXTDCameraShake": "📳 Radiance Camera Shake",
    "FXTDPhysicalCamera": "📷 Radiance Physical Camera",
}

