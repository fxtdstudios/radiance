"""
FXTD Resolution - Professional Resolution & Aspect Ratio Node
Version: 1.0.0
Author: FXTD Studios

Industry-standard resolution presets for:
- Social Media (Instagram, TikTok, YouTube, Twitter, Facebook, LinkedIn)
- Film & Cinema (4K DCI, 2K DCI, IMAX, Anamorphic scopes)
- Television (UHD, HD, SD formats)
- Photography (Common print sizes, megapixel targets)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, Any, List
import math


class FXTDResolution:
    """
    Professional Resolution Generator with Industry Presets
    
    Generates empty latent images or resolution specifications with:
    - Social media optimized sizes
    - Film/TV broadcast standards
    - Custom aspect ratio support
    - Visual preview with guidelines
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESOLUTION PRESETS
    # ═══════════════════════════════════════════════════════════════════════════
    
    PRESETS = {
        # ─────────────────────────────────────────────────────────────────────
        # SOCIAL MEDIA
        # ─────────────────────────────────────────────────────────────────────
        "── SOCIAL MEDIA ──": None,
        "Instagram Square (1:1)": (1080, 1080, "1:1"),
        "Instagram Portrait (4:5)": (1080, 1350, "4:5"),
        "Instagram Story/Reel (9:16)": (1080, 1920, "9:16"),
        "Instagram Landscape (1.91:1)": (1080, 566, "1.91:1"),
        "TikTok (9:16)": (1080, 1920, "9:16"),
        "YouTube Thumbnail (16:9)": (1280, 720, "16:9"),
        "YouTube Short (9:16)": (1080, 1920, "9:16"),
        "Twitter/X Post (16:9)": (1200, 675, "16:9"),
        "Twitter/X Header (3:1)": (1500, 500, "3:1"),
        "Facebook Post (1.91:1)": (1200, 628, "1.91:1"),
        "Facebook Cover (2.63:1)": (820, 312, "2.63:1"),
        "LinkedIn Post (1.91:1)": (1200, 627, "1.91:1"),
        "LinkedIn Banner (4:1)": (1584, 396, "4:1"),
        "Pinterest Pin (2:3)": (1000, 1500, "2:3"),
        "Snapchat (9:16)": (1080, 1920, "9:16"),
        
        # ─────────────────────────────────────────────────────────────────────
        # FILM & CINEMA
        # ─────────────────────────────────────────────────────────────────────
        "── FILM & CINEMA ──": None,
        "4K DCI (1.90:1)": (4096, 2160, "1.90:1"),
        "4K DCI Scope (2.39:1)": (4096, 1716, "2.39:1"),
        "4K DCI Flat (1.85:1)": (4096, 2214, "1.85:1"),
        "2K DCI (1.90:1)": (2048, 1080, "1.90:1"),
        "2K DCI Scope (2.39:1)": (2048, 858, "2.39:1"),
        "2K DCI Flat (1.85:1)": (2048, 1107, "1.85:1"),
        "IMAX (1.43:1)": (5616, 3924, "1.43:1"),
        "IMAX Digital (1.90:1)": (4096, 2160, "1.90:1"),
        "Anamorphic 2.39:1": (2880, 1206, "2.39:1"),
        "Anamorphic 2.76:1 Ultra Panavision": (2880, 1044, "2.76:1"),
        "Academy Ratio (1.375:1)": (2048, 1489, "1.375:1"),
        "VistaVision (1.5:1)": (3072, 2048, "1.5:1"),
        "Super 35 (1.85:1)": (2048, 1107, "1.85:1"),
        
        # ─────────────────────────────────────────────────────────────────────
        # TELEVISION & BROADCAST
        # ─────────────────────────────────────────────────────────────────────
        "── TELEVISION ──": None,
        "8K UHD (16:9)": (7680, 4320, "16:9"),
        "4K UHD (16:9)": (3840, 2160, "16:9"),
        "1080p Full HD (16:9)": (1920, 1080, "16:9"),
        "1080i HD (16:9)": (1920, 1080, "16:9"),
        "720p HD (16:9)": (1280, 720, "16:9"),
        "576p PAL (16:9)": (1024, 576, "16:9"),
        "480p NTSC (16:9)": (854, 480, "16:9"),
        "4:3 SD (4:3)": (640, 480, "4:3"),
        
        # ─────────────────────────────────────────────────────────────────────
        # PHOTOGRAPHY & PRINT
        # ─────────────────────────────────────────────────────────────────────
        "── PHOTOGRAPHY ──": None,
        "Full Frame 3:2": (3000, 2000, "3:2"),
        "Medium Format 4:3": (4000, 3000, "4:3"),
        "Square 1:1": (2000, 2000, "1:1"),
        "Panoramic 3:1": (3000, 1000, "3:1"),
        "Ultra-Wide 21:9": (2520, 1080, "21:9"),
        
        # ─────────────────────────────────────────────────────────────────────
        # AI/FLUX OPTIMIZED
        # ─────────────────────────────────────────────────────────────────────
        "── AI OPTIMIZED ──": None,
        "SDXL Square (1:1)": (1024, 1024, "1:1"),
        "SDXL Portrait (3:4)": (896, 1152, "3:4"),
        "SDXL Landscape (4:3)": (1152, 896, "4:3"),
        "SDXL Wide (16:9)": (1344, 768, "16:9"),
        "SDXL Tall (9:16)": (768, 1344, "9:16"),
        "FLUX 1MP Square": (1024, 1024, "1:1"),
        "FLUX 1MP Wide (16:9)": (1344, 768, "16:9"),
        "FLUX 1MP Portrait (9:16)": (768, 1344, "9:16"),
    }
    
    ASPECT_RATIOS = {
        "1:1": 1.0,
        "4:5": 0.8,
        "9:16": 0.5625,
        "16:9": 1.7778,
        "3:2": 1.5,
        "2:3": 0.6667,
        "4:3": 1.3333,
        "3:4": 0.75,
        "21:9": 2.3333,
        "1.85:1": 1.85,
        "2.39:1": 2.39,
        "2.76:1": 2.76,
        "1.43:1": 1.43,
        "1.90:1": 1.90,
        "1.91:1": 1.91,
    }
    
    MEGAPIXEL_TARGETS = ["0.5", "1.0", "1.5", "2.0", "4.0", "8.0", "Custom"]
    
    # Model types with their latent channel counts
    MODEL_TYPES = {
        "SDXL / SD1.5": 4,
        "Flux / SD3": 16,
    }
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_list = list(cls.PRESETS.keys())
        aspect_list = list(cls.ASPECT_RATIOS.keys())
        model_type_list = list(cls.MODEL_TYPES.keys())
        
        return {
            "required": {
                "preset": (preset_list, {"default": "FLUX 1MP Square"}),
            },
            "optional": {
                "model_type": (model_type_list, {"default": "Flux / SD3"}),
                "megapixels": (cls.MEGAPIXEL_TARGETS, {"default": "1.0"}),
                "custom_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "custom_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "custom_aspect": (aspect_list, {"default": "1:1"}),
                "use_custom": ("BOOLEAN", {"default": False}),
                "divisible_by": ([8, 16, 32, 64], {"default": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "swap_dimensions": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "preview", "width", "height", "info")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/Utilities"
    DESCRIPTION = """Professional Resolution Generator with industry presets:
• Social Media (Instagram, TikTok, YouTube, Twitter, etc.)
• Film & Cinema (4K DCI, 2K, IMAX, Anamorphic scopes)
• Television (8K/4K UHD, HD, SD)
• AI Optimized (SDXL, FLUX)
Outputs empty latent + visual preview."""
    
    def _round_to_divisible(self, value: int, divisor: int) -> int:
        """Round value to nearest multiple of divisor."""
        return round(value / divisor) * divisor
    
    def _create_preview_image(self, width: int, height: int, preset_name: str, 
                              aspect_str: str) -> torch.Tensor:
        """Create a visual preview image showing the resolution and aspect ratio."""
        # Create preview at reasonable size
        preview_max = 512
        scale = min(preview_max / width, preview_max / height)
        
        pw = int(width * scale)
        ph = int(height * scale)
        
        # Create image with dark background
        img = Image.new('RGB', (preview_max, preview_max), color=(20, 20, 28))
        draw = ImageDraw.Draw(img)
        
        # Calculate centered position for aspect ratio box
        x_offset = (preview_max - pw) // 2
        y_offset = (preview_max - ph) // 2
        
        # Draw aspect ratio frame
        # Outer glow
        for i in range(3, 0, -1):
            alpha = int(80 / i)
            draw.rectangle(
                [x_offset - i, y_offset - i, x_offset + pw + i, y_offset + ph + i],
                outline=(0, 136 + alpha, 255),
                width=1
            )
        
        # Main frame
        draw.rectangle(
            [x_offset, y_offset, x_offset + pw, y_offset + ph],
            outline=(220, 60, 60),
            width=2
        )
        
        # Fill with subtle gradient effect
        for y in range(y_offset + 2, y_offset + ph - 2):
            alpha = int(20 + 10 * (y - y_offset) / ph)
            draw.line([(x_offset + 2, y), (x_offset + pw - 2, y)], 
                     fill=(alpha, alpha, alpha + 10))
        
        # Draw center crosshair
        cx = x_offset + pw // 2
        cy = y_offset + ph // 2
        draw.line([(cx - 20, cy), (cx + 20, cy)], fill=(100, 100, 120), width=1)
        draw.line([(cx, cy - 20), (cx, cy + 20)], fill=(100, 100, 120), width=1)
        
        # Rule of thirds guides
        third_w = pw // 3
        third_h = ph // 3
        for i in range(1, 3):
            # Vertical lines
            x = x_offset + i * third_w
            draw.line([(x, y_offset), (x, y_offset + ph)], fill=(60, 60, 80), width=1)
            # Horizontal lines
            y = y_offset + i * third_h
            draw.line([(x_offset, y), (x_offset + pw, y)], fill=(60, 60, 80), width=1)
        
        # Try to use a nice font, fall back to default
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            info_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
        
        # Draw resolution text
        res_text = f"{width}x{height}"
        ratio_text = f"({aspect_str})"
        
        # Resolution - centered in box
        text_bbox = draw.textbbox((0, 0), res_text, font=title_font)
        text_w = text_bbox[2] - text_bbox[0]
        draw.text((cx - text_w // 2, cy - 20), res_text, fill=(220, 60, 60), font=title_font)
        
        # Aspect ratio - below resolution
        ratio_bbox = draw.textbbox((0, 0), ratio_text, font=info_font)
        ratio_w = ratio_bbox[2] - ratio_bbox[0]
        draw.text((cx - ratio_w // 2, cy + 10), ratio_text, fill=(0, 170, 255), font=info_font)
        
        # Bottom info bar
        info_text = f"Resolution: {width} x {height}"
        draw.text((10, preview_max - 25), info_text, fill=(180, 180, 200), font=info_font)
        
        # Convert to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)
    
    def generate(self, preset: str,
                 model_type: str = "Flux / SD3",
                 megapixels: str = "1.0",
                 custom_width: int = 1024,
                 custom_height: int = 1024,
                 custom_aspect: str = "1:1",
                 use_custom: bool = False,
                 divisible_by: int = 64,
                 batch_size: int = 1,
                 swap_dimensions: bool = False):
        
        # Handle category separators
        if preset.startswith("──"):
            preset = "SDXL Square (1:1)"
        
        # Determine dimensions
        if use_custom:
            width = custom_width
            height = custom_height
            aspect_str = custom_aspect
        elif preset in self.PRESETS and self.PRESETS[preset] is not None:
            width, height, aspect_str = self.PRESETS[preset]
        else:
            width, height, aspect_str = 1024, 1024, "1:1"
        
        # Apply megapixel scaling if not custom
        if not use_custom and megapixels != "Custom":
            mp = float(megapixels)
            current_mp = (width * height) / 1_000_000
            scale = math.sqrt(mp / current_mp)
            width = int(width * scale)
            height = int(height * scale)
        
        # Ensure divisibility
        width = self._round_to_divisible(width, divisible_by)
        height = self._round_to_divisible(height, divisible_by)
        
        # Swap dimensions if requested
        if swap_dimensions:
            width, height = height, width
        
        # Ensure minimum size
        width = max(divisible_by, width)
        height = max(divisible_by, height)
        
        # Get latent channels based on model type
        # SDXL/SD1.5 = 4 channels, Flux/SD3 = 16 channels
        latent_channels = self.MODEL_TYPES.get(model_type, 16)
        
        # Create empty latent with correct channel count
        latent = torch.zeros([batch_size, latent_channels, height // 8, width // 8])
        
        # Create preview image
        preview = self._create_preview_image(width, height, preset, aspect_str)
        
        # Build info string
        mp_actual = (width * height) / 1_000_000
        info = f"""═══ FXTD RESOLUTION ═══
Preset: {preset}
Model: {model_type}
Resolution: {width} x {height}
Aspect Ratio: {aspect_str}
Megapixels: {mp_actual:.2f} MP
Latent: {latent_channels}ch
═══════════════════════"""
        
        return {"ui": {"resolution": [f"{width}x{height}"]}, 
                "result": ({"samples": latent}, preview, width, height, info)}


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "FXTDResolution": FXTDResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTDResolution": "📐 Radiance Resolution",
}
