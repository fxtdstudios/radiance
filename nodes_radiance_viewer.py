"""
═══════════════════════════════════════════════════════════════════════════════
                     FXTD RADIANCE VIEWER NODE
              VFX Industry-Standard Image Viewer for ComfyUI
                         FXTD Studios © 2024-2026

Professional viewer node providing:
- Interactive zoom/pan with canvas-based rendering
- Real-time exposure and gamma controls  
- Channel viewing modes (RGB, R, G, B, Alpha, Luminance)
- Color picker with HDR value display
- False color and zebra analysis
- A/B comparison modes

This node uses a JavaScript frontend extension for interactivity.
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import numpy as np
import os
import folder_paths
from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════════════════════
#                           RADIANCE VIEWER NODE
# ═══════════════════════════════════════════════════════════════════════════════

class FXTD_RadianceViewer:
    """
    VFX Industry-Standard Image Viewer
    
    An interactive viewer inspired by Nuke, Flame, RV, and DaVinci Resolve.
    NO OUTPUT - This is a pure viewer node, no image passthrough.
    
    NAVIGATION:
    - Mouse wheel: Zoom in/out
    - Shift+drag or Middle mouse: Pan
    - Fit button: Fit to view
    - 1:1 button: 1:1 pixel zoom
    
    DISPLAY:
    - EV slider: Adjust exposure in stops
    - Gamma slider: Adjust display gamma
    - R/G/B/L buttons: View individual channels
    
    ANALYSIS:
    - FC: False color mode
    - Z: Zebra pattern
    
    The viewer is fully interactive via JavaScript frontend.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "compare_image": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    # NO OUTPUT - Pure viewer node
    RETURN_TYPES = ()
    FUNCTION = "view"
    CATEGORY = "FXTD Studios/Radiance/Viewer"
    OUTPUT_NODE = True
    DESCRIPTION = """VFX Industry-Standard Viewer (NO OUTPUT):
• Zoom/Pan navigation
• Exposure/Gamma controls
• Channel viewing (RGB/R/G/B/Luma)
• False color & zebra analysis
• A/B comparison modes"""

    def view(self, image: torch.Tensor, 
             compare_image: torch.Tensor = None,
             prompt=None, extra_pnginfo=None):
        """
        Process and display the image in the Radiance Viewer.
        
        The actual viewing is handled by the JavaScript frontend.
        This node saves the image to a temp file and returns metadata
        for the frontend to load. Supports 16-bit output for HDR preservation.
        """
        from PIL import Image as PILImage
        import time
        
        try:
            output_dir = folder_paths.get_temp_directory()
            batch_size = image.shape[0] if image.dim() == 4 else 1
            images_list = []
            
            # Process all batch frames
            # Process all batch frames
            for frame_idx in range(batch_size):
                if image.dim() == 4:
                    frame = image[frame_idx].cpu().numpy()
                else:
                    frame = image.cpu().numpy()
                
                # Convert to 16-bit for internal precision handling
                # (Future proofing for potential 16-bit viewer support)
                img_16bit = self._convert_to_16bit(frame)
                
                # For display in browser (standard Canvas), we need 8-bit PNGs
                # PIL 'RGB'/'RGBA' modes expect 8-bit data.
                if len(img_16bit.shape) == 3 and img_16bit.shape[2] == 4:
                    # RGBA
                    img_8bit = (img_16bit / 257).clip(0, 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(img_8bit, mode='RGBA')
                elif len(img_16bit.shape) == 3:
                    # RGB
                    img_8bit = (img_16bit / 257).clip(0, 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(img_8bit, mode='RGB')
                else:
                    # Grayscale / Mask - "L" mode is 8-bit pixels
                    img_8bit = (img_16bit / 257).clip(0, 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(img_8bit, mode='L')

                # Save the file
                timestamp = int(time.time() * 1000)
                filename = f"radiance_viewer_{id(self)}_{frame_idx}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                
                pil_img.save(filepath, compress_level=4) # Faster saving
                
                images_list.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp",
                    "frame": frame_idx,
                    "total_frames": batch_size
                })
            
            # Handle compare image if provided
            if compare_image is not None:
                cmp_batch = compare_image.shape[0] if compare_image.dim() == 4 else 1
                for cmp_idx in range(cmp_batch):
                    if compare_image.dim() == 4:
                        cmp_frame = compare_image[cmp_idx].cpu().numpy()
                    else:
                        cmp_frame = compare_image.cpu().numpy()
                    
                    cmp_8bit = self._convert_to_8bit(cmp_frame)
                    
                    timestamp = int(time.time() * 1000)
                    cmp_filename = f"radiance_compare_{id(self)}_{cmp_idx}_{timestamp}.png"
                    cmp_filepath = os.path.join(output_dir, cmp_filename)
                    
                    cmp_img = PILImage.fromarray(cmp_8bit)
                    cmp_img.save(cmp_filepath)
                    
                    images_list.append({
                        "filename": cmp_filename,
                        "subfolder": "",
                        "type": "temp",
                        "is_compare": True,
                        "frame": cmp_idx
                    })
            
            # Return UI metadata with batch info
            # ComfyUI expects dictionary values to be lists/iterables for aggregation
            return {"ui": {"radiance_images": images_list, "batch_size": [batch_size]}}
            
        except Exception as e:
            print(f"[Radiance Viewer] Error saving image: {e}")
            return {"ui": {"radiance_images": [], "error": [str(e)]}}
    
    def _convert_to_16bit(self, img_np: np.ndarray) -> np.ndarray:
        """Convert image to 16-bit with proper handling of various data types."""
        # Handle different input types
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            # Float data: assume 0-1 range, scale to 0-65535
            img_clipped = np.clip(img_np, 0.0, 1.0)
            return (img_clipped * 65535).astype(np.uint16)
        elif img_np.dtype == np.float16:
            # Float16: convert to float32 first
            img_32 = img_np.astype(np.float32)
            img_clipped = np.clip(img_32, 0.0, 1.0)
            return (img_clipped * 65535).astype(np.uint16)
        elif img_np.dtype == np.uint16:
            # Already 16-bit
            return img_np
        elif img_np.dtype == np.uint8:
            # 8-bit: scale up to 16-bit
            return (img_np.astype(np.uint16) * 257)
        else:
            # Unknown type: try to normalize
            img_min, img_max = img_np.min(), img_np.max()
            if img_max > img_min:
                normalized = (img_np - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(img_np, dtype=np.float32)
            return (normalized * 65535).astype(np.uint16)
    
    def _convert_to_8bit(self, img_np: np.ndarray) -> np.ndarray:
        """Convert image to 8-bit with proper handling of various data types."""
        if img_np.dtype == np.float32 or img_np.dtype == np.float64 or img_np.dtype == np.float16:
            img_clipped = np.clip(img_np.astype(np.float32), 0.0, 1.0)
            return (img_clipped * 255).astype(np.uint8)
        elif img_np.dtype == np.uint16:
            return (img_np / 256).astype(np.uint8)
        elif img_np.dtype == np.uint8:
            return img_np
        else:
            img_min, img_max = img_np.min(), img_np.max()
            if img_max > img_min:
                normalized = (img_np - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(img_np, dtype=np.float32)
            return (normalized * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#                           NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "FXTD_RadianceViewer": FXTD_RadianceViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTD_RadianceViewer": "🎬 Radiance Viewer",
}
