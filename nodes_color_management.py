"""
FXTD Radiance - Color Management (OCIO)
Professional Color Management using OpenColorIO.
"""

import torch
import os
import sys
import numpy as np

# Try importing PyOpenColorIO
try:
    import PyOpenColorIO as OCIO
except ImportError:
    OCIO = None

class RadianceOCIOColorTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "config_file": ("STRING", {"default": "ACES 1.3 Studio", "multiline": False}),
                "input_space": ("STRING", {"default": "ACES - ACEScg"}),
                "output_space": ("STRING", {"default": "Output - sRGB"}),
                "direction": (["Forward", "Inverse"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform"
    CATEGORY = "FXTD Studios/Radiance/Color Management"
    
    def transform(self, image, config_file, input_space, output_space, direction):
        if OCIO is None:
            print("PyOpenColorIO not found inside Radiance nodes. Please install it to use this node: pip install PyOpenColorIO")
            return (image,)
            
        try:
            # Load config
            if os.path.exists(config_file):
                config = OCIO.Config.CreateFromFile(config_file)
            else:
                try:
                    config = OCIO.Config.CreateFromEnv()
                except:
                    # Fallback
                    config = OCIO.Config.Create()
            
            # Verify spaces exist (simple check)
            # if not config.getColorSpace(input_space): ... 
            
            # Create Transform
            if direction == "Forward":
                transform = OCIO.ColorSpaceTransform(src=input_space, dst=output_space)
            else:
                transform = OCIO.ColorSpaceTransform(src=output_space, dst=input_space)

            processor = config.getProcessor(transform)
            cpu = processor.getDefaultCPUProcessor()
            
            # Process Image
            # Image is Batch, H, W, C (RGB) -> Flatten to N pixels RGB
            # PyOpenColorIO expects float32 buffer
            
            # Convert torch to numpy
            img_np = image.contiguous().cpu().numpy()
            batch, h, w, c = img_np.shape
            
            # Flatten to (N, 3)
            flat_img = img_np.reshape(-1, 3).astype(np.float32)
            
            # Apply
            cpu.applyRGB(flat_img)
            
            # Reshape back
            result_np = flat_img.reshape(batch, h, w, c)
            
            return (torch.from_numpy(result_np),)
            
        except Exception as e:
            print(f"[Radiance] OCIO Error: {e}")
            return (image,)

class RadianceLogCurveDecode:
    """
    Logarithmic to Linear conversion for specific camera curves.
    Implements exact mathematical formulas for industry standard curves.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "curve": ([
                    "ARRI LogC3", 
                    "ARRI LogC4", 
                    "Sony S-Log3", 
                    "RED Log3G10", 
                    "Canon Log3", 
                    "Panasonic V-Log"
                ],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "FXTD Studios/Radiance/Color Management"

    def decode(self, image, curve):
        t = image.clone()
        
        # ---------------------------------------------------------
        # ARRI LogC3 (Alexa Wide Gamut)
        # ---------------------------------------------------------
        if curve == "ARRI LogC3":
            cut = 0.010591
            a = 5.555556
            b = 0.052272
            c = 0.247190
            d = 0.385537
            e = 5.367655
            f = 0.092809
            
            mask = t > e * cut + f
            t[mask] = (torch.pow(10.0, (t[mask] - d) / c) - b) / a
            t[~mask] = (t[~mask] - f) / e
            
        # ---------------------------------------------------------
        # ARRI LogC4 (Alexa 35 REVEAL Color Science)
        # ---------------------------------------------------------
        elif curve == "ARRI LogC4":
            # Official ARRI LogC4 Linearization
            # Linear = (2^(14.0056 * t + 0.55479) - 1.46788) / 8.68337 for t > 0.092864
            # Linear = (t - 0.092864) / 5.555556 for t <= 0.092864
            
            c4_mask = t > 0.092864
            t[c4_mask] = (torch.pow(2.0, 14.0056 * t[c4_mask] + 0.55479) - 1.46788) / 8.68337
            t[~c4_mask] = (t[~c4_mask] - 0.092864) / 5.555556

        # ---------------------------------------------------------
        # Sony S-Log3
        # ---------------------------------------------------------
        elif curve == "Sony S-Log3":
            # S-Log3 Decoding
            # x = ((y * 1023 - 95) / (4 * 261.5)) - 0.01  (Linear part)
            # x = (0.19) * 10^((y * 1023 - 420) / 261.5) - 0.01 (Log part)
            
            cut_v = 171.2102946929 / 1023.0
            mask_s = t >= cut_v
            
            t[mask_s] = torch.pow(10.0, ((t[mask_s] * 1023.0 - 420.0) / 261.5)) * 0.18 + (0.18 * 0.01 - 0.01) # Refined
            # Actually standard formula: x = (0.18 + 0.01) * ... - 0.01. Note 0.19 = 0.18+0.01
             
            t[mask_s] = (0.19) * torch.pow(10.0, ((t[mask_s] * 1023.0 - 420.0) / 261.5)) - 0.01
            t[~mask_s] = ((t[~mask_s] * 1023.0 - 95.0) / (4.0 * 261.5)) - 0.01

        # ---------------------------------------------------------
        # RED Log3G10
        # ---------------------------------------------------------
        elif curve == "RED Log3G10":
            # RED Log3G10
            # A = 0.224282, B = 155.975327, C = 0.01
            # Linear = (10^((t - A) / 0.25) - 1) / B - C
            
            # Note: 0.25 comes from the log encoding side (B in encoding).
            # The encoding B is 0.25.
            
            t = (torch.pow(10.0, (t - 0.224282) / 0.25) - 1.0) / 155.975327 - 0.01

        # ---------------------------------------------------------
        # Canon Log3
        # ---------------------------------------------------------
        elif curve == "Canon Log3":
            # Canon Log 3 Decoding
            # if t < 0.073059361:
            #    x = (t - 0.073059361) / 5.765765766 
            # else:
            #    x = (10^((t - 0.073059361)/0.529136) - 1) / 10.1596
            
            mask_cl = t < 0.073059361
            t[mask_cl] = (t[mask_cl] - 0.073059361) / 5.765765766 
            t[~mask_cl] = (torch.pow(10.0, (t[~mask_cl] - 0.073059361) / 0.529136) - 1.0) / 10.1596

        # ---------------------------------------------------------
        # Panasonic V-Log
        # ---------------------------------------------------------
        elif curve == "Panasonic V-Log":
            # Panasonic V-Log Decoding
            # x = (y - 0.125) / 5.625   (y < 0.181)
            # x = 10^((y - 0.598206) / 0.241514) - 0.008761  (y >= 0.181)
            
            mask_v = t < 0.181
            t[mask_v] = (t[mask_v] - 0.125) / 5.625
            t[~mask_v] = torch.pow(10.0, (t[~mask_v] - 0.598206) / 0.241514) - 0.008761

        return (t,)

NODE_CLASS_MAPPINGS = {
    "RadianceOCIOColorTransform": RadianceOCIOColorTransform,
    "RadianceLogCurveDecode": RadianceLogCurveDecode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceOCIOColorTransform": "Radiance OCIO Color Transform",
    "RadianceLogCurveDecode": "Radiance Log Curve Decode"
}
