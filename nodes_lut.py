"""
FXTD Radiance - LUT Nodes
Professional 3D LUT application and Color Matrix transformations.

Performance optimized with caching, vectorized operations, and comprehensive error handling.
"""

import torch
import numpy as np
import os
import re
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import hashlib


@dataclass
class LUTData:
    """Container for parsed LUT information with metadata."""
    lut_tensor: torch.Tensor
    size: int
    domain_min: List[float]
    domain_max: List[float]
    filepath: str
    file_hash: str


class LUTCache:
    """Global LUT cache to avoid re-parsing files."""
    _cache: Dict[str, LUTData] = {}
    _max_cache_size: int = 20
    
    @classmethod
    def get_file_hash(cls, filepath: str) -> str:
        """Generate hash of file for cache invalidation."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    @classmethod
    def get(cls, filepath: str) -> Optional[LUTData]:
        """Retrieve cached LUT if valid."""
        if filepath in cls._cache:
            cached = cls._cache[filepath]
            # Validate cache with file hash
            current_hash = cls.get_file_hash(filepath)
            if current_hash == cached.file_hash:
                return cached
            else:
                # File changed, remove from cache
                del cls._cache[filepath]
        return None
    
    @classmethod
    def set(cls, filepath: str, lut_data: LUTData):
        """Store LUT in cache with size management."""
        # Implement LRU-style eviction if cache is full
        if len(cls._cache) >= cls._max_cache_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(cls._cache))
            del cls._cache[oldest_key]
        cls._cache[filepath] = lut_data
    
    @classmethod
    def clear(cls):
        """Clear entire cache."""
        cls._cache.clear()


class RadianceLUTApply:
    """
    Apply a 3D LUT (.cube) to an image with professional-grade color transformation.
    
    Features:
    - Supports standard LUT sizes: 17³, 33³, 65³, and custom sizes
    - GPU-accelerated trilinear interpolation
    - Intelligent caching system for performance
    - Domain scaling support for HDR workflows
    - Multiple log space encodings (Log10, Log2, Ln)
    - Strength-based blending
    """
    
    # Supported log encodings
    LOG_ENCODINGS = ["Log10", "Log2", "Natural Log (Ln)"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (cls.get_lut_files(),),
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
                "log_space": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Log Input", 
                    "label_off": "Linear/sRGB Input"
                }),
            },
            "optional": {
                "log_encoding": (cls.LOG_ENCODINGS, {"default": "Log10"}),
                "clamp_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clamp output to 0-1 range. Disable for HDR workflows to preserve super-white values."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_lut"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = """Apply 3D LUT (.cube) to images with professional controls.
    
Features:
• Cached LUT parsing for performance
• GPU-accelerated trilinear interpolation
• Domain scaling for HDR support
• Multiple log space encodings
• Strength-based blending"""

    @staticmethod
    def get_lut_files() -> List[str]:
        """Retrieve available LUT files from standard locations."""
        try:
            import folder_paths
            luts = folder_paths.get_filename_list("luts")
            return luts if luts else ["No LUT files found - add .cube files to ComfyUI/models/luts"]
        except (ImportError, Exception):
            # Fallback: scan local directory
            luts_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "models", 
                "luts"
            )
            if os.path.exists(luts_dir):
                cube_files = [f for f in os.listdir(luts_dir) if f.lower().endswith('.cube')]
                return cube_files if cube_files else ["No .cube files found"]
            return ["Create directory: ComfyUI/models/luts"]

    @staticmethod
    def parse_cube_file(filepath: str) -> LUTData:
        """
        Parse .cube file and return LUT data with metadata.
        
        Supports:
        - Standard .cube format with LUT_3D_SIZE
        - DOMAIN_MIN and DOMAIN_MAX for HDR
        - Comments and metadata lines
        - Auto-detection of LUT size from data
        
        Raises:
            ValueError: If file format is invalid or data is corrupted
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LUT file not found: {filepath}")
        
        lut_data = []
        lut_size = None
        domain_min = [0.0, 0.0, 0.0]
        domain_max = [1.0, 1.0, 1.0]
        title = None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse metadata
                    if line.startswith('TITLE'):
                        title = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else None
                        continue
                    
                    if line.startswith('LUT_3D_SIZE'):
                        try:
                            lut_size = int(line.split()[1])
                            if lut_size < 2 or lut_size > 256:
                                raise ValueError(f"LUT size {lut_size} out of valid range [2, 256]")
                        except (IndexError, ValueError) as e:
                            raise ValueError(f"Invalid LUT_3D_SIZE on line {line_num}: {e}")
                        continue
                    
                    if line.startswith('DOMAIN_MIN'):
                        try:
                            domain_min = [float(x) for x in line.split()[1:4]]
                            if len(domain_min) != 3:
                                raise ValueError("DOMAIN_MIN must have 3 values")
                        except (IndexError, ValueError) as e:
                            raise ValueError(f"Invalid DOMAIN_MIN on line {line_num}: {e}")
                        continue
                    
                    if line.startswith('DOMAIN_MAX'):
                        try:
                            domain_max = [float(x) for x in line.split()[1:4]]
                            if len(domain_max) != 3:
                                raise ValueError("DOMAIN_MAX must have 3 values")
                        except (IndexError, ValueError) as e:
                            raise ValueError(f"Invalid DOMAIN_MAX on line {line_num}: {e}")
                        continue
                    
                    # Skip 1D LUT declarations (we only support 3D)
                    if line.startswith('LUT_1D_SIZE'):
                        continue
                    
                    # Parse RGB values
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            rgb = [float(x) for x in parts[:3]]
                            # Validate RGB values are reasonable
                            if any(abs(v) > 1000 for v in rgb):
                                print(f"Warning: Extreme RGB value on line {line_num}: {rgb}")
                            lut_data.append(rgb)
                        except ValueError:
                            # Skip malformed lines silently (could be metadata)
                            continue
            
            if not lut_data:
                raise ValueError("No valid RGB data found in LUT file")
            
            # Auto-detect size if not specified
            if lut_size is None:
                data_len = len(lut_data)
                lut_size = round(data_len ** (1/3))
                print(f"Auto-detected LUT size: {lut_size}³ from {data_len} entries")
            
            # Validate data length
            expected_len = lut_size ** 3
            if len(lut_data) != expected_len:
                raise ValueError(
                    f"LUT data length mismatch: found {len(lut_data)} entries, "
                    f"expected {expected_len} for size {lut_size}³"
                )
            
            # Validate domain ranges
            for i in range(3):
                if domain_min[i] >= domain_max[i]:
                    raise ValueError(
                        f"Invalid domain range: min[{i}]={domain_min[i]} >= max[{i}]={domain_max[i]}"
                    )
            
            # Convert to tensor and reshape to 3D grid
            lut_array = np.array(lut_data, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3)
            lut_tensor = torch.from_numpy(lut_array)
            
            # Generate file hash for caching
            file_hash = LUTCache.get_file_hash(filepath)
            
            return LUTData(
                lut_tensor=lut_tensor,
                size=lut_size,
                domain_min=domain_min,
                domain_max=domain_max,
                filepath=filepath,
                file_hash=file_hash
            )
            
        except Exception as e:
            raise ValueError(f"Error parsing LUT file '{filepath}': {str(e)}")

    @staticmethod
    @torch.jit.script
    def trilinear_interpolate_optimized(
        image: torch.Tensor, 
        lut: torch.Tensor, 
        lut_size: int,
        domain_min: torch.Tensor,
        domain_max: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized GPU trilinear interpolation with domain scaling.
        
        Uses vectorized operations and minimal tensor allocations.
        Supports HDR workflows via domain_min/domain_max scaling.
        
        Args:
            image: [B, H, W, C] tensor
            lut: [S, S, S, 3] lookup table
            lut_size: Size of LUT (S)
            domain_min: [3] minimum domain values
            domain_max: [3] maximum domain values
        
        Returns:
            Transformed image [B, H, W, 3]
        """
        # Scale input from domain range to [0, 1]
        domain_range = domain_max - domain_min
        coords_normalized = (image[..., :3] - domain_min) / domain_range
        coords_normalized = torch.clamp(coords_normalized, 0.0, 1.0)
        
        # Scale to LUT index space [0, lut_size-1]
        coords = coords_normalized * (lut_size - 1)
        
        # Integer and fractional parts
        coords_floor = torch.floor(coords).long()
        coords_floor = torch.clamp(coords_floor, 0, lut_size - 2)
        coords_ceil = coords_floor + 1
        coords_frac = coords - coords_floor.float()
        
        # Extract coordinates
        x0, y0, z0 = coords_floor[..., 0], coords_floor[..., 1], coords_floor[..., 2]
        x1, y1, z1 = coords_ceil[..., 0], coords_ceil[..., 1], coords_ceil[..., 2]
        xd, yd, zd = coords_frac[..., 0:1], coords_frac[..., 1:2], coords_frac[..., 2:3]
        
        # Gather 8 corner values (vectorized)
        c000 = lut[x0, y0, z0]
        c001 = lut[x0, y0, z1]
        c010 = lut[x0, y1, z0]
        c011 = lut[x0, y1, z1]
        c100 = lut[x1, y0, z0]
        c101 = lut[x1, y0, z1]
        c110 = lut[x1, y1, z0]
        c111 = lut[x1, y1, z1]
        
        # Trilinear interpolation (optimized with fused operations)
        # Interpolate along x-axis
        c00 = c000 + (c100 - c000) * xd
        c01 = c001 + (c101 - c001) * xd
        c10 = c010 + (c110 - c010) * xd
        c11 = c011 + (c111 - c011) * xd
        
        # Interpolate along y-axis
        c0 = c00 + (c10 - c00) * yd
        c1 = c01 + (c11 - c01) * yd
        
        # Interpolate along z-axis
        result = c0 + (c1 - c0) * zd
        
        return result

    @staticmethod
    def convert_from_log_space(image: torch.Tensor, log_encoding: str) -> torch.Tensor:
        """
        Convert image from log space to linear.
        
        Args:
            image: Input tensor in log space
            log_encoding: Type of log encoding used
        
        Returns:
            Linear space tensor
        """
        if log_encoding == "Log10":
            return torch.pow(10.0, image)
        elif log_encoding == "Log2":
            return torch.pow(2.0, image)
        elif log_encoding == "Natural Log (Ln)":
            return torch.exp(image)
        else:
            # Fallback to Log10
            return torch.pow(10.0, image)

    def apply_lut(
        self, 
        image: torch.Tensor, 
        lut_file: str, 
        strength: float, 
        log_space: bool,
        log_encoding: str = "Log10",
        clamp_output: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        Apply 3D LUT transformation to image.
        
        Args:
            image: Input image tensor [B, H, W, C]
            lut_file: Name of .cube file
            strength: Blend strength (0=original, 1=full LUT)
            log_space: Whether input is in log space
            log_encoding: Type of log encoding if log_space=True
            clamp_output: Whether to clamp output to [0, 1]
        
        Returns:
            Tuple containing transformed image
        """
        # Early exit for zero strength
        if strength == 0.0:
            return (image,)
        
        # Resolve LUT file path
        try:
            import folder_paths
            lut_path = folder_paths.get_full_path("luts", lut_file)
        except (ImportError, Exception):
            luts_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "models", 
                "luts"
            )
            lut_path = os.path.join(luts_dir, lut_file)
        
        if not os.path.exists(lut_path):
            print(f"ERROR: LUT file not found: {lut_path}")
            return (image,)
        
        # Try to get from cache
        lut_data = LUTCache.get(lut_path)
        
        # Parse if not cached
        if lut_data is None:
            try:
                lut_data = self.parse_cube_file(lut_path)
                LUTCache.set(lut_path, lut_data)
                print(f"Loaded and cached LUT: {lut_file} (size: {lut_data.size}³)")
            except Exception as e:
                print(f"ERROR: Failed to parse LUT file '{lut_file}': {e}")
                return (image,)
        
        # Prepare image
        device = image.device
        img_processed = image.clone()
        
        # Convert from log space if needed
        if log_space:
            img_processed = self.convert_from_log_space(img_processed, log_encoding)
        
        # Move LUT to same device as image
        lut_tensor = lut_data.lut_tensor.to(device)
        
        # Prepare domain tensors
        domain_min = torch.tensor(lut_data.domain_min, dtype=torch.float32, device=device)
        domain_max = torch.tensor(lut_data.domain_max, dtype=torch.float32, device=device)
        
        # Apply LUT using optimized trilinear interpolation
        result = self.trilinear_interpolate_optimized(
            img_processed,
            lut_tensor,
            lut_data.size,
            domain_min,
            domain_max
        )
        
        # Preserve alpha channel if present
        if image.shape[-1] > 3:
            result = torch.cat([result, img_processed[..., 3:]], dim=-1)
        
        # Blend based on strength
        if strength < 1.0:
            result = image * (1.0 - strength) + result * strength
        
        # Clamp output to valid range (only if requested - disabled by default for HDR)
        if clamp_output:
            result = torch.clamp(result, 0.0, 1.0)
        else:
            # HDR: Only clamp negatives to preserve super-whites
            result = torch.clamp(result, min=0.0)
        
        return (result,)


class RadianceGPUColorMatrix:
    """
    Apply custom color matrix transformations with professional controls.
    
    Features:
    - 3x3 RGB and 4x4 RGBA matrix support
    - Pre-built matrix presets (Sepia, B&W, etc.)
    - Optional offset/bias addition
    - Robust input parsing with validation
    - GPU-accelerated batch processing
    """
    
    # Preset matrices for common color operations
    MATRIX_PRESETS = {
        "Custom": None,
        "Identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Sepia": [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        "B&W (Average)": [[0.333, 0.333, 0.333], [0.333, 0.333, 0.333], [0.333, 0.333, 0.333]],
        "B&W (Luminance)": [[0.299, 0.587, 0.114], [0.299, 0.587, 0.114], [0.299, 0.587, 0.114]],
        "Invert": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        "Channel Swap (RBG)": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (list(cls.MATRIX_PRESETS.keys()), {"default": "Custom"}),
                "matrix_type": (["RGB (3x3)", "RGBA (4x4)"], {"default": "RGB (3x3)"}),
                "r_vector": ("STRING", {"default": "1.0, 0.0, 0.0", "multiline": False}),
                "g_vector": ("STRING", {"default": "0.0, 1.0, 0.0", "multiline": False}),
                "b_vector": ("STRING", {"default": "0.0, 0.0, 1.0", "multiline": False}),
            },
            "optional": {
                "a_vector": ("STRING", {"default": "0.0, 0.0, 0.0, 1.0", "multiline": False}),
                "offset": ("STRING", {"default": "0.0, 0.0, 0.0", "multiline": False}),
                "clamp_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clamp output to 0-1 range. Disable for HDR workflows to preserve super-white values."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_matrix"
    CATEGORY = "FXTD Studios/Radiance/Color"
    DESCRIPTION = """Apply custom color matrix transformation.
    
Features:
• Pre-built presets (Sepia, B&W, Invert, etc.)
• Custom 3x3 or 4x4 matrices
• Optional offset/bias
• GPU-accelerated processing
• Robust input validation"""

    @staticmethod
    def parse_vector(vector_str: str, expected_len: int = 3) -> List[float]:
        """
        Parse comma or space-separated string to float list with validation.
        
        Args:
            vector_str: Input string (e.g., "1.0, 0.5, 0.2" or "1.0 0.5 0.2")
            expected_len: Expected number of values
        
        Returns:
            List of parsed float values
        
        Raises:
            ValueError: If parsing fails or length doesn't match
        """
        try:
            # Handle both comma and space-separated values
            # Remove extra whitespace and split
            cleaned = re.sub(r'\s+', ' ', vector_str.strip())
            cleaned = cleaned.replace(',', ' ')
            parts = cleaned.split()
            
            values = [float(x) for x in parts]
            
            if len(values) != expected_len:
                raise ValueError(
                    f"Expected {expected_len} values, got {len(values)}. "
                    f"Input: '{vector_str}'"
                )
            
            # Validate for extreme values that might indicate errors
            for i, v in enumerate(values):
                if abs(v) > 1000:
                    print(f"Warning: Extreme value {v} at position {i} in vector '{vector_str}'")
            
            return values
            
        except ValueError as e:
            print(f"ERROR parsing vector '{vector_str}': {e}")
            # Return identity/zero vector as safe fallback
            if expected_len == 3:
                return [1.0, 0.0, 0.0] if "r" in vector_str.lower() else \
                       [0.0, 1.0, 0.0] if "g" in vector_str.lower() else \
                       [0.0, 0.0, 1.0]
            elif expected_len == 4:
                return [0.0, 0.0, 0.0, 1.0]
            else:
                return [0.0] * expected_len

    def apply_matrix(
        self, 
        image: torch.Tensor, 
        preset: str,
        matrix_type: str, 
        r_vector: str, 
        g_vector: str, 
        b_vector: str, 
        a_vector: Optional[str] = None, 
        offset: Optional[str] = None,
        clamp_output: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        Apply color matrix transformation to image.
        
        Args:
            image: Input image [B, H, W, C]
            preset: Matrix preset name or "Custom"
            matrix_type: "RGB (3x3)" or "RGBA (4x4)"
            r_vector: Red channel coefficients
            g_vector: Green channel coefficients
            b_vector: Blue channel coefficients
            a_vector: Alpha channel coefficients (for 4x4 only)
            offset: Color bias to add after matrix multiplication
            clamp_output: Whether to clamp result to [0, 1]
        
        Returns:
            Tuple containing transformed image
        """
        device = image.device
        B, H, W, C = image.shape
        
        # Use preset if selected
        if preset != "Custom" and preset in self.MATRIX_PRESETS:
            preset_matrix = self.MATRIX_PRESETS[preset]
            if preset_matrix:
                r_vals, g_vals, b_vals = preset_matrix
            else:
                # Parse custom vectors
                r_vals = self.parse_vector(r_vector, 3)
                g_vals = self.parse_vector(g_vector, 3)
                b_vals = self.parse_vector(b_vector, 3)
        else:
            # Parse custom vectors
            r_vals = self.parse_vector(r_vector, 3)
            g_vals = self.parse_vector(g_vector, 3)
            b_vals = self.parse_vector(b_vector, 3)
        
        # Reshape image for efficient batch matrix multiplication
        img_flat = image.reshape(-1, C)
        
        if matrix_type == "RGB (3x3)":
            # Build 3x3 matrix
            matrix = torch.tensor(
                [r_vals, g_vals, b_vals], 
                dtype=torch.float32, 
                device=device
            )
            
            # Apply matrix: result = image @ matrix.T
            # Only transform RGB channels
            rgb_channels = img_flat[..., :3]
            result = torch.matmul(rgb_channels, matrix.T)
            
            # Add offset if provided
            if offset is not None and offset.strip():
                try:
                    offset_vals = self.parse_vector(offset, 3)
                    offset_tensor = torch.tensor(
                        offset_vals, 
                        dtype=torch.float32, 
                        device=device
                    )
                    result = result + offset_tensor
                except Exception as e:
                    print(f"Warning: Failed to parse offset '{offset}': {e}")
            
            # Preserve alpha channel if present
            if C > 3:
                result = torch.cat([result, img_flat[..., 3:]], dim=-1)
            
        else:  # RGBA (4x4)
            # Parse alpha vector
            if a_vector is not None and a_vector.strip():
                a_vals = self.parse_vector(a_vector, 4)
            else:
                a_vals = [0.0, 0.0, 0.0, 1.0]
            
            # Build 4x4 matrix
            matrix = torch.tensor([
                r_vals + [0.0],
                g_vals + [0.0],
                b_vals + [0.0],
                a_vals
            ], dtype=torch.float32, device=device)
            
            # Ensure image has 4 channels
            if C == 3:
                alpha = torch.ones((img_flat.shape[0], 1), dtype=torch.float32, device=device)
                img_flat = torch.cat([img_flat, alpha], dim=-1)
            
            # Apply matrix
            result = torch.matmul(img_flat, matrix.T)
            
            # Add offset if provided
            if offset is not None and offset.strip():
                try:
                    offset_parts = self.parse_vector(offset, -1)  # Accept any length
                    # Pad to 4 if needed
                    while len(offset_parts) < 4:
                        offset_parts.append(0.0)
                    offset_vals = offset_parts[:4]
                    
                    offset_tensor = torch.tensor(
                        offset_vals, 
                        dtype=torch.float32, 
                        device=device
                    )
                    result = result + offset_tensor
                except Exception as e:
                    print(f"Warning: Failed to parse offset '{offset}': {e}")
            
            # Match original channel count
            if C == 3:
                result = result[..., :3]
        
        # Reshape back to image dimensions
        result = result.reshape(B, H, W, -1)
        
        # Clamp to valid range if requested (disabled by default for HDR)
        if clamp_output:
            result = torch.clamp(result, 0.0, 1.0)
        else:
            # HDR: Only clamp negatives to preserve super-whites
            result = torch.clamp(result, min=0.0)
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "RadianceLUTApply": RadianceLUTApply,
    "RadianceGPUColorMatrix": RadianceGPUColorMatrix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLUTApply": "◆ Radiance LUT Apply",
    "RadianceGPUColorMatrix": "◆ Radiance GPU Color Matrix"
}