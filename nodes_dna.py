
import torch
import numpy as np
import json
import zlib
import struct
import time
from typing import Dict, Any, Tuple, Optional

# Get version from package
try:
    from . import __version__ as FXTD_VERSION
except ImportError:
    FXTD_VERSION = "3.2.0"

class FXTD_DigitalDNA:
    """
    Core engine for FXTD Digital DNA (Signature Architecture).
    Embeds invisible, lossless metadata into 32-bit floating point images.
    """
    
    # Magic header to identify FXTD DNA (32 bits)
    # "FXTD" in ASCII binary: 01000110 01011000 01010100 01000100
    MAGIC_HEADER = "01000110010110000101010001000100" 
    VERSION = "1.0"

    @staticmethod
    def _float_to_int_bits(f_val):
        """Reinterpret float32 bits as int32."""
        s = struct.pack('>f', f_val)
        return struct.unpack('>I', s)[0]

    @staticmethod
    def _int_bits_to_float(i_val):
        """Reinterpret int32 bits as float32."""
        s = struct.pack('>I', i_val)
        return struct.unpack('>f', s)[0]

    @classmethod
    def encode(cls, image: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Embed metadata into the image tensor's LSBs.
        Returns a new tensor with the embedded signature.
        """
        # Ensure we work with CPU numpy for bit manipulation
        device = image.device
        img_np = image.detach().cpu().numpy().astype(np.float32)
        
        # Prepare payload
        payload = {
            "dna_ver": cls.VERSION,
            "data": metadata
        }
        json_str = json.dumps(payload)
        # Compress to save space and obfuscate
        compressed = zlib.compress(json_str.encode('utf-8'))
        
        # Convert bytes to bit string
        bits = ''.join(f'{byte:08b}' for byte in compressed)
        
        # Add length prefix (32 bits) so we know when to stop reading
        length_bin = f'{len(bits):032b}'
        
        # Full bit stream: Header + Length + Data
        full_stream = cls.MAGIC_HEADER + length_bin + bits
        
        # Check capacity
        total_pixels = img_np.size
        if len(full_stream) > total_pixels:
            print(f"[FXTD DNA] Error: Image too small for signature ({len(full_stream)} bits > {total_pixels} pixels)")
            return image
            
        # Embed bits
        # Flatten image for easy indexing
        flat_img = img_np.reshape(-1)
        
        # Create a writable copy for modification
        # We manually manipulate the bits by viewing as int32
        # Note: In numpy, view() shares memory, but we want to modify values via int representation
        # Ideally, we do this element-wise or vectorized. Vectorized is faster.
        
        # Vectorized approach:
        # 1. Take the first N pixels where N is stream length
        # 2. Convert to int representation
        # 3. Set LSB to match stream
        # 4. Convert back to float
        
        stream_len = len(full_stream)
        target_indices = np.arange(stream_len)
        
        # Get float values
        float_vals = flat_img[target_indices]
        
        # Reinterpret as int32 (read-only usually if checking view)
        # We use frombuffer for safe reinterpretation
        int_vals = np.frombuffer(float_vals.tobytes(), dtype=np.uint32).copy()
        
        # Prepare bit mask (clear LSB)
        # 0xFFFFFFFE = ...11111110
        clear_mask = np.uint32(0xFFFFFFFE)
        int_vals &= clear_mask
        
        # Create additive bits from stream
        # Map '1' -> 1, '0' -> 0
        stream_bits = np.array([int(b) for b in full_stream], dtype=np.uint32)
        
        # Combine
        int_vals |= stream_bits
        
        # Convert back to float
        new_floats = np.frombuffer(int_vals.tobytes(), dtype=np.float32)
        
        # Apply back to flattened image
        flat_img[target_indices] = new_floats
        
        # Reshape back to original
        result_np = flat_img.reshape(img_np.shape)
        
        return torch.from_numpy(result_np).to(device)

    @classmethod
    def decode(cls, image: torch.Tensor) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Attempt to read FXTD DNA from an image.
        Returns (is_valid, metadata_dict).
        """
        img_np = image.detach().cpu().numpy().astype(np.float32)
        flat_img = img_np.reshape(-1)
        
        # Read header bits first
        header_len = len(cls.MAGIC_HEADER)
        if flat_img.size < header_len:
            return False, None
            
        # Get raw bits from first chunk
        # Reinterpret logic
        check_len = header_len + 32 # Header + Length
        # Safety check
        if flat_img.size < check_len:
            return False, None
            
        # Extract bits 
        float_vals = flat_img[:check_len]
        int_vals = np.frombuffer(float_vals.tobytes(), dtype=np.uint32)
        
        # Get LSBs
        lsbs = int_vals & 1
        
        # Reconstruct string
        extracted_bits = "".join(str(b) for b in lsbs)
        
        # Check specific magic header
        extracted_header = extracted_bits[:header_len]
        if extracted_header != cls.MAGIC_HEADER:
            return False, None
            
        # Read length
        length_bin = extracted_bits[header_len:header_len+32]
        try:
            payload_len = int(length_bin, 2)
        except ValueError:
            return False, None
            
        # Check if we have enough pixels for the full payload
        total_len = header_len + 32 + payload_len
        if flat_img.size < total_len:
            return False, None
            
        # Now read the full payload
        # Optimization: We already read the first chunk, but easier to read all needed now
        float_payload = flat_img[header_len+32 : total_len]
        int_payload = np.frombuffer(float_payload.tobytes(), dtype=np.uint32)
        payload_lsbs = int_payload & 1
        
        payload_bits = "".join(str(b) for b in payload_lsbs)
        
        # Convert bits back to bytes
        try:
            byte_array = bytearray()
            for i in range(0, len(payload_bits), 8):
                byte = payload_bits[i:i+8]
                if len(byte) == 8:
                    byte_array.append(int(byte, 2))
            
            # Decompress
            json_str = zlib.decompress(byte_array).decode('utf-8')
            data = json.loads(json_str)
            
            return True, data
        except Exception as e:
            # print(f"DNA Decode Error: {e}")
            return False, None


class FXTDSignatureMixin:
    """
    Mixin for ComfyUI Nodes to easily sign their output.
    """
    def sign_image(self, image: torch.Tensor, extra_metadata: Dict[str, Any] = None) -> torch.Tensor:
        """Sign the image with this node's signature."""
        
        # Basic node info
        node_class = self.__class__.__name__
        
        metadata = {
            "created_by": "FXTD Studio",
            "node": node_class,
            "timestamp": time.time(),
            "fxtd_ver": FXTD_VERSION
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
            
        return FXTD_DigitalDNA.encode(image, metadata)


# --------------------------------------------------------------------------------
#                                   NODES
# --------------------------------------------------------------------------------

class FXTDDNAReader:
    """
    Node to verify and read FXTD Digital DNA from images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_signed", "signature_data")
    FUNCTION = "read_dna"
    CATEGORY = "FXTD Studios/Radiance/Utilities"
    DESCRIPTION = "Verifies if an image contains a valid FXTD Digital DNA signature."

    def read_dna(self, image):
        # Handle batch - just check the first image for now
        # Ideally we check all, but metadata usually uniform across batch
        if image.dim() == 4:
            tgt_image = image[0]
        else:
            tgt_image = image
            
        is_valid, data = FXTD_DigitalDNA.decode(tgt_image)
        
        if is_valid:
            info_str = json.dumps(data.get("data", {}), indent=2)
            print(f"[FXTD DNA] Valid Signature Found: {info_str}")
        else:
            info_str = "No valid signature found."
            
        return (is_valid, info_str)
