<div align="center">

![RADIANCE](docs/RADIANCE.png)


![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-GPL--3.0-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-purple)
![GPU](https://img.shields.io/badge/GPU-Accelerated-orange)

</div>

**Professional HDR Image Processing Suite for ComfyUI**

*Industry-grade color grading, film effects, and HDR processing*

[Installation](#installation) - [Node Reference](#node-reference) - [Features](#features) - [Documentation](https://fxtd.org/radiance_docs/)

---

## Features

- **Professional HDR Processing** - 32-bit floating-point pipeline
- **Film Effects** - 30+ camera sensors, 20+ film stocks
- **Industry Scopes** - Histogram, Waveform, Vectorscope
- **GPU Accelerated** - 10-50x faster with CUDA
- **Pro Viewer** - Flame/Nuke-style interactive viewer
- **Camera Simulation** - White balance, lens effects, presets
- **EXR/HDR Support** - Full OpenEXR read/write
- **Unified Loading** - Simplified model loading workflow

---

## Installation

### Option 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "radiance"
3. Click Install

### Option 2: Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FXTDStudio/radiance.git
pip install -r FXTDStudio_Radiance/requirements.txt
```

### Option 3: Manual Download
1. Download the latest release
2. Extract to `ComfyUI/custom_nodes/radiance`
3. Install dependencies: `pip install -r requirements.txt`

---

## Node Categories

| Category | Nodes | Description |
|----------|-------|-------------|
| **HDR** | 24 | Professional HDR processing |
| **Viewer** | 7 | Scopes and analysis tools |
| **Film** | 5 | Film grain and effects |
| **Upscale** | 7 | Professional upscaling |
| **EXR** | 3 | OpenEXR file support |
| **Camera** | 3 | Camera simulation |
| **Sampler** | 2 | Flux-optimized sampling |
| **Prompt** | 3 | Prompt engineering |
| **Loader** | 1 | Unified model loader |

**Total: 55 Nodes**

---

## Quick Start

### Basic HDR Workflow
```
Image -> ImageToFloat32 -> Float32ColorCorrect -> HDRToneMap -> SaveImageEXR
```

### Film Look Workflow
```
Image -> FXTDFilmGrain -> FXTDLensEffects -> FXTDFilmLook -> Output
```

### Pro Viewer Setup
```
Image -> Radiance Pro Viewer (FXTD)
```
Then use keyboard shortcuts: `F` fit, `1` 100%, `H` histogram, `W` waveform, `V` vectorscope

---

## Pro Viewer Shortcuts

| Key | Action |
|-----|--------|
| `F` | Fit to view |
| `1` | 100% zoom |
| `R/G/B` | View channel |
| `L` | Luminance |
| `C` | RGB (color) |
| `H` | Toggle histogram |
| `W` | Toggle waveform |
| `V` | Toggle vectorscope |
| `E` | False color |
| `Z` | Zebra pattern |
| `A` | A/B comparison |
| `P` | Fullscreen |

---

## HDR Nodes

### Image Processing
| Node | Description |
|------|-------------|
| Image to Float32 | Convert to 32-bit floating point |
| Float32 Color Correct | Professional color correction |
| HDR Expand Dynamic Range | Increase dynamic range |
| HDR Tone Map | 10+ tone mapping operators |
| GPU HDR Tone Map | GPU-accelerated tone mapping |

### Log Curves
| Node | Description |
|------|-------------|
| Log Curve Encode | ARRI LogC3/4, S-Log3, V-Log, ACEScct |
| Log Curve Decode | Decode log footage to linear |

### Color Spaces
| Node | Description |
|------|-------------|
| Color Space Convert | sRGB, ACEScg, Rec.2020, XYZ |
| ACES 2.0 Output Transform | Full ACES 2.0 pipeline |
| DaVinci Wide Gamut | DWG color space |
| ARRI Wide Gamut 4 | AWG4 color space |
| OCIO Color Transform | OpenColorIO integration |

### HDR Tools
| Node | Description |
|------|-------------|
| HDR Exposure Blend | Mertens, Laplacian pyramid |
| Shadow/Highlight Recovery | Lift shadows, recover highlights |
| HDR Histogram | HDR-aware histogram analysis |
| LUT Apply | Apply 3D LUTs (.cube, .3dl) |

### Export
| Node | Description |
|------|-------------|
| Save EXR (32-bit) | OpenEXR export |
| Load EXR | OpenEXR import |
| Save 16-bit PNG/TIFF | 16-bit export |
| Save HDRI | HDR environment map |
| HDR 360 Generate | 360 HDRI generation |

---

## Film Effect Nodes

| Node | Description |
|------|-------------|
| FXTD Film Grain | Photorealistic grain simulation |
| FXTD Grain Advanced | Fine-grained control |
| FXTD Lens Effects | Halation, bloom, chromatic aberration |
| FXTD Film Look | Complete film emulation |
| FXTD Pro Film Effects | All-in-one film processing |

### Camera Presets
RED, ARRI ALEXA, Sony Venice, Blackmagic URSA, Canon C300, Panasonic Varicam

### Film Stock Presets
Kodak Vision3 250D/500T, Kodak Portra 400/800, Fuji Eterna, Ilford HP5, CineStill 800T

---

## Upscale Nodes

| Node | Description |
|------|-------------|
| FXTD Pro Upscale | Professional Lanczos/Mitchell |
| FXTD Upscale By Size | Target resolution upscaling |
| FXTD Upscale Tiled | Large image tile processing |
| FXTD AI Upscale | RealESRGAN, SUPIR models |
| FXTD Sharpen 32-bit | GPU-accelerated sharpening |
| FXTD Downscale 32-bit | Anti-aliased downscaling |
| FXTD Bit Depth Convert | Dithered bit depth conversion |

---

## Viewer Nodes

| Node | Description |
|------|-------------|
| FXTD Master Viewer | Comprehensive image viewer |
| FXTD Scope Viewer | Histogram, waveform, vectorscope |
| FXTD Pixel Sampler | Color picker with HDR values |
| Radiance Grade (FXTD) | GPU-accelerated color grading |
| Radiance Viewer (FXTD) | Fast preview with scopes |
| Radiance Pro Viewer (FXTD) | Industry-level interactive viewer |
| Radiance Pro Viewer Advanced | Full analysis features |

---

## Sampler Nodes

| Node | Description |
|------|-------------|
| Radiance Sampler (FXTD) | Flux-optimized sampler |
| Radiance Sampler Pro (FXTD) | Advanced sampling control |

### Flux Features
- Native sigma shifting for high-res detail
- Integrated guidance control
- Optimized Euler + Simple scheduler

---

## Camera Nodes

| Node | Description |
|------|-------------|
| White Balance | Temperature/tint adjustment |
| Camera Simulation | Sensor characteristics, presets |
| Lens Simulation | Lens aberrations, vignette, distortion |

---

## Prompt Nodes

| Node | Description |
|------|-------------|
| Cinematic Prompt Machine | Cinematic prompt generator |
| Simple to Flux | Convert simple prompts for Flux |
| Prompt Presets | Saved prompt templates |

---

## Loaders

| Node | Description |
|------|-------------|
| Radiance Unified Loader | Single node for Checkpoint, CLIP, VAE |

---

## External Models

### [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
The node will automatically download models from HuggingFace to `models/depth_anything`.
- **Small**: Fast, good for video
- **Base**: Balanced
- **Large**: Best quality

### AI Upscale Models
- **[RealESRGAN](https://github.com/xinntao/Real-ESRGAN)**: x2/x4 upscaling (Standard & Anime)
- **[SUPIR](https://github.com/Fanghua-Yu/SUPIR)**: High-fidelity restoration (v0Q, v0F)

---

## GPU Acceleration

All critical operations are GPU-accelerated with automatic CPU fallback:

| Operation | GPU Speedup |
|-----------|-------------|
| Tone Mapping | 20-50x |
| Log Curves | 20x |
| Color Grading | 25x |
| Histogram | 10x |
| Waveform | 10x |
| Gaussian Blur | 20x |
| LUT Application | 10x |

---

## External Libraries

Radiance is built on industry-standard open source libraries:

- **[OpenColorIO (OCIO)](https://opencolorio.org/)**: Standard color management
- **[OpenEXR](https://openexr.com/)**: Professional HDR image format
- **[colour-science](https://www.colour-science.org/)**: Comprehensive color science for Python
- **[PyTorch](https://pytorch.org/)**: GPU acceleration and tensor operations
- **[OpenCV](https://opencv.org/)**: Real-time computer vision library

---

## Documentation

- **[Official Documentation](https://fxtd.org/radiance_docs/)**
- **Website**: [www.fxtd.org](https://www.fxtd.org)
- **LinkedIn**: [FXTD Studios](https://www.linkedin.com/company/fxtdstudios)

---

## Contributing

We welcome contributions from the community! Whether it's adding new features, fixing bugs, or improving documentation, your help is appreciated.

Please check our [Contributing Guide](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and code of conduct.

---

## License

GPL-3.0 License - See [LICENSE](LICENSE)

---

## Credits


**Core Technology**

- ComfyUI Architecture
- PyTorch (GPU Acceleration)
- OpenCV & OpenEXR
- Pillow (Image I/O)
- NumPy

**Color Science & Algorithms**

- **ACES Color System:** Academy of Motion Picture Arts and Sciences
- **Filmic Tone Mapping:** John Hable (Uncharted 2), Stephen Hill, and Krzysztof Narkowicz
- **AgX Color Transform:** Troy Sobotka
- **Reinhard Tone Mapping:** Erik Reinhard
- **Color Math:** Colour-Science for Python
- **3D LUT Processing:** Standard .cube format (Adobe/DaVinci)

**Camera & Film Emulation**

- **Digital Cinema Cameras:** ARRI (Alexa 35, ARRI IMAX), RED (V-Raptor, Komodo), Sony (Venice 2, FX6), Blackmagic (Pocket 4K/6K), Canon (C70, R5C), Panavision (DXL2)
- **Log Curves:** ARRI (LogC3/C4), Sony (S-Log3), RED (Log3G10), Panasonic (V-Log), Canon (Log3)
- **Film Stocks:** Kodak Vision3 (500T, 250D, 50D, 200T), Fujifilm Eterna (500T, Vivid), Kodak Ektachrome
- **Lens Presets:** Panavision C-Series, Cooke Anamorphic/i, Zeiss Supreme Prime, Canon K35, Lomo Round Front
- **ARRI Textures:** ARRI Look Library compatible grain/halation characteristics

**AI Models**

- **Depth Estimation:** Depth Anything V2 (HuggingFace Transformers)
- **Upscaling:** RealESRGAN, SUPIR

**Viewer Technology**

- JavaScript Canvas API
- WebGL (planned)
- HiDPI/Retina rendering


**Created by**

FXTD Studios

---

[Back to Top](#fxtd-studio-radiance)
