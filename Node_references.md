# FXTD Studio Radiance - Node Reference

Complete documentation for all 53 nodes in the FXTD Studio Radiance suite.

---

## Table of Contents

- [HDR Processing Nodes](#hdr-processing-nodes)
- [Viewer Nodes](#viewer-nodes)
- [Upscale Nodes](#upscale-nodes)
- [Film & Lens Effect Nodes](#film--lens-effect-nodes)
- [Camera Simulation Nodes](#camera-simulation-nodes)
- [EXR Export Nodes](#exr-export-nodes)
- [Prompt Engineering Nodes](#prompt-engineering-nodes)

---

## HDR Processing Nodes

### 🎨 Image to Float32
**Node ID:** `ImageToFloat32`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

Convert image tensor to 32-bit floating point precision. Ensures full HDR range is preserved without clamping.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | - | Input image |
| normalize | BOOLEAN | False | Normalize to 0-1 range |
| source_gamma | FLOAT | 2.2 | Source gamma for linearization |

**Output:** IMAGE (float32)

---

### 🎨 Float32 Color Correct
**Node ID:** `Float32ColorCorrect`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

Professional color correction in 32-bit float space. Preserves full dynamic range.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | - | Input image |
| exposure | FLOAT | 0.0 | Exposure adjustment (stops) |
| contrast | FLOAT | 1.0 | Contrast multiplier |
| brightness | FLOAT | 0.0 | Brightness offset |
| saturation | FLOAT | 1.0 | Saturation multiplier |
| lift_r/g/b | FLOAT | 0.0 | Shadow color lift |
| gain_r/g/b | FLOAT | 1.0 | Highlight color gain |

**Output:** IMAGE

---

### 🌅 HDR Expand Dynamic Range
**Node ID:** `HDRExpandDynamicRange`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

Expand image dynamic range for HDR output. Simulates extended stops of dynamic range from SDR source.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | - | Input image |
| source_gamma | FLOAT | 2.2 | Source gamma |
| highlight_recovery | FLOAT | 1.0 | Highlight expansion |
| black_point | FLOAT | 0.0 | Black level |
| target_stops | FLOAT | 14.0 | Target dynamic range |
| highlight_rolloff | FLOAT | 1.5 | Shoulder softness |

**Output:** IMAGE

---

### 🌅 HDR Tone Map
**Node ID:** `HDRToneMap`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

GPU-accelerated tone mapping with 12+ operators for HDR to SDR conversion.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | - | HDR input |
| operator | LIST | filmic_aces | Tone mapping algorithm |
| exposure | FLOAT | 0.0 | Pre-tonemap exposure |
| gamma | FLOAT | 2.2 | Output gamma |
| white_point | FLOAT | 1.0 | White point reference |
| saturation | FLOAT | 1.0 | Output saturation |
| use_gpu | BOOLEAN | True | Enable GPU acceleration |

**Operators:** Reinhard, Reinhard Extended, Reinhard Luminance, ACES Filmic, ACES Fitted, Hable (Uncharted 2), AgX, AgX Punchy, Gran Turismo, Khronos PBR, Drago, Exponential

**Output:** IMAGE

---

### ⚡ GPU HDR Tone Map
**Node ID:** `GPUHDRToneMap`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

High-performance GPU-only tone mapping for maximum speed.

---

### 🔄 Color Space Convert
**Node ID:** `ColorSpaceConvert`  
**Category:** `FXTD STUDIO/Radiance/HDR/Color Management`

Convert between color spaces: sRGB, ACEScg, Rec.709, Rec.2020, DCI-P3.

---

### 📈 Log Curve Encode
**Node ID:** `LogCurveEncode`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

Encode linear image to log curve. Supports: ARRI LogC3, ARRI LogC4, Sony S-Log3, Panasonic V-Log, Canon Log 3, ACEScct, DaVinci Intermediate.

---

### 📉 Log Curve Decode
**Node ID:** `LogCurveDecode`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

Decode log-encoded image back to linear.

---

### 📊 HDR Histogram
**Node ID:** `HDRHistogram`  
**Category:** `FXTD STUDIO/Radiance/HDR/Analysis`

Generate extended-range histogram for HDR images.

---

### 🔀 HDR Exposure Blend
**Node ID:** `HDRExposureBlend`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

Blend multiple exposures to create HDR composite.

---

### 🌓 HDR Shadow/Highlight Recovery
**Node ID:** `HDRShadowHighlightRecovery`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

Recover detail in shadows and highlights without affecting midtones.

---

### 🌈 OCIO Color Transform
**Node ID:** `OCIOColorTransform`  
**Category:** `FXTD STUDIO/Radiance/HDR/Color Management`

Apply OpenColorIO transforms using config files. Requires OCIO installation.

---

### 📋 OCIO List Colorspaces
**Node ID:** `OCIOListColorspaces`  
**Category:** `FXTD STUDIO/Radiance/HDR/Color Management`

List available colorspaces from OCIO config.

---

### 🎬 LUT Apply
**Node ID:** `LUTApply`  
**Category:** `FXTD STUDIO/Radiance/HDR/Color Management`

Apply 3D LUT files (.cube, .3dl, .spi3d) with high-quality interpolation.

---

### ⚡ GPU Color Matrix
**Node ID:** `GPUColorMatrix`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

GPU-accelerated 3x3 color matrix operations.

---

### ⚡ GPU Tensor Ops
**Node ID:** `GPUTensorOps`  
**Category:** `FXTD STUDIO/Radiance/HDR/Processing`

Low-level GPU tensor operations for advanced workflows.

---

### 🌐 HDR 360 Generate
**Node ID:** `HDR360Generate`  
**Category:** `FXTD STUDIO/Radiance/HDR/360`

Generate HDR panoramas from source images. Supports equirectangular, cube map, mirror ball, angular map projections.

---

### 💾 Save HDRI
**Node ID:** `SaveHDRI`  
**Category:** `FXTD STUDIO/Radiance/HDR/360`

Save HDR panoramas as HDRI environment maps (EXR/HDR/TIFF).

---

### 💾 Save EXR (32-bit)
**Node ID:** `SaveImageEXR`  
**Category:** `FXTD STUDIO/Radiance/HDR/Export`

Save images as 32-bit EXR with metadata.

---

### 📂 Load EXR
**Node ID:** `LoadImageEXR`  
**Category:** `FXTD STUDIO/Radiance/HDR/Import`

Load EXR files with full HDR range preservation.

---

### 💾 Save 16-bit PNG/TIFF
**Node ID:** `SaveImage16bit`  
**Category:** `FXTD STUDIO/Radiance/HDR/Export`

Save in 16-bit PNG or TIFF format.

---

### 🎬 ACES 2.0 Output Transform
**Node ID:** `ACES2OutputTransform`  
**Category:** `FXTD STUDIO/Radiance/HDR/ACES`

Apply ACES 2.0 output transform for SDR, HDR (PQ/HLG), or cinema output (DCI-P3).

| Outputs | Description |
|---------|-------------|
| SDR | sRGB/Rec.709, P3-D65 |
| HDR | Rec.2100 PQ (1000/2000/4000 nits), HLG |
| Cinema | DCI-P3 D60, DCI-P3 D65 |

---

### 🎨 DaVinci Wide Gamut
**Node ID:** `DaVinciWideGamut`  
**Category:** `FXTD STUDIO/Radiance/HDR/Color Management`

Convert to/from DaVinci Wide Gamut and DaVinci Intermediate.

---

### 📷 ARRI Wide Gamut 4
**Node ID:** `ARRIWideGamut4`  
**Category:** `FXTD STUDIO/Radiance/HDR/Color Management`

Convert to/from ARRI Wide Gamut 4 (AWG4) for Alexa 35.

---

## Viewer Nodes

### 🎬 FXTD Master Viewer
**Node ID:** `FXTDMasterViewer`  
**Category:** `FXTD STUDIO/Radiance/Viewer`

The ultimate professional HDR image viewer combining all analysis features.

| Feature | Description |
|---------|-------------|
| Tone Mappers | 12+ operators |
| False Color | ARRI, RED, Sony, Blackmagic presets |
| Zebra | Overexposure/underexposure detection |
| Scopes | Histogram, Waveform, Vectorscope |
| Comparison | A/B split, difference, checkerboard |
| Grids | Rule of thirds, golden ratio, center cross |

**Outputs:** preview_image, scope_image, info

---

### 📊 FXTD Scope Viewer
**Node ID:** `FXTDScopeViewer`  
**Category:** `FXTD STUDIO/Radiance/Viewer`

Generate standalone professional scopes: histogram, waveform (luma/RGB), vectorscope.

---

### 🔍 FXTD Pixel Sampler
**Node ID:** `FXTDPixelSampler`  
**Category:** `FXTD STUDIO/Radiance/Viewer`

Sample and display precise pixel values from HDR images with crosshair marker.

**Outputs:** marked_image, info, red, green, blue, luminance

---

## Upscale Nodes

### ⬆️ FXTD Pro Upscale
**Node ID:** `FXTDProUpscale`  
**Category:** `FXTD STUDIO/Radiance/Upscale`

Professional 32-bit upscaler optimized for Flux workflows.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| scale_factor | FLOAT | 2.0 | Upscale multiplier |
| method | LIST | lanczos | Algorithm |
| preset | LIST | Flux Balanced | Optimization preset |
| sharpening | FLOAT | 0.3 | Post-upscale sharpening |

**Methods:** Nearest, Bilinear, Bicubic, Lanczos, Lanczos4, Mitchell, Catmull-Rom, Hermite, Gaussian

---

### 📐 FXTD Upscale By Size
**Node ID:** `FXTDUpscaleBySize`  
**Category:** `FXTD STUDIO/Radiance/Upscale`

Upscale to exact pixel dimensions with aspect ratio preservation options.

---

### 🔲 FXTD Upscale Tiled
**Node ID:** `FXTDUpscaleTiled`  
**Category:** `FXTD STUDIO/Radiance/Upscale`

Tile-based upscaler for very large images with seamless blending.

---

### 🔪 FXTD Sharpen 32-bit
**Node ID:** `FXTDSharpen32bit`  
**Category:** `FXTD STUDIO/Radiance/Upscale`

GPU-accelerated 32-bit sharpening with Unsharp Mask, High Pass, and Multi-Scale methods.

---

### ⬇️ FXTD Downscale 32-bit
**Node ID:** `FXTDDownscale32bit`  
**Category:** `FXTD STUDIO/Radiance/Upscale`

High-quality 32-bit downscaling with anti-aliasing.

---

### 🎚️ FXTD Bit Depth Convert
**Node ID:** `FXTDBitDepthConvert`  
**Category:** `FXTD STUDIO/Radiance/Upscale`

Convert between bit depths (8/10/16/32-bit) with optional dithering: Floyd-Steinberg, Ordered, Blue Noise.

---

### 🤖 FXTD AI Upscale
**Node ID:** `FXTDAIUpscale`  
**Category:** `FXTD STUDIO/Radiance/Upscale`

AI-powered upscaling using neural network models.

| Supported Models |
|------------------|
| RealESRGAN_x4plus |
| RealESRGAN_x4plus_anime_6B |
| RealESRGAN_x2plus |
| ESRGAN_4x |
| 4x-UltraSharp |
| 4x-AnimeSharp |
| SwinIR_4x |
| HAT_4x |
| SUPIR-v0F_fp16 |
| SUPIR-v0Q_fp16 |

---

## Film & Lens Effect Nodes

### 🎬 FXTD Pro Film Effects (NEW - Industry Level)
**Node ID:** `FXTDProFilmEffects`  
**Category:** `FXTD STUDIO/Radiance/Film/Effects`

The ultimate industry-level film effects node combining all film and lens effects in one GPU-accelerated package.

| Master Presets | Description |
|----------------|-------------|
| 35mm Clean | Modern 35mm cinema - clean with subtle character |
| 16mm Gritty | 16mm indie film - visible grain, character |
| Alexa Natural | ARRI Alexa digital cinema - minimal processing |
| RED Raw | RED cinema camera - sharp, minimal grain |
| 70mm IMAX | Large format IMAX - ultra clean, vast |
| Super 8 Vintage | Super 8 home movie - heavy grain, instability |
| VHS Degraded | VHS tape artifact - heavy degradation |
| Digital Clean | Modern digital - no film artifacts |
| Cinematic Blockbuster | Hollywood blockbuster - polished with character |
| 70s Grindhouse | 1970s exploitation film - damaged, saturated |
| Music Video | Modern music video - stylized, high contrast |

| Effect Section | Parameters |
|----------------|------------|
| **Film Grain** | intensity, size, softness, shadow_boost, highlight_protect |
| **Halation** | intensity, threshold, size, RGB color picker |
| **Chromatic Aberration** | intensity (radial dispersion) |
| **Bloom** | intensity, threshold, size |
| **Vignette** | intensity, falloff, roundness |
| **Lens Distortion** | k1, k2 (barrel/pincushion) |
| **Diffusion** | Pro-Mist style highlight glow |
| **Gate Weave** | Frame instability amplitude |

**GPU:** ✅ Full acceleration with CPU fallback  
**Outputs:** processed_image, effect_info

---

### 🎞️ FXTD Film Grain
**Node ID:** `FXTDFilmGrain`  
**Category:** `FXTD STUDIO/Radiance/Film/Effects`

GPU-accelerated film grain with camera and film stock presets.

| Camera Presets | Film Stock Presets |
|----------------|-------------------|
| ARRI Alexa 35 | Kodak Vision3 500T 5219 |
| ARRI Alexa Mini LF | Kodak Vision3 250D 5207 |
| RED V-Raptor XL 8K | Kodak Vision3 50D 5203 |
| RED Komodo 6K | Kodak Vision3 200T 5213 |
| Sony Venice 2 | Kodak 5248 (70s Look) |
| Sony A7S III | Fuji Eterna 500T 8573 |
| Blackmagic URSA Mini Pro 12K | Fuji Eterna 250D 8563 |
| Blackmagic Pocket 4K | CineStill 800T |
| Canon C70 | CineStill 50D |
| Canon R5 C | |
| Panavision DXL2 | |
| IMAX Digital | |

---

### 📷 FXTD Lens Effects
**Node ID:** `FXTDLensEffects`  
**Category:** `FXTD STUDIO/Radiance/Film/Effects`

Professional lens effects with cinema lens presets.

| Feature | Description |
|---------|-------------|
| Chromatic Aberration | RGB fringing simulation |
| Vignette | Optical falloff with adjustable softness |
| Bloom | Highlight glow with threshold |
| Lens Flare | Anamorphic and spherical |

| Lens Presets |
|--------------|
| Cooke S7/i |
| Zeiss Master Prime |
| Arri Signature Prime |
| Panavision Primo 70 |
| Leica Summilux-C |
| Angenieux Optimo |

---

### 🎬 FXTD Film Look
**Node ID:** `FXTDFilmLook`  
**Category:** `FXTD STUDIO/Radiance/Film/Effects`

Complete film emulation: camera + film stock + lens effects combined.

**Outputs:** processed_image, look_info

---

### 🎚️ FXTD Grain Advanced
**Node ID:** `FXTDFilmGrainAdvanced`  
**Category:** `FXTD STUDIO/Radiance/Film/Effects`

Full manual control over all grain parameters including per-channel grain size, luminance response curves, and custom halation color.

---

## Camera Simulation Nodes

### 🎨 FXTD White Balance
**Node ID:** `FXTDWhiteBalance`  
**Category:** `FXTD STUDIO/Radiance/Camera/Color`

Professional white balance adjustment using color temperature (Kelvin) and tint.

| Presets | Temperature |
|---------|-------------|
| Daylight | 5500K |
| Cloudy | 6500K |
| Shade | 7500K |
| Tungsten | 3200K |
| Fluorescent | 4000K |
| Candlelight | 1850K |
| Blue Hour | 9000K |

| Input | Type | Description |
|-------|------|-------------|
| temperature | INT | 1000-15000K |
| tint | FLOAT | Green-magenta shift |
| source_temperature | INT | Original temp for correction |

**GPU:** ✅

---

### 🔍 FXTD Depth of Field
**Node ID:** `FXTDDepthOfField`  
**Category:** `FXTD STUDIO/Radiance/Camera/Lens`

Cinematic depth of field blur with optional depth map input.

| Input | Type | Description |
|-------|------|-------------|
| depth_map | IMAGE | Optional depth input |
| focus_distance | FLOAT | 0-1 focus point |
| focus_range | FLOAT | In-focus zone |
| blur_amount | FLOAT | Max blur strength |
| bokeh_shape | LIST | Circle, Hexagon, Octagon, Anamorphic |
| highlight_boost | FLOAT | Bokeh brightness |

**GPU:** ✅

---

### 💨 FXTD Motion Blur
**Node ID:** `FXTDMotionBlur`  
**Category:** `FXTD STUDIO/Radiance/Camera/Motion`

Directional, radial, or zoom motion blur.

| Blur Types | Description |
|------------|-------------|
| Directional | Linear blur with angle control |
| Radial | Rotational blur around center |
| Zoom | In/out blur from center point |

| Input | Type | Description |
|-------|------|-------------|
| amount | FLOAT | Blur strength |
| angle | FLOAT | Direction (directional mode) |
| center_x/y | FLOAT | Blur center point |
| samples | INT | Quality (4-64) |

**GPU:** ✅

---

### 📷 FXTD Rolling Shutter
**Node ID:** `FXTDRollingShutter`  
**Category:** `FXTD STUDIO/Radiance/Camera/Sensor`

Simulate CMOS rolling shutter artifacts.

| Feature | Description |
|---------|-------------|
| Skew | Diagonal distortion |
| Wobble | Jello effect |
| Flash Banding | Partial exposure bands |

| Shutter Direction |
|-------------------|
| Vertical (default) |
| Horizontal |
| Both |

**GPU:** ✅

---

### 📦 FXTD Compression Artifacts
**Node ID:** `FXTDCompressionArtifacts`  
**Category:** `FXTD STUDIO/Radiance/Camera/Pipeline`

Add compression artifacts for degraded video look.

| Artifact Type | Description |
|---------------|-------------|
| JPEG | DCT blocking, mosquito noise |
| Banding | Color posterization |
| Both | Combined artifacts |

| Input | Type | Description |
|-------|------|-------------|
| quality | INT | 1-100 (lower = more artifacts) |
| block_size | INT | DCT block size |
| banding_levels | INT | Color quantization steps |

---

### 📳 FXTD Camera Shake
**Node ID:** `FXTDCameraShake`  
**Category:** `FXTD STUDIO/Radiance/Camera/Motion`

Handheld camera shake with Perlin-like motion.

| Presets | Shake X | Shake Y | Rotation |
|---------|---------|---------|----------|
| Subtle Handheld | 2.0 | 0.5 | 0.3 |
| Documentary | 4.0 | 1.0 | 0.5 |
| Action Cam | 8.0 | 2.0 | 1.0 |
| Earthquake | 20.0 | 5.0 | 3.0 |
| Vehicle Interior | 6.0 | 3.0 | 0.2 |
| Nervous Hold | 3.0 | 2.0 | 0.8 |

**GPU:** ✅

---

## EXR Export Nodes

### 💾 FXTD Save EXR
**Node ID:** `FXTDSaveEXR`  
**Category:** `FXTD STUDIO/Radiance/Export/EXR`

Save images as EXR files with full HDR and metadata support.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| bit_depth | LIST | 16-bit Half | HALF or FLOAT |
| compression | LIST | ZIP | None, RLE, ZIPS, ZIP, PIZ, PXR24, B44, B44A, DWAA, DWAB |
| alpha_mode | LIST | None | None, From Image, Solid White, Solid Black |

---

### 📚 FXTD Save EXR Multi-Layer
**Node ID:** `FXTDSaveEXRMultiLayer`  
**Category:** `FXTD STUDIO/Radiance/Export/EXR`

Save multiple layers into a single multi-layer EXR file.

| Layers | Description |
|--------|-------------|
| beauty | Main render |
| diffuse, specular, emission | Light passes |
| normal, depth | Geometry data |
| alpha, ao, shadow | Utility passes |
| uv, motion, crypto | Technical passes |
| custom1/2/3 | User-defined |

---

### 🎬 FXTD Save EXR Sequence
**Node ID:** `FXTDSaveEXRSequence`  
**Category:** `FXTD STUDIO/Radiance/Export/EXR`

Save image sequence as EXR files with VFX-standard naming (1001+).

**Outputs:** output_path, first_frame, last_frame

---

### 🔀 FXTD EXR Channel Merge
**Node ID:** `FXTDEXRChannelMerge`  
**Category:** `FXTD STUDIO/Radiance/Export/EXR`

Merge separate images into EXR channels.

---

### 🎭 FXTD Save EXR Cryptomatte
**Node ID:** `FXTDSaveEXRCryptomatte`  
**Category:** `FXTD STUDIO/Radiance/Export/EXR`

Save Cryptomatte-compatible EXR for Nuke/Fusion/After Effects.

---

## Prompt Engineering Nodes

### 🎬 FXTD Cinematic Prompt Machine
**Node ID:** `FXTDCinematicPromptMachine`  
**Category:** `FXTD STUDIO/Radiance/Utilities`

The ultimate director's console for prompt generation using filmmaking terminology. Outputs text strings for manual CLIP encoding.

| Feature | Description |
|---------|-------------|
| **Style Presets** | 14 one-click presets for instant cinematic looks |
| **Full Manual Control** | Override any setting for complete customization |
| **Smart Auto-Negative** | Context-aware negative prompt generation |

| Style Presets |
|---------------|
| 🎬 Classic Hollywood |
| 🌃 Film Noir |
| 🚀 Sci-Fi Cinematic |
| 🌆 Cyberpunk |
| 🎭 Drama / Emotional |
| 🏔️ Epic Landscape |
| 👤 Portrait Pro |
| 📰 Documentary |
| 🎨 Artistic / Painterly |
| 📼 Retro VHS |
| 🌅 Golden Hour Magic |
| 🌙 Moody Night |
| ⚡ Action / Dynamic |
| 🎪 Wes Anderson |

| Input | Options |
|-------|---------|
| base_prompt | Your core subject/scene description |
| style_preset | One-click preset or "None (Custom)" for manual |
| framing | ECU, CU, MCU, MS, Cowboy, Wide, EWS, Establishing, OTS, POV, Low/High Angle, Dutch, etc. |
| camera_type | ARRI Alexa, RED, Sony Venice, Blackmagic, Canon, Panavision, IMAX, Super 8, etc. |
| lens_focal | 14mm to 600mm, Anamorphic, Tilt-Shift, Cinema Primes (Cooke, Zeiss, ARRI, Panavision) |
| aperture_dof | f/0.95 to f/22 with descriptive labels |
| lighting | Rembrandt, Chiaroscuro, Film Noir, Golden Hour, Volumetric Fog, Cyberpunk Neon, etc. |
| style_aesthetic | 23 styles from Photorealistic to Anime to Blade Runner |
| film_stock | Kodak Vision3, Portra, Cinestill 800T, etc. |
| shutter_speed | Motion blur control |
| color_grading | Teal & Orange, Bleach Bypass, Cyberpunk Neon, etc. |
| aspect_ratio | 16:9, 2.39:1 Anamorphic, 4:3, 1:1, 21:9 |
| year_era | 1800-2100 for period looks |
| auto_negative | Smart negative prompt generation |

**Outputs:** final_prompt, negative_prompt

---

### 🎬 FXTD Cinematic Encoder (NEW - All-in-One)
**Node ID:** `FXTDCinematicPromptEncoder`  
**Category:** `FXTD STUDIO/Radiance/Utilities`

**All-in-one** cinematic prompt builder with **direct CLIP encoding**. Eliminates the need for separate CLIP Text Encode nodes.

| Feature | Description |
|---------|-------------|
| **Direct CONDITIONING Output** | Ready for sampler input |
| **Style Presets** | 14 one-click presets |
| **CLIP Skip Support** | Control encoding depth (0-24 layers) |
| **Smart Auto-Negative** | Context-aware negative generation |
| **Text Output** | Also outputs prompt text for debugging |

| Workflow Simplification |
|-------------------------|
| **Before:** Prompt Machine → CLIP Encode (×2) → Sampler |
| **After:** Cinematic Encoder → Sampler |

| Input | Type | Description |
|-------|------|-------------|
| clip | CLIP | CLIP model for encoding |
| base_prompt | STRING | Your subject/scene |
| style_preset | LIST | One-click style preset |
| framing | LIST | Camera framing |
| camera_type | LIST | Camera body |
| lens_focal | LIST | Lens choice |
| aperture_dof | LIST | Depth of field |
| lighting | LIST | Lighting style |
| style_aesthetic | LIST | Visual aesthetic |
| clip_skip | INT | Layers to skip (0-24, default: 0) |
| + all optional inputs from Prompt Machine |

| Output | Type | Description |
|--------|------|-------------|
| positive | CONDITIONING | Encoded positive prompt |
| negative | CONDITIONING | Encoded negative prompt |
| final_prompt | STRING | Generated prompt text |
| negative_prompt | STRING | Generated negative text |

**Quality Score:** 10/10 ⭐⭐⭐⭐⭐

---



## Quick Reference Table

| Node | Category | GPU | Description |
|------|----------|-----|-------------|
| ImageToFloat32 | HDR/Processing | ✅ | Convert to float32 |
| Float32ColorCorrect | HDR/Processing | ✅ | Color correction |
| HDRExpandDynamicRange | HDR/Processing | ✅ | Expand DR |
| HDRToneMap | HDR/Processing | ✅ | Tone mapping |
| GPUHDRToneMap | HDR/Processing | ✅ | Fast tone map |
| ColorSpaceConvert | Color Management | ✅ | Color space |
| LogCurveEncode | HDR/Processing | ✅ | Log encoding |
| LogCurveDecode | HDR/Processing | ✅ | Log decoding |
| HDRHistogram | HDR/Analysis | ❌ | Histogram |
| HDRExposureBlend | HDR/Processing | ❌ | Exposure blend |
| HDRShadowHighlightRecovery | HDR/Processing | ❌ | Recovery |
| OCIOColorTransform | Color Management | ❌ | OCIO |
| OCIOListColorspaces | Color Management | ❌ | OCIO list |
| LUTApply | Color Management | ❌ | 3D LUT |
| GPUColorMatrix | HDR/Processing | ✅ | Color matrix |
| GPUTensorOps | HDR/Processing | ✅ | Tensor ops |
| HDR360Generate | HDR/360 | ❌ | 360 pano |
| SaveHDRI | HDR/360 | ❌ | Save HDRI |
| SaveImageEXR | HDR/Export | ❌ | Save EXR |
| LoadImageEXR | HDR/Import | ❌ | Load EXR |
| SaveImage16bit | HDR/Export | ❌ | Save 16-bit |
| ACES2OutputTransform | HDR/ACES | ❌ | ACES 2.0 |
| DaVinciWideGamut | Color Management | ❌ | DaVinci WG |
| ARRIWideGamut4 | Color Management | ❌ | ARRI AWG4 |
| FXTDMasterViewer | Viewer | ✅ | Master viewer |
| FXTDScopeViewer | Viewer | ❌ | Scopes |
| FXTDPixelSampler | Viewer | ❌ | Pixel sample |
| FXTDProUpscale | Upscale | ✅ | Pro upscale |
| FXTDUpscaleBySize | Upscale | ❌ | Size upscale |
| FXTDUpscaleTiled | Upscale | ❌ | Tiled upscale |
| FXTDSharpen32bit | Upscale | ✅ | Sharpening |
| FXTDDownscale32bit | Upscale | ✅ | Downscale |
| FXTDBitDepthConvert | Upscale | ❌ | Bit depth |
| FXTDAIUpscale | Upscale | ✅ | AI upscale |
| **FXTDProFilmEffects** | **Film/Effects** | **✅** | **Industry-level combined** |
| FXTDFilmGrain | Film/Effects | ✅ | Film grain |
| FXTDLensEffects | Film/Effects | ❌ | Lens effects |
| FXTDFilmLook | Film/Effects | ❌ | Film look |
| FXTDFilmGrainAdvanced | Film/Effects | ❌ | Advanced grain |
| FXTDSaveEXR | Export/EXR | ❌ | Save EXR |
| FXTDSaveEXRMultiLayer | Export/EXR | ❌ | Multi-layer |
| FXTDSaveEXRSequence | Export/EXR | ❌ | Sequence |
| FXTDEXRChannelMerge | Export/EXR | ❌ | Merge |
| FXTDSaveEXRCryptomatte | Export/EXR | ❌ | Cryptomatte |
| FXTDCinematicPromptMachine | Utilities | ❌ | Prompt gen + presets |
| **FXTDCinematicPromptEncoder** | **Utilities** | **❌** | **All-in-one CLIP encoder** |

---

*FXTD Studio Radiance v3.0.0 - FXTD Studios © 2024-2026*
