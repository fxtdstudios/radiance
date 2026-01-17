# FXTD Studio Radiance - Example Workflows

Pre-built workflow files to help you get started quickly.

## Available Workflows

### 🚀 FXTD_QuickStart_Basic.json
**Beginner-friendly basic Flux workflow**

A minimal setup demonstrating:
- Model loading (Flux)
- FXTD Radiance Sampler
- Pro Viewer preview
- Image saving

Perfect for first-time users!

---

### 🎬 FXTD_Cinema_LogC3_Export.json
**Professional cinema pipeline**

Complete workflow for:
- HDR expansion
- LogC3 encoding
- EXR export

Output is ready for DaVinci Resolve color grading.

---

### 🎞️ FXTD_FilmLook_Pipeline.json
**Film emulation with effects**

Features:
- 2x upscaling
- CineStill 800T film stock
- Lens effects (halation, vignette)
- Pro Viewer analysis

Creates authentic cinematic output.

---

## How to Use

1. Open ComfyUI
2. Click **Load** (or drag-drop the JSON)
3. Select any workflow from this folder
4. Adjust settings as needed
5. Run!

## Requirements

Make sure you have the required models:
- `flux1-dev.safetensors` or `flux1-schnell.safetensors`
- `clip_l.safetensors` + `t5xxl_fp16.safetensors`
- `ae.safetensors` (Flux VAE)

---

*FXTD Studios © 2024-2026*
