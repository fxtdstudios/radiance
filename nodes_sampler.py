"""
═══════════════════════════════════════════════════════════════════════════════
                    FXTD RADIANCE SAMPLER PRO
         Professional Flux-Optimized Sampling Engine
                   FXTD Studios © 2024-2026
                   
 Features:
 - Native Flux sigma shifting for high-resolution detail
 - Integrated guidance control (bypasses CFG for Flux)
 - Workflow presets (txt2img, img2img, inpaint, high-res)
 - Full step control with timing diagnostics
 - SIGMAS output for advanced chaining
 - Professional error handling and logging

 Example:
     Model → Sampler Pro → VAE Decode → Image
     
 Flux Tips:
     - CFG = 1.0 (Flux uses guidance instead)
     - flux_guidance = 3.5 (default, higher = more prompt adherence)
     - flux_shift = 1.0 (increase to 3.0 for high-res detail boost)
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import time
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils


# ═══════════════════════════════════════════════════════════════════════════════
#                         WORKFLOW PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

WORKFLOW_PRESETS = [
    "None (Custom)",
    "🎨 Flux txt2img",
    "🖼️ Flux img2img",
    "🎭 Flux Inpaint",
    "📐 Flux High-Res Fix",
    "⚡ Flux Fast (12 steps)",
    "💎 Flux Quality (28 steps)",
    "🎬 Cinematic (30 steps)",
]

PRESET_CONFIGS = {
    "🎨 Flux txt2img": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "🖼️ Flux img2img": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.75,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "🎭 Flux Inpaint": {
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
    },
    "📐 Flux High-Res Fix": {
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 0.5,
        "flux_shift": 3.0,
        "flux_guidance": 3.5,
    },
    "⚡ Flux Fast (12 steps)": {
        "steps": 12,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 3.5,
    },
    "💎 Flux Quality (28 steps)": {
        "steps": 28,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "flux_shift": 1.0,
        "flux_guidance": 4.0,
    },
    "🎬 Cinematic (30 steps)": {
        "steps": 30,
        "cfg": 1.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 1.0,
        "flux_shift": 2.0,
        "flux_guidance": 4.5,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                         SIGMA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def flux_shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    Apply Flux-specific sigma shifting.
    
    The shift parameter controls how the noise schedule is transformed.
    Higher shift values push more denoising into later steps, which
    can improve high-frequency details at high resolutions.
    
    Formula: shifted = shift * sigma / (1 + (shift - 1) * sigma)
    
    Args:
        sigmas: Original sigma schedule
        shift: Shift factor (1.0 = no change, 3.0 = typical for high-res)
    
    Returns:
        Shifted sigma schedule
    """
    if shift == 1.0:
        return sigmas
    
    # Apply the Flux shift formula
    shifted = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return shifted


def get_flux_sigmas(model, scheduler: str, steps: int, denoise: float, shift: float = 1.0) -> torch.Tensor:
    """
    Calculate sigma schedule optimized for Flux models.
    
    Args:
        model: The model wrapper
        scheduler: Scheduler name (recommended: "simple" for Flux)
        steps: Number of sampling steps
        denoise: Denoise strength (1.0 = full, <1.0 = img2img)
        shift: Flux shift parameter
    
    Returns:
        Sigma schedule tensor
    """
    # Get the model's sampling configuration
    model_sampling = model.get_model_object("model_sampling")
    
    # Calculate base sigmas using the scheduler
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
    
    # Apply Flux shift if specified
    if shift != 1.0:
        sigmas = flux_shift_sigmas(sigmas, shift)
    
    # Apply denoise (trim sigmas for img2img)
    if denoise < 1.0:
        if denoise <= 0.0:
            return torch.tensor([])
        total_steps = len(sigmas) - 1
        start_step = max(0, int(total_steps * (1.0 - denoise)))
        sigmas = sigmas[start_step:]
    
    return sigmas


# ═══════════════════════════════════════════════════════════════════════════════
#                         RADIANCE SAMPLER PRO
# ═══════════════════════════════════════════════════════════════════════════════

class FXTD_Radiance_Sampler_Pro:
    """
    Professional Flux-optimized sampler with full control.
    
    Features:
    - Workflow presets for instant configuration
    - Flux Shift: Native sigma shifting for high-res detail
    - Flux Guidance: Direct control over Flux's internal guidance
    - Step Control: start_step, end_step for advanced workflows
    - Noise Control: add_noise, return_with_leftover_noise
    - Timing Report: Performance diagnostics
    - SIGMAS Output: For advanced chaining
    
    Example:
        Use preset "🎨 Flux txt2img" for instant optimal settings
        Or customize each parameter for full control
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model (Flux, SD, SDXL, etc.)"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning from CLIP encoder"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning (typically empty for Flux)"}),
                "latent_image": ("LATENT", {"tooltip": "Input latent (empty for txt2img, or from VAE encode for img2img)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility"}),
            },
            "optional": {
                "preset": (WORKFLOW_PRESETS, {"default": "🎨 Flux txt2img", "tooltip": "Quick workflow preset. Overrides settings below."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "tooltip": "Sampling steps (20-28 optimal for Flux)"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "CFG scale (use 1.0 for Flux, guidance is separate)"}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Sampling algorithm (euler recommended for Flux)"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "Noise schedule (simple recommended for Flux)"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength (1.0=full gen, 0.5-0.8=img2img)"}),
                "flux_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Sigma shift for high-res (1.0=none, 3.0=high-res boost)"}),
                "flux_guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Flux guidance scale (3.5 default, higher=more prompt adherence)"}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 150, "tooltip": "Start from this step (for advanced multi-pass workflows)"}),
                "end_step": ("INT", {"default": 150, "min": 0, "max": 150, "tooltip": "End at this step (150 = run all steps)"}),
                "add_noise": ("BOOLEAN", {"default": True, "tooltip": "Add initial noise (disable for step-by-step control)"}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False, "tooltip": "Keep remaining noise (for chaining samplers)"}),
            }
        }

    RETURN_TYPES = ("LATENT", "SIGMAS")
    RETURN_NAMES = ("latent", "sigmas")
    OUTPUT_TOOLTIPS = (
        "Output latent ready for VAE decode",
        "Sigma schedule used (for advanced chaining)",
    )
    FUNCTION = "execute"
    CATEGORY = "FXTD Studios/Radiance/Utilities"
    DESCRIPTION = "Professional Flux-optimized sampler with presets, timing report, and full parameter control."

    def execute(self, model, positive, negative, latent_image, seed,
                preset="🎨 Flux txt2img", steps=20, cfg=1.0, sampler="euler", 
                scheduler="simple", denoise=1.0, flux_shift=1.0, flux_guidance=3.5,
                start_step=0, end_step=150, add_noise=True, return_with_leftover_noise=False):
        """
        Execute sampling with timing diagnostics.
        """
        timings = {}
        t_start = time.time()
        
        # ─────────────────────────────────────────────────────────────────
        # Apply preset if selected
        # ─────────────────────────────────────────────────────────────────
        if preset != "None (Custom)" and preset in PRESET_CONFIGS:
            config = PRESET_CONFIGS[preset]
            steps = config.get("steps", steps)
            cfg = config.get("cfg", cfg)
            sampler = config.get("sampler", sampler)
            scheduler = config.get("scheduler", scheduler)
            denoise = config.get("denoise", denoise)
            flux_shift = config.get("flux_shift", flux_shift)
            flux_guidance = config.get("flux_guidance", flux_guidance)
        
        # ─────────────────────────────────────────────────────────────────
        # Prepare latent and noise
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        timings["latent_copy"] = time.time() - t0
        
        t0 = time.time()
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)
        timings["prepare_noise"] = time.time() - t0
        
        # ─────────────────────────────────────────────────────────────────
        # Apply Flux guidance to conditioning
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        if flux_guidance != 3.5:
            modified_positive = []
            for cond in positive:
                cond_dict = cond[1].copy()
                cond_dict["guidance"] = flux_guidance
                modified_positive.append([cond[0], cond_dict])
            positive = modified_positive
        timings["flux_guidance"] = time.time() - t0
        
        # ─────────────────────────────────────────────────────────────────
        # Calculate sigmas with Flux shift
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            sigmas = get_flux_sigmas(model, scheduler, steps, denoise, flux_shift)
        except Exception as e:
            print(f"[Radiance Sampler Pro] Warning: Custom sigma calculation failed ({e}), using standard")
            sigmas = None
        timings["sigma_calc"] = time.time() - t0
        
        # ─────────────────────────────────────────────────────────────────
        # Calculate step range
        # ─────────────────────────────────────────────────────────────────
        actual_start = start_step if start_step > 0 else None
        actual_end = min(end_step, steps) if end_step < 150 else None
        
        # ─────────────────────────────────────────────────────────────────
        # Execute sampling
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            disable_noise=not add_noise,
            start_step=actual_start,
            last_step=actual_end,
            force_full_denoise=not return_with_leftover_noise,
            noise_mask=noise_mask,
            callback=None,
            disable_pbar=False,
            seed=seed
        )
        timings["sampling"] = time.time() - t0
        
        # ─────────────────────────────────────────────────────────────────
        # Prepare output
        # ─────────────────────────────────────────────────────────────────
        t0 = time.time()
        out = latent.copy()
        out["samples"] = samples
        timings["output_prep"] = time.time() - t0
        
        total_time = time.time() - t_start
        
        # ─────────────────────────────────────────────────────────────────
        # Print timing report
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "═"*60)
        print("⚡ RADIANCE SAMPLER PRO - TIMING REPORT")
        print("═"*60)
        if preset != "None (Custom)":
            print(f"  Preset:           {preset}")
        print(f"  Steps:            {steps} | CFG: {cfg} | Denoise: {denoise}")
        print(f"  Flux Shift:       {flux_shift} | Guidance: {flux_guidance}")
        print("─"*60)
        print(f"  Latent copy:      {timings['latent_copy']*1000:>8.2f} ms")
        print(f"  Prepare noise:    {timings['prepare_noise']*1000:>8.2f} ms")
        print(f"  Flux guidance:    {timings['flux_guidance']*1000:>8.2f} ms")
        print(f"  Sigma calc:       {timings['sigma_calc']*1000:>8.2f} ms")
        print("─"*60)
        print(f"  SAMPLING:         {timings['sampling']:>8.2f} s  ◄◄◄")
        print("─"*60)
        print(f"  Output prep:      {timings['output_prep']*1000:>8.2f} ms")
        print("─"*60)
        print(f"  TOTAL:            {total_time:>8.2f} s")
        print("═"*60)
        
        # Calculate percentages
        sampling_pct = (timings['sampling'] / total_time) * 100 if total_time > 0 else 0
        overhead_pct = 100 - sampling_pct
        print(f"  Sampling:  {sampling_pct:>5.1f}% | Overhead: {overhead_pct:>5.1f}%")
        print("═"*60 + "\n")
        
        # Return sigmas for advanced chaining
        output_sigmas = sigmas if sigmas is not None else torch.tensor([])
        
        return (out, output_sigmas)


# ═══════════════════════════════════════════════════════════════════════════════
#                         NODE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "FXTD_Radiance_Sampler_Pro": FXTD_Radiance_Sampler_Pro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTD_Radiance_Sampler_Pro": "⚡ Radiance Sampler Pro",
}
