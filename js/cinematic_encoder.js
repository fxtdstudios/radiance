/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                    FXTD CINEMATIC ENCODER WIDGET
 *              Prompt Preview + Randomize Button for 10/10 UX
 *                        FXTD Studios © 2024-2026
 * ═══════════════════════════════════════════════════════════════════════════════
 */

import { app } from "../../scripts/app.js";

// Style presets data (must match Python)
const STYLE_PRESETS = [
    "None (Custom)",
    "🎬 Classic Hollywood",
    "🌃 Film Noir",
    "🚀 Sci-Fi Cinematic",
    "🌆 Cyberpunk",
    "🎭 Drama / Emotional",
    "🏔️ Epic Landscape",
    "👤 Portrait Pro",
    "📰 Documentary",
    "🎨 Artistic / Painterly",
    "📼 Retro VHS",
    "🌅 Golden Hour Magic",
    "🌙 Moody Night",
    "⚡ Action / Dynamic",
    "🎪 Wes Anderson",
];

const FRAMING = [
    "None", "Extreme Close-Up (ECU)", "Close-Up (CU)", "Medium Close-Up (MCU)",
    "Medium Shot (MS)", "Cowboy Shot (American Shot)", "Full Body Shot (Wide)",
    "Extreme Wide Shot (EWS)", "Establishing Shot", "Over-The-Shoulder (OTS)",
    "Point of View (POV)", "Low Angle (Hero Shot)", "High Angle (Vulnerability)",
    "Bird's Eye View (Overhead)", "Worm's Eye View", "Dutch Angle (Canted)",
    "Symmetrical Composition", "Rule of Thirds"
];

const CAMERAS = [
    "None", "ARRI Alexa 65 (IMAX)", "ARRI Alexa Mini LF", "ARRI Alexa 35",
    "Sony Venice 2", "Sony FX9", "Sony A7S III",
    "RED V-Raptor XL", "RED Komodo", "RED Monstro 8K VV",
    "Panavision Millennium DXL2", "Panavision Panaflex Gold II (35mm)",
    "IMAX 15/70mm Film Camera", "Bolex H16 (16mm Film)", "Super 8mm Camera"
];

const LIGHTING = [
    "None", "Rembrandt Lighting", "Chiaroscuro (High Contrast)", "Film Noir Lighting",
    "Split Lighting", "Butterfly Lighting", "Paramount Lighting",
    "Soft Window Light", "Golden Hour (Magic Hour)", "Blue Hour",
    "Cinematic Haze / Volumetric Fog", "God Rays (Crepuscular Rays)",
    "Neon Cyberpunk Lighting", "Practical Lighting", "Bioluminescence"
];

app.registerExtension({
    name: "FXTD.CinematicEncoder",

    async nodeCreated(node) {
        if (node.comfyClass !== "FXTDCinematicPromptEncoder") return;

        // Add custom widget for prompt preview
        const previewWidget = node.addWidget("text", "prompt_preview", "", () => { }, {
            multiline: true,
            inputEl: null,
        });
        previewWidget.computeSize = () => [node.size[0] - 20, 60];
        previewWidget.serializeValue = () => undefined; // Don't save this

        // Style the preview
        if (previewWidget.inputEl) {
            previewWidget.inputEl.readOnly = true;
            previewWidget.inputEl.style.background = "rgba(0, 168, 255, 0.1)";
            previewWidget.inputEl.style.border = "1px solid rgba(0, 168, 255, 0.3)";
            previewWidget.inputEl.style.color = "#aaccff";
            previewWidget.inputEl.style.fontSize = "10px";
            previewWidget.inputEl.style.fontFamily = "monospace";
        }

        // Randomize button
        node.addWidget("button", "🎲 Randomize Style", null, () => {
            randomizeSettings(node);
        });

        // Update preview when inputs change
        const originalOnPropertyChanged = node.onPropertyChanged;
        node.onPropertyChanged = function (property, value) {
            if (originalOnPropertyChanged) originalOnPropertyChanged.call(this, property, value);
            updatePreview(node, previewWidget);
        };

        // Also update on widget change
        const originalOnWidgetChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value, old_value, widget) {
            if (originalOnWidgetChanged) originalOnWidgetChanged.call(this, name, value, old_value, widget);
            updatePreview(node, previewWidget);
        };

        // Initial update
        setTimeout(() => updatePreview(node, previewWidget), 100);
    }
});

function randomizeSettings(node) {
    const widgets = node.widgets;
    if (!widgets) return;

    // Find and randomize each widget
    for (const widget of widgets) {
        switch (widget.name) {
            case "style_preset":
                widget.value = STYLE_PRESETS[Math.floor(Math.random() * (STYLE_PRESETS.length - 1)) + 1];
                break;
            case "framing":
                widget.value = FRAMING[Math.floor(Math.random() * (FRAMING.length - 1)) + 1];
                break;
            case "camera_type":
                widget.value = CAMERAS[Math.floor(Math.random() * (CAMERAS.length - 1)) + 1];
                break;
            case "lighting":
                widget.value = LIGHTING[Math.floor(Math.random() * (LIGHTING.length - 1)) + 1];
                break;
            case "year_era":
                widget.value = 1950 + Math.floor(Math.random() * 80); // 1950-2030
                break;
        }
    }

    // Trigger update
    node.setDirtyCanvas(true);

    // Find and update preview
    const previewWidget = widgets.find(w => w.name === "prompt_preview");
    if (previewWidget) {
        updatePreview(node, previewWidget);
    }
}

function updatePreview(node, previewWidget) {
    if (!previewWidget) return;

    const widgets = node.widgets;
    if (!widgets) return;

    // Get current values
    let basePrompt = "", framing = "", camera = "", lighting = "", style = "";

    for (const widget of widgets) {
        switch (widget.name) {
            case "base_prompt": basePrompt = widget.value || "A cinematic scene..."; break;
            case "framing": framing = widget.value; break;
            case "camera_type": camera = widget.value; break;
            case "lighting": lighting = widget.value; break;
            case "style_aesthetic": style = widget.value; break;
        }
    }

    // Build preview (simplified version)
    let parts = [];

    if (framing && framing !== "None") {
        parts.push(`${framing} of ${basePrompt}`);
    } else {
        parts.push(basePrompt);
    }

    if (camera && camera !== "None") {
        parts.push(`Shot on ${camera}`);
    }

    if (lighting && lighting !== "None") {
        parts.push(`${lighting}`);
    }

    if (style && style !== "None") {
        parts.push(style);
    }

    const preview = parts.join(". ").substring(0, 200) + (parts.join(". ").length > 200 ? "..." : "");

    previewWidget.value = preview;

    // Update DOM if exists
    if (previewWidget.inputEl) {
        previewWidget.inputEl.value = preview;
    }

    node.setDirtyCanvas(true);
}
