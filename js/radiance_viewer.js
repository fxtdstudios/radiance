/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                         FXTD RADIANCE VIEWER
 *                  VFX Industry-Standard Image Viewer
 *                        FXTD Studios © 2024-2026
 * 
 * Full Feature Set:
 * - Fullscreen (native browser), keyboard shortcuts
 * - Pixel probe with float values + copy to clipboard
 * - A/B comparison (wipe, side-by-side, difference)
 * - Professional scopes (Histogram, Waveform, Vectorscope) + overlay mode
 * - Annotations (circle, arrow, rectangle)
 * - Grid overlay (rule of thirds, center)
 * - Export snapshot, reset controls
 * ═══════════════════════════════════════════════════════════════════════════════
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

class RadianceViewer {
    constructor(node, container) {
        this.node = node;
        this.container = container;

        this.theme = {
            bg: '#0a0a0f',
            panel: 'rgba(16, 16, 24, 0.95)',
            panelBorder: 'rgba(60, 70, 100, 0.25)',
            accent: '#00a8ff',
            text: '#e8e8f0',
            textDim: '#707088',
        };

        // State
        this.image = null;
        this.compareImage = null;
        this.imageWidth = 0;
        this.imageHeight = 0;
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.exposure = 0.0;
        this.gamma = 1.0;
        this.channel = 'rgb';
        this.falseColor = false;
        this.zebra = false;
        this.colorspace = 'sRGB';

        // Batch Navigation
        this.currentFrame = 0;
        this.totalFrames = 1;
        this.frameImages = [];
        this.frameCompareImages = [];

        // Safe Area Guides
        this.safeAreaMode = 'none'; // none, action, title, both

        // Focus Peaking
        this.focusPeaking = false;
        this.focusPeakingColor = '#ff0000';
        this.focusPeakingThreshold = 30;

        // Pixel Loupe
        this.showLoupe = true;
        this.loupeSize = 80;
        this.loupeMagnification = 8;

        // Comparison
        this.compareMode = 'none';
        this.wipePosition = 0.5;
        this.isDraggingWipe = false;

        // Scopes
        this.showHistogram = false;
        this.showWaveform = false;
        this.showVectorscope = false;
        this.scopeOverlay = false;

        // Annotations
        this.annotations = [];
        this.isAnnotating = false;
        this.annotationTool = 'pen'; // pen, circle, arrow, rect, text
        this.annotationStart = null;
        this.annotationColor = '#ff4444';
        this.annotationLineWidth = 3;
        this.currentPath = null; // For pen drawing

        // Grid & Safe Areas
        this.showGrid = false;
        this.gridMode = 0; // 0=off, 1=thirds, 2=safe areas, 3=center

        // Fullscreen
        this.isFullscreen = false;

        // Pixel data
        this.imageData = null;
        this.lastPixelColor = null;

        this.initialized = false; // Track if we've set initial size

        // Run & Prompt
        this.showPromptPanel = false;
        this.promptPanel = null;
        this.isQueueing = false;

        // Progress
        this.progressBar = null;
        this.progressText = null;
        this.progressStart = 0;

        this.progressHistory = [];

        // Color Space / LUT
        this.displayLut = 'None';
        this.lutOptions = ['None', 'sRGB', 'Rec.709', 'LogC3', 'ACEScg'];

        this.init();
    }

    init() {
        this.createUI();

        this.setupProgressUI();
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        requestAnimationFrame(() => this.resize());
    }

    createUI() {
        const t = this.theme;

        this.container.style.cssText = `
            position: relative;
            width: 100%;
            height: 100%;
            min-height: 300px;
            background: linear-gradient(180deg, #0f0f14 0%, #08080c 100%);
            border-radius: 8px;
            overflow: hidden;
            user-select: none;
            border: 1px solid ${t.panelBorder};
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            display: flex;
            flex-direction: column;
        `;

        // Toolbar
        this.toolbar = document.createElement('div');
        this.toolbar.style.cssText = `
            flex: 0 0 32px;
            display: flex;
            align-items: center;
            gap: 2px;
            padding: 0 6px;
            background: ${t.panel};
            border-bottom: 1px solid ${t.panelBorder};
            overflow-x: auto;
        `;
        this.container.appendChild(this.toolbar);
        this.createToolbar();

        // Main area
        this.mainArea = document.createElement('div');
        this.mainArea.style.cssText = `flex: 1; display: flex; position: relative; overflow: hidden;`;
        this.container.appendChild(this.mainArea);

        // Canvas wrapper
        this.canvasWrapper = document.createElement('div');
        this.canvasWrapper.style.cssText = `flex: 1; position: relative; overflow: hidden;`;
        this.mainArea.appendChild(this.canvasWrapper);

        // Main canvas
        this.canvas = document.createElement('canvas');
        this.canvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair;`;
        this.canvas.tabIndex = 0;
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });
        this.canvasWrapper.appendChild(this.canvas);

        // Overlay canvas
        this.overlayCanvas = document.createElement('canvas');
        this.overlayCanvas.style.cssText = `position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;`;
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.canvasWrapper.appendChild(this.overlayCanvas);

        // Scope panel
        this.scopePanel = document.createElement('div');
        this.scopePanel.style.cssText = `
            display: none;
            flex: 0 0 160px;
            flex-direction: column;
            background: rgba(0,0,0,0.9);
            border-left: 1px solid ${t.panelBorder};
            padding: 4px;
            overflow-y: auto;
        `;
        this.mainArea.appendChild(this.scopePanel);
        this.createScopes();

        // Pixel probe tooltip
        this.probeTooltip = document.createElement('div');
        this.probeTooltip.style.cssText = `
            position: absolute;
            display: none;
            background: rgba(0,0,0,0.92);
            border: 1px solid ${t.panelBorder};
            border-radius: 4px;
            padding: 6px 8px;
            font-size: 9px;
            color: ${t.text};
            pointer-events: none;
            z-index: 1000;
            white-space: pre;
            cursor: pointer;
        `;
        this.canvasWrapper.appendChild(this.probeTooltip);

        // Status bar
        this.statusBar = document.createElement('div');
        this.statusBar.style.cssText = `
            flex: 0 0 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 8px;
            background: ${t.panel};
            border-top: 1px solid ${t.panelBorder};
            font-size: 9px;
            color: ${t.textDim};
        `;
        this.container.appendChild(this.statusBar);

        this.cursorInfo = document.createElement('span');
        this.cursorInfo.textContent = 'X: — Y: —';
        this.statusBar.appendChild(this.cursorInfo);

        this.colorInfo = document.createElement('span');
        this.colorInfo.textContent = 'RGB: — — —';
        this.colorInfo.style.cursor = 'pointer';
        this.colorInfo.title = 'Click to copy';
        this.colorInfo.onclick = () => this.copyColor();
        this.statusBar.appendChild(this.colorInfo);

        this.colorspaceInfo = document.createElement('span');
        this.colorspaceInfo.textContent = 'sRGB';
        this.colorspaceInfo.style.color = t.accent;
        this.statusBar.appendChild(this.colorspaceInfo);

        this.imageInfo = document.createElement('span');
        this.imageInfo.textContent = '— × — | 100%';
        this.statusBar.appendChild(this.imageInfo);
    }

    createScopes() {
        // Histogram
        this.histogramCanvas = document.createElement('canvas');
        this.histogramCanvas.width = 256;
        this.histogramCanvas.height = 50;
        this.histogramCanvas.style.cssText = 'width: 100%; height: 50px; display: none;';
        this.histogramCtx = this.histogramCanvas.getContext('2d');
        this.histogramLabel = this.createLabel('Histogram', true);
        this.scopePanel.appendChild(this.histogramLabel);
        this.scopePanel.appendChild(this.histogramCanvas);

        // Waveform
        this.waveformCanvas = document.createElement('canvas');
        this.waveformCanvas.width = 256;
        this.waveformCanvas.height = 50;
        this.waveformCanvas.style.cssText = 'width: 100%; height: 50px; display: none;';
        this.waveformCtx = this.waveformCanvas.getContext('2d');
        this.waveformLabel = this.createLabel('Waveform', true);
        this.scopePanel.appendChild(this.waveformLabel);
        this.scopePanel.appendChild(this.waveformCanvas);

        // Vectorscope
        this.vectorscopeCanvas = document.createElement('canvas');
        this.vectorscopeCanvas.width = 140;
        this.vectorscopeCanvas.height = 140;
        this.vectorscopeCanvas.style.cssText = 'width: 100%; aspect-ratio: 1; display: none;';
        this.vectorscopeCtx = this.vectorscopeCanvas.getContext('2d');
        this.vectorscopeLabel = this.createLabel('Vectorscope', true);
        this.scopePanel.appendChild(this.vectorscopeLabel);
        this.scopePanel.appendChild(this.vectorscopeCanvas);
    }

    createLabel(text, hidden = false) {
        const lbl = document.createElement('div');
        lbl.textContent = text;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 8px; margin: 4px 0 2px; display: ${hidden ? 'none' : 'block'};`;
        return lbl;
    }

    createToolbar() {
        // Fullscreen & Export
        this.addButton('⛶', () => this.toggleFullscreen(), 'Fullscreen');
        this.addButton('💾', () => this.exportSnapshot(), 'Export');
        this.addButton('↺', () => this.resetControls(), 'Reset');
        this.addSep();

        // Run & Editor
        this.runButton = this.addButton('▶', () => this.runWorkflow(), 'Run (Shift+Enter)');
        this.runButton.style.color = '#4f4'; // Green hue

        this.promptButton = this.addButton('P', () => this.togglePromptPanel(), 'Prompt Editor (P)');
        this.addSep();

        // EV
        this.addLbl('EV');
        this.evSlider = this.addSlider(-5, 5, 0, 0.1, v => { this.exposure = v; this.render(); });
        this.addSep();

        // Gamma
        this.addLbl('γ');
        this.gammaSlider = this.addSlider(0.2, 3.0, 1.0, 0.1, v => { this.gamma = v; this.render(); });
        this.addSep();

        // Zoom
        this.addButton('Fit', () => this.fitToView(), 'F');
        this.addButton('1:1', () => this.setZoom(1.0), '1');
        this.addSep();

        // LUT Dropdown
        this.addLbl('LUT');
        const lutSel = document.createElement('select');
        lutSel.style.cssText = `
            background: #181820;
            color: ${this.theme.text};
            border: 1px solid ${this.theme.panelBorder};
            border-radius: 4px;
            padding: 2px 4px;
            font-size: 11px;
            outline: none;
            cursor: pointer;
        `;
        this.lutOptions.forEach(opt => {
            const el = document.createElement('option');
            el.value = opt;
            el.textContent = opt;
            lutSel.appendChild(el);
        });
        lutSel.value = this.displayLut;
        lutSel.onchange = (e) => { this.displayLut = e.target.value; this.render(); };
        this.toolbar.appendChild(lutSel);

        this.addSep();

        // Channels (added Alpha)
        ['RGB', 'R', 'G', 'B', 'A', 'L'].forEach(ch => {
            this.addButton(ch, () => {
                this.channel = ch.toLowerCase() === 'l' ? 'luma' : ch.toLowerCase();
                this.render();
            }, ch === 'A' ? 'Alpha Channel' : '');
        });
        this.addSep();

        // Batch Navigation
        this.prevFrameBtn = this.addButton('◀', () => this.prevFrame(), 'Previous Frame (←)');
        this.frameDisplay = document.createElement('span');
        this.frameDisplay.textContent = '1/1';
        this.frameDisplay.style.cssText = `color: ${this.theme.text}; font-size: 9px; min-width: 32px; text-align: center;`;
        this.toolbar.appendChild(this.frameDisplay);
        this.nextFrameBtn = this.addButton('▶', () => this.nextFrame(), 'Next Frame (→)');
        this.addSep();

        // View
        this.addButton('FC', () => { this.falseColor = !this.falseColor; this.zebra = false; this.focusPeaking = false; this.render(); }, 'False Color (E)');
        this.addButton('Z', () => { this.zebra = !this.zebra; this.falseColor = false; this.focusPeaking = false; this.render(); }, 'Zebra');
        this.focusPeakingBtn = this.addButton('FP', () => { this.focusPeaking = !this.focusPeaking; this.falseColor = false; this.zebra = false; this.render(); }, 'Focus Peaking (K)');
        this.gridBtn = this.addButton('▦', () => this.cycleGridMode(), 'Grid / Safe Areas (G)');
        this.addButton('📺', () => this.cycleSafeAreas(), 'Safe Areas (S)');
        this.loupeBtn = this.addButton('🔍', () => { this.showLoupe = !this.showLoupe; }, 'Pixel Loupe (Q)');
        this.addSep();

        // Compare
        this.addButton('A|B', () => this.cycleCompareMode(), 'Compare (A)');
        this.addSep();

        // Scopes
        this.addButton('H', () => this.toggleScope('histogram'), 'Histogram');
        this.addButton('W', () => this.toggleScope('waveform'), 'Waveform');
        this.addButton('V', () => this.toggleScope('vectorscope'), 'Vectorscope');
        this.addButton('◫', () => { this.scopeOverlay = !this.scopeOverlay; this.renderOverlay(); }, 'Overlay');
        this.addSep();

        // Annotations
        this.annotBtns = {};

        // Color Picker
        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.value = this.annotationColor;
        colorInput.style.cssText = `width: 20px; height: 18px; border: none; padding: 0; background: none; cursor: pointer;`;
        colorInput.oninput = (e) => this.annotationColor = e.target.value;
        this.toolbar.appendChild(colorInput);

        // Size Selector
        const sizeSel = document.createElement('select');
        [1, 2, 3, 5, 8, 12].forEach(s => {
            const opt = document.createElement('option');
            opt.value = s; opt.text = s + 'px';
            if (s === this.annotationLineWidth) opt.selected = true;
            sizeSel.appendChild(opt);
        });
        sizeSel.style.cssText = `
            background: #1a1a28; color: ${this.theme.textDim}; border: 1px solid ${this.theme.panelBorder};
            border-radius: 3px; font-size: 9px; margin-right: 2px; height: 18px;
        `;
        sizeSel.onchange = (e) => this.annotationLineWidth = parseInt(e.target.value);
        this.toolbar.appendChild(sizeSel);

        this.annotBtns.pen = this.addButton('✎', () => this.setAnnotationTool('pen'), 'Pen');
        this.annotBtns.arrow = this.addButton('→', () => this.setAnnotationTool('arrow'), 'Arrow');
        this.annotBtns.rect = this.addButton('□', () => this.setAnnotationTool('rect'), 'Rectangle');
        this.annotBtns.circle = this.addButton('○', () => this.setAnnotationTool('circle'), 'Circle');
        this.annotBtns.text = this.addButton('T', () => this.setAnnotationTool('text'), 'Text');

        this.addButton('↩', () => this.undoAnnotation(), 'Undo');
        this.addButton('⟲', () => this.clearAnnotations(), 'Clear All');
    }

    addLbl(text) {
        const lbl = document.createElement('span');
        lbl.textContent = text;
        lbl.style.cssText = `color: ${this.theme.textDim}; font-size: 9px;`;
        this.toolbar.appendChild(lbl);
    }

    addButton(text, onClick, title = '') {
        const btn = document.createElement('button');
        btn.textContent = text;
        btn.title = title;
        btn.style.cssText = `
            padding: 2px 5px;
            background: linear-gradient(180deg, #1a1a28 0%, #12121a 100%);
            border: 1px solid ${this.theme.panelBorder};
            border-radius: 3px;
            color: ${this.theme.textDim};
            cursor: pointer;
            font-size: 9px;
        `;
        btn.onmouseenter = () => btn.style.color = this.theme.text;
        btn.onmouseleave = () => { if (!btn.classList.contains('active')) btn.style.color = this.theme.textDim; };
        btn.onclick = onClick;
        this.toolbar.appendChild(btn);
        return btn;
    }

    addSlider(min, max, value, step, onChange) {
        const w = document.createElement('div');
        w.style.cssText = 'display: flex; align-items: center; gap: 2px;';

        const input = document.createElement('input');
        input.type = 'range';
        input.min = min; input.max = max; input.value = value; input.step = step;
        input.style.cssText = `width: 38px; cursor: pointer; accent-color: ${this.theme.accent};`;

        const disp = document.createElement('span');
        disp.textContent = value.toFixed(1);
        disp.style.cssText = `min-width: 18px; text-align: right; color: ${this.theme.accent}; font-size: 8px;`;

        input.oninput = () => { disp.textContent = parseFloat(input.value).toFixed(1); onChange(parseFloat(input.value)); };
        w.appendChild(input);
        w.appendChild(disp);
        this.toolbar.appendChild(w);
        return { input, disp };
    }

    addSep() {
        const s = document.createElement('div');
        s.style.cssText = `width: 1px; height: 12px; background: ${this.theme.panelBorder}; margin: 0 2px;`;
        this.toolbar.appendChild(s);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RESET & COPY
    // ═══════════════════════════════════════════════════════════════════════════

    resetControls() {
        this.exposure = 0.0;
        this.gamma = 1.0;
        this.channel = 'rgb';
        this.falseColor = false;
        this.zebra = false;
        this.evSlider.input.value = 0;
        this.evSlider.disp.textContent = '0.0';
        this.gammaSlider.input.value = 1;
        this.gammaSlider.disp.textContent = '1.0';
        this.render();
    }

    copyColor() {
        if (this.lastPixelColor) {
            const { r, g, b } = this.lastPixelColor;
            const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
            navigator.clipboard.writeText(hex).then(() => {
                this.colorInfo.textContent = `Copied: ${hex}`;
                setTimeout(() => {
                    if (this.lastPixelColor) {
                        this.colorInfo.textContent = `RGB: ${this.lastPixelColor.r} ${this.lastPixelColor.g} ${this.lastPixelColor.b}`;
                    }
                }, 1000);
            });
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          ANNOTATIONS
    // ═══════════════════════════════════════════════════════════════════════════

    setAnnotationTool(tool) {
        this.annotationTool = tool;
        this.isAnnotating = true;
        this.canvas.style.cursor = 'crosshair';

        // Highlight active
        Object.keys(this.annotBtns).forEach(k => {
            const btn = this.annotBtns[k];
            if (k === tool) {
                btn.style.background = this.theme.accent;
                btn.style.color = '#fff';
                btn.classList.add('active');
            } else {
                btn.style.background = '';
                btn.style.color = this.theme.textDim;
                btn.classList.remove('active');
            }
        });
    }

    undoAnnotation() {
        if (this.annotations.length > 0) {
            this.annotations.pop();
            this.renderOverlay();
        }
    }

    addAnnotation(type, x1, y1, x2, y2, extra = {}) {
        this.annotations.push({
            type, x1, y1, x2, y2,
            color: this.annotationColor,
            width: this.annotationLineWidth,
            ...extra
        });
        this.renderOverlay();
    }

    clearAnnotations() {
        this.annotations = [];
        this.renderOverlay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          BATCH NAVIGATION
    // ═══════════════════════════════════════════════════════════════════════════

    prevFrame() {
        if (this.totalFrames <= 1) return;
        this.currentFrame = (this.currentFrame - 1 + this.totalFrames) % this.totalFrames;
        this.loadCurrentFrame();
    }

    nextFrame() {
        if (this.totalFrames <= 1) return;
        this.currentFrame = (this.currentFrame + 1) % this.totalFrames;
        this.loadCurrentFrame();
    }

    loadCurrentFrame() {
        if (this.frameImages[this.currentFrame]) {
            this.image = this.frameImages[this.currentFrame];
            this.updateFrameDisplay();
            this.render();
            this.updateScopes();
        }
    }

    updateFrameDisplay() {
        if (this.frameDisplay) {
            this.frameDisplay.textContent = `${this.currentFrame + 1}/${this.totalFrames}`;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          GRID & SAFE AREAS
    // ═══════════════════════════════════════════════════════════════════════════

    cycleGridMode() {
        // 0=off, 1=thirds, 2=safe areas, 3=center, 4=both
        this.gridMode = (this.gridMode + 1) % 5;
        this.showGrid = this.gridMode > 0;

        const labels = ['Off', 'Thirds', 'Safe', 'Center', 'All'];
        if (this.gridBtn) {
            this.gridBtn.title = `Grid: ${labels[this.gridMode]} (G)`;
        }
        this.renderOverlay();
    }

    cycleSafeAreas() {
        const modes = ['none', 'action', 'title', 'both'];
        const idx = modes.indexOf(this.safeAreaMode);
        this.safeAreaMode = modes[(idx + 1) % modes.length];
        this.renderOverlay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          KEYBOARD SHORTCUTS
    // ═══════════════════════════════════════════════════════════════════════════

    setupKeyboardShortcuts() {
        const handler = (e) => {
            if (e.target.tagName === 'INPUT') return;
            this.handleKey(e);
        };
        this.canvas.addEventListener('keydown', handler);
        document.addEventListener('keydown', (e) => { if (this.isFullscreen) handler(e); });
    }

    handleKey(e) {
        const key = e.key.toLowerCase();
        switch (key) {
            case 'f': this.fitToView(); break;
            case '1': this.setZoom(1.0); break;
            case 'r': this.channel = 'r'; this.render(); break;
            case 'g': if (!e.ctrlKey) { this.cycleGridMode(); } break;
            case 'b': this.channel = 'b'; this.render(); break;
            case 'l': this.channel = 'luma'; this.render(); break;
            case 'c': this.channel = 'rgb'; this.render(); break;
            case 'h': this.toggleScope('histogram'); break;
            case 'w': this.toggleScope('waveform'); break;
            case 'v': this.toggleScope('vectorscope'); break;
            case 'e': this.falseColor = !this.falseColor; this.zebra = false; this.focusPeaking = false; this.render(); break;
            case 'k': this.focusPeaking = !this.focusPeaking; this.falseColor = false; this.zebra = false; this.render(); break;
            case 'q': this.showLoupe = !this.showLoupe; break;
            case 'a':
                if (e.shiftKey) { this.channel = 'a'; this.render(); }
                else { this.cycleCompareMode(); }
                break;
            case 's': if (!e.ctrlKey) this.cycleSafeAreas(); break;
            case 'arrowleft': this.prevFrame(); break;
            case 'arrowright': this.nextFrame(); break;
            case 'escape': if (this.isFullscreen) this.exitFullscreen(); break;
            case '=': case '+': this.adjustEV(0.5); break;
            case '-': this.adjustEV(-0.5); break;
            case '0': this.resetControls(); break;
            case 'p': if (!e.ctrlKey) this.togglePromptPanel(); break;
            case 'enter': if (e.shiftKey) this.runWorkflow(); break;
        }
    }

    adjustEV(delta) {
        this.exposure = Math.max(-5, Math.min(5, this.exposure + delta));
        this.evSlider.input.value = this.exposure;
        this.evSlider.disp.textContent = this.exposure.toFixed(1);
        this.render();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          FULLSCREEN
    // ═══════════════════════════════════════════════════════════════════════════

    toggleFullscreen() {
        if (!this.isFullscreen) {
            const elem = this.container;
            if (elem.requestFullscreen) elem.requestFullscreen();
            else if (elem.webkitRequestFullscreen) elem.webkitRequestFullscreen();
            this.isFullscreen = true;

            const handler = () => {
                if (!document.fullscreenElement && !document.webkitFullscreenElement) {
                    this.isFullscreen = false;
                    document.removeEventListener('fullscreenchange', handler);
                }
                requestAnimationFrame(() => { this.resize(); this.render(); });
            };
            document.addEventListener('fullscreenchange', handler);
            document.addEventListener('webkitfullscreenchange', handler);
        } else {
            this.exitFullscreen();
        }
    }

    exitFullscreen() {
        if (document.exitFullscreen) document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
        this.isFullscreen = false;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          EXPORT
    // ═══════════════════════════════════════════════════════════════════════════

    exportSnapshot() {
        if (!this.image) return;
        const exp = document.createElement('canvas');
        exp.width = this.imageWidth;
        exp.height = this.imageHeight;
        const ctx = exp.getContext('2d');
        this.renderImage(ctx, this.image);
        this.annotations.forEach(a => this.drawAnnotation(ctx, a, 1));

        const link = document.createElement('a');
        link.download = `radiance_${Date.now()}.png`;
        link.href = exp.toDataURL('image/png');
        link.click();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          SCOPES
    // ═══════════════════════════════════════════════════════════════════════════

    toggleScope(scope) {
        if (scope === 'histogram') {
            this.showHistogram = !this.showHistogram;
            this.histogramCanvas.style.display = this.showHistogram ? 'block' : 'none';
            this.histogramLabel.style.display = this.showHistogram ? 'block' : 'none';
        } else if (scope === 'waveform') {
            this.showWaveform = !this.showWaveform;
            this.waveformCanvas.style.display = this.showWaveform ? 'block' : 'none';
            this.waveformLabel.style.display = this.showWaveform ? 'block' : 'none';
        } else if (scope === 'vectorscope') {
            this.showVectorscope = !this.showVectorscope;
            this.vectorscopeCanvas.style.display = this.showVectorscope ? 'block' : 'none';
            this.vectorscopeLabel.style.display = this.showVectorscope ? 'block' : 'none';
        }

        const any = this.showHistogram || this.showWaveform || this.showVectorscope;
        this.scopePanel.style.display = any ? 'flex' : 'none';
        if (this.image) this.updateScopes();
    }

    cycleCompareMode() {
        if (!this.compareImage) { this.compareMode = 'none'; return; }
        const modes = ['none', 'wipe', 'sidebyside', 'difference'];
        this.compareMode = modes[(modes.indexOf(this.compareMode) + 1) % modes.length];
        this.render();
    }

    updateScopes() {
        if (this.showHistogram) this.updateHistogram();
        if (this.showWaveform) this.updateWaveform();
        if (this.showVectorscope) this.updateVectorscope();
        if (this.scopeOverlay) this.renderOverlay();
    }

    updateHistogram() {
        if (!this.image || !this.imageData) return;
        const data = this.imageData; // Use cached data

        const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);
        for (let i = 0; i < data.length; i += 4) { hR[data[i]]++; hG[data[i + 1]]++; hB[data[i + 2]]++; }

        let max = 1;
        for (let i = 0; i < 256; i++) max = Math.max(max, hR[i], hG[i], hB[i]);

        this.histogramData = { hR, hG, hB, max };

        const hCtx = this.histogramCtx;
        const w = this.histogramCanvas.width, h = this.histogramCanvas.height;
        hCtx.fillStyle = '#0a0a0f';
        hCtx.fillRect(0, 0, w, h);

        const draw = (hist, color) => {
            hCtx.strokeStyle = color; hCtx.lineWidth = 1; hCtx.beginPath();
            for (let i = 0; i < 256; i++) {
                const y = h - (hist[i] / max) * h;
                if (i === 0) hCtx.moveTo(i, y); else hCtx.lineTo(i, y);
            }
            hCtx.stroke();
        };
        hCtx.globalAlpha = 0.7;
        draw(hR, '#ff4444'); draw(hG, '#44ff44'); draw(hB, '#4444ff');
        hCtx.globalAlpha = 1;
    }

    updateWaveform() {
        if (!this.image || !this.imageData) return;
        const data = this.imageData; // Use cached data

        const wCtx = this.waveformCtx;
        const w = this.waveformCanvas.width, h = this.waveformCanvas.height;
        wCtx.fillStyle = '#0a0a0f';
        wCtx.fillRect(0, 0, w, h);

        const step = Math.max(1, Math.floor(this.imageWidth / w));
        wCtx.globalAlpha = 0.1;
        for (let col = 0; col < this.imageWidth; col += step) {
            const x = Math.floor((col / this.imageWidth) * w);
            for (let row = 0; row < this.imageHeight; row += 3) {
                const idx = (row * this.imageWidth + col) * 4;
                const luma = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
                const y = h - (luma / 255) * h;
                wCtx.fillStyle = `rgb(${data[idx]},${data[idx + 1]},${data[idx + 2]})`;
                wCtx.fillRect(x, y, 1, 1);
            }
        }
        wCtx.globalAlpha = 1;
    }

    updateVectorscope() {
        if (!this.image || !this.imageData) return;
        const data = this.imageData; // Use cached data

        const vCtx = this.vectorscopeCtx;
        const size = this.vectorscopeCanvas.width;
        const cx = size / 2, cy = size / 2, rad = size / 2 - 6;

        vCtx.fillStyle = '#0a0a0f';
        vCtx.fillRect(0, 0, size, size);

        vCtx.strokeStyle = '#222'; vCtx.lineWidth = 1;
        for (let r = 0.25; r <= 1; r += 0.25) { vCtx.beginPath(); vCtx.arc(cx, cy, rad * r, 0, Math.PI * 2); vCtx.stroke(); }

        const targets = [{ a: 103, c: '#f00' }, { a: 167, c: '#ff0' }, { a: 241, c: '#0f0' }, { a: 283, c: '#0ff' }, { a: 347, c: '#00f' }, { a: 61, c: '#f0f' }];
        targets.forEach(t => {
            const ang = (t.a - 90) * Math.PI / 180;
            vCtx.fillStyle = t.c;
            vCtx.beginPath();
            vCtx.arc(cx + Math.cos(ang) * rad * 0.75, cy + Math.sin(ang) * rad * 0.75, 2, 0, Math.PI * 2);
            vCtx.fill();
        });

        vCtx.globalAlpha = 0.03;
        const step = Math.max(1, Math.floor(data.length / 4 / 30000));
        for (let i = 0; i < data.length; i += 4 * step) {
            const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
            const y = r * 0.299 + g * 0.587 + b * 0.114;
            const u = (b - y) * 0.492, v = (r - y) * 0.877;
            vCtx.fillStyle = `rgb(${data[i]},${data[i + 1]},${data[i + 2]})`;
            vCtx.fillRect(cx + u * rad * 2, cy - v * rad * 2, 1, 1);
        }
        vCtx.globalAlpha = 1;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          OVERLAY RENDERING
    // ═══════════════════════════════════════════════════════════════════════════

    renderOverlay() {
        const ctx = this.overlayCtx;
        const w = this.overlayCanvas.width, h = this.overlayCanvas.height;
        ctx.clearRect(0, 0, w, h);

        // Grid
        if (this.showGrid) this.drawGrid(ctx, w, h);

        // Annotations
        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        this.annotations.forEach(a => this.drawAnnotation(ctx, a, 1 / this.zoom));

        // Draw current stroke if active
        if (this.isAnnotating && this.currentPath && this.annotationTool === 'pen') {
            this.drawAnnotation(ctx, {
                type: 'pen',
                points: this.currentPath,
                color: this.annotationColor,
                width: this.annotationLineWidth
            }, 1 / this.zoom);
        } else if (this.isAnnotating && this.annotationStart && this.annotationTool !== 'text') {
            // Preview shapes (simplified logic: we don't have mouse pos here easily without storing it, 
            // but `renderOverlay` is called usually when done or panning. 
            // For drag interaction, we need mouse move to trigger render.
        }

        ctx.restore();

        // Scope overlay
        if (this.scopeOverlay && this.histogramData) this.drawHistogramOverlay(ctx, w, h);
    }

    drawGrid(ctx, w, h) {
        // Grid mode: 1=thirds, 2=safe areas, 3=center, 4=all

        // Rule of thirds (mode 1 or 4)
        if (this.gridMode === 1 || this.gridMode === 4) {
            ctx.strokeStyle = 'rgba(255,255,255,0.2)';
            ctx.lineWidth = 1;
            for (let i = 1; i <= 2; i++) {
                ctx.beginPath();
                ctx.moveTo(w * i / 3, 0); ctx.lineTo(w * i / 3, h);
                ctx.moveTo(0, h * i / 3); ctx.lineTo(w, h * i / 3);
                ctx.stroke();
            }
        }

        // Safe areas (mode 2 or 4, or via safeAreaMode)
        const showSafeFromGrid = this.gridMode === 2 || this.gridMode === 4;
        const showActionSafe = showSafeFromGrid || this.safeAreaMode === 'action' || this.safeAreaMode === 'both';
        const showTitleSafe = showSafeFromGrid || this.safeAreaMode === 'title' || this.safeAreaMode === 'both';

        // Action Safe (93% - broadcast safe)
        if (showActionSafe) {
            ctx.strokeStyle = 'rgba(0, 200, 255, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([8, 4]);
            const actionMargin = 0.035; // 3.5% margin = 93% visible
            ctx.beginPath();
            ctx.rect(
                w * actionMargin, h * actionMargin,
                w * (1 - 2 * actionMargin), h * (1 - 2 * actionMargin)
            );
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = 'rgba(0, 200, 255, 0.6)';
            ctx.font = '9px sans-serif';
            ctx.fillText('Action Safe 93%', w * actionMargin + 4, h * actionMargin + 12);
        }

        // Title Safe (80% - text safe)
        if (showTitleSafe) {
            ctx.strokeStyle = 'rgba(255, 200, 0, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            const titleMargin = 0.10; // 10% margin = 80% visible
            ctx.beginPath();
            ctx.rect(
                w * titleMargin, h * titleMargin,
                w * (1 - 2 * titleMargin), h * (1 - 2 * titleMargin)
            );
            ctx.stroke();
            ctx.setLineDash([]);

            // Label
            ctx.fillStyle = 'rgba(255, 200, 0, 0.6)';
            ctx.font = '9px sans-serif';
            ctx.fillText('Title Safe 80%', w * titleMargin + 4, h * titleMargin + 12);
        }

        // Center cross (mode 3 or 4)
        if (this.gridMode === 3 || this.gridMode === 4) {
            ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(w / 2 - 30, h / 2); ctx.lineTo(w / 2 + 30, h / 2);
            ctx.moveTo(w / 2, h / 2 - 30); ctx.lineTo(w / 2, h / 2 + 30);
            ctx.stroke();

            // Center circle
            ctx.beginPath();
            ctx.arc(w / 2, h / 2, 5, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    drawAnnotation(ctx, a, lineScale) {
        ctx.strokeStyle = a.color;
        ctx.fillStyle = a.color;
        ctx.lineWidth = (a.width || 3) * lineScale;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();

        if (a.type === 'pen') {
            if (a.points && a.points.length > 0) {
                ctx.moveTo(a.points[0].x, a.points[0].y);
                for (let i = 1; i < a.points.length; i++) {
                    ctx.lineTo(a.points[i].x, a.points[i].y);
                }
            }
            ctx.stroke();
            return;
        }

        if (a.type === 'text') {
            ctx.font = `bold ${Math.max(10, (a.width || 3) * 5)}px sans-serif`;
            // Draw background for readability
            ctx.save();
            ctx.scale(1 / this.zoom, 1 / this.zoom); // Keep text constant size or scale? Let's scale with image so it stays fixed in place
            ctx.restore();
            // Actually, keep it simple: text scales with image

            ctx.shadowColor = 'black';
            ctx.shadowBlur = 4 * lineScale;
            ctx.lineWidth = 1 * lineScale; // reset for text stroke

            ctx.fillText(a.text, a.x1, a.y1);
            // ctx.strokeText(a.text, a.x1, a.y1); // optional outline
            return;
        }

        if (a.type === 'circle') {
            const rx = Math.abs(a.x2 - a.x1) / 2, ry = Math.abs(a.y2 - a.y1) / 2;
            ctx.ellipse((a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2, rx, ry, 0, 0, Math.PI * 2);
        } else if (a.type === 'arrow') {
            ctx.moveTo(a.x1, a.y1); ctx.lineTo(a.x2, a.y2);
            const ang = Math.atan2(a.y2 - a.y1, a.x2 - a.x1);
            const hl = (12 + (a.width || 3)) * lineScale;
            ctx.lineTo(a.x2 - hl * Math.cos(ang - 0.4), a.y2 - hl * Math.sin(ang - 0.4));
            ctx.moveTo(a.x2, a.y2);
            ctx.lineTo(a.x2 - hl * Math.cos(ang + 0.4), a.y2 - hl * Math.sin(ang + 0.4));
        } else if (a.type === 'rect') {
            ctx.rect(a.x1, a.y1, a.x2 - a.x1, a.y2 - a.y1);
        }
        ctx.stroke();
    }

    drawHistogramOverlay(ctx, w, h) {
        const { hR, hG, hB, max } = this.histogramData;
        const hx = w - 130, hy = 8, hw = 120, hh = 40;

        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(hx - 4, hy - 4, hw + 8, hh + 8);

        const draw = (hist, color) => {
            ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const x = hx + (i / 255) * hw, y = hy + hh - (hist[i] / max) * hh;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };
        ctx.globalAlpha = 0.6;
        draw(hR, '#f44'); draw(hG, '#4f4'); draw(hB, '#44f');
        ctx.globalAlpha = 1;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          EVENTS
    // ═══════════════════════════════════════════════════════════════════════════

    setupEventListeners() {
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left, my = e.clientY - rect.top;
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.02, Math.min(100, this.zoom * factor));
            this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
            this.panY = my - (my - this.panY) * (newZoom / this.zoom);
            this.zoom = newZoom;
            this.updateInfo();
            this.render();
        });

        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left, my = e.clientY - rect.top;
            const x = (mx - this.panX) / this.zoom;
            const y = (my - this.panY) / this.zoom;

            if (this.compareMode === 'wipe' && Math.abs(mx - rect.width * this.wipePosition) < 10) {
                this.isDraggingWipe = true; return;
            }

            if (this.isAnnotating && e.button === 0) {
                if (this.annotationTool === 'text') {
                    const text = prompt("Enter text markup:");
                    if (text) {
                        this.addAnnotation('text', x, y, x, y, { text });
                    }
                    return;
                }

                if (this.annotationTool === 'pen') {
                    this.currentPath = [{ x, y }];
                }

                this.annotationStart = { x, y };
                return;
            }

            if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
                this.isPanning = true;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.canvas.style.cursor = 'grabbing';
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left, my = e.clientY - rect.top;

            if (this.isDraggingWipe) {
                this.wipePosition = Math.max(0.02, Math.min(0.98, mx / rect.width));
                this.render(); return;
            }

            if (this.isPanning) {
                this.panX += e.clientX - this.lastMouseX;
                this.panY += e.clientY - this.lastMouseY;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.render();
            }

            // Handle Drawing Preview
            if (this.isAnnotating && this.annotationStart) {
                const x = (mx - this.panX) / this.zoom;
                const y = (my - this.panY) / this.zoom;

                if (this.annotationTool === 'pen' && this.currentPath) {
                    this.currentPath.push({ x, y });
                    this.renderOverlay();
                } else if (['circle', 'arrow', 'rect'].includes(this.annotationTool)) {
                    // We need to render a preview. 
                    // The most efficient way is to just redraw overlay with a temporary shape.
                    this.renderOverlay();

                    // Draw preview directly here to avoid state complexity? 
                    // Or better, add a "preview" param to renderOverlay or just use a temp field.
                    // Let's use direct drawing on overlay context for performance? 
                    // Actually `renderOverlay` clears everything. So we must redraw all annotations + preview.

                    // Let's do a manual preview draw after `renderOverlay` call?
                    // No, `renderOverlay` is called above. Let's make `renderOverlay` aware of drag.
                    const ctx = this.overlayCtx;
                    ctx.save();
                    ctx.translate(this.panX, this.panY);
                    ctx.scale(this.zoom, this.zoom);
                    this.drawAnnotation(ctx, {
                        type: this.annotationTool,
                        x1: this.annotationStart.x,
                        y1: this.annotationStart.y,
                        x2: x,
                        y2: y,
                        color: this.annotationColor,
                        width: this.annotationLineWidth
                    }, 1 / this.zoom);
                    ctx.restore();
                }
            }

            this.updateCursor(e);
            this.updateProbe(e);
        });

        this.canvas.addEventListener('mouseup', (e) => {
            if (this.isAnnotating && this.annotationStart) {
                const rect = this.canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left, my = e.clientY - rect.top;
                const x2 = (mx - this.panX) / this.zoom, y2 = (my - this.panY) / this.zoom;

                if (this.annotationTool === 'pen' && this.currentPath) {
                    this.addAnnotation('pen', 0, 0, 0, 0, { points: this.currentPath });
                    this.currentPath = null;
                } else if (this.annotationTool !== 'text') {
                    this.addAnnotation(this.annotationTool, this.annotationStart.x, this.annotationStart.y, x2, y2);
                }

                this.annotationStart = null;
                this.renderOverlay();
            }
        });

        window.addEventListener('mouseup', () => {
            this.isPanning = false;
            this.isDraggingWipe = false;
            if (!this.isAnnotating) this.canvas.style.cursor = 'crosshair';
        });

        this.canvas.addEventListener('mouseleave', () => { this.probeTooltip.style.display = 'none'; });

        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(this.canvasWrapper);
        }
    }

    resize() {
        const rect = this.canvasWrapper.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        this.canvas.width = Math.floor(rect.width);
        this.canvas.height = Math.floor(rect.height);
        this.overlayCanvas.width = this.canvas.width;
        this.overlayCanvas.height = this.canvas.height;
        this.render();
    }

    updateCursor(e) {
        if (!this.image) return;
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const imgX = Math.floor((mx - this.panX) / this.zoom);
        const imgY = Math.floor((my - this.panY) / this.zoom);
        if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
            this.cursorInfo.textContent = `X: ${imgX} Y: ${imgY}`;
        } else {
            this.cursorInfo.textContent = 'X: — Y: —';
        }
    }

    updateProbe(e) {
        if (!this.image || !this.imageData) { this.probeTooltip.style.display = 'none'; return; }
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const imgX = Math.floor((mx - this.panX) / this.zoom);
        const imgY = Math.floor((my - this.panY) / this.zoom);

        if (imgX >= 0 && imgX < this.imageWidth && imgY >= 0 && imgY < this.imageHeight) {
            const idx = (imgY * this.imageWidth + imgX) * 4;
            const r = this.imageData[idx], g = this.imageData[idx + 1], b = this.imageData[idx + 2], a = this.imageData[idx + 3];
            this.lastPixelColor = { r, g, b, a };
            const luma = (r * 0.2126 + g * 0.7152 + b * 0.0722).toFixed(0);
            const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;

            this.colorInfo.textContent = `RGB: ${r} ${g} ${b}`;
            this.probeTooltip.innerHTML = `<b>${imgX}, ${imgY}</b>\n8-bit: ${r} ${g} ${b} ${a !== 255 ? 'A:' + a : ''}\nFloat: ${(r / 255).toFixed(3)} ${(g / 255).toFixed(3)} ${(b / 255).toFixed(3)}\nHex: ${hex}\nLuma: ${luma}`;
            this.probeTooltip.style.display = 'block';
            this.probeTooltip.style.left = `${mx + 12}px`;
            this.probeTooltip.style.top = `${my + 12}px`;

            // Draw pixel loupe on overlay
            if (this.showLoupe) {
                this.drawLoupe(mx, my, imgX, imgY);
            }
        } else {
            this.probeTooltip.style.display = 'none';
            this.colorInfo.textContent = 'RGB: — — —';
            this.lastPixelColor = null;
        }
    }

    drawLoupe(mx, my, imgX, imgY) {
        const ctx = this.overlayCtx;
        const size = this.loupeSize;
        const mag = this.loupeMagnification;
        const halfPixels = Math.floor(size / mag / 2);

        // Position loupe in corner opposite to cursor
        let lx = mx > this.canvas.width / 2 ? 10 : this.canvas.width - size - 10;
        let ly = my > this.canvas.height / 2 ? 10 : this.canvas.height - size - 10;

        // Draw loupe background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(lx - 2, ly - 2, size + 4, size + 4, 4);
        ctx.fill();
        ctx.stroke();

        // Draw magnified pixels
        for (let py = -halfPixels; py <= halfPixels; py++) {
            for (let px = -halfPixels; px <= halfPixels; px++) {
                const sx = imgX + px, sy = imgY + py;
                if (sx >= 0 && sx < this.imageWidth && sy >= 0 && sy < this.imageHeight) {
                    const idx = (sy * this.imageWidth + sx) * 4;
                    const r = this.imageData[idx], g = this.imageData[idx + 1], b = this.imageData[idx + 2];
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                } else {
                    ctx.fillStyle = '#222';
                }

                const dx = lx + (px + halfPixels) * mag;
                const dy = ly + (py + halfPixels) * mag;
                ctx.fillRect(dx, dy, mag, mag);

                // Highlight center pixel
                if (px === 0 && py === 0) {
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(dx, dy, mag, mag);
                }
            }
        }

        // Draw crosshair
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lx + size / 2, ly); ctx.lineTo(lx + size / 2, ly + size);
        ctx.moveTo(lx, ly + size / 2); ctx.lineTo(lx + size, ly + size / 2);
        ctx.stroke();
    }

    updateInfo() {
        this.imageInfo.textContent = `${this.imageWidth} × ${this.imageHeight} | ${(this.zoom * 100).toFixed(0)}%`;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          IMAGE
    // ═══════════════════════════════════════════════════════════════════════════

    setImage(img) {
        this.image = img;
        this.imageWidth = img.width;
        this.imageHeight = img.height;
        this.diffCanvas = null; // Clear cache

        const off = document.createElement('canvas');
        off.width = img.width; off.height = img.height;
        const ctx = off.getContext('2d');
        ctx.drawImage(img, 0, 0);
        this.imageData = ctx.getImageData(0, 0, img.width, img.height).data;

        if (this.node && !this.isFullscreen) {
            // Auto-resize vertically to fit image aspect ratio while respecting current width
            const nodeW = this.node.size[0];
            const aspect = this.imageHeight / this.imageWidth;
            const clampedAspect = Math.min(2.0, Math.max(0.5, aspect));
            const newH = Math.max(300, nodeW * clampedAspect + 80);

            this.node.setSize([nodeW, newH]);
            this.node.setDirtyCanvas(true, true);
        }

        // Wait for ComfyUI/Browser layout update
        setTimeout(() => {
            this.resize();
            this.fitToView();
            this.updateInfo();
            this.updateScopes();
        }, 50);
    }

    setCompareImage(img) {
        this.compareImage = img;
        this.diffCanvas = null; // Clear difference cache
        if (this.compareMode === 'none') this.compareMode = 'wipe';
        this.render();
    }

    fitToView() {
        if (!this.image) return;
        const w = this.canvas.width, h = this.canvas.height;
        if (w < 10 || h < 10) {
            // Retry if canvas is somehow not ready
            requestAnimationFrame(() => this.fitToView());
            return;
        }

        let z = Math.min(w / this.imageWidth, h / this.imageHeight);
        // Add a small margin (5%) if it's tight
        z = z * 0.95;

        this.zoom = z;
        this.panX = (w - this.imageWidth * this.zoom) / 2;
        this.panY = (h - this.imageHeight * this.zoom) / 2;
        this.updateInfo();
        this.render();
    }

    setZoom(z) {
        const cx = this.canvas.width / 2, cy = this.canvas.height / 2;
        // Zoom towards center of view, not 0,0
        const oldZ = this.zoom;

        // Calculate world point at center
        const wx = (cx - this.panX) / oldZ;
        const wy = (cy - this.panY) / oldZ;

        this.zoom = z;
        // Recalculate pan to keep world point at center
        this.panX = cx - wx * this.zoom;
        this.panY = cy - wy * this.zoom;

        this.updateInfo();
        this.render();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RENDERING
    // ═══════════════════════════════════════════════════════════════════════════

    render() {
        const ctx = this.ctx;
        const w = this.canvas.width, h = this.canvas.height;
        if (w === 0 || h === 0) return;

        ctx.fillStyle = this.theme.bg;
        ctx.fillRect(0, 0, w, h);

        if (!this.image) {
            ctx.fillStyle = this.theme.textDim;
            ctx.font = '12px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No image loaded', w / 2, h / 2);
            return;
        }

        // Ensure high quality scaling
        ctx.imageSmoothingEnabled = this.zoom < 1.0; // Smooth when downscaling, pixelated when upscaling?
        if (this.zoom > 2.0) ctx.imageSmoothingEnabled = false; // Pixel art look for high zoom
        else ctx.imageSmoothingQuality = 'high';

        if (this.compareMode === 'sidebyside' && this.compareImage) {
            this.renderSideBySide(ctx, w, h);
        } else if (this.compareMode === 'difference' && this.compareImage) {
            this.renderDifference(ctx, w, h);
        } else {
            ctx.save();
            ctx.translate(this.panX, this.panY);
            ctx.scale(this.zoom, this.zoom);
            this.renderImage(ctx, this.image);
            ctx.restore();

            if (this.compareMode === 'wipe' && this.compareImage) this.renderWipe(ctx, w, h);
        }

        this.renderOverlay();
    }

    renderImage(ctx, img) {
        if (this.exposure !== 0 || this.gamma !== 1.0 || this.channel !== 'rgb' || this.falseColor || this.zebra || this.focusPeaking || this.displayLut !== 'None') {
            this.renderProcessed(ctx, img);
        } else {
            ctx.drawImage(img, 0, 0);
        }
    }

    renderWipe(ctx, w, h) {
        const wipeX = w * this.wipePosition;
        ctx.save();
        ctx.beginPath(); ctx.rect(wipeX, 0, w - wipeX, h); ctx.clip();
        ctx.translate(this.panX, this.panY); ctx.scale(this.zoom, this.zoom);
        ctx.drawImage(this.compareImage, 0, 0);
        ctx.restore();

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(wipeX, 0); ctx.lineTo(wipeX, h); ctx.stroke();

        // Handle
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.beginPath(); ctx.arc(wipeX, h / 2, 10, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(wipeX, h / 2, 4, 0, Math.PI * 2); ctx.fill();
    }

    renderSideBySide(ctx, w, h) {
        const hw = w / 2;
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, hw, h); ctx.clip();
        ctx.translate(this.panX * 0.5, this.panY); ctx.scale(this.zoom * 0.5, this.zoom); // Zoom needs adjustment for half width? No, keep relative
        // Actually for SxS usually we behave as two separate viewports or just cropped
        // Let's implement cropped view for better comparison
        ctx.drawImage(this.image, 0, 0); ctx.restore();

        ctx.save(); ctx.beginPath(); ctx.rect(hw, 0, hw, h); ctx.clip();
        ctx.translate(hw + this.panX * 0.5, this.panY); ctx.scale(this.zoom * 0.5, this.zoom);
        ctx.drawImage(this.compareImage, 0, 0); ctx.restore();

        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(hw, 0); ctx.lineTo(hw, h); ctx.stroke();
    }

    renderDifference(ctx, w, h) {
        if (!this.diffCanvas) {
            this.diffCanvas = document.createElement('canvas');
            this.diffCanvas.width = this.imageWidth;
            this.diffCanvas.height = this.imageHeight;
            const dCtx = this.diffCanvas.getContext('2d');

            // Draw A
            dCtx.drawImage(this.image, 0, 0);
            const dA = dCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);

            // Draw B to temporary canvas to get data
            const tmp = document.createElement('canvas');
            tmp.width = this.imageWidth; tmp.height = this.imageHeight;
            const tCtx = tmp.getContext('2d');
            tCtx.drawImage(this.compareImage, 0, 0);
            const dB = tCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);

            // Compute Difference
            for (let i = 0; i < dA.data.length; i += 4) {
                const r = Math.abs(dA.data[i] - dB.data[i]);
                const g = Math.abs(dA.data[i + 1] - dB.data[i + 1]);
                const b = Math.abs(dA.data[i + 2] - dB.data[i + 2]);

                // Visualization: Boost difference
                dA.data[i] = Math.min(255, r * 4);
                dA.data[i + 1] = Math.min(255, g * 4);
                dA.data[i + 2] = Math.min(255, b * 4);
                dA.data[i + 3] = 255;
            }
            dCtx.putImageData(dA, 0, 0);
        }

        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        ctx.drawImage(this.diffCanvas, 0, 0);
        ctx.restore();
    }

    renderProcessed(ctx, img) {
        const off = document.createElement('canvas');
        off.width = this.imageWidth; off.height = this.imageHeight;
        const offCtx = off.getContext('2d');
        offCtx.drawImage(img, 0, 0);

        const imageData = offCtx.getImageData(0, 0, this.imageWidth, this.imageHeight);
        const data = imageData.data;
        const expMult = Math.pow(2, this.exposure);
        const invGamma = 1.0 / this.gamma;

        for (let i = 0; i < data.length; i += 4) {
            let r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;

            // Apply exposure before gamma
            r *= expMult; g *= expMult; b *= expMult;

            // Apply LUT / Color Space
            if (this.displayLut !== 'None') {
                [r, g, b] = this.applyColorTransform(r, g, b, this.displayLut);
            }

            // Apply Gamma
            if (this.gamma !== 1.0) {
                r = Math.pow(Math.max(0, r), invGamma);
                g = Math.pow(Math.max(0, g), invGamma);
                b = Math.pow(Math.max(0, b), invGamma);
            }

            if (this.channel === 'r') { g = r; b = r; }
            else if (this.channel === 'g') { r = g; b = g; }
            else if (this.channel === 'b') { r = b; g = b; }
            else if (this.channel === 'a') {
                // Alpha channel - show as grayscale with checkerboard for transparency
                const a = data[i + 3] / 255;
                const x = (i / 4) % this.imageWidth, y = Math.floor((i / 4) / this.imageWidth);
                const checker = ((Math.floor(x / 8) + Math.floor(y / 8)) % 2) === 0 ? 0.3 : 0.5;
                r = g = b = a * 1.0 + (1 - a) * checker;
            }
            else if (this.channel === 'luma') { const l = r * 0.2126 + g * 0.7152 + b * 0.0722; r = g = b = l; }

            if (this.falseColor) {
                const l = r * 0.2126 + g * 0.7152 + b * 0.0722;
                const fc = this.getFalseColor(l);
                r = fc.r; g = fc.g; b = fc.b;
            }

            if (this.zebra) {
                const l = r * 0.2126 + g * 0.7152 + b * 0.0722;
                const x = (i / 4) % this.imageWidth, y = Math.floor((i / 4) / this.imageWidth);
                if (l > 0.95 && (x + y) % 8 < 4) { r = 1; g = 0; b = 0; }
                if (l < 0.02 && (x + y) % 8 < 4) { r = 0; g = 0; b = 1; }
            }

            // Clip final values
            data[i] = Math.min(255, Math.max(0, r * 255));
            data[i + 1] = Math.min(255, Math.max(0, g * 255));
            data[i + 2] = Math.min(255, Math.max(0, b * 255));
        }

        // Focus peaking - apply as post-process for edge detection
        if (this.focusPeaking) {
            this.applyFocusPeaking(imageData, this.imageWidth, this.imageHeight);
        }

        offCtx.putImageData(imageData, 0, 0);
        ctx.drawImage(off, 0, 0);
    }

    applyFocusPeaking(imageData, width, height) {
        const data = imageData.data;
        const threshold = this.focusPeakingThreshold;

        // Create edge buffer
        const edges = new Uint8Array(width * height);

        // Sobel edge detection on luminance
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;

                // Get luminance of surrounding pixels
                const getLuma = (ox, oy) => {
                    const i = ((y + oy) * width + (x + ox)) * 4;
                    return data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
                };

                // Sobel operators
                const gx = -getLuma(-1, -1) + getLuma(1, -1) +
                    -2 * getLuma(-1, 0) + 2 * getLuma(1, 0) +
                    -getLuma(-1, 1) + getLuma(1, 1);

                const gy = -getLuma(-1, -1) - 2 * getLuma(0, -1) - getLuma(1, -1) +
                    getLuma(-1, 1) + 2 * getLuma(0, 1) + getLuma(1, 1);

                const magnitude = Math.sqrt(gx * gx + gy * gy);
                edges[y * width + x] = magnitude > threshold ? 1 : 0;
            }
        }

        // Apply peaks as colored overlay
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (edges[y * width + x]) {
                    const idx = (y * width + x) * 4;
                    // Red highlight for sharp edges
                    data[idx] = 255;
                    data[idx + 1] = Math.floor(data[idx + 1] * 0.3);
                    data[idx + 2] = Math.floor(data[idx + 2] * 0.3);
                }
            }
        }
    }

    getFalseColor(l) {
        if (l < 0.01) return { r: 0.1, g: 0, b: 0.3 };
        if (l < 0.08) return { r: 0, g: 0, b: 0.8 };
        if (l < 0.20) return { r: 0.3, g: 0.3, b: 0.3 };
        if (l < 0.40) return { r: 0, g: 0.7, b: 0 };
        if (l < 0.60) return { r: 0.8, g: 0.8, b: 0 };
        if (l < 0.80) return { r: 1, g: 0.5, b: 0 };
        if (l < 0.95) return { r: 1, g: 0, b: 0 };
        return { r: 1, g: 0, b: 1 };
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RUN & PROMPT
    // ═══════════════════════════════════════════════════════════════════════════

    runWorkflow() {
        if (this.isQueueing) return;
        this.isQueueing = true;
        this.runButton.textContent = '⏳';

        // Use ComfyUI's queue prompt function
        app.queuePrompt(0).then(() => {
            this.isQueueing = false;
            this.runButton.textContent = '▶';
        }).catch(() => {
            this.isQueueing = false;
            this.runButton.textContent = '❌';
            setTimeout(() => this.runButton.textContent = '▶', 1000);
        });
    }

    togglePromptPanel() {
        this.showPromptPanel = !this.showPromptPanel;

        if (this.showPromptPanel) {
            this.promptButton.classList.add('active');
            this.promptButton.style.background = this.theme.accent;
            this.promptButton.style.color = '#fff';
            this.createPromptPanel();
        } else {
            this.promptButton.classList.remove('active');
            this.promptButton.style.background = '';
            this.promptButton.style.color = this.theme.textDim;
            if (this.promptPanel) this.promptPanel.remove();
            this.promptPanel = null;
        }
    }

    createPromptPanel() {
        if (this.promptPanel) this.promptPanel.remove();

        const t = this.theme;
        this.promptPanel = document.createElement('div');
        this.promptPanel.style.cssText = `
            position: absolute;
            top: 40px;
            right: 10px;
            width: 300px;
            max-height: 80%;
            background: rgba(16, 16, 24, 0.95);
            border: 1px solid ${t.panelBorder};
            border-radius: 4px;
            padding: 8px;
            overflow-y: auto;
            z-index: 100;
            display: flex;
            flex-direction: column;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        `;

        // Heuristic: Scan for ANY node that looks like it has a text/prompt widget
        const nodes = app.graph._nodes.filter(n => {
            if (!n.widgets) return false;

            // Check specific types first
            if (n.type && (n.type.includes("CLIPTextEncode") || n.type.includes("Prompt"))) return true;

            // Heuristic checking of widgets
            const hasTextWidget = n.widgets.some(w => {
                // Check widget names likely to be prompts
                const name = (w.name || "").toLowerCase();
                // check if value is string and not a small config string (like "enable")
                const isString = typeof w.value === "string";

                return isString && (
                    name === "text" ||
                    name === "string" ||
                    name.includes("prompt") ||
                    w.type === "customtext"
                );
            });

            return hasTextWidget;
        });

        // Sort by vertical position
        nodes.sort((a, b) => a.pos[1] - b.pos[1]);

        if (nodes.length === 0) {
            const msg = document.createElement('div');
            msg.textContent = "No prompt nodes found.";
            msg.style.color = t.textDim;
            msg.style.fontSize = "11px";
            this.promptPanel.appendChild(msg);
        } else {
            nodes.forEach(node => {
                const wrapper = document.createElement('div');
                wrapper.style.display = 'flex';
                wrapper.style.flexDirection = 'column';
                wrapper.style.gap = '4px';

                const label = document.createElement('div');
                label.textContent = node.title || node.type;
                label.style.cssText = `color: ${t.accent}; font-size: 11px; font-weight: bold; cursor: pointer;`;
                label.title = "Jump to node";
                label.onclick = () => {
                    app.canvas.centerOnNode(node);
                    app.canvas.selectNode(node);
                };

                wrapper.appendChild(label);

                // Render ALL widgets
                if (node.widgets) {
                    node.widgets.forEach(w => {
                        // Skip converted or hidden widgets
                        if (w.type === 'converted-widget' || w.name === '_temp') return;

                        const wContainer = document.createElement('div');
                        wContainer.style.cssText = 'display: flex; flex-direction: column; gap: 2px; margin-bottom: 4px;';

                        // Label for parameters (skip for main prompt text to save space, or keep small?)
                        const isMainText = (w.type === 'customtext' || w.name === 'text');
                        if (!isMainText) {
                            const wl = document.createElement('div');
                            wl.textContent = w.name;
                            wl.style.cssText = `color: ${t.textDim}; font-size: 9px;`;
                            wContainer.appendChild(wl);
                        }

                        let input;

                        // 1. Text / String
                        if (w.type === 'customtext' || w.type === 'text' || (!w.type && typeof w.value === 'string')) {
                            input = document.createElement('textarea');
                            input.value = w.value;
                            input.style.cssText = `
                                width: 100%;
                                height: ${isMainText ? '60px' : '30px'};
                                background: #0a0a0f;
                                border: 1px solid ${t.panelBorder};
                                color: ${t.text};
                                font-size: 11px;
                                padding: 4px;
                                resize: vertical;
                                font-family: inherit;
                            `;
                            input.addEventListener('input', (e) => {
                                w.value = e.target.value;
                                if (w.callback) w.callback(w.value);
                            });
                            // Focus helpers
                            input.addEventListener('focus', () => { app.canvas.selectNode(node); wrapper.style.borderLeft = `2px solid ${t.accent}`; });
                            input.addEventListener('blur', () => { wrapper.style.borderLeft = 'none'; });
                        }
                        // 2. Number
                        else if (w.type === 'number' || typeof w.value === 'number') {
                            input = document.createElement('input');
                            input.type = 'number';
                            input.value = w.value;
                            if (w.options) {
                                if (w.options.min !== undefined) input.min = w.options.min;
                                if (w.options.max !== undefined) input.max = w.options.max;
                                if (w.options.step !== undefined) input.step = w.options.step;
                            }
                            input.style.cssText = `
                                width: 100%;
                                background: #0a0a0f;
                                border: 1px solid ${t.panelBorder};
                                color: ${t.text};
                                font-size: 11px;
                                padding: 2px 4px;
                            `;
                            input.addEventListener('input', (e) => {
                                let val = parseFloat(e.target.value);
                                if (w.options) {
                                    if (w.options.min !== undefined) val = Math.max(w.options.min, val);
                                    if (w.options.max !== undefined) val = Math.min(w.options.max, val);
                                }
                                w.value = val;
                                if (w.callback) w.callback(w.value);
                            });
                        }
                        // 3. Combo
                        else if (w.type === 'combo') {
                            input = document.createElement('select');
                            input.style.cssText = `
                                width: 100%;
                                background: #0a0a0f;
                                border: 1px solid ${t.panelBorder};
                                color: ${t.text};
                                font-size: 11px;
                                padding: 2px;
                            `;
                            if (w.options && w.options.values) {
                                w.options.values.forEach(v => {
                                    const opt = document.createElement('option');
                                    opt.value = v;
                                    opt.textContent = v;
                                    input.appendChild(opt);
                                });
                            }
                            input.value = w.value;
                            input.addEventListener('change', (e) => {
                                w.value = e.target.value;
                                if (w.callback) w.callback(w.value);
                            });
                        }
                        // 4. Toggle / Boolean
                        else if (w.type === 'toggle' || typeof w.value === 'boolean') {
                            const row = document.createElement('div');
                            row.style.cssText = 'display: flex; align-items: center; gap: 6px;';
                            input = document.createElement('input');
                            input.type = 'checkbox';
                            input.checked = w.value;
                            input.addEventListener('change', (e) => {
                                w.value = e.target.checked;
                                if (w.callback) w.callback(w.value);
                            });
                            row.appendChild(input);

                            // Move label here for checkboxes
                            if (wContainer.firstChild) wContainer.firstChild.remove(); // Remove top label
                            const lbl = document.createElement('span');
                            lbl.textContent = w.name;
                            lbl.style.cssText = `color: ${t.textDim}; font-size: 11px;`;
                            row.appendChild(lbl);

                            wContainer.appendChild(row);
                            input = null; // Handled wrappers
                        }

                        if (input) wContainer.appendChild(input);
                        wrapper.appendChild(wContainer);
                    });
                }

                this.promptPanel.appendChild(wrapper);
            });

            // Add Run Button Inside Panel
            const runBtnPanel = document.createElement('button');
            runBtnPanel.textContent = 'Apply & Queue (Shift+Enter)';
            runBtnPanel.style.textTransform = 'uppercase';
            runBtnPanel.style.cssText = `
                margin-top: 8px;
                padding: 6px;
                background: ${t.accent};
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                font-weight: bold;
            `;
            runBtnPanel.onclick = () => this.runWorkflow();
            this.promptPanel.appendChild(runBtnPanel);
        }

        this.container.appendChild(this.promptPanel);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          PROGRESS & STATUS
    // ═══════════════════════════════════════════════════════════════════════════

    setupProgressUI() {
        const t = this.theme;
        this.progressContainer = document.createElement('div');
        this.progressContainer.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: rgba(0,0,0,0.5);
            z-index: 50;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        this.progressBar = document.createElement('div');
        this.progressBar.style.cssText = `
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, ${t.accent}, #4f4);
            transition: width 0.1s linear;
            box-shadow: 0 0 10px ${t.accent};
        `;

        this.progressText = document.createElement('div');
        this.progressText.style.cssText = `
            position: absolute;
            bottom: 6px;
            right: 10px;
            font-size: 10px;
            font-family: monospace;
            color: rgba(255,255,255,0.8);
            text-shadow: 0 1px 2px black;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
            background: rgba(0,0,0,0.6);
            padding: 2px 6px;
            border-radius: 4px;
        `;

        this.progressContainer.appendChild(this.progressBar);
        this.container.appendChild(this.progressContainer);
        this.container.appendChild(this.progressText);

        // API Events
        api.addEventListener("execution_start", () => {
            this.progressStart = Date.now();
            this.progressHistory = [];
            this.showProgress(true);
        });

        api.addEventListener("progress", ({ detail }) => {
            const { value, max } = detail;
            const pct = (value / max) * 100;
            this.progressBar.style.width = `${pct}%`;

            // ETA Calculation
            const now = Date.now();
            if (value > 0) {
                const elapsed = (now - this.progressStart) / 1000;
                const timePerStep = elapsed / value;
                const remaining = (max - value) * timePerStep;

                // Simple formatting
                const eta = remaining < 60 ? `${remaining.toFixed(1)}s` : `${Math.floor(remaining / 60)}m ${Math.floor(remaining % 60)}s`;
                this.progressText.textContent = `Step ${value}/${max} | ETA: ${eta}`;
            }
        });

        api.addEventListener("executed", ({ detail }) => {
            // Hide progress eventually if queue empty, but ComfyUI usually handles global progress
        });

        api.addEventListener("status", ({ detail }) => {
            if (!detail || detail.exec_info.queue_remaining === 0) {
                this.showProgress(false);
            }
        });
    }

    showProgress(show) {
        this.progressContainer.style.opacity = show ? 1 : 0;
        this.progressText.style.opacity = show ? 1 : 0;
        if (!show) {
            setTimeout(() => {
                this.progressBar.style.width = '0%';
                this.progressText.textContent = '';
            }, 300); // Wait for fade out
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          COLOR TRANSFORMS
    // ═══════════════════════════════════════════════════════════════════════════

    applyColorTransform(r, g, b, mode) {
        // Helper: Linear to sRGB
        const lin2srgb = (c) => c > 0.0031308 ? 1.055 * Math.pow(c, 1 / 2.4) - 0.055 : 12.92 * c;
        // Helper: Linear to Rec.709
        const lin2rec709 = (c) => c < 0.018 ? 4.5 * c : 1.099 * Math.pow(c, 0.45) - 0.099;

        switch (mode) {
            case 'sRGB':
                // Assume Input is Linear, Output sRGB
                return [lin2srgb(r), lin2srgb(g), lin2srgb(b)];
            case 'Rec.709':
                // Assume Input is Linear, Output Rec.709
                return [lin2rec709(r), lin2rec709(g), lin2rec709(b)];
            case 'LogC3':
                // ARRI LogC3 to Rec709 LUT approximation
                r = (r > 0.1496582 ? (Math.pow(10.0, (r - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (r / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                g = (g > 0.1496582 ? (Math.pow(10.0, (g - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (g / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                b = (b > 0.1496582 ? (Math.pow(10.0, (b - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272) : (b / 0.1496582) * (Math.pow(10.0, (0.1496582 - 0.385537) / 0.2471896) - 0.052272) / (1.0 - 0.052272));
                return [lin2rec709(Math.max(0, r)), lin2rec709(Math.max(0, g)), lin2rec709(Math.max(0, b))];
            case 'ACEScg':
                // ACEScg -> Rec.709 (Simple Tonemap)
                // Matrix (AP1 -> Rec.709)
                let rr = r * 1.70485868 - g * 0.62171602 - b * 0.08329937;
                let gg = -r * 0.19644612 + g * 1.26432540 + b * 0.03212072;
                let bb = -r * 0.01776686 - g * 0.00403754 + b * 1.02179971;
                // Simple Tonemap
                rr = rr / (rr + 1); gg = gg / (gg + 1); bb = bb / (bb + 1);
                return [lin2rec709(Math.max(0, rr)), lin2rec709(Math.max(0, gg)), lin2rec709(Math.max(0, bb))];
            default:
                return [r, g, b];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                          NODE REGISTRATION
// ═══════════════════════════════════════════════════════════════════════════════

app.registerExtension({
    name: "FXTD.RadianceViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FXTD_RadianceViewer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const container = document.createElement('div');
            container.id = `radiance-viewer-${this.id}`;
            this.addDOMWidget("viewer", "viewer", container, { serialize: false, hideOnZoom: false });

            // Force container properties to ensure it expands
            container.style.display = 'flex';
            container.style.width = '100%';
            container.style.height = '100%';

            this.radianceViewer = new RadianceViewer(this, container);
            this.size = [900, 600];
            this.outputs = [];
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            onResize?.apply(this, arguments);
            if (this.radianceViewer) {
                // Debounce resize to prevent thrashing
                if (this._resizeTimer) clearTimeout(this._resizeTimer);
                this._resizeTimer = setTimeout(() => {
                    this.radianceViewer.resize();
                }, 20);
            }
        };

        nodeType.prototype.onExecuted = function (message) {
            if (!message?.radiance_images?.length) return;

            const viewer = this.radianceViewer;
            if (!viewer) return;

            // Separate main images from compare images
            const mainImages = message.radiance_images.filter(img => !img.is_compare);
            const compareImages = message.radiance_images.filter(img => img.is_compare);

            // Reset frame arrays
            viewer.frameImages = [];
            viewer.frameCompareImages = [];
            viewer.totalFrames = mainImages.length;
            viewer.currentFrame = 0;

            // Load all main frames
            let loadedCount = 0;
            mainImages.forEach((imgData, idx) => {
                const img = new Image();
                img.crossOrigin = 'anonymous';

                img.onload = () => {
                    viewer.frameImages[idx] = img;
                    loadedCount++;

                    // Set first image immediately
                    if (idx === 0) {
                        viewer.setImage(img);
                    }

                    // Update frame display when all loaded
                    if (loadedCount === mainImages.length) {
                        viewer.updateFrameDisplay();
                    }
                };

                img.onerror = (e) => {
                    console.error("[Radiance Viewer] Failed to load image:", imgData.filename, e);
                    if (idx === 0) {
                        const ctx = viewer.ctx;
                        ctx.fillStyle = viewer.theme.textDim;
                        ctx.font = '12px sans-serif';
                        ctx.textAlign = 'center';
                        ctx.fillText('Error loading image', viewer.canvas.width / 2, viewer.canvas.height / 2);
                    }
                };

                img.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`);
            });

            // Load compare images
            compareImages.forEach((imgData, idx) => {
                const cmp = new Image();
                cmp.crossOrigin = 'anonymous';
                cmp.onload = () => {
                    viewer.frameCompareImages[idx] = cmp;
                    // Set first compare image
                    if (idx === 0) {
                        viewer.setCompareImage(cmp);
                    }
                };
                // Silent fail for compare images or optional log
                cmp.onerror = (e) => console.warn("[Radiance Viewer] Failed to load compare image:", imgData.filename);

                cmp.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || '')}&type=${imgData.type || 'temp'}`);
            });
        };
    }
});
