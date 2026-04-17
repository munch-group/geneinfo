"""
segment_viewer_gl.py — WebGL2 genomic segment viewer (anywidget)
================================================================

pip install anywidget traitlets numpy pandas

Track types
-----------
  segment   — instanced GPU rects; auto-LOD to density histogram
  heatmap   — R8 texture; handles 10 k+ individuals at 60 fps
  gene      — exon blocks + intron connectors (2D canvas)
  scatter   — point cloud (GL POINTS)
  line      — connected line (GL LINE_STRIP)
  fill      — fill-between two curves, pos/neg coloring (GL TRIANGLE_STRIP)
  histogram — vertical bars (instanced rects)

Performance
-----------
  All pan/zoom is uniform-only (zero CPU/GPU buffer work per frame).
  Segment culling uses sorted attribute-pointer offsets.

API
---
    viewer = SegmentViewer(chrom_sizes)
    viewer.add_segment_track(df, 'Segments', group_by='pop', individual_col='ind')
    viewer.add_heatmap_track(df, 'Haplotypes', individual_col='sample', group_col='pop')
    viewer.add_gene_track(genes_df, exons_df, name='Genes')
    viewer.add_scatter_track(df, 'Fst', x='pos', y='fst', group_by='pop')
    viewer.add_line_track(df, 'Rate', x='pos', y='rate')
    viewer.add_fill_track(df, 'CI', x='pos', y_lo='lo', y_hi='hi')
    viewer.add_histogram_track(df, 'Depth', x='pos', y='depth')
    viewer.zoom_to('chr1', 50_000_000, window=10_000_000)
    viewer
"""

from __future__ import annotations
import base64
from typing import Dict, List, Optional, Tuple, Union
import anywidget
import numpy as np
import pandas as pd
import traitlets

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
.sv-root {
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
    font-size: 12px;
    border: 1px solid var(--sv-border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--sv-bg);
    color: var(--sv-fg);
    user-select: none;
    -webkit-user-select: none;
}
.sv-toolbar {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    background: var(--sv-panel);
    border-bottom: 1px solid var(--sv-border);
    flex-wrap: wrap;
    min-height: 32px;
}
.sv-label-sm { font-size: 10px; color: var(--sv-muted); letter-spacing: 0.08em; text-transform: uppercase; }
.sv-toolbar select, .sv-toolbar input {
    padding: 3px 7px;
    border: 1px solid var(--sv-input-border);
    border-radius: 3px;
    font-size: 11px;
    font-family: inherit;
    background: var(--sv-input-bg);
    color: var(--sv-input-fg);
    outline: none;
    transition: border-color 0.12s;
}
.sv-toolbar input { width: 210px; }
.sv-toolbar select:focus, .sv-toolbar input:focus { border-color: var(--sv-accent); }
.sv-btn {
    padding: 2px 9px;
    border: 1px solid var(--sv-input-border);
    border-radius: 3px;
    cursor: pointer;
    background: var(--sv-input-bg);
    color: var(--sv-input-fg);
    font-family: inherit;
    font-size: 14px;
    line-height: 1.3;
    transition: background 0.1s, border-color 0.1s;
}
.sv-btn:hover { background: var(--sv-panel); border-color: var(--sv-accent); }
.sv-sep { flex: 1; }
.sv-lod-badge {
    font-size: 9px;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--sv-bg);
    border: 1px solid var(--sv-input-border);
    color: var(--sv-muted);
    letter-spacing: 0.06em;
}
.sv-wrap {
    position: relative;
    overflow: hidden;
    background: var(--sv-bg);
}
.sv-glcanvas { position: absolute; top: 0; left: 0; display: block; cursor: crosshair; }
.sv-glcanvas.dragging { cursor: crosshair; }
.sv-overlay { position: absolute; top: 0; left: 0; display: block; pointer-events: none; }
.sv-tooltip {
    position: absolute;
    background: var(--sv-input-bg);
    color: var(--sv-fg);
    padding: 4px 9px;
    border-radius: 4px;
    font-size: 10px;
    pointer-events: none;
    display: none;
    z-index: 20;
    white-space: pre;
    border: 1px solid var(--sv-input-border);
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# JavaScript ESM module
# ─────────────────────────────────────────────────────────────────────────────
_JS = r"""
export function render({ model, el }) {

// ══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════
const LABEL_W    = 96;    // left label column, CSS px
const SCALEBAR_H = 26;    // top ruler height, CSS px
const PAD_V      = 3;     // vertical padding inside track, CSS px
const DENS_THRESH = 1200; // bp/px above which we switch to density LOD
const INST_STRIDE = 28;   // 7 float32s per instance: start,end,yLo,yHi,r,g,b
const DENS_STRIDE = 8;    // 2 float32s per vertex: x, h

// NDC helpers (CSS y → WebGL NDC y, origin top-left vs bottom-left)
const cssNDC = (cssY, H) => 1.0 - 2.0 * cssY / H;

// ══════════════════════════════════════════════════════════════════════════════
// DOM
// ══════════════════════════════════════════════════════════════════════════════
el.className = 'sv-root';
el.innerHTML = `
  <div class="sv-toolbar">
    <span class="sv-label-sm">chr</span>
    <select class="sv-chrom"></select>
    <input  class="sv-pos" placeholder="chr1:0–248956422" />
    <button class="sv-btn sv-zi" title="Zoom in (+)">＋</button>
    <button class="sv-btn sv-zo" title="Zoom out (−)">－</button>
    <button class="sv-btn sv-rs" title="Reset view">⌂</button>
    <button class="sv-btn sv-hmr" title="Recompute heatmap(s) for current view" style="display:none">⟲</button>
    <button class="sv-btn sv-hmg" title="Restore global heatmap view" style="display:none">◱</button>
    <div class="sv-sep"></div>
    <span class="sv-lod-badge sv-lod">▬ segments</span>
  </div>
  <div class="sv-wrap">
    <canvas class="sv-glcanvas"></canvas>
    <canvas class="sv-overlay"></canvas>
    <div class="sv-tooltip"></div>
  </div>
`;

const chromSel  = el.querySelector('.sv-chrom');
const posInput  = el.querySelector('.sv-pos');
const zoomInBtn = el.querySelector('.sv-zi');
const zoomOutBtn= el.querySelector('.sv-zo');
const resetBtn  = el.querySelector('.sv-rs');
const hmRecBtn  = el.querySelector('.sv-hmr');
const hmGlobBtn = el.querySelector('.sv-hmg');
const lodBadge  = el.querySelector('.sv-lod');
const wrap      = el.querySelector('.sv-wrap');
const glCanvas  = el.querySelector('.sv-glcanvas');
const ov        = el.querySelector('.sv-overlay');
const tooltip   = el.querySelector('.sv-tooltip');
const octx      = ov.getContext('2d');

// ══════════════════════════════════════════════════════════════════════════════
// THEME
// ══════════════════════════════════════════════════════════════════════════════
let th = {};

function applyTheme() {
  const t = model.get('theme') || {};
  th = { ...t };
  const root = el.style;
  root.setProperty('--sv-bg',           t.bg           || '#13131a');
  root.setProperty('--sv-fg',           t.fg           || '#d0d0e8');
  root.setProperty('--sv-panel',        t.panel        || '#1c1c26');
  root.setProperty('--sv-border',       t.border       || '#252530');
  root.setProperty('--sv-input-bg',     t.input_bg     || '#0d0d14');
  root.setProperty('--sv-input-fg',     t.input_fg     || '#c0c0dc');
  root.setProperty('--sv-input-border', t.input_border || '#33334a');
  root.setProperty('--sv-accent',       t.accent       || '#4466ee');
  root.setProperty('--sv-muted',        t.muted        || '#666688');
  root.setProperty('--sv-label',        t.label        || '#9090c0');
}
applyTheme();

// ══════════════════════════════════════════════════════════════════════════════
// WebGL2 SETUP
// ══════════════════════════════════════════════════════════════════════════════
const gl = glCanvas.getContext('webgl2', { antialias: false, alpha: false });
if (!gl) {
  wrap.innerHTML = '<div style="padding:20px;color:#f66">WebGL2 not available in this environment.</div>';
  return;
}
const dpr = Math.min(window.devicePixelRatio || 1, 2);

// ─── Shader sources ────────────────────────────────────────────────────────
const VS_RECT = `#version 300 es
precision highp float;
in vec2  aCorner;
in float iStart, iEnd, iYLo, iYHi;
in vec3  iColor;
uniform float uVS, uVE, uTT, uTB, uXL;
flat out vec3 vColor;
void main() {
  float gx = mix(iStart, iEnd, aCorner.x);
  float t  = (gx - uVS) / (uVE - uVS);
  float xN = clamp(mix(uXL, 1.0, t), -1.0, 1.0);
  float yN  = mix(uTT, uTB, mix(iYLo, iYHi, aCorner.y));
  gl_Position = vec4(xN, yN, 0.0, 1.0);
  vColor = iColor;
}`;

const FS_RECT = `#version 300 es
precision mediump float;
flat in vec3 vColor;
out vec4 fragColor;
void main() { fragColor = vec4(vColor, 1.0); }`;

const VS_DENS = `#version 300 es
precision highp float;
in float aX, aH;
uniform float uVS, uVE, uTT, uTB, uPointSize, uXL;
void main() {
  float t  = (aX - uVS) / (uVE - uVS);
  float xN = mix(uXL, 1.0, t);
  float yN = mix(uTB, uTT, aH);
  gl_Position = vec4(xN, yN, 0.0, 1.0);
  gl_PointSize = uPointSize;
}`;

const FS_DENS = `#version 300 es
precision mediump float;
uniform vec3 uColor;
uniform float uAlpha;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, uAlpha); }`;

const VS_TEX = `#version 300 es
precision highp float;
in vec2 aPos, aTex;
out vec2 vTex;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); vTex = aTex; }`;

const FS_TEX = `#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform vec3 uBg, uFg;
in vec2 vTex;
out vec4 fragColor;
void main() {
  float v = texture(uTex, vTex).r;
  fragColor = vec4(mix(uBg, uFg, v), 1.0);
}`;

// ─── Compile / link helpers ────────────────────────────────────────────────
function compileShader(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}

function makeProgram(vs, fs) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compileShader(gl.VERTEX_SHADER,   vs));
  gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

// ─── Programs ─────────────────────────────────────────────────────────────
const rectProg = makeProgram(VS_RECT, FS_RECT);
const densProg = makeProgram(VS_DENS, FS_DENS);
const texProg  = makeProgram(VS_TEX,  FS_TEX);

// rect program locations
const rLoc = {
  aCorner: gl.getAttribLocation (rectProg, 'aCorner'),
  iStart:  gl.getAttribLocation (rectProg, 'iStart'),
  iEnd:    gl.getAttribLocation (rectProg, 'iEnd'),
  iYLo:    gl.getAttribLocation (rectProg, 'iYLo'),
  iYHi:    gl.getAttribLocation (rectProg, 'iYHi'),
  iColor:  gl.getAttribLocation (rectProg, 'iColor'),
  uVS:     gl.getUniformLocation(rectProg, 'uVS'),
  uVE:     gl.getUniformLocation(rectProg, 'uVE'),
  uTT:     gl.getUniformLocation(rectProg, 'uTT'),
  uTB:     gl.getUniformLocation(rectProg, 'uTB'),
  uXL:     gl.getUniformLocation(rectProg, 'uXL'),
};

// density/line/scatter program locations
const dLoc = {
  aX:         gl.getAttribLocation (densProg, 'aX'),
  aH:         gl.getAttribLocation (densProg, 'aH'),
  uVS:        gl.getUniformLocation(densProg, 'uVS'),
  uVE:        gl.getUniformLocation(densProg, 'uVE'),
  uTT:        gl.getUniformLocation(densProg, 'uTT'),
  uTB:        gl.getUniformLocation(densProg, 'uTB'),
  uColor:     gl.getUniformLocation(densProg, 'uColor'),
  uAlpha:     gl.getUniformLocation(densProg, 'uAlpha'),
  uPointSize: gl.getUniformLocation(densProg, 'uPointSize'),
  uXL:        gl.getUniformLocation(densProg, 'uXL'),
};

// texture program locations
const tLoc = {
  aPos: gl.getAttribLocation (texProg, 'aPos'),
  aTex: gl.getAttribLocation (texProg, 'aTex'),
  uTex: gl.getUniformLocation(texProg, 'uTex'),
  uBg:  gl.getUniformLocation(texProg, 'uBg'),
  uFg:  gl.getUniformLocation(texProg, 'uFg'),
};

// ─── Static unit quad buffer (shared) ─────────────────────────────────────
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  0,0, 1,0, 1,1,
  0,0, 1,1, 0,1,
]), gl.STATIC_DRAW);

// Scratch buffer for heatmap quads (reused each draw)
const hmQuadBuf = gl.createBuffer();

// ─── GPU data stores: [tid][chrom][gid] ───────────────────────────────────
const gpuSeg  = {};   // { buf, starts, maxLen, count }
const gpuDens = {};   // [{ nBins, binWidth, buf, count }, ...]
const gpuHM   = {};   // { tex, nInd, nWin }
const gpuXY   = {};   // { buf, count }  — for scatter, line
const gpuFill = {};   // { posBuf, posCount, negBuf, negCount }
const gpuHist = {};   // { buf, starts, maxLen, count }  — instanced rects
const rawXY   = {};   // tooltip: [tid][chrom][gid] = Float32Array [x,y,...]
const rawFill = {};   // tooltip: [tid][chrom][gid] = Float32Array [x,lo,hi,...]
const rawHist = {};   // tooltip: [tid][chrom][gid] = { data: Float32Array, binWidth }

// Per-chromosome pan/zoom clamp range. Defaults to the whole chromosome;
// tightens to a recomputed heatmap's [xStart, xEnd] while any heatmap track
// on that chrom is in "view" mode.
const panRange = {};  // { [chrom]: { lo, hi } }
function panLo(ch) { return panRange[ch]?.lo ?? 0; }
function panHi(ch) { return panRange[ch]?.hi ?? (chromSizes[ch] || 1); }

// ══════════════════════════════════════════════════════════════════════════════
// BINARY DECODE
// ══════════════════════════════════════════════════════════════════════════════
function b64F32(b64) {
  if (!b64) return new Float32Array(0);
  const bin = atob(b64);
  const u8  = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

function b64U8(b64) {
  if (!b64) return new Uint8Array(0);
  const bin = atob(b64);
  const u8  = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

function hexRGB(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

// ══════════════════════════════════════════════════════════════════════════════
// GPU BUFFER BUILDERS
// ══════════════════════════════════════════════════════════════════════════════

// Build instanced rect buffer: 7 floats/instance [start,end,yLo,yHi,r,g,b]
// segs: Float32Array — [s, e, ...] (stride 2) or [s, e, ind, ...] (stride 3)
function buildSegGPU(segs, yLo, yHi, color, nInd) {
  const stride = nInd > 0 ? 3 : 2;
  const n = (segs.length / stride) | 0;
  if (n === 0) return null;
  const [r, g, b] = hexRGB(color);
  const inst   = new Float32Array(n * 7);
  const starts = new Float32Array(n);
  let maxLen = 0;
  const bandH = yHi - yLo;
  const rowH  = nInd > 0 ? bandH / nInd : bandH;
  for (let i = 0; i < n; i++) {
    const s   = segs[i * stride];
    const e   = segs[i * stride + 1];
    const ind = nInd > 0 ? segs[i * stride + 2] : 0;
    starts[i] = s;
    maxLen = Math.max(maxLen, e - s);
    const iy0 = yLo + ind * rowH;
    const iy1 = iy0 + rowH;
    const o = i * 7;
    inst[o]   = s; inst[o+1] = e;
    inst[o+2] = iy0; inst[o+3] = iy1;
    inst[o+4] = r; inst[o+5] = g; inst[o+6] = b;
  }
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, inst, gl.STATIC_DRAW);
  return { buf, starts, maxLen, count: n };
}

// Build density step-function buffer: flat-topped rectangles per bin
function buildDensGPU(counts, chromSz) {
  const n = counts.length;
  if (n === 0) return null;
  const winSz = chromSz / n;
  let maxVal = 0;
  for (let i = 0; i < n; i++) maxVal = Math.max(maxVal, counts[i]);
  if (maxVal === 0) maxVal = 1;
  const verts = new Float32Array(n * 4 * 2);
  let vi = 0;
  for (let i = 0; i < n; i++) {
    const xL = i * winSz;
    const xR = (i + 1) * winSz;
    const h  = counts[i] / maxVal;
    verts[vi++] = xL; verts[vi++] = 0;
    verts[vi++] = xL; verts[vi++] = h;
    verts[vi++] = xR; verts[vi++] = 0;
    verts[vi++] = xR; verts[vi++] = h;
  }
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  return { buf, count: n * 4 };
}

// Build heatmap texture
function buildHeatmapGPU(u8, nInd, nWin, xStart, xEnd) {
  if (!u8 || u8.length === 0) return null;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, nWin, nInd, 0, gl.RED, gl.UNSIGNED_BYTE, u8);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { tex, nInd, nWin, xStart, xEnd };
}

// Build XY buffer for scatter / line: normalize y to [0,1]
function buildXYGPU(data, yMin, yMax) {
  const n = (data.length / 2) | 0;
  if (n === 0) return null;
  const range = yMax - yMin || 1;
  const verts = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    verts[i * 2]     = data[i * 2];      // x (genomic)
    verts[i * 2 + 1] = (data[i * 2 + 1] - yMin) / range;  // h normalized
  }
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  return { buf, count: n };
}

// Build fill-between buffers: split into positive (above baseline) and negative (below)
function buildFillGPU(data, yMin, yMax, baseline) {
  const n = (data.length / 3) | 0;
  if (n === 0) return null;
  const range = yMax - yMin || 1;
  const baseN = (baseline - yMin) / range;  // normalized baseline

  // Positive fill: triangle strip from max(yLo, baseline) to max(yHi, baseline)
  const posVerts = [];
  const negVerts = [];
  for (let i = 0; i < n; i++) {
    const x   = data[i * 3];
    const lo  = (data[i * 3 + 1] - yMin) / range;
    const hi  = (data[i * 3 + 2] - yMin) / range;

    // Positive region (above baseline)
    const pLo = Math.max(lo, baseN);
    const pHi = Math.max(hi, baseN);
    posVerts.push(x, pLo, x, pHi);

    // Negative region (below baseline)
    const nLo = Math.min(lo, baseN);
    const nHi = Math.min(hi, baseN);
    negVerts.push(x, nLo, x, nHi);
  }

  function makeBuf(arr) {
    const f = new Float32Array(arr);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, f, gl.STATIC_DRAW);
    return buf;
  }
  return {
    posBuf: makeBuf(posVerts), posCount: n * 2,
    negBuf: makeBuf(negVerts), negCount: n * 2,
  };
}

// Build histogram bars as instanced rects
function buildHistGPU(data, yMin, yMax, binWidth, color) {
  const n = (data.length / 2) | 0;
  if (n === 0) return null;
  const [r, g, b] = hexRGB(color);
  const range = yMax - yMin || 1;
  const inst   = new Float32Array(n * 7);
  const starts = new Float32Array(n);
  let maxLen = 0;
  const halfW = binWidth / 2;
  for (let i = 0; i < n; i++) {
    const x = data[i * 2];
    const y = data[i * 2 + 1];
    const s = x - halfW;
    const e = x + halfW;
    const baseN = (0 - yMin) / range;
    const hN    = (y - yMin) / range;
    // yLo = baseline (bottom), yHi = value (top) — in 0..1 track space
    const yLo = Math.min(baseN, hN);
    const yHi = Math.max(baseN, hN);
    starts[i] = s;
    maxLen = Math.max(maxLen, e - s);
    const o = i * 7;
    inst[o]   = s; inst[o+1] = e;
    inst[o+2] = yLo; inst[o+3] = yHi;
    inst[o+4] = r; inst[o+5] = g; inst[o+6] = b;
  }
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, inst, gl.STATIC_DRAW);
  return { buf, starts, maxLen, count: n };
}

// ══════════════════════════════════════════════════════════════════════════════
// DATA UPLOAD (from model → GPU)
// ══════════════════════════════════════════════════════════════════════════════
function uploadTrackData() {
  const td   = model.get('track_data') || {};
  const cfgs = model.get('track_configs');

  for (const cfg of cfgs) {
    const tid = cfg.id;
    const d   = td[tid];
    if (!d) continue;

    if (cfg.type === 'segment') {
      const raw   = d.segs || {};
      const dens  = d.dens || {};
      const nG    = cfg.groups.length;
      const gap   = 0.08;
      const weights = cfg.groups.map(g => g.nInd || 1);
      const totalW  = weights.reduce((a, b) => a + b, 0);
      const usable  = 1.0 - gap;
      const cumW    = [0];
      for (let i = 0; i < nG; i++) cumW.push(cumW[i] + weights[i]);
      if (!gpuSeg[tid])  gpuSeg[tid]  = {};
      if (!gpuDens[tid]) gpuDens[tid] = {};
      for (const ch of Object.keys(raw)) {
        gpuSeg[tid][ch]  = {};
        gpuDens[tid][ch] = {};
        const csz = chromSizes[ch] || 1;
        for (let gi = 0; gi < nG; gi++) {
          const grp = cfg.groups[gi];
          const gid = grp.id;
          const yLo = gap * 0.5 + usable * cumW[gi] / totalW;
          const yHi = gap * 0.5 + usable * cumW[gi + 1] / totalW;
          const segs = b64F32(raw[ch]?.[gid]);
          const nInd = grp.nInd || 0;
          gpuSeg[tid][ch][gid] = buildSegGPU(segs, yLo, yHi, grp.color, nInd);
          const densLevels = dens[ch]?.[gid] || {};
          const levels = [];
          for (const [nBins, b64] of Object.entries(densLevels)) {
            const cnts = b64F32(b64);
            if (!cnts || cnts.length === 0) continue;
            const gpu = buildDensGPU(cnts, csz);
            if (gpu) levels.push({ nBins: parseInt(nBins), binWidth: csz / parseInt(nBins), ...gpu });
          }
          levels.sort((a, b) => a.nBins - b.nBins);
          gpuDens[tid][ch][gid] = levels;
        }
      }
    } else if (cfg.type === 'heatmap') {
      if (!gpuHM[tid]) gpuHM[tid] = {};
      for (const ch of Object.keys(d)) {
        gpuHM[tid][ch] = {};
        for (const [gid, info] of Object.entries(d[ch])) {
          const u8 = b64U8(info.data);
          gpuHM[tid][ch][gid] = buildHeatmapGPU(
            u8, info.nInd, info.nWin,
            info.xStart ?? 0,
            info.xEnd   ?? (chromSizes[ch] || 1),
          );
        }
      }
    } else if (cfg.type === 'gene') {
      geneData[tid] = d;
    } else if (cfg.type === 'scatter' || cfg.type === 'line') {
      if (!gpuXY[tid]) gpuXY[tid] = {};
      if (!rawXY[tid]) rawXY[tid] = {};
      const yMin = cfg.yMin ?? 0;
      const yMax = cfg.yMax ?? 1;
      for (const ch of Object.keys(d)) {
        gpuXY[tid][ch] = {};
        rawXY[tid][ch] = {};
        for (const [gid, b64] of Object.entries(d[ch])) {
          const arr = b64F32(b64);
          rawXY[tid][ch][gid] = arr;
          gpuXY[tid][ch][gid] = buildXYGPU(arr, yMin, yMax);
        }
      }
    } else if (cfg.type === 'fill') {
      if (!gpuFill[tid]) gpuFill[tid] = {};
      if (!rawFill[tid]) rawFill[tid] = {};
      const yMin = cfg.yMin ?? 0;
      const yMax = cfg.yMax ?? 1;
      const baseline = cfg.baseline ?? 0;
      for (const ch of Object.keys(d)) {
        gpuFill[tid][ch] = {};
        rawFill[tid][ch] = {};
        for (const [gid, b64] of Object.entries(d[ch])) {
          const arr = b64F32(b64);
          rawFill[tid][ch][gid] = arr;
          gpuFill[tid][ch][gid] = buildFillGPU(arr, yMin, yMax, baseline);
        }
      }
    } else if (cfg.type === 'histogram') {
      if (!gpuHist[tid]) gpuHist[tid] = {};
      if (!rawHist[tid]) rawHist[tid] = {};
      const yMin = cfg.yMin ?? 0;
      const yMax = cfg.yMax ?? 1;
      const binWidth = cfg.binWidth ?? 1;
      for (const ch of Object.keys(d)) {
        gpuHist[tid][ch] = {};
        rawHist[tid][ch] = {};
        for (const [gid, b64] of Object.entries(d[ch])) {
          const arr = b64F32(b64);
          rawHist[tid][ch][gid] = { data: arr, binWidth };
          const color = cfg.groups.find(g => g.id === gid)?.color || '#4488cc';
          gpuHist[tid][ch][gid] = buildHistGPU(arr, yMin, yMax, binWidth, color);
        }
      }
    }
  }

  // Recompute per-chrom pan clamp as the intersection of all heatmap domains
  // on that chrom. If any heatmap is at full-chrom (or there are no heatmaps),
  // the clamp for that chrom is cleared.
  for (const ch of Object.keys(panRange)) delete panRange[ch];
  for (const cfg of cfgs) {
    if (cfg.type !== 'heatmap') continue;
    const d = td[cfg.id];
    if (!d) continue;
    for (const ch of Object.keys(d)) {
      const csz = chromSizes[ch] || 1;
      let lo = 0, hi = csz;
      for (const info of Object.values(d[ch])) {
        const s = info.xStart ?? 0;
        const e = info.xEnd   ?? csz;
        lo = Math.max(lo, s);
        hi = Math.min(hi, e);
      }
      if (lo > 0 || hi < csz) {
        const prev = panRange[ch];
        panRange[ch] = prev
          ? { lo: Math.max(prev.lo, lo), hi: Math.min(prev.hi, hi) }
          : { lo, hi };
      }
    }
  }

  // Clamp current viewport into the new allowed range.
  const lo = panLo(vp.chrom), hi = panHi(vp.chrom);
  let ns = vp.start, ne = vp.end;
  const range = ne - ns;
  if (range >= hi - lo) {
    ns = lo; ne = hi;
  } else {
    if (ns < lo) { ns = lo; ne = ns + range; }
    if (ne > hi) { ne = hi; ns = ne - range; }
  }
  if (ns !== vp.start || ne !== vp.end) {
    vp = { chrom: vp.chrom, start: ns, end: ne };
    syncVp();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// BINARY SEARCH
// ══════════════════════════════════════════════════════════════════════════════
function bisect(arr, val) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; arr[m] < val ? lo = m + 1 : hi = m; }
  return lo;
}

// ══════════════════════════════════════════════════════════════════════════════
// DRAW CALLS
// ══════════════════════════════════════════════════════════════════════════════

// Draw instanced rectangles (segment track + histogram bars)
let curXL = -1.0;  // NDC x at label edge, set per frame

function drawRects(gpuData, vs, ve, tt, tb) {
  if (!gpuData || gpuData.count === 0) return;
  const lo = Math.max(0, bisect(gpuData.starts, vs - gpuData.maxLen) - 1);
  const hi = Math.min(gpuData.count, bisect(gpuData.starts, ve) + 1);
  const count = hi - lo;
  if (count <= 0) return;
  const byteOff = lo * INST_STRIDE;
  gl.useProgram(rectProg);
  gl.uniform1f(rLoc.uVS, vs);
  gl.uniform1f(rLoc.uVE, ve);
  gl.uniform1f(rLoc.uTT, tt);
  gl.uniform1f(rLoc.uTB, tb);
  gl.uniform1f(rLoc.uXL, curXL);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.enableVertexAttribArray(rLoc.aCorner);
  gl.vertexAttribPointer(rLoc.aCorner, 2, gl.FLOAT, false, 8, 0);
  gl.vertexAttribDivisor(rLoc.aCorner, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, gpuData.buf);
  const S = INST_STRIDE;
  gl.enableVertexAttribArray(rLoc.iStart);
  gl.vertexAttribPointer(rLoc.iStart, 1, gl.FLOAT, false, S, byteOff + 0);
  gl.vertexAttribDivisor(rLoc.iStart, 1);
  gl.enableVertexAttribArray(rLoc.iEnd);
  gl.vertexAttribPointer(rLoc.iEnd, 1, gl.FLOAT, false, S, byteOff + 4);
  gl.vertexAttribDivisor(rLoc.iEnd, 1);
  gl.enableVertexAttribArray(rLoc.iYLo);
  gl.vertexAttribPointer(rLoc.iYLo, 1, gl.FLOAT, false, S, byteOff + 8);
  gl.vertexAttribDivisor(rLoc.iYLo, 1);
  gl.enableVertexAttribArray(rLoc.iYHi);
  gl.vertexAttribPointer(rLoc.iYHi, 1, gl.FLOAT, false, S, byteOff + 12);
  gl.vertexAttribDivisor(rLoc.iYHi, 1);
  gl.enableVertexAttribArray(rLoc.iColor);
  gl.vertexAttribPointer(rLoc.iColor, 3, gl.FLOAT, false, S, byteOff + 16);
  gl.vertexAttribDivisor(rLoc.iColor, 1);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
}

// Shared setup for densProg-based draws (density, line, scatter, fill)
function setupDensProg(vs, ve, tt, tb, color, alpha) {
  const [r, g, b] = hexRGB(color);
  gl.useProgram(densProg);
  gl.uniform1f(dLoc.uVS, vs);
  gl.uniform1f(dLoc.uVE, ve);
  gl.uniform1f(dLoc.uTT, tt);
  gl.uniform1f(dLoc.uTB, tb);
  gl.uniform3f(dLoc.uColor, r, g, b);
  gl.uniform1f(dLoc.uAlpha, alpha);
  gl.uniform1f(dLoc.uPointSize, 1.0);
  gl.uniform1f(dLoc.uXL, curXL);
}

function bindDensBuf(buf) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.enableVertexAttribArray(dLoc.aX);
  gl.vertexAttribPointer(dLoc.aX, 1, gl.FLOAT, false, DENS_STRIDE, 0);
  gl.vertexAttribDivisor(dLoc.aX, 0);
  gl.enableVertexAttribArray(dLoc.aH);
  gl.vertexAttribPointer(dLoc.aH, 1, gl.FLOAT, false, DENS_STRIDE, 4);
  gl.vertexAttribDivisor(dLoc.aH, 0);
}

// Draw density area (segment LOD)
function drawDensArea(gpuData, vs, ve, tt, tb, color, alpha) {
  if (!gpuData || gpuData.count === 0) return;
  setupDensProg(vs, ve, tt, tb, color, alpha);
  bindDensBuf(gpuData.buf);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, gpuData.count);
}

// Draw line strip
function drawLine(gpuData, vs, ve, tt, tb, color, alpha) {
  if (!gpuData || gpuData.count === 0) return;
  setupDensProg(vs, ve, tt, tb, color, alpha);
  bindDensBuf(gpuData.buf);
  gl.drawArrays(gl.LINE_STRIP, 0, gpuData.count);
}

// Draw scatter points
function drawPoints(gpuData, vs, ve, tt, tb, color, alpha, pointSize) {
  if (!gpuData || gpuData.count === 0) return;
  setupDensProg(vs, ve, tt, tb, color, alpha);
  gl.uniform1f(dLoc.uPointSize, pointSize * dpr);
  bindDensBuf(gpuData.buf);
  gl.drawArrays(gl.POINTS, 0, gpuData.count);
}

// Draw fill-between (pos/neg split)
function drawFill(gpuData, vs, ve, tt, tb, colorPos, colorNeg, alpha) {
  if (!gpuData) return;
  if (gpuData.posCount > 0) {
    setupDensProg(vs, ve, tt, tb, colorPos, alpha);
    bindDensBuf(gpuData.posBuf);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, gpuData.posCount);
  }
  if (gpuData.negCount > 0) {
    setupDensProg(vs, ve, tt, tb, colorNeg, alpha);
    bindDensBuf(gpuData.negBuf);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, gpuData.negCount);
  }
}

// Draw heatmap texture for one group band
function drawHeatmapBand(hmData, vs, ve, chromSz, tt, tb, color) {
  if (!hmData) return;
  const [r, g, b] = hexRGB(color);
  const dom0 = hmData.xStart ?? 0;
  const dom1 = hmData.xEnd   ?? chromSz;
  const span = (dom1 - dom0) || 1;
  const ts0 = (vs - dom0) / span;
  const ts1 = (ve - dom0) / span;
  const xL = curXL;
  const xR = 1.0;
  const q = new Float32Array([
    xL, tt,  ts0, 0,
    xR, tt,  ts1, 0,
    xL, tb,  ts0, 1,
    xR, tb,  ts1, 1,
  ]);
  gl.bindBuffer(gl.ARRAY_BUFFER, hmQuadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, q, gl.DYNAMIC_DRAW);
  gl.useProgram(texProg);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, hmData.tex);
  gl.uniform1i(tLoc.uTex, 0);
  const [bgR, bgG, bgB] = hexRGB(th.bg || '#13131a');
  gl.uniform3f(tLoc.uBg, bgR, bgG, bgB);
  gl.uniform3f(tLoc.uFg, r, g, b);
  const stride = 16;
  gl.enableVertexAttribArray(tLoc.aPos);
  gl.vertexAttribPointer(tLoc.aPos, 2, gl.FLOAT, false, stride, 0);
  gl.vertexAttribDivisor(tLoc.aPos, 0);
  gl.enableVertexAttribArray(tLoc.aTex);
  gl.vertexAttribPointer(tLoc.aTex, 2, gl.FLOAT, false, stride, 8);
  gl.vertexAttribDivisor(tLoc.aTex, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// ══════════════════════════════════════════════════════════════════════════════
// OVERLAY: scale bar + track labels (2D canvas)
// ══════════════════════════════════════════════════════════════════════════════

function fmtBp(bp, step) {
  // `step` is the tick spacing in bp; use it to pick a precision that makes
  // consecutive tick labels distinguishable. Falls back to a value-based rule
  // when no step is supplied.
  if (step != null && step > 0) {
    const pick = (unit) => {
      const d = Math.max(0, Math.ceil(-Math.log10(step / unit)));
      return (bp / unit).toFixed(d);
    };
    if (Math.abs(bp) >= 1e6 || step >= 1e6) return pick(1e6) + ' Mb';
    if (Math.abs(bp) >= 1e3 || step >= 1e3) return pick(1e3) + ' kb';
    return Math.round(bp) + ' bp';
  }
  if (bp >= 1e6) return (bp / 1e6).toFixed(bp >= 10e6 ? 1 : 2) + ' Mb';
  if (bp >= 1e3) return (bp / 1e3).toFixed(bp >= 10e3 ? 1 : 2) + ' kb';
  return Math.round(bp) + ' bp';
}

function fmtY(v) {
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'k';
  if (Number.isInteger(v)) return v.toString();
  return v.toPrecision(3);
}

// ── Gene track: full 2D rendering (backbone, arrows, exons, names) ─────────
function drawGeneTrack2D(cfg, trackY, vs, ve, W_css) {
  const drawW  = W_css - LABEL_W;
  const range  = ve - vs;
  const entry  = geneData?.[cfg.id]?.[vp.chrom];
  const genes  = entry?.records ?? (Array.isArray(entry) ? entry : []);
  const color  = cfg.color || '#4488cc';

  octx.fillStyle = th.bg || '#13131a';
  octx.fillRect(LABEL_W, trackY, drawW, cfg.height);
  if (!genes.length) return;

  // Pack rows tightly at the top of the track at a fixed 28 px per row.
  // The number of rows used for this chrom may be less than cfg.rows (the
  // global max across all chroms), so any unused space is simply left empty
  // at the bottom rather than stretched into the gene layout.
  const nRowsChrom = Math.max(1, entry?.rows ?? cfg.rows ?? 1);
  const rowH       = Math.min(28, cfg.height / nRowsChrom);
  const exonH  = Math.min(14, Math.max(6, rowH * 0.45));
  const labelBand = 9;  // space reserved above each exon for the label
  const arrowSep = 70;
  const arrowA   = 4;

  octx.save();
  octx.beginPath();
  octx.rect(LABEL_W, trackY, drawW, cfg.height);
  octx.clip();

  octx.font = '8px monospace';

  // Build a list of just the visible genes with their pixel extents.
  const vis = [];
  for (const gene of genes) {
    if (gene.e < vs || gene.s > ve) continue;
    const gx0 = LABEL_W + Math.max(0,     (gene.s - vs) / range * drawW);
    const gx1 = LABEL_W + Math.min(drawW, (gene.e - vs) / range * drawW);
    const gw  = gx1 - gx0;
    if (gw < 0.5) continue;
    vis.push({ gene, gx0, gx1, gw, rowIdx: gene.row | 0 });
  }

  // Feasibility pass: tentatively place every visible gene's label, aborting
  // labelling for the whole track if any fails. Show labels only when all fit.
  const labelUsed = Array.from({length: nRowsChrom}, () => []);
  const gap = 3;
  let showLabels = true;
  for (const v of vis) {
    if (!v.gene.n) { v.cx = null; continue; }
    const tw = octx.measureText(v.gene.n).width;
    v.tw = tw;
    const cands = [
      (v.gx0 + v.gx1) / 2,          // centered
      v.gx1 + gap + tw / 2,         // right-flush
      v.gx0 - gap - tw / 2,         // left-flush
    ];
    const used = labelUsed[v.rowIdx];
    let placed = null;
    for (const cx of cands) {
      const lx0 = cx - tw / 2 - 1, lx1 = cx + tw / 2 + 1;
      if (lx0 < LABEL_W || lx1 > W_css) continue;
      let overlaps = false;
      for (const [a, b] of used) {
        if (lx0 < b && lx1 > a) { overlaps = true; break; }
      }
      if (overlaps) continue;
      placed = { cx, lx0, lx1 };
      break;
    }
    if (!placed) { showLabels = false; break; }
    v.cx = placed.cx;
    used.push([placed.lx0, placed.lx1]);
  }

  // Draw pass.
  for (const v of vis) {
    const gene = v.gene;
    const rowIdx = v.rowIdx;
    // Place exon top ~labelBand px below the row's top edge so the label
    // sits above the exon and stays inside the track's clip rect (row 0
    // labels were previously truncated at trackY).
    const rowTop = trackY + rowIdx * rowH;
    const exonY  = rowTop + labelBand + 1;
    const midY   = exonY + exonH / 2;
    const bodyY  = midY - 1;
    const gx0 = v.gx0, gx1 = v.gx1, gw = v.gw;

    octx.fillStyle = th.muted || '#666688';
    octx.fillRect(gx0, bodyY, gw, 2);

    const dir = gene.strand === '+' ? 1 : -1;
    octx.strokeStyle = th.muted || '#666688';
    octx.lineWidth   = 1.2;
    octx.lineJoin    = 'round';
    for (let ax = gx0 + arrowSep * 0.5; ax < gx1 - 6; ax += arrowSep) {
      octx.beginPath();
      octx.moveTo(ax - dir * arrowA, midY - arrowA);
      octx.lineTo(ax,                midY);
      octx.lineTo(ax - dir * arrowA, midY + arrowA);
      octx.stroke();
    }

    const exons = gene.exons?.length ? gene.exons : [[gene.s, gene.e]];
    octx.fillStyle = color;
    for (const [es, ee] of exons) {
      if (ee < vs || es > ve) continue;
      const ex0 = LABEL_W + Math.max(0,     (es - vs) / range * drawW);
      const ex1 = LABEL_W + Math.min(drawW, (ee - vs) / range * drawW);
      const ew  = Math.max(1, ex1 - ex0);
      if (ew > 4) {
        const r = Math.min(2, exonH / 2, ew / 2);
        octx.beginPath();
        octx.roundRect(ex0, exonY, ew, exonH, r);
        octx.fill();
      } else {
        octx.fillRect(ex0, exonY, ew, exonH);
      }
    }

    if (showLabels && v.cx != null && gene.n) {
      octx.fillStyle    = th.label || '#9090c0';
      octx.textAlign    = 'center';
      octx.textBaseline = 'bottom';
      octx.fillText(gene.n, v.cx, exonY - 1);
    }
  }
  octx.restore();
}

// ── Y-axis ticks for quantitative tracks ───────────────────────────────────
function drawYAxis(cfg, trackY) {
  const yMin = cfg.yMin ?? 0;
  const yMax = cfg.yMax ?? 1;
  const h = cfg.height - PAD_V * 2;
  const nTicks = Math.min(5, Math.max(2, Math.floor(h / 20)));
  octx.font         = '8px monospace';
  octx.textAlign    = 'right';
  octx.textBaseline = 'middle';
  octx.fillStyle    = th.muted || '#666688';
  octx.strokeStyle  = th.border || '#252530';
  octx.lineWidth    = 0.5;
  for (let i = 0; i <= nTicks; i++) {
    const frac = i / nTicks;
    const val  = yMin + (yMax - yMin) * (1 - frac);
    const py   = trackY + PAD_V + frac * h;
    octx.fillText(fmtY(val), LABEL_W - 4, py);
    // Gridline
    octx.beginPath();
    octx.moveTo(LABEL_W, py);
    octx.lineTo(LABEL_W + 4, py);
    octx.stroke();
  }
}

function drawOverlay(cfgs, vs, ve, W_css, H_css) {
  octx.clearRect(0, 0, ov.width, ov.height);
  octx.save();
  octx.scale(dpr, dpr);

  // ── Left label panel background ─────────────────────────────────────────
  octx.fillStyle = th.panel || '#1c1c26';
  octx.fillRect(0, 0, LABEL_W, H_css);
  octx.strokeStyle = th.border || '#252530';
  octx.lineWidth   = 1;
  octx.beginPath();
  octx.moveTo(LABEL_W - 0.5, 0);
  octx.lineTo(LABEL_W - 0.5, H_css);
  octx.stroke();

  // ── Scale bar background ────────────────────────────────────────────────
  octx.fillStyle = th.bg || '#13131a';
  octx.fillRect(LABEL_W, 0, W_css - LABEL_W, SCALEBAR_H);
  octx.strokeStyle = th.border || '#252530';
  octx.lineWidth   = 1;
  octx.beginPath();
  octx.moveTo(LABEL_W, SCALEBAR_H - 0.5);
  octx.lineTo(W_css, SCALEBAR_H - 0.5);
  octx.stroke();

  // ── Axis ticks ──────────────────────────────────────────────────────────
  const drawW = W_css - LABEL_W;
  const range  = ve - vs;
  const exp    = Math.pow(10, Math.floor(Math.log10(range)));
  const nice   = [1, 2, 5, 10, 20, 50].map(x => x * exp / 10)
                    .find(x => drawW / (range / x) > 55) || exp;
  const tFirst = Math.ceil(vs / nice) * nice;

  octx.font         = '9px monospace';
  octx.textAlign    = 'center';
  octx.textBaseline = 'top';

  for (let t = tFirst; t < ve; t += nice) {
    const px = LABEL_W + (t - vs) / range * drawW;
    if (px < LABEL_W + 18 || px > W_css - 18) continue;
    octx.strokeStyle = th.border || '#252530';
    octx.lineWidth   = 1;
    octx.beginPath();
    octx.moveTo(px, SCALEBAR_H - 5);
    octx.lineTo(px, SCALEBAR_H);
    octx.stroke();
    octx.fillStyle = th.muted || '#666688';
    octx.fillText(fmtBp(t, nice), px, 4);
  }

  // ── Gene tracks first (behind labels) ───────────────────────────────────
  let cssY = SCALEBAR_H;
  for (const cfg of cfgs) {
    if (cfg.type === 'gene') drawGeneTrack2D(cfg, cssY, vs, ve, W_css);
    cssY += cfg.height;
  }

  // ── Track labels + separators ────────────────────────────────────────────
  cssY = SCALEBAR_H;
  for (const cfg of cfgs) {
    // Track separator
    octx.strokeStyle = th.border || '#252530';
    octx.lineWidth   = 1;
    octx.beginPath();
    octx.moveTo(0, cssY + cfg.height - 0.5);
    octx.lineTo(W_css, cssY + cfg.height - 0.5);
    octx.stroke();

    // Track name (left panel)
    octx.save();
    octx.beginPath();
    octx.rect(0, cssY, LABEL_W - 4, cfg.height);
    octx.clip();
    octx.fillStyle    = th.label || '#9090c0';
    octx.font         = 'bold 10px monospace';
    octx.textAlign    = 'left';
    octx.textBaseline = 'top';
    octx.fillText(cfg.name, 6, cssY + 5);

    if (cfg.type === 'segment' && cfg.groups.length > 1) {
      const nG   = cfg.groups.length;
      const gap  = 0.08;
      const sWeights = cfg.groups.map(g => g.nInd || 1);
      const sTotalW  = sWeights.reduce((a, b) => a + b, 0);
      const sUsable  = cfg.height * (1 - gap);
      const sCumW    = [0];
      for (let i = 0; i < nG; i++) sCumW.push(sCumW[i] + sWeights[i]);
      for (let gi = 0; gi < nG; gi++) {
        const bandTop = sUsable * sCumW[gi] / sTotalW;
        const bandBot = sUsable * sCumW[gi + 1] / sTotalW;
        const gy = cssY + PAD_V + bandTop + (bandBot - bandTop) * 0.5;
        const [r, g, b] = hexRGB(cfg.groups[gi].color);
        octx.fillStyle = `rgba(${(r*255)|0},${(g*255)|0},${(b*255)|0},0.85)`;
        octx.font      = '8px monospace';
        octx.fillText(cfg.groups[gi].name, 6, gy);
      }
    }
    if (cfg.type === 'heatmap' && cfg.groups.length > 1) {
      const nG  = cfg.groups.length;
      const hLWeights = cfg.groups.map(g => g.nInd || 1);
      const hLTotalW  = hLWeights.reduce((a, b) => a + b, 0);
      const hLCumW    = [0];
      for (let i = 0; i < nG; i++) hLCumW.push(hLCumW[i] + hLWeights[i]);
      for (let gi = 0; gi < nG; gi++) {
        const bandTop = cfg.height * hLCumW[gi] / hLTotalW;
        const bandBot = cfg.height * hLCumW[gi + 1] / hLTotalW;
        const gy = cssY + bandTop + (bandBot - bandTop) / 2;
        const [r, g, b] = hexRGB(cfg.groups[gi].color);
        octx.fillStyle = `rgba(${(r*255)|0},${(g*255)|0},${(b*255)|0},0.85)`;
        octx.font      = '8px monospace';
        octx.fillText(`${cfg.groups[gi].name} (${cfg.groups[gi].nInd})`, 6, gy);
      }
    }
    // Group labels for quantitative tracks (scatter/line/fill/histogram)
    if (['scatter', 'line', 'fill', 'histogram'].includes(cfg.type) && cfg.groups.length > 1) {
      for (let gi = 0; gi < cfg.groups.length; gi++) {
        const grp = cfg.groups[gi];
        const [r, g, b] = hexRGB(grp.color);
        octx.fillStyle = `rgba(${(r*255)|0},${(g*255)|0},${(b*255)|0},0.85)`;
        octx.font      = '8px monospace';
        octx.fillText(grp.name, 6, cssY + 18 + gi * 10);
      }
    }
    octx.restore();

    // Y-axis ticks for quantitative tracks
    if (['scatter', 'line', 'fill', 'histogram'].includes(cfg.type)) {
      drawYAxis(cfg, cssY);
    }

    cssY += cfg.height;
  }

  octx.restore();
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN RENDER
// ══════════════════════════════════════════════════════════════════════════════
let rafId = null;

function scheduleRender() {
  if (rafId) return;
  rafId = requestAnimationFrame(render);
}

function render() {
  rafId = null;
  const W_css = wrap.offsetWidth || 800;
  const cfgs  = model.get('track_configs');
  if (!cfgs || !cfgs.length) return;

  const totalH_css = SCALEBAR_H + cfgs.reduce((s, c) => s + c.height, 0);
  const W_gl   = Math.round(W_css * dpr);
  const H_gl   = Math.round(totalH_css * dpr);

  if (glCanvas.width !== W_gl || glCanvas.height !== H_gl) {
    glCanvas.width  = W_gl;  glCanvas.height  = H_gl;
    ov.width        = W_gl;  ov.height        = H_gl;
    glCanvas.style.cssText = `width:${W_css}px;height:${totalH_css}px`;
    ov.style.cssText       = `width:${W_css}px;height:${totalH_css}px`;
    wrap.style.height = totalH_css + 'px';
  }

  gl.viewport(0, 0, W_gl, H_gl);
  const [bgR, bgG, bgB] = hexRGB(th.bg || '#13131a');
  gl.clearColor(bgR, bgG, bgB, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const vs  = vp.start, ve = vp.end, ch = vp.chrom;
  const csz = chromSizes[ch] || 1;
  curXL = LABEL_W / W_css * 2.0 - 1.0;  // NDC x at label edge
  const bpPx = (ve - vs) / (W_css - LABEL_W);
  const useDens = bpPx > DENS_THRESH;
  lodBadge.textContent = useDens ? '▒ density' : '▬ segments';

  let cssY = SCALEBAR_H;
  for (const cfg of cfgs) {
    const trackTop = cssY;
    const trackBot = cssY + cfg.height;

    gl.scissor(
      LABEL_W * dpr,
      H_gl - trackBot * dpr,
      (W_css - LABEL_W) * dpr,
      cfg.height * dpr
    );
    gl.enable(gl.SCISSOR_TEST);

    const tt = cssNDC(trackTop + PAD_V, totalH_css);
    const tb = cssNDC(trackBot - PAD_V, totalH_css);

    if (cfg.type === 'segment') {
      for (const grp of cfg.groups) {
        const gid = grp.id;
        if (useDens) {
          const levels = gpuDens[cfg.id]?.[ch]?.[gid] || [];
          const pxPerBp = (W_css - LABEL_W) / (ve - vs);
          let level = levels[0];
          for (const lv of levels) {
            if (lv.binWidth * pxPerBp >= 2) level = lv;
          }
          if (level) {
            drawDensArea(level, vs, ve, tt, tb, grp.color, 0.80);
            drawDensArea(level, vs, ve, tt, tb, grp.color, 0.40);
          }
        } else {
          drawRects(gpuSeg[cfg.id]?.[ch]?.[gid], vs, ve, tt, tb);
        }
      }
    } else if (cfg.type === 'heatmap') {
      const nG = cfg.groups.length;
      const hWeights = cfg.groups.map(g => g.nInd || 1);
      const hTotalW  = hWeights.reduce((a, b) => a + b, 0);
      const hUsable  = cfg.height;
      const hCumW    = [0];
      for (let i = 0; i < nG; i++) hCumW.push(hCumW[i] + hWeights[i]);
      for (let gi = 0; gi < nG; gi++) {
        const grp  = cfg.groups[gi];
        const bandTT = cssNDC(trackTop + PAD_V + hUsable * hCumW[gi] / hTotalW, totalH_css);
        const bandTB = cssNDC(trackTop + PAD_V + hUsable * hCumW[gi + 1] / hTotalW, totalH_css);
        drawHeatmapBand(
          gpuHM[cfg.id]?.[ch]?.[grp.id],
          vs, ve, csz, bandTT, bandTB, grp.color
        );
      }
    } else if (cfg.type === 'gene') {
      // Rendered entirely on 2D overlay — no GL needed
    } else if (cfg.type === 'scatter') {
      const pointSize = cfg.pointSize || 3;
      for (const grp of cfg.groups) {
        drawPoints(gpuXY[cfg.id]?.[ch]?.[grp.id], vs, ve, tt, tb, grp.color, 0.85, pointSize);
      }
    } else if (cfg.type === 'line') {
      for (const grp of cfg.groups) {
        drawLine(gpuXY[cfg.id]?.[ch]?.[grp.id], vs, ve, tt, tb, grp.color, 1.0);
      }
    } else if (cfg.type === 'fill') {
      for (const grp of cfg.groups) {
        const colorPos = grp.colorPos || grp.color;
        const colorNeg = grp.colorNeg || grp.color;
        drawFill(gpuFill[cfg.id]?.[ch]?.[grp.id], vs, ve, tt, tb, colorPos, colorNeg, 0.6);
      }
    } else if (cfg.type === 'histogram') {
      for (const grp of cfg.groups) {
        drawRects(gpuHist[cfg.id]?.[ch]?.[grp.id], vs, ve, tt, tb);
      }
    }

    gl.disable(gl.SCISSOR_TEST);
    cssY += cfg.height;
  }

  drawOverlay(cfgs, vs, ve, W_css, totalH_css);
}

// ══════════════════════════════════════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════════════════════════════════════
let chromSizes = model.get('chrom_sizes');
let vp         = { ...model.get('viewport') };
let geneData   = {};

function buildChromSel() {
  chromSel.innerHTML = '';
  for (const ch of Object.keys(chromSizes)) {
    const o = document.createElement('option');
    o.value = o.textContent = ch;
    chromSel.append(o);
  }
  chromSel.value = vp.chrom;
}
buildChromSel();

function updatePosBox() {
  posInput.value = `${vp.chrom}:${Math.round(vp.start).toLocaleString()}–${Math.round(vp.end).toLocaleString()}`;
}
updatePosBox();

// ══════════════════════════════════════════════════════════════════════════════
// INTERACTIONS
// ══════════════════════════════════════════════════════════════════════════════
let isDragging = false, dragX0 = 0, dragVp0 = null;
let wheelTimer = null;

function syncVp() { model.set('viewport', { ...vp }); model.save_changes(); }
const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;

glCanvas.addEventListener('mousedown', e => {
  isDragging = true; dragX0 = e.clientX; dragVp0 = { ...vp };
  glCanvas.classList.add('dragging');
});

// ── Tooltip helpers ─────────────────────────────────────────────────────────
function trackAtY(offsetY) {
  const cfgs = model.get('track_configs');
  if (!cfgs) return null;
  let cssY = SCALEBAR_H;
  for (const cfg of cfgs) {
    if (offsetY >= cssY && offsetY < cssY + cfg.height)
      return { cfg, trackTop: cssY, localY: offsetY - cssY };
    cssY += cfg.height;
  }
  return null;
}

// ── Tooltip format engine ───────────────────────────────────────────────────
// Applies a Python-style format string: {key} or {key:.Nf}
function fmtTip(fmt, obj) {
  let ok = true;
  const result = fmt.replace(/\{(\w+)(?::([^}]+))?\}/g, (_, key, spec) => {
    if (!(key in obj)) { ok = false; return `{${key}}`; }
    const v = obj[key];
    if (spec && typeof v === 'number') {
      const m = spec.match(/^\.(\d+)f$/);
      if (m) return v.toFixed(parseInt(m[1]));
    }
    if (typeof v === 'number' && !spec) return v.toPrecision(4);
    return String(v);
  });
  return ok ? result : null;
}

// ── Tooltip data extractors ─────────────────────────────────────────────────

function bisectStride(arr, val, stride) {
  let lo = 0, hi = (arr.length / stride) | 0;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    arr[m * stride] < val ? lo = m + 1 : hi = m;
  }
  return lo;
}

function tipDataGene(pos, chrom, cfg) {
  const entry = geneData?.[cfg.id]?.[chrom];
  const genes = entry?.records ?? (Array.isArray(entry) ? entry : []);
  const hits = [];
  for (const g of genes) {
    if (pos >= g.s && pos <= g.e)
      hits.push({ name: g.n || 'unnamed', strand: g.strand === '+' ? '\u2192' : '\u2190', start: g.s, end: g.e });
  }
  return hits.length ? hits : null;
}

function tipDataXY(pos, chrom, cfg) {
  const items = [];
  for (const grp of cfg.groups) {
    const arr = rawXY?.[cfg.id]?.[chrom]?.[grp.id];
    if (!arr || arr.length < 2) continue;
    const n = (arr.length / 2) | 0;
    const idx = bisectStride(arr, pos, 2);
    let best = -1, bestDist = Infinity;
    for (const c of [idx - 1, idx]) {
      if (c >= 0 && c < n) {
        const d = Math.abs(arr[c * 2] - pos);
        if (d < bestDist) { bestDist = d; best = c; }
      }
    }
    if (best >= 0) items.push({ group: grp.name, value: arr[best * 2 + 1], x: arr[best * 2] });
  }
  return items.length ? items : null;
}

function tipDataFill(pos, chrom, cfg) {
  const items = [];
  for (const grp of cfg.groups) {
    const arr = rawFill?.[cfg.id]?.[chrom]?.[grp.id];
    if (!arr || arr.length < 3) continue;
    const n = (arr.length / 3) | 0;
    const idx = bisectStride(arr, pos, 3);
    let best = -1, bestDist = Infinity;
    for (const c of [idx - 1, idx]) {
      if (c >= 0 && c < n) {
        const d = Math.abs(arr[c * 3] - pos);
        if (d < bestDist) { bestDist = d; best = c; }
      }
    }
    if (best >= 0) items.push({ group: grp.name, lo: arr[best * 3 + 1], hi: arr[best * 3 + 2], x: arr[best * 3] });
  }
  return items.length ? items : null;
}

function tipDataHist(pos, chrom, cfg) {
  const binW = cfg.binWidth ?? 1;
  const halfW = binW / 2;
  const items = [];
  for (const grp of cfg.groups) {
    const raw = rawHist?.[cfg.id]?.[chrom]?.[grp.id];
    if (!raw) continue;
    const arr = raw.data;
    if (!arr || arr.length < 2) continue;
    const n = (arr.length / 2) | 0;
    const idx = bisectStride(arr, pos, 2);
    for (const c of [idx - 1, idx]) {
      if (c >= 0 && c < n) {
        const cx = arr[c * 2];
        if (pos >= cx - halfW && pos <= cx + halfW) {
          items.push({ group: grp.name, value: arr[c * 2 + 1], x: cx });
          break;
        }
      }
    }
  }
  return items.length ? items : null;
}

function tipDataSegment(pos, chrom, cfg, localY) {
  const nG = cfg.groups.length;
  if (nG === 0 || localY < 0) return null;
  const gap = 0.08;
  const weights = cfg.groups.map(g => g.nInd || 1);
  const totalW = weights.reduce((a, b) => a + b, 0);
  const frac = localY / cfg.height;
  let cumW = 0;
  for (let gi = 0; gi < nG; gi++) {
    const bandTop = gap * 0.5 + (1 - gap) * cumW / totalW;
    cumW += weights[gi];
    const bandBot = gap * 0.5 + (1 - gap) * cumW / totalW;
    if (frac >= bandTop && frac <= bandBot) return { group: cfg.groups[gi].name };
  }
  return nG === 1 ? { group: cfg.groups[0].name } : null;
}

function tipDataHeatmap(pos, chrom, cfg, localY) {
  if (localY < 0) return null;
  const nG = cfg.groups.length;
  const weights = cfg.groups.map(g => g.nInd || 1);
  const totalW = weights.reduce((a, b) => a + b, 0);
  const frac = (localY - PAD_V) / cfg.height;
  let cumW = 0;
  for (let gi = 0; gi < nG; gi++) {
    const bandTop = cumW / totalW;
    cumW += weights[gi];
    const bandBot = cumW / totalW;
    if (frac >= bandTop && frac <= bandBot) {
      const grp = cfg.groups[gi];
      const indFrac = (frac - bandTop) / (bandBot - bandTop);
      const indIdx = Math.min(Math.floor(indFrac * (grp.nInd || 1)), (grp.nInd || 1) - 1);
      return { group: grp.name, individual: indIdx + 1, nInd: grp.nInd };
    }
  }
  return null;
}

// ── Tooltip dispatch ────────────────────────────────────────────────────────

// Default formatters (replicate previous hardcoded behaviour)
function defaultFmtGene(items) { return items.map(d => `${d.name} ${d.strand}`).join(', '); }
function defaultFmtXY(items, nGroups) { return items.map(d => (nGroups > 1 ? `${d.group}: ` : '') + d.value.toPrecision(4)).join(', '); }
function defaultFmtFill(items, nGroups) { return items.map(d => (nGroups > 1 ? `${d.group}: ` : '') + `${d.lo.toPrecision(4)}\u2013${d.hi.toPrecision(4)}`).join(', '); }
function defaultFmtHist(items, nGroups) { return items.map(d => (nGroups > 1 ? `${d.group}: ` : '') + d.value.toPrecision(4)).join(', '); }
function defaultFmtSegment(d) { return d.group; }
function defaultFmtHeatmap(d) { return `${d.group} [${d.individual}/${d.nInd}]`; }

function tipForTrack(pos, chrom, cfg, localY) {
  if (cfg.tipFmt === false) return null;
  let data, keys;
  switch (cfg.type) {
    case 'gene':
      data = tipDataGene(pos, chrom, cfg);
      keys = 'name, strand, start, end';
      if (!data) return null;
      if (cfg.tipFmt) {
        const parts = data.map(d => fmtTip(cfg.tipFmt, d)).filter(s => s !== null);
        return parts.length ? parts.join(', ') : `keys: ${keys}`;
      }
      return defaultFmtGene(data);

    case 'scatter': case 'line':
      data = tipDataXY(pos, chrom, cfg);
      keys = 'group, value, x';
      if (!data) return null;
      if (cfg.tipFmt) {
        const parts = data.map(d => fmtTip(cfg.tipFmt, d)).filter(s => s !== null);
        return parts.length ? parts.join(', ') : `keys: ${keys}`;
      }
      return defaultFmtXY(data, cfg.groups.length);

    case 'fill':
      data = tipDataFill(pos, chrom, cfg);
      keys = 'group, lo, hi, x';
      if (!data) return null;
      if (cfg.tipFmt) {
        const parts = data.map(d => fmtTip(cfg.tipFmt, d)).filter(s => s !== null);
        return parts.length ? parts.join(', ') : `keys: ${keys}`;
      }
      return defaultFmtFill(data, cfg.groups.length);

    case 'histogram':
      data = tipDataHist(pos, chrom, cfg);
      keys = 'group, value, x';
      if (!data) return null;
      if (cfg.tipFmt) {
        const parts = data.map(d => fmtTip(cfg.tipFmt, d)).filter(s => s !== null);
        return parts.length ? parts.join(', ') : `keys: ${keys}`;
      }
      return defaultFmtHist(data, cfg.groups.length);

    case 'segment':
      data = tipDataSegment(pos, chrom, cfg, localY);
      keys = 'group';
      if (!data) return null;
      if (cfg.tipFmt) return fmtTip(cfg.tipFmt, data) ?? `keys: ${keys}`;
      return defaultFmtSegment(data);

    case 'heatmap':
      data = tipDataHeatmap(pos, chrom, cfg, localY);
      keys = 'group, individual, nInd';
      if (!data) return null;
      if (cfg.tipFmt) return fmtTip(cfg.tipFmt, data) ?? `keys: ${keys}`;
      return defaultFmtHeatmap(data);

    default: return null;
  }
}

const onMove = e => {
  if (!isDragging) {
    const mx = e.offsetX - LABEL_W;
    if (mx < 0 || e.target !== glCanvas) { tooltip.style.display = 'none'; return; }
    const W = glCanvas.offsetWidth - LABEL_W;
    const pos = Math.round(vp.start + (mx / W) * (vp.end - vp.start));
    const lines = [`${vp.chrom}:${pos.toLocaleString()}`];
    const cfgs = model.get('track_configs');
    if (cfgs) {
      const hit = trackAtY(e.offsetY);
      let cssY = SCALEBAR_H;
      for (const cfg of cfgs) {
        const localY = (hit && hit.cfg.id === cfg.id) ? hit.localY : -1;
        const tip = tipForTrack(pos, vp.chrom, cfg, localY);
        const label = cfg.tipLabel ?? '';
        if (tip) {
          lines.push(label ? `${label} ${tip}` : tip);
        } else if (label && cfg.tipFmt !== false) {
          lines.push(label);
        }
        cssY += cfg.height;
      }
    }
    tooltip.textContent = lines.join('\n');
    tooltip.style.cssText = `display:block;left:${e.offsetX + 14}px;top:${e.offsetY - 6}px`;
    return;
  }
  const W     = glCanvas.offsetWidth - LABEL_W;
  const bpPx  = (dragVp0.end - dragVp0.start) / W;
  const shift = -(e.clientX - dragX0) * bpPx;
  const range = dragVp0.end - dragVp0.start;
  const lo    = panLo(dragVp0.chrom), hi = panHi(dragVp0.chrom);
  const ns    = clamp(dragVp0.start + shift, lo, hi - range);
  vp = { chrom: dragVp0.chrom, start: ns, end: ns + range };
  scheduleRender();
};

const onUp = () => {
  if (!isDragging) return;
  isDragging = false;
  glCanvas.classList.remove('dragging');
  syncVp(); updatePosBox();
};

window.addEventListener('mousemove', onMove);
window.addEventListener('mouseup',   onUp);
glCanvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

glCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  const W      = glCanvas.offsetWidth - LABEL_W;
  const mx     = clamp(e.offsetX - LABEL_W, 0, W);
  const frac   = mx / W;
  const range  = vp.end - vp.start;
  const anchor = vp.start + frac * range;
  const zs     = model.get('zoom_speed') || 1.3;
  const factor = e.deltaY > 0 ? zs : 1 / zs;
  const lo     = panLo(vp.chrom), hi = panHi(vp.chrom);
  const span   = hi - lo;
  const nr     = clamp(range * factor, 500, span);
  const ns     = clamp(anchor - frac * nr, lo, hi - nr);
  vp = { chrom: vp.chrom, start: ns, end: ns + nr };
  scheduleRender();
  clearTimeout(wheelTimer);
  wheelTimer = setTimeout(() => { syncVp(); updatePosBox(); }, 150);
}, { passive: false });

glCanvas.addEventListener('dblclick', e => {
  const W     = glCanvas.offsetWidth - LABEL_W;
  const frac  = clamp(e.offsetX - LABEL_W, 0, W) / W;
  const range = vp.end - vp.start;
  const lo    = panLo(vp.chrom), hi = panHi(vp.chrom);
  const nr    = Math.max(500, range / 2);
  const anchor = vp.start + frac * range;
  const ns    = clamp(anchor - frac * nr, lo, hi - nr);
  vp = { chrom: vp.chrom, start: ns, end: ns + nr };
  scheduleRender(); syncVp(); updatePosBox();
});

const onKey = e => {
  if (!el.matches(':hover, :focus-within')) return;
  let { start: ns, end: ne } = vp;
  const range = ne - ns;
  const lo    = panLo(vp.chrom), hi = panHi(vp.chrom);
  const span  = hi - lo;
  const step  = range * 0.15;
  if (e.key === 'ArrowRight') { ns = clamp(ns + step, lo, hi - range); ne = ns + range; }
  if (e.key === 'ArrowLeft')  { ns = clamp(ns - step, lo, hi - range); ne = ns + range; }
  if (e.key === '+' || e.key === '=') {
    const nr = Math.max(500, range / 1.5), mid = (ns + ne) / 2;
    ns = clamp(mid - nr/2, lo, hi - nr); ne = ns + nr;
  }
  if (e.key === '-') {
    const nr = Math.min(span, range * 1.5), mid = (ns + ne) / 2;
    ns = clamp(mid - nr/2, lo, hi - nr); ne = ns + nr;
  }
  if (ns !== vp.start || ne !== vp.end) {
    vp = { chrom: vp.chrom, start: ns, end: ne };
    scheduleRender(); syncVp(); updatePosBox();
  }
};
window.addEventListener('keydown', onKey);

chromSel.addEventListener('change', () => {
  const ch = chromSel.value;
  vp = { chrom: ch, start: panLo(ch), end: panHi(ch) };
  scheduleRender(); syncVp(); updatePosBox();
});

posInput.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const m = posInput.value.replace(/,/g, '').match(/^(?:([\w.]+):)?(\d+)[–\-](\d+)$/);
  if (!m) return;
  const ch = m[1] || vp.chrom;
  const s = +m[2], en = +m[3];
  if (en > s && chromSizes[ch] != null) {
    const lo = panLo(ch), hi = panHi(ch);
    const ns = clamp(s,  lo, hi);
    const ne = clamp(en, lo, hi);
    if (ne > ns) {
      vp = { chrom: ch, start: ns, end: ne };
      chromSel.value = ch; scheduleRender(); syncVp();
    }
  }
});

const doZoom = f => {
  const mid = (vp.start + vp.end) / 2;
  const lo  = panLo(vp.chrom), hi = panHi(vp.chrom);
  const span = hi - lo;
  const nr  = clamp((vp.end - vp.start) * f, 500, span);
  const ns  = clamp(mid - nr / 2, lo, hi - nr);
  vp = { chrom: vp.chrom, start: ns, end: ns + nr };
  scheduleRender(); syncVp(); updatePosBox();
};

zoomInBtn.addEventListener ('click', () => doZoom(0.5));
zoomOutBtn.addEventListener('click', () => doZoom(2.0));
resetBtn.addEventListener  ('click', () => {
  vp = { chrom: vp.chrom, start: panLo(vp.chrom), end: panHi(vp.chrom) };
  scheduleRender(); syncVp(); updatePosBox();
});

function updateHeatmapBtns() {
  const cfgs = model.get('track_configs') || [];
  const hasHM = cfgs.some(c => c.type === 'heatmap');
  hmRecBtn.style.display  = hasHM ? '' : 'none';
  hmGlobBtn.style.display = hasHM ? '' : 'none';
}
updateHeatmapBtns();

function randId() {
  return 'cmd-' + Math.random().toString(36).slice(2) + '-' + Date.now().toString(36);
}
function invokeCmd(name, msg) {
  // Match anywidget's command envelope so the @command-decorated Python
  // handler is routed correctly — plain model.send(...) with a custom
  // envelope does not reliably reach on_msg in VS Code's Jupyter.
  model.send({ id: randId(), kind: 'anywidget-command', name, msg });
}

hmRecBtn.addEventListener('click', () => {
  const cfgs = model.get('track_configs') || [];
  for (const c of cfgs) {
    if (c.type === 'heatmap') {
      invokeCmd('_cmd_heatmap_recompute', {
        tid:  c.id,
        chrom: vp.chrom,
        xStart: Math.floor(vp.start),
        xEnd:   Math.ceil(vp.end),
      });
    }
  }
});
hmGlobBtn.addEventListener('click', () => {
  const cfgs = model.get('track_configs') || [];
  for (const c of cfgs) {
    if (c.type === 'heatmap') {
      invokeCmd('_cmd_heatmap_reset', { tid: c.id });
    }
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// MODEL CHANGE LISTENERS
// ══════════════════════════════════════════════════════════════════════════════
model.on('change:theme', () => { applyTheme(); scheduleRender(); });
model.on('change:chrom_sizes', () => {
  chromSizes = model.get('chrom_sizes'); buildChromSel(); scheduleRender();
});
model.on('change:track_configs', () => { updateHeatmapBtns(); scheduleRender(); });
model.on('change:track_data', () => { uploadTrackData(); scheduleRender(); });

model.on('change:viewport', () => {
  if (!isDragging) {
    vp = { ...model.get('viewport') };
    chromSel.value = vp.chrom;
    updatePosBox(); scheduleRender();
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// RESIZE OBSERVER
// ══════════════════════════════════════════════════════════════════════════════
const ro = new ResizeObserver(() => scheduleRender());
ro.observe(wrap);

// ══════════════════════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════════════════════
uploadTrackData();
scheduleRender();

return () => {
  ro.disconnect();
  window.removeEventListener('mousemove', onMove);
  window.removeEventListener('mouseup',   onUp);
  window.removeEventListener('keydown',   onKey);
  clearTimeout(wheelTimer);
  if (rafId) cancelAnimationFrame(rafId);
};

} // end render()
"""


# ─────────────────────────────────────────────────────────────────────────────
# Widget
# ─────────────────────────────────────────────────────────────────────────────
class SegmentViewer(anywidget.AnyWidget):
    """
    WebGL2 genomic segment viewer.

    Tracks
    ------
    add_segment_track   — instanced GPU rects; auto-LOD to density histogram
    add_heatmap_track   — R8 texture; handles 10 k+ individuals
    add_gene_track      — exon blocks + intron connectors
    add_scatter_track   — point cloud
    add_line_track      — connected line
    add_fill_track      — fill-between two curves, pos/neg coloring
    add_histogram_track — vertical bars

    All pan/zoom is uniform-only (zero CPU/GPU buffer work per frame).
    Segment viewport culling uses sorted attribute-pointer offsets.
    """

    _esm = _JS
    _css = _CSS

    chrom_sizes   = traitlets.Dict({}).tag(sync=True)
    track_configs = traitlets.List([]).tag(sync=True)
    track_data    = traitlets.Dict({}).tag(sync=True)
    viewport      = traitlets.Dict({'chrom': '', 'start': 0, 'end': 0}).tag(sync=True)
    zoom_speed    = traitlets.Float(1.3).tag(sync=True)
    theme         = traitlets.Dict({
        'bg':           '#13131a',
        'fg':           '#d0d0e8',
        'panel':        '#1c1c26',
        'border':       '#252530',
        'input_bg':     '#0d0d14',
        'input_fg':     '#c0c0dc',
        'input_border': '#33334a',
        'accent':       '#4466ee',
        'muted':        '#666688',
        'label':        '#9090c0',
    }).tag(sync=True)

    _PALETTE = [
        '#4488cc', '#ee5566', '#33aa77', '#ddaa22',
        '#66bbcc', '#cc44aa', '#88aa44', '#aa6644',
    ]

    def __init__(self, chrom_sizes: Dict[str, int], **kw):
        super().__init__(**kw)
        self.chrom_sizes = {str(k): int(v) for k, v in chrom_sizes.items()}
        first = next(iter(self.chrom_sizes))
        self.viewport = {'chrom': first, 'start': 0, 'end': self.chrom_sizes[first]}
        self._heatmap_sources: Dict[str, Dict] = {}
        self._heatmap_global:  Dict[str, Dict] = {}

    @anywidget.experimental.command
    def _cmd_heatmap_recompute(self, msg, buffers):
        self._rebin_heatmap_for_view(
            msg['tid'],
            str(msg['chrom']),
            int(msg['xStart']),
            int(msg['xEnd']),
        )
        return (None, [])

    @anywidget.experimental.command
    def _cmd_heatmap_reset(self, msg, buffers):
        self._restore_heatmap_global(msg['tid'])
        return (None, [])

    def _bin_heatmap_region(
        self,
        tid: str,
        chrom: str,
        x_start: int,
        x_end: int,
        windows: int,
    ) -> Dict:
        """Rebin heatmap data for a single chrom over [x_start, x_end]."""
        src = self._heatmap_sources.get(tid)
        if src is None:
            return {}
        df         = src['df']
        groups     = src['groups']
        group_col  = src['group_col']
        ind_col    = src['individual_col']
        inds_by_gi = src['inds_by_group']

        span = max(1, x_end - x_start)
        out: Dict = {}

        cdf_all = df[df['chrom'].astype(str) == chrom]
        for gi, group in enumerate(groups):
            gid  = str(gi)
            inds = inds_by_gi[gid]
            nInd = len(inds)
            ind_idx = {ind: i for i, ind in enumerate(inds)}
            gdf  = cdf_all[cdf_all[group_col] == group] if group_col else cdf_all

            matrix = np.zeros((nInd, windows), dtype=np.uint8)
            if not gdf.empty:
                starts_arr = gdf['start'].to_numpy()
                ends_arr   = gdf['end'].to_numpy()
                inds_arr   = gdf[ind_col].to_numpy()
                for k in range(len(gdf)):
                    ii = ind_idx.get(inds_arr[k])
                    if ii is None: continue
                    s = max(starts_arr[k], x_start)
                    e = min(ends_arr[k],   x_end)
                    if e <= s: continue
                    b0 = int(max(0, (s - x_start) / span * windows))
                    b1 = int(min(windows - 1, (e - x_start) / span * windows)) + 1
                    matrix[ii, b0:b1] = np.minimum(
                        255, matrix[ii, b0:b1].astype(int) + 80
                    )

            out[gid] = {
                'data':   self._pack_u8(matrix.ravel()),
                'nInd':   nInd,
                'nWin':   windows,
                'xStart': int(x_start),
                'xEnd':   int(x_end),
            }
        return out

    def _rebin_heatmap_for_view(
        self, tid: str, chrom: str, x_start: int, x_end: int,
    ) -> None:
        src = self._heatmap_sources.get(tid)
        if src is None:
            return
        new_entry = self._bin_heatmap_region(
            tid, chrom, x_start, x_end, src['windows']
        )
        if not new_entry:
            return
        hm = {**self.track_data.get(tid, {})}
        hm[chrom] = new_entry
        self.track_data = {**self.track_data, tid: hm}

    def _restore_heatmap_global(self, tid: str) -> None:
        cached = self._heatmap_global.get(tid)
        if cached is None:
            return
        self.track_data = {**self.track_data, tid: {
            ch: {gid: {**entry} for gid, entry in groups.items()}
            for ch, groups in cached.items()
        }}

    def _tid(self): return f't{len(self.track_configs)}'

    @staticmethod
    def _pack_f32(arr: np.ndarray) -> str:
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

    @staticmethod
    def _pack_u8(arr: np.ndarray) -> str:
        return base64.b64encode(arr.astype(np.uint8).tobytes()).decode()

    # ── add_segment_track ────────────────────────────────────────────────────
    def add_segment_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        group_by: Optional[str] = None,
        individual_col: Optional[str] = None,
        color_map: Optional[Dict] = None,
        height: Optional[int] = None,
        density_windows: Union[tuple, int] = (256, 1024, 4096),
        tip_fmt: Optional[str] = None,
        tip_label: Optional[str] = None,
    ) -> 'SegmentViewer':
        """
        Segment track with GPU instanced rendering.

        Parameters
        ----------
        df              : DataFrame with columns [chrom, start, end, ...].
        name            : Track display label.
        group_by        : Column whose unique values become separate sub-rows.
        individual_col  : Column identifying individuals.  When provided,
                          each individual gets its own horizontal row within
                          the group band (stacked haplotype view).  Track height
                          auto-adapts so each row is at least 1 px.
        color_map       : {group_value: '#rrggbb'}.
        height          : Track height in CSS px.  If not specified, defaults
                          to the larger of 90 and the number of rows (one
                          pixel per row).
        density_windows : Tuple of bin counts for multi-resolution density LOD.
                          A single int is also accepted (treated as one level).
        tip_fmt         : Python format string for tooltip.
                          Available keys: ``{group}``.
        """
        if isinstance(density_windows, int):
            density_windows = (density_windows,)
        tid = self._tid()
        if group_by and group_by in df.columns:
            groups = sorted(df[group_by].dropna().unique(), key=str)
        else:
            groups = ['all']
            group_by = None

        if isinstance(color_map, str):
            color_map = {g: color_map for g in groups}
        if color_map is None:
            color_map = {g: self._PALETTE[i % len(self._PALETTE)]
                         for i, g in enumerate(groups)}

        # Build global per-group individual maps (consistent across chromosomes)
        use_ind = individual_col and individual_col in df.columns
        grp_ind_map: Dict[str, Dict] = {}
        grp_nind: Dict[str, int] = {}
        if use_ind:
            for gi, group in enumerate(groups):
                gdf = df[df[group_by] == group] if group_by else df
                ind_uniq = sorted(gdf[individual_col].dropna().unique(), key=str)
                grp_ind_map[str(gi)] = {v: i for i, v in enumerate(ind_uniq)}
                grp_nind[str(gi)] = len(ind_uniq)

        if height is None:
            total_rows = sum(grp_nind.values()) if use_ind else len(groups)
            height = max(90, total_rows)

        seg_out: Dict  = {}
        dens_out: Dict = {}

        for chrom_val, cdf in df.groupby('chrom'):
            chrom = str(chrom_val)
            csz   = self.chrom_sizes.get(chrom, int(cdf['end'].max()))
            seg_out[chrom]  = {}
            dens_out[chrom] = {}

            for gi, group in enumerate(groups):
                gid = str(gi)
                gdf = cdf[cdf[group_by] == group] if group_by else cdf
                if gdf.empty:
                    seg_out[chrom][gid]  = ''
                    dens_out[chrom][gid] = ''
                    continue

                gdf = gdf.sort_values('start')
                if use_ind:
                    ind_map = grp_ind_map[gid]
                    ind_idx = np.array([ind_map[v] for v in gdf[individual_col].values],
                                       dtype=np.float32)
                    arr = np.empty(len(gdf) * 3, dtype=np.float32)
                    arr[0::3] = gdf['start'].to_numpy()
                    arr[1::3] = gdf['end'].to_numpy()
                    arr[2::3] = ind_idx
                else:
                    arr = np.empty(len(gdf) * 2, dtype=np.float32)
                    arr[0::2] = gdf['start'].to_numpy()
                    arr[1::2] = gdf['end'].to_numpy()
                seg_out[chrom][gid] = self._pack_f32(arr)

                starts = gdf['start'].to_numpy()
                levels = {}
                for n in density_windows:
                    bins = np.clip(
                        (starts / csz * n).astype(int), 0, n - 1
                    )
                    counts = np.zeros(n, dtype=np.float32)
                    np.add.at(counts, bins, 1)
                    levels[str(n)] = self._pack_f32(counts)
                dens_out[chrom][gid] = levels

        cfg = {
            'id': tid, 'type': 'segment', 'name': name, 'height': height,
            'groups': [
                {'id': str(i), 'name': str(g),
                 'nInd': grp_nind.get(str(i), 0),
                 'color': color_map.get(g, self._PALETTE[i % len(self._PALETTE)])}
                for i, g in enumerate(groups)
            ],
            **(({'tipFmt': tip_fmt} if tip_fmt is not None else {})),
            'tipLabel': tip_label if tip_label is not None else f'{name}:',
        }

        self.track_data    = {**self.track_data, tid: {'segs': seg_out, 'dens': dens_out}}
        self.track_configs = [*self.track_configs, cfg]
        return self

    # ── add_heatmap_track ────────────────────────────────────────────────────
    def add_heatmap_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        individual_col: str = 'sample',
        group_col: Optional[str] = None,
        sort_by: Optional[str] = None,
        color_map: Optional[Dict] = None,
        height: Optional[int] = None,
        windows: int = 1000,
        tip_fmt: Optional[str] = None,
        tip_label: Optional[str] = None,
    ) -> 'SegmentViewer':
        """
        Heatmap track — one row per individual, GPU texture rendering.

        Parameters
        ----------
        df             : DataFrame with [chrom, start, end, individual_col, ...].
        name           : Track display label.
        individual_col : Column identifying each haplotype / sample row.
        group_col      : Column for grouping individuals into coloured bands.
        sort_by        : Column to sort individuals by within each group.
        color_map      : {group_value: '#rrggbb'}.
        height         : Track height in CSS px.  If not specified, defaults
                         to the larger of 90 and the total number of
                         individuals (one pixel per row).
        windows        : Number of pre-computed density bins (whole chromosome).
        tip_fmt        : Python format string for tooltip.
                         Available keys: ``{group}``, ``{individual}``,
                         ``{nInd}``.
        """
        tid = self._tid()

        if group_col and group_col in df.columns:
            groups = list(df[group_col].dropna().unique())
        else:
            groups = ['all']
            group_col = None

        if isinstance(color_map, str):
            color_map = {g: color_map for g in groups}
        if color_map is None:
            color_map = {g: self._PALETTE[i % len(self._PALETTE)]
                         for i, g in enumerate(groups)}

        # Resolve per-group individual ordering once — stable across rebins.
        inds_by_group: Dict[str, List] = {}
        grp_meta: List = []
        for gi, group in enumerate(groups):
            gdf = df[df[group_col] == group] if group_col else df
            inds = list(gdf[individual_col].unique())
            if sort_by and sort_by in gdf.columns:
                order = gdf.groupby(individual_col)[sort_by].first().sort_values().index
                inds  = [i for i in order if i in set(inds)]
            inds_by_group[str(gi)] = inds
            grp_meta.append({
                'id':    str(gi),
                'name':  str(group),
                'color': color_map.get(group, self._PALETTE[gi % len(self._PALETTE)]),
                'nInd':  len(inds),
            })

        # Retain source data so we can rebin on demand.
        needed_cols = ['chrom', 'start', 'end', individual_col]
        if group_col and group_col not in needed_cols:
            needed_cols.append(group_col)
        if sort_by and sort_by in df.columns and sort_by not in needed_cols:
            needed_cols.append(sort_by)
        self._heatmap_sources[tid] = {
            'df':             df[needed_cols].copy(),
            'groups':         groups,
            'group_col':      group_col,
            'individual_col': individual_col,
            'sort_by':        sort_by,
            'windows':        windows,
            'inds_by_group':  inds_by_group,
        }

        # Build initial (whole-chromosome) binning via the helper.
        hm_out: Dict = {}
        for chrom_val in df['chrom'].astype(str).unique():
            chrom = str(chrom_val)
            csz   = self.chrom_sizes.get(chrom)
            if csz is None:
                csz = int(df.loc[df['chrom'].astype(str) == chrom, 'end'].max())
            hm_out[chrom] = self._bin_heatmap_region(tid, chrom, 0, csz, windows)

        # Cache global snapshot for cheap reset.
        self._heatmap_global[tid] = {
            ch: {gid: {**entry} for gid, entry in groups_.items()}
            for ch, groups_ in hm_out.items()
        }

        if height is None:
            total_rows = sum(g['nInd'] for g in grp_meta)
            height = max(90, total_rows)

        cfg = {
            'id': tid, 'type': 'heatmap', 'name': name, 'height': height,
            'groups': grp_meta,
            **(({'tipFmt': tip_fmt} if tip_fmt is not None else {})),
            'tipLabel': tip_label if tip_label is not None else f'{name}:',
        }
        self.track_data    = {**self.track_data, tid: hm_out}
        self.track_configs = [*self.track_configs, cfg]
        return self

    # ── add_gene_track ───────────────────────────────────────────────────────
    @staticmethod
    def _collapse_exons(transcripts) -> List:
        """Merge overlapping exons across all transcripts into a single list."""
        all_exons = sorted(ex for t in transcripts for ex in t)
        if not all_exons:
            return []
        merged = [list(all_exons[0])]
        for s, e in all_exons[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    def add_gene_track(
        self,
        genes_data,
        exons_df=None,
        *,
        name: str = 'Genes',
        color: str = '#4488cc',
        height: Optional[int] = None,
        collapse: bool = True,
        tip_fmt: Optional[str] = None,
        tip_label: Optional[str] = None,
    ) -> 'SegmentViewer':
        """
        Gene / exon annotation track.

        Parameters
        ----------
        genes_data : Either a dict mapping chrom -> list of
                     (gene_name, chrom, start, end, strand, transcripts)
                     tuples (as returned by geneinfo), or a DataFrame with
                     [chrom, start, end, name, strand].
        exons_df   : Optional DataFrame with [chrom, gene_name, start, end].
                     Only used with the DataFrame form of genes_data.
        name       : Track display label.
        color      : Exon block colour.
        height     : Track height in CSS px.  If not specified, auto-sizes
                     to the required number of rows (positive strand above,
                     negative strand below, each with sub-lanes on overlap).
        collapse   : If True (default), merge exons across all transcripts
                     into a single union set.  If False, each transcript
                     becomes its own row in the gene track.
        tip_fmt    : Python format string for tooltip.
                     Available keys: ``{name}``, ``{strand}``, ``{start}``,
                     ``{end}``.
        """
        tid   = self._tid()
        gdata: Dict = {}

        if isinstance(genes_data, dict) and not isinstance(genes_data, pd.DataFrame):
            # genes_by_chrom dict: {chrom: [(name, chrom, start, end, strand, transcripts), ...]}
            for chrom, gene_list in genes_data.items():
                chrom = str(chrom)
                recs: List = []
                for entry in gene_list:
                    gname, _, gs, ge, strand, transcripts = entry
                    strand = str(strand) if strand in ('+', '-') else '+'
                    if collapse:
                        exons = self._collapse_exons(transcripts)
                        recs.append({
                            's': int(gs), 'e': int(ge),
                            'n': str(gname), 'strand': strand, 'exons': exons,
                        })
                    else:
                        for ti, tex in enumerate(transcripts):
                            exons = [[int(s), int(e)] for s, e in tex]
                            # transcript extent may be narrower than gene extent
                            ts = min((s for s, _ in exons), default=int(gs))
                            te = max((e for _, e in exons), default=int(ge))
                            suffix = f' t{ti+1}' if len(transcripts) > 1 else ''
                            recs.append({
                                's': int(ts), 'e': int(te),
                                'n': f'{gname}{suffix}', 'strand': strand,
                                'exons': exons,
                            })
                gdata[chrom] = recs
        else:
            # DataFrame form (legacy API)
            genes_df = genes_data
            for chrom_val, cdf in genes_df.groupby('chrom'):
                chrom = str(chrom_val)
                recs = []
                for _, row in cdf.iterrows():
                    gname  = str(row.get('name',   ''))
                    strand = str(row.get('strand', '+'))
                    exons: List = []
                    if exons_df is not None:
                        ecol = 'gene_name' if 'gene_name' in exons_df.columns else 'name'
                        mask = (
                            (exons_df['chrom'] == chrom_val) &
                            (exons_df[ecol]   == row.get('name', ''))
                        )
                        exons = [
                            [int(r['start']), int(r['end'])]
                            for _, r in exons_df[mask].iterrows()
                        ]
                    recs.append({
                        's': int(row['start']), 'e': int(row['end']),
                        'n': gname, 'strand': strand, 'exons': exons,
                    })
                gdata[chrom] = recs

        # Pack all genes (both strands) into the minimum number of non-overlapping
        # rows with a greedy interval lane-packing. Strand is preserved on each
        # record and encoded via arrow direction at draw time.
        max_rows = 1
        rows_per_chrom: Dict[str, int] = {}
        for chrom, recs in gdata.items():
            n_lanes = self._assign_lanes(recs, offset=0)
            n = max(1, n_lanes)
            rows_per_chrom[chrom] = n
            max_rows = max(max_rows, n)

        # Emit data as {chrom: {'records': [...], 'rows': n}} so the JS can
        # size rows for the displayed chromosome only (avoids empty space when
        # one chrom needs more rows than the currently viewed one).
        gdata_out = {ch: {'records': recs, 'rows': rows_per_chrom[ch]}
                     for ch, recs in gdata.items()}

        if height is None:
            # Per row: ~9 px label band + ~11 px exon + ~8 px bottom gap so
            # the next row's label cannot touch the previous row's exon.
            height = max(28, max_rows * 28 + 2)

        cfg = {
            'id': tid, 'type': 'gene', 'name': name,
            'height': height, 'color': color,
            'rows': max_rows,
            'groups': [],
            **(({'tipFmt': tip_fmt} if tip_fmt is not None else {})),
            'tipLabel': tip_label if tip_label is not None else f'{name}:',
        }
        self.track_data    = {**self.track_data, tid: gdata_out}
        self.track_configs = [*self.track_configs, cfg]
        return self

    @staticmethod
    def _assign_lanes(recs: List, offset: int = 0) -> int:
        """Greedy interval lane-packing. Mutates each rec to add 'row' = offset+lane.
        Returns the number of lanes used."""
        if not recs:
            return 0
        lanes: List[int] = []  # last-used end for each lane
        for r in sorted(recs, key=lambda x: (x.get('s', 0), x.get('e', 0))):
            placed = False
            for li, last_end in enumerate(lanes):
                if r['s'] >= last_end:
                    lanes[li] = r['e']
                    r['row'] = offset + li
                    placed = True
                    break
            if not placed:
                r['row'] = offset + len(lanes)
                lanes.append(r['e'])
        return len(lanes)

    # ── add_scatter_track ────────────────────────────────────────────────────
    def add_scatter_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        x: str = 'pos',
        y: str = 'value',
        group_by: Optional[str] = None,
        color_map: Optional[Dict] = None,
        height: int = 60,
        y_range: Optional[Tuple[float, float]] = None,
        point_size: int = 3,
        tip_fmt: Optional[str] = None,
        tip_label: Optional[str] = None,
    ) -> 'SegmentViewer':
        """
        Scatter track — point cloud.

        Parameters
        ----------
        df         : DataFrame with [chrom, x, y, ...].
        name       : Track display label.
        x          : Column for genomic position.
        y          : Column for y-axis value.
        group_by   : Column whose unique values become separate groups.
        color_map  : {group_value: '#rrggbb'}.
        height     : Track height in CSS px.
        y_range    : (yMin, yMax) — auto-computed from data if None.
        point_size : Point diameter in CSS px.
        tip_fmt    : Python format string for tooltip.
                     Available keys: ``{group}``, ``{value}``, ``{x}``.
        """
        return self._add_xy_track(df, name, 'scatter', x, y, group_by,
                                  color_map, height, y_range, point_size=point_size,
                                  tip_fmt=tip_fmt, tip_label=tip_label)

    # ── add_line_track ───────────────────────────────────────────────────────
    def add_line_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        x: str = 'pos',
        y: str = 'value',
        group_by: Optional[str] = None,
        color_map: Optional[Dict] = None,
        height: int = 60,
        y_range: Optional[Tuple[float, float]] = None,
        step: Optional[str] = None,
        tip_fmt: Optional[str] = None,
        tip_label: Optional[str] = None,
    ) -> 'SegmentViewer':
        """
        Line track — connected line per group.

        Parameters
        ----------
        df       : DataFrame with [chrom, x, y, ...].
        name     : Track display label.
        x        : Column for genomic position.
        y        : Column for y-axis value.
        group_by : Column whose unique values become separate groups.
        color_map: {group_value: '#rrggbb'}.
        height   : Track height in CSS px.
        y_range  : (yMin, yMax) — auto-computed from data if None.
        step     : Step mode for staircase line, matching matplotlib's
                   ``step`` parameter.  One of ``'pre'``, ``'post'``, or
                   ``'mid'``.  None (default) gives smooth interpolation.
        tip_fmt  : Python format string for tooltip.
                   Available keys: ``{group}``, ``{value}``, ``{x}``.
        """
        return self._add_xy_track(df, name, 'line', x, y, group_by,
                                  color_map, height, y_range, tip_fmt=tip_fmt,
                                  tip_label=tip_label, step=step)

    def _add_xy_track(self, df, name, track_type, x, y, group_by,
                      color_map, height, y_range, point_size=3, tip_fmt=None, tip_label=None, step=None):
        """Shared implementation for scatter and line tracks."""
        tid = self._tid()
        if group_by and group_by in df.columns:
            groups = sorted(df[group_by].dropna().unique(), key=str)
        else:
            groups = ['all']
            group_by = None

        if color_map is None:
            color_map = {g: self._PALETTE[i % len(self._PALETTE)]
                         for i, g in enumerate(groups)}

        # Compute y range
        if y_range is not None:
            yMin, yMax = y_range
        else:
            yMin = float(df[y].min())
            yMax = float(df[y].max())
            pad = (yMax - yMin) * 0.05 or 0.5
            yMin -= pad
            yMax += pad

        xy_out: Dict = {}
        for chrom_val, cdf in df.groupby('chrom'):
            chrom = str(chrom_val)
            xy_out[chrom] = {}
            for gi, group in enumerate(groups):
                gid = str(gi)
                gdf = cdf[cdf[group_by] == group] if group_by else cdf
                if gdf.empty:
                    xy_out[chrom][gid] = ''
                    continue
                gdf = gdf.sort_values(x)
                n = len(gdf)
                arr = np.empty(n * 2, dtype=np.float32)
                arr[0::2] = gdf[x].to_numpy(dtype=np.float32)
                arr[1::2] = gdf[y].to_numpy(dtype=np.float32)
                if step and n > 1:
                    xs = arr[0::2]; ys = arr[1::2]
                    if step == 'post':
                        m = 2 * n - 1
                        sx = np.empty(m, dtype=np.float32)
                        sy = np.empty(m, dtype=np.float32)
                        sx[0::2] = xs;  sx[1::2] = xs[1:]
                        sy[0::2] = ys;  sy[1::2] = ys[:-1]
                    elif step == 'pre':
                        m = 2 * n - 1
                        sx = np.empty(m, dtype=np.float32)
                        sy = np.empty(m, dtype=np.float32)
                        sx[0::2] = xs;  sx[1::2] = xs[:-1]
                        sy[0::2] = ys;  sy[1::2] = ys[1:]
                    elif step == 'mid':
                        mids = (xs[:-1] + xs[1:]) / 2
                        m = 2 * n
                        sx = np.empty(m, dtype=np.float32)
                        sy = np.empty(m, dtype=np.float32)
                        sx[0::2] = np.concatenate([[xs[0]], mids])
                        sx[1::2] = np.concatenate([mids, [xs[-1]]])
                        sy[0::2] = ys;  sy[1::2] = ys
                    else:
                        raise ValueError(f"step must be 'pre', 'post', or 'mid', got {step!r}")
                    arr = np.empty(m * 2, dtype=np.float32)
                    arr[0::2] = sx; arr[1::2] = sy
                xy_out[chrom][gid] = self._pack_f32(arr)

        cfg = {
            'id': tid, 'type': track_type, 'name': name, 'height': height,
            'yMin': yMin, 'yMax': yMax, 'pointSize': point_size,
            'groups': [
                {'id': str(i), 'name': str(g),
                 'color': color_map.get(g, self._PALETTE[i % len(self._PALETTE)])}
                for i, g in enumerate(groups)
            ],
            **(({'tipFmt': tip_fmt} if tip_fmt is not None else {})),
            'tipLabel': tip_label if tip_label is not None else f'{name}:',
        }
        self.track_data    = {**self.track_data, tid: xy_out}
        self.track_configs = [*self.track_configs, cfg]
        return self

    # ── add_fill_track ───────────────────────────────────────────────────────
    def add_fill_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        x: str = 'pos',
        y: Optional[str] = None,
        y_lo: Optional[str] = None,
        y_hi: Optional[str] = None,
        group_by: Optional[str] = None,
        color_map: Optional[Dict] = None,
        color_pos: Optional[str] = None,
        color_neg: Optional[str] = None,
        height: int = 60,
        y_range: Optional[Tuple[float, float]] = None,
        baseline: Optional[float] = None,
        step: Optional[str] = None,
        tip_fmt: Optional[str] = None,
        tip_label: Optional[str] = None,
    ) -> 'SegmentViewer':
        """
        Fill-between track — filled area between two curves or from a
        single curve to a baseline.

        Supports different colors above and below a baseline.

        Either supply ``y`` (fill from curve to baseline) or both
        ``y_lo`` and ``y_hi`` (fill between two curves).

        Parameters
        ----------
        df        : DataFrame with [chrom, x, ...] plus the y column(s).
        name      : Track display label.
        x         : Column for genomic position.
        y         : Column for a single y value.  Fill extends from this
                    curve to *baseline*.  Mutually exclusive with y_lo/y_hi.
        y_lo      : Column for lower y boundary.
        y_hi      : Column for upper y boundary.
        group_by  : Column whose unique values become separate groups.
        color_map : {group_value: '#rrggbb'} — used for both pos/neg if
                    color_pos/color_neg not specified.
        color_pos : Colour for region above baseline.
        color_neg : Colour for region below baseline.
        height    : Track height in CSS px.
        y_range   : (yMin, yMax) — auto-computed from data if None.
        baseline  : Y value at which pos/neg colours switch.  Defaults to
                    0 when *y* is used, or 0.0 when *y_lo*/*y_hi* are used.
        step      : Step mode for staircase fill, matching matplotlib's
                    ``fill_between(step=...)``.  One of ``'pre'``,
                    ``'post'``, or ``'mid'``.  None (default) gives smooth
                    linear interpolation.
        tip_fmt   : Python format string for tooltip.
                    Available keys: ``{group}``, ``{lo}``, ``{hi}``, ``{x}``.
        """
        # Resolve single-y vs lo/hi mode
        if y is not None and (y_lo is not None or y_hi is not None):
            raise ValueError("Specify either 'y' or 'y_lo'/'y_hi', not both.")
        if y is not None:
            single_y = True
            if baseline is None:
                baseline = 0
        else:
            single_y = False
            if y_lo is None:
                y_lo = 'lo'
            if y_hi is None:
                y_hi = 'hi'
            if baseline is None:
                baseline = 0.0

        tid = self._tid()
        if group_by and group_by in df.columns:
            groups = sorted(df[group_by].dropna().unique(), key=str)
        else:
            groups = ['all']
            group_by = None

        if color_map is None:
            color_map = {g: self._PALETTE[i % len(self._PALETTE)]
                         for i, g in enumerate(groups)}

        if y_range is not None:
            yMin, yMax = y_range
        elif single_y:
            yMin = float(min(df[y].min(), baseline))
            yMax = float(max(df[y].max(), baseline))
            pad = (yMax - yMin) * 0.05 or 0.5
            yMin -= pad
            yMax += pad
        else:
            yMin = float(min(df[y_lo].min(), df[y_hi].min()))
            yMax = float(max(df[y_lo].max(), df[y_hi].max()))
            pad = (yMax - yMin) * 0.05 or 0.5
            yMin -= pad
            yMax += pad

        fill_out: Dict = {}
        for chrom_val, cdf in df.groupby('chrom'):
            chrom = str(chrom_val)
            fill_out[chrom] = {}
            for gi, group in enumerate(groups):
                gid = str(gi)
                gdf = cdf[cdf[group_by] == group] if group_by else cdf
                if gdf.empty:
                    fill_out[chrom][gid] = ''
                    continue
                gdf = gdf.sort_values(x)
                n = len(gdf)
                arr = np.empty(n * 3, dtype=np.float32)
                arr[0::3] = gdf[x].to_numpy(dtype=np.float32)
                if single_y:
                    yvals = gdf[y].to_numpy(dtype=np.float32)
                    arr[1::3] = np.minimum(yvals, baseline)
                    arr[2::3] = np.maximum(yvals, baseline)
                else:
                    arr[1::3] = gdf[y_lo].to_numpy(dtype=np.float32)
                    arr[2::3] = gdf[y_hi].to_numpy(dtype=np.float32)
                if step and n > 1:
                    xs = arr[0::3]; los = arr[1::3]; his = arr[2::3]
                    if step == 'post':
                        m = 2 * n - 1
                        sx  = np.empty(m, dtype=np.float32)
                        slo = np.empty(m, dtype=np.float32)
                        shi = np.empty(m, dtype=np.float32)
                        sx[0::2] = xs;    sx[1::2] = xs[1:]
                        slo[0::2] = los;  slo[1::2] = los[:-1]
                        shi[0::2] = his;  shi[1::2] = his[:-1]
                    elif step == 'pre':
                        m = 2 * n - 1
                        sx  = np.empty(m, dtype=np.float32)
                        slo = np.empty(m, dtype=np.float32)
                        shi = np.empty(m, dtype=np.float32)
                        sx[0::2] = xs;    sx[1::2] = xs[:-1]
                        slo[0::2] = los;  slo[1::2] = los[1:]
                        shi[0::2] = his;  shi[1::2] = his[1:]
                    elif step == 'mid':
                        mids = (xs[:-1] + xs[1:]) / 2
                        m = 2 * n
                        sx  = np.empty(m, dtype=np.float32)
                        slo = np.empty(m, dtype=np.float32)
                        shi = np.empty(m, dtype=np.float32)
                        sx[0::2] = np.concatenate([[xs[0]], mids])
                        sx[1::2] = np.concatenate([mids, [xs[-1]]])
                        slo[0::2] = los;  slo[1::2] = los
                        shi[0::2] = his;  shi[1::2] = his
                    else:
                        raise ValueError(f"step must be 'pre', 'post', or 'mid', got {step!r}")
                    arr = np.empty(m * 3, dtype=np.float32)
                    arr[0::3] = sx; arr[1::3] = slo; arr[2::3] = shi
                fill_out[chrom][gid] = self._pack_f32(arr)

        cfg = {
            'id': tid, 'type': 'fill', 'name': name, 'height': height,
            'yMin': yMin, 'yMax': yMax, 'baseline': baseline,
            'groups': [
                {'id': str(i), 'name': str(g),
                 'color': color_map.get(g, self._PALETTE[i % len(self._PALETTE)]),
                 'colorPos': color_pos or color_map.get(g, self._PALETTE[i % len(self._PALETTE)]),
                 'colorNeg': color_neg or '#888888'}
                for i, g in enumerate(groups)
            ],
            **(({'tipFmt': tip_fmt} if tip_fmt is not None else {})),
            'tipLabel': tip_label if tip_label is not None else f'{name}:',
        }
        self.track_data    = {**self.track_data, tid: fill_out}
        self.track_configs = [*self.track_configs, cfg]
        return self

    # ── add_histogram_track ──────────────────────────────────────────────────
    def add_histogram_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        x: str = 'pos',
        y: str = 'value',
        group_by: Optional[str] = None,
        color_map: Optional[Dict] = None,
        height: int = 60,
        y_range: Optional[Tuple[float, float]] = None,
        bin_width: Optional[float] = None,
        tip_fmt: Optional[str] = None,
        tip_label: Optional[str] = None,
    ) -> 'SegmentViewer':
        """
        Histogram track — vertical bars.

        Parameters
        ----------
        df        : DataFrame with [chrom, x, y, ...].
        name      : Track display label.
        x         : Column for genomic position (bin center).
        y         : Column for bar height.
        group_by  : Column whose unique values become separate groups.
        color_map : {group_value: '#rrggbb'}.
        height    : Track height in CSS px.
        y_range   : (yMin, yMax) — auto-computed from data if None.
        bin_width : Width of each bar in bp.  Auto-computed from data spacing
                    if None.
        tip_fmt   : Python format string for tooltip.
                    Available keys: ``{group}``, ``{value}``, ``{x}``.
        """
        tid = self._tid()
        if group_by and group_by in df.columns:
            groups = sorted(df[group_by].dropna().unique(), key=str)
        else:
            groups = ['all']
            group_by = None

        if color_map is None:
            color_map = {g: self._PALETTE[i % len(self._PALETTE)]
                         for i, g in enumerate(groups)}

        if y_range is not None:
            yMin, yMax = y_range
        else:
            yMin = min(0, float(df[y].min()))
            yMax = float(df[y].max())
            pad = (yMax - yMin) * 0.05 or 0.5
            yMax += pad

        # Auto bin width from median spacing
        if bin_width is None:
            xs = df.sort_values(x)[x].to_numpy()
            if len(xs) > 1:
                diffs = np.diff(xs)
                bin_width = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
            else:
                bin_width = 1.0

        hist_out: Dict = {}
        for chrom_val, cdf in df.groupby('chrom'):
            chrom = str(chrom_val)
            hist_out[chrom] = {}
            for gi, group in enumerate(groups):
                gid = str(gi)
                gdf = cdf[cdf[group_by] == group] if group_by else cdf
                if gdf.empty:
                    hist_out[chrom][gid] = ''
                    continue
                gdf = gdf.sort_values(x)
                arr = np.empty(len(gdf) * 2, dtype=np.float32)
                arr[0::2] = gdf[x].to_numpy(dtype=np.float32)
                arr[1::2] = gdf[y].to_numpy(dtype=np.float32)
                hist_out[chrom][gid] = self._pack_f32(arr)

        cfg = {
            'id': tid, 'type': 'histogram', 'name': name, 'height': height,
            'yMin': yMin, 'yMax': yMax, 'binWidth': bin_width,
            'groups': [
                {'id': str(i), 'name': str(g),
                 'color': color_map.get(g, self._PALETTE[i % len(self._PALETTE)])}
                for i, g in enumerate(groups)
            ],
            **(({'tipFmt': tip_fmt} if tip_fmt is not None else {})),
            'tipLabel': tip_label if tip_label is not None else f'{name}:',
        }
        self.track_data    = {**self.track_data, tid: hist_out}
        self.track_configs = [*self.track_configs, cfg]
        return self

    # ── Navigation ───────────────────────────────────────────────────────────
    def set_viewport(self, chrom: str, start: int, end: int) -> 'SegmentViewer':
        csz = self.chrom_sizes.get(str(chrom), end)
        self.viewport = {
            'chrom': str(chrom),
            'start': int(max(0, start)),
            'end':   int(min(csz, end)),
        }
        return self

    def zoom_to(self, chrom: str, center: int, window: int = 1_000_000) -> 'SegmentViewer':
        """Centre view on *center* ± *window*/2 bp."""
        half = window // 2
        return self.set_viewport(chrom, center - half, center + half)
