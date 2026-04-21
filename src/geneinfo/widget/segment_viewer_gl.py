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
    viewer = Tracks(chrom_sizes)
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
from typing import Any
import anywidget
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import traitlets


# ─────────────────────────────────────────────────────────────────────────────
# Colour resolution
# ─────────────────────────────────────────────────────────────────────────────
def resolve_color(c: Any) -> str | None:
    """Resolve a matplotlib-style colour spec to a CSS-compatible hex string.

    Parameters
    ----------
    c : Any
        Anything matplotlib understands: a named colour (``'red'``), a
        cycle reference (``'C0'``..``'C9'``), a hex string (``'#rrggbb'``
        or ``'#rrggbbaa'``), an ``(r, g, b)`` or ``(r, g, b, a)`` tuple in
        ``[0, 1]``, or ``None``.

    Returns
    -------
    str or None
        Hex string. ``#RRGGBBAA`` when the input carries an alpha
        channel, ``#RRGGBB`` otherwise. ``None`` is passed through.
    """
    if c is None:
        return None
    # Resolve via matplotlib → RGBA tuple → hex. keep_alpha=True retains the
    # alpha channel when the input carries one (e.g. '#rrggbbaa' or a 4-tuple).
    rgba = mcolors.to_rgba(c)
    return mcolors.to_hex(rgba, keep_alpha=True)


def _split_alpha(spec: Any) -> tuple[str | None, float | None]:
    """Split a matplotlib colour spec into ``(opaque_hex, alpha_or_None)``.

    Mirrors matplotlib's compositional ``color`` + ``alpha`` semantics: a
    spec that carries alpha (``'#rrggbbaa'`` or a 4-tuple) returns the
    opaque RGB hex and the alpha as a float; a fully opaque spec returns
    ``alpha=None`` so the caller can keep its own default.

    Parameters
    ----------
    spec : Any
        Any matplotlib colour spec, or ``None``.

    Returns
    -------
    tuple of (str or None, float or None)
        ``(opaque_hex, alpha)``. ``alpha`` is ``None`` when the spec is
        fully opaque or ``None``; otherwise a float in ``[0, 1)``.
    """
    if spec is None:
        return None, None
    rgba = mcolors.to_rgba(spec)
    a = float(rgba[3])
    rgb_hex = mcolors.to_hex(rgba, keep_alpha=False)
    return rgb_hex, (None if a == 1.0 else a)


def _resolve_color_mapping(m: Any) -> Any:
    """Resolve colours in a dict or list, returning a new container.

    Parameters
    ----------
    m : Any
        A colour spec, a list of specs, a dict whose values are specs (or
        nested dicts/lists), or ``None``.

    Returns
    -------
    Any
        A new container of the same shape with each string value passed
        through :func:`resolve_color`. Non-string scalars (e.g. numeric
        layout knobs like ``sidebar_w``) are passed through unchanged so
        theme dicts can carry mixed value types. ``None`` returns ``None``.
    """
    if m is None:
        return None
    if isinstance(m, dict):
        out = {}
        for k, v in m.items():
            if isinstance(v, (dict, list)):
                out[k] = _resolve_color_mapping(v)
            elif isinstance(v, str):
                out[k] = resolve_color(v)
            else:
                out[k] = v
        return out
    if isinstance(m, list):
        return [resolve_color(v) if isinstance(v, str) else v for v in m]
    if isinstance(m, str):
        return resolve_color(m)
    return m


def _build_heatmap_lut(
    values: pd.Series,
    palette: Any,
    vmin: float | None,
    vmax: float | None,
) -> dict:
    """Materialise a per-segment colouring spec into a ready-to-upload LUT.

    The R8 heatmap texture carries a single byte per (individual, window)
    cell. Byte 0 is reserved as a "no data" sentinel; values 1..255 index
    into a companion RGB lookup table. Continuous mode maps a scalar
    column to 1..255 via ``vmin/vmax`` normalisation; discrete mode maps
    each unique category to an index in 1..K.

    Parameters
    ----------
    values : pandas.Series
        The column named by ``value_col``; used to infer ``vmin/vmax``
        (continuous) or the category set (discrete).
    palette : str | list | dict | matplotlib.colors.Colormap
        Continuous: a named colormap (``'viridis'``) or a ``Colormap``
        instance. Discrete: a dict ``{category: color}`` or a list of
        colours (paired with sorted unique categories).
    vmin, vmax : float, optional
        Continuous-mode normalisation range. Auto-inferred from
        ``values`` when either is ``None``.

    Returns
    -------
    dict
        Keys: ``mode`` (``'discrete' | 'continuous'``), ``lut_u8``
        (``(N, 3) uint8``), ``cat_to_idx`` (discrete only;
        ``{category: 1-based-index}``), ``vmin``, ``vmax``, and
        ``palette_ref`` (the resolved ``Colormap`` for continuous, the
        ``{cat: hex}`` dict for discrete — used by ``Tracks.colorbar``).
    """
    import matplotlib.cm as mcm

    is_dict_palette = isinstance(palette, dict)
    is_listlike_palette = isinstance(palette, (list, tuple))
    is_cmap_name = isinstance(palette, str)
    is_cmap_obj = isinstance(palette, mcolors.Colormap)

    # Discrete path: dict or list/tuple of colours.
    if is_dict_palette or is_listlike_palette:
        if is_dict_palette:
            cats = list(palette.keys())
            colors = [palette[c] for c in cats]
        else:
            cats = sorted(pd.Series(values).dropna().unique().tolist())
            if len(palette) < len(cats):
                raise ValueError(
                    f"palette has {len(palette)} colours but value_col has "
                    f"{len(cats)} categories"
                )
            colors = list(palette[: len(cats)])
        if len(cats) > 255:
            raise ValueError(
                f"discrete palette capped at 255 categories, got {len(cats)}"
            )
        rgb = np.array(
            [mcolors.to_rgb(resolve_color(c)) for c in colors], dtype=np.float64,
        )
        rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        # Always emit a 256-row LUT (index 0 = "no data", 1..K = categories,
        # K+1..255 = unused zero-filled). This keeps the JS shader's
        # ``(idx + 0.5) / 256.0`` lookup uniform across modes.
        lut_u8 = np.zeros((256, 3), dtype=np.uint8)
        lut_u8[1:1 + len(cats), :] = rgb_u8
        cat_to_idx = {c: i + 1 for i, c in enumerate(cats)}
        palette_ref = {c: mcolors.to_hex(rgb[i]) for i, c in enumerate(cats)}
        return {
            'mode':        'discrete',
            'lut_u8':      lut_u8,
            'cat_to_idx':  cat_to_idx,
            'vmin':        None,
            'vmax':        None,
            'palette_ref': palette_ref,
        }

    # Continuous path: colormap name or Colormap instance.
    if is_cmap_name or is_cmap_obj:
        cmap = mcm.get_cmap(palette) if is_cmap_name else palette
        numeric = pd.to_numeric(values, errors='coerce').dropna()
        if numeric.empty:
            raise ValueError("value_col has no numeric data for continuous palette")
        v_lo = float(numeric.min()) if vmin is None else float(vmin)
        v_hi = float(numeric.max()) if vmax is None else float(vmax)
        if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
            v_hi = v_lo + 1.0
        # 255 usable slots (index 0 is "no data"); sample cmap at bin centres.
        xs = (np.arange(255) + 0.5) / 255.0
        rgba = cmap(xs)
        lut_u8 = np.zeros((256, 3), dtype=np.uint8)
        lut_u8[1:, :] = np.clip(np.round(rgba[:, :3] * 255.0), 0, 255).astype(np.uint8)
        return {
            'mode':        'continuous',
            'lut_u8':      lut_u8,
            'cat_to_idx':  None,
            'vmin':        v_lo,
            'vmax':        v_hi,
            'palette_ref': cmap,
        }

    raise TypeError(
        f"palette must be a colormap name, Colormap, dict, or list; got {type(palette).__name__}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Built-in themes
# ─────────────────────────────────────────────────────────────────────────────
DARK_THEME: dict[str, Any] = {
    'bg':           '#1F1F1F',
    'fg':           '#f2f2f8',
    'panel':        '#1F1F1F',
    'border':       '#1F1F1F',
    'input_bg':     '#2f2f2f',
    'input_fg':     '#f2f2f8',
    'input_border': '#4a4a55',
    'focus_border': '#88a0ff',
    'axis_text':    '#9090a0',
    'track_label':  '#c0c0d0',
    'gene_exon':    '#88c0ee',
    'gene_spine':   '#a8a8c0',
    'gene_label':   '#d0d0e0',
    # Lightened highlight palette — sibling hues to the light-theme values
    # but pushed up in lightness so they read against a dark background.
    'highlight_fill':    '#ff6d7f',
    'highlight_spine':   '#ee66f2',
    'highlight_outline': '#ffffff',
    'highlight_label':   '#5fd88a',
    'highlight_halo':    '#79CAD32E',
    # Layout
    'sidebar_w':    96,    # left label column width, CSS px
}

LIGHT_THEME: dict[str, Any] = {
    'bg':           '#FAF9F6',
    'fg':           '#1a1a2e',
    'panel':        '#FAF9F6',
    'border':       '#FAF9F6',
    'input_bg':     '#ffffff',
    'input_fg':     '#1a1a2e',
    'input_border': '#c0c0cc',
    'focus_border': '#3355dd',
    'axis_text':    '#888899',
    'track_label':  '#555566',
    'gene_exon':    '#4488cc',
    'gene_spine':   '#666688',
    'gene_label':   '#000000',
    'highlight_fill':    '#e03a4e',
    'highlight_spine':   '#d80ce6',
    'highlight_outline': '#000000',
    'highlight_label':   '#169f4a',
    'highlight_halo':    '#79CAD32E',
    # Layout
    'sidebar_w':    96,    # left label column width, CSS px
}


def _detect_default_theme() -> dict[str, Any]:
    """Auto-detect a sensible default theme.

    Returns
    -------
    dict of str to Any
        :data:`DARK_THEME` or :data:`LIGHT_THEME`. If
        ``vscodenb.is_vscode_dark_theme`` is importable, its verdict
        decides; otherwise dark is returned.
    """
    try:
        from vscodenb import is_vscode_dark_theme  # type: ignore
        is_dark, _ = is_vscode_dark_theme()
        return DARK_THEME if is_dark else LIGHT_THEME
    except Exception:
        return DARK_THEME


_default_theme: dict[str, Any] = _detect_default_theme()


def set_default_theme(theme: str | dict[str, Any]) -> None:
    """Set the theme used by newly created :class:`Tracks` instances.

    Parameters
    ----------
    theme : str or dict of str to Any
        One of:

        * ``'dark'``  — use the built-in dark theme.
        * ``'light'`` — use the built-in light theme.
        * ``'auto'``  — re-run the vscodenb-based auto-detection.
        * a dict — a custom theme, merged onto the dark theme so missing
          keys fall back to sane defaults. Mixed value types are allowed
          (colour strings plus numeric layout knobs like ``sidebar_w``).

    Raises
    ------
    ValueError
        If ``theme`` is a string other than ``'dark'``, ``'light'``, or
        ``'auto'``.
    TypeError
        If ``theme`` is neither a string nor a dict.
    """
    global _default_theme
    if isinstance(theme, str):
        key = theme.lower()
        if key == 'dark':
            _default_theme = dict(DARK_THEME)
        elif key == 'light':
            _default_theme = dict(LIGHT_THEME)
        elif key == 'auto':
            _default_theme = _detect_default_theme()
        else:
            raise ValueError(
                f"Unknown theme name {theme!r}; use 'dark', 'light', 'auto', "
                f"or pass a dict."
            )
    elif isinstance(theme, dict):
        _default_theme = {**DARK_THEME, **_resolve_color_mapping(theme)}
    else:
        raise TypeError(f"theme must be a str or dict, got {type(theme).__name__}")


def get_default_theme() -> dict[str, Any]:
    """Return a copy of the currently active default theme.

    Returns
    -------
    dict of str to Any
        A shallow copy of the module-level default theme. Mutating the
        returned dict does not affect the module state; pass it to
        :func:`set_default_theme` to install changes.
    """
    return dict(_default_theme)


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
.sv-label-sm { font-size: 10px; color: var(--sv-axis-text); letter-spacing: 0.08em; text-transform: uppercase; }
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
.sv-toolbar select:focus, .sv-toolbar input:focus { border-color: var(--sv-focus-border); }
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
.sv-btn:hover { background: var(--sv-panel); border-color: var(--sv-focus-border); }
.sv-sep { flex: 1; }
.sv-lod-badge {
    font-size: 9px;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--sv-bg);
    border: 1px solid var(--sv-input-border);
    color: var(--sv-axis-text);
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
let   LABEL_W    = 96;    // left label column, CSS px (theme: sidebar_w)
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
    <button class="sv-btn sv-pff" title="Pan 90% of the view left">&laquo;</button>
    <button class="sv-btn sv-phf" title="Pan half a view left">&lsaquo;</button>
    <button class="sv-btn sv-phr" title="Pan half a view right">&rsaquo;</button>
    <button class="sv-btn sv-pfr" title="Pan 90% of the view right">&raquo;</button>
    <button class="sv-btn sv-rs" title="Reset view">⌂</button>
    <button class="sv-btn sv-hmr" title="Recompute heatmap(s) for current view" style="display:none">⟲</button>
    <button class="sv-btn sv-hmg" title="Restore global heatmap view" style="display:none">◱</button>
    <button class="sv-btn sv-gl"  title="Show all gene labels (when &lt;100 visible)" style="display:none">A</button>
    <button class="sv-btn sv-snap" title="Copy current view to clipboard">⧉</button>
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
const panFFBtn  = el.querySelector('.sv-pff');
const panHFBtn  = el.querySelector('.sv-phf');
const panHRBtn  = el.querySelector('.sv-phr');
const panFRBtn  = el.querySelector('.sv-pfr');
const hmRecBtn  = el.querySelector('.sv-hmr');
const hmGlobBtn = el.querySelector('.sv-hmg');
const geneLblBtn= el.querySelector('.sv-gl');
const snapBtn   = el.querySelector('.sv-snap');
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
  root.setProperty('--sv-focus-border', t.focus_border || '#4466ee');
  root.setProperty('--sv-axis-text',    t.axis_text    || '#666688');
  root.setProperty('--sv-track-label',  t.track_label  || '#9090c0');
  // Layout values (numeric, not CSS colours).
  const w = Number(t.sidebar_w);
  LABEL_W = Number.isFinite(w) && w > 0 ? w : 96;
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
uniform float uVS, uVE, uTT, uTB, uXL, uMinDx;
flat out vec3 vColor;
void main() {
  // Expand bars to a minimum width (in genomic units) about their centre so
  // sub-pixel bars don't vanish on raster snap when zoomed out.
  float c   = 0.5 * (iStart + iEnd);
  float h   = max(0.5 * (iEnd - iStart), 0.5 * uMinDx);
  float s   = c - h;
  float e   = c + h;
  float gx  = mix(s, e, aCorner.x);
  float t   = (gx - uVS) / (uVE - uVS);
  float xN  = clamp(mix(uXL, 1.0, t), -1.0, 1.0);
  // iYLo / iYHi are fractional positions in the track box where 0 = top
  // and 1 = bottom (matches the segment-track convention).
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
precision highp float;
uniform sampler2D uTex;
uniform vec3  uBg, uFg;
uniform float uNWin;
in vec2 vTex;
out vec4 fragColor;
void main() {
  // When nWin > plotPx, a single fragment covers many texels on the X axis.
  // Hardware bilinear (or nearest) sampling would shimmer as the sampled pair
  // changes with sub-pixel pan; instead, we explicitly box-average across the
  // texel span this fragment covers. Y keeps nearest sampling so individual
  // rows remain crisp.
  float dUdx = abs(dFdx(vTex.x));           // tex-space width of one fragment
  float span = dUdx * uNWin;                // in texel units
  int N = int(min(span, 64.0)) + 1;         // sample count; cap for safety
  float v = 0.0;
  // Centre-aligned samples that evenly cover [vTex.x - dUdx/2, vTex.x + dUdx/2].
  float step = dUdx / float(N);
  float x0   = vTex.x - dUdx * 0.5 + step * 0.5;
  for (int i = 0; i < 64; i++) {
    if (i >= N) break;
    float u = x0 + step * float(i);
    v += textureLod(uTex, vec2(u, vTex.y), 0.0).r;
  }
  v /= float(N);
  fragColor = vec4(mix(uBg, uFg, v), 1.0);
}`;

// Value-mode heatmap: R8 texel is an index (0 = no data, 1..255 = palette slot)
// looked up in a 1-D RGB LUT. Fragment "averaging" here is a *nearest* pick at
// the fragment centre — averaging across category indices would blend colour
// identities, which is meaningless for discrete palettes and only weakly useful
// for continuous ones. Clients can increase ``windows`` when they want finer
// resolution; the binning already applies last-write-wins within a bin.
const FS_TEX_LUT = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform sampler2D uLUT;
uniform vec3  uBg;
in vec2 vTex;
out vec4 fragColor;
void main() {
  float raw = textureLod(uTex, vTex, 0.0).r;   // 0..1 (byte / 255)
  int idx = int(raw * 255.0 + 0.5);
  if (idx == 0) { fragColor = vec4(uBg, 1.0); return; }
  // Sample the LUT at the texel centre of the palette slot.
  float u = (float(idx) + 0.5) / 256.0;
  vec3 c = textureLod(uLUT, vec2(u, 0.5), 0.0).rgb;
  fragColor = vec4(c, 1.0);
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
const rectProg    = makeProgram(VS_RECT, FS_RECT);
const densProg    = makeProgram(VS_DENS, FS_DENS);
const texProg     = makeProgram(VS_TEX,  FS_TEX);
const texProgLUT  = makeProgram(VS_TEX,  FS_TEX_LUT);

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
  uMinDx:  gl.getUniformLocation(rectProg, 'uMinDx'),
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
  aPos:  gl.getAttribLocation (texProg, 'aPos'),
  aTex:  gl.getAttribLocation (texProg, 'aTex'),
  uTex:  gl.getUniformLocation(texProg, 'uTex'),
  uBg:   gl.getUniformLocation(texProg, 'uBg'),
  uFg:   gl.getUniformLocation(texProg, 'uFg'),
  uNWin: gl.getUniformLocation(texProg, 'uNWin'),
};

// texture-LUT program locations (value-mode heatmap)
const tLocLUT = {
  aPos: gl.getAttribLocation (texProgLUT, 'aPos'),
  aTex: gl.getAttribLocation (texProgLUT, 'aTex'),
  uTex: gl.getUniformLocation(texProgLUT, 'uTex'),
  uLUT: gl.getUniformLocation(texProgLUT, 'uLUT'),
  uBg:  gl.getUniformLocation(texProgLUT, 'uBg'),
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
const gpuHMLUT = {};  // [tid] = { tex, size } — 1-D palette for value-mode heatmaps
const gpuXY   = {};   // { buf, count }  — for scatter, line
const gpuFill = {};   // { posBuf, posCount, negBuf, negCount }
const gpuHist = {};   // [tid][ch][gid] = { base: rectGpu|null, levels: [{nBins, binWidth, ...rectGpu}] }
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
  // Accept #rgb, #rrggbb, or #rrggbbaa — ignore alpha here, return 0..1 RGB.
  let s = hex.startsWith('#') ? hex.slice(1) : hex;
  if (s.length === 3) s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
  if (s.length === 8) s = s.slice(0, 6);
  const r = parseInt(s.slice(0, 2), 16) / 255;
  const g = parseInt(s.slice(2, 4), 16) / 255;
  const b = parseInt(s.slice(4, 6), 16) / 255;
  return [r, g, b];
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

// Stacked density: input is stride-3 (binIndex, yLo, yHi) per bin in
// normalised [0,1] space.  We emit a triangle strip spanning yLo..yHi for
// each bin so the same drawDensArea shader paints per-group stacked bands.
function buildDensStackedGPU(arr, chromSz, nBins) {
  const n = (arr.length / 3) | 0;
  if (n === 0) return null;
  const winSz = chromSz / nBins;
  const verts = new Float32Array(n * 4 * 2);
  let vi = 0;
  for (let i = 0; i < n; i++) {
    const bi = arr[i * 3];
    const lo = arr[i * 3 + 1];
    const hi = arr[i * 3 + 2];
    const xL = bi * winSz;
    const xR = (bi + 1) * winSz;
    verts[vi++] = xL; verts[vi++] = lo;
    verts[vi++] = xL; verts[vi++] = hi;
    verts[vi++] = xR; verts[vi++] = lo;
    verts[vi++] = xR; verts[vi++] = hi;
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
  // We do our own X-axis box-filter in the fragment shader (see FS_TEX), so
  // keep per-texel reads crisp and rely on NEAREST sampling vertically. Mipmaps
  // would also downsample in the nInd (row) dimension, which is semantically
  // wrong: we want to preserve individuals, not blend them.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { tex, nInd, nWin, xStart, xEnd };
}

// Build palette LUT texture for value-mode heatmap.
// Input: Uint8Array of length size*3 (RGB triples, index 0 is bg/"no data").
function buildHeatmapLUTGPU(rgb, size) {
  if (!rgb || rgb.length === 0) return null;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, size, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, rgb);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { tex, size };
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

// Build histogram bars as instanced rects.
// The shared rect shader treats iYLo / iYHi as fractional positions in the
// track box where 0 = top and 1 = bottom (segment convention). So a bar of
// height `y` with baseline 0 spans from data-fraction `min(0, y)` (bottom of
// bar) to `max(0, y)` (top of bar), then we flip to top-down NDC convention.
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
    const dataLo = Math.min(baseN, hN);  // bottom of bar in data-frac
    const dataHi = Math.max(baseN, hN);  // top    of bar in data-frac
    // Flip to top-down: 0 = top, 1 = bottom.
    const yLo = 1.0 - dataHi;            // top    of bar (NDC convention)
    const yHi = 1.0 - dataLo;            // bottom of bar (NDC convention)
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

// Stacked histogram: input stride is (x, yLo, yHi) — pre-computed in Python
// so each bar spans [yLo, yHi] in data units. No baseline inference needed.
// Same top-down convention flip as buildHistGPU.
function buildHistStackedGPU(data, yMin, yMax, binWidth, color) {
  const n = (data.length / 3) | 0;
  if (n === 0) return null;
  const [r, g, b] = hexRGB(color);
  const range = yMax - yMin || 1;
  const inst   = new Float32Array(n * 7);
  const starts = new Float32Array(n);
  let maxLen = 0;
  const halfW = binWidth / 2;
  for (let i = 0; i < n; i++) {
    const x   = data[i * 3];
    const lo  = data[i * 3 + 1];
    const hi  = data[i * 3 + 2];
    const s = x - halfW;
    const e = x + halfW;
    const loN = (lo - yMin) / range;
    const hiN = (hi - yMin) / range;
    const dataLo = Math.min(loN, hiN);
    const dataHi = Math.max(loN, hiN);
    const yLo = 1.0 - dataHi;  // top    of bar (NDC convention)
    const yHi = 1.0 - dataLo;  // bottom of bar (NDC convention)
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
      const stacked = !!cfg.stacked;
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
            const n = parseInt(nBins);
            const gpu = stacked
              ? buildDensStackedGPU(cnts, csz, n)
              : buildDensGPU(cnts, csz);
            if (gpu) levels.push({ nBins: n, binWidth: csz / n, ...gpu });
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
      // Value-mode heatmap: upload 1-D palette LUT once per track.
      if (cfg.mode && cfg.mode !== 'density' && cfg.lut) {
        const lutBytes = b64U8(cfg.lut);
        const size = (lutBytes.length / 3) | 0;
        if (size > 0) gpuHMLUT[tid] = buildHeatmapLUTGPU(lutBytes, size);
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
      const stacked  = !!cfg.stacked;
      for (const ch of Object.keys(d)) {
        gpuHist[tid][ch] = {};
        rawHist[tid][ch] = {};
        const csz = chromSizes[ch] || 1;
        for (const [gid, payload] of Object.entries(d[ch])) {
          const color = cfg.groups.find(g => g.id === gid)?.color || '#4488cc';
          // Empty group: nothing to upload.
          if (payload === '' || payload == null) {
            rawHist[tid][ch][gid] = { data: new Float32Array(0), binWidth, stacked };
            gpuHist[tid][ch][gid] = { base: null, levels: [] };
            continue;
          }
          // Both stacked and non-stacked ship as { base, lods }; only the
          // builder differs (stride-3 stacked vs stride-2 unstacked).
          const baseB64 = payload.base || '';
          const lodMap  = payload.lods || {};
          const baseArr = b64F32(baseB64);
          rawHist[tid][ch][gid] = { data: baseArr, binWidth, stacked };
          const buildBase = stacked ? buildHistStackedGPU : buildHistGPU;
          const baseGpu = baseArr && baseArr.length
            ? buildBase(baseArr, yMin, yMax, binWidth, color)
            : null;
          // Build a GPU buffer for each LOD level. For stacked, levels are
          // pre-built stride-3 (x, lo, hi) arrays from Python; for
          // non-stacked, they're length-n bin values that we expand into
          // (centre, value) pairs here.
          const levels = [];
          for (const [nStr, b64] of Object.entries(lodMap)) {
            const n = parseInt(nStr);
            const vals = b64F32(b64);
            if (!vals || vals.length === 0) continue;
            const lvlBinW = csz / n;
            let gpu;
            if (stacked) {
              gpu = buildHistStackedGPU(vals, yMin, yMax, lvlBinW, color);
            } else {
              const xy = new Float32Array(n * 2);
              for (let i = 0; i < n; i++) {
                xy[i * 2]     = (i + 0.5) * lvlBinW;
                xy[i * 2 + 1] = vals[i];
              }
              gpu = buildHistGPU(xy, yMin, yMax, lvlBinW, color);
            }
            if (gpu) levels.push({ nBins: n, binWidth: lvlBinW, ...gpu });
          }
          levels.sort((a, b) => a.nBins - b.nBins);  // coarsest → finest
          gpuHist[tid][ch][gid] = { base: baseGpu, levels };
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

function drawRects(gpuData, vs, ve, tt, tb, minDx) {
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
  gl.uniform1f(rLoc.uMinDx, minDx || 0.0);
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

// Draw heatmap texture for one group band.
// `lutData` is the optional value-mode palette texture; when present, `color`
// (the group's density-mode tint) is ignored and `texProgLUT` renders via LUT.
function drawHeatmapBand(hmData, vs, ve, chromSz, tt, tb, color, lutData) {
  if (!hmData) return;
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
  const [bgR, bgG, bgB] = hexRGB(th.bg || '#13131a');
  const stride = 16;
  if (lutData && lutData.tex) {
    gl.useProgram(texProgLUT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, hmData.tex);
    gl.uniform1i(tLocLUT.uTex, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, lutData.tex);
    gl.uniform1i(tLocLUT.uLUT, 1);
    gl.uniform3f(tLocLUT.uBg, bgR, bgG, bgB);
    gl.enableVertexAttribArray(tLocLUT.aPos);
    gl.vertexAttribPointer(tLocLUT.aPos, 2, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(tLocLUT.aPos, 0);
    gl.enableVertexAttribArray(tLocLUT.aTex);
    gl.vertexAttribPointer(tLocLUT.aTex, 2, gl.FLOAT, false, stride, 8);
    gl.vertexAttribDivisor(tLocLUT.aTex, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // Restore unit 0 as the active texture for subsequent draws.
    gl.activeTexture(gl.TEXTURE0);
    return;
  }
  const [r, g, b] = hexRGB(color);
  gl.useProgram(texProg);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, hmData.tex);
  gl.uniform1i(tLoc.uTex, 0);
  gl.uniform3f(tLoc.uBg, bgR, bgG, bgB);
  gl.uniform3f(tLoc.uFg, r, g, b);
  gl.uniform1f(tLoc.uNWin, hmData.nWin || 1);
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
  const color      = cfg.color     || th.gene_exon  || '#4488cc';
  const spineBase  = th.gene_spine || th.axis_text   || '#666688';
  const labelBase  = th.gene_label || th.track_label || '#9090c0';
  const hlS    = cfg.highlightStyles || {};
  const fillColor    = hlS.fillColor    || cfg.highlightColor || th.highlight_fill    || '#e03a4e';
  const spineColor   = hlS.spineColor   || th.highlight_spine   || '#d80ce6';
  const outlineColor = hlS.outlineColor || th.highlight_outline || '#000000';
  const labelColor   = hlS.labelColor   || th.highlight_label   || '#169f4a';
  const haloColor    = hlS.haloColor    || th.highlight_halo    || '#79CAD32E';

  const h = trackH(cfg);
  octx.fillStyle = th.bg || '#13131a';
  octx.fillRect(LABEL_W, trackY, drawW, h);
  if (!genes.length) return;

  // Pack rows tightly at the top of the track. The number of rows used for
  // this chrom may be less than cfg.rows (the global max across all chroms),
  // so any unused space is simply left empty at the bottom rather than
  // stretched into the gene layout.
  const nRowsChrom = Math.max(1, entry?.rows ?? cfg.rows ?? 1);
  // Keep rows at their natural size (≤30 px) even when the caller supplied a
  // much larger explicit height. The surplus space is redistributed: each
  // row is placed on a ``rowStride``-tall slot and the unused portion of each
  // slot becomes vertical air, with half a slot of padding above the first
  // row and below the last so the stack is centred. When ``h`` is tight
  // (auto mode or small explicit heights) ``rowStride === rowH`` and
  // ``rowPad === 0``, reproducing the original layout exactly.
  const rawRowH   = h / nRowsChrom;
  const rowH      = Math.min(30, rawRowH);
  const exonH     = Math.min(14, Math.max(6, rowH * 0.45));
  const labelBand = 11; // space reserved above each exon for the label (fits up to 10 px font)
  const rowStride = Math.max(rowH, rawRowH);
  const rowPad    = Math.max(0, (rawRowH - rowH) / 2);
  const arrowSep  = 70;
  const arrowA    = 4;

  const has = (gene, k) => !!(gene.hl && gene.hl.indexOf(k) >= 0);
  const fontFor = (gene, sizePx) => {
    const bold   = has(gene, 'bold')   ? 'bold '    : '';
    const italic = has(gene, 'italic') ? 'italic ' : '';
    return `${italic}${bold}${sizePx}px monospace`;
  };

  octx.save();
  octx.beginPath();
  octx.rect(LABEL_W, trackY, drawW, h);
  octx.clip();

  // Build a list of just the visible genes with their pixel extents.
  // Keep both the *clipped* extents (for drawing body/exons inside the track)
  // and the *true* (un-clipped) pixel midpoint (for stable label placement —
  // using the clipped midpoint makes feasibility non-monotonic in zoom).
  const vis = [];
  for (const gene of genes) {
    if (gene.e < vs || gene.s > ve) continue;
    // ``gene.s``/``gene.e`` may be inflated by label_padding (an invisible
    // prefix that participates in row packing and label centering). The true
    // gene footprint ``gs``/``ge`` (falling back to ``s``/``e`` when no
    // padding was applied) is what we actually draw as spine, exons and
    // arrows — the padded extent stays invisible.
    const gs = (gene.gs != null) ? gene.gs : gene.s;
    const ge = (gene.ge != null) ? gene.ge : gene.e;
    // Floor gene width at 1 px so sub-pixel genes remain visible at extreme
    // zoom-out instead of disappearing entirely. Matches the per-exon floor
    // applied further down so a single-exon gene and its fallback exon agree.
    const gx0Raw = LABEL_W + Math.max(0,     (gs - vs) / range * drawW);
    const gx1Raw = LABEL_W + Math.min(drawW, (ge - vs) / range * drawW);
    const gx0 = gx0Raw;
    const gx1 = Math.max(gx0 + 1, gx1Raw);
    const gw  = gx1 - gx0;
    // True (un-clipped) left edge of the gene body in pixel space — used to
    // anchor arrow markers so they stay fixed relative to the gene as the
    // user pans, rather than relative to the visible viewport edge.
    const gxTrueLeft = LABEL_W + (gs - vs) / range * drawW;
    // Label midpoint uses the *true* gene bounds so the name stays centred
    // on the visible gene body. The invisible prefix only serves to open up
    // a neighbouring row when genes are close — it must not drag the label
    // off the visible footprint.
    const cxTrue = LABEL_W + ((gs + ge) / 2 - vs) / range * drawW;
    vis.push({ gene, gx0, gx1, gw, cxTrue, gxTrueLeft, rowIdx: gene.row | 0 });
  }

  // Feasibility: try to place every visible gene's label at a given font size.
  // Labels are always centred on the gene body's *true* midpoint (computed
  // from gs/ge, i.e. the un-padded extent — and un-clipped, so the inter-
  // label distance grows monotonically with zoom). A label that falls off
  // a viewport edge is simply not drawn; it does not fail the tier. The
  // font string is built per gene so bold/italic highlights affect width.
  const tryFontSize = (sizePx) => {
    octx.font = `${sizePx}px monospace`;
    const pad = octx.measureText('M').width / 2;
    const used = Array.from({length: nRowsChrom}, () => []);
    const cxs = new Array(vis.length).fill(null);
    for (let i = 0; i < vis.length; i++) {
      const v = vis[i];
      if (!v.gene.n) continue;
      octx.font = fontFor(v.gene, sizePx);
      const tw = octx.measureText(v.gene.n).width;
      const cx  = v.cxTrue;
      const lx0 = cx - tw / 2 - pad, lx1 = cx + tw / 2 + pad;
      if (lx0 < LABEL_W || lx1 > W_css) continue;
      const u = used[v.rowIdx];
      for (const [a, b] of u) {
        if (lx0 < b && lx1 > a) return null;
      }
      cxs[i] = { cx, tw };
      u.push([lx0, lx1]);
    }
    return cxs;
  };

  const fontSizes = [10, 8, 6];
  let labelSize = null;
  let labelCxs = null;

  // Force-all mode: when the user has toggled the gene-label button and the
  // view contains fewer than GENE_LABEL_FORCE_MAX genes, place every visible
  // label without collision checks.
  if (forceAllLabels && vis.length < GENE_LABEL_FORCE_MAX) {
    const sz = 8;
    octx.font = `${sz}px monospace`;
    const cxs = new Array(vis.length).fill(null);
    for (let i = 0; i < vis.length; i++) {
      const v = vis[i];
      if (!v.gene.n) continue;
      octx.font = fontFor(v.gene, sz);
      const tw = octx.measureText(v.gene.n).width;
      cxs[i] = { cx: v.cxTrue, tw };
    }
    labelSize = sz; labelCxs = cxs;
  } else {
    // Prefer the largest tier that fits every visible gene; otherwise no
    // labels are drawn.
    for (const sz of fontSizes) {
      const cxs = tryFontSize(sz);
      if (cxs) { labelSize = sz; labelCxs = cxs; break; }
    }
  }

  // Halo pass (behind everything): fill a rounded rect slightly larger than
  // the exon band for genes with `halo` active.
  for (const v of vis) {
    if (!has(v.gene, 'halo')) continue;
    const rowTop = trackY + rowPad + v.rowIdx * rowStride;
    const exonY  = rowTop + labelBand + 1;
    const pad = 3;
    const hy = exonY - pad;
    const hh = exonH + pad * 2;
    const hx0 = v.gx0 - pad;
    const hw  = v.gw + pad * 2;
    octx.fillStyle = haloColor;
    octx.beginPath();
    octx.roundRect(hx0, hy, hw, hh, Math.min(6, hh / 2));
    octx.fill();
  }

  // Main draw pass.
  for (let i = 0; i < vis.length; i++) {
    const v = vis[i];
    const gene = v.gene;
    const rowIdx = v.rowIdx;
    const rowTop = trackY + rowPad + rowIdx * rowStride;
    const exonY  = rowTop + labelBand + 1;
    const midY   = exonY + exonH / 2;
    const bodyY  = midY - 1;
    const gx0 = v.gx0, gx1 = v.gx1, gw = v.gw;

    // Backbone / spine.
    octx.fillStyle = has(gene, 'stroke') ? spineColor : spineBase;
    octx.fillRect(gx0, bodyY, gw, 2);

    // Arrows: inherit spine colour when 'stroke' active, else muted.
    // Anchor to the gene's true (un-clipped) left edge so markers stay fixed
    // relative to the gene when panning — rather than drifting with the
    // viewport edge.
    const dir = gene.strand === '+' ? 1 : -1;
    octx.strokeStyle = has(gene, 'stroke') ? spineColor : spineBase;
    octx.lineWidth   = 1.2;
    octx.lineJoin    = 'round';
    const axStart = v.gxTrueLeft + arrowSep * 0.5;
    // Skip instances before the visible region.
    let ax0 = axStart;
    if (ax0 < gx0 + 2) {
      const skip = Math.ceil((gx0 + 2 - ax0) / arrowSep);
      ax0 += skip * arrowSep;
    }
    for (let ax = ax0; ax < gx1 - 6; ax += arrowSep) {
      octx.beginPath();
      octx.moveTo(ax - dir * arrowA, midY - arrowA);
      octx.lineTo(ax,                midY);
      octx.lineTo(ax - dir * arrowA, midY + arrowA);
      octx.stroke();
    }

    // Exons (fill + optional outline). When no explicit exons are supplied,
    // draw a single block spanning the true gene footprint (which equals
    // ``s``/``e`` when no padding was applied).
    const gs = (gene.gs != null) ? gene.gs : gene.s;
    const ge = (gene.ge != null) ? gene.ge : gene.e;
    const exons = gene.exons?.length ? gene.exons : [[gs, ge]];
    const exonFill    = has(gene, 'fill')    ? fillColor    : color;
    const doOutline   = has(gene, 'outline');
    for (const [es, ee] of exons) {
      if (ee < vs || es > ve) continue;
      const ex0 = LABEL_W + Math.max(0,     (es - vs) / range * drawW);
      const ex1 = LABEL_W + Math.min(drawW, (ee - vs) / range * drawW);
      const ew  = Math.max(1, ex1 - ex0);
      if (ew > 4) {
        const r = Math.min(2, exonH / 2, ew / 2);
        octx.beginPath();
        octx.roundRect(ex0, exonY, ew, exonH, r);
        octx.fillStyle = exonFill;
        octx.fill();
        if (doOutline) {
          octx.strokeStyle = outlineColor;
          octx.lineWidth   = 1;
          octx.stroke();
        }
      } else {
        octx.fillStyle = exonFill;
        octx.fillRect(ex0, exonY, ew, exonH);
        if (doOutline) {
          octx.strokeStyle = outlineColor;
          octx.lineWidth   = 1;
          octx.strokeRect(ex0 + 0.5, exonY + 0.5, ew - 1, exonH - 1);
        }
      }
    }

    // Label (per-gene font for bold/italic, per-gene colour, optional underline).
    if (labelSize != null && labelCxs && labelCxs[i] != null && gene.n) {
      const { cx, tw } = labelCxs[i];
      octx.font         = fontFor(gene, labelSize);
      octx.fillStyle    = has(gene, 'color') ? labelColor : labelBase;
      octx.textAlign    = 'center';
      octx.textBaseline = 'bottom';
      octx.fillText(gene.n, cx, exonY - 1);
      if (has(gene, 'underline')) {
        octx.fillRect(cx - tw / 2, exonY - 1, tw, 1);
      }
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
  octx.fillStyle    = th.axis_text || '#666688';
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
    octx.fillStyle = th.axis_text || '#666688';
    octx.fillText(fmtBp(t, nice), px, 4);
  }

  // ── Gene tracks first (behind labels) ───────────────────────────────────
  let cssY = SCALEBAR_H;
  for (const cfg of cfgs) {
    if (cfg.type === 'gene') drawGeneTrack2D(cfg, cssY, vs, ve, W_css);
    cssY += trackH(cfg);
  }

  // ── Track labels + separators ────────────────────────────────────────────
  cssY = SCALEBAR_H;
  for (const cfg of cfgs) {
    const ch = trackH(cfg);
    // Track separator
    octx.strokeStyle = th.border || '#252530';
    octx.lineWidth   = 1;
    octx.beginPath();
    octx.moveTo(0, cssY + ch - 0.5);
    octx.lineTo(W_css, cssY + ch - 0.5);
    octx.stroke();

    // Track name (left panel)
    octx.save();
    octx.beginPath();
    octx.rect(0, cssY, LABEL_W - 4, ch);
    octx.clip();
    octx.fillStyle    = th.track_label || '#9090c0';
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

    cssY += ch;
  }

  // ── User-specified vertical blocks (filled spans across all tracks) ──────
  const vblocks = model.get('vblocks') || [];
  if (vblocks.length) {
    octx.save();
    for (const b of vblocks) {
      if (!b || b.chrom !== vp.chrom) continue;
      const bs = +b.start, be = +b.end;
      if (!(be > bs)) continue;
      // Clip to viewport.
      const s = Math.max(bs, vs);
      const e = Math.min(be, ve);
      if (!(e > s)) continue;
      const x0 = LABEL_W + (s - vs) / range * drawW;
      const x1 = LABEL_W + (e - vs) / range * drawW;
      const w  = Math.max(0.5, x1 - x0);
      const a  = (b.alpha == null) ? 0.1
                 : Math.max(0, Math.min(1, +b.alpha));
      octx.globalAlpha = a;
      octx.fillStyle   = b.color || '#ffcc44';
      octx.fillRect(x0, SCALEBAR_H, w, H_css - SCALEBAR_H);
      const edge = +b.edgewidth || 0;
      if (edge > 0) {
        octx.strokeStyle = b.edgecolor || b.color || '#ffcc44';
        octx.lineWidth   = edge;
        octx.setLineDash(
          Array.isArray(b.dash) && b.dash.length
            ? b.dash.map(n => +n) : []
        );
        octx.strokeRect(x0 + 0.5, SCALEBAR_H + 0.5,
                        w - 1, H_css - SCALEBAR_H - 1);
        octx.setLineDash([]);
      }
    }
    octx.restore();
  }

  // ── User-specified vertical marker lines ─────────────────────────────────
  const vlines = model.get('vlines') || [];
  if (vlines.length) {
    octx.save();
    octx.lineCap = 'butt';
    for (const v of vlines) {
      if (!v || v.chrom !== vp.chrom) continue;
      const pos = +v.pos;
      if (!(pos >= vs && pos <= ve)) continue;
      const px = LABEL_W + (pos - vs) / range * drawW;
      if (px < LABEL_W || px > W_css) continue;
      const lw = +v.width || 1.0;
      const a  = (v.alpha == null) ? 1.0
                 : Math.max(0, Math.min(1, +v.alpha));
      octx.globalAlpha = a;
      octx.strokeStyle = v.color || '#ff4444';
      octx.lineWidth   = lw;
      const dashArr = Array.isArray(v.dash) && v.dash.length
        ? v.dash.map(n => +n) : [];
      octx.setLineDash(dashArr);
      octx.beginPath();
      // Snap to half-pixel only when lineWidth is odd, to keep the line crisp.
      const xSnap = (lw % 2 === 1) ? Math.round(px) + 0.5 : Math.round(px);
      octx.moveTo(xSnap, SCALEBAR_H);
      octx.lineTo(xSnap, H_css);
      octx.stroke();
    }
    octx.setLineDash([]);
    octx.restore();
  }

  octx.restore();
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN RENDER
// ══════════════════════════════════════════════════════════════════════════════
let rafId = null;

// Effective on-screen height of a track for the current chromosome. Gene
// tracks whose height was auto-computed from the *global* max row count
// shrink to match only the *current* chromosome's row count, so chromosomes
// that need fewer lanes don't leave a big empty band at the bottom.
function trackH(cfg) {
  if (cfg.type === 'gene' && cfg.heightAuto && cfg.rowsPerChrom) {
    const n = cfg.rowsPerChrom[vp.chrom];
    if (n != null) return Math.min(cfg.height, Math.max(30, n * 30 + 2));
  }
  return cfg.height;
}

function scheduleRender() {
  if (rafId) return;
  rafId = requestAnimationFrame(render);
}

function render() {
  rafId = null;
  const W_css = wrap.offsetWidth || 800;
  const cfgs  = model.get('track_configs');
  if (!cfgs || !cfgs.length) return;

  const totalH_css = SCALEBAR_H + cfgs.reduce((s, c) => s + trackH(c), 0);
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
    const cfgH     = trackH(cfg);
    const trackBot = cssY + cfgH;

    gl.scissor(
      LABEL_W * dpr,
      H_gl - trackBot * dpr,
      (W_css - LABEL_W) * dpr,
      cfgH * dpr
    );
    gl.enable(gl.SCISSOR_TEST);

    const tt = cssNDC(trackTop + PAD_V, totalH_css);
    const tb = cssNDC(trackBot - PAD_V, totalH_css);

    if (cfg.type === 'segment') {
      const segStacked = !!cfg.stacked;
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
            if (segStacked) {
              // Stacked: solid opaque fill so adjacent group bands read as
              // distinct colours rather than bleeding into each other.
              drawDensArea(level, vs, ve, tt, tb, grp.color, 1.0);
            } else {
              drawDensArea(level, vs, ve, tt, tb, grp.color, 0.80);
              drawDensArea(level, vs, ve, tt, tb, grp.color, 0.40);
            }
          }
        } else {
          drawRects(gpuSeg[cfg.id]?.[ch]?.[gid], vs, ve, tt, tb, bpPx / dpr);
        }
      }
    } else if (cfg.type === 'heatmap') {
      const nG = cfg.groups.length;
      const hWeights = cfg.groups.map(g => g.nInd || 1);
      const hTotalW  = hWeights.reduce((a, b) => a + b, 0);
      const hUsable  = cfgH;
      const hCumW    = [0];
      for (let i = 0; i < nG; i++) hCumW.push(hCumW[i] + hWeights[i]);
      const lutData = (cfg.mode && cfg.mode !== 'density') ? gpuHMLUT[cfg.id] : null;
      for (let gi = 0; gi < nG; gi++) {
        const grp  = cfg.groups[gi];
        const bandTT = cssNDC(trackTop + PAD_V + hUsable * hCumW[gi] / hTotalW, totalH_css);
        const bandTB = cssNDC(trackTop + PAD_V + hUsable * hCumW[gi + 1] / hTotalW, totalH_css);
        drawHeatmapBand(
          gpuHM[cfg.id]?.[ch]?.[grp.id],
          vs, ve, csz, bandTT, bandTB, grp.color, lutData
        );
      }
    } else if (cfg.type === 'gene') {
      // Rendered entirely on 2D overlay — no GL needed
    } else if (cfg.type === 'scatter') {
      const pointSize = cfg.pointSize || 3;
      const alpha     = cfg.alpha ?? 0.85;
      for (const grp of cfg.groups) {
        drawPoints(gpuXY[cfg.id]?.[ch]?.[grp.id], vs, ve, tt, tb, grp.color, alpha, pointSize);
      }
    } else if (cfg.type === 'line') {
      const alpha = cfg.alpha ?? 1.0;
      for (const grp of cfg.groups) {
        drawLine(gpuXY[cfg.id]?.[ch]?.[grp.id], vs, ve, tt, tb, grp.color, alpha);
      }
    } else if (cfg.type === 'fill') {
      for (const grp of cfg.groups) {
        const colorPos = grp.colorPos || grp.color;
        const colorNeg = grp.colorNeg || grp.color;
        drawFill(gpuFill[cfg.id]?.[ch]?.[grp.id], vs, ve, tt, tb, colorPos, colorNeg, 0.6);
      }
    } else if (cfg.type === 'histogram') {
      const pxPerBp = (W_css - LABEL_W) / (ve - vs);
      for (const grp of cfg.groups) {
        const slot = gpuHist[cfg.id]?.[ch]?.[grp.id];
        if (!slot) continue;
        // Treat the original bars as one extra "finest" level (binWidth =
        // cfg.binWidth) and pick the finest level whose bins still cover
        // >= 2 CSS px on screen. Mirrors the segment-track LOD heuristic.
        let pick = null;
        if (slot.levels) {
          for (const lv of slot.levels) {
            if (lv.binWidth * pxPerBp >= 2) pick = lv;
          }
        }
        if (slot.base && (cfg.binWidth || 0) * pxPerBp >= 2) {
          pick = slot.base;
        }
        // Nothing fits the threshold (extreme zoom-out): fall back to the
        // coarsest LOD level, or the original bars if no LODs exist.
        if (!pick) {
          pick = (slot.levels && slot.levels[0]) || slot.base;
        }
        drawRects(pick, vs, ve, tt, tb, bpPx / dpr);
      }
    }

    gl.disable(gl.SCISSOR_TEST);
    cssY += cfgH;
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
    const ch = trackH(cfg);
    if (offsetY >= cssY && offsetY < cssY + ch)
      return { cfg, trackTop: cssY, localY: offsetY - cssY };
    cssY += ch;
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
    const gs = (g.gs != null) ? g.gs : g.s;
    const ge = (g.ge != null) ? g.ge : g.e;
    if (pos >= gs && pos <= ge)
      hits.push({ name: g.n || 'unnamed', strand: g.strand === '+' ? '\u2192' : '\u2190', start: gs, end: ge });
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
    const stride = raw.stacked ? 3 : 2;
    const n = (arr.length / stride) | 0;
    const idx = bisectStride(arr, pos, stride);
    for (const c of [idx - 1, idx]) {
      if (c >= 0 && c < n) {
        const cx = arr[c * stride];
        if (pos >= cx - halfW && pos <= cx + halfW) {
          const value = raw.stacked
            ? (arr[c * stride + 2] - arr[c * stride + 1])
            : arr[c * stride + 1];
          items.push({ group: grp.name, value, x: cx });
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
function defaultFmtGene(items) { return items.map(d => d.name).join(', '); }
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
        cssY += trackH(cfg);
      }
    }
    tooltip.textContent = lines.join('\n');
    tooltip.style.cssText = `display:block;left:${e.offsetX + 14}px;top:${e.offsetY - 6}px`;
    return;
  }
  const W     = glCanvas.offsetWidth - LABEL_W;
  const bpPx  = (dragVp0.end - dragVp0.start) / W;
  const ps    = model.get('pan_speed') || 1.0;
  const shift = -(e.clientX - dragX0) * bpPx * ps;
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
  const zs     = model.get('zoom_speed') || 1.02;
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

// Pan by a fraction of the current view width, clamped to the chromosome
// bounds. Positive fraction pans right, negative pans left.
const doPan = f => {
  const w   = vp.end - vp.start;
  const lo  = panLo(vp.chrom), hi = panHi(vp.chrom);
  const ns  = clamp(vp.start + w * f, lo, hi - w);
  vp = { chrom: vp.chrom, start: ns, end: ns + w };
  scheduleRender(); syncVp(); updatePosBox();
};

panFFBtn.addEventListener('click', () => doPan(-0.9));
panHFBtn.addEventListener('click', () => doPan(-0.5));
panHRBtn.addEventListener('click', () => doPan( 0.5));
panFRBtn.addEventListener('click', () => doPan( 0.9));
resetBtn.addEventListener  ('click', () => {
  vp = { chrom: vp.chrom, start: panLo(vp.chrom), end: panHi(vp.chrom) };
  scheduleRender(); syncVp(); updatePosBox();
});

function updateHeatmapBtns() {
  const cfgs = model.get('track_configs') || [];
  const hasHM = cfgs.some(c => c.type === 'heatmap');
  hmRecBtn.style.display  = hasHM ? '' : 'none';
  hmGlobBtn.style.display = hasHM ? '' : 'none';
  const hasGene = cfgs.some(c => c.type === 'gene');
  geneLblBtn.style.display = hasGene ? '' : 'none';
}
updateHeatmapBtns();

// Force-show-all-labels mode for gene tracks (overrides collision-based
// placement when fewer than GENE_LABEL_FORCE_MAX genes are visible).
let forceAllLabels = false;
const GENE_LABEL_FORCE_MAX = 100;
const syncGeneLblBtn = () => {
  geneLblBtn.style.opacity = forceAllLabels ? '1' : '0.55';
  geneLblBtn.title = forceAllLabels
    ? 'Hide overlapping gene labels (revert to adaptive placement)'
    : 'Show all gene labels (when <100 visible)';
};
syncGeneLblBtn();
geneLblBtn.addEventListener('click', () => {
  forceAllLabels = !forceAllLabels;
  syncGeneLblBtn();
  scheduleRender();
});

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

snapBtn.addEventListener('click', async () => {
  // Render synchronously so the WebGL drawing buffer still has valid pixels
  // (the context was created without preserveDrawingBuffer, so otherwise the
  // buffer may be cleared by the time we copy it).
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  render();

  const w = glCanvas.width, h = glCanvas.height;
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const octx2 = out.getContext('2d');
  octx2.drawImage(glCanvas, 0, 0);
  octx2.drawImage(ov,       0, 0);

  const flash = (ok) => {
    const prev = snapBtn.textContent;
    snapBtn.textContent = ok ? '✓' : '!';
    setTimeout(() => { snapBtn.textContent = prev; }, 900);
  };

  try {
    if (!navigator.clipboard || !window.ClipboardItem) throw new Error('clipboard unsupported');
    const blob = await new Promise(res => out.toBlob(res, 'image/png'));
    if (!blob) throw new Error('toBlob failed');
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    flash(true);
  } catch (err) {
    console.warn('[Tracks] clipboard copy failed, falling back to download:', err);
    // Fallback: trigger a download so the user still gets the image.
    out.toBlob(blob => {
      if (!blob) return flash(false);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'segment-viewer.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      flash(true);
    }, 'image/png');
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
model.on('change:vlines', () => { scheduleRender(); });
model.on('change:vblocks', () => { scheduleRender(); });

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
class Tracks(anywidget.AnyWidget):
    """WebGL2 genomic segment viewer rendered as a Jupyter widget.

    Build a stack of tracks by chaining the ``add_*_track`` methods, then
    display the instance directly in a notebook cell. All pan / zoom is
    uniform-only (zero CPU/GPU buffer work per frame); segment viewport
    culling uses sorted attribute-pointer offsets.

    Track types
    -----------
    :meth:`add_segment_track`
        Instanced GPU rectangles with automatic level-of-detail (LOD)
        density fallback when zoomed out.
    :meth:`add_heatmap_track`
        R8 texture rendering; handles 10 k+ individuals at 60 fps.
    :meth:`add_gene_track`
        Exon blocks plus intron connectors (drawn on the 2D overlay).
    :meth:`add_scatter_track`
        Point cloud (``gl.POINTS``).
    :meth:`add_line_track`
        Connected polyline per group (``gl.LINE_STRIP``).
    :meth:`add_fill_track`
        Fill-between two curves with optional pos/neg colour split.
    :meth:`add_histogram_track`
        Vertical bars with multi-resolution LOD.

    Toolbar
    -------
    The widget renders a single-row toolbar above the canvas:

    Chromosome dropdown
        Switches the visible chromosome and resets the viewport to the
        full chromosome span.
    Position box
        Free-text region selector. Accepts ``chr1:1,000–2,000`` or
        ``chr1:1000-2000`` (the chromosome prefix is optional and
        defaults to the current one). Press ``Enter`` to jump.
    ``＋``  *Zoom in*
        Halves the visible window around its centre.
    ``－``  *Zoom out*
        Doubles the visible window around its centre (clamped to the
        chromosome).
    ``⌂``  *Reset view*
        Restores the viewport to the full chromosome (or, when a
        heatmap defines a tighter pan range, that range).
    ``⟲``  *Recompute heatmap(s)*
        Visible only when at least one heatmap track is present.
        Re-bins every heatmap at the current viewport for higher
        resolution. Routed through the ``_cmd_heatmap_recompute``
        command to :meth:`_rebin_heatmap_for_view`.
    ``◱``  *Restore global heatmap*
        Visible only when at least one heatmap track is present.
        Restores the original whole-chromosome heatmap pixels via
        :meth:`_restore_heatmap_global`.
    ``⧉``  *Snapshot*
        Composites the WebGL and overlay canvases and writes a PNG to
        the system clipboard (falls back to a file download when
        clipboard access is denied). The button briefly flashes ``✓``
        on success or ``!`` on failure.
    LOD badge
        Read-only indicator showing the current segment-track render
        mode: ``▬ segments`` when individual rectangles are drawn,
        ``▒ density`` when the density LOD is active.

    Pan and zoom
    ------------
    Mouse wheel
        Scroll wheel zooms in/out anchored on the cursor's genomic
        position. The zoom-step factor per wheel tick is
        :attr:`zoom_speed`.
    Click-drag
        Press and drag inside the plot area to pan along the
        chromosome.
    Double-click
        Halves the visible window around the click position (a faster
        alternative to the ``＋`` button when targeting a specific
        location).
    Keyboard (when the widget has focus or hover)
        ``←`` / ``→`` pan by 15 % of the current window.
        ``+`` / ``=`` zoom in by 1.5×; ``-`` zooms out by 1.5×.
    Tooltip
        Hovering over a track displays a tooltip whose contents come
        from the track's ``tip_fmt`` / ``tip_label``. The tooltip
        disappears on ``mouseleave``.

    Synchronised attributes
    -----------------------
    These traits round-trip between Python and the browser; assigning
    them from Python updates the rendered view immediately.

    chrom_sizes : dict of str to int
        Chromosome name → length in base pairs. Set once via
        :meth:`__init__`; re-assigning it rebuilds the dropdown and
        the pan-clamp ranges.
    track_configs : list of dict
        One config per track. Maintained by the ``add_*_track``
        methods; do not mutate directly.
    track_data : dict
        Track-id-keyed payloads consumed by the JS uploader. Maintained
        by the ``add_*_track`` methods; do not mutate directly.
    viewport : dict
        ``{'chrom': str, 'start': int, 'end': int}``. Use
        :meth:`set_viewport` or :meth:`zoom_to` to change it from
        Python; the JS side updates it in response to pan/zoom and
        toolbar input.
    zoom_speed : float, default ``1.02``
        Multiplicative zoom step per wheel-tick. Higher values zoom
        faster.
    pan_speed : float, default ``1.0``
        Multiplier for click-drag pan velocity (genomic units per CSS
        pixel of mouse movement).
    theme : dict
        Colour palette and layout knobs. See :data:`DARK_THEME` and
        :data:`LIGHT_THEME` for the recognised keys; numeric values
        like ``sidebar_w`` (CSS pixel width of the left label column)
        live alongside the colour strings. Mutating the trait via
        ``viewer.theme = {...}`` re-applies the theme without reload.

    Examples
    --------
    >>> viewer = Tracks(chrom_sizes={'chr1': 248_956_422})
    >>> (viewer
    ...  .add_gene_track(assembly='hg38')
    ...  .zoom_to('chr1', 50_000_000, window=10_000_000))
    >>> viewer  # display in a notebook cell
    """

    _esm = _JS
    _css = _CSS

    chrom_sizes   = traitlets.Dict({}).tag(sync=True)
    track_configs = traitlets.List([]).tag(sync=True)
    track_data    = traitlets.Dict({}).tag(sync=True)
    viewport      = traitlets.Dict({'chrom': '', 'start': 0, 'end': 0}).tag(sync=True)
    vlines        = traitlets.List([]).tag(sync=True)
    vblocks       = traitlets.List([]).tag(sync=True)
    zoom_speed    = traitlets.Float(1.02).tag(sync=True)
    pan_speed     = traitlets.Float(1.0).tag(sync=True)
    theme         = traitlets.Dict({
        'bg':           '#13131a',
        'fg':           '#d0d0e8',
        'panel':        '#1c1c26',
        'border':       '#252530',
        'input_bg':     '#0d0d14',
        'input_fg':     '#c0c0dc',
        'input_border': '#33334a',
        'focus_border': '#4466ee',
        'axis_text':    '#666688',
        'track_label':  '#9090c0',
        'gene_exon':    '#4488cc',
        'gene_spine':   '#666688',
        'gene_label':   '#9090c0',
        'sidebar_w':    96,
    }).tag(sync=True)

    _PALETTE = [
        '#4488cc', '#ee5566', '#33aa77', '#ddaa22',
        '#66bbcc', '#cc44aa', '#88aa44', '#aa6644',
    ]

    @traitlets.validate('theme')
    def _validate_theme(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Normalise theme values whenever the trait is assigned.

        Parameters
        ----------
        proposal : dict
            The traitlets proposal object whose ``'value'`` key carries
            the user-supplied theme dict.

        Returns
        -------
        dict of str to Any
            A new theme dict with every string colour spec resolved to a
            CSS-compatible hex through :func:`_resolve_color_mapping`.
        """
        return _resolve_color_mapping(proposal['value'])

    def __init__(self, chrom_sizes: dict[str, int], **kw):
        """Construct an empty viewer for a given set of chromosome sizes.

        Parameters
        ----------
        chrom_sizes : dict of str to int
            Mapping from chromosome name to length in base pairs. Keys
            and values are coerced to ``str`` and ``int`` respectively.
            The first key (in iteration order) becomes the initial
            viewport.
        **kw
            Extra keyword arguments forwarded to
            :class:`anywidget.AnyWidget`. The recognised extra is
            ``theme=``, which overrides the module-level default set by
            :func:`set_default_theme` and is normalised through
            :func:`_resolve_color_mapping` so matplotlib-style colour
            specs (``'C0'``, ``'red'``, …) can be supplied.
        """
        # Respect an explicit ``theme=`` kwarg; otherwise apply the module-level
        # default (see ``set_default_theme``).
        if 'theme' in kw:
            kw['theme'] = _resolve_color_mapping(kw['theme'])
        else:
            kw['theme'] = dict(_default_theme)
        super().__init__(**kw)
        self.chrom_sizes = {str(k): int(v) for k, v in chrom_sizes.items()}
        first = next(iter(self.chrom_sizes))
        self.viewport = {'chrom': first, 'start': 0, 'end': self.chrom_sizes[first]}
        self._heatmap_sources: dict[str, dict] = {}
        self._heatmap_global:  dict[str, dict] = {}

    @anywidget.experimental.command
    def _cmd_heatmap_recompute(
        self, msg: dict[str, Any], buffers: list,
    ) -> tuple[None, list]:
        """Handle the JS-side ``heatmap_recompute`` command.

        Parameters
        ----------
        msg : dict
            Command payload with ``'tid'``, ``'chrom'``, ``'xStart'``,
            ``'xEnd'``.
        buffers : list
            Binary buffers carried with the command (unused).

        Returns
        -------
        tuple of (None, list)
            The anywidget command-result protocol — no payload, no
            buffers.
        """
        self._rebin_heatmap_for_view(
            msg['tid'],
            str(msg['chrom']),
            int(msg['xStart']),
            int(msg['xEnd']),
        )
        return (None, [])

    @anywidget.experimental.command
    def _cmd_heatmap_reset(
        self, msg: dict[str, Any], buffers: list,
    ) -> tuple[None, list]:
        """Handle the JS-side ``heatmap_reset`` command.

        Parameters
        ----------
        msg : dict
            Command payload with ``'tid'``.
        buffers : list
            Binary buffers carried with the command (unused).

        Returns
        -------
        tuple of (None, list)
            The anywidget command-result protocol — no payload, no
            buffers.
        """
        self._restore_heatmap_global(msg['tid'])
        return (None, [])

    def _bin_heatmap_region(
        self,
        tid: str,
        chrom: str,
        x_start: int,
        x_end: int,
        windows: int,
    ) -> dict:
        """Rebin heatmap data for a single chromosome window.

        Parameters
        ----------
        tid : str
            Track ID assigned by :meth:`_tid`.
        chrom : str
            Chromosome name to rebin.
        x_start, x_end : int
            Half-open base-pair window ``[x_start, x_end)`` covered by
            the rebinned matrix.
        windows : int
            Number of bins along the x axis in the resulting matrix.

        Returns
        -------
        dict
            Per-group payload ready to drop into ``track_data[tid][chrom]``.
            Keys are stringified group indices; values carry ``data``,
            ``nInd``, ``nWin``, ``xStart``, ``xEnd``. Empty when the
            track id is unknown.
        """
        src = self._heatmap_sources.get(tid)
        if src is None:
            return {}
        df         = src['df']
        groups     = src['groups']
        group_col  = src['group_col']
        ind_col    = src['individual_col']
        inds_by_gi = src['inds_by_group']
        # Per-segment shading contribution (matplotlib-style alpha → 0..255).
        # Default alpha=0.5 → step 127 (≈ legacy 80, slightly bolder).
        # alpha=1.0 → step 255 (single segment fully saturates the cell).
        alpha = float(src.get('alpha', 0.5))
        step  = max(1, min(255, int(round(255 * alpha))))

        mode       = src.get('mode', 'density')
        value_col  = src.get('value_col')
        vmin       = src.get('vmin')
        vmax       = src.get('vmax')
        cat_to_idx = src.get('cat_to_idx') or {}

        span = max(1, x_end - x_start)
        out: dict = {}

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
                # Pre-resolve palette-slot byte per segment when in value mode
                # (0 = skip / no data, 1..255 = LUT slot).
                if mode == 'discrete':
                    vals_arr = gdf[value_col].to_numpy()
                    slot_arr = np.fromiter(
                        (cat_to_idx.get(v, 0) for v in vals_arr),
                        count=len(vals_arr), dtype=np.uint16,
                    )
                elif mode == 'continuous':
                    vals_arr = pd.to_numeric(gdf[value_col], errors='coerce').to_numpy()
                    lo, hi = float(vmin), float(vmax)
                    rng = hi - lo if hi > lo else 1.0
                    norm = (vals_arr - lo) / rng
                    norm = np.clip(norm, 0.0, 1.0)
                    # Map [0, 1] → [1, 255]; NaN segments fall back to 0 (skip).
                    slot_arr = np.where(
                        np.isfinite(vals_arr),
                        np.clip(np.round(norm * 254.0) + 1.0, 1.0, 255.0),
                        0.0,
                    ).astype(np.uint16)
                else:
                    slot_arr = None

                for k in range(len(gdf)):
                    ii = ind_idx.get(inds_arr[k])
                    if ii is None: continue
                    s = max(starts_arr[k], x_start)
                    e = min(ends_arr[k],   x_end)
                    if e <= s: continue
                    b0 = int(max(0, (s - x_start) / span * windows))
                    b1 = int(min(windows - 1, (e - x_start) / span * windows)) + 1
                    if slot_arr is None:
                        # Density: accumulate alpha, clamp at 255.
                        matrix[ii, b0:b1] = np.minimum(
                            255, matrix[ii, b0:b1].astype(int) + step
                        )
                    else:
                        # Value mode: last write wins within overlapping
                        # segments. Skip sentinel-0 slots entirely.
                        slot = int(slot_arr[k])
                        if slot == 0:
                            continue
                        matrix[ii, b0:b1] = slot

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
        """Rebuild and install heatmap pixels for the visible window.

        Parameters
        ----------
        tid : str
            Track ID.
        chrom : str
            Chromosome currently in view.
        x_start, x_end : int
            Half-open base-pair viewport ``[x_start, x_end)``.
        """
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
        """Restore the original whole-chromosome heatmap pixels.

        Parameters
        ----------
        tid : str
            Track ID. A no-op when no cached global view exists for the
            track (i.e. the heatmap has never been rebinned).
        """
        cached = self._heatmap_global.get(tid)
        if cached is None:
            return
        self.track_data = {**self.track_data, tid: {
            ch: {gid: {**entry} for gid, entry in groups.items()}
            for ch, groups in cached.items()
        }}

    def _tid(self) -> str:
        """Return a fresh, monotonically increasing track ID.

        Returns
        -------
        str
            ``'t0'``, ``'t1'``, … one per call, derived from the
            current length of ``track_configs``.
        """
        return f't{len(self.track_configs)}'

    @staticmethod
    def _pack_f32(arr: np.ndarray) -> str:
        """Pack a numpy array as base64-encoded little-endian float32 bytes.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array; cast to ``float32`` before serialisation.

        Returns
        -------
        str
            ASCII base64 of the raw byte buffer, ready to ship through
            the traitlets sync layer to the JS-side ``b64F32`` decoder.
        """
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

    @staticmethod
    def _pack_u8(arr: np.ndarray) -> str:
        """Pack a numpy array as base64-encoded uint8 bytes.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array; cast to ``uint8`` before serialisation.

        Returns
        -------
        str
            ASCII base64 of the raw byte buffer, ready to ship through
            the traitlets sync layer to the JS-side ``b64U8`` decoder.
        """
        return base64.b64encode(arr.astype(np.uint8).tobytes()).decode()

    # ── add_segment_track ────────────────────────────────────────────────────
    def add_segment_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        group_by: str | None = None,
        individual_col: str | None = None,
        color_map: dict | None = None,
        height: int | None = None,
        density_windows: tuple | int = (256, 1024, 4096),
        stack: bool = False,
        tip_fmt: str | None = None,
        tip_label: str | None = None,
    ) -> 'Tracks':
        """Add a segment track rendered with GPU-instanced rectangles.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format frame with columns ``chrom``, ``start``, ``end``,
            and any of the optional ``group_by`` / ``individual_col``
            columns referenced below.
        name : str
            Track display label shown in the left panel.
        group_by : str, optional
            Column whose unique values become separate sub-rows.
        individual_col : str, optional
            Column identifying individuals. When provided, each individual
            gets its own horizontal row within the group band (stacked
            haplotype view); track height auto-adapts so each row is at
            least 1 px.
        color_map : dict, optional
            ``{group_value: '#rrggbb'}``. When omitted, colours cycle
            through the built-in palette in group order.
        height : int, optional
            Track height in CSS pixels. Defaults to ``max(90, nrows)``,
            i.e. at least one pixel per row.
        density_windows : tuple of int or int, default ``(256, 1024, 4096)``
            Bin counts for the multi-resolution density LOD. A single
            ``int`` is treated as one level.
        stack : bool, default False
            If True, groups' density contributions stack on top of each
            other in the zoomed-out density view (rather than
            overlapping). Normalisation uses the combined max across all
            bins. Requires ``group_by`` with more than one group; ignored
            when ``individual_col`` is set.
        tip_fmt : str, optional
            Python format string for tooltips. Available key:
            ``{group}``.
        tip_label : str, optional
            Heading shown above the tooltip body. Defaults to
            ``f'{name}:'``.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.
        """
        color_map = _resolve_color_mapping(color_map)
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
        grp_ind_map: dict[str, dict] = {}
        grp_nind: dict[str, int] = {}
        if use_ind:
            for gi, group in enumerate(groups):
                gdf = df[df[group_by] == group] if group_by else df
                ind_uniq = sorted(gdf[individual_col].dropna().unique(), key=str)
                grp_ind_map[str(gi)] = {v: i for i, v in enumerate(ind_uniq)}
                grp_nind[str(gi)] = len(ind_uniq)

        if height is None:
            total_rows = sum(grp_nind.values()) if use_ind else len(groups)
            height = max(90, total_rows)

        # Stacking is only meaningful in the density overlay with >1 group and
        # no per-individual layout.
        stacked = bool(stack) and len(groups) > 1 and not use_ind

        seg_out: dict  = {}
        dens_out: dict = {}

        for chrom_val, cdf in df.groupby('chrom'):
            chrom = str(chrom_val)
            csz   = self.chrom_sizes.get(chrom, int(cdf['end'].max()))
            seg_out[chrom]  = {}
            dens_out[chrom] = {}

            # First pass — segment rects and raw per-group per-level count arrays.
            counts_per_level: dict[int, dict[str, np.ndarray]] = {n: {} for n in density_windows}
            for gi, group in enumerate(groups):
                gid = str(gi)
                gdf = cdf[cdf[group_by] == group] if group_by else cdf
                if gdf.empty:
                    seg_out[chrom][gid] = ''
                    for n in density_windows:
                        counts_per_level[n][gid] = np.zeros(n, dtype=np.float32)
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
                for n in density_windows:
                    bins = np.clip((starts / csz * n).astype(int), 0, n - 1)
                    counts = np.zeros(n, dtype=np.float32)
                    np.add.at(counts, bins, 1)
                    counts_per_level[n][gid] = counts

            # Second pass — serialise counts as either plain per-group arrays
            # (non-stacked) or stride-3 (x_bin_left_norm, yLo, yHi) arrays in
            # normalised [0,1] space (stacked).  In the stacked case we use a
            # shared column-sum maximum so groups scale consistently.
            for gi, group in enumerate(groups):
                gid = str(gi)
                if seg_out[chrom].get(gid) == '':
                    dens_out[chrom][gid] = ''
                    continue
                levels: dict[str, str] = {}
                for n in density_windows:
                    counts = counts_per_level[n][gid]
                    if stacked:
                        # column sum across all groups at each bin
                        col_sum = np.zeros(n, dtype=np.float64)
                        for gj in range(len(groups)):
                            col_sum += counts_per_level[n][str(gj)].astype(np.float64)
                        maxVal = float(col_sum.max()) or 1.0
                        cum_prev = np.zeros(n, dtype=np.float64)
                        for gj in range(gi):
                            cum_prev += counts_per_level[n][str(gj)].astype(np.float64)
                        yLo = cum_prev / maxVal
                        yHi = (cum_prev + counts.astype(np.float64)) / maxVal
                        # stride-3: xBinLeft (as bin index 0..n-1), yLo, yHi.
                        # JS will convert xBinLeft to chromosome coords via nBins.
                        arr = np.empty(n * 3, dtype=np.float32)
                        arr[0::3] = np.arange(n, dtype=np.float32)
                        arr[1::3] = yLo.astype(np.float32)
                        arr[2::3] = yHi.astype(np.float32)
                        levels[str(n)] = self._pack_f32(arr)
                    else:
                        levels[str(n)] = self._pack_f32(counts)
                dens_out[chrom][gid] = levels

        cfg = {
            'id': tid, 'type': 'segment', 'name': name, 'height': height,
            'stacked': stacked,
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
        group_col: str | None = None,
        sort_by: str | None = None,
        color_map: dict | None = None,
        value_col: str | None = None,
        palette: Any = None,
        vmin: float | None = None,
        vmax: float | None = None,
        height: int | None = None,
        windows: int = 1000,
        alpha: float = 0.5,
        tip_fmt: str | None = None,
        tip_label: str | None = None,
    ) -> 'Tracks':
        """Add a heatmap track — one row per individual, GPU texture rendering.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format frame with columns ``chrom``, ``start``, ``end``,
            and at least the column named by ``individual_col``.
        name : str
            Track display label shown in the left panel.
        individual_col : str, default ``'sample'``
            Column identifying each haplotype / sample row.
        group_col : str, optional
            Column for grouping individuals into coloured bands.
        sort_by : str, optional
            Column used to sort individuals within each group.
        color_map : dict, optional
            ``{group_value: '#rrggbb'}``. When omitted, colours cycle
            through the built-in palette in group order. Only used in
            density mode (when ``value_col`` is ``None``).
        value_col : str, optional
            Column whose per-segment values drive colouring. When
            omitted, the track renders in density mode (overlapping
            segments accumulate alpha, tinted per group).
        palette : str, list, dict, or matplotlib.colors.Colormap, optional
            Palette used when ``value_col`` is given. A colormap name
            (``'viridis'``) or ``Colormap`` instance triggers continuous
            mode; a ``{category: colour}`` dict or list of colours
            triggers discrete mode (up to 255 categories). Ignored when
            ``value_col`` is ``None``.
        vmin, vmax : float, optional
            Continuous-mode normalisation range; auto-inferred from
            ``df[value_col]`` when omitted. Ignored in discrete mode.
        height : int, optional
            Track height in CSS pixels. Defaults to ``max(90, n_individuals)``,
            i.e. at least one pixel per row.
        windows : int, default 1000
            Number of pre-computed density bins covering the whole
            chromosome.
        alpha : float, default 0.5
            Per-segment opacity (matplotlib-style). Density mode only —
            each overlapping segment adds ``alpha`` to the cell shading,
            clamped at ``1.0``. Must lie in ``(0, 1]``.
        tip_fmt : str, optional
            Python format string for tooltips. Available keys:
            ``{group}``, ``{individual}``, ``{nInd}``.
        tip_label : str, optional
            Heading shown above the tooltip body. Defaults to
            ``f'{name}:'``.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.

        Raises
        ------
        ValueError
            If ``alpha`` is not in ``(0, 1]``, if ``palette`` is supplied
            without ``value_col``, if ``value_col`` is missing from
            ``df``, or if a discrete palette has more than 255 categories.
        """
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        alpha = float(alpha)
        color_map = _resolve_color_mapping(color_map)
        tid = self._tid()

        # ── Resolve colour mode (density | discrete | continuous) ──────────
        if palette is not None and value_col is None:
            raise ValueError("palette requires value_col to be set")
        if value_col is not None and value_col not in df.columns:
            raise ValueError(f"value_col '{value_col}' not in df.columns")
        if value_col is not None and palette is None:
            raise ValueError(
                "value_col requires palette (colormap name, Colormap, dict, or list)"
            )
        lut_info: dict | None = None
        if value_col is not None:
            lut_info = _build_heatmap_lut(df[value_col], palette, vmin, vmax)
        mode = lut_info['mode'] if lut_info else 'density'

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
        inds_by_group: dict[str, list] = {}
        grp_meta: list = []
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
        if value_col and value_col not in needed_cols:
            needed_cols.append(value_col)
        self._heatmap_sources[tid] = {
            'df':             df[needed_cols].copy(),
            'groups':         groups,
            'group_col':      group_col,
            'individual_col': individual_col,
            'sort_by':        sort_by,
            'windows':        windows,
            'alpha':          alpha,
            'inds_by_group':  inds_by_group,
            'mode':           mode,
            'value_col':      value_col,
            'vmin':           lut_info['vmin'] if lut_info else None,
            'vmax':           lut_info['vmax'] if lut_info else None,
            'cat_to_idx':     lut_info['cat_to_idx'] if lut_info else None,
            'palette_ref':    lut_info['palette_ref'] if lut_info else None,
            'name':           name,
        }

        # Build initial (whole-chromosome) binning via the helper.
        hm_out: dict = {}
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
            'mode':  mode,
            **(({'tipFmt': tip_fmt} if tip_fmt is not None else {})),
            'tipLabel': tip_label if tip_label is not None else f'{name}:',
        }
        if lut_info is not None:
            # LUT ships as 256 × 3 uint8 bytes; JS side decodes into an RGB8
            # texture keyed to byte index (0 reserved for "no data").
            cfg['lut'] = self._pack_u8(lut_info['lut_u8'].reshape(-1))
        self.track_data    = {**self.track_data, tid: hm_out}
        self.track_configs = [*self.track_configs, cfg]
        return self

    def colorbar(
        self,
        track_name: str,
        *,
        ax: Any = None,
        orientation: str = 'horizontal',
        label: str | None = None,
    ):
        """Build a matplotlib colorbar or legend for a value-mode heatmap.

        Renders outside the widget — drop into a cell alongside the
        ``Tracks`` display to give the palette a legible scale. Density-
        mode tracks have no palette to show and raise ``ValueError``.

        Parameters
        ----------
        track_name : str
            Name of a heatmap track added via :meth:`add_heatmap_track`.
        ax : matplotlib.axes.Axes, optional
            Target axes. When omitted, a small figure is created with
            dimensions appropriate to ``orientation``.
        orientation : {'horizontal', 'vertical'}, default ``'horizontal'``
            Colorbar orientation (continuous mode only).
        label : str, optional
            Axis label. Defaults to the track's ``value_col``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the colorbar / legend.

        Raises
        ------
        KeyError
            If no heatmap track is named ``track_name``.
        ValueError
            If the named track is in density mode (no palette).
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        # Locate the heatmap source by track name (cfg.name → tid).
        tid = None
        for cfg in self.track_configs:
            if cfg.get('type') == 'heatmap' and cfg.get('name') == track_name:
                tid = cfg['id']
                break
        if tid is None:
            raise KeyError(f"no heatmap track named {track_name!r}")
        src = self._heatmap_sources.get(tid)
        if src is None or src.get('mode', 'density') == 'density':
            raise ValueError(
                f"track {track_name!r} is in density mode; no palette to show"
            )

        mode        = src['mode']
        palette_ref = src.get('palette_ref')
        axis_label  = label if label is not None else (src.get('value_col') or '')

        if mode == 'continuous':
            fig = ax.figure if ax is not None else plt.figure(
                figsize=(4.0, 0.6) if orientation == 'horizontal' else (0.6, 4.0)
            )
            if ax is None:
                ax = fig.add_subplot(111)
            norm = mcolors.Normalize(vmin=src['vmin'], vmax=src['vmax'])
            cb = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=palette_ref),
                cax=ax, orientation=orientation,
            )
            if axis_label:
                cb.set_label(axis_label)
            return fig

        # Discrete legend: one Patch per category.
        fig = ax.figure if ax is not None else plt.figure(figsize=(4.0, 0.6))
        if ax is None:
            ax = fig.add_subplot(111)
        ax.set_axis_off()
        handles = [
            Patch(facecolor=color, edgecolor='none', label=str(cat))
            for cat, color in palette_ref.items()
        ]
        ax.legend(
            handles=handles,
            loc='center',
            ncol=min(len(handles), 6),
            frameon=False,
            title=axis_label or None,
        )
        return fig

    # ── add_gene_track ───────────────────────────────────────────────────────
    @staticmethod
    def _collapse_exons(
        transcripts: list[list[tuple[int, int]]],
    ) -> list[list[int]]:
        """Merge overlapping exons across all transcripts into a single list.

        Parameters
        ----------
        transcripts : list of list of (int, int)
            Each transcript is a list of ``(start, end)`` pairs in base
            pairs.

        Returns
        -------
        list of [int, int]
            Sorted, non-overlapping ``[start, end]`` intervals covering
            the union of all input exons.
        """
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

    _HIGHLIGHT_KEYS = (
        'fill', 'stroke', 'outline', 'color',
        'bold', 'italic', 'underline', 'halo',
    )

    def add_gene_track(
        self,
        genes_data: dict | pd.DataFrame | None = None,
        exons_df: pd.DataFrame | None = None,
        *,
        assembly: str | None = None,
        name: str = 'Genes',
        color: str | None = None,
        height: int | None = None,
        collapse: bool = True,
        label_padding: int = 0,
        highlight: list[str] | dict[str, list[str]] | None = None,
        highlight_color: str | None = None,
        highlight_fill_color:    str | None = None,
        highlight_spine_color:   str | None = None,
        highlight_outline_color: str | None = None,
        highlight_label_color:   str | None = None,
        highlight_halo_color:    str | None = None,
        tip_fmt: str | None = None,
        tip_label: str | None = None,
    ) -> 'Tracks':
        """Add a gene / exon annotation track.

        Parameters
        ----------
        genes_data : dict or pandas.DataFrame, optional
            Either a dict mapping chromosome to a list of
            ``(gene_name, chrom, start, end, strand, transcripts)``
            tuples (as returned by ``geneinfo``), or a DataFrame with
            columns ``chrom``, ``start``, ``end``, ``name``, ``strand``.
            Mutually exclusive with ``assembly``; exactly one must be
            provided.
        exons_df : pandas.DataFrame, optional
            Frame with ``chrom``, ``gene_name``, ``start``, ``end``.
            Only used with the DataFrame form of ``genes_data``.
        assembly : str, optional
            Assembly identifier (e.g. ``'hg38'``). When given,
            ``genes_data`` is fetched automatically via
            :func:`geneinfo.coords.gene_coords_region` for every
            chromosome in :func:`geneinfo.coords.chromosome_lengths`.
            Mutually exclusive with ``genes_data``.
        name : str, default ``'Genes'``
            Track display label.
        color : str, optional
            Exon block colour for non-highlighted genes (matplotlib
            colour spec). Falls back to ``theme['gene_exon']``.
        height : int, optional
            Track height in CSS pixels. Defaults to the required number
            of rows (positive strand above, negative below, each with
            sub-lanes on overlap).
        collapse : bool, default True
            If True, merge exons across all transcripts into a single
            union set. If False, each transcript becomes its own row.
        label_padding : int, default 0
            Genomic bp of extra space reserved on the label-facing side
            of every gene — to the left for ``+``-strand genes and to
            the right for ``-``-strand genes — when assigning rows and
            placing labels. Modeled as an invisible prefix attached to
            each gene: row packing, label centring and label collision
            all see the padded extent, while the spine, exons, arrows
            and tooltip range continue to reflect the gene's true
            bounds. Genes too close to a preceding gene are bumped to
            a lower row so the invisible prefix can hold the label.
            ``0`` reproduces the prior overlap-only packing exactly.
        highlight : list of str or dict of str to list of str, optional
            Either a list of gene names (treated as ``{'fill': [...]}``)
            or a dict mapping a property key to a list of gene names.
            Valid keys are ``'fill'``, ``'stroke'``, ``'outline'``,
            ``'color'``, ``'bold'``, ``'italic'``, ``'underline'``,
            ``'halo'``. A gene may appear under multiple keys; all
            applicable properties are applied independently.
        highlight_color : str, optional
            Deprecated alias for ``highlight_fill_color``. Retained for
            backward compatibility with the list form of ``highlight``.
        highlight_fill_color : str, optional
            Exon fill colour for ``'fill'`` highlights.
        highlight_spine_color : str, optional
            Backbone colour for ``'stroke'`` highlights.
        highlight_outline_color : str, optional
            Exon outline colour for ``'outline'`` highlights.
        highlight_label_color : str, optional
            Gene-name text colour for ``'color'`` highlights.
        highlight_halo_color : str, optional
            Halo rectangle colour (may include alpha) for ``'halo'``
            highlights.
        tip_fmt : str, optional
            Python format string for tooltips. Available keys:
            ``{name}``, ``{strand}``, ``{start}``, ``{end}``.
        tip_label : str, optional
            Heading shown above the tooltip body. Defaults to
            ``f'{name}:'``.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.

        Raises
        ------
        ValueError
            If neither or both of ``genes_data`` and ``assembly`` are
            given, or if ``highlight`` carries an unknown property key.
        """
        if (genes_data is None) == (assembly is None):
            raise ValueError(
                "add_gene_track requires exactly one of `genes_data` or "
                "`assembly` to be specified."
            )

        if label_padding < 0:
            raise ValueError("label_padding must be non-negative.")

        if assembly is not None:
            from ..coords import gene_coords_region, chromosome_lengths
            chrom_names, _ = zip(*chromosome_lengths(assembly=assembly))
            genes_data = {
                chrom: gene_coords_region(
                    chrom, assembly=assembly, include_strand=True
                )
                for chrom in chrom_names
            }

        # Resolve matplotlib-style colour specs (e.g. 'C0', named colours).
        color                   = resolve_color(color)
        highlight_color         = resolve_color(highlight_color)
        highlight_fill_color    = resolve_color(highlight_fill_color)
        highlight_spine_color   = resolve_color(highlight_spine_color)
        highlight_outline_color = resolve_color(highlight_outline_color)
        highlight_label_color   = resolve_color(highlight_label_color)
        highlight_halo_color    = resolve_color(highlight_halo_color)

        # Backward-compatible alias: highlight_color overrides the fill colour.
        if highlight_color is not None:
            highlight_fill_color = highlight_color

        # Normalise `highlight` into {key: set(gene_names)}.
        hl_sets: dict[str, set] = {}
        if highlight is None:
            pass
        elif isinstance(highlight, dict):
            for k, names in highlight.items():
                if k not in self._HIGHLIGHT_KEYS:
                    raise ValueError(
                        f"Unknown highlight key {k!r}; valid keys are "
                        f"{self._HIGHLIGHT_KEYS}"
                    )
                hl_sets[k] = {str(g) for g in (names or [])}
        else:
            # list / iterable: apply default 'fill' property to listed genes.
            hl_sets['fill'] = {str(g) for g in highlight}

        def active_keys(gname: str) -> list[str]:
            return [k for k in self._HIGHLIGHT_KEYS
                    if gname in hl_sets.get(k, ())]
        tid   = self._tid()
        gdata: dict = {}

        if isinstance(genes_data, dict) and not isinstance(genes_data, pd.DataFrame):
            # genes_by_chrom dict: {chrom: [(name, chrom, start, end, strand, transcripts), ...]}
            for chrom, gene_list in genes_data.items():
                chrom = str(chrom)
                recs: list = []
                for entry in gene_list:
                    gname, _, gs, ge, strand, transcripts = entry
                    strand = str(strand) if strand in ('+', '-') else '+'
                    hl = active_keys(str(gname))
                    if collapse:
                        exons = self._collapse_exons(transcripts)
                        rec = {
                            's': int(gs), 'e': int(ge),
                            'n': str(gname), 'strand': strand, 'exons': exons,
                        }
                        if hl: rec['hl'] = hl
                        recs.append(rec)
                    else:
                        for ti, tex in enumerate(transcripts):
                            exons = [[int(s), int(e)] for s, e in tex]
                            # transcript extent may be narrower than gene extent
                            ts = min((s for s, _ in exons), default=int(gs))
                            te = max((e for _, e in exons), default=int(ge))
                            suffix = f' t{ti+1}' if len(transcripts) > 1 else ''
                            rec = {
                                's': int(ts), 'e': int(te),
                                'n': f'{gname}{suffix}', 'strand': strand,
                                'exons': exons,
                            }
                            if hl: rec['hl'] = hl
                            recs.append(rec)
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
                    exons: list = []
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
                    rec = {
                        's': int(row['start']), 'e': int(row['end']),
                        'n': gname, 'strand': strand, 'exons': exons,
                    }
                    hl = active_keys(gname)
                    if hl: rec['hl'] = hl
                    recs.append(rec)
                gdata[chrom] = recs

        # Apply label_padding uniformly by extending each record's packing
        # extent on the side where the gene name reads from — the left for
        # ``+``-strand genes, the right for ``-``-strand. The true gene bounds
        # are preserved as ``gs``/``ge`` so the renderer still draws spine,
        # exons and arrows over the real gene footprint. Because ``s``/``e``
        # drive row packing, label midpoint, viewport culling and inter-label
        # collision detection, inflating them is equivalent to prepending an
        # invisible exon: the same code path handles padding=0 and >0.
        if label_padding:
            for recs in gdata.values():
                for rec in recs:
                    rec['gs'] = rec['s']
                    rec['ge'] = rec['e']
                    if rec.get('strand') == '+':
                        rec['s'] = rec['s'] - label_padding
                    else:
                        rec['e'] = rec['e'] + label_padding

        # Pack all genes (both strands) into the minimum number of non-overlapping
        # rows with a greedy interval lane-packing. Strand is preserved on each
        # record and encoded via arrow direction at draw time.
        max_rows = 1
        rows_per_chrom: dict[str, int] = {}
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

        height_auto = height is None
        if height_auto:
            # Per row: ~11 px label band + ~11 px exon + ~8 px bottom gap so
            # the next row's label cannot touch the previous row's exon.
            height = max(30, max_rows * 30 + 2)

        hl_styles: dict[str, str] = {}
        if highlight_fill_color    is not None: hl_styles['fillColor']    = highlight_fill_color
        if highlight_spine_color   is not None: hl_styles['spineColor']   = highlight_spine_color
        if highlight_outline_color is not None: hl_styles['outlineColor'] = highlight_outline_color
        if highlight_label_color   is not None: hl_styles['labelColor']   = highlight_label_color
        if highlight_halo_color    is not None: hl_styles['haloColor']    = highlight_halo_color

        cfg = {
            'id': tid, 'type': 'gene', 'name': name,
            'height': height, 'color': color,
            'highlightStyles': hl_styles,
            'rows': max_rows,
            'rowsPerChrom': rows_per_chrom,
            'heightAuto': height_auto,
            'labelPadding': label_padding,
            'groups': [],
            **(({'tipFmt': tip_fmt} if tip_fmt is not None else {})),
            'tipLabel': tip_label if tip_label is not None else f'{name}:',
        }
        if highlight_fill_color is not None:
            cfg['highlightColor'] = highlight_fill_color  # legacy alias
        self.track_data    = {**self.track_data, tid: gdata_out}
        self.track_configs = [*self.track_configs, cfg]
        return self

    @staticmethod
    def _assign_lanes(recs: list[dict[str, Any]], offset: int = 0) -> int:
        """Pack overlapping gene records into the fewest possible rows.

        Classic first-fit interval colouring: records are sorted by
        ``(start, end)`` and each is placed in the lowest-index row
        whose current rightmost edge is at or before the record's
        start. This is optimal for interval graphs (the number of rows
        used equals the maximum clique size, i.e. the deepest stack of
        overlapping intervals) and keeps upper rows as densely packed
        as possible. Strand does not influence row choice — arrow
        direction already encodes strand per-gene.

        Parameters
        ----------
        recs : list of dict
            Gene records carrying at least ``'s'`` (start) and ``'e'``
            (end) keys.
        offset : int, default 0
            Constant added to every assigned lane index so the caller
            can stack multiple lane-packs vertically.

        Returns
        -------
        int
            Total number of lanes used, ``0`` when ``recs`` is empty.
        """
        if not recs:
            return 0

        NEG_INF = float('-inf')
        lane_end: list[float] = []

        for r in sorted(recs, key=lambda x: (x.get('s', 0), x.get('e', 0))):
            placed_row = None
            for row in range(len(lane_end)):
                if r['s'] >= lane_end[row]:
                    placed_row = row
                    break

            if placed_row is None:
                placed_row = len(lane_end)
                lane_end.append(NEG_INF)

            lane_end[placed_row] = r['e']
            r['row'] = offset + placed_row

        return len(lane_end)

    # ── add_scatter_track ────────────────────────────────────────────────────
    def add_scatter_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        x: str = 'pos',
        y: str = 'value',
        group_by: str | None = None,
        color_map: dict | None = None,
        height: int = 60,
        y_range: tuple[float, float] | None = None,
        point_size: int = 3,
        alpha: float | None = None,
        color: Any = None,
        c: Any = None,
        s: float | None = None,
        size: float | None = None,
        marker: str | None = None,
        tip_fmt: str | None = None,
        tip_label: str | None = None,
    ) -> 'Tracks':
        """Add a scatter track — point cloud rendered with ``gl.POINTS``.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format frame with columns ``chrom``, the column named by
            ``x``, the column named by ``y``, and any optional
            ``group_by`` column.
        name : str
            Track display label shown in the left panel.
        x : str, default ``'pos'``
            Column for genomic position.
        y : str, default ``'value'``
            Column for the y-axis value.
        group_by : str, optional
            Column whose unique values become separate groups (each
            drawn with its own colour).
        color_map : dict, optional
            ``{group_value: '#rrggbb'}``. Overridden by ``color``/``c``
            when those are supplied.
        height : int, default 60
            Track height in CSS pixels.
        y_range : tuple of (float, float), optional
            ``(y_min, y_max)``. Auto-computed from data with 5% padding
            when omitted.
        point_size : int, default 3
            Point diameter in CSS pixels. Aliases: ``s`` / ``size``.
        alpha : float, optional
            Opacity in ``[0, 1]`` applied to every group. Defaults to
            ``0.85`` (the prior visual). If ``color`` carries an alpha
            channel and ``alpha`` is not given, that alpha is lifted out.
        color, c : Any, optional
            Single matplotlib-style colour applied to all groups.
            Overrides ``color_map``. Examples: ``'C0'``, ``'red'``,
            ``'#ff8800'``, ``'#ff880040'``, ``(1, 0, 0, 0.5)``.
        s, size : float, optional
            Aliases for ``point_size``.
        marker : str, optional
            Accepted only as ``None`` / ``'.'`` / ``'s'`` (filled square,
            the only shape WebGL ``gl.POINTS`` produces). Other values
            raise :class:`NotImplementedError` rather than silently
            rendering a square.
        tip_fmt : str, optional
            Python format string for tooltips. Available keys:
            ``{group}``, ``{value}``, ``{x}``.
        tip_label : str, optional
            Heading shown above the tooltip body. Defaults to
            ``f'{name}:'``.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.

        Raises
        ------
        TypeError
            If conflicting aliases are passed (e.g. both ``s`` and
            ``point_size``, or both ``color`` and ``c``).
        ValueError
            If ``alpha`` is outside ``[0, 1]``.
        NotImplementedError
            If ``marker`` is anything other than ``None`` / ``'.'`` /
            ``'s'``.
        """
        return self._add_xy_track(
            df, name, 'scatter', x, y, group_by,
            color_map, height, y_range, point_size=point_size,
            tip_fmt=tip_fmt, tip_label=tip_label,
            alpha=alpha, color=color, c=c, s=s, size=size, marker=marker,
        )

    # ── add_line_track ───────────────────────────────────────────────────────
    def add_line_track(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        x: str = 'pos',
        y: str = 'value',
        group_by: str | None = None,
        color_map: dict | None = None,
        height: int = 60,
        y_range: tuple[float, float] | None = None,
        step: str | None = None,
        alpha: float | None = None,
        color: Any = None,
        c: Any = None,
        lw: float | None = None,
        linewidth: float | None = None,
        ls: str | None = None,
        linestyle: str | None = None,
        tip_fmt: str | None = None,
        tip_label: str | None = None,
    ) -> 'Tracks':
        """Add a line track — one connected polyline per group.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format frame with columns ``chrom``, the column named by
            ``x``, the column named by ``y``, and any optional
            ``group_by`` column.
        name : str
            Track display label shown in the left panel.
        x : str, default ``'pos'``
            Column for genomic position.
        y : str, default ``'value'``
            Column for the y-axis value.
        group_by : str, optional
            Column whose unique values become separate groups (each
            drawn as its own polyline).
        color_map : dict, optional
            ``{group_value: '#rrggbb'}``. Overridden by ``color``/``c``
            when those are supplied.
        height : int, default 60
            Track height in CSS pixels.
        y_range : tuple of (float, float), optional
            ``(y_min, y_max)``. Auto-computed from data with 5% padding
            when omitted.
        step : str, optional
            Step mode for a staircase line, matching matplotlib's
            ``step`` parameter. One of ``'pre'``, ``'post'``, ``'mid'``.
            ``None`` gives a straight-line interpolation between samples.
        alpha : float, optional
            Opacity in ``[0, 1]`` applied to every group. Defaults to
            ``1.0``. If ``color`` carries an alpha channel and ``alpha``
            is not given, that alpha is lifted out.
        color, c : Any, optional
            Single matplotlib-style colour applied to all groups.
            Overrides ``color_map``. Examples: ``'C0'``, ``'red'``,
            ``'#ff8800'``, ``'#ff880040'``, ``(1, 0, 0, 0.5)``.
        lw, linewidth : float, optional
            Accepted only as ``None`` or ``1`` — WebGL2 fixes line width
            at 1 device pixel. Other values raise
            :class:`NotImplementedError`. For thicker bands use
            :meth:`add_fill_track`.
        ls, linestyle : str, optional
            Accepted only as ``None`` / ``'-'`` / ``'solid'``. Dashed
            and dotted lines raise :class:`NotImplementedError`.
        tip_fmt : str, optional
            Python format string for tooltips. Available keys:
            ``{group}``, ``{value}``, ``{x}``.
        tip_label : str, optional
            Heading shown above the tooltip body. Defaults to
            ``f'{name}:'``.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.

        Raises
        ------
        TypeError
            If conflicting aliases are passed (e.g. both ``lw`` and
            ``linewidth``, or both ``color`` and ``c``).
        ValueError
            If ``alpha`` is outside ``[0, 1]`` or ``step`` is not in
            ``{None, 'pre', 'post', 'mid'}``.
        NotImplementedError
            For ``linewidth`` other than ``None`` / ``1`` or ``linestyle``
            other than ``None`` / ``'-'`` / ``'solid'``.
        """
        return self._add_xy_track(
            df, name, 'line', x, y, group_by,
            color_map, height, y_range, tip_fmt=tip_fmt,
            tip_label=tip_label, step=step,
            alpha=alpha, color=color, c=c,
            lw=lw, linewidth=linewidth, ls=ls, linestyle=linestyle,
        )

    def _add_xy_track(
        self,
        df: pd.DataFrame,
        name: str,
        track_type: str,
        x: str,
        y: str,
        group_by: str | None,
        color_map: dict | None,
        height: int,
        y_range: tuple[float, float] | None,
        point_size: int = 3,
        tip_fmt: str | None = None,
        tip_label: str | None = None,
        step: str | None = None,
        alpha: float | None = None,
        color: Any = None,
        c: Any = None,
        s: float | None = None,
        size: float | None = None,
        marker: str | None = None,
        lw: float | None = None,
        linewidth: float | None = None,
        ls: str | None = None,
        linestyle: str | None = None,
    ) -> 'Tracks':
        """Shared implementation for scatter and line tracks.

        Accepts matplotlib-style aliases (``c``/``color``, ``s``/``size``,
        ``alpha``, ``marker``, ``lw``/``linewidth``, ``ls``/``linestyle``)
        and routes the supported subset to the renderer; rejects the
        rest with explicit errors so silent no-ops don't mislead the
        caller.

        Parameters
        ----------
        df, name, x, y, group_by, color_map, height, y_range, tip_fmt, tip_label
            See :meth:`add_scatter_track` and :meth:`add_line_track` for
            the user-facing semantics; values flow through unchanged.
        track_type : {'scatter', 'line'}
            Which renderer branch to emit. Drives which of the
            scatter-only / line-only kwargs are accepted.
        point_size, alpha, color, c, s, size, marker, lw, linewidth, ls, linestyle
            matplotlib-style styling kwargs. The validator collapses
            aliases, lifts an alpha channel out of ``color`` when the
            user did not set ``alpha`` explicitly, and raises for
            kwargs that the WebGL2 renderer cannot honour today.
        step : {'pre', 'post', 'mid', None}, optional
            Line-only staircase mode (see :meth:`add_line_track`).

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.

        Raises
        ------
        TypeError
            On conflicting aliases or alias use on the wrong track type.
        ValueError
            If ``alpha`` is outside ``[0, 1]`` or ``step`` is unknown.
        NotImplementedError
            For unsupported ``marker`` shapes, ``linewidth`` ≠ 1, or
            non-solid ``linestyle`` values.
        """
        # ── matplotlib alias collapse ─────────────────────────────────────
        if c is not None and color is not None:
            raise TypeError("pass either `color` or `c`, not both")
        if color is None:
            color = c

        if track_type == 'scatter':
            sz_aliases = [v for v in (s, size) if v is not None]
            if len(sz_aliases) > 1:
                raise TypeError("pass at most one of `s`, `size`")
            if sz_aliases and point_size != 3:
                raise TypeError(
                    "pass either `point_size` or `s`/`size`, not both"
                )
            if sz_aliases:
                point_size = sz_aliases[0]
            if marker is not None and marker not in ('.', 's'):
                raise NotImplementedError(
                    f"marker={marker!r}: only filled-square points are "
                    f"rendered (the WebGL `gl.POINTS` primitive). Pass "
                    f"`marker=None`, `'.'`, or `'s'`, or omit the kwarg."
                )
            for nm, v in (('lw', lw), ('linewidth', linewidth),
                          ('ls', ls), ('linestyle', linestyle)):
                if v is not None:
                    raise TypeError(
                        f"`{nm}` is not valid for scatter tracks"
                    )
        else:  # 'line'
            for nm, v in (('s', s), ('size', size), ('marker', marker)):
                if v is not None:
                    raise TypeError(
                        f"`{nm}` is not valid for line tracks"
                    )
            lw_aliases = [v for v in (lw, linewidth) if v is not None]
            if len(lw_aliases) > 1:
                raise TypeError("pass at most one of `lw`, `linewidth`")
            if lw_aliases and lw_aliases[0] not in (1, 1.0):
                raise NotImplementedError(
                    f"linewidth={lw_aliases[0]!r}: WebGL2 fixes line width "
                    f"at 1 device pixel. For thicker bands consider "
                    f"`add_fill_track`."
                )
            ls_aliases = [v for v in (ls, linestyle) if v is not None]
            if len(ls_aliases) > 1:
                raise TypeError("pass at most one of `ls`, `linestyle`")
            if ls_aliases and ls_aliases[0] not in ('-', 'solid'):
                raise NotImplementedError(
                    f"linestyle={ls_aliases[0]!r}: only solid lines are "
                    f"currently supported."
                )

        # ── colour + alpha resolution ─────────────────────────────────────
        # If `color` carries alpha and the user did not set `alpha`, lift
        # the alpha out so the JS uAlpha uniform receives it. The colour
        # is always shipped as opaque hex (the JS hexRGB strips alpha).
        single_color_hex: str | None = None
        if color is not None:
            opaque_hex, color_alpha = _split_alpha(color)
            single_color_hex = opaque_hex
            if alpha is None and color_alpha is not None:
                alpha = color_alpha

        if alpha is not None:
            alpha = float(alpha)
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"alpha must be in [0, 1]; got {alpha!r}")

        color_map = _resolve_color_mapping(color_map)
        tid = self._tid()
        if group_by and group_by in df.columns:
            groups = sorted(df[group_by].dropna().unique(), key=str)
        else:
            groups = ['all']
            group_by = None

        if single_color_hex is not None:
            # Track-level `color`/`c` overrides per-group colour_map entirely.
            color_map = {g: single_color_hex for g in groups}
        elif color_map is None:
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

        xy_out: dict = {}
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
            **(({'alpha': alpha} if alpha is not None else {})),
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
        y: str | None = None,
        y_lo: str | None = None,
        y_hi: str | None = None,
        group_by: str | None = None,
        color_map: dict | None = None,
        color_pos: str | None = 'C0',
        color_neg: str | None = 'C3',
        height: int = 60,
        y_range: tuple[float, float] | None = None,
        baseline: float | None = None,
        step: str | None = None,
        tip_fmt: str | None = None,
        tip_label: str | None = None,
    ) -> 'Tracks':
        """Add a fill-between track.

        Either supply ``y`` (fill from a single curve to ``baseline``) or
        both ``y_lo`` and ``y_hi`` (fill between two curves). When
        ``baseline`` is set, the area above and below it can be drawn in
        distinct colours.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format frame with columns ``chrom``, the column named
            by ``x``, plus the relevant y column(s).
        name : str
            Track display label shown in the left panel.
        x : str, default ``'pos'``
            Column for genomic position.
        y : str, optional
            Column for a single y value. Fill extends from this curve to
            ``baseline``. Mutually exclusive with ``y_lo`` / ``y_hi``.
        y_lo : str, optional
            Column for the lower y boundary.
        y_hi : str, optional
            Column for the upper y boundary.
        group_by : str, optional
            Column whose unique values become separate groups.
        color_map : dict, optional
            ``{group_value: '#rrggbb'}``. When omitted, group colours
            cycle through ``'C0'``..``'C9'``.
        color_pos : str, optional, default ``'C0'``
            Colour for the region above ``baseline``. Resolved with
            :func:`resolve_color`.
        color_neg : str, optional, default ``'C3'``
            Colour for the region below ``baseline``.
        height : int, default 60
            Track height in CSS pixels.
        y_range : tuple of (float, float), optional
            ``(y_min, y_max)``. Auto-computed from data with 5% padding
            when omitted.
        baseline : float, optional
            Y value at which the positive / negative colour split
            occurs. Defaults to ``0`` in single-``y`` mode. In
            ``y_lo`` / ``y_hi`` mode it defaults to ``None``, which
            disables dual colouring (each group is drawn in a single
            colour from ``color_map``). Pass an explicit number in lo/hi
            mode to re-enable the split.
        step : str, optional
            Step mode for a staircase fill, matching matplotlib's
            ``fill_between(step=...)``. One of ``'pre'``, ``'post'``,
            ``'mid'``. ``None`` gives smooth linear interpolation.
        tip_fmt : str, optional
            Python format string for tooltips. Available keys:
            ``{group}``, ``{lo}``, ``{hi}``, ``{x}``.
        tip_label : str, optional
            Heading shown above the tooltip body. Defaults to
            ``f'{name}:'``.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.

        Raises
        ------
        ValueError
            If both ``y`` and ``y_lo`` / ``y_hi`` are supplied or if
            ``step`` is not in ``{None, 'pre', 'post', 'mid'}``.
        """
        color_map = _resolve_color_mapping(color_map)
        color_pos = resolve_color(color_pos)
        color_neg = resolve_color(color_neg)
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
            # In lo/hi mode, leave baseline as None unless the user gave one.
            # No baseline -> no pos/neg split, single colour per group.

        tid = self._tid()
        if group_by and group_by in df.columns:
            groups = sorted(df[group_by].dropna().unique(), key=str)
        else:
            groups = ['all']
            group_by = None

        default_palette = [resolve_color(f'C{i}') for i in range(10)]
        if color_map is None:
            color_map = {g: default_palette[i % len(default_palette)]
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

        fill_out: dict = {}
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

        dual = baseline is not None

        def _group_cfg(i, g):
            base = color_map.get(g, default_palette[i % len(default_palette)])
            entry = {'id': str(i), 'name': str(g), 'color': base}
            if dual:
                entry['colorPos'] = color_pos or base
                entry['colorNeg'] = color_neg or '#888888'
            return entry

        cfg = {
            'id': tid, 'type': 'fill', 'name': name, 'height': height,
            'yMin': yMin, 'yMax': yMax, 'baseline': baseline,
            'groups': [_group_cfg(i, g) for i, g in enumerate(groups)],
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
        group_by: str | None = None,
        color_map: dict | None = None,
        height: int = 60,
        y_range: tuple[float, float] | None = None,
        bin_width: float | None = None,
        stack: bool = True,
        density_windows: tuple | int = (256, 1024, 4096),
        aggregate: str = 'mean',
        tip_fmt: str | None = None,
        tip_label: str | None = None,
    ) -> 'Tracks':
        """Add a histogram track — vertical bars centred on each ``x``.

        Parameters
        ----------
        df : pandas.DataFrame
            Long-format frame with columns ``chrom``, the column named
            by ``x``, and the column named by ``y``.
        name : str
            Track display label shown in the left panel.
        x : str, default ``'pos'``
            Column for genomic position (the bin centre).
        y : str, default ``'value'``
            Column for bar height.
        group_by : str, optional
            Column whose unique values become separate groups.
        color_map : dict, optional
            ``{group_value: '#rrggbb'}``. Defaults to the built-in
            palette cycled in group order.
        height : int, default 60
            Track height in CSS pixels.
        y_range : tuple of (float, float), optional
            ``(y_min, y_max)``. Auto-computed from data with 5% padding
            when omitted; always includes the baseline ``0``.
        bin_width : float, optional
            Width of each bar in base pairs at the finest zoom level.
            Auto-computed from the median spacing of ``x`` when omitted.
        stack : bool, default True
            If True, group contributions at each x-bin stack on top of
            each other (positive values above ``0``, negative below).
            Group order determines stacking order. Set to False to
            overlap bars from different groups instead.
        density_windows : tuple of int or int, default ``(256, 1024, 4096)``
            Bin counts for the multi-resolution LOD, matching the
            segment-track convention. When zoomed out, the renderer
            picks the finest level whose bin maps to at least ~2 CSS
            pixels, keeping the visible bar count bounded by
            chromosome size. Pass an empty tuple to disable LOD (always
            render the original bars).
        aggregate : {'mean', 'sum', 'max'}, default ``'mean'``
            How to combine multiple input bins that fall inside a single
            LOD bin. Only relevant when LOD is engaged.
        tip_fmt : str, optional
            Python format string for tooltips. Available keys:
            ``{group}``, ``{value}``, ``{x}``.
        tip_label : str, optional
            Heading shown above the tooltip body. Defaults to
            ``f'{name}:'``.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.

        Raises
        ------
        ValueError
            If ``aggregate`` is not one of ``'mean'``, ``'sum'``,
            ``'max'``.
        """
        color_map = _resolve_color_mapping(color_map)
        if isinstance(density_windows, int):
            density_windows = (density_windows,)
        density_windows = tuple(int(n) for n in density_windows if int(n) > 0)
        if aggregate not in ('mean', 'sum', 'max'):
            raise ValueError(
                f"aggregate must be 'mean', 'sum', or 'max'; got {aggregate!r}"
            )
        tid = self._tid()
        if group_by and group_by in df.columns:
            groups = sorted(df[group_by].dropna().unique(), key=str)
        else:
            groups = ['all']
            group_by = None

        if color_map is None:
            color_map = {g: self._PALETTE[i % len(self._PALETTE)]
                         for i, g in enumerate(groups)}

        # Auto bin width from median spacing
        if bin_width is None:
            xs = df.sort_values(x)[x].to_numpy()
            if len(xs) > 1:
                diffs = np.diff(xs)
                bin_width = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
            else:
                bin_width = 1.0

        stacked = bool(stack) and len(groups) > 1

        # Compute y-range.  When stacking, the effective maximum is the
        # column-wise sum of positive group contributions (and minimum is the
        # sum of negative contributions), not the largest single bar.
        if y_range is not None:
            yMin, yMax = y_range
        elif stacked:
            # Per-(chrom, x) column totals split into +/-.
            pos_sum = np.zeros(1, dtype=np.float64)
            neg_sum = np.zeros(1, dtype=np.float64)
            for chrom_val, cdf in df.groupby('chrom'):
                # sum of y per x, separately for positive and negative contributions
                pos_by_x = cdf.loc[cdf[y] > 0].groupby(x)[y].sum()
                neg_by_x = cdf.loc[cdf[y] < 0].groupby(x)[y].sum()
                if not pos_by_x.empty:
                    pos_sum = np.append(pos_sum, float(pos_by_x.max()))
                if not neg_by_x.empty:
                    neg_sum = np.append(neg_sum, float(neg_by_x.min()))
            yMin = float(min(0.0, neg_sum.min()))
            yMax = float(max(0.0, pos_sum.max()))
            pad = (yMax - yMin) * 0.05 or 0.5
            yMax += pad
        else:
            # Always include the baseline (0) in the visible range so bars
            # have a stable foot to grow from.
            yMin = min(0.0, float(df[y].min()))
            yMax = max(0.0, float(df[y].max()))
            pad = (yMax - yMin) * 0.05 or 0.5
            yMax += pad
            if yMin < 0:
                yMin -= pad

        hist_out: dict = {}
        for chrom_val, cdf in df.groupby('chrom'):
            chrom = str(chrom_val)
            hist_out[chrom] = {}

            if stacked:
                csz = self.chrom_sizes.get(
                    chrom, int(cdf[x].max()) + int(bin_width)
                )

                # ── Original-resolution stacked bars ───────────────────────
                # For each x, maintain running pos/neg cursors that advance
                # as each group adds its value (positive above 0, negative
                # below 0).
                x_cursor: dict[float, dict[str, float]] = {}
                base_per_group: dict[str, np.ndarray] = {}
                xs_per_group:   dict[str, np.ndarray] = {}
                ys_per_group:   dict[str, np.ndarray] = {}
                for gi, group in enumerate(groups):
                    gid = str(gi)
                    gdf = cdf[cdf[group_by] == group] if group_by else cdf
                    if gdf.empty:
                        base_per_group[gid] = np.zeros(0, dtype=np.float32)
                        xs_per_group[gid]   = np.zeros(0, dtype=np.float64)
                        ys_per_group[gid]   = np.zeros(0, dtype=np.float64)
                        continue
                    gdf = gdf.sort_values(x)
                    xs_arr = gdf[x].to_numpy(dtype=np.float64)
                    ys_arr = gdf[y].to_numpy(dtype=np.float64)
                    n = len(xs_arr)
                    arr = np.empty(n * 3, dtype=np.float32)
                    for k in range(n):
                        xv = float(xs_arr[k])
                        yv = float(ys_arr[k])
                        cur = x_cursor.setdefault(xv, {'pos': 0.0, 'neg': 0.0})
                        if yv >= 0:
                            lo = cur['pos']; hi = cur['pos'] + yv
                            cur['pos'] = hi
                        else:
                            hi = cur['neg']; lo = cur['neg'] + yv
                            cur['neg'] = lo
                        arr[k * 3]     = xv
                        arr[k * 3 + 1] = lo
                        arr[k * 3 + 2] = hi
                    base_per_group[gid] = arr
                    xs_per_group[gid]   = xs_arr
                    ys_per_group[gid]   = ys_arr

                # ── LOD aggregations ──────────────────────────────────────
                # For each level n, aggregate each group's values into n
                # equal bins, then re-stack the per-bin per-group aggregates
                # in group order (positive contributions above 0, negative
                # below 0).
                lods_per_group: dict[str, dict[str, str]] = {
                    str(gi): {} for gi in range(len(groups))
                }
                if csz > 0:
                    for nbin in density_windows:
                        # Per-group, per-bin aggregate value at this level.
                        agg: dict[str, np.ndarray] = {}
                        for gi in range(len(groups)):
                            gid = str(gi)
                            xs_g = xs_per_group[gid]
                            ys_g = ys_per_group[gid]
                            lvl = np.zeros(nbin, dtype=np.float64)
                            if xs_g.size:
                                bins = np.clip(
                                    (xs_g / csz * nbin).astype(int),
                                    0, nbin - 1,
                                )
                                sums = np.zeros(nbin, dtype=np.float64)
                                np.add.at(sums, bins, ys_g)
                                if aggregate == 'sum':
                                    lvl = sums
                                elif aggregate == 'max':
                                    lvl_init = np.full(
                                        nbin, -np.inf, dtype=np.float64
                                    )
                                    np.maximum.at(lvl_init, bins, ys_g)
                                    lvl = np.where(
                                        np.isfinite(lvl_init), lvl_init, 0.0
                                    )
                                else:  # 'mean'
                                    counts = np.zeros(nbin, dtype=np.float64)
                                    np.add.at(counts, bins, 1)
                                    with np.errstate(
                                        divide='ignore', invalid='ignore'
                                    ):
                                        lvl = np.where(
                                            counts > 0, sums / counts, 0.0
                                        )
                            agg[str(gi)] = lvl

                        # Re-stack per-bin in group order.
                        pos_cursor = np.zeros(nbin, dtype=np.float64)
                        neg_cursor = np.zeros(nbin, dtype=np.float64)
                        bin_w = csz / nbin
                        x_centres = (np.arange(nbin) + 0.5) * bin_w
                        for gi in range(len(groups)):
                            gid = str(gi)
                            v = agg[gid]
                            lo = np.where(v >= 0, pos_cursor,
                                          neg_cursor + v)
                            hi = np.where(v >= 0, pos_cursor + v,
                                          neg_cursor)
                            pos_cursor = np.where(v >= 0,
                                                  pos_cursor + v, pos_cursor)
                            neg_cursor = np.where(v < 0,
                                                  neg_cursor + v, neg_cursor)
                            arr = np.empty(nbin * 3, dtype=np.float32)
                            arr[0::3] = x_centres.astype(np.float32)
                            arr[1::3] = lo.astype(np.float32)
                            arr[2::3] = hi.astype(np.float32)
                            lods_per_group[gid][str(nbin)] = self._pack_f32(arr)

                for gi in range(len(groups)):
                    gid = str(gi)
                    base = base_per_group[gid]
                    if base.size == 0 and not lods_per_group[gid]:
                        hist_out[chrom][gid] = ''
                        continue
                    hist_out[chrom][gid] = {
                        'base': self._pack_f32(base) if base.size else '',
                        'lods': lods_per_group[gid],
                    }
            else:
                csz = self.chrom_sizes.get(chrom, int(cdf[x].max()) + int(bin_width))
                for gi, group in enumerate(groups):
                    gid = str(gi)
                    gdf = cdf[cdf[group_by] == group] if group_by else cdf
                    if gdf.empty:
                        hist_out[chrom][gid] = {'base': '', 'lods': {}}
                        continue
                    gdf = gdf.sort_values(x)
                    xs_arr = gdf[x].to_numpy(dtype=np.float64)
                    ys_arr = gdf[y].to_numpy(dtype=np.float64)
                    base_arr = np.empty(len(gdf) * 2, dtype=np.float32)
                    base_arr[0::2] = xs_arr.astype(np.float32)
                    base_arr[1::2] = ys_arr.astype(np.float32)
                    base_b64 = self._pack_f32(base_arr)

                    # Build LOD aggregations: each level n is a length-n array
                    # giving the per-bin aggregate of the user's y values that
                    # fall inside that bin (along the chromosome).
                    lods: dict[str, str] = {}
                    if csz > 0:
                        for n in density_windows:
                            bins = np.clip(
                                (xs_arr / csz * n).astype(int), 0, n - 1
                            )
                            sums = np.zeros(n, dtype=np.float64)
                            np.add.at(sums, bins, ys_arr)
                            if aggregate == 'sum':
                                lvl = sums
                            elif aggregate == 'max':
                                lvl = np.full(n, np.nan, dtype=np.float64)
                                # np.maximum.at handles duplicate indices.
                                lvl_init = np.full(n, -np.inf, dtype=np.float64)
                                np.maximum.at(lvl_init, bins, ys_arr)
                                # Bins that received no data: leave as 0.
                                lvl = np.where(np.isfinite(lvl_init),
                                               lvl_init, 0.0)
                            else:  # 'mean'
                                counts = np.zeros(n, dtype=np.float64)
                                np.add.at(counts, bins, 1)
                                with np.errstate(divide='ignore',
                                                 invalid='ignore'):
                                    lvl = np.where(counts > 0,
                                                   sums / counts, 0.0)
                            lods[str(n)] = self._pack_f32(
                                lvl.astype(np.float32)
                            )
                    hist_out[chrom][gid] = {'base': base_b64, 'lods': lods}

        cfg = {
            'id': tid, 'type': 'histogram', 'name': name, 'height': height,
            'yMin': yMin, 'yMax': yMax, 'binWidth': bin_width,
            'stacked': stacked,
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

    # ── Overlays ─────────────────────────────────────────────────────────────
    def add_vlines(
        self,
        positions,
        chrom: str | None = None,
        *,
        color: str = '#ff4444',
        linewidth: float = 1.0,
        alpha: float = 1.0,
        dash: list | None = None,
        replace: bool = False,
    ) -> 'Tracks':
        """Add thin vertical guide lines across all tracks.

        Parameters
        ----------
        positions : int | Iterable[int] | dict[str, Iterable[int]]
            Base-pair positions. If a dict, keys are chromosome names
            and values are iterables of positions on that chromosome;
            the ``chrom`` argument is ignored in that case.
        chrom : str, optional
            Chromosome for the positions. Defaults to the current
            viewport chromosome. Ignored when ``positions`` is a dict.
        color : str, default ``'#ff4444'``
            Matplotlib-style colour spec; resolved through
            :func:`_resolve_color_mapping` so names like ``'C0'`` or
            ``'red'`` work.
        linewidth : float, default ``1.0``
            Line width in CSS pixels.
        alpha : float, default ``1.0``
            Opacity in ``[0, 1]``. Values outside the range are clamped.
        dash : list of int, optional
            Canvas dash pattern, e.g. ``[4, 3]``. ``None`` = solid.
        replace : bool, default ``False``
            If True, replace existing vlines. Otherwise append.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.
        """
        resolved = _resolve_color_mapping(color)

        if isinstance(positions, dict):
            items = [(str(c), positions[c]) for c in positions]
        else:
            c = chrom if chrom is not None else self.viewport.get('chrom', '')
            if isinstance(positions, (int, float)):
                positions = [positions]
            items = [(str(c), positions)]

        new_entries = []
        for c, pos_iter in items:
            for p in pos_iter:
                new_entries.append({
                    'chrom': c,
                    'pos':   int(p),
                    'color': resolved,
                    'width': float(linewidth),
                    'alpha': float(max(0.0, min(1.0, alpha))),
                    'dash':  list(dash) if dash else [],
                })

        self.vlines = new_entries if replace else [*self.vlines, *new_entries]
        return self

    def clear_vlines(self) -> 'Tracks':
        """Remove all vertical marker lines.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.
        """
        self.vlines = []
        return self

    def add_vblocks(
        self,
        spans,
        chrom: str | None = None,
        *,
        color: str = '#ffcc44',
        alpha: float = 0.1,
        edgecolor: str | None = None,
        edgewidth: float = 0.0,
        dash: list | None = None,
        replace: bool = False,
    ) -> 'Tracks':
        """Add shaded vertical blocks spanning all tracks.

        Parameters
        ----------
        spans : tuple | Iterable[tuple] | dict[str, Iterable[tuple]]
            ``(start, end)`` pairs in base pairs. If a dict, keys are
            chromosome names and values are iterables of pairs on that
            chromosome; the ``chrom`` argument is ignored in that case.
        chrom : str, optional
            Chromosome for the spans. Defaults to the current viewport
            chromosome. Ignored when ``spans`` is a dict.
        color : str, default ``'#ffcc44'``
            Fill colour.
        alpha : float, default ``0.1``
            Fill opacity in ``[0, 1]``.
        edgecolor : str, optional
            Outline colour. Defaults to ``color`` if an ``edgewidth`` is
            given.
        edgewidth : float, default ``0.0``
            Outline width in CSS pixels. ``0`` disables the outline.
        dash : list of int, optional
            Dash pattern for the outline, e.g. ``[4, 3]``.
        replace : bool, default ``False``
            If True, replace existing vblocks. Otherwise append.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.
        """
        fill  = _resolve_color_mapping(color)
        edge  = _resolve_color_mapping(edgecolor) if edgecolor else ''

        def _pairs(it):
            out = []
            if (isinstance(it, (tuple, list)) and len(it) == 2
                    and all(isinstance(x, (int, float)) for x in it)):
                out.append((int(it[0]), int(it[1])))
            else:
                for pair in it:
                    out.append((int(pair[0]), int(pair[1])))
            return out

        if isinstance(spans, dict):
            items = [(str(c), _pairs(spans[c])) for c in spans]
        else:
            c = chrom if chrom is not None else self.viewport.get('chrom', '')
            items = [(str(c), _pairs(spans))]

        a = float(max(0.0, min(1.0, alpha)))
        new_entries = []
        for c, pairs in items:
            for s, e in pairs:
                if e < s:
                    s, e = e, s
                new_entries.append({
                    'chrom':     c,
                    'start':     int(s),
                    'end':       int(e),
                    'color':     fill,
                    'alpha':     a,
                    'edgecolor': edge,
                    'edgewidth': float(edgewidth),
                    'dash':      list(dash) if dash else [],
                })

        self.vblocks = new_entries if replace else [*self.vblocks, *new_entries]
        return self

    def clear_vblocks(self) -> 'Tracks':
        """Remove all vertical blocks.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.
        """
        self.vblocks = []
        return self

    # ── Navigation ───────────────────────────────────────────────────────────
    def set_viewport(self, chrom: str, start: int, end: int) -> 'Tracks':
        """Move the viewport to a new chromosome and base-pair window.

        Parameters
        ----------
        chrom : str
            Chromosome name (must exist in the widget's ``chrom_sizes``).
        start : int
            Inclusive start position in base pairs. Negative values are
            clamped to ``0``.
        end : int
            Exclusive end position in base pairs. Values past the end
            of the chromosome are clamped to its length.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.
        """
        csz = self.chrom_sizes.get(str(chrom), end)
        self.viewport = {
            'chrom': str(chrom),
            'start': int(max(0, start)),
            'end':   int(min(csz, end)),
        }
        return self

    def zoom_to(self, chrom: str, center: int, window: int = 1_000_000) -> 'Tracks':
        """Centre the viewport on a position with a fixed window width.

        Parameters
        ----------
        chrom : str
            Chromosome name.
        center : int
            Genomic position to centre on, in base pairs.
        window : int, default ``1_000_000``
            Total window width in base pairs. The viewport is set to
            ``[center - window // 2, center + window // 2)`` then
            clamped to the chromosome bounds by :meth:`set_viewport`.

        Returns
        -------
        Tracks
            ``self``, to support fluent chaining.
        """
        half = window // 2
        return self.set_viewport(chrom, center - half, center + half)
