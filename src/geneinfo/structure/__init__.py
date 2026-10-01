"""Interactive 3D protein structure viewer for notebooks (JupyterLab, Jupyter Notebook, VS Code).

    from structure_viewer import show_structure
    show_structure("TTLL10", highlight="362-365, 375, 377")

The first argument is a gene/protein name or a UniProt accession (shown as the AlphaFold model)
or a PDB ID (experimental structure). Structures are downloaded once and cached in ./structures.
Rendering uses 3Dmol.js, loaded from a CDN, so the notebook front end needs internet access. Each
viewer is plain HTML/JavaScript output with its own controls, so no widget extensions are needed.

    from structure_viewer import alphamissense
    alphamissense("TTLL10")  # mean AlphaMissense pathogenicity per residue, as a list
"""
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

from IPython.display import HTML

__all__ = ["show_structure", "load_structure", "alphamissense"]

# --------------------------------------------------------------------------------------------
# Viewer: the HTML/JavaScript each call to show_structure outputs
# --------------------------------------------------------------------------------------------

JS_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"

VIEWER_CSS = """
.ttv { font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: #222; max-width: 100%; }
.ttv-controls, .ttv-footer { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px;
    padding: 6px 8px; background: #f3f4f6; border: 1px solid #d0d4da; }
.ttv-controls { border-bottom: none; }
.ttv-footer { border-top: none; }
.ttv label { display: inline-flex; align-items: center; gap: 4px; margin: 0; }
.ttv input[type=text], .ttv select, .ttv button { font: inherit; color: #222; background: #fff;
    border: 1px solid #b8bec7; border-radius: 3px; padding: 2px 6px; margin: 0; }
.ttv input[type=checkbox] { margin: 0; }
.ttv button { cursor: pointer; }
.ttv button:hover { background: #e6e9ee; }
.ttv input[type=color] { width: 30px; height: 24px; padding: 1px; border: 1px solid #b8bec7; background: #fff; }
.ttv-viewer { position: relative; width: 100%; background: #fff;
    outline: 1px solid #d0d4da; outline-offset: -1px; }  /* outline, not border: 3Dmol sizes the canvas to the box */
.ttv-item { display: inline-flex; align-items: center; gap: 4px; }
.ttv-swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px; border: 1px solid rgba(0,0,0,.3); }
.ttv button.ttv-remove { padding: 0 5px; line-height: 1.2; }
.ttv-msg { color: #b00020; }
.ttv-note, .ttv-source { color: #666; }
.ttv-swatch-base { border-style: dashed; border-color: #666; }
.ttv-bar { display: inline-block; width: 90px; height: 11px; border-radius: 2px; border: 1px solid rgba(0,0,0,.3); }
.ttv-msg:empty, .ttv-key:empty, .ttv-legend:empty { display: none; }
"""

CONTROLS_HTML = """
<div class="ttv-controls">
  <label>Residues <input class="ttv-res" type="text" placeholder="e.g. 362-365, 375 or all" size="20"></label>
  <input class="ttv-color" type="color" value="#ff0000" title="Highlight colour">
  <label>add as <select class="ttv-kind">
    <option value="highlight">highlight</option><option value="sticks">sticks</option></select></label>
  <button class="ttv-add">Add</button>
  <button class="ttv-clear">Clear</button>
  <label>Show as <select class="ttv-show">
    <option value="cartoon">cartoon</option><option value="sticks">sticks</option>
    <option value="cartoon+sticks">cartoon + sticks</option></select></label>
  <label>Colour by <select class="ttv-base">
    <option value="lightgrey">grey</option><option value="plddt">pLDDT</option>
    <option value="alphamissense">AlphaMissense</option>
    <option value="spectrum">rainbow N&rarr;C</option></select></label>
  <label><input class="ttv-sticks" type="checkbox"> side chains</label>
  <label><input class="ttv-elements" type="checkbox"> element colours</label>
  <label><input class="ttv-labels" type="checkbox"> labels</label>
  <label><input class="ttv-spin" type="checkbox"> spin</label>
  <button class="ttv-zoom-sel">Zoom to highlights</button>
  <button class="ttv-zoom-all">Zoom to all</button>
  <button class="ttv-copy-rot" title="Copy the current orientation as an (x, y, z) tuple for show_structure(rotation=...)">Copy rotation</button>
  <button class="ttv-save" title="Save the current view as a PNG image at 4&times; the on-screen size">Save PNG</button>
</div>"""

VIEWER_HTML = r"""
<style>__CSS__</style>
<div id="__ID__" class="ttv" data-lm-suppress-shortcuts="true" style="width: __WIDTH__">
  __CONTROLS__
  <div class="ttv-viewer" style="height: __HEIGHT__"></div>
  <div class="ttv-footer">
    <span class="ttv-source">__SOURCE__</span><span class="ttv-key"></span><span class="ttv-legend"></span>
    <span class="ttv-msg">Loading 3D viewer&hellip; If this message stays, the notebook's JavaScript did not run:
      re-run the cell (in JupyterLab, trust the notebook).</span>
  </div>
</div>
<script>
(function () {
  const cfg = __CONFIG__;
  const PLDDT_BANDS = [[90, 101, "#0053D6", ">90"], [70, 90, "#65CBF3", "70-90"],
                       [50, 70, "#FFDB13", "50-70"], [0, 50, "#FF7D45", "<50"]];
  const AM_STOPS = [[0, "#3b6fd4"], [0.45, "#c9c9c9"], [1, "#c8102e"]];  // AlphaMissense: benign - ambiguous - pathogenic
  const PALETTE = ["#ff0000", "#ff00ff", "#00a651", "#8e44ad", "#8c510a", "#111111"];  // stand out from grey and pLDDT colours

  const plddtColour = (b) => (PLDDT_BANDS.find(([lo, hi]) => b >= lo && b < hi) || PLDDT_BANDS[3])[2];

  function amColour(score) {  // AM_STOPS interpolated in 21 steps, so residues can be styled in batches
    const s = Math.round(Math.min(1, Math.max(0, score)) * 20) / 20;
    let i = 1;
    while (i < AM_STOPS.length - 1 && s > AM_STOPS[i][0]) i++;
    const [x0, c0] = AM_STOPS[i - 1], [x1, c1] = AM_STOPS[i], t = (s - x0) / (x1 - x0);
    const rgb = (c) => [1, 3, 5].map((k) => parseInt(c.slice(k, k + 2), 16));
    const a = rgb(c0), b = rgb(c1);
    return "#" + a.map((v, k) => Math.round(v + (b[k] - v) * t).toString(16).padStart(2, "0")).join("");
  }
  const LABEL_STYLE = {fontSize: 11, fontColor: "black", backgroundColor: "white", backgroundOpacity: 0.7};
  const ELEMENT_COLORS = {N: "#3050F8", O: "#FF0D0D", S: "#E6C200", SE: "#FFA100", H: "#FFFFFF"};
  // side chain plus CA, so the sticks join the cartoon; proline's N closes its ring
  const isSideChain = (a) => !["C", "O", "OXT"].includes(a.atom) && (a.atom !== "N" || a.resn === "PRO");

  // Orientation as (x, y, z) angles in degrees: from the default view, rotate x degrees about x, then
  // y about y, then z about z, as viewer.rotate(x, "x") etc. do. 3Dmol post-multiplies its quaternion,
  // so this is q = qx * qy * qz (quaternions as [x, y, z, w]).
  function anglesToQuaternion(deg) {
    const axis = (i, d) => { const q = [0, 0, 0, Math.cos(d * Math.PI / 360)]; q[i] = Math.sin(d * Math.PI / 360); return q; };
    const mul = (a, b) => [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                           a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                           a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                           a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
    return mul(mul(axis(0, deg[0]), axis(1, deg[1])), axis(2, deg[2]));
  }

  function quaternionToAngles(q) {  // inverse of anglesToQuaternion, via the rotation matrix Rx Ry Rz
    const n = Math.hypot(...q), [x, y, z, w] = q.map((v) => v / n);
    const r02 = 2 * (x * z + y * w), deg = (rad) => rad * 180 / Math.PI;
    let a, b = Math.asin(Math.max(-1, Math.min(1, r02))), c;
    if (Math.abs(r02) < 0.999999) {
      a = Math.atan2(-2 * (y * z - x * w), 1 - 2 * (x * x + y * y));
      c = Math.atan2(-2 * (x * y - z * w), 1 - 2 * (y * y + z * z));
    } else {  // y = +-90 degrees: x and z rotate about the same axis, so put it all in x
      a = Math.atan2(2 * (y * z + x * w), 1 - 2 * (x * x + z * z));
      c = 0;
    }
    return [a, b, c].map((v) => Math.round(deg(v) * 10) / 10 + 0);  // + 0 turns -0 into 0
  }

  function load3Dmol(url) {
    if (window.$3Dmol) return Promise.resolve();
    if (!window.__ttvLoad3Dmol) {
      window.__ttvLoad3Dmol = new Promise(function (resolve, reject) {
        // 3Dmol.js is a UMD bundle: if an AMD loader (require.js) is on the page, as in some notebook
        // front ends, it registers as a module and never defines window.$3Dmol. Offering CommonJS
        // globals while it loads makes it initialise immediately (the same trick py3Dmol uses).
        const had = {exports: "exports" in window, module: "module" in window};
        const saved = {exports: window.exports, module: window.module};
        if (typeof window.exports !== "object") window.exports = {};
        if (typeof window.module !== "object") window.module = {};
        const restore = function () {
          for (const k of ["exports", "module"]) { if (had[k]) window[k] = saved[k]; else delete window[k]; }
        };
        const tag = document.createElement("script");
        tag.src = url;
        tag.onload = function () {
          restore();
          if (window.$3Dmol) resolve(); else reject(new Error("3Dmol.js loaded but did not initialise"));
        };
        tag.onerror = function () {
          restore();
          window.__ttvLoad3Dmol = null;
          reject(new Error("could not load 3Dmol.js from " + url + " (no internet access?)"));
        };
        document.head.appendChild(tag);
      });
    }
    return window.__ttvLoad3Dmol;
  }

  // Front ends differ in when they attach output to the page, so wait until the viewer box is visible.
  function whenVisible(id) {
    return new Promise(function (resolve, reject) {
      const start = Date.now();
      (function poll() {
        const root = document.getElementById(id);
        const box = root && root.querySelector(".ttv-viewer");
        if (box && box.isConnected && box.clientWidth > 0) return resolve(root);
        if (Date.now() - start > 120000) return reject(new Error("viewer element never became visible"));
        setTimeout(poll, 100);
      })();
    });
  }

  const span = (a, b) => (a === b ? String(a) : a + "-" + b);

  function compact(res) {  // [362, 363, 364, 365, 375] -> "362-365, 375"
    const runs = [];
    for (const r of res) {
      if (runs.length && r === runs[runs.length - 1][1] + 1) runs[runs.length - 1][1] = r;
      else runs.push([r, r]);
    }
    return runs.map(([a, b]) => span(a, b)).join(", ");
  }

  // Residues to draw with sticks and labels: everything except consecutive stretches longer than
  // cfg.maxDetail (e.g. a whole domain), which stay cartoon only. Scattered residues are never limited.
  function detailResidues(res) {
    const out = [];
    let run = [];
    for (const r of res.concat([Infinity])) {
      if (run.length && r === run[run.length - 1] + 1) { run.push(r); continue; }
      if (run.length <= cfg.maxDetail) out.push(...run);
      run = [r];
    }
    return out;
  }

  // Inverted wheel zoom. 3Dmol binds its wheel handler when a viewer is created, so the prototype is
  // patched (once, before the first viewer); only viewers flagged _ttvInvertZoom are affected. The
  // event is passed through a Proxy with the wheel direction negated; trackpad pinches (ctrl+wheel)
  // are left as they are, so pinching out still zooms in.
  function patchWheelZoom() {
    const proto = $3Dmol.GLViewer && $3Dmol.GLViewer.prototype;
    if (!proto || !proto._handleMouseScroll || proto._ttvWheelPatched) return;
    const original = proto._handleMouseScroll;
    proto._handleMouseScroll = function (ev) {
      if (!this._ttvInvertZoom || ev.ctrlKey) return original.call(this, ev);
      return original.call(this, new Proxy(ev, {get: function (target, key) {
        if (key === "detail" || key === "wheelDelta") return -target[key];
        const value = target[key];
        return typeof value === "function" ? value.bind(target) : value;
      }}));
    };
    proto._ttvWheelPatched = true;
  }

  function init(root) {
    const q = (sel) => root.querySelector(sel);
    const msg = (text) => { q(".ttv-msg").textContent = text || ""; };
    msg("");
    patchWheelZoom();
    const viewer = $3Dmol.createViewer(q(".ttv-viewer"), {backgroundColor: "white"});
    viewer._ttvInvertZoom = true;
    root.viewer = viewer;  // handy for debugging from the browser console
    // keep the canvas matched to its box when the notebook or window changes width
    if (window.ResizeObserver) new ResizeObserver(() => viewer.resize()).observe(q(".ttv-viewer"));
    viewer.addModel(cfg.pdb, "pdb");

    const cas = viewer.selectedAtoms({atom: "CA"});
    const present = new Set(cas.map((a) => a.resi));
    const first = Math.min(...present), last = Math.max(...present);
    const byProperty = () => state.base === "plddt" || state.base === "alphamissense";
    // residues grouped by colour for the property-based colourings (optionally restricted to `res`)
    function colourGroups(res) {
      const groups = new Map();
      for (const a of cas) {
        if (res && !res.has(a.resi)) continue;
        const colour = state.base === "plddt" ? plddtColour(a.b)
                     : cfg.am[a.resi] === undefined ? "#d0d0d0" : amColour(cfg.am[a.resi]);
        if (!groups.has(colour)) groups.set(colour, []);
        groups.get(colour).push(a.resi);
      }
      return groups;
    }
    const allRes = [...present].sort((a, b) => a - b);
    const rainbow = {prop: "resi", gradient: new $3Dmol.Gradient.Sinebow(last, first)};  // = cartoon "spectrum"
    const state = {groups: cfg.groups.map((g) => ({color: g.color, res: g.res.slice()})),
                   stickGroups: cfg.stickGroups.map((g) => ({res: g.res.slice()})),
                   show: cfg.show, base: cfg.base, sticks: cfg.sticks, labels: cfg.labels,
                   elements: cfg.elements};

    viewer.setHoverable({}, true,
      function (atom, v) {
        if (!atom.label) {
          const am = cfg.am && cfg.am[atom.resi] !== undefined ? ", AlphaMissense " + cfg.am[atom.resi].toFixed(2) : "";
          atom.label = v.addLabel(atom.resn + " " + atom.resi + "  (" + cfg.bLabel + " " + atom.b.toFixed(1) + am + ")",
            {position: atom, backgroundColor: "black", backgroundOpacity: 0.8, fontColor: "white", fontSize: 12});
        }
      },
      function (atom, v) {
        if (atom.label) { v.removeLabel(atom.label); delete atom.label; }
      });

    // residues drawn as all-atom sticks in the base colouring
    function stickResidues() {
      return new Set(state.show === "cartoon" ? state.stickGroups.flatMap((g) => g.res) : allRes);
    }

    // draw `res` (narrowed by the extra selection `sel`) as `shape` ("stick"/"sphere") in the base colouring
    // stick colouring: all atoms in `carbon`, or N/O/S/... by element when state.elements is on
    const elem = (carbon) => {
      const map = {};
      for (const e in ELEMENT_COLORS) map[e] = state.elements ? ELEMENT_COLORS[e] : carbon;
      map.C = carbon;
      return {prop: "elem", map: map};
    };

    function addBase(res, radius, sel, shape) {
      const add = (sub, scheme) => viewer.addStyle(Object.assign({resi: sub}, sel || {}),
                                                   {[shape || "stick"]: {radius: radius, colorscheme: scheme}});
      if (byProperty()) {
        for (const [colour, sub] of colourGroups(new Set(res))) add(sub, elem(colour));
      } else if (state.base === "spectrum") {
        add(res, rainbow);
      } else {
        add(res, elem(state.base));
      }
    }

    function applyStyles() {
      viewer.removeAllLabels();
      const cartoon = state.show !== "sticks";
      // 1. the whole chain in the base colouring
      viewer.setStyle({}, {});
      if (cartoon) {
        if (byProperty()) {
          viewer.setStyle({}, {cartoon: {color: "#d0d0d0"}});
          for (const [colour, res] of colourGroups()) viewer.setStyle({resi: res}, {cartoon: {color: colour}});
        } else {
          viewer.setStyle({}, {cartoon: {color: state.base}});  // a colour, or "spectrum"
        }
      }
      const inSticks = stickResidues();
      if (inSticks.size) addBase([...inSticks], cfg.stickRadius);
      // 2. highlights on top (a group without a colour keeps the base colouring)
      for (const g of state.groups) {
        const scheme = elem(g.color);
        const add = (res, radius, sel, shape) => g.color === null ? addBase(res, radius, sel, shape)
          : viewer.addStyle(Object.assign({resi: res}, sel), {[shape]: {radius: radius, colorscheme: scheme}});
        if (cartoon && g.color !== null) viewer.addStyle({resi: g.res}, {cartoon: {color: g.color}});
        // residues already drawn as sticks: recolour the whole residue (slightly thicker)
        const drawn = g.res.filter((r) => inSticks.has(r));
        if (drawn.length) add(drawn, cfg.highlightRadius, {}, "stick");
        const detail = detailResidues(g.res);
        const rest = detail.filter((r) => !inSticks.has(r));
        if (state.sticks && rest.length) {  // add the side chain in the highlight colour
          add(rest, cfg.highlightRadius, {predicate: isSideChain}, "stick");
          // glycine has no side chain: mark its CA with a small sphere instead
          add(rest, 2 * cfg.highlightRadius, {resn: "GLY", atom: "CA"}, "sphere");
        }
        if (state.labels) {  // anchored on CA so the label does not cover the side chain
          for (const ca of viewer.selectedAtoms({resi: detail, atom: "CA"})) {
            viewer.addLabel(ca.resn + ca.resi, Object.assign({position: ca}, LABEL_STYLE));
          }
        }
      }
      viewer.render();
    }

    function swatch(color) {
      const s = document.createElement("span");
      s.className = "ttv-swatch";
      s.style.background = color;
      return s;
    }

    function renderLegend() {
      const key = q(".ttv-key"), legend = q(".ttv-legend");
      key.textContent = "";
      if (state.base === "plddt") {
        key.append("pLDDT:");
        for (const [, , color, text] of PLDDT_BANDS) {
          const item = document.createElement("span");
          item.className = "ttv-item";
          item.append(swatch(color), text);
          key.append(" ", item);
        }
      } else if (state.base === "alphamissense") {
        const bar = document.createElement("span");
        bar.className = "ttv-bar";
        bar.style.background = "linear-gradient(to right, " + AM_STOPS.map(([x, c]) => c + " " + x * 100 + "%").join(", ") + ")";
        key.append("AlphaMissense (mean over substitutions): benign ", bar, " pathogenic");
      }
      legend.textContent = "";
      const inSticks = stickResidues();
      state.stickGroups.forEach(function (g, i) {
        const item = document.createElement("span");
        item.className = "ttv-item";
        const tag = document.createElement("span");
        tag.className = "ttv-note";
        tag.textContent = "sticks:";
        item.append(tag, g.res.length === allRes.length ? "all" : compact(g.res));
        if (cfg.controls) {
          const x = document.createElement("button");
          x.className = "ttv-remove";
          x.textContent = "×";
          x.title = "Remove these sticks";
          x.onclick = function () { state.stickGroups.splice(i, 1); restyle(); };
          item.append(x);
        }
        legend.append(item);
      });
      state.groups.forEach(function (g, i) {
        const item = document.createElement("span");
        item.className = "ttv-item";
        const sw = swatch(g.color === null ? "transparent" : g.color);
        if (g.color === null) { sw.classList.add("ttv-swatch-base"); sw.title = "In the base colouring"; }
        item.append(sw, compact(g.res));
        const nDetail = detailResidues(g.res).length;
        if (nDetail < g.res.length && (state.sticks || state.labels) && g.res.some((r) => !inSticks.has(r))) {
          const note = document.createElement("span");
          note.className = "ttv-note";
          note.textContent = nDetail ? "(stretches over " + cfg.maxDetail + " as cartoon only)" : "(cartoon only)";
          note.title = "Consecutive stretches of more than " + cfg.maxDetail +
                       " residues are drawn without side chains and labels";
          item.append(note);
        }
        if (cfg.controls) {
          const x = document.createElement("button");
          x.className = "ttv-remove";
          x.textContent = "\u00d7";
          x.title = "Remove this group";
          x.onclick = function () { state.groups.splice(i, 1); restyle(); };
          item.append(x);
        }
        legend.append(item);
      });
      if (cfg.controls && !state.groups.length && !state.stickGroups.length) legend.textContent = "No highlights";
    }

    function restyle() { applyStyles(); renderLegend(); }

    function highlighted() {
      return [...new Set(state.groups.flatMap((g) => g.res))].sort((a, b) => a - b);
    }

    function zoomTo(res, ms) { viewer.zoomTo(res && res.length ? {resi: res} : {}, ms); }

    // "362-365, 375 377" -> residue numbers present in the structure; reports what was skipped
    function parse(text) {
      if (text.trim().toLowerCase() === "all") return {res: allRes.slice(), outside: [], bad: []};
      const res = new Set(), outside = [], bad = [];
      for (const tok of text.replace(/[\[\]()]/g, " ").trim().replace(/\s*-\s*/g, "-").split(/[\s,;]+/).filter(Boolean)) {
        const m = tok.match(/^(\d+)(?:-(\d+))?$/);
        if (!m) { bad.push(tok); continue; }
        let a = Number(m[1]), b = m[2] === undefined ? a : Number(m[2]);
        if (a > b) [a, b] = [b, a];
        for (let r = Math.max(a, first); r <= Math.min(b, last); r++) if (present.has(r)) res.add(r);
        if (a < first) outside.push(span(a, Math.min(b, first - 1)));
        if (b > last) outside.push(span(Math.max(a, last + 1), b));
      }
      return {res: [...res].sort((x, y) => x - y), outside: outside, bad: bad};
    }

    // The current view as a PNG at `scale` times the on-screen size (longest side at most 8192 px,
    // which all WebGL implementations handle). 3Dmol sizes its canvas and labels by
    // window.devicePixelRatio, so that is raised briefly while the view is re-rendered.
    function savePNG(scale) {
      const box = q(".ttv-viewer");
      const ratio = Math.min(scale, 8192 / Math.max(box.clientWidth, box.clientHeight));
      const own = Object.getOwnPropertyDescriptor(window, "devicePixelRatio");
      let uri;
      try {
        Object.defineProperty(window, "devicePixelRatio", {value: ratio, configurable: true});
        viewer.resize();
        viewer.render();
        uri = viewer.pngURI();
      } finally {
        if (own) Object.defineProperty(window, "devicePixelRatio", own); else delete window.devicePixelRatio;
        viewer.resize();
        viewer.render();
      }
      // a link with `download` saves the file in browsers; VS Code turns it into a save dialog
      const link = document.createElement("a");
      link.href = uri;
      link.download = cfg.fileName + ".png";
      link.style.display = "none";
      root.append(link);
      link.click();
      link.remove();
    }
    root.savePNG = savePNG;  // e.g. root.savePNG(8) from the browser console

    // clipboard API where allowed, else the older execCommand route (some webviews block the API)
    function copyText(text) {
      const fallback = function () {
        const area = document.createElement("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.opacity = "0";
        root.append(area);
        area.select();
        const ok = document.execCommand("copy");
        area.remove();
        if (!ok) throw new Error("the browser refused");
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text).catch(fallback);
      }
      return Promise.resolve().then(fallback);
    }

    function wireControls() {
      const input = q(".ttv-res"), picker = q(".ttv-color"), base = q(".ttv-base");
      const nextColour = () => {  // red is hard to see on the AlphaMissense colouring
        const palette = state.base === "alphamissense" ? PALETTE.slice(1) : PALETTE;
        picker.value = palette[state.groups.length % palette.length];
      };
      if (!cfg.plddt) base.querySelector('option[value="plddt"]').remove();  // B column is not pLDDT
      if (!cfg.am) base.querySelector('option[value="alphamissense"]').remove();  // no scores for this protein
      if (![...base.options].some((o) => o.value === state.base)) base.add(new Option(state.base, state.base));
      base.value = state.base;
      q(".ttv-show").value = state.show;
      q(".ttv-sticks").checked = state.sticks;
      q(".ttv-labels").checked = state.labels;
      q(".ttv-elements").checked = state.elements;
      q(".ttv-spin").checked = cfg.spin;
      nextColour();

      function add() {
        const p = parse(input.value);
        if (p.bad.length) return msg("Cannot read " + p.bad.join(", ") + ". Use numbers and ranges, e.g. 362-365, 375");
        if (p.res.length) {
          if (q(".ttv-kind").value === "sticks") {
            state.stickGroups.push({res: p.res});
          } else {
            state.groups.push({color: picker.value, res: p.res});
            nextColour();
          }
          input.value = "";
        }
        msg(p.outside.length ? "Not in the structure (" + first + "-" + last + "), skipped: " + p.outside.join(", ")
            : p.res.length ? "" : "Type residue numbers first, e.g. 362-365, 375");
        restyle();
      }

      q(".ttv-add").onclick = add;
      input.addEventListener("keydown", function (e) {
        if (e.key === "Enter") { e.preventDefault(); add(); }
      });
      // keep key presses in the controls away from the notebook's keyboard shortcuts
      for (const type of ["keydown", "keypress", "keyup"]) {
        q(".ttv-controls").addEventListener(type, (e) => e.stopPropagation());
      }
      q(".ttv-clear").onclick = function () { state.groups = []; state.stickGroups = []; msg(""); nextColour(); restyle(); };
      base.onchange = function () { state.base = base.value; restyle(); };
      q(".ttv-show").onchange = function (e) { state.show = e.target.value; restyle(); };
      q(".ttv-sticks").onchange = function (e) { state.sticks = e.target.checked; restyle(); };
      q(".ttv-labels").onchange = function (e) { state.labels = e.target.checked; restyle(); };
      q(".ttv-elements").onchange = function (e) { state.elements = e.target.checked; restyle(); };
      q(".ttv-spin").onchange = function (e) { viewer.spin(e.target.checked ? "y" : false); };
      q(".ttv-zoom-sel").onclick = function () { zoomTo(highlighted(), 500); };
      q(".ttv-zoom-all").onclick = function () { zoomTo(null, 500); };
      q(".ttv-copy-rot").onclick = function () {
        const button = this, text = "(" + quaternionToAngles(viewer.getView().slice(4, 8)).join(", ") + ")";
        copyText(text).then(function () {
          msg("");
          button.textContent = "Copied";
          button.title = "Copied " + text;
          setTimeout(function () { button.textContent = "Copy rotation"; }, 1500);
        }, function (err) { msg("Could not copy (" + err.message + "); rotation=" + text); });
      };
      q(".ttv-save").onclick = function () {
        try { savePNG(4); } catch (err) { console.error(err); msg("Could not save the image: " + err.message); }
      };
    }

    restyle();
    if (cfg.rotation) {
      const view = viewer.getView();
      view.splice(4, 4, ...anglesToQuaternion(cfg.rotation));
      viewer.setView(view);
    }
    zoomTo(cfg.zoom ? highlighted() : null, 0);
    viewer.render();
    if (cfg.spin) viewer.spin("y");
    if (cfg.controls) wireControls();
  }

  Promise.all([load3Dmol(cfg.jsUrl), whenVisible(cfg.id)])
    .then(function (r) { init(r[1]); })
    .catch(function (err) {
      console.error(err);
      const root = document.getElementById(cfg.id);
      if (root) root.querySelector(".ttv-msg").textContent = "Viewer error: " + err.message;
    });
})();
</script>
"""

# --------------------------------------------------------------------------------------------
# Structures: resolve a name, download the model once, cache it on disk and in memory
# --------------------------------------------------------------------------------------------

_ORGANISMS = {
    "human": 9606, "mouse": 10090, "rat": 10116, "zebrafish": 7955, "fly": 7227, "drosophila": 7227,
    "worm": 6239, "c. elegans": 6239, "yeast": 559292, "arabidopsis": 3702, "chicken": 9031,
    "cow": 9913, "pig": 9823, "dog": 9615, "chimpanzee": 9598, "macaque": 9544, "e. coli": 83333,
}
_UNIPROT_ACCESSION = re.compile(r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?")
_PDB_ID = re.compile(r"[0-9][A-Za-z0-9]{3}")


@dataclass
class Structure:
    label: str                            # what was asked for, e.g. "TTLL10 (human)"
    source: str                           # where it came from, shown under the viewer
    pdb: str                              # PDB-format coordinates, a single chain
    residues: dict                        # residue number -> (three-letter name, B column)
    plddt: bool                           # True if the B column holds AlphaFold's pLDDT
    alphamissense: dict = None            # residue number -> mean AlphaMissense pathogenicity (human AlphaFold models)


_STRUCTURES = {}  # in-memory cache for this kernel session


def _taxid(organism):
    if isinstance(organism, Integral) or str(organism).isdigit():
        return int(organism)
    try:
        return _ORGANISMS[str(organism).strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown organism {organism!r}; use an NCBI taxonomy ID or one of: "
                         + ", ".join(_ORGANISMS)) from None


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _download(url, path):
    try:
        urllib.request.urlretrieve(url, path)
    except urllib.error.HTTPError as e:
        raise ValueError(f"Download failed ({e.code}) for {url}") from None


def _resolve_name(name, taxid, cache_dir):
    """Gene or protein name -> (UniProt accession, label) via the UniProt REST API, cached in names.json."""
    index_path = cache_dir / "names.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    key = f"{name.upper()}|{taxid}"
    if key not in index:
        base = f"(gene_exact:{name} OR id:{name}) AND organism_id:{taxid}"
        results = []
        for query in (base + " AND reviewed:true", base):  # prefer Swiss-Prot entries
            url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
                {"query": query, "fields": "accession,gene_primary,organism_name", "format": "json", "size": 1})
            results = _fetch_json(url).get("results", [])
            if results:
                break
        if not results:
            raise ValueError(f"No UniProt entry found for {name!r} in organism {taxid}")
        hit = results[0]
        gene = (hit.get("genes") or [{}])[0].get("geneName", {}).get("value", name)
        organism = hit["organism"].get("commonName") or hit["organism"]["scientificName"]
        index[key] = [hit["primaryAccession"], f"{gene} ({organism.lower()})"]
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
    return tuple(index[key])


def _alphamissense_means(path):
    """Residue number -> mean AlphaMissense pathogenicity of the 19 substitutions, from the AlphaFold DB CSV."""
    total, count = {}, {}
    with path.open() as f:
        next(f)  # header: protein_variant,am_pathogenicity,am_class
        for line in f:
            variant, score, _ = line.rstrip("\n").split(",")  # e.g. M1A,0.1552,LBen
            resi = int(variant[1:-1])
            total[resi] = total.get(resi, 0.0) + float(score)
            count[resi] = count.get(resi, 0) + 1
    return {resi: total[resi] / count[resi] for resi in total}


def _alphafold_model(accession, cache_dir):
    """The AlphaFold DB model for a UniProt accession, downloaded once, plus its AlphaMissense table
    (human proteins only). Returns (PDB text, model id, {residue: mean pathogenicity} or None)."""
    index_path = cache_dir / "alphafold.json"  # accession -> file names
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    info = index.get(accession)
    if info is None or not (cache_dir / info["pdb"]).exists():
        try:
            entries = _fetch_json(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")
        except urllib.error.HTTPError as e:
            raise ValueError(f"No AlphaFold model for UniProt {accession} (HTTP {e.code})") from None
        except urllib.error.URLError:
            cached = sorted(cache_dir.glob(f"AF-{accession}-F1-model_v*.pdb"))  # offline: use what we have
            if not cached:
                raise
            info = {"pdb": cached[-1].name, "am": None}
        else:
            entry = next((e for e in entries if e["entryId"] == f"AF-{accession}-F1"), None)
            if entry is None:
                raise ValueError(f"No AlphaFold model for UniProt {accession}")
            info = {"pdb": Path(entry["pdbUrl"]).name, "am": None}
            if not (cache_dir / info["pdb"]).exists():
                _download(entry["pdbUrl"], cache_dir / info["pdb"])
            if entry.get("amAnnotationsUrl"):
                am_path = cache_dir / Path(entry["amAnnotationsUrl"]).name
                try:
                    if not am_path.exists():
                        _download(entry["amAnnotationsUrl"], am_path)
                    info["am"] = am_path.name
                except ValueError as e:
                    print(f"Note: AlphaMissense scores not available: {e}")
            index[accession] = info
            index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
    pdb = (cache_dir / info["pdb"]).read_text()
    am = _alphamissense_means(cache_dir / info["am"]) if info.get("am") else None
    return pdb, info["pdb"].split("-model")[0], am  # e.g. "AF-Q6ZVT0-F1"


def _pdb_entry(pdb_id, chain, cache_dir):
    """One chain (first model) of an RCSB PDB entry, downloaded once. Returns (pdb text, chain, all chains)."""
    pdb_id = pdb_id.upper()
    path = cache_dir / f"{pdb_id}.pdb"
    if not path.exists():
        _download(f"https://files.rcsb.org/download/{pdb_id}.pdb", path)
    atoms = []
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):  # first model only (NMR ensembles)
            break
        if line.startswith("ATOM"):
            atoms.append(line)
    chains = list(dict.fromkeys(line[21] for line in atoms))
    if not chains:
        raise ValueError(f"No protein chains in PDB {pdb_id}")
    if chain is None:
        chain = chains[0]
    elif chain not in chains:
        raise ValueError(f"PDB {pdb_id} has no chain {chain!r}; chains: {', '.join(chains)}")
    return "\n".join(line for line in atoms if line[21] == chain) + "\nEND\n", chain, chains


def _residues(pdb):
    """residue number -> (three-letter name, B column) from the CA atoms."""
    return {int(line[22:26]): (line[17:20], float(line[60:66]))
            for line in pdb.splitlines() if line.startswith("ATOM") and line[12:16].strip() == "CA"}


def load_structure(protein, organism="human", chain=None, cache_dir="structures"):
    """Fetch (once) and return the Structure for a gene/protein name, UniProt accession or PDB ID.

    Names and accessions give the AlphaFold model; PDB IDs give one chain of the experimental structure.
    """
    cache_dir = Path(cache_dir)
    key = (str(protein).strip().upper(), _taxid(organism), chain, str(cache_dir.resolve()))
    if key in _STRUCTURES:
        return _STRUCTURES[key]
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = str(protein).strip()
    if _PDB_ID.fullmatch(name):
        pdb, chain, chains = _pdb_entry(name, chain, cache_dir)
        residues = _residues(pdb)
        source = (f"PDB {name.upper()} chain {chain}" + (f" (chains: {', '.join(chains)})" if len(chains) > 1 else "")
                  + f" \u00b7 {len(residues)} residues")
        structure = Structure(f"PDB {name.upper()}", source, pdb, residues, plddt=False)
    else:
        if _UNIPROT_ACCESSION.fullmatch(name):
            accession, label = name, f"UniProt {name}"
        else:
            accession, label = _resolve_name(name, _taxid(organism), cache_dir)
            label = f"{label} · UniProt {accession}"
        pdb, model, am = _alphafold_model(accession, cache_dir)
        residues = _residues(pdb)
        structure = Structure(label, f"{label} · AlphaFold {model} · {len(residues)} residues",
                              pdb, residues, plddt=True, alphamissense=am)
    _STRUCTURES[key] = structure
    return structure


# --------------------------------------------------------------------------------------------
# Residue specifications
# --------------------------------------------------------------------------------------------

def expand_residues(spec):
    """Turn a residue specification into a sorted list of residue numbers.

    Accepts an int (375), an iterable of ints/strings ([362, 375], range(362, 366), ["362-365", 375])
    or a string with commas/spaces and ranges ("362-365, 375 377", "[3, 15, 16]").
    """
    if spec is None:
        return []
    if isinstance(spec, Integral):
        items = [spec]
    elif isinstance(spec, str):
        spec = re.sub(r"[\[\]()]", " ", spec)  # allow a pasted list like "[3, 15, 16]"
        items = re.split(r"[,;\s]+", re.sub(r"\s*-\s*", "-", spec.strip()))
    else:
        items = list(spec)
    residues = set()
    for item in items:
        if isinstance(item, Integral):
            residues.add(int(item))
            continue
        item = str(item).strip()
        if not item:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", item)
        if m:
            a, b = sorted((int(m[1]), int(m[2])))
            residues.update(range(a, b + 1))
        elif item.isdigit():
            residues.add(int(item))
        else:
            raise ValueError(f"Cannot parse residue specification {item!r}")
    return sorted(residues)


def compact(residues):
    """[362, 363, 364, 365, 375, 377] -> '362-365, 375, 377'"""
    runs = []
    for r in sorted(residues):
        if runs and r == runs[-1][1] + 1:
            runs[-1][1] = r
        else:
            runs.append([r, r])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


def parse_residues(spec, residues):
    """Like expand_residues, but keeps only numbers present in `residues` (reporting the rest); "all" = every residue."""
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return sorted(residues)
    requested = expand_residues(spec)
    missing = [r for r in requested if r not in residues]
    if missing:
        print(f"Warning: not in the structure ({min(residues)}-{max(residues)}), ignored: {compact(missing)}")
    return [r for r in requested if r in residues]


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

def alphamissense(protein, residues=None, organism="human", cache_dir="structures"):
    """Mean AlphaMissense pathogenicity per residue, as a list of floats.

    Each value is the mean over the 19 possible substitutions at that residue (as shown in AlphaFold DB
    and used by show_structure(base="alphamissense")): near 0 = benign, near 1 = pathogenic.

    protein   : a gene or protein name ("TTLL10") or a UniProt accession ("Q6ZVT0"); human only, since
                AlphaMissense scores exist for human AlphaFold models only.
    residues  : residues to return, e.g. "362-365, 375" or [362, 375] (same forms as `highlight` in
                show_structure). Default: all residues of the model, so element i is residue i + 1.
                Values come in increasing residue order.
    organism, cache_dir : as for show_structure.

    Residues without a score are nan. Use numpy.array(...) for an array.
    """
    structure = load_structure(protein, organism, None, cache_dir)
    if not structure.alphamissense:
        raise ValueError("AlphaMissense scores exist for human AlphaFold models only; "
                         f"none for {structure.label}")
    selected = sorted(structure.residues) if residues is None else parse_residues(residues, structure.residues)
    return [structure.alphamissense.get(r, float("nan")) for r in selected]


# --------------------------------------------------------------------------------------------
# The viewer
# --------------------------------------------------------------------------------------------

def show_structure(protein, highlight=None, highlight_color="red", base="lightgrey", show="cartoon",
                   sticks_for=None, side_chains=True, element_colors=False, labels=True, stick_radius=0.2, highlight_radius=0.3,
                   max_detail=30, zoom=False, rotation=None, spin=False, controls=True, width="100%", height=600,
                   organism="human", chain=None, cache_dir="structures"):
    """Show a protein structure as a rotatable 3D cartoon with highlighted residues.

    protein   : a gene or protein name ("TTLL10", "TTL10_HUMAN"; looked up in UniProt for `organism`)
                or a UniProt accession ("Q6ZVT0"), both shown as the AlphaFold model, or a PDB ID
                ("1UBQ") for an experimental structure (one chain, see `chain`). Downloaded once and
                cached in `cache_dir`. Residue numbers follow the model (UniProt numbering for AlphaFold).
    highlight : residues to highlight, e.g. 375, [362, 375], range(362, 366), "362-365, 375, 377",
                or a dict {colour: residues} to highlight several groups in different colours.
    highlight_color : colour for `highlight` when it is not a dict (CSS colour name or hex code), or
                None to keep the base colouring (the residues are only marked by side chains and labels).
                In a dict, None as the colour does the same for that group.
    base      : colour of the rest of the chain: a colour name/hex, "spectrum" (rainbow from N- to
                C-terminus), "plddt" (AlphaFold confidence; AlphaFold models only) or "alphamissense"
                (mean AlphaMissense pathogenicity of the 19 possible substitutions at each residue,
                blue = benign, red = pathogenic; human AlphaFold models only; pick a highlight colour
                such as "magenta" or "green" that stands out from it).
    show      : how to draw the whole chain: "cartoon" (default), "sticks" (all atoms) or "cartoon+sticks".
    sticks_for: residues to draw as all-atom sticks in the base colouring on top of the cartoon,
                e.g. "355-385", [362, 375] or "all". Highlighted residues among them are recoloured.
    side_chains : draw the side chains of highlighted residues as sticks in the highlight colour.
    element_colors : colour N, O and S atoms of all sticks by element (blue, red, yellow), with only
                the carbons in the highlight/base colour. Default False: sticks are one colour.
    labels    : label the highlighted residues with name and number.
    stick_radius    : radius (Å) of the sticks drawn with `show` or `sticks_for`.
    highlight_radius: radius (Å) of the sticks of highlighted residues.
    max_detail: consecutive stretches longer than this are drawn as cartoon only (no sticks or
                labels), so highlighting a whole domain stays readable. Scattered residues are
                always drawn with sticks, however many there are.
    zoom      : zoom in on the highlighted residues instead of the whole protein.
    rotation  : orientation as (x, y, z) angles in degrees, e.g. (30, -45, 0): starting from the default
                view, rotate x degrees about the x axis, then y about y, then z about z. The
                "Copy rotation" button copies the current orientation in this form (rotation only;
                zoom and centring still follow `zoom`).
    spin      : rotate the structure continuously.
    controls  : show the controls for adding highlights, changing colours, zooming, copying the
                rotation (see `rotation`) and saving the view as a PNG image (4x the on-screen size,
                e.g. 4000 px wide for a 1000 px viewer).
    width, height : size of the viewer, in pixels (int) or as a CSS length string. The default
                width "100%" fills the notebook's width; e.g. width=1200, height=900 for a fixed
                size, or height="70vh" for 70% of the window height.
    organism  : for names: "human" (default), "mouse", "rat", ... or an NCBI taxonomy ID.
    chain     : for PDB IDs: the chain to show (default: the first protein chain).
    cache_dir : folder where downloaded structures are kept (default "structures").
    """
    structure = load_structure(protein, organism, chain, cache_dir)
    if show not in ("cartoon", "sticks", "cartoon+sticks"):
        raise ValueError('show must be "cartoon", "sticks" or "cartoon+sticks"')
    if base == "plddt" and not structure.plddt:
        raise ValueError(f'base="plddt" needs an AlphaFold model; {structure.label} is an experimental structure')
    if base == "alphamissense" and not structure.alphamissense:
        raise ValueError('base="alphamissense" needs AlphaMissense scores, which exist for human AlphaFold '
                         f'models only; none for {structure.label}')
    if isinstance(highlight, dict):
        groups = [(c, parse_residues(r, structure.residues)) for c, r in highlight.items()]
    else:
        groups = [(highlight_color, parse_residues(highlight, structure.residues))] if highlight is not None else []
    if rotation is not None:
        try:
            rotation = [float(a) for a in rotation]
        except (TypeError, ValueError):
            rotation = None
        if rotation is None or len(rotation) != 3:
            raise ValueError("rotation must be three angles in degrees, e.g. (30, -45, 0)")
    css_len = lambda v: f"{int(v)}px" if isinstance(v, (int, float)) else str(v)
    viewer_id = f"ttv-{uuid.uuid4().hex}"
    config = {
        "id": viewer_id, "jsUrl": JS_URL, "pdb": structure.pdb,
        "plddt": structure.plddt, "bLabel": "pLDDT" if structure.plddt else "B-factor",
        "am": {r: round(v, 3) for r, v in structure.alphamissense.items()} if structure.alphamissense else None,
        "groups": [{"color": c, "res": r} for c, r in groups if r],
        "stickGroups": [{"res": parse_residues(sticks_for, structure.residues)}] if sticks_for is not None else [],
        "show": show, "base": base, "sticks": side_chains, "elements": element_colors, "labels": labels, "maxDetail": max_detail,
        "stickRadius": float(stick_radius), "highlightRadius": float(highlight_radius),
        "zoom": zoom, "rotation": rotation, "spin": spin, "controls": controls,
        "fileName": re.sub(r"[^A-Za-z0-9_.-]+", "_", str(protein).strip()) or "structure",
    }
    page = (VIEWER_HTML
            .replace("__CSS__", VIEWER_CSS)
            .replace("__CONTROLS__", CONTROLS_HTML if controls else "")
            .replace("__ID__", viewer_id)
            .replace("__SOURCE__", html.escape(structure.source))
            .replace("__WIDTH__", css_len(width))
            .replace("__HEIGHT__", css_len(height))
            .replace("__CONFIG__", json.dumps(config).replace("</", "<\\/")))  # keep "</script>" out
    return HTML(page)
