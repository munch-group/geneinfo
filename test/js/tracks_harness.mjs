// Behavioural test of the Tracks widget's browser side (data fetching,
// upload bookkeeping, failure handling) in Node, with a stub DOM and a stub
// WebGL2 context that records uploads. Nothing is actually rendered.
//
// Usage: node tracks_harness.mjs <dir>
//   <dir>/widget.mjs   the widget's ES module (Tracks._esm)
//   <dir>/cases.json   model states and kernel replies, written by
//                      test/test_tracks_widget.py from real Python payloads
//
// Prints one PASS/FAIL line per check; exits 1 if any check fails.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const dir = process.argv[2];
const C = JSON.parse(fs.readFileSync(path.join(dir, 'cases.json'), 'utf8'));

let fails = 0;
const ok = (cond, msg) => { console.log((cond ? 'PASS ' : 'FAIL ') + msg); if (!cond) fails++; };

// ── Stub browser environment ────────────────────────────────────────────────
const stub = new Proxy(function () {}, {
  get: (t, p) => p === Symbol.toPrimitive ? () => 0 : (p === 'then' ? undefined : stub),
  apply: () => stub, set: () => true,
});
function withFallback(obj) {
  return new Proxy(obj, {
    get: (t, p) => (p in t ? t[p] : (typeof p === 'symbol' ? undefined : stub)),
    set: (t, p, v) => { t[p] = v; return true; },
  });
}
let now = 1000;
Object.defineProperty(globalThis, 'performance', { value: { now: () => now }, configurable: true });
let rafQ = [], inFrame = false;
globalThis.requestAnimationFrame = f => { rafQ.push(f); return rafQ.length; };
globalThis.cancelAnimationFrame = () => {};
let timers = [];                          // timers are driven by the test
globalThis.setTimeout = f => { timers.push(f); return timers.length; };
function runTimers() { const t = timers; timers = []; t.forEach(f => f()); }
globalThis.clearTimeout = () => {};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.window = withFallback({ addEventListener() {}, removeEventListener() {}, devicePixelRatio: 1 });
globalThis.document = withFallback({ addEventListener() {}, removeEventListener() {} });
globalThis.getComputedStyle = () => withFallback({ getPropertyValue: () => '' });
globalThis.devicePixelRatio = 1;

const origLog = console.log;
let errors = [], warns = [];
console.error = (...a) => errors.push(a.map(String).join(' '));
console.warn  = (...a) => warns.push(a.map(String).join(' '));

// WebGL2 stub: counts buffer/texture uploads made outside animation frames
// (i.e. data uploads, not per-frame scratch buffers); getError is scriptable.
// texImage2D checks the data size the way WebGL does: each row but the last
// is padded to the UNPACK_ALIGNMENT, and too little data is INVALID_OPERATION.
let uploads = 0, nextGlErrors = [], ctxLost = false, glErrFlag = 0, texRejects = 0;
const GL = { NO_ERROR: 0, OUT_OF_MEMORY: 0x505, INVALID_OPERATION: 0x502,
             UNPACK_ALIGNMENT: 0x0CF5, RED: 0x1903, RGB: 0x1907, RGBA: 0x1908 };
const TEXEL_BYTES = { [GL.RED]: 1, [GL.RGB]: 3, [GL.RGBA]: 4 };
function mkGL() {
  let align = 4;
  return new Proxy({}, { get: (t, p) => {
    if (p in GL) return GL[p];
    if (typeof p === 'string' && /^[A-Z0-9_]+$/.test(p)) return 0x9000 + p.length;
    switch (p) {
      case 'pixelStorei': return (pname, v) => { if (pname === GL.UNPACK_ALIGNMENT) align = v; };
      case 'texImage2D': return (tg, lvl, ifmt, w, h, b, fmt, type, data) => {
        if (!inFrame) uploads++;
        const row = w * (TEXEL_BYTES[fmt] || 4);
        const need = h > 0 ? (h - 1) * Math.ceil(row / align) * align + row : 0;
        if (data && data.length < need) { glErrFlag ||= GL.INVALID_OPERATION; texRejects++; }
      };
      case 'bufferData': case 'bufferSubData': case 'texSubImage2D':
        return () => { if (!inFrame) uploads++; };
      case 'getError': return () => {
        if (nextGlErrors.length) return nextGlErrors.shift();
        const e = glErrFlag; glErrFlag = 0; return e;
      };
      case 'isContextLost': return () => ctxLost;
      case 'getShaderParameter': case 'getProgramParameter': return () => true;
      case 'getParameter': return () => 8192;
      case 'getAttribLocation': return () => 0;
      case 'getExtension': return () => null;
      default: return () => ({});
    }
  }});
}
let texts = [];   // every string drawn on a 2-D overlay
function mk2D() {
  return new Proxy({}, {
    get: (t, p) => {
      if (p === 'fillText') return s => texts.push(String(s));
      if (p === 'measureText') return s => ({ width: String(s).length * 7 });
      if (p in t) return t[p];
      return () => {};
    },
    set: (t, p, v) => { t[p] = v; return true; },
  });
}
let listeners = {};
function mkEl(cls) {
  return withFallback({
    style: withFallback({ setProperty() {}, removeProperty() {} }), dataset: {},
    width: 300, height: 150, offsetWidth: 1200, offsetHeight: 400,
    value: '', textContent: '', innerHTML: '', options: [],
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener: (ev, f) => { (listeners[cls + ':' + ev] ||= []).push(f); },
    removeEventListener() {}, setAttribute() {}, removeAttribute() {},
    hasAttribute: () => cls === 'sv-help-pop',
    append() {}, appendChild() {}, contains: () => false, focus() {}, blur() {},
    querySelectorAll: () => [],
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1200, height: 400, right: 1200, bottom: 400 }),
    getContext: kind => (kind === 'webgl2' ? mkGL() : mk2D()),
  });
}
function mkRoot() {
  const els = {};
  return withFallback({
    innerHTML: '', style: withFallback({ setProperty() {} }), classList: { add() {}, remove() {} },
    querySelector: sel => (els[sel] ||= mkEl(sel.replace(/^\./, ''))),
    querySelectorAll: () => [], appendChild() {}, append() {}, addEventListener() {},
  });
}

// ── Binary helpers ──────────────────────────────────────────────────────────
// Buffers arrive as DataViews at odd byte offsets, as they can over a real
// comm, so the widget's alignment copy is exercised.
function b64ToView(b64) {
  const bin = Buffer.from(b64, 'base64');
  const ab = new ArrayBuffer(bin.length + 1);
  new Uint8Array(ab, 1).set(bin);
  return new DataView(ab, 1, bin.length);
}
const clone = x => JSON.parse(JSON.stringify(x));
function putViews(obj, paths, b64s) {
  paths.forEach((p, i) => {
    let o = obj;
    for (let j = 0; j < p.length - 1; j++) o = o[p[j]];
    o[p[p.length - 1]] = b64ToView(b64s[i]);
  });
  return obj;
}

// ── Mock kernel + model ─────────────────────────────────────────────────────
// The kernel answers _cmd_fetch_track from the replies Python produced; the
// revision in each reply is the kernel's current one. Replies are queued and
// delivered when the test calls kernel.flush(), so latency and a busy kernel
// can be simulated.
function mkKernel(caseName) {
  const replies = C[caseName].replies;
  return {
    revs: clone(C[caseName].state.track_revs || {}),
    queue: [], requests: [], broken: {},
    handle(content, model) {
      if (content.kind !== 'anywidget-command') return;
      this.requests.push({ name: content.name, ...content.msg });
      if (content.name !== '_cmd_fetch_track') {
        this.queue.push(() => model._emit('msg:custom', { id: content.id, kind: 'anywidget-command-response', response: null }, []));
        return;
      }
      // Like the real kernel, the reply is built when the request is
      // processed (flush), with the revision current at that moment.
      const { tid, chrom } = content.msg;
      this.queue.push(() => {
        const rep = replies[tid] && replies[tid][chrom];
        let resp, bufs = [];
        if (this.broken[tid]) resp = { tid, chrom, error: this.broken[tid] };
        else if (!rep) resp = { tid, chrom, error: 'track no longer exists' };
        else {
          resp = { ...clone(rep.response), rev: this.revs[tid] };
          bufs = rep.buffers.map(b64ToView);
        }
        model._emit('msg:custom', { id: content.id, kind: 'anywidget-command-response', response: resp }, bufs);
      });
    },
    flush() { const q = this.queue; this.queue = []; q.forEach(f => f()); },
  };
}
function mkModel(state, kernel) {
  const h = {};
  const model = {
    state,
    get: k => state[k],
    set: (k, v) => { state[k] = v; },
    save_changes() {},
    on: (ev, f) => (h[ev] ||= []).push(f),
    off: (ev, f) => { h[ev] = (h[ev] || []).filter(g => g !== f); },
    send: content => kernel.handle(content, model),
    _emit: (ev, ...a) => (h[ev] || []).slice().forEach(f => f(...a)),
    // Like a Backbone set(): all attributes first, then one change event each.
    fire(upd) {
      const keys = Object.keys(upd);
      for (const k of keys) state[k] = upd[k];
      for (const k of keys) model._emit('change:' + k);
    },
  };
  return model;
}
function frames(n = 3) {
  for (let i = 0; i < n; i++) {
    const q = rafQ; rafQ = [];
    inFrame = true;
    try { q.forEach(f => f(0)); } finally { inFrame = false; }
  }
}

const { render } = await import(pathToFileURL(path.join(dir, 'widget.mjs')).href);

function baseState(caseName) {
  const st = clone(C[caseName].state);
  st.embedded_data = st.embedded_data || {};
  return st;
}
function mount(caseName, { state, kernel, fresh = true } = {}) {
  if (fresh) { listeners = {}; uploads = 0; errors = []; warns = []; texts = []; rafQ = []; timers = []; }
  kernel = kernel || mkKernel(caseName);
  const model = mkModel(state || baseState(caseName), kernel);
  let threw = null;
  try { render({ model, el: mkRoot() }); } catch (e) { threw = e; }
  return { model, kernel, threw };
}
const fetches = k => k.requests.filter(r => r.name === '_cmd_fetch_track');
const drawn = s => texts.includes(s);
const drawnStarting = s => texts.some(t => t.startsWith(s));
function redraw() { texts = []; frames(); }

// Uploads made by the empty viewer itself (static quad buffers).
let m = mount('empty'); frames();
const SETUP = uploads;

// ── 1. Initial display: fetch the chromosome in view, show "loading…" ─────
m = mount('seven');
frames();
ok(!m.threw && errors.length === 0, `mount: no exceptions ${errors[0] || ''}`);
const nTracks = C.seven.state.track_configs.length;
let f = fetches(m.kernel);
ok(f.length === nTracks && f.every(r => r.chrom === 'chr1'),
   `mount requests one slice per track for the chrom in view (${f.length} of ${nTracks})`);
ok(uploads === SETUP, 'nothing uploaded before the kernel answers');
ok(drawn('loading…'), 'bands say "loading…" while waiting');
m.kernel.flush(); redraw();
const chr1Uploads = uploads - SETUP;
ok(chr1Uploads > 0 && errors.length === 0, `replies uploaded (${chr1Uploads} uploads) ${errors[0] || ''}`);
ok(!drawn('loading…'), 'no "loading…" once loaded');
ok(drawn('P') && drawn('H'), 'track labels drawn');

// ── 2. Slow kernel: notice changes to "waiting for kernel…" ───────────────
m = mount('seven'); frames();
now += 5000; runTimers(); redraw();
ok(drawn('waiting for kernel…'), 'after 4 s without a reply the band says "waiting for kernel…"');
m.kernel.flush(); redraw();
ok(!drawn('waiting for kernel…'), 'notice gone once the reply arrives');

// ── 3. Chromosome switch fetches only that chromosome, once ──────────────
m = mount('seven'); frames(); m.kernel.flush(); frames();
m.kernel.requests = [];
m.model.fire({ viewport: { chrom: 'chr2', start: 0, end: C.seven.state.chrom_sizes.chr2 } });
frames();
f = fetches(m.kernel);
ok(f.length === nTracks && f.every(r => r.chrom === 'chr2'), `switch to chr2 requests chr2 only (${f.length})`);
m.kernel.flush(); frames();
m.kernel.requests = [];
m.model.fire({ viewport: { chrom: 'chr1', start: 0, end: C.seven.state.chrom_sizes.chr1 } });
frames();
ok(fetches(m.kernel).length === 0, 'switching back to chr1 uses what is already loaded');

// ── 4. Adding tracks to a displayed viewer fetches only the new ones ─────
m = mount('two'); frames(); m.kernel.flush(); frames();
const k7 = mkKernel('seven');
m.kernel.handle = k7.handle.bind(k7); m.kernel.flush = k7.flush.bind(k7);
k7.requests = [];
const s7 = C.seven.state;
m.model.fire({ track_configs: clone(s7.track_configs), track_revs: clone(s7.track_revs) });
frames();
ok(fetches(k7).length === nTracks - C.two.state.track_configs.length,
   `adding ${nTracks - C.two.state.track_configs.length} tracks requests only those (${fetches(k7).length})`);
uploads = 0; k7.flush(); frames();
ok(errors.length === 0, 'added tracks upload without errors');

// ── 5. A revision bump refetches only that track; old data stays drawn ───
m = mount('seven'); frames(); m.kernel.flush(); frames();
m.kernel.requests = []; uploads = 0;
m.kernel.revs.t1 = 100;
m.model.fire({ track_revs: { ...m.model.state.track_revs, t1: 100 } });
redraw();
f = fetches(m.kernel);
ok(f.length === 1 && f[0].tid === 't1', `revision bump of t1 requests t1 only (${f.map(r => r.tid)})`);
ok(!drawn('loading…'), 'stale data keeps drawing while the new slice loads (no flicker)');
m.kernel.flush(); frames();
ok(uploads > 0 && errors.length === 0, 'new revision uploaded');

// ── 6. A reply that predates the current revision is dropped and refetched
m = mount('seven'); frames(); m.kernel.flush(); frames();
m.kernel.requests = [];
m.kernel.revs.t1 = 200;
m.model.fire({ track_revs: { ...m.model.state.track_revs, t1: 200 } });
frames();
m.kernel.revs.t1 = 199;        // kernel answers with an older revision
uploads = 0; m.kernel.flush(); frames();
ok(uploads === 0, 'stale reply not uploaded');
ok(fetches(m.kernel).length === 2, 'stale reply triggers a new fetch');

// ── 7. Kernel-side error: shown on the band, not retried ─────────────────
{
  const k = mkKernel('seven');
  k.broken.t1 = 'KeyError: boom';
  m = mount('seven', { kernel: k });
}
frames(); m.kernel.flush(); redraw();
ok(drawnStarting('could not load track: KeyError: boom'), 'kernel error drawn on the band');
ok(errors.some(e => e.includes('KeyError: boom')), 'kernel error logged to the console');
m.kernel.requests = [];
m.model.fire({ track_revs: { ...m.model.state.track_revs } }); frames();
ok(fetches(m.kernel).length === 0, 'failed slice not re-requested at the same revision');

// ── 8. Corrupt payload: that slice fails, the others load ────────────────
{
  const st = baseState('seven');
  const k = mkKernel('seven');
  const h = k.handle.bind(k);
  k.handle = (content, model) => {
    h(content, model);
    if (content.msg && content.msg.tid === 't2') {   // segment track: truncate its int32 buffer
      const deliver = k.queue.pop();
      k.queue.push(() => {
        const orig = model._emit;
        model._emit = (ev, msg, bufs) => orig(ev, msg, bufs.map(b => new DataView(b.buffer, b.byteOffset, 3)));
        deliver();
        model._emit = orig;
      });
    }
  };
  m = mount('seven', { state: st, kernel: k }); frames(); k.flush(); redraw();
  ok(!m.threw, 'corrupt slice: no exception escapes');
  ok(drawnStarting('could not load track:'), 'corrupt slice: error drawn on its band');
  ok(drawn('P') && drawn('H'), 'other tracks still drawn');
}

// ── 9. GPU out of memory during an upload ─────────────────────────────────
m = mount('seven'); frames();
nextGlErrors = [0, GL.OUT_OF_MEMORY];   // first call clears stale errors, second reports OOM
m.kernel.flush(); redraw();
nextGlErrors = [];
ok(errors.some(e => e.includes('GPU out of memory')), 'OOM reported as a load failure');
ok(drawnStarting('could not load track: GPU out of memory'), 'OOM drawn on the band');

// ── 10. Context loss: notice instead of blank; restore refetches ──────────
m = mount('seven'); frames(); m.kernel.flush(); frames();
ctxLost = true;
for (const fn of listeners['sv-glcanvas:webglcontextlost'] || []) fn({ preventDefault() {} });
redraw();
ok(drawnStarting('WebGL context lost'), 'context loss draws a notice');
ok(drawn('P'), 'context loss still draws labels');
m.kernel.requests = [];
m.model.fire({ track_revs: { ...m.model.state.track_revs, t0: 300 } }); frames();
ok(fetches(m.kernel).length === 0, 'no fetches while the context is lost');
ctxLost = false; uploads = 0; m.kernel.requests = [];
for (const fn of listeners['sv-glcanvas:webglcontextrestored'] || []) fn({});
frames();
ok(fetches(m.kernel).length === nTracks, `restore refetches the chrom in view (${fetches(m.kernel).length})`);
m.kernel.revs.t0 = 300; m.kernel.flush(); redraw();
// The restored context also rebuilds the viewer's own static buffers.
ok(uploads === chr1Uploads + SETUP && !drawnStarting('WebGL context lost'),
   `restore re-uploads everything (${uploads} vs ${chr1Uploads + SETUP}) and clears the notice`);

// ── 11. Embedded data: drawn without a kernel ─────────────────────────────
{
  const st = baseState('embedded');
  putViews(st.embedded_data, C.embedded.embedded_paths, C.embedded.embedded_buffers);
  m = mount('embedded', { state: st }); frames();
  ok(fetches(m.kernel).length === 0, 'embedded viewer sends no fetches');
  ok(uploads > SETUP && errors.length === 0, `embedded data uploaded at mount ${errors[0] || ''}`);
  redraw();
  ok(!drawn('loading…'), 'embedded viewer never says "loading…"');
}

// ── 12. Removing and clearing tracks; late replies for removed tracks ─────
m = mount('seven'); frames();
const st7 = C.seven.state;
m.model.fire({ track_configs: st7.track_configs.slice(1),
               track_revs: Object.fromEntries(Object.entries(st7.track_revs).filter(([k]) => k !== 't0')) });
m.kernel.flush(); frames();
m.model.fire({ track_configs: [], track_revs: {} });
frames();
ok(errors.length === 0, `remove + clear with replies in flight: no errors ${errors[0] || ''}`);

// ── 13. Hidden tracks are fetched in the background, after visible ones ──
{
  const st = baseState('seven');
  st.track_configs = st.track_configs.map(c => (c.id === 't1' || c.id === 't3') ? { ...c, hidden: true } : c);
  m = mount('seven', { state: st }); frames();
  const order = fetches(m.kernel).map(r => r.tid);
  ok(order.length === nTracks && order.slice(-2).sort().join() === 't1,t3',
     `hidden tracks fetched too, last (${order})`);
  m.kernel.flush(); frames();
  m.kernel.requests = []; uploads = 0;
  m.model.fire({ track_configs: st.track_configs.map(c => ({ ...c, hidden: undefined })) });
  redraw();
  ok(fetches(m.kernel).length === 0 && uploads === 0 && !drawn('loading…'),
     'showing hidden tracks needs no fetch, upload or "loading…"');
  m.kernel.requests = [];
  m.model.fire({ track_configs: [...st.track_configs].reverse() });   // select_tracks-style reorder
  redraw();
  ok(fetches(m.kernel).length === 0 && !drawn('loading…'), 'reordering tracks needs no fetch');
}

// ── 14. Two views of one model: each uses its own replies ─────────────────
{
  listeners = {}; uploads = 0; errors = []; rafQ = [];
  const k = mkKernel('seven');
  const shared = baseState('seven');
  const model = mkModel(shared, k);
  render({ model, el: mkRoot() });
  render({ model, el: mkRoot() });
  frames();
  ok(fetches(k).length === 2 * nTracks, `two views each request their slices (${fetches(k).length})`);
  uploads = 0; k.flush(); frames();
  ok(uploads === 2 * chr1Uploads && errors.length === 0,
     `each view uploads its own replies once (${uploads} vs ${2 * chr1Uploads})`);
}

// ── 15. Heatmaps whose bin count is not a multiple of 4 ───────────────────
// A bp-sized zoom_windows resolves to arbitrary counts (here 100 and 50 bins
// on the two chromosomes, and 23 after a rebin), which WebGL rejected under
// the default 4-byte row alignment, leaving the band blank.
if (C.odd) {
  const sizes = C.odd.state.chrom_sizes;
  m = mount('odd'); frames(); m.kernel.flush(); redraw();
  ok(errors.length === 0 && texRejects === 0 && !drawnStarting('could not load track'),
     `odd-width heatmap uploads on chr1 (${texRejects} rejected) ${errors[0] || ''}`);
  m.model.fire({ viewport: { chrom: 'chr2', start: 0, end: sizes.chr2 } });
  frames(); m.kernel.flush(); redraw();
  ok(errors.length === 0 && texRejects === 0, `odd-width heatmap uploads on chr2 ${errors[0] || ''}`);
  // The rebinned detail view arrives as a new revision of the track.
  const k = mkKernel('odd_rebinned');
  m.kernel.handle = k.handle.bind(k); m.kernel.flush = k.flush.bind(k);
  m.model.fire({ viewport: { chrom: 'chr1', start: 100000, end: 330000 } }); frames();
  m.model.fire({ track_revs: clone(C.odd_rebinned.state.track_revs) }); frames();
  k.flush(); redraw();
  ok(fetches(k).length > 0 && errors.length === 0 && texRejects === 0
     && !drawnStarting('could not load track'),
     `rebinned 23-bin detail view uploads ${errors[0] || ''}`);
}

console.log = origLog;
console.log(fails ? `\n${fails} FAILED` : '\nALL PASSED');
process.exit(fails ? 1 : 0);
