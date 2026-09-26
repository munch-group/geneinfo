# Changelog

## [Unreleased]

### Widget (`geneinfo.widget.segment_viewer_gl`)

End-to-end pass across API, trait validation, GL resource management, UI
feedback, accessibility, and performance of the `Tracks` WebGL2 viewer.

**API renames + removal methods**
- `add_heatmap_track`: `group_col` renamed to `group_by` (old kwarg emits
  `DeprecationWarning`); `individual_col` no longer defaults to `'sample'`
  and is now required.
- `add_histogram_track`: `stack` default flipped to `False` to match
  `add_segment_track`.
- `add_ucsc_track`: first positional renamed `track_name` → `ucsc_track`;
  old `track_name=...` kwarg still works with a `DeprecationWarning`.
- `density_windows` renamed to `zoom_windows` on every track type, and
  `add_heatmap_track`'s `windows` renamed to match (both old names emit a
  `DeprecationWarning`; passing an old name together with `zoom_windows`
  is a `TypeError`). "Density" was literal only for `add_segment_track` —
  elsewhere the levels hold values aggregated by `aggregate`, not
  densities — and the heatmap spelled the same knob two ways. The new
  name says what the knob controls: which resolution is drawn at which
  zoom. The `cfg` key sent to the browser follows (`densityWindows` →
  `zoomWindows`).
- `y_range_per_lod` renamed to `y_range_per_zoom` on the point / line /
  fill tracks (same deprecation shim), dropping the internal LOD jargon
  from the public API.
- `zoom_windows=False` is accepted as a readable spelling of `zoom_windows=()`
  on every track that keeps a level stack (segment / point / line / histogram /
  fill): both disable LOD and ship only the raw data. `True` is still
  rejected — "on" would have to say at which resolutions — as are bools used
  as individual entries (`(256, False)`). `add_heatmap_track` still rejects
  an empty spec, since it always bins.
- New `remove_track(name_or_index)` and `clear_tracks()` on `Tracks` that
  clean up `track_configs`, `track_data`, and the per-track heatmap
  caches.
- Internal `_tid` switched to a monotonic counter so removed ids are
  never reused.

**Trait validators**
- Added `@validate('viewport')`: enforces chrom ∈ `chrom_sizes` and
  `0 ≤ start ≤ end ≤ length`.
- `set_viewport` now raises `KeyError` on unknown chromosomes (kept
  clamping for valid-chrom out-of-range start/end); `zoom_to` validates
  the chromosome at the top regardless of `center`.
- `_validate_theme` now **merges** partial overrides onto the current
  theme, whitelists keys against `DARK_THEME.keys()`, and ships a single
  source of truth (`traitlets.Dict(dict(DARK_THEME))`).
- `_resolve_color_mapping` grew a recursion guard for cyclic input.
- `add_vlines` / `add_spans` reject `str` explicitly and validate
  numeric / pair shape.

**Resource management (GL + listeners)**
- Per-shape GPU disposers and a `disposeAllGpu()` walk clean every
  buffer/texture the widget owns.
- `uploadTrackData` now disposes each `(tid, chrom)` slot before
  re-assigning it (fixes VRAM leak on re-upload).
- `webglcontextlost` / `webglcontextrestored` listeners rebuild programs
  and re-upload `track_data`; `scheduleRender`/`render` short-circuit
  while the context is lost.
- `keydown` moved off `window` onto a focusable `.sv-wrap` (`tabindex=0`);
  skips text-field targets and `preventDefault`s only the keys it
  handles.
- Tooltip `mousemove` scoped to `glCanvas`; drag-pan listener attached
  to `window` only for the duration of a drag.
- Track removal (from Python) now disposes the track's GL resources via
  a JS-side tid-diff on `change:track_configs`.

**UI feedback**
- Invalid `posInput` entries flash red (`.sv-input-error`) with a
  specific tooltip hint; state clears on next keystroke or blur.
- `hmRecBtn` disables + shows a spinning ⟳ while any heatmap rebin is in
  flight; re-enables on the next `change:track_data`.
- Empty track bands now render a centred "no data in view" label.
- Snapshot button: ✓ for clipboard success, ⇓ for download fallback, !
  for failure, each with a matching tooltip.
- "WebGL2 not available" replaced with a styled `.sv-error` block
  including a short "what to check" list.

**Accessibility + help**
- `aria-label` on every icon button, `role="toolbar"`, and aria on the
  chrom `select` / position `input`.
- `glCanvas`: `tabindex=0`, `role="img"`, descriptive `aria-label`;
  overlay canvas and tooltip `aria-hidden`.
- Visible focus ring on buttons, inputs, selects, and canvas via
  `:focus-visible`.
- New `?` toolbar button opens a popover listing mouse / keyboard /
  button legend and a link to `munch-group.org/geneinfo`.
- Small vertical dividers (`.sv-sep-v`) between toolbar clusters.

**`add_fill_track` argument resolution**
- A lone boundary now fills against `baseline` (`0` by default) instead of
  reaching for a column that isn't there: `y_hi='v'` no longer raises
  `KeyError` looking for a `'lo'` column, and `y_lo='v'` behaves
  symmetrically. `y` is an alias for `y_hi`, accepted whenever `y_lo` is
  absent, so `y=`, `y_hi=`, and `y_lo=` alone all produce the same
  zero-anchored fill with the pos/neg colour split intact.
- `group_by` now requires a two-curve band and raises `ValueError` next to a
  single curve. Groups are told apart by colour, but a single curve has
  already spent its colours on the baseline split — `color_pos` / `color_neg`
  apply to every group alike — so the groups rendered identically. This
  rejects calls that previously "worked" but drew indistinguishable fills.
- A genuine `y_lo` + `y_hi` pair is still a two-curve band, and omitting
  all three still falls back to the `'lo'` / `'hi'` columns — both
  unchanged. `y` together with `y_hi` is now a `TypeError` (same curve,
  two spellings); `y` with `y_lo` remains a `ValueError`.

**Zoom limit**
- New `min_span` trait (default `500`) sets the narrowest viewport in base
  pairs — the point past which zooming in stops. It replaces a `500` hard-coded
  separately into the wheel, double-click and `+` key handlers, so the three
  can no longer drift apart. Set `Tracks(..., min_span=100)`, or assign it live
  like `zoom_speed` / `pan_speed`. The floor is clamped to the chromosome's own
  span, so a `min_span` larger than a short chromosome cannot lock the view.

**Exact genomic coordinates (breaking payload change)**
- Genomic positions now travel as `int32` and are rebased against the viewport
  start *in integer space* inside the vertex shaders
  (`float(aXi - uVSi) / (uVE - uVS)`). They were `float32` end to end, whose
  24-bit mantissa is exact only below 16,777,216 — above that positions snapped
  onto a grid of 4 bp, then 8, then 16, which looked like the data was still
  being aggregated even with `zoom_windows=False`. Twelve samples 1 bp apart at
  60 Mb used to collapse onto 4 distinct x values; they now resolve to 12, and
  to 12 at 240 Mb as well.
- Applies to every WebGL path: line, scatter, fill and the segment density area
  (`VS_DENS`), histogram bars and segment rects (`VS_RECT`), and arcs
  (`VS_ARC`). Heatmaps are unaffected — their quad carries texture fractions
  and is bin-limited — as are the gene track, vlines and spans, which draw on
  the 2D canvas in double precision.
- Vertex strides are unchanged; only the leading 4 bytes of each vertex change
  type, so `vertexAttribIPointer(..., gl.INT, ...)` reads what
  `vertexAttribPointer` used to. New `Tracks._pack_i32` / `_pack_xy` /
  `_pack_xlohi` replace `_pack_f32` on the position-carrying payloads, and
  `_step_expand` / `_step_expand3` now return components rather than a narrowed
  interleaved buffer.
- **Breaking, deliberately:** a notebook saved with the previous payloads will
  render garbled positions until its cells are re-run. There is no legacy
  decode path.
- Derived positions (LOD bin centres, `step='mid'` midpoints) are rounded to
  whole base pairs, half up — `np.rint`'s round-half-to-even made midpoints
  cluster in pairs. Half a base pair is orders of magnitude below one pixel.

**Gene track**
- `add_gene_track`'s `label_padding` is now measured in **kilobases**, not
  base pairs: pass `200` for 200 kb. Fractions are allowed (`0.5` = 500 bp).
  This is a silent unit change for existing callers, so a value of 10,000 kb
  (10 Mb) or more — what a base-pair value carried over from the old form
  looks like — emits a `UserWarning` telling you to divide by 1000.

- `label_padding` is now reserved evenly on both sides of a gene instead of
  entirely on one side chosen by strand. The old scheme left one of the four
  strand pairings unprotected: a `+` gene followed by a `-` gene padded *away*
  from the gap between them, so they stayed on one row and their labels could
  collide, while a `-` followed by a `+` was padded twice over. Since the
  renderer centres each label on the gene's true midpoint, symmetric padding
  is what actually matches the drawing. Two neighbours are now bumped apart
  exactly when their gap is under `label_padding`, whatever their strands —
  which is what same-strand pairs already did, so calibrated values keep their
  meaning.

**Overlay DataFrame input**
- `add_vlines` and `add_spans` accept a DataFrame alongside the existing
  scalar / iterable / dict forms, with column names given by `pos=` (vlines)
  and `start=` / `end=` (spans). Rows with a missing value are skipped, and a
  non-numeric column is rejected with a dtype error rather than failing later
  in `int()`.
- `chrom=` gains a third reading in the DataFrame form: a column name, so one
  frame can carry several chromosomes. Omitted, it uses a `'chrom'` column
  when the frame has one and otherwise falls back to the viewport chromosome,
  so `add_vlines(df)` works for both a genome-wide frame and a bare list of
  positions. A plain chromosome name still works (`add_vlines(df, 'chr7')`);
  a column of that name wins over a chromosome of the same name, and a value
  matching neither raises instead of silently falling back.

- Both gained `group_by=` (plus `color_map=` / `palette=`), colouring each
  line or span by a column's value using the same palette machinery as the
  track methods. It is mutually exclusive with `color=` — a flat colour would
  overwrite every group's — and requires DataFrame input, both enforced with
  `TypeError`. Group order is by string form, so colours are stable across
  chromosomes and independent of row order. `add_spans`'s `color` default
  moved from `'#ffcc44'` to `None` (resolved to the same value internally) so
  an explicitly passed colour is distinguishable from the built-in one; the
  rendered default is unchanged.

**Level of detail**
- `add_fill_track` gained `zoom_windows` (plus `aggregate` and
  `y_range_per_zoom`), bringing it in line with the segment / heatmap /
  point / line / histogram tracks — it was the only value-over-position
  track that always shipped and drew every raw sample. The payload is now
  `{base, lods, binWidth}` per (chrom, group) and the renderer picks the
  coarsest level whose bins still cover ≥ 2 CSS px, falling back to the
  raw samples when zoomed in. Bare-base64 fill payloads still decode.
- `aggregate` defaults to `'envelope'` (per-bin `min` of the lower and
  `max` of the upper boundary) so a coarse level is always a superset of
  the band it summarises and spikes survive zooming out; `'mean'`
  averages each boundary instead. In single-`y` mode the aggregate is
  applied to `y` before the split against `baseline`, keeping the
  pos/neg colours from smearing.
- Tooltips (`rawFill`) keep reading the raw samples at every zoom.
- `add_ucsc_track(kind='fill', zoom_windows=...)` no longer raises
  `TypeError` — the kwarg is forwarded like it is for the other kinds.

**Performance**
- Module-level `_step_expand` and `_aggregate_bin` helpers shared by the
  xy and histogram paths (eliminates the nested closure and a duplicated
  aggregation loop).
- `_step_expand3` / `_aggregate_bin_band` band variants of the same,
  replacing the staircase loop that was inlined in `add_fill_track`;
  `_aggregate_bin` also learned `'min'`.
- `add_segment_track` stacked density: per-level column-sum and
  running `cum_prev` computed once instead of being rebuilt for every
  group (prev: O(G²×L), now: O(G×L)).
- `add_histogram_track` stacked base-bar build: vectorised with a wide
  pivot + `cumsum` instead of a per-row Python loop; equivalence
  verified against the old algorithm including duplicate-x-within-group.
- New `_prep_groups` / `_commit_track` helpers on `Tracks` used by
  segment / heatmap / histogram / fill / xy / gene.
- Gene track split into `_gene_records_from_dict`,
  `_gene_records_from_df`, `_apply_label_padding` so `add_gene_track`
  reads top-down.
- JS hot paths (wheel, drag-pan, tooltip, render) now read `pan_speed`,
  `zoom_speed`, and `track_configs` from cached locals refreshed via
  `model.on('change:...')` listeners.
