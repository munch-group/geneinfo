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

**Performance**
- Module-level `_step_expand` and `_aggregate_bin` helpers shared by the
  xy and histogram paths (eliminates the nested closure and a duplicated
  aggregation loop).
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
