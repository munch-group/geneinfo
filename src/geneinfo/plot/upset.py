"""
upset.py  v0.6.0
==================
UpSet-style intersection plot where the count bars are replaced by stacked
gene (or any element) name labels that grow upward from a shared baseline,
visually resembling a bar chart.

**Column layout** is decided independently per column based on the number of
genes it contains, controlled by *two_col_threshold* (default 15):

* **≤ threshold** → single centered column at 1 × *col_width*.
* **> threshold** → paired two-sub-column layout at 2 × *col_width*.
  Shorter names are left-aligned on the left sub-column; longer names are
  right-aligned on the right sub-column, so each row spans the full column
  width in proportion to combined name length.

The dot in the matrix panel is always centred in its column regardless of
width.  The figure background is transparent.

**Wrapping**: when *wrap=True* the *max_cols* intersections-per-row limit is
respected and additional rows are added below, producing a single figure with
multiple stacked gene+matrix panel pairs — one per row of columns.

**Intersection semantics** are controlled by *mode*:

* ``'exclusive'`` (default) — a gene appears in exactly one column.
* ``'inclusive'`` — a gene appears in every column whose listed sets are a
  subset of the sets that gene belongs to.

Quickstart
----------
    from geneinfo.plot import upset
    import matplotlib.pyplot as plt

    sets = {
        "Cancer drivers": {"TP53", "BRCA1", "KRAS", "PTEN", "EGFR", "APC"},
        "PI3K pathway":   {"PTEN", "KRAS", "AKT1", "MTOR", "EGFR", "PIK3CA"},
        "DNA repair":     {"BRCA1", "TP53", "ATM", "CHEK2", "PTEN", "MLH1"},
    }
    fig = upset(sets, title="Gene set overlaps")
    fig.savefig("overlaps.pdf", bbox_inches="tight", transparent=True)
    plt.show()
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Optional, List
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from ..genelist import GeneList, GeneListCollection

__all__ = ["upset", "compute_intersections"]
__version__ = "0.6.0"


# ── default colour palette ─────────────────────────────────────────────────────

_PALETTE: list[str] = [
    "#7F77DD",
    "#1D9E75",
    "#D85A30",
    "#378ADD",
    "#639922",
    "#BA7517",
    "#D4537E",
    "#E24B4A",
]


# ── intersection computation ───────────────────────────────────────────────────


def compute_intersections(
    sets: Mapping[str, set],
    *,
    min_size: int = 1,
    sort_by: str = "size",
    max_cols: int = 30,
    singletons: bool = True,
    mode: str = "exclusive",
) -> list[tuple[frozenset[str], list[str]]]:
    """
    Compute pairwise/n-way intersections.

    Parameters
    ----------
    sets :
        Mapping of label -> element collection.
    min_size :
        Discard intersections smaller than this.
    sort_by :
        ``'size'`` - largest intersection first (default).
        ``'degree'`` - most sets involved first, then by size.
    max_cols :
        Truncate output to at most this many intersections after sorting.
        When called from ``upset`` with ``wrap=True`` this cap is
        applied per row rather than to the total.
    singletons :
        If ``False``, remove degree-1 intersections before the ``max_cols``
        cap so they cannot crowd out genuine multi-set overlaps.
    mode :
        ``'exclusive'`` (default) — each column shows elements that belong
        to *exactly* the listed sets and no others.

        ``'inclusive'`` — each column shows elements that belong to *at
        least* the listed sets; they may also appear in other sets.  A gene
        shared by A, B, and C will appear in the A∩B, A∩C, B∩C, and A∩B∩C
        columns simultaneously.

    Returns
    -------
    list of ``(frozenset[str], list[str])``
        Each tuple is *(set_labels_in_intersection, sorted_element_list)*.
    """
    names = list(sets.keys())
    n = len(names)
    results: list[tuple[frozenset[str], list[str]]] = []

    if mode not in ("exclusive", "inclusive"):
        raise ValueError(f"mode must be 'exclusive' or 'inclusive', got {mode!r}")

    for mask in range(1, 1 << n):
        inc = [names[i] for i in range(n) if mask >> i & 1]
        exc = [names[i] for i in range(n) if not (mask >> i & 1)]

        if not singletons and len(inc) == 1:
            continue

        els: set = set(sets[inc[0]])
        for s in inc[1:]:
            els &= sets[s]

        if mode == "exclusive":
            for s in exc:
                els -= sets[s]

        if len(els) >= min_size:
            results.append((frozenset(inc), sorted(els)))

    if sort_by == "size":
        results.sort(key=lambda x: -len(x[1]))
    elif sort_by == "degree":
        results.sort(key=lambda x: (-len(x[0]), -len(x[1])))
    elif sort_by == "groups":
        results.sort(key=lambda x: sum(names.index(y) for y in x[0]))        
    elif sort_by == "ordered":
        results.sort(key=lambda x: min(names.index(y) for y in x[0]))        
    else:
        raise ValueError(f"sort_by must be 'size' or 'degree', got {sort_by!r}")

    return results[:max_cols]


# ── main API ───────────────────────────────────────────────────────────────────


def upset(
    sets: Mapping[str, "set | list"]|List[GeneList],
    *,
    # ── data ──────────────────────────────────────────────────────────────────
    min_size: int = 1,
    sort_by: str = "size",
    max_cols: int = 15,
    max_labels: Optional[int] = None,
    singletons: bool = False,
    mode: str = "exclusive",
    # ── wrapping ──────────────────────────────────────────────────────────────
    wrap: bool = False,
    # ── color ─────────────────────────────────────────────────────────────────
    color: bool = True,
    cmap: str = "tab10",
    # ── layout ────────────────────────────────────────────────────────────────
    col_width: float = 1.0,
    gene_row_h: float = 0.14,
    set_row_h: float = 0.28,
    size_bar_cols: float = 4.0,
    gene_pad: float = 0.04,
    two_col_threshold: int = 15,
    two_col_ratio: float = 1.5,
    figsize: Optional[tuple[float, float]] = None,
    # ── gene label typography ─────────────────────────────────────────────────
    gene_fontsize: float = 6.5,
    gene_fontfamily: str = "monospace",
    gene_color: Optional[str] = None,
    gene_fontweight: str = "normal",
    # ── count label typography ────────────────────────────────────────────────
    count_fontsize: Optional[float] = None,
    count_color: Optional[str] = None,
    # ── set label typography ──────────────────────────────────────────────────
    label_fontsize: Optional[float] = None,
    label_color: Optional[str] = None,
    label_fontweight: str = "bold",
    label_fontfamily: str = "sans-serif",
    # ── title ─────────────────────────────────────────────────────────────────
    title: Optional[str] = None,
    title_fontsize: Optional[float] = None,
    title_fontweight: str = "bold",
    # ── dots ──────────────────────────────────────────────────────────────────
    dot_size: float = 5.5,
    dot_filled_color: Optional[str] = None,
    dot_empty_color: Optional[str] = None,
    # ── grid & structure ──────────────────────────────────────────────────────
    col_divider_color: Optional[str] = None,
    col_divider_lw: float = 0.4,
    baseline_color: Optional[str] = None,
    baseline_lw: float = 0.9,
    row_grid_color: Optional[str] = None,
    row_grid_lw: float = 0.4,
    size_bar_color: Optional[str] = None,
    size_bar_alpha: float = 0.40,
    # ── annotations ───────────────────────────────────────────────────────────
    show_set_sizes: bool = True,
    show_size_bars: bool = True,
    bars: bool = False,
    show_count: bool = True,
    truncate_suffix: str = "...",
    # ── highlighting ──────────────────────────────────────────────────────────
    highlight: Optional[list[frozenset[str]]] = None,
    highlight_color: str = "#F0EEE8",
) -> plt.Figure:
    """
    Draw an UpSet-style intersection plot where stacked gene names replace the
    conventional count bars, growing upward from a shared baseline.

    **Column layout** is decided independently per column:

    * Columns with **≤ two_col_threshold** genes use a **single centered
      stack** at 1 × *col_width*.
    * Columns with **> two_col_threshold** genes use a **paired two-sub-column
      layout** at 2 × *col_width*: shorter names left-aligned on the left
      half, longer names right-aligned on the right half.

    The dot in the matrix panel is always centred in its column.  The figure
    background is fully transparent.

    **Wrapping**: when *wrap=True*, all intersections are computed without the
    *max_cols* cap, then laid out *max_cols* columns at a time.  Each group of
    columns occupies one gene-panel + matrix-panel pair, and the pairs are
    stacked vertically within a single figure.  Set-name labels and size bars
    are repeated on every row so each row is self-contained.

    **Intersection semantics** are controlled by *mode*:

    * ``'exclusive'`` (default) — a gene appears in exactly one column.
    * ``'inclusive'`` — a gene appears in every column whose listed sets are
      a subset of the sets that gene belongs to.

    Parameters
    ----------
    sets :
        Gene (or any element) sets to compare.  Values can be ``set`` or
        ``list``; duplicates are ignored.
    min_size :
        Discard intersections with fewer than this many elements.
    sort_by :
        Column ordering: ``'size'`` — largest intersection first (default);
        ``'degree'`` — most sets involved first, then by size.
    max_cols :
        Without wrapping: hard cap on the number of columns displayed.
        With ``wrap=True``: number of intersection columns per row.
        Singletons are filtered *before* this cap when ``singletons=False``.
    max_labels :
        Cap on gene labels shown per column.  ``None`` = unlimited.
    singletons :
        If ``False`` (default), degree-1 intersections (genes unique to a
        single set) are excluded before the *max_cols* cap is applied.
    mode :
        ``'exclusive'`` (default) or ``'inclusive'``.  See above.
    wrap :
        If ``True``, ignore the *max_cols* truncation and instead lay out
        all intersections across multiple rows, *max_cols* columns each.
        Each row is a complete gene-panel + matrix-panel pair with set
        labels repeated on the left.  Default ``False``.
    color :
        If ``True`` (default), each set gets a distinct colour from *cmap*
        applied to its label, size bar, and filled dots.  Gene name text
        always uses the theme foreground.  ``False`` → monochrome.
    cmap :
        Matplotlib colormap name for ``color=True``.  Defaults to
        ``'tab10'``; auto-upgraded to ``'tab20'`` for > 10 sets.
    col_width :
        Width of a narrow (single-column) intersection column in inches.
        Wide (two-column) columns are 2 × *col_width*.
    gene_row_h :
        Vertical space per gene-label row in inches.
    set_row_h :
        Height of each set row in the dot-matrix panel in inches.
    size_bar_cols :
        Width of the set-label + size-bar region in *col_width* units.
        Overridden when ``bars=False``.
    gene_pad :
        Gap between a column divider and the nearest gene-name character,
        as a fraction of a column width.  Wide columns only.  Default 0.04.
    two_col_threshold :
        Columns with **more than** this many genes use the wide two-column
        layout (2 × *col_width*); others use the narrow single layout.
        Default 15.
    figsize :
        Override auto figure size ``(width_in, height_in)``.
    gene_fontsize :
        Font size for gene name labels (pt).  Default 6.5.
    gene_fontfamily :
        Font family for gene name labels.  Default ``'monospace'``.
    gene_color :
        Colour for gene name labels.  ``None`` → theme foreground.
    gene_fontweight :
        Font weight for gene name labels.  Default ``'normal'``.
    count_fontsize :
        Font size for intersection-count labels.  Defaults to
        ``gene_fontsize - 0.5``.
    count_color :
        Colour for count labels and truncation markers.
        ``None`` → theme foreground at ~45 % opacity.
    label_fontsize :
        Font size for set name labels.  Defaults to ``gene_fontsize + 0.5``.
    label_color :
        Colour for set name labels.  ``None`` → theme foreground (or set
        colour when ``color=True``).
    label_fontweight :
        Font weight for set name labels.  Default ``'bold'``.
    label_fontfamily :
        Font family for set name labels.  Default ``'sans-serif'``.
    title :
        Optional figure title string (placed above the first row).
    title_fontsize :
        Font size for the figure title.  Defaults to ``gene_fontsize + 4``.
    title_fontweight :
        Font weight for the figure title.  Default ``'bold'``.
    dot_size :
        Dot-matrix circle diameter (matplotlib marker-size units).
    dot_filled_color :
        Colour of filled dots.  ``None`` → theme foreground (or set colour).
    dot_empty_color :
        Colour of empty dots.  ``None`` → theme foreground at ~18 % opacity.
    col_divider_color :
        Colour of vertical column-divider lines.
        ``None`` → theme foreground at ~15 % opacity.
    col_divider_lw :
        Line width of column dividers.  Default 0.4.
    baseline_color :
        Colour of the horizontal baseline between gene and matrix panels.
        ``None`` → theme foreground at ~50 % opacity.
    baseline_lw :
        Line width of the baseline.  Default 0.9.
    row_grid_color :
        Colour of horizontal lines between set rows in the matrix panel.
        ``None`` → theme foreground at ~12 % opacity.
    row_grid_lw :
        Line width of row-grid lines.  Default 0.4.
    size_bar_color :
        Colour of set-size bars.  ``None`` → theme foreground (or set colour).
    size_bar_alpha :
        Opacity of set-size bars.  Default 0.40.
    show_set_sizes :
        Append total element count to each set label.  Default ``True``.
    show_size_bars :
        Draw horizontal set-size bars.  Ignored when ``bars=False``.
    bars :
        If ``False`` (default), hide size bars and use a narrow label-only
        margin.  ``True`` shows the classic UpSet-style size bars.
    show_count :
        Print intersection size above each gene-label column.  Default ``True``.
    truncate_suffix :
        Appended when a column is capped by *max_labels*.  Default ``'...'``.
    highlight :
        List of ``frozenset`` of set-name strings whose columns are shaded
        with *highlight_color*.
    highlight_color :
        Background fill for highlighted columns.  Default ``'#F0EEE8'``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    Basic usage::

        upset({"A": {"x","y"}, "B": {"y","z"}, "C": {"z","x"}})
        

    Wrap all intersections across multiple rows, 12 columns per row::

        fig = upset(my_sets, wrap=True, max_cols=12, singletons=True)
        fig.savefig("wrapped.pdf", bbox_inches="tight", transparent=True)

    Inclusive mode with per-set colour::

        upset(
            my_sets,
            mode="inclusive",
            color=True,
            cmap="Set2",
            gene_fontsize=7,
            title="Shared gene sets",
        )

    Dark theme::

        with plt.style.context("dark_background"):
            upset(my_sets)

    Highlight a specific intersection::

        upset(
            my_sets,
            highlight=[frozenset(["Cancer drivers", "DNA repair"])],
        )
    """
    if isinstance(sets, dict):
        sets = {k: set(v) for k, v in sets.items()}
    else:
        sets = {x.name(): set(x) for x in sets}
    names = list(sets.keys())
    n_sets = len(names)

    if n_sets < 2:
        raise ValueError("upset requires at least 2 sets.")

    highlight_set: set[frozenset[str]] = set(highlight) if highlight else set()

    # bars=False collapses the size-bar region
    if not bars:
        show_size_bars = False
        size_bar_cols = 1.5

    # resolve font sizes
    _count_fs = count_fontsize if count_fontsize is not None else gene_fontsize - 0.5
    _label_fs = label_fontsize if label_fontsize is not None else gene_fontsize + 0.5
    _title_fs = title_fontsize if title_fontsize is not None else gene_fontsize + 4

    # theme foreground
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    _fg = mpl.rcParams.get("text.color", "black")

    def _with_alpha(c: str, a: float) -> tuple:
        r, g, b, _ = mcolors.to_rgba(c)
        return (r, g, b, a)

    _gene_color    = gene_color      if gene_color      is not None else _fg
    _count_color   = count_color     if count_color     is not None else _with_alpha(_fg, 0.45)
    _label_color   = label_color     if label_color     is not None else _fg
    _dot_filled    = dot_filled_color if dot_filled_color is not None else _fg
    _dot_empty     = dot_empty_color  if dot_empty_color  is not None else _with_alpha(_fg, 0.18)
    _col_div_color = col_divider_color if col_divider_color is not None else _with_alpha(_fg, 0.15)
    _baseline_color = baseline_color if baseline_color is not None else _with_alpha(_fg, 0.50)
    _row_grid_color = row_grid_color if row_grid_color is not None else _with_alpha(_fg, 0.12)
    _size_bar_color = size_bar_color if size_bar_color is not None else _fg

    # per-set colours
    _cmap_name = cmap
    if _cmap_name == "tab10" and n_sets > 10:
        _cmap_name = "tab20"
    _cmap = plt.colormaps[_cmap_name]
    _set_colors: list = [_cmap(i / max(n_sets, 1)) for i in range(n_sets)]

    # ── compute intersections ──────────────────────────────────────────────────
    # When wrapping, compute all intersections (no cap); chunking is done below.
    _compute_cap = 10_000 if wrap else max_cols
    all_intersections = compute_intersections(
        sets, min_size=min_size, sort_by=sort_by, max_cols=_compute_cap,
        singletons=singletons, mode=mode,
    )
    if not all_intersections:
        raise ValueError(
            f"No intersections found with min_size={min_size}.  "
            "Try lowering min_size or verifying that sets share elements."
        )

    # split into chunks (one chunk = one row of the figure)
    if wrap:
        chunks = [all_intersections[i:i + max_cols]
                  for i in range(0, len(all_intersections), max_cols)]
    else:
        chunks = [all_intersections]

    n_chunks = len(chunks)

    # ── helpers shared across all chunks ──────────────────────────────────────

    def _two_col_layout(genes: list[str], trunc: bool) -> tuple[list[str], list[str]]:
        by_len = sorted(genes, key=len)
        mid   = (len(by_len) + 1) // 2
        left  = by_len[:mid]
        right = by_len[mid:][::-1]
        if trunc:
            if len(left) > len(right):
                right.append("...")
            else:
                left.append("...")
        return left, right

    def _apply_label_cap(intersections):
        display, truncated = [], []
        for _, genes in intersections:
            if max_labels is not None and len(genes) > max_labels:
                display.append(genes[:max_labels])
                truncated.append(True)
            else:
                display.append(list(genes))
                truncated.append(False)
        return display, truncated

    def _col_geometry(intersections):
        """Return (col_is_wide, col_widths, col_starts, total_x)."""
        col_is_wide = [len(g) > two_col_threshold for _, g in intersections]
        col_widths  = [two_col_ratio if w else 1 for w in col_is_wide]
        col_starts  = []
        x = 0
        for w in col_widths:
            col_starts.append(x)
            x += w
        return col_is_wide, col_widths, col_starts, x

    def _n_rows(d, t, wide):
        if wide:
            return math.ceil((len(d) + int(t)) / 2)
        return len(d) + int(t)

    # precompute per-chunk geometry to determine figure sizing
    chunk_data = []
    for chunk in chunks:
        disp, trunc = _apply_label_cap(chunk)
        wide, widths, starts, total_x = _col_geometry(chunk)
        mr = max((_n_rows(d, t, w) for d, t, w in zip(disp, trunc, wide)), default=1)
        chunk_data.append(dict(
            intersections=chunk,
            display=disp, truncated=trunc,
            col_is_wide=wide, col_widths=widths, col_starts=starts,
            total_x=total_x, max_rows=max(mr, 1),
        ))

    count_rows  = 1 if show_count else 0
    max_total_x = max(cd["total_x"] for cd in chunk_data)

    def _gene_panel_h(max_rows):
        return (max_rows + count_rows) * gene_row_h + 0.20

    matrix_h = n_sets * set_row_h + 0.20

    # all rows share the same physical width (widest chunk sets the scale)
    if figsize is None:
        fig_w = (size_bar_cols + max_total_x) * col_width + 0.20
        fig_h = sum(_gene_panel_h(cd["max_rows"]) for cd in chunk_data) \
                + n_chunks * matrix_h \
                + (0.30 if title else 0.08)
        figsize = (fig_w, fig_h)

    label_frac = (size_bar_cols * col_width) / figsize[0]

    with matplotlib.rc_context({'interactive': False}):
        # ── figure ────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize, facecolor="none")
        fig.patch.set_alpha(0.0)

        # GridSpec: 2 rows per chunk (gene panel + matrix panel)
        height_ratios = []
        for cd in chunk_data:
            height_ratios.append(_gene_panel_h(cd["max_rows"]))
            height_ratios.append(matrix_h)

        gs = gridspec.GridSpec(
            2 * n_chunks, 1,
            figure=fig,
            height_ratios=height_ratios,
            hspace=0.0,
            top=0.94 if title else 0.99,
            bottom=0.01,
            left=label_frac,
            right=0.995,
        )

        max_total = max(len(sets[nm]) for nm in names) or 1
        bar_right = -0.35
        bar_left  = -size_bar_cols + 0.20

        # ── draw each chunk ───────────────────────────────────────────────────────
        for ri, cd in enumerate(chunk_data):
            chunk        = cd["intersections"]
            display      = cd["display"]
            truncated    = cd["truncated"]
            col_is_wide  = cd["col_is_wide"]
            col_widths   = cd["col_widths"]
            col_starts   = cd["col_starts"]
            total_x      = cd["total_x"]
            max_rows     = cd["max_rows"]
            n_cols_chunk = len(chunk)

            ax_gene = fig.add_subplot(gs[2 * ri])
            ax_mat  = fig.add_subplot(gs[2 * ri + 1])

            for ax in (ax_gene, ax_mat):
                ax.set_facecolor("none")
                ax.patch.set_visible(False)

            # ── gene panel ────────────────────────────────────────────────────────
            total_rows = max_rows + count_rows
            # use max_total_x so all rows share the same x scale → dots align
            ax_gene.set_xlim(-size_bar_cols, max_total_x)
            ax_gene.set_ylim(0, total_rows)
            ax_gene.axis("off")

            # highlight
            for ci, (in_sets, _) in enumerate(chunk):
                if in_sets in highlight_set:
                    x0 = col_starts[ci]
                    ax_gene.axvspan(x0, x0 + col_widths[ci],
                                    color=highlight_color, linewidth=0, zorder=0)

            # column dividers
            divider_xs = set()
            for ci in range(n_cols_chunk):
                divider_xs.add(col_starts[ci])
            divider_xs.add(total_x)
            for xd in sorted(divider_xs):
                ax_gene.axvline(xd, color=_col_div_color,
                                linewidth=col_divider_lw, zorder=0)

            ax_gene.axhline(0, color=_baseline_color, linewidth=baseline_lw, zorder=3)

            for ci, (in_sets, _) in enumerate(chunk):
                genes = display[ci]
                trunc = truncated[ci]
                wide  = col_is_wide[ci]
                x0    = col_starts[ci]
                cw    = col_widths[ci]
                cx    = x0 + cw / 2.0

                if wide:
                    left_genes, right_genes = _two_col_layout(genes, trunc)
                    n_rows_col = max(len(left_genes), len(right_genes))
                    for ri2, gene in enumerate(left_genes):
                        is_trunc = gene == "..."
                        ax_gene.text(x0 + gene_pad, ri2 + 0.5,
                                    truncate_suffix if is_trunc else gene,
                                    ha="left", va="center",
                                    fontsize=gene_fontsize, fontfamily=gene_fontfamily,
                                    fontweight=gene_fontweight,
                                    color=_count_color if is_trunc else _gene_color,
                                    clip_on=True)
                    for ri2, gene in enumerate(right_genes):
                        is_trunc = gene == "..."
                        ax_gene.text(x0 + cw - gene_pad, ri2 + 0.5,
                                    truncate_suffix if is_trunc else gene,
                                    ha="right", va="center",
                                    fontsize=gene_fontsize, fontfamily=gene_fontfamily,
                                    fontweight=gene_fontweight,
                                    color=_count_color if is_trunc else _gene_color,
                                    clip_on=True)
                else:
                    n_rows_col = len(genes) + int(trunc)
                    for ri2, gene in enumerate(genes):
                        ax_gene.text(cx, ri2 + 0.5, gene,
                                    ha="center", va="center",
                                    fontsize=gene_fontsize, fontfamily=gene_fontfamily,
                                    fontweight=gene_fontweight,
                                    color=_gene_color, clip_on=True)
                    if trunc:
                        ax_gene.text(cx, len(genes) + 0.5, truncate_suffix,
                                    ha="center", va="center",
                                    fontsize=gene_fontsize, color=_count_color, clip_on=True)

                if show_count:
                    n_total = len(chunk[ci][1])
                    ax_gene.text(cx, n_rows_col + 0.55, str(n_total),
                                ha="center", va="bottom",
                                fontsize=_count_fs, color=_count_color, clip_on=True)

            # ── matrix panel ──────────────────────────────────────────────────────
            ax_mat.set_xlim(-size_bar_cols, max_total_x)
            ax_mat.set_ylim(n_sets - 0.5, -0.5)
            ax_mat.axis("off")

            # highlight
            for ci, (in_sets, _) in enumerate(chunk):
                if in_sets in highlight_set:
                    x0 = col_starts[ci]
                    ax_mat.axvspan(x0, x0 + col_widths[ci],
                                color=highlight_color, linewidth=0, zorder=0)

            # row grid lines
            for i in range(n_sets - 1):
                ax_mat.axhline(i + 0.5, color=_row_grid_color, linewidth=row_grid_lw,
                            xmin=0, xmax=1, zorder=1)

            # set labels + size bars (repeated on every row)
            for i, name in enumerate(names):
                total   = len(sets[name])
                set_col = _set_colors[i] if color else _size_bar_color
                if show_size_bars:
                    bar_w = (total / max_total) * (bar_right - bar_left)
                    ax_mat.barh(i, bar_w, left=bar_left, height=0.28,
                                color=set_col, alpha=size_bar_alpha,
                                linewidth=0, zorder=2)
                label = f"{name} ({total})" if show_set_sizes else name
                ax_mat.text(bar_right - 0.10, i, label,
                            ha="right", va="center",
                            fontsize=_label_fs, fontweight=label_fontweight,
                            fontfamily=label_fontfamily,
                            color=set_col if color else _label_color,
                            clip_on=False)

            # dots
            for ci, (in_sets, _) in enumerate(chunk):
                cx     = col_starts[ci] + col_widths[ci] / 2.0
                filled = {i for i, nm in enumerate(names) if nm in in_sets}
                for i in range(n_sets):
                    c = (_set_colors[i] if color else _dot_filled) if i in filled else _dot_empty
                    ax_mat.plot(cx, i, "o", ms=dot_size, color=c,
                                zorder=3, markeredgewidth=0)

            # column dividers in matrix
            for xd in sorted(divider_xs):
                ax_mat.axvline(xd, color=_col_div_color, linewidth=col_divider_lw, zorder=0)

        if title:
            fig.suptitle(title, fontsize=_title_fs, fontweight=title_fontweight, ha="center")

    return fig