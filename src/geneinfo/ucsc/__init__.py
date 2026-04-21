"""
UCSC REST API track access with per-track-type dispatch.

Public API
----------
- ``list_ucsc_tracks(assembly)`` / ``search_ucsc_tracks(*queries, assembly=...)``
- ``get_ucsc_track_meta(track_name, assembly)``
- ``get_ucsc_track(track_name, assembly, chrom=..., start=..., end=..., ...)``

The goal of :func:`get_ucsc_track` is to return a pandas ``DataFrame`` with a
predictable schema ``chrom, start, end, value`` (plus any extra fields that
were present in the source payload) for every track type where that mapping
is meaningful. Track types that do not map onto a numeric-per-interval model
(bam/cram/hic/vcfTabix, composite parents, ...) raise informative errors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
import re
import sys
import textwrap
import time
import unicodedata
import warnings

import pandas as pd
import requests
from rapidfuzz import process, fuzz

from ..utils import shelve_it


UCSC_API = "https://api.genome.ucsc.edu"

# Fraction of ``maxItemsLimit`` at which we treat a single response as
# "likely truncated" and recursively bisect the region. The UCSC API returns
# exactly maxItemsLimit items when it hits the cap, but results can also be
# truncated at values slightly below (e.g. due to server-side limits on
# certain track types), so we leave a small margin.
_TRUNCATION_RATIO = 0.95

# Minimum region size we're willing to bisect further (bp). Protects against
# infinite recursion on pathologically dense loci and aligns with the
# smallest useful per-base resolution for most tracks.
_MIN_BISECT_BP = 10_000

# --- track-type dispatch tables ---------------------------------------------

BED_LIKE = {
    "bed", "bigBed", "narrowPeak", "broadPeak", "bedGraph",
    "genePred", "bigGenePred", "pgSnp", "peptideMapping",
    "gvf", "bedDetail", "bed9", "bed12", "factorSource",
    "bedRnaElements", "bedMethyl", "bedLogR",
}
WIG_LIKE = {"wig", "bigWig"}
INTERACT = {"interact", "bigInteract"}
# Track types that don't return dataframe-shaped data via the REST API.
BINARY_OR_ALIGNMENT = {"bam", "cram", "hic", "vcfTabix", "vcf", "bigChain", "bigMaf", "maf"}

# Priority order for choosing a "value" column in bed-like tracks.
_VALUE_COLUMN_PREFERENCE = (
    "value", "val", "signalValue", "signal", "score", "obs_exp",
    "pValue", "qValue", "fdr", "logP", "log2FoldChange",
)

# Column-name normalizations applied to bed-like records.
# Some UCSC schemas use bed conventions (chromStart/chromEnd), others use
# genePred conventions (txStart/txEnd), and a few use bare tStart/tEnd.
_BED_RENAME = {
    "chromStart": "start",
    "chromEnd": "end",
    "txStart": "start",
    "txEnd": "end",
    "tStart": "start",
    "tEnd": "end",
}


class UcscApiError(RuntimeError):
    """Raised when the UCSC REST API returns an error envelope or non-OK HTTP."""


class UnsupportedTrackType(NotImplementedError):
    """Raised when a track's type is not representable as chrom/start/end/value."""


# --- low-level HTTP ---------------------------------------------------------


def _request_json(path: str, params: dict, *, retries: int = 3, backoff: float = 1.5) -> dict:
    """GET ``{UCSC_API}{path}`` with ``params`` and decode JSON.

    Retries on 5xx with exponential backoff. Raises :class:`UcscApiError` on
    persistent failure or on an explicit ``{"error": ...}`` envelope.
    """
    url = f"{UCSC_API}{path}"
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            last_exc = exc
        else:
            if resp.status_code < 500:
                if not resp.ok:
                    raise UcscApiError(
                        f"UCSC API {resp.status_code} for {url} params={params}: {resp.text[:500]}"
                    )
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise UcscApiError(
                        f"UCSC API returned non-JSON for {url} params={params}: {exc}"
                    ) from exc
                if isinstance(data, dict) and "error" in data and len(data) <= 2:
                    raise UcscApiError(f"UCSC API error for {url} params={params}: {data['error']}")
                return data
            last_exc = UcscApiError(f"UCSC API {resp.status_code} for {url}: {resp.text[:500]}")
        time.sleep(backoff ** attempt)
    raise UcscApiError(f"UCSC API call failed after {retries} attempts: {last_exc}")


# --- track metadata ---------------------------------------------------------


_TRACK_META_CACHE: dict[str, dict] = {}


@shelve_it()
def _fetch_ucsc_tracks_meta(assembly: str) -> dict:
    """Fetch and cache (shelve) the per-track metadata dict for an assembly."""
    data = _request_json("/list/tracks", {"genome": assembly})
    if assembly not in data:
        raise UcscApiError(f"Assembly {assembly!r} not present in /list/tracks response")
    return data[assembly]


def _fetch_all_tracks(assembly: str) -> dict:
    """Return the per-track metadata dict, hot in-process + shelve-cached on disk."""
    if assembly in _TRACK_META_CACHE:
        return _TRACK_META_CACHE[assembly]
    meta = _fetch_ucsc_tracks_meta(assembly)
    _TRACK_META_CACHE[assembly] = meta
    return meta


@shelve_it()
def _fetch_ucsc_track_data(
    assembly: str,
    track_name: str,
    chrom: str | None,
    start: int | None,
    end: int | None,
) -> dict:
    """Fetch and cache (shelve) the raw ``/getData/track`` envelope.

    Arguments are explicit and stringifiable so ``shelve_it`` produces a
    stable key per (assembly, track, region) tuple.
    """
    params: dict[str, str] = {"genome": assembly, "track": track_name}
    if chrom is not None:
        params["chrom"] = chrom
        if start is not None and end is not None:
            params["start"] = str(start)
            params["end"] = str(end)
    return _request_json("/getData/track", params)


def _is_container(meta: dict) -> bool:
    """True iff this entry is a composite/view container (never a data-bearing leaf)."""
    if not isinstance(meta, dict):
        return False
    return (
        meta.get("compositeContainer") == "TRUE"
        or meta.get("compositeViewContainer") == "TRUE"
    )


def _flatten_tracks(tracks: dict, parent: str | None = None) -> dict:
    """Flatten nested subtrack entries into a single name -> metadata dict.

    UCSC's ``/list/tracks`` response uses ad-hoc nesting: a container entry
    has ``compositeContainer='TRUE'`` (or ``compositeViewContainer='TRUE'``)
    and holds its children under arbitrary keys, with each child being a
    dict that itself looks like a track. There is also a legacy
    ``subtracks`` key on some entries — we honour both.
    """
    flat = {}
    for name, meta in tracks.items():
        if not isinstance(meta, dict):
            continue
        entry = dict(meta)
        if parent is not None:
            entry.setdefault("_parent", parent)
        flat[name] = entry

        # Legacy: explicit `subtracks` dict.
        subtracks = meta.get("subtracks")
        if isinstance(subtracks, dict):
            flat.update(_flatten_tracks(subtracks, parent=name))

        # Ad-hoc nested children: any dict-valued entry that itself looks
        # like a track (has a ``type`` or ``shortLabel``).
        for k, v in meta.items():
            if k in {"subtracks", "subGroups"}:
                continue
            if isinstance(v, dict) and ("type" in v or "shortLabel" in v):
                flat.update(_flatten_tracks({k: v}, parent=name))
    return flat


def get_ucsc_track_meta(track_name: str, assembly: str) -> dict:
    """Return the UCSC metadata dict for a single track (searching subtracks)."""
    tracks = _flatten_tracks(_fetch_all_tracks(assembly))
    if track_name not in tracks:
        raise KeyError(f"Track {track_name!r} not found in assembly {assembly!r}")
    return tracks[track_name]


def list_ucsc_subtracks(track_name: str, assembly: str) -> pd.DataFrame:
    """Return a DataFrame of leaf subtracks under a composite/view container.

    Columns: ``name``, ``type``, ``shortLabel``. For a non-container track
    this returns an empty DataFrame.
    """
    meta = get_ucsc_track_meta(track_name, assembly)
    leaves = _walk_leaves(meta) if _is_container(meta) else []
    return pd.DataFrame(leaves, columns=["name", "type", "shortLabel"])


# --- listing / searching ----------------------------------------------------


def _normalize(name: str) -> str:
    name = name.lower()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^\w\s]", "", name)
    return name.strip()


def list_ucsc_tracks(assembly: str, *, label_wrap: int = 80) -> None:
    """Print all tracks for an assembly (including flattened subtracks)."""
    search_ucsc_tracks(assembly=assembly, label_wrap=label_wrap)


def search_ucsc_tracks(*queries: str, assembly: str, label_wrap: int = 80) -> None:
    """Print tracks matching fuzzy queries (or all tracks when no query given)."""
    if assembly is None:
        raise ValueError("assembly is required")

    tracks = _flatten_tracks(_fetch_all_tracks(assembly))
    pairs = sorted(
        ((n, m.get("longLabel", "")) for n, m in tracks.items()),
        key=lambda t: t[0].upper(),
    )
    if not pairs:
        return

    if not queries:
        entries = pairs
    else:
        names, _labels = zip(*pairs)
        normalized = [_normalize(n) for n in names]
        scores: dict[tuple[str, int], float] = defaultdict(float)
        for query in queries:
            for word in query.split():
                hits = process.extract(
                    _normalize(word), normalized,
                    scorer=fuzz.WRatio, score_cutoff=80.0, limit=100,
                )
                for name, score, index in hits:
                    scores[(name, index)] += score
        ranked = sorted(((s, k) for k, s in scores.items()), reverse=True)
        entries = [pairs[idx] for _score, (_name, idx) in ranked]

    if not entries:
        return
    ljust = max(len(n) for n, _ in entries) + 2
    for name, label in entries:
        wrapped = "\n".join(textwrap.wrap(
            label, width=label_wrap, subsequent_indent=" " * ljust,
        ))
        print(name.ljust(ljust) + wrapped)


# --- adapters ---------------------------------------------------------------


def _pick_value_column(df: pd.DataFrame, override: str | None) -> str | None:
    """Return the name of the column that should be exposed as ``value``."""
    if override is not None:
        if override not in df.columns:
            raise KeyError(f"value_column={override!r} not in track columns {list(df.columns)}")
        return override
    for candidate in _VALUE_COLUMN_PREFERENCE:
        if candidate in df.columns and pd.api.types.is_numeric_dtype(df[candidate]):
            return candidate
    # Fall back: first numeric, non-coordinate column.
    skip = {"chrom", "start", "end", "chromStart", "chromEnd"}
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _coerce_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Try to convert each given column to numeric when it parses cleanly."""
    for col in cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only adopt the conversion if the original was essentially numeric
        # (avoid turning a string-typed categorical into mostly-NaN).
        non_null_ratio = converted.notna().mean() if len(converted) else 0.0
        if non_null_ratio >= 0.9:
            df[col] = converted


def _reorder_schema(df: pd.DataFrame, value_col: str | None) -> pd.DataFrame:
    """Reorder so ``chrom, start, end, value, ...`` come first when present."""
    front = [c for c in ("chrom", "start", "end") if c in df.columns]
    if value_col is not None and value_col != "value":
        df = df.rename(columns={value_col: "value"})
        value_col = "value"
    if value_col and value_col in df.columns:
        front.append("value")
    remaining = [c for c in df.columns if c not in front]
    return df[front + remaining]


def _adapt_bed_like(payload, envelope: dict, value_column: str | None) -> pd.DataFrame:
    if not isinstance(payload, list):
        # Some bed-like tracks still come back keyed by chrom when the
        # request spans multiple chromosomes. Flatten them the same way
        # as the wig path.
        return _adapt_wig_like(payload, envelope, value_column)
    if not payload:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    df = pd.DataFrame.from_records(payload)
    df = df.rename(columns={k: v for k, v in _BED_RENAME.items() if k in df.columns})
    # Some tracks return start/end as strings.
    _coerce_numeric_inplace(df, ("start", "end"))
    _coerce_numeric_inplace(df, [c for c in df.columns if c not in {"chrom", "start", "end"}])
    value_col = _pick_value_column(df, value_column)
    return _reorder_schema(df, value_col)


def _adapt_wig_like(payload, envelope: dict, value_column: str | None) -> pd.DataFrame:
    # Wig payloads can be either a flat list or a dict keyed by chromosome.
    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame(columns=["chrom", "start", "end", "value"])
        df = pd.DataFrame.from_records(payload)
        if "chrom" not in df.columns and "chrom" in envelope:
            df["chrom"] = envelope["chrom"]
    elif isinstance(payload, dict):
        frames = []
        for chrom, rows in payload.items():
            if not isinstance(rows, list) or not rows:
                continue
            sub = pd.DataFrame.from_records(rows)
            sub["chrom"] = chrom
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=["chrom", "start", "end", "value"])
        df = pd.concat(frames, ignore_index=True)
    else:
        raise UcscApiError(f"Unexpected wig payload type: {type(payload).__name__}")

    df = df.rename(columns={k: v for k, v in _BED_RENAME.items() if k in df.columns})
    _coerce_numeric_inplace(df, ("start", "end"))
    _coerce_numeric_inplace(df, [c for c in df.columns if c not in {"chrom", "start", "end"}])
    value_col = _pick_value_column(df, value_column)
    return _reorder_schema(df, value_col)


def _adapt_interact(payload, envelope: dict, value_column: str | None, *, side: str = "source") -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["chrom", "start", "end", "value"])
    df = pd.DataFrame.from_records(payload)
    prefix = "source" if side == "source" else "target"
    mapping = {
        f"{prefix}Chrom": "chrom",
        f"{prefix}Start": "start",
        f"{prefix}End": "end",
    }
    missing = [k for k in mapping if k not in df.columns]
    if missing:
        # Fall through to bed-like handling if the payload doesn't look like interact.
        return _adapt_bed_like(payload, envelope, value_column)
    df = df.rename(columns=mapping)
    _coerce_numeric_inplace(df, ("start", "end"))
    _coerce_numeric_inplace(df, [c for c in df.columns if c not in {"chrom", "start", "end"}])
    value_col = _pick_value_column(df, value_column)
    return _reorder_schema(df, value_col)


# --- dispatch ---------------------------------------------------------------


def _track_type_family(track_type: str | None) -> str:
    """Map a raw UCSC ``type`` string to one of: bed, wig, interact, binary, unknown.

    Besides the explicit lookup tables, the function recognizes suffix/prefix
    patterns that clearly indicate a family (e.g. ``bigDbSnp``, ``bigBarChart``,
    ``bigPsl``, ``bigChain``, ``wigMaf``, ...) so that we don't emit
    "unknown type" warnings for tracks the UCSC API actually returns as
    bed-like rows.
    """
    if not track_type:
        return "unknown"
    head = track_type.split()[0]
    if head in BED_LIKE:
        return "bed"
    if head in WIG_LIKE:
        return "wig"
    if head in INTERACT:
        return "interact"
    if head in BINARY_OR_ALIGNMENT:
        return "binary"
    # Heuristic fallbacks based on naming conventions.
    lowered = head.lower()
    if lowered.startswith("wig") or lowered.endswith("wig"):
        return "wig"
    if lowered.startswith("big") and lowered not in {"bigchain", "bigmaf"}:
        # bigBed, bigGenePred, bigNarrowPeak, bigPsl, bigDbSnp, bigBarChart,
        # bigLolly, bigRmsk, bigSnp, ... all serialize to bed-like rows.
        return "bed"
    return "unknown"


def _walk_leaves(meta: dict) -> list[tuple[str, str, str]]:
    """Return ``[(name, type, shortLabel), ...]`` for every leaf under ``meta``.

    Leaves are entries that are not themselves containers. Walks both the
    legacy ``subtracks`` dict and ad-hoc nested track-shaped values.
    """
    leaves: list[tuple[str, str, str]] = []

    def visit(name: str, entry: dict) -> None:
        if _is_container(entry):
            # Recurse into children; the container itself isn't a leaf.
            subtracks = entry.get("subtracks")
            if isinstance(subtracks, dict):
                for k, v in subtracks.items():
                    if isinstance(v, dict):
                        visit(k, v)
            for k, v in entry.items():
                if k in {"subtracks", "subGroups"}:
                    continue
                if isinstance(v, dict) and ("type" in v or "shortLabel" in v):
                    visit(k, v)
        else:
            leaves.append((
                name,
                str(entry.get("type", "")).split()[0] if entry.get("type") else "",
                str(entry.get("shortLabel", "")),
            ))

    # Iterate the container's direct children, same discovery rules as visit().
    subtracks = meta.get("subtracks")
    if isinstance(subtracks, dict):
        for k, v in subtracks.items():
            if isinstance(v, dict):
                visit(k, v)
    for k, v in meta.items():
        if k in {"subtracks", "subGroups"}:
            continue
        if isinstance(v, dict) and ("type" in v or "shortLabel" in v):
            visit(k, v)
    return leaves


def _format_subtrack_table(leaves: list[tuple[str, str, str]], max_rows: int = 50) -> str:
    if not leaves:
        return ""
    shown = leaves[:max_rows]
    name_w = max(len(n) for n, _, _ in shown)
    type_w = max((len(t) for _, t, _ in shown), default=0)
    lines = [f"  {n.ljust(name_w)}  {t.ljust(type_w)}  {lbl}" for n, t, lbl in shown]
    body = "\n".join(lines)
    if len(leaves) > max_rows:
        body += f"\n  ... and {len(leaves) - max_rows} more"
    return body


def _check_container(meta: dict, track_name: str, assembly: str) -> None:
    """Raise a helpful error if the track is a composite/view container.

    Important: ``compositeTrack='on'`` and ``superTrack='on'`` are set on
    *leaf* tracks that belong to a composite/super-track too — they are
    NOT container markers. The authoritative markers are
    ``compositeContainer='TRUE'`` and ``compositeViewContainer='TRUE'``.

    The raised error enumerates every leaf subtrack (name, type, short
    label) so the caller can pick the right one without a second lookup.
    """
    if not _is_container(meta):
        return
    leaves = _walk_leaves(meta)
    table = _format_subtrack_table(leaves)
    if table:
        msg = (
            f"Track {track_name!r} on {assembly!r} is a composite/view container; "
            f"pick one of its subtracks instead. Available subtracks "
            f"({len(leaves)} total):\n{table}"
        )
    else:
        msg = (
            f"Track {track_name!r} on {assembly!r} is a composite/view container "
            f"with no listed subtracks."
        )
    raise UnsupportedTrackType(msg)


# --- batching helpers -------------------------------------------------------


def _envelope_truncated(envelope: dict) -> bool:
    """True if the UCSC response likely hit its item cap.

    UCSC returns ``itemsReturned`` and ``maxItemsLimit`` fields in each
    envelope. When ``itemsReturned`` is at or near the cap we assume the
    response was truncated and must be subdivided.
    """
    items = envelope.get("itemsReturned")
    cap = envelope.get("maxItemsLimit") or envelope.get("maxItemsOutput")
    if items is None or cap is None:
        return False
    try:
        return int(items) >= int(cap) * _TRUNCATION_RATIO
    except (TypeError, ValueError):
        return False


def _fetch_region_batched(
    assembly: str,
    track_name: str,
    chrom: str,
    start: int,
    end: int,
) -> list[dict]:
    """Fetch a single-chromosome region, subdividing if the response truncates.

    Returns a list of raw UCSC envelopes covering ``[start, end)`` with no
    gaps. Each envelope is individually shelved via
    :func:`_fetch_ucsc_track_data`, so re-fetching the same (possibly
    subdivided) region costs nothing.
    """
    envelope = _fetch_ucsc_track_data(assembly, track_name, chrom, start, end)

    if not _envelope_truncated(envelope):
        return [envelope]

    # Can't subdivide further: keep the truncated envelope and warn.
    if end - start <= _MIN_BISECT_BP:
        warnings.warn(
            f"Track {track_name!r}: region {chrom}:{start}-{end} hit UCSC's "
            f"item cap and is already smaller than {_MIN_BISECT_BP} bp; "
            f"results for this region are truncated.",
            stacklevel=3,
        )
        return [envelope]

    mid = (start + end) // 2
    left = _fetch_region_batched(assembly, track_name, chrom, start, mid)
    right = _fetch_region_batched(assembly, track_name, chrom, mid, end)
    return left + right


def _concat_envelope_payloads(
    envelopes: list[dict], track_name: str,
) -> tuple[list | dict, dict]:
    """Merge the per-envelope payloads under ``track_name``.

    Bed-like payloads (lists of records) are concatenated. Wig-like
    payloads (dicts keyed by chrom) are merged chrom-by-chrom. Returns
    ``(combined_payload, representative_envelope)`` — the envelope is used
    by adapters for fields like ``chrom``/``start``/``end``.
    """
    list_payloads: list[list] = []
    dict_payloads: list[dict] = []
    for env in envelopes:
        p = env.get(track_name, [])
        if isinstance(p, list):
            list_payloads.append(p)
        elif isinstance(p, dict):
            dict_payloads.append(p)

    if dict_payloads and not list_payloads:
        merged: dict[str, list] = {}
        for p in dict_payloads:
            for chrom, rows in p.items():
                if not isinstance(rows, list):
                    continue
                merged.setdefault(chrom, []).extend(rows)
        return merged, envelopes[0] if envelopes else {}

    if list_payloads and not dict_payloads:
        combined: list = []
        for p in list_payloads:
            combined.extend(p)
        return combined, envelopes[0] if envelopes else {}

    # Mixed or empty: fall back to the first envelope's shape.
    if envelopes:
        return envelopes[0].get(track_name, []), envelopes[0]
    return [], {}


def _assembly_chromosomes(assembly: str) -> list[str]:
    """Return the list of chromosomes for an assembly in their native order."""
    # Deferred import — keeps ``geneinfo.ucsc`` usable without triggering the
    # coords module's network dependencies until we actually need chrom sizes.
    from ..coords import chromosome_lengths
    return [name for name, _length in chromosome_lengths(assembly=assembly)]


# --- public entry point -----------------------------------------------------


def get_ucsc_track(
    track_name: str,
    assembly: str,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
    *,
    as_frame: bool = True,
    value_column: str | None = None,
    interact_side: str = "source",
    force_type: str | None = None,
) -> pd.DataFrame | dict | list[dict]:
    """Fetch a UCSC track and return a normalized ``DataFrame``.

    Fetches are automatically batched to stay below UCSC's
    ``maxItemsLimit``:

    - When ``chrom`` is omitted, one request is made per chromosome in the
      assembly.
    - When a single-chromosome request comes back truncated
      (``itemsReturned >= maxItemsLimit * 0.95``), the region is bisected
      recursively until every response is un-truncated, or until the
      candidate region drops below
      :data:`_MIN_BISECT_BP` bp (at which point a warning is emitted).

    Each sub-request is individually shelve-cached, so re-running the same
    query costs nothing beyond the adapter step.

    Parameters
    ----------
    track_name
        UCSC track identifier (as listed by :func:`search_ucsc_tracks`).
    assembly
        Genome assembly, e.g. ``'hg38'``.
    chrom, start, end
        Optional region restriction. ``chrom`` is auto-prefixed with
        ``'chr'`` when missing. When ``chrom`` is ``None`` the entire
        assembly is fetched chromosome-by-chromosome.
    as_frame
        When ``False`` return the raw JSON payloads from UCSC. If the
        fetch was batched, a ``list`` of envelopes is returned; otherwise
        a single envelope ``dict``.
    value_column
        Override the column selected as ``value``. By default the first
        numeric column matching :data:`_VALUE_COLUMN_PREFERENCE` wins, then
        the first other numeric column.
    interact_side
        For ``interact`` / ``bigInteract`` tracks, whether to expose the
        ``source*`` (default) or ``target*`` side as ``chrom/start/end``.
    force_type
        Override the dispatch family (``'bed'``, ``'wig'``, ``'interact'``).
        Useful when the metadata ``type`` is missing or inaccurate.

    Returns
    -------
    pandas.DataFrame
        Columns ``chrom, start, end, value`` first (when derivable), then
        any remaining original columns. The index is a fresh ``RangeIndex``.

    Raises
    ------
    UnsupportedTrackType
        For track types whose payload cannot be flattened into a value
        dataframe (bam/cram/hic/vcfTabix/...) or for composite/super-track
        containers. The message identifies acceptable alternatives.
    UcscApiError
        For HTTP or API-level failures.
    """
    if chrom is not None and not str(chrom).startswith("chr"):
        chrom = f"chr{chrom}"

    meta = get_ucsc_track_meta(track_name, assembly)
    _check_container(meta, track_name, assembly)
    track_type = meta.get("type", "")
    family = force_type or _track_type_family(track_type)

    if family == "binary":
        big_url = meta.get("bigDataUrl")
        hint = f" Download the source file directly: {big_url}" if big_url else ""
        raise UnsupportedTrackType(
            f"Track {track_name!r} has type {track_type!r} which returns "
            f"binary/alignment data that cannot be represented as chrom/start/end/value.{hint}"
        )

    # Collect one or more envelopes, batched to stay under UCSC's item cap.
    envelopes: list[dict] = []
    if chrom is None:
        # Genome-wide: iterate chromosomes. We cannot pass a bare
        # start/end without chrom through the API, so those are ignored
        # here (they only apply inside a single chromosome).
        for c in _assembly_chromosomes(assembly):
            try:
                length = None
                # We don't actually need the length — UCSC handles a
                # missing ``end``, but having it lets us subdivide. Ask
                # the coords module again only if the first envelope
                # turns out to be truncated.
                env0 = _fetch_ucsc_track_data(assembly, track_name, c, None, None)
                if _envelope_truncated(env0):
                    from ..coords import chromosome_lengths
                    length = dict(chromosome_lengths(assembly=assembly)).get(c)
                    if length is None:
                        envelopes.append(env0)
                        warnings.warn(
                            f"Track {track_name!r} on {c}: response truncated "
                            f"and chromosome length unavailable for "
                            f"subdivision.",
                            stacklevel=2,
                        )
                        continue
                    envelopes.extend(
                        _fetch_region_batched(assembly, track_name, c, 0, int(length))
                    )
                else:
                    envelopes.append(env0)
            except UcscApiError as exc:
                warnings.warn(
                    f"Track {track_name!r} on {c}: {exc}; skipping this chromosome.",
                    stacklevel=2,
                )
    elif start is not None and end is not None:
        envelopes.extend(
            _fetch_region_batched(assembly, track_name, chrom, int(start), int(end))
        )
    else:
        # Single chromosome, no explicit region: fetch and subdivide if needed.
        env0 = _fetch_ucsc_track_data(assembly, track_name, chrom, None, None)
        if _envelope_truncated(env0):
            from ..coords import chromosome_lengths
            length = dict(chromosome_lengths(assembly=assembly)).get(chrom)
            if length is not None:
                envelopes.extend(
                    _fetch_region_batched(assembly, track_name, chrom, 0, int(length))
                )
            else:
                envelopes.append(env0)
        else:
            envelopes.append(env0)

    if as_frame is False:
        return envelopes if len(envelopes) != 1 else envelopes[0]

    payload, representative = _concat_envelope_payloads(envelopes, track_name)

    if family == "bed":
        df = _adapt_bed_like(payload, representative, value_column)
    elif family == "wig":
        df = _adapt_wig_like(payload, representative, value_column)
    elif family == "interact":
        df = _adapt_interact(payload, representative, value_column, side=interact_side)
    else:
        try:
            df = _adapt_bed_like(payload, representative, value_column)
        except Exception:
            df = _adapt_wig_like(payload, representative, value_column)
        if df.empty and payload:
            raise UnsupportedTrackType(
                f"Track {track_name!r} has type {track_type!r} which is not "
                f"recognized. Pass force_type='bed'|'wig'|'interact' to override, "
                f"or call with as_frame=False to inspect the raw payload."
            )
        if track_type and family == "unknown":
            warnings.warn(
                f"Track {track_name!r} has unknown type {track_type!r}; "
                f"attempted best-effort bed/wig parsing.",
                stacklevel=2,
            )

    return df.reset_index(drop=True)


__all__ = [
    "get_ucsc_track",
    "get_ucsc_track_meta",
    "list_ucsc_subtracks",
    "list_ucsc_tracks",
    "search_ucsc_tracks",
    "UcscApiError",
    "UnsupportedTrackType",
]
