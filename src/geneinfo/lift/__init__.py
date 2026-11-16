"""
Lifting genomic intervals between assemblies.

Wraps the UCSC ``liftOver`` binary (bioconda package ``ucsc-liftover``) and
downloads UCSC chain files on demand. Intervals use BED conventions: 0-based,
half-open coordinates. Chromosome names must match the chain files (e.g.
``chr1``, not ``1``).
"""

import gzip
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

__all__ = ['get_chain_file', 'liftover']

UCSC_CHAIN_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/{src}/liftOver/{name}'
UCSC_ALIAS_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/{asm}/bigZips/{asm}.chromAlias.txt'

DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'geneinfo' / 'chains'

BED_COLUMNS = ['chrom', 'start', 'end', 'name', 'score', 'strand',
               'thickStart', 'thickEnd', 'itemRgb', 'blockCount',
               'blockSizes', 'blockStarts']

PathLike = Union[str, os.PathLike]


def _chain_name(from_assembly: str, to_assembly: str) -> str:
    """UCSC file name of the chain lifting ``from_assembly`` to ``to_assembly``."""
    return f'{from_assembly}To{to_assembly[0].upper()}{to_assembly[1:]}.over.chain.gz'


def get_chain_file(from_assembly: str, to_assembly: str,
                   cache_dir: Optional[PathLike] = None,
                   overwrite: bool = False) -> Path:
    """
    Path to a UCSC chain file, downloading it if it is not cached.

    Parameters
    ----------
    from_assembly :
        UCSC name of the source assembly, e.g. ``'hg38'``.
    to_assembly :
        UCSC name of the target assembly, e.g. ``'papAnu4'``.
    cache_dir :
        Directory for the downloaded chain files. Defaults to
        ``~/.cache/geneinfo/chains``.
    overwrite :
        Download again even if the file is cached.

    Returns
    -------
    :
        Path to the gzipped chain file.

    Examples
    --------
    ```python
    get_chain_file('hg38', 'papAnu4')
    ```
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    name = _chain_name(from_assembly, to_assembly)
    path = cache_dir / name
    if path.exists() and not overwrite:
        return path

    url = UCSC_CHAIN_URL.format(src=from_assembly, name=name)
    if not _download(url, path):
        index = UCSC_CHAIN_URL.format(src=from_assembly, name='')
        raise ValueError(
            f'UCSC has no chain file {name}. See {index} for the chain '
            'files available, or pass a chain file path explicitly.')
    return path


def _download(url: str, path: Path) -> bool:
    """Download ``url`` to ``path``. Returns False if the server has no such file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.name
    with requests.get(url, stream=True, timeout=60) as response:
        if response.status_code == 404:
            return False
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0)) or None
        tmp_path = path.with_name(path.name + '.part')
        try:
            with open(tmp_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True,
                                                 desc=name, leave=False) as bar:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return True


def _chrom_alias_map(assembly: str, cache_dir: Optional[PathLike] = None) -> Optional[dict]:
    """
    Map from chromosome names of all styles (RefSeq, GenBank, Ensembl, ...)
    to UCSC names for an assembly, from the UCSC chromAlias table. UCSC names
    map to themselves. Returns None, with a warning, if the table is not
    available.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    path = cache_dir / f'{assembly}.chromAlias.txt'
    if not path.exists():
        url = UCSC_ALIAS_URL.format(asm=assembly)
        try:
            found = _download(url, path)
        except requests.RequestException as e:
            warnings.warn(f'Could not download the chromosome alias table for {assembly} '
                          f'({e}). Chromosome names are used as given.', stacklevel=4)
            return None
        if not found:
            warnings.warn(f'UCSC has no chromosome alias table for {assembly!r}. '
                          'Chromosome names are used as given.', stacklevel=4)
            return None
    aliases = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            ucsc, *others = line.rstrip('\n').split('\t')
            for name in others:
                if name:
                    aliases.setdefault(name, ucsc)
            aliases[ucsc] = ucsc  # UCSC names take precedence
    return aliases


def _translate_chroms(chroms: np.ndarray, assembly: str,
                      cache_dir: Optional[PathLike]) -> np.ndarray:
    """Translate chromosome names to UCSC names, warning about unknown names."""
    aliases = _chrom_alias_map(assembly, cache_dir)
    if aliases is None:
        return chroms
    unknown = sorted({c for c in chroms if c not in aliases})
    if unknown:
        shown = ', '.join(unknown[:10]) + (', ...' if len(unknown) > 10 else '')
        warnings.warn(f'{len(unknown)} chromosome name(s) are not known in {assembly}, '
                      f'and intervals on them will not map: {shown}', stacklevel=3)
    return np.array([aliases.get(c, c) for c in chroms], dtype=object)


def _liftover_binary() -> str:
    binary = shutil.which('liftOver')
    if binary is None:
        raise FileNotFoundError(
            'The UCSC liftOver binary is not on PATH. Install it with '
            '"conda install -c bioconda ucsc-liftover" (or "pixi add ucsc-liftover").')
    return binary


def _read_bed(path: PathLike) -> pd.DataFrame:
    """Read a BED file (optionally gzipped) without a header line."""
    # count leading 'track', 'browser' and comment lines
    opener = gzip.open if str(path).endswith('.gz') else open
    n_header = 0
    with opener(path, 'rt') as f:
        for line in f:
            if not line.startswith(('track', 'browser', '#')) and line.strip():
                break
            n_header += 1
    df = pd.read_csv(path, sep='\t', header=None, skiprows=n_header, dtype={0: str})
    extra = [f'col{i}' for i in range(len(BED_COLUMNS), df.shape[1])]
    df.columns = (BED_COLUMNS + extra)[:df.shape[1]]
    for col in ('start', 'end'):
        df[col] = pd.to_numeric(df[col])
    return df


def _read_input(intervals, chrom_col: str, start_col: str, end_col: str,
                pos_col: Optional[str], pos_base: int):
    """
    Return a validated copy of the input, and the chromosomes, starts and ends
    of the intervals to lift as numpy arrays (BED coordinates). With
    ``pos_col``, each position is lifted as the one-base interval it denotes.
    """
    if isinstance(intervals, (str, os.PathLike)):
        df = _read_bed(intervals)
        chrom_col, start_col, end_col = 'chrom', 'start', 'end'
    elif isinstance(intervals, pd.DataFrame):
        df = intervals.copy()
    else:
        raise TypeError('intervals must be a pandas DataFrame or a path to a BED file')

    coord_cols = [chrom_col, start_col, end_col] if pos_col is None else [chrom_col, pos_col]
    missing = [c for c in coord_cols if c not in df.columns]
    if missing:
        raise ValueError(f'intervals are missing column(s) {missing}')
    if df[coord_cols].isna().any().any():
        raise ValueError(f'intervals contain missing values in {coord_cols}')
    for col in coord_cols[1:]:
        values = pd.to_numeric(df[col])
        if not np.array_equal(values, values.round()):
            raise ValueError(f'column {col!r} contains non-integer coordinates')

    chroms = df[chrom_col].astype(str).to_numpy()
    if pos_col is None:
        starts = df[start_col].to_numpy(dtype='int64')
        ends = df[end_col].to_numpy(dtype='int64')
        if (starts < 0).any() or (ends < starts).any():
            raise ValueError('intervals must satisfy 0 <= start <= end')
    else:
        starts = df[pos_col].to_numpy(dtype='int64') - pos_base
        ends = starts + 1
        if (starts < 0).any():
            raise ValueError(f'positions must be >= {pos_base} when pos_base={pos_base}')
    return df, chroms, starts, ends


def _available_cores() -> int:
    """
    Number of cores available: SLURM_CPUS_PER_TASK inside a SLURM job,
    otherwise the cores this process may run on.
    """
    slurm = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass
    return os.process_cpu_count() or 1


def _available_memory() -> Optional[int]:
    """
    Bytes of memory available: the SLURM job's allocation inside a SLURM job,
    otherwise the system's available memory. None if unknown.
    """
    try:
        if os.environ.get('SLURM_MEM_PER_NODE'):
            return int(os.environ['SLURM_MEM_PER_NODE']) * 2**20
        if os.environ.get('SLURM_MEM_PER_CPU'):
            return int(os.environ['SLURM_MEM_PER_CPU']) * 2**20 * _available_cores()
    except ValueError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _max_jobs_for_memory(chain_file: Path, chunk_size: int) -> Optional[int]:
    """
    Number of liftOver processes that fit in 80% of available memory. Each
    process holds the parsed chain file (measured at about 15 times the size
    of a gzipped chain file) plus about 300 bytes per interval.
    """
    memory = _available_memory()
    if memory is None:
        return None
    size = chain_file.stat().st_size
    per_process = (15 if chain_file.suffix == '.gz' else 5) * size + 300 * chunk_size
    return max(1, int(0.8 * memory // per_process))


def _lift_coords(chroms, starts, ends, chain_file: PathLike, min_match: float,
                 n_jobs: int, auto_jobs: bool = False) -> pd.DataFrame:
    """
    Lift intervals with liftOver.

    Only unique intervals are lifted, split across up to ``n_jobs`` liftOver
    processes (further limited by available memory if ``auto_jobs``). Returns a frame with columns chrom, start, end aligned to the
    input positions: one row per input interval, NaN where the interval does
    not map.
    """
    n = len(chroms)
    if n == 0:
        return _align_coords(None, n)
    chain_file = Path(chain_file)
    if not chain_file.exists():
        raise FileNotFoundError(f'chain file not found: {chain_file}')

    keys = pd.DataFrame({'chrom': np.asarray(chroms, dtype=str),
                         'start': np.asarray(starts, dtype='int64'),
                         'end': np.asarray(ends, dtype='int64')})
    # index of each interval's unique interval, numbered in order of first appearance
    codes = keys.groupby(['chrom', 'start', 'end'], sort=False).ngroup().to_numpy()
    unique = keys.drop_duplicates().reset_index(drop=True)
    lifted = _run_liftover(unique, chain_file, min_match, n_jobs, auto_jobs)
    return lifted.iloc[codes].reset_index(drop=True)


# minimum number of intervals per liftOver process: each process spends about
# a second loading the chain file, and lifts roughly 2,000 intervals per second
MIN_CHUNK_SIZE = 10_000


def _run_liftover(bed: pd.DataFrame, chain_file: Path, min_match: float,
                  n_jobs: int, auto_jobs: bool = False) -> pd.DataFrame:
    """Run liftOver on intervals in parallel chunks; returns coordinates aligned to ``bed``."""
    n = len(bed)
    n_chunks = max(1, min(n_jobs, -(-n // MIN_CHUNK_SIZE)))
    if auto_jobs and n_chunks > 1:
        max_jobs = _max_jobs_for_memory(chain_file, -(-n // n_chunks))
        if max_jobs is not None:
            n_chunks = min(n_chunks, max_jobs)
    bounds = np.linspace(0, n, n_chunks + 1).astype(int)
    bed = bed.assign(id=np.arange(n))
    binary = _liftover_binary()
    with tempfile.TemporaryDirectory(prefix='geneinfo_liftover_') as tmp:
        tmp = Path(tmp)
        procs = []
        try:
            for i in range(n_chunks):
                in_bed, out_bed = tmp / f'in{i}.bed', tmp / f'out{i}.bed'
                bed.iloc[bounds[i]:bounds[i + 1]].to_csv(in_bed, sep='\t', header=False,
                                                          index=False)
                cmd = [binary, f'-minMatch={min_match}', str(in_bed), str(chain_file),
                       str(out_bed), str(tmp / f'unmapped{i}.bed')]
                procs.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                               stderr=subprocess.PIPE, text=True), out_bed))
            mapped = []
            for proc, out_bed in procs:
                _, stderr = proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f'liftOver failed (exit code {proc.returncode}):\n{stderr}')
                if out_bed.stat().st_size > 0:
                    mapped.append(pd.read_csv(out_bed, sep='\t', header=None,
                                              names=['chrom', 'start', 'end', 'id'],
                                              dtype={'chrom': str}))
        finally:
            for proc, _ in procs:  # don't leave processes running after an error
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    if not mapped:
        return _align_coords(None, n)
    mapped = pd.concat(mapped, ignore_index=True)
    # without -multiple, liftOver writes at most one line per input interval
    assert not mapped['id'].duplicated().any()
    return _align_coords(mapped.set_index('id'), n)


def _align_coords(mapped: Optional[pd.DataFrame], n: int) -> pd.DataFrame:
    """
    Align coordinates indexed by input position (0..n-1) to all n positions,
    with NaN for positions not in ``mapped``.
    """
    if mapped is None:
        mapped = pd.DataFrame({'chrom': [], 'start': [], 'end': []})
    out = mapped.reindex(range(n))
    # float64 so that missing coordinates are NaN
    return pd.DataFrame({'chrom': out['chrom'].astype(object),
                         'start': out['start'].astype('float64'),
                         'end': out['end'].astype('float64')})


def _add_columns(df: pd.DataFrame, coords: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add the columns of ``coords`` to ``df`` positionally, prefixed with ``prefix``."""
    names = [f'{prefix}_{c}' for c in coords.columns]
    clash = [c for c in names if c in df.columns]
    if clash:
        raise ValueError(f'intervals already have column(s) {clash}')
    for name, col in zip(names, coords.columns):
        df[name] = coords[col].to_numpy()
    return df


def _chain_files(chain_file, path: list, cache_dir, arg: str) -> list:
    """Chain files for each step along a path of assemblies, downloading those not given."""
    n_steps = len(path) - 1
    if chain_file is None:
        return [get_chain_file(a, b, cache_dir=cache_dir) for a, b in zip(path, path[1:])]
    if isinstance(chain_file, (str, os.PathLike)):
        chain_file = [chain_file]
    chain_file = list(chain_file)
    if len(chain_file) != n_steps:
        raise ValueError(f'{arg} must have one chain file for each of the {n_steps} '
                         f'step(s) {" -> ".join(path)}')
    return chain_file


def liftover(intervals: Union[pd.DataFrame, PathLike], from_assembly: str, to_assembly: str,
             reciprocal: bool = False, min_match: float = 0.95,
             assembly_over: Optional[Union[str, Sequence[str]]] = None,
             chain_file: Optional[Union[PathLike, Sequence[PathLike]]] = None,
             back_chain_file: Optional[Union[PathLike, Sequence[PathLike]]] = None,
             chrom_col: str = 'chrom', start_col: str = 'start', end_col: str = 'end',
             pos_col: Optional[str] = None, pos_base: int = 0,
             chrom_aliases: bool = True, n_jobs: Optional[int] = None,
             cache_dir: Optional[PathLike] = None) -> pd.DataFrame:
    """
    Lift intervals or positions from one assembly to another, and optionally back again.

    With ``reciprocal=True``, lifted intervals are lifted back to test that
    they map one to one: an interval is reciprocally mapped if it lifts to the
    other assembly and the lifted interval lifts back to exactly the original
    chromosome, start and end.

    Parameters
    ----------
    intervals :
        Data frame with chromosome, start and end columns (or chromosome and
        position columns, see ``pos_col``), or the path to a BED file.
        Interval coordinates are 0-based and half-open, as in BED.
    from_assembly :
        UCSC name of the assembly the intervals are in, e.g. ``'hg38'``.
    to_assembly :
        UCSC name of the assembly to lift to, e.g. ``'papAnu4'``.
    reciprocal :
        Also lift the lifted intervals back to ``from_assembly`` and test that
        they map to their original coordinates.
    min_match :
        Minimum fraction of an interval's bases that must remap
        (liftOver ``-minMatch``). Used in both directions. Has no effect on
        positions, which either map or do not.
    assembly_over :
        UCSC name of an intermediate assembly, for when UCSC has no chain file
        between ``from_assembly`` and ``to_assembly``. Intervals are then lifted
        from ``from_assembly`` to ``assembly_over`` and on to ``to_assembly``,
        and back the same way with ``reciprocal=True``. A list of names lifts
        through several assemblies in turn. Each step applies ``min_match``,
        so intervals that change in an intermediate step can be lost.
    chain_file :
        Chain file lifting ``from_assembly`` to ``to_assembly``, to use instead
        of downloading it from UCSC. With ``assembly_over``, a list of chain
        files, one for each step.
    back_chain_file :
        Chain file lifting ``to_assembly`` to ``from_assembly``, to use instead
        of downloading it from UCSC. With ``assembly_over``, a list of chain
        files, one for each step. Only used with ``reciprocal=True``.
    chrom_col, start_col, end_col :
        Names of the coordinate columns of a data frame. Ignored for BED files,
        which are read with standard BED column names.
    pos_col :
        Name of a column of single positions. If given, these positions are
        lifted instead of intervals, and ``start_col`` and ``end_col`` are
        ignored.
    pos_base :
        Coordinate base of the positions in ``pos_col``: 0 (as BED starts) or
        1 (as in VCF and GFF files). Lifted positions are reported in the same
        base.
    chrom_aliases :
        Translate chromosome names to the UCSC names used in chain files
        (e.g. ``'NC_000023.11'`` or ``'X'`` to ``'chrX'`` for hg38), using the
        UCSC chromosome alias table of ``from_assembly``. Names that are not in
        the table are used as given, with a warning.
    n_jobs :
        Number of liftOver processes to run in parallel. Defaults to all
        available cores (``SLURM_CPUS_PER_TASK`` inside a SLURM job, otherwise
        the cores this process may use), but no more than fit in memory: each
        process loads the whole chain file, which can take close to 1 GB.
        Memory is the SLURM job's allocation (``SLURM_MEM_PER_NODE`` or
        ``SLURM_MEM_PER_CPU``) or the system's available memory. A given
        ``n_jobs`` is used as is. Each process gets at least 10,000
        intervals, so small inputs use fewer processes. Intervals (or
        positions) that occur several times are lifted only once.
    cache_dir :
        Directory for downloaded chain and chromosome alias files. Defaults to
        ``~/.cache/geneinfo/chains``.

    Returns
    -------
    :
        A copy of the intervals, with the same index and row order. Added
        columns ``<to_assembly>_chrom``, ``<to_assembly>_start`` and
        ``<to_assembly>_end`` hold the interval lifted to the other assembly,
        with UCSC chromosome names, and the boolean column ``mapped`` is True
        if the interval lifted. With ``reciprocal=True``, the columns
        ``<from_assembly>_back_chrom``, ``<from_assembly>_back_start`` and
        ``<from_assembly>_back_end`` hold the lifted interval lifted back, and
        the boolean column ``reciprocal`` is True if it lifted back to exactly
        its original coordinates. With ``pos_col``, each pair of start and end
        columns is replaced by a single ``_pos`` column, e.g.
        ``<to_assembly>_pos``. Coordinates are NaN where a lift fails. An
        interval with ``mapped`` True and ``reciprocal`` False either did not
        lift back (its back coordinates are NaN) or lifted back to different
        coordinates.

    Examples
    --------
    ```python
    df = pd.DataFrame(dict(chrom=['chrX'], start=[1_000_000], end=[1_010_000]))
    liftover(df, 'hg38', 'papAnu4')
    lifted = liftover(df, 'hg38', 'papAnu4', reciprocal=True)
    lifted.loc[lifted.reciprocal]

    # no direct chain from T2T (hs1) to baboon: go through hg38
    liftover(df, 'hs1', 'papAnu4', assembly_over='hg38')

    snps = pd.DataFrame(dict(chrom=['chrX'], pos=[1_000_000]))
    liftover(snps, 'hg38', 'papAnu4', pos_col='pos', pos_base=1)
    ```
    """
    if from_assembly == to_assembly:
        raise ValueError('from_assembly and to_assembly must differ')
    if back_chain_file is not None and not reciprocal:
        raise ValueError('back_chain_file is only used with reciprocal=True')
    if pos_base not in (0, 1):
        raise ValueError('pos_base must be 0 or 1')
    auto_jobs = n_jobs is None
    if auto_jobs:
        n_jobs = _available_cores()
    elif n_jobs < 1:
        raise ValueError('n_jobs must be at least 1')
    df, chroms, starts, ends = _read_input(intervals, chrom_col, start_col, end_col,
                                           pos_col, pos_base)
    if chrom_aliases:
        chroms = _translate_chroms(chroms, from_assembly, cache_dir)
    for col in ('mapped', 'reciprocal') if reciprocal else ('mapped',):
        if col in df.columns:
            raise ValueError(f'intervals already have a {col!r} column')
    if assembly_over is None:
        assembly_over = []
    elif isinstance(assembly_over, str):
        assembly_over = [assembly_over]
    path = [from_assembly, *assembly_over, to_assembly]
    if len(set(path)) < len(path):
        raise ValueError(f'assemblies must all differ: {path}')
    chains = _chain_files(chain_file, path, cache_dir, 'chain_file')
    if reciprocal:
        back_chains = _chain_files(back_chain_file, path[::-1], cache_dir, 'back_chain_file')

    def output(coords):
        # BED intervals for output; positions are reported in pos_base
        if pos_col is None:
            return coords
        return pd.DataFrame({'chrom': coords['chrom'], 'pos': coords['start'] + pos_base})

    def lift(coords, chain_files):
        # lift through each chain in turn; rows that fail a step stay NaN
        for chain in chain_files:
            ok = coords['chrom'].notna().to_numpy()
            lifted = _lift_coords(coords['chrom'].to_numpy()[ok], coords['start'].to_numpy()[ok],
                                  coords['end'].to_numpy()[ok], chain, min_match, n_jobs, auto_jobs)
            lifted.index = np.flatnonzero(ok)  # positions in the input
            coords = _align_coords(lifted, len(coords))
        return coords

    fwd = lift(pd.DataFrame({'chrom': chroms, 'start': starts, 'end': ends}), chains)
    mapped = fwd['chrom'].notna().to_numpy()
    df = _add_columns(df, output(fwd), to_assembly)
    if not reciprocal:
        df['mapped'] = mapped
        return df

    back = lift(fwd, back_chains)
    same = (back['chrom'].notna().to_numpy()
            & (back['chrom'].to_numpy() == chroms)
            & (back['start'].to_numpy() == starts)
            & (back['end'].to_numpy() == ends))

    df = _add_columns(df, output(back), f'{from_assembly}_back')
    df['mapped'] = mapped
    df['reciprocal'] = same
    return df
