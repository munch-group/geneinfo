import gzip
import os
import shutil
import warnings

import pandas as pd
import pytest

import requests

import geneinfo.lift
from geneinfo.lift import _chain_name, get_chain_file, liftover

pytestmark = pytest.mark.skipif(shutil.which('liftOver') is None,
                                reason='UCSC liftOver binary not on PATH')

# Forward chain, chrA -> chrB:
#   chrA[0,400)    -> chrB[100,500)
#   chrA[400,500)     deleted in chrB
#   chrA[500,1000) -> chrB[500,1000)
FORWARD = 'chain 1000 chrA 1200 + 0 1000 chrB 1100 + 100 1000 1\n400\t100\t0\n500\n\n'

# Backward chain, chrB -> chrA:
#   chrB[100,500) -> chrA[0,400)
#   chrB[500,800) -> chrA[600,900)   (not the inverse of the forward chain)
#   chrB[800,...)    not covered
BACKWARD = 'chain 1000 chrB 1100 + 100 800 chrA 1200 + 0 900 2\n400\t0\t200\n300\n\n'

# Chromosome alias table for the made-up assembly aaa1 (UCSC chromAlias format)
ALIASES = '# ucsc\tgenbank\trefseq\tassembly\nchrA\tCP000001.1\tNC_000001.1\tA\n'


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    """Default cache with an alias table for aaa1, so tests never download."""
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'aaa1.chromAlias.txt').write_text(ALIASES)
    monkeypatch.setattr(geneinfo.lift, 'DEFAULT_CACHE_DIR', cache)
    return cache


@pytest.fixture
def chains(tmp_path):
    fwd = tmp_path / 'aToB.over.chain.gz'
    with gzip.open(fwd, 'wt') as f:
        f.write(FORWARD)
    back = tmp_path / 'bToA.over.chain'
    back.write_text(BACKWARD)
    return fwd, back


@pytest.fixture
def intervals():
    return pd.DataFrame({
        'chrom': ['chrA', 'chrA', 'chrA', 'chrA', 'chrA'],
        'start': [10, 420, 600, 850, 20],
        'end': [50, 480, 700, 950, 20],
        'label': ['ok', 'deleted', 'mismatch', 'no_back', 'zero_length'],
    }, index=[10, 11, 12, 13, 14])


def test_chain_name():
    assert _chain_name('hg38', 'papAnu4') == 'hg38ToPapAnu4.over.chain.gz'
    assert _chain_name('papAnu4', 'hg38') == 'papAnu4ToHg38.over.chain.gz'


def test_get_chain_file_uses_cache(tmp_path):
    cached = tmp_path / _chain_name('aaa1', 'bbb2')
    cached.write_text('x')
    assert get_chain_file('aaa1', 'bbb2', cache_dir=tmp_path) == cached


def test_liftover(chains, intervals):
    fwd, _ = chains
    out = liftover(intervals, 'aaa1', 'bbb2', chain_file=fwd)
    assert list(out.index) == list(intervals.index)
    assert list(out.columns) == list(intervals.columns) + [
        'bbb2_chrom', 'bbb2_start', 'bbb2_end', 'mapped']
    assert out['mapped'].tolist() == [True, False, True, True, True]
    assert out.loc[10, ['bbb2_chrom', 'bbb2_start', 'bbb2_end']].tolist() == ['chrB', 110, 150]
    assert out.loc[11, ['bbb2_chrom', 'bbb2_start', 'bbb2_end']].isna().all()
    assert out.loc[14, ['bbb2_start', 'bbb2_end']].tolist() == [120, 120]
    assert out['bbb2_start'].dtype == 'float64' and out['bbb2_end'].dtype == 'float64'
    assert out.loc[11, 'bbb2_chrom'] is not pd.NA and out.loc[11, 'bbb2_start'] != out.loc[11, 'bbb2_start']
    # input is not modified
    assert 'bbb2_chrom' not in intervals.columns


def test_reciprocal(chains, intervals):
    fwd, back = chains
    out = liftover(intervals, 'aaa1', 'bbb2', reciprocal=True,
                   chain_file=fwd, back_chain_file=back)
    assert list(out.index) == list(intervals.index)
    assert list(out.columns) == list(intervals.columns) + [
        'bbb2_chrom', 'bbb2_start', 'bbb2_end',
        'aaa1_back_chrom', 'aaa1_back_start', 'aaa1_back_end',
        'mapped', 'reciprocal']
    # labels: ok, deleted, mismatch, no_back, zero_length
    assert out['mapped'].tolist() == [True, False, True, True, True]
    assert out['reciprocal'].tolist() == [True, False, False, False, True]
    for col in ('bbb2_start', 'bbb2_end', 'aaa1_back_start', 'aaa1_back_end'):
        assert out[col].dtype == 'float64'
    assert out['mapped'].dtype == bool and out['reciprocal'].dtype == bool
    # missing values are NaN, not pd.NA
    for value in out.loc[11, ['bbb2_chrom', 'bbb2_start', 'aaa1_back_chrom', 'aaa1_back_end']]:
        assert value is not pd.NA and pd.isna(value) and value != value
    assert out.loc[12, ['bbb2_start', 'aaa1_back_start', 'aaa1_back_end']].tolist() == [600, 700, 800]
    assert out.loc[13, ['aaa1_back_chrom', 'aaa1_back_start']].isna().all()
    assert out.loc[11, ['bbb2_chrom', 'aaa1_back_chrom']].isna().all()


def test_min_match(chains):
    fwd, back = chains
    df = pd.DataFrame({'chrom': ['chrA'], 'start': [350], 'end': [550]})  # half deleted
    strict = liftover(df, 'aaa1', 'bbb2', reciprocal=True,
                      chain_file=fwd, back_chain_file=back)
    assert strict[['mapped', 'reciprocal']].values.tolist() == [[False, False]]
    loose = liftover(df, 'aaa1', 'bbb2', chain_file=fwd, min_match=0.4)
    assert loose[['bbb2_start', 'bbb2_end']].iloc[0].tolist() == [450, 550]


def test_bed_input(chains, tmp_path):
    fwd, back = chains
    bed = tmp_path / 'in.bed'
    bed.write_text('track name=test\nchrA\t10\t50\tfirst\t0\t+\nchrA\t420\t480\tsecond\t0\t-\n')
    out = liftover(bed, 'aaa1', 'bbb2', reciprocal=True,
                   chain_file=fwd, back_chain_file=back)
    assert out[['chrom', 'start', 'end', 'name', 'score', 'strand']].values.tolist() == [
        ['chrA', 10, 50, 'first', 0, '+'], ['chrA', 420, 480, 'second', 0, '-']]
    assert out['mapped'].tolist() == [True, False]
    assert out['reciprocal'].tolist() == [True, False]


def test_custom_columns(chains):
    fwd, back = chains
    df = pd.DataFrame({'seqname': ['chrA'], 'from': [10], 'to': [50]})
    out = liftover(df, 'aaa1', 'bbb2', reciprocal=True,
                   chain_file=fwd, back_chain_file=back,
                   chrom_col='seqname', start_col='from', end_col='to')
    assert out['reciprocal'].tolist() == [True]


def test_unknown_chromosome_and_empty(chains):
    fwd, back = chains
    df = pd.DataFrame({'chrom': ['chrZ'], 'start': [1], 'end': [5]})
    with pytest.warns(UserWarning, match='not known in aaa1.*chrZ'):
        out = liftover(df, 'aaa1', 'bbb2', reciprocal=True,
                       chain_file=fwd, back_chain_file=back)
    assert out[['mapped', 'reciprocal']].values.tolist() == [[False, False]]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        empty = liftover(df.iloc[:0], 'aaa1', 'bbb2', reciprocal=True,
                         chain_file=fwd, back_chain_file=back)
    assert len(empty) == 0 and {'mapped', 'reciprocal'} <= set(empty.columns)


@pytest.mark.parametrize('bad, message', [
    (pd.DataFrame({'chrom': ['chrA'], 'start': [50], 'end': [10]}), 'start <= end'),
    (pd.DataFrame({'chrom': ['chrA'], 'start': [None], 'end': [10]}), 'missing'),
    (pd.DataFrame({'chrom': ['chrA'], 'start': [1.5], 'end': [10]}), 'non-integer'),
    (pd.DataFrame({'chr': ['chrA'], 'start': [1], 'end': [10]}), 'missing column'),
    (pd.DataFrame({'chrom': ['chrA'], 'start': [1], 'end': [10], 'mapped': ['x']}), 'mapped'),
])
def test_validation(chains, bad, message):
    fwd, back = chains
    with pytest.raises(ValueError, match=message):
        liftover(bad, 'aaa1', 'bbb2', reciprocal=True,
                 chain_file=fwd, back_chain_file=back)


def test_back_chain_requires_reciprocal(chains, intervals):
    fwd, back = chains
    with pytest.raises(ValueError, match='reciprocal=True'):
        liftover(intervals, 'aaa1', 'bbb2', chain_file=fwd, back_chain_file=back)


def test_reciprocal_column_allowed_one_way(chains):
    fwd, _ = chains
    df = pd.DataFrame({'chrom': ['chrA'], 'start': [10], 'end': [50], 'reciprocal': ['x']})
    out = liftover(df, 'aaa1', 'bbb2', chain_file=fwd)
    assert out['mapped'].tolist() == [True]


# Reverse chain, chrA -> chrC on the minus strand:
#   chrA[0,100) -> chrC[400,500) reversed, so 0-based base x maps to 499 - x
REVERSE = 'chain 1000 chrA 1200 + 0 100 chrC 500 - 0 100 3\n100\n\n'


def test_positions(chains):
    fwd, back = chains
    snps = pd.DataFrame({'chrom': ['chrA', 'chrA', 'chrA', 'chrA'],
                         'pos': [10, 450, 600, 850],
                         'id': ['ok', 'deleted', 'mismatch', 'no_back']})
    out = liftover(snps, 'aaa1', 'bbb2', pos_col='pos', chain_file=fwd)
    assert list(out.columns) == ['chrom', 'pos', 'id', 'bbb2_chrom', 'bbb2_pos', 'mapped']
    assert out['bbb2_pos'].tolist()[:1] == [110] and pd.isna(out.loc[1, 'bbb2_pos'])
    assert out['mapped'].tolist() == [True, False, True, True]

    out = liftover(snps, 'aaa1', 'bbb2', pos_col='pos', reciprocal=True,
                   chain_file=fwd, back_chain_file=back)
    assert list(out.columns) == ['chrom', 'pos', 'id', 'bbb2_chrom', 'bbb2_pos',
                                 'aaa1_back_chrom', 'aaa1_back_pos', 'mapped', 'reciprocal']
    assert out['reciprocal'].tolist() == [True, False, False, False]
    assert out.loc[2, ['bbb2_pos', 'aaa1_back_pos']].tolist() == [600, 700]
    assert out['bbb2_pos'].dtype == 'float64'


def test_positions_one_based(chains):
    fwd, back = chains
    snps = pd.DataFrame({'chrom': ['chrA', 'chrA'], 'pos': [1, 11]})
    out = liftover(snps, 'aaa1', 'bbb2', pos_col='pos', pos_base=1, reciprocal=True,
                   chain_file=fwd, back_chain_file=back)
    assert out['bbb2_pos'].tolist() == [101, 111]
    assert out['aaa1_back_pos'].tolist() == [1, 11]
    assert out['reciprocal'].all()


def test_positions_reverse_strand(tmp_path):
    chain = tmp_path / 'aToC.over.chain'
    chain.write_text(REVERSE)
    snps = pd.DataFrame({'chrom': ['chrA'], 'pos': [10]})
    zero = liftover(snps, 'aaa1', 'ccc3', pos_col='pos', chain_file=chain)
    assert zero['ccc3_pos'].tolist() == [489]    # 499 - 10
    one = liftover(snps, 'aaa1', 'ccc3', pos_col='pos', pos_base=1, chain_file=chain)
    assert one['ccc3_pos'].tolist() == [491]     # 1-based base 10 is 0-based 9 -> 490 -> 491


def test_positions_ignore_interval_columns(chains):
    fwd, _ = chains
    # start/end columns are neither required nor used
    snps = pd.DataFrame({'chr': ['chrA'], 'position': [10], 'start': ['x']})
    out = liftover(snps, 'aaa1', 'bbb2', chrom_col='chr', pos_col='position', chain_file=fwd)
    assert out['bbb2_pos'].tolist() == [110]


@pytest.mark.parametrize('df, kwargs, message', [
    (pd.DataFrame({'chrom': ['chrA'], 'pos': [10]}), dict(pos_base=2), 'pos_base'),
    (pd.DataFrame({'chrom': ['chrA'], 'pos': [0]}), dict(pos_base=1), '>= 1'),
    (pd.DataFrame({'chrom': ['chrA'], 'pos': [-1]}), dict(), '>= 0'),
    (pd.DataFrame({'chrom': ['chrA'], 'pos': [None]}), dict(), 'missing'),
    (pd.DataFrame({'chrom': ['chrA']}), dict(), 'missing column'),
])
def test_position_validation(chains, df, kwargs, message):
    fwd, _ = chains
    with pytest.raises(ValueError, match=message):
        liftover(df, 'aaa1', 'bbb2', pos_col='pos', chain_file=fwd, **kwargs)


def test_chrom_aliases(chains):
    fwd, back = chains
    df = pd.DataFrame({'chrom': ['NC_000001.1', 'A', 'CP000001.1', 'chrA'],
                       'start': [10, 10, 600, 10], 'end': [50, 50, 700, 50]})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = liftover(df, 'aaa1', 'bbb2', reciprocal=True,
                       chain_file=fwd, back_chain_file=back)
    assert out['chrom'].tolist() == df['chrom'].tolist()  # input names unchanged
    assert out['bbb2_chrom'].tolist() == ['chrB'] * 4
    assert out['aaa1_back_chrom'].tolist() == ['chrA'] * 4
    assert out['reciprocal'].tolist() == [True, True, False, True]


def test_chrom_aliases_positions(chains):
    fwd, _ = chains
    snps = pd.DataFrame({'chrom': ['NC_000001.1'], 'pos': [11]})
    out = liftover(snps, 'aaa1', 'bbb2', pos_col='pos', pos_base=1, chain_file=fwd)
    assert out[['bbb2_chrom', 'bbb2_pos']].values.tolist() == [['chrB', 111]]


def test_chrom_aliases_off(chains):
    fwd, _ = chains
    df = pd.DataFrame({'chrom': ['NC_000001.1'], 'start': [10], 'end': [50]})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = liftover(df, 'aaa1', 'bbb2', chain_file=fwd, chrom_aliases=False)
    assert out['mapped'].tolist() == [False]


def test_chrom_aliases_unavailable(chains, monkeypatch):
    fwd, _ = chains

    def offline(url, path):
        raise requests.ConnectionError('offline')

    monkeypatch.setattr(geneinfo.lift, '_download', offline)
    df = pd.DataFrame({'chrom': ['chrA'], 'start': [10], 'end': [50]})
    with pytest.warns(UserWarning, match='alias table for zzz9'):
        out = liftover(df, 'zzz9', 'bbb2', chain_file=fwd)
    assert out['mapped'].tolist() == [True]  # names used as given

    monkeypatch.setattr(geneinfo.lift, '_download', lambda url, path: False)  # 404
    with pytest.warns(UserWarning, match="no chromosome alias table for 'zzz9'"):
        liftover(df, 'zzz9', 'bbb2', chain_file=fwd)


def test_duplicates_lifted_once(chains, monkeypatch):
    fwd, back = chains
    calls = []
    run = geneinfo.lift._run_liftover

    def spy(bed, *args):
        calls.append(len(bed))
        return run(bed, *args)

    monkeypatch.setattr(geneinfo.lift, '_run_liftover', spy)
    df = pd.DataFrame({'chrom': ['chrA', 'chrA', 'chrA', 'chrA', 'chrA', 'chrA'],
                       'start': [10, 600, 10, 420, 600, 10],
                       'end': [50, 700, 50, 480, 700, 50]})
    out = liftover(df, 'aaa1', 'bbb2', reciprocal=True, chain_file=fwd, back_chain_file=back)
    assert calls == [3, 2]  # unique intervals forward; unique mapped intervals back
    assert out['bbb2_start'].tolist()[:3] == [110, 600, 110] and pd.isna(out.loc[3, 'bbb2_start'])
    assert out['reciprocal'].tolist() == [True, False, True, False, False, True]
    # same interval on another chromosome is not a duplicate
    df = pd.DataFrame({'chrom': ['chrA', 'chrZ'], 'start': [10, 10], 'end': [50, 50]})
    with pytest.warns(UserWarning, match='chrZ'):
        out = liftover(df, 'aaa1', 'bbb2', chain_file=fwd)
    assert out['mapped'].tolist() == [True, False]


def test_parallel_matches_serial(chains, intervals, monkeypatch):
    fwd, back = chains
    serial = liftover(intervals, 'aaa1', 'bbb2', reciprocal=True, n_jobs=1,
                      chain_file=fwd, back_chain_file=back)
    monkeypatch.setattr(geneinfo.lift, 'MIN_CHUNK_SIZE', 1)
    n_procs = []
    popen = geneinfo.lift.subprocess.Popen

    def counting_popen(*args, **kwargs):
        n_procs.append(1)
        return popen(*args, **kwargs)

    monkeypatch.setattr(geneinfo.lift.subprocess, 'Popen', counting_popen)
    parallel = liftover(intervals, 'aaa1', 'bbb2', reciprocal=True, n_jobs=3,
                        chain_file=fwd, back_chain_file=back)
    assert len(n_procs) == 6  # 3 chunks in each direction
    pd.testing.assert_frame_equal(serial, parallel)


def test_available_cores(monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '7')
    assert geneinfo.lift._available_cores() == 7
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', 'garbage')
    assert geneinfo.lift._available_cores() == os.process_cpu_count()
    monkeypatch.delenv('SLURM_CPUS_PER_TASK')
    assert geneinfo.lift._available_cores() == os.process_cpu_count()


def test_n_jobs_validation(chains, intervals):
    fwd, _ = chains
    with pytest.raises(ValueError, match='n_jobs'):
        liftover(intervals, 'aaa1', 'bbb2', chain_file=fwd, n_jobs=0)


def test_available_memory_slurm(monkeypatch):
    monkeypatch.delenv('SLURM_MEM_PER_NODE', raising=False)
    monkeypatch.setenv('SLURM_MEM_PER_CPU', '1000')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    assert geneinfo.lift._available_memory() == 4000 * 2**20
    monkeypatch.setenv('SLURM_MEM_PER_NODE', '8000')
    assert geneinfo.lift._available_memory() == 8000 * 2**20


def test_default_jobs_limited_by_memory(chains, intervals, monkeypatch):
    fwd, back = chains
    monkeypatch.setattr(geneinfo.lift, 'MIN_CHUNK_SIZE', 1)
    monkeypatch.setattr(geneinfo.lift, '_available_cores', lambda: 4)
    n_procs = []
    popen = geneinfo.lift.subprocess.Popen

    def counting_popen(*args, **kwargs):
        n_procs.append(1)
        return popen(*args, **kwargs)

    monkeypatch.setattr(geneinfo.lift.subprocess, 'Popen', counting_popen)
    per_process = 15 * fwd.stat().st_size + 300 * 2   # 5 intervals in 4 chunks of <= 2
    monkeypatch.setattr(geneinfo.lift, '_available_memory', lambda: int(2.5 * per_process / 0.8))
    liftover(intervals, 'aaa1', 'bbb2', chain_file=fwd)
    assert len(n_procs) == 2      # 4 cores, but memory for 2 processes
    n_procs.clear()
    liftover(intervals, 'aaa1', 'bbb2', chain_file=fwd, n_jobs=4)
    assert len(n_procs) == 4      # explicit n_jobs is not capped
    n_procs.clear()
    monkeypatch.setattr(geneinfo.lift, '_available_memory', lambda: None)
    liftover(intervals, 'aaa1', 'bbb2', chain_file=fwd)
    assert len(n_procs) == 4      # unknown memory: all cores


# Chains between chrB (bbb2) and chrC (ccc3): a plain offset of 1000
B_TO_C = 'chain 1000 chrB 1100 + 0 1100 chrC 2200 + 1000 2100 4\n1100\n\n'
C_TO_B = 'chain 1000 chrC 2200 + 1000 2100 chrB 1100 + 0 1100 5\n1100\n\n'


@pytest.fixture
def cached_chains(cache_dir):
    """UCSC-named chain files in the default cache: aaa1 <-> bbb2 <-> ccc3."""
    for (a, b), text in {('aaa1', 'bbb2'): FORWARD, ('bbb2', 'aaa1'): BACKWARD,
                         ('bbb2', 'ccc3'): B_TO_C, ('ccc3', 'bbb2'): C_TO_B}.items():
        with gzip.open(cache_dir / _chain_name(a, b), 'wt') as f:
            f.write(text)
    return cache_dir


def test_assembly_over(cached_chains, intervals):
    out = liftover(intervals, 'aaa1', 'ccc3', assembly_over='bbb2', reciprocal=True)
    assert list(out.columns) == list(intervals.columns) + [
        'ccc3_chrom', 'ccc3_start', 'ccc3_end',
        'aaa1_back_chrom', 'aaa1_back_start', 'aaa1_back_end', 'mapped', 'reciprocal']
    # labels: ok, deleted, mismatch, no_back, zero_length
    assert out.loc[10, ['ccc3_chrom', 'ccc3_start', 'ccc3_end']].tolist() == ['chrC', 1110, 1150]
    assert out['mapped'].tolist() == [True, False, True, True, True]
    assert out['reciprocal'].tolist() == [True, False, False, False, True]
    assert out.loc[12, ['ccc3_start', 'aaa1_back_start']].tolist() == [1600, 700]
    # a list of one intermediate assembly is the same
    same = liftover(intervals, 'aaa1', 'ccc3', assembly_over=['bbb2'], reciprocal=True)
    pd.testing.assert_frame_equal(out, same)


def test_assembly_over_positions(cached_chains):
    snps = pd.DataFrame({'chrom': ['NC_000001.1'], 'pos': [11]})
    out = liftover(snps, 'aaa1', 'ccc3', assembly_over='bbb2', pos_col='pos', pos_base=1,
                   reciprocal=True)
    assert out[['ccc3_chrom', 'ccc3_pos', 'aaa1_back_pos', 'reciprocal']].values.tolist() == [
        ['chrC', 1111, 11, True]]


def test_assembly_over_explicit_chains(chains, tmp_path, intervals):
    fwd, back = chains
    b_to_c, c_to_b = tmp_path / 'bToC.chain', tmp_path / 'cToB.chain'
    b_to_c.write_text(B_TO_C)
    c_to_b.write_text(C_TO_B)
    out = liftover(intervals, 'aaa1', 'ccc3', assembly_over='bbb2', reciprocal=True,
                   chain_file=[fwd, b_to_c], back_chain_file=[c_to_b, back])
    assert out['reciprocal'].tolist() == [True, False, False, False, True]
    with pytest.raises(ValueError, match='one chain file for each of the 2 step'):
        liftover(intervals, 'aaa1', 'ccc3', assembly_over='bbb2', chain_file=fwd)


def test_assembly_over_validation(cached_chains, intervals):
    with pytest.raises(ValueError, match='must all differ'):
        liftover(intervals, 'aaa1', 'ccc3', assembly_over='aaa1')
