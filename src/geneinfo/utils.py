import os, glob, sys
import re
import pandas as pd
from pandas.api.types import is_object_dtype
import numpy as np
from math import isclose, floor, log10
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from pandas.api.types import is_object_dtype
from math import sqrt
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union
from itertools import zip_longest
from matplotlib.patches import Rectangle
import seaborn as sns
import requests
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union
import warnings
from itertools import chain
import shelve
from collections.abc import Sequence
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import fisher_exact

from .intervals import *


chrom_lengths = {'hg19': {'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 
                            'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 
                            'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895,
                            'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753, 
                            'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 
                            'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566},
                    'hg38': {'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555, 
                            'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636, 
                            'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309, 
                            'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345, 
                            'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167, 
                            'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415}}    

# TODO: make the centromeres fit each assembly!
centromeres = {
    'chr1':    (121700000, 125100000),
    'chr10':   (38000000, 41600000),
    'chr11':   (51000000, 55800000),
    'chr12':   (33200000, 37800000),
    'chr13':   (16500000, 18900000),
    'chr14':   (16100000, 18200000),
    'chr15':   (17500000, 20500000),
    'chr16':   (35300000, 38400000),
    'chr17':   (22700000, 27400000),
    'chr18':   (15400000, 21500000),
    'chr19':   (24200000, 28100000),
    'chr2':    (91800000, 96000000),
    'chr20':   (25700000, 30400000),
    'chr21':   (10900000, 13000000),
    'chr22':   (13700000, 17400000),
    'chr3':    (87800000, 94000000),
    'chr4':    (48200000, 51800000),
    'chr5':    (46100000, 51400000),
    'chr6':    (58500000, 62600000),
    'chr7':    (58100000, 62100000),
    'chr8':    (43200000, 47200000),
    'chr9':    (42200000, 45500000),
    'chrX':    (58100000, 63800000),
    'chrY':    (10300000, 10400000)}    


def _chrom_sort_key(chrom):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', chrom)]

def chrom_sort_key(chrom):
    """
    Function for use as key in sorting chromosomes. Works for both
    Python lists and numpy arrays/pandas series.
    """
    if isinstance(chrom, (list, tuple)):
        return chrom_sort_key(chrom[0])
    elif isinstance(chrom, Sequence):
        return [_chrom_sort_key(x) for x in chrom]
    else:
        return _chrom_sort_key(chrom)

def dummy_data():
    df_list = []
    for chrom, length in chrom_lengths['hg38'].items():
        sample = length // 500_000
        Fs = length
        f = np.linspace(5, 30, sample)
        x = np.linspace(0, Fs, sample)
        y = np.sin(2* np.pi * f * x / Fs)
        chrom = [chrom]*sample
        df = pd.DataFrame(dict(x=x, y=y, chrom=chrom)).sort_values('x')
        df['start'] = df.x.shift()
        df['end'] = df.x
        df = df.iloc[1:-1, :]
        df = stairs(df)
        df_list.append(df)
    df = pd.concat(df_list)
    return df


def dummy_segments():
    import random
    segments = []
    for chrom, chrom_len in chrom_lengths['hg38'].items():
        p = sorted([random.randint(0, chrom_len) for _ in range(int(chrom_len//10_000_000))])
        segments.extend([(chrom, *t) for t in zip(p[0::2], p[1::2])])
    return segments
    
use_cache = True
verbose_retrieval = False

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

class cache_disabled():
    def __enter__(self):
        global use_cache
        use_cache = False

    def __exit__(self, type, value, traceback):
        global disable_cache
        use_cache = True

def fisher_test(one, other, background, min_dist=None, return_counts=False):

    a, b = one, other    

    not_in_background = a ^ (a & background)
    if not_in_background:
        print(f'Removed {len(not_in_background)} genes not in background set', file=sys.stderr)
        print(not_in_background)
    if min_dist is not None:
        try:
            a = one._distance_prune(other, *min_dist)
        except AttributeError as e:
            print('Distance pruning only works for GeneList objects.')
            raise e
        
    M = len(background) 
    N = len(background & a) 
    n = len(background & b)
    x = len(background & a & b)
    table = [[  x,           n - x          ],
            [ N - x,        M - (n + N) + x]]
    if return_counts:
        return float(fisher_exact(table, alternative='greater').pvalue), table
    return float(fisher_exact(table, alternative='greater').pvalue)  

def shelve_it() -> Callable:
    """
    Decorator to cache the result of a function in a shelve file.

    Parameters
    ----------
    file_name : 
        Path to the shelve file
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    def decorator(func):
        if not use_cache:
            return func
        def new_func(*args, **kwargs):
            with shelve.open(os.path.join(CACHE_DIR, func.__name__)) as cache:
                key = '*'.join(map(str, args)) + '///' + '**'.join([f'{k}={v}' for k, v in kwargs.items()])
                if key not in cache:
                    if verbose_retrieval:
                        print(f' {func.__name__} retrieving ...', file=sys.stderr)
                    cache[key] = func(*args, **kwargs)
                return cache[key]
        return new_func

    return decorator


def clear_cache(func_name=None):
    """
    Clear the cache of a shelve file.
    """
    if not os.path.exists(CACHE_DIR):
        return
    if func_name is None:
        for file in glob.glob(f'{CACHE_DIR}/*.db'):
            os.remove(file)
    else:
        os.remove(f'{CACHE_DIR}/{func_name}.db')


def _horizon(row, i, cut):
    """
    Compute the values for the three 
    positive and negative intervals.
    """
    val = getattr(row, i)

    if np.isnan(val):
        for i in range(8):
            yield 0
        # for nan color
        yield cut
    else:
        if val < 0:
            for i in range(4):
                yield 0

        val = abs(val)
        for i in range(3):
            yield min(cut, val)
            val = max(0, val-cut)
        yield int(not isclose(val, 0, abs_tol=1e-8)) * cut

        if val >= 0:
            for i in range(4):
                yield 0

        # for nan color
        yield 0


def horizon(df, y=None, ax=None,
                cut=None, # float, takes precedence over quantile_span
                quantile_span = None,
                x='start',
                beginzero=True, 
                offset=0,
                height=None,
                palette='iker',
                colors = None,
                **kwargs):
                # colours = ['#314E9F', '#36AAA8', '#D7E2D4'] + ['midnightblue'] + \
                #           ['#F5DE90', '#F5DE90', '#A51023'] + ['darkred'] + ['whitesmoke']):
                # colors = sns.color_palette("Blues", 3) + ['midnightblue'] + \
                #           sns.color_palette("Reds", 3) + ['darkred'] + ['lightgrey']):

    """
    Horizon bar plot made allowing multiple chromosomes and multiple samples.
    """
    
    palettes = dict(iker = ['#CCE2DF', '#59A9A8', '#374E9B', 'midnightblue',
                            '#F2DE9A', '#DA8630', '#972428', 'darkred',
                             '#D3D3D3'],
                    bluered = sns.color_palette("Blues", 3) + ['midnightblue'] + \
                              sns.color_palette("Reds", 3) + ['darkred'] + \
                              ['lightgrey'],
                          )                          

    if colors is None:
        colors = palettes[palette.replace('_r', '')]

    if palette.endswith('_r'):
        colors = colors[3:6][::-1] + colors[0:3][::-1] + colors[6:]

    # set cut if not set
    if cut is None:
        cut = np.max([np.max(df[y]), np.max(-df[y])]) / 3
    elif quantile_span:
        cut=max(np.abs(np.nanquantile(df[col], quantile_span[0])), 
                np.abs(np.nanquantile(df[col], quantile_span[1]))) / 3,

    # make the data frame to plot
    row_iter = df.itertuples()
    col_iterators = zip(*(_horizon(row, y, cut) for row in row_iter))
    col_names = ['yp1', 'yp2', 'yp3', 'yp4', 
                 'yn1', 'yn2', 'yn3', 'yn4', 'nan']

    df2 = (df[[y, x]]
            .assign(**dict(zip(col_names, col_iterators)))
            )
    df2 = pd.DataFrame(dict((col, list(chain.from_iterable(zip(df2[col].values, df2[col].values)))) for col in df2))

    # make the plot
    with sns.axes_style("ticks"):

        # ingore UserWarning from seaborn that tight_layout is not applied
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # first y tick
            ytic1 = round(cut/3, -int(floor(log10(abs(cut/3)))))

            scale = 1
            if height is not None:
                ymax = max(df2[col_name].max() for col_name in col_names)
                ymin = max(df2[col_name].min() for col_name in col_names)
                scale = height/(ymax-ymin)

            for col_name, color in zip(col_names, colors):
                ax.fill_between(
                    df2[x], 
                    df2[col_name]*scale+offset, 
                    y2=offset,
                    color=color,
                    linewidth=0,
                    capstyle='butt',
                **kwargs)


def map_interval(chrom, start, end, strand, map_from, map_to, species='homo_sapiens'):
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    start, end = int(start), int(end)    
    api_url = f"http://rest.ensembl.org/map/{species}/{map_from}/{chrom}:{start}..{end}:{strand}/{map_to}"
    params = {'content-type': 'application/json'}
    response = requests.get(api_url, params=params)
    if not response.ok:
        response.raise_for_status()
    return response.json()


class nice:

    def __rlshift__(self, df):
        "Left align columns of params frame: df << nice()"

        def make_pretty(styler):

            def commas(v):
                if type(v) is int:
                    s = str(v)[::-1]
                    return ','.join([s[i:i+3] for i in range(0, len(s), 3)])[::-1]
                else:
                    return v

            return styler.format(commas)

        s = df.style.pipe(make_pretty)
        s.set_table_styles(
            {c: [{'selector': '', 'props': [('text-align', 'left')]}] 
                 for c in df.columns if is_object_dtype(df[c]) and c != 'strand'},
            overwrite=False
        )
        display(s)


def tabulate_genes(words, ncols=None):
    n = len(words)
    col_width = max(map(len, words)) + 1
    if ncols is None:
        ncols = max(100//col_width, 1+sqrt(n/col_width))
    nrows = int(n/ncols) + 1
    rows = []
    for r in range(0, n, nrows):
        rows.append(words[r:r+nrows])
    for row in list(zip_longest(*rows, fillvalue='')):
        line = []
        for gene in row:
            line.append(gene.ljust(col_width))
        print(''.join(line))

def fit_lowess(df, x='x', y='y', frac=0.005, it=0, **kwargs):
    return df.groupby('chrom').apply(lambda _df: _fit_lowess(_df, x=x, y=y, frac=frac, it=it, **kwargs)).reset_index()

def _fit_lowess(df, x='x', y='y', frac=0.005, it=0, **kwargs):
    sorted_df = df.sort_values(by=x)
    filtered = lowess(sorted_df[y], sorted_df[x], is_sorted=True, frac=frac, it=it, **kwargs)
    return pd.DataFrame({x: filtered[:,0], y: filtered[:,1]})

def abline(slope, intercept, ax=None):
    "Add a straight line through the plot"
    if ax is None:
        ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--', color='grey')

def add_band(x_low, x_high, y_low=None, y_high=None, ax=None, color='gray', linewidth=0, alpha=0.5, zorder=0, **kwargs):
    "Plot a gray block on x interval"
    if ax is None:
        ax = plt.gca()
    if y_low is None:
        y_low, _ = ax.get_ylim()
    if y_high is None:
        _, y_high = ax.get_ylim()
    g = ax.add_patch(Rectangle((x_low, y_low), x_high-x_low, y_high-y_low, 
                 facecolor=color,
                 linewidth=linewidth,
                 alpha=alpha,
                 zorder=zorder,
                 **kwargs))

def stairs(df, start='start', end='end', pos='pos', endtrim=0):
    "Turn a df with start, end into one with pos to plot as stairs"
    df1 = df.copy(deep=True)
    df2 = df.copy(deep=True)
    df1[pos] = df1[start]
    df2[pos] = df2[end] - endtrim
    return pd.concat([df1, df2]).sort_values([start, end])
