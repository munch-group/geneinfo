import os, glob, sys
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
from pathlib import Path
from collections import UserList
from statsmodels.nonparametric.smoothers_lowess import lowess

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


class GeneList(UserList):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: add alias mapping to GeneList
    def download_gene_aliases():
        """
        Download mapping from any alias to the cannonical hgcn name for use in set operations.
        """
        ...

    def _tabulate(self):
        """
        Turn list into square'ish matrix
        """
        n = len(self)
        col_width = max(map(len, self)) + 1
        ncols = min(max(100//col_width, 1+sqrt(n/col_width)), 150//col_width)
        # ncols = min(max(80//col_width, 1+sqrt(n/col_width)), 80//col_width)
        nrows = int(n/ncols) + 1
        rows = []
        for r in range(0, n, nrows):
            rows.append(self[r:r+nrows])
        return rows, col_width
        
    def __repr__(self):
        rows, col_width = self._tabulate()
        repr = []
        for row in list(zip_longest(*rows, fillvalue='')):
            line = []
            for gene in row:
                line.append(gene.ljust(col_width))
            repr.append(''.join(line))
        return('\n'.join(repr))

    def _repr_html_(self):
        rows, col_width = self._tabulate()
        style = 'style="background: transparent!important; line-height: 10px!important;text-align: left!important"'
        table = [f'<table data-quarto-disable-processing="true" {style}>']
        for row in list(zip_longest(*rows, fillvalue='')):
            table.append(f'<tr {style}>')
            for gene in row:
                if hasattr(self, '_highlight') and gene in self._highlight:
                    table.append(f'<td {style}><b>{gene}</b></td>')
                else:
                    table.append(f'<td {style}>{gene}</td>')
            table.append('</tr>')
        table.append('</table>')
        if hasattr(self, '_highlight'):
            delattr(self, '_highlight')        
        return '\n'.join(table)

    def expand_amplicon_abbrev(self):

        new_list = []
        for gene_name in self:
            abbrev = gene_name.rsplit('_', 1)[0]
            abbrev = abbrev.replace('-', '_')
            if abbrev in AMPL_ABBREV_MAP.keys():
                new_list.extend(AMPL_ABBREV_MAP[abbrev])
            else:
                new_list.append(gene_name)

        # new_list = []
        # for gene_name in old_list:
        #     if gene_name.startswith('amplicon') and '/' in gene_name:
        #         prefix, *variants = gene_name.split('/')
        #         first_amplicon = re.split(r'[_-]+', prefix, 2)[-1]
        #         new_list.append(first_amplicon)
        #         for var in variants:
        #             ampl_name = first_amplicon[:-1] + var
        #             new_list.append(ampl_name)
        #     else:
        #         new_list.append(gene_name)

        self.data = sorted(set(new_list))
    
    def __str__(self):
        return repr(self)

    def __lshift__(self, other):
        setattr(self, '_highlight', list(other))
        return self
        
    def __or__(self, other):
        return GeneList(sorted(set(self.data + other.data)))

    def __and__(self, other):
        return GeneList(sorted(set(self.data).intersection(set(other.data))))

    def __xor__(self, other):
        inter = set(self.data).intersection(set(other.data))
        union = set(self.data + other.data)
        return GeneList(sorted(union.difference(inter)))

AMPL_ABBREV_MAP = {    
 'amplicon_chrX_CPXCR1': ['CPXCR1'],
 'amplicon_chrX_CSAG1/2/3': ['CSAG1', 'CSAG2', 'CSAG3'],
 'amplicon_chrX_CT45A1/2/3//6/7/8/9/10': ['CT45A1', 'CT45A2', 'CT45A3', 'CT45A6', 'CT45A7', 'CT45A8', 'CT45A9', 'CT45A10'],
 'amplicon_chrX_CT47A1/2/3/4/5/6/7/8/9/10/11/12/B1': ['CT47A1', 'CT47A2', 'CT47A3', 'CT47A4', 'CT47A5', 'CT47A6', 'CT47A7', 'CT47A8', 'CT47A9', 'CT47A10', 'CT47A11', 'CT47A12', 'CT47B1'],
 'amplicon_chrX_CT55': ['CT55'],
 'amplicon_chrX_CT83': ['CT83'],
 'amplicon_chrX_CTAG1A/1B/2': ['CTAG1A', 'CTAG1B', 'CTAG2'],
 'amplicon_chrX_CXorf49/B': ['CXorf49', 'CXorf49B'],
 'amplicon_chrX_CXorf51A/B': ['CXorf51A', 'CXorf51B'],
 'amplicon_chrX_DDX53': ['DDX53'],
 'amplicon_chrX_DMRTC1/B/FAM236A/B/C/D': ['DMRTC1', 'DMRTC1B', 'FAM236A', 'FAM236B', 'FAM236C', 'FAM236D'],
 'amplicon_chrX_EOLA1/2/HSFX3/4': ['EOLA1', 'EOLA2', 'HSFX3', 'HSFX4'],
 'amplicon_chrX_ETD1/B/ZNF75D': ['ETD1', 'ETD1B', 'ZNF75D'],
 'amplicon_chrX_F8/F8A1/2/3/H2AB1/2/3': ['F8', 'F8A1', 'F8A2', 'F8A3', 'H2AB1', 'H2AB2', 'H2AB3'],
 'amplicon_chrX_FAM156A/B': ['FAM156A', 'FAM156B'],
 'amplicon_chrX_FAM47A/B/C': ['FAM47A', 'FAM47B', 'FAM47C'],
 'amplicon_chrX_G6PD/IKBKG': ['G6PD', 'IKBKG'],
 'amplicon_chrX_GAGE10/1/2A/13/12B/12C/12D/12E/12F/12G/12H/12J': ['GAGE10', 'GAGE1', 'GAGE2A', 'GAGE13', 'GAGE12B', 'GAGE12C', 'GAGE12D', 'GAGE12E', 'GAGE12F', 'GAGE12G', 'GAGE12H', 'GAGE12J'],
 'amplicon_chrX_HSFX1/2': ['HSFX1', 'HSFX2'],
 'amplicon_chrX_IL3RA/P2RY8/SLC25A6': ['IL3RA', 'P2RY8', 'SLC25A6'],
 'amplicon_chrX_MAGEA4': ['MAGEA4'],
 'amplicon_chrX_MAGEA12/A2/A2B/A3/A6': ['MAGEA12', 'MAGEA2', 'MAGEA2B', 'MAGEA3', 'MAGEA6'],
 'amplicon_chrX_MAGEA9/9B': ['MAGEA9', 'MAGEA9B'],
 'amplicon_chrX_MAGEB6': ['MAGEB6'],
 'amplicon_chrX_MAGEC1': ['MAGEC1'],
 'amplicon_chrX_MBTPS2/YY2': ['MBTPS2', 'YY2'],
 'amplicon_chrX_NSDHL': ['NSDHL'],
 'amplicon_chrX_NUDT10/11': ['NUDT10', 'NUDT11'],
 'amplicon_chrX_NXF2/2B/5': ['NXF2', 'NXF2B', 'NXF5'],
 'amplicon_chrX_PABPC1L2A/B': ['PABPC1L2A', 'PABPC1L2B'],
 'amplicon_chrX_PAGE2/2B/5': ['PAGE2', 'PAGE2B', 'PAGE5'],
 'amplicon_chrX_RHOXF2/B': ['RHOXF2', 'RHOXF2B'],
 'amplicon_chrX_SPACA5/B': ['SPACA5', 'SPACA5B'],
 'amplicon_chrX_SPANXA1/A2/N1/N2/N3/N4/N5/B1/C/D': ['SPANXA1', 'SPANXA2', 'SPANXN1', 'SPANXN2', 'SPANXN3', 'SPANXN4', 'SPANXN5', 'SPANXB1', 'SPANXC', 'SPANXD'],
 'amplicon_chrX_SSX1/2/2B/344B/5/7': ['SSX1', 'SSX2', 'SSX2B', 'SS344B', 'SSX5', 'SSX7'],
 'amplicon_chrX_SUPT20HL1/2': ['SUPT20HL1', 'SUPT20HL2'],
 'amplicon_chrX_TCEAL2/3/4/5/6': ['TCEAL2', 'TCEAL3', 'TCEAL4', 'TCEAL5', 'TCEAL6'],
 'amplicon_chrX_TCP11X1/2': ['TCP11X1', 'TCP11X2'],
 'amplicon_chrX_TEX28': ['TEX28'],
 'amplicon_chrX_TMEM185A': ['TMEM185A'],
 'amplicon_chrX_VCX/2/3A/3B': ['VCX', 'VCX2', 'VCX3A', 'VCX3B'],
 'amplicon_chrX_XAGE1A/B': ['XAGE1A', 'XAGE1B'],
 'amplicon_chrX_XAGE3': ['XAGE3'],
 'amplicon_chrX_XAGE5': ['XAGE5'],
}

class GeneListCollection(object):

    def __init__(self, url:str=None, google_sheet:str=None, tab='Sheet1'):

        assert url or google_sheet, 'Either file/url or google_sheet id must be provided.'

        if url is None:
            url = f'https://docs.google.com/spreadsheets/d/{google_sheet}/gviz/tq?tqx=out:csv&sheet={tab}'

        self.desc = []
        for desc in pd.read_csv(url, header=None, low_memory=False).iloc[0]:
            if str(desc) == 'nan':
                self.desc.append('')
            else:
                self.desc.append(desc.replace('\n', ' '))
        self.df = pd.read_csv(url, header=1, low_memory=False)
        self.df = self.df.loc[:, [not x.startswith('Unnamed') for x in self.df.columns]]
        self.names = self.df.columns.tolist()

    def all_genes(self):
        names = []
        for label in self.names:
            names.extend(self.get(label))
        return sorted(set(names))
    
    def get(self, name):
        sr = self.df[name]
        sr = self.df.loc[~sr.isnull(), name]
        # lst = sorted(self.expand_amplicon_abbrev(sr.tolist()))
        lst = sr.tolist()
        return GeneList(lst)

    def _repr_html_(self):
        out = ['| label | description |', '|:---|:---|']
        for name, desc in zip(self.names, self.desc):
            if pd.isnull(desc):
                desc = ''
            # out.append(f"- **{(name+':**').ljust(130)} {desc}")
            out.append(f"| **{name}** | {desc} |")
            
        display(Markdown('\n'.join(out)))

    def __repr__(self):
        return ""
  
    def __iter__(self):
         yield from self.names

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
