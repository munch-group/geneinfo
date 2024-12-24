import time
import warnings
from itertools import chain
# import gc

import numpy as np
import pandas as pd

# # Make inline plots vector graphics instead of raster graphics
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('pdf', 'svg')

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from math import isclose, floor, log10


from IPython.display import Markdown, display, Image, SVG, HTML
import numpy as np
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import sys
import os
import re
import json
import pickle
import subprocess
import pandas as pd
from pandas.api.types import is_object_dtype
from math import log10, sqrt
import shutil
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union
from itertools import zip_longest

import matplotlib.axes
from matplotlib.patches import Rectangle, Polygon
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina', 'png')

from goatools.base import download_go_basic_obo
#from goatools.base import download_ncbi_associations
from goatools.obo_parser import GODag, OBOReader
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot
from goatools.cli.ncbi_gene_results_to_python import ncbi_tsv_to_py
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.go_enrichment import GOEnrichmentRecord
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.go_search import GoSearch
from goatools.godag.go_tasks import CurNHigher
from goatools.godag_plot import plot_gos, plot_goid2goobj, plot_results, plt_goea_results
import requests
from Bio import Entrez

from .intervals import *

def map_interval(chrom, start, end, strand, map_from, map_to, species='homo_sapiens'):
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    start, end = int(start), int(end)    
    api_url = f"http://rest.ensembl.org/map/{species}/{map_from}/{chrom}:{start}..{end}:{strand}/{map_to}"
    params = {'content-type': 'application/json'}
    response = requests.get(api_url, params=params)
    if not response.ok:
        response.raise_for_status()
    #null = '' # json may include 'null' variables 
    return response.json()#eval(response.content.decode())


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


class GeneList(list):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        n = len(self)
        col_width = max(map(len, self)) + 1
        ncols = max(100//col_width, 1+sqrt(n/col_width))
        nrows = int(n/ncols) + 1
        rows = []
        for r in range(0, n, nrows):
            rows.append(self[r:r+nrows])
        repr = []
        for row in list(zip_longest(*rows, fillvalue='')):
            line = []
            for gene in row:
                line.append(gene.ljust(col_width))
            repr.append(''.join(line))
        return('\n'.join(repr))


class GoogleSheet(object):

    def __init__(self, SHEET_ID='1JSjSLuto3jqdEnnG7JqzeC_1pUZw76n7XueVAYrUOpk', SHEET_NAME='Sheet1'):
        url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
        self.desc = []
        for desc in pd.read_csv(url, header=None, low_memory=False).iloc[0]:
            if str(desc) == 'nan':
                self.desc.append('')
            else:
                self.desc.append(desc.replace('\n', ' '))
        self.df = pd.read_csv(url, header=1, low_memory=False)
        self.df = self.df.loc[:, [not x.startswith('Unnamed') for x in self.df.columns]]
        self.names = self.df.columns.tolist()

    def get(self, name):
        sr = self.df[name]
        return GeneList(sorted(sr[~sr.isnull()]))

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
  