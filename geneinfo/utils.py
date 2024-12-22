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

def horizon(row, i, cut):
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
    
def horizonplot(df, y=None, ax=None,
                cut=None, # float, takes precedence over quantile_span
                quantile_span = None,
                start='start',
                beginzero=True, 
                colors = ['#CCE2DF', '#59A9A8', '#374E9B', 'midnightblue',
                          '#F2DE9A', '#DA8630', '#972428', 'darkred',
                          '#D3D3D3']):
                # colours = ['#314E9F', '#36AAA8', '#D7E2D4'] + ['midnightblue'] + \
                #           ['#F5DE90', '#F5DE90', '#A51023'] + ['darkred'] + ['whitesmoke']):
                # colors = sns.color_palette("Blues", 3) + ['midnightblue'] + \
                #           sns.color_palette("Reds", 3) + ['darkred'] + ['lightgrey']):

    """
    Horizon bar plot made allowing multiple chromosomes and multiple samples.
    """

    # set cut if not set
    if cut is None:
        cut = np.max([np.max(df[y]), np.max(-df[y])]) / 3
    elif quantile_span:
        cut=max(np.abs(np.nanquantile(df[col], quantile_span[0])), 
                np.abs(np.nanquantile(df[col], quantile_span[1]))) / 3,

    # make the data frame to plot
    row_iter = df.itertuples()
    col_iterators = zip(*(horizon(row, y, cut) for row in row_iter))
    col_names = ['yp1', 'yp2', 'yp3', 'yp4', 
                 'yn1', 'yn2', 'yn3', 'yn4', 'nan']

    df2 = (df[[y, start]]
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

            for col_name, color in zip(col_names, colors):
                plt.setp(g.fig.texts, text="") # hack to make y facet labels align...
                # map barplots to each facet
                ax.fill_between(
                    df2[start], 
                    df2[col_name], 
                    y2=0,
                    color=color,
                    linewidth=0,
                    capstyle='butt')

n = 1000

fig, ax = plt.subplots(1, 1, figsize=(10, 0.5))

df = pd.DataFrame({'chrom': ['chr1']*n,
                'start': list(range(1*n)), 
                 'pi': list(np.sin(np.linspace(-np.pi, 10*np.pi, 1*n)))                   
                  })
sample = n
Fs = ci.max_chrom_size
# f = 50
f = np.linspace(5, 50, sample)
x = np.linspace(0, Fs, sample)
y = np.sin(2* np.pi * f * x / Fs)
df['pi'] = y


horizonplot(df, y='pi', ax=ax)
ax.set_axis_off()