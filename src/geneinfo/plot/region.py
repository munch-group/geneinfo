

from collections import defaultdict
from math import log10
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

import matplotlib.axes
import matplotlib.pyplot as plt

from ..intervals import *
from ..coords import gene_coords_region


def _plot_gene(name, txstart, txend, exons, offset, line_width, 
               min_visible_width, font_size, ax, highlight=False, clip_on=True):

    color='black'

    line = ax.plot([txstart, txend], [offset, offset], color=color, 
                   linewidth=line_width/5, alpha=0.5)
    line[0].set_solid_capstyle('butt')
    for start, end in exons:
        end = max(start+min_visible_width, end)
        line = ax.plot([start, end], [offset, offset], linewidth=line_width, 
                       color=color)
        line[0].set_solid_capstyle('butt')
        
    if highlight is True:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', 
                verticalalignment='center', fontsize=font_size, clip_on=clip_on,
                weight='bold', color='red')
    elif type(highlight) is dict:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', 
                verticalalignment='center',
            fontsize=font_size, clip_on=clip_on, 
            **highlight)
    else:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', 
                verticalalignment='center', fontsize=font_size, color=color, 
                clip_on=clip_on)


def gene_plot(chrom:str, start:str, end:str, assembly:str, highlight:List[Dict]=[], 
              hard_limits:bool=False, exact_exons:bool=False, figsize:tuple=None, 
              aspect:float=1, despine:bool=False, clip_on:bool=True, 
              gene_density:float=60, font_size:int=None, return_axes:int=1
              ) -> Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]:
    """
    Plots gene ideograms for a chromosomal region and returns axes for 
    plotting along the same chromosome coordinates.

    Parameters
    ----------
    chrom : 
        Chromosome identifier
    start : 
        Start of region
    end : 
        End of region (end base not included)
    assembly : 
        Genome assembly identifier
    highlight : 
        List or dictionary of genes to highlight on gene plot (see Examples), 
        by default []
    hard_limits : 
        Whether to truncate plot in the middle of a gene, by default False so 
        that genes are fully plotted.
    exact_exons : 
        Whether to plot exon coordinates exactly, by default False so that exons 
        are plotted as a minimum width.
    figsize : 
        Figure size specified as a (width, height) tuple, by default None 
        honoring the default matplotlib settings.
    aspect : 
        Size of gene plot height relative to the total height of the other axes, 
        by default 1
    despine : 
        Whether to remove top and right frame borders, by default False
    clip_on : 
        Argument passed to axes.Text, by default True
    gene_density : 
        Controls the density of gene ideograms in the plot, by default 60
    font_size : 
        Gene label font size, by default None, in which case it is calculated 
        based on the region size.
    return_axes : 
        The number of vertically stacked axes to return for plotting over the 
        gene plot, by default 1

    Returns
    -------
    :
        A single axes or a list of axes for plotting data over the gene plot.

    Examples
    --------
    ```python
    import geneinfo.information as gi
    # Set email for Entrez queries
    gi.email('your@email.com')

    # Highlight a single gene
    ax = gene_plot('chr1', 1000000, 2000000, 'hg38', highlight='TP53')
    ax.scatter(chrom_coordinates, values)

    # Highlight multiple genes
    ax = gene_plot('chr1', 1000000, 2000000, 'hg38', highlight=['TP53', 'BRCA1'])
    ax.scatter(chrom_coordinates, values)

    # Highlight genes with custom styles
    ax = gene_plot('chr1', 1000000, 2000000, 'hg38', 
                    highlight={'TP53': {'color': 'blue', 'weight': 'bold'}})
    ax.scatter(chrom_coordinates, values)

    # Muli-gene highlight with custom styles
    ax = gene_plot('chr1', 1000000, 2000000, 'hg38', 
                    highlight={'TP53': {'color': 'blue', 'weight': 'bold'}, 
                               'BRCA1': {'color': 'red'}})
    ax.scatter(chrom_coordinates, values)

    # Multiple axes for plotting over gene plot
    axes = gene_plot('chr1', 1000000, 2000000, 'hg38', return_axes=2)
    ax1, ax2 = axes
    ax1.scatter(chrom_coordinates, values1)
    ax2.scatter(chrom_coordinates, values2)

    # Custom figure size and aspect ratio
    axes = gene_plot('chr1', 1000000, 2000000, 'hg38', figsize=(10, 4), aspect=0.5)
    ax1, ax2 = axes
    ax1.scatter(chrom_coordinates, values1)
    ax2.scatter(chrom_coordinates, values2)
    ```

    """

    fig, axes = plt.subplots(return_axes+1, 1, figsize=figsize, sharex='col', 
                             sharey='row', 
                             gridspec_kw={'height_ratios': 
                                          [1/return_axes]*return_axes + [aspect]})
    plt.subplots_adjust(wspace=0, hspace=0.15)

    genes = gene_coords_region(chrom, start, end, assembly)

    for i, (name, chrom, txstart, txend, transcripts) in enumerate(genes):
        if len(transcripts) > 1:
            exons = [exon for transcript in transcripts for exon in transcript]
            exons = collapse(sorted(exons))
        else:
            exons = transcripts[0]
        genes[i] = (name, chrom, txstart, txend, exons)

    line_width = max(6, int(50 / log10(end - start)))-2
    if font_size is None:
        font_size = max(6, int(50 / log10(end - start)))
    label_width = font_size * (end - start) / gene_density
    if exact_exons:
        min_visible_exon_width = 0
    else:
        min_visible_exon_width = (end - start) / 1000

    plotted_intervals = defaultdict(list)
    for name, chrom, txstart, txend, exons in genes:
        # exons = [e for t in exons for e in t]
        gene_interval = [txstart-label_width, txend]
        max_gene_rows = 1000
        for offset in range(1, max_gene_rows, 1):
            if not intersect([gene_interval], plotted_intervals[offset]) and \
                not intersect([gene_interval], plotted_intervals[offset-1]) and \
                not intersect([gene_interval], plotted_intervals[offset+1]) and \
                not intersect([gene_interval], plotted_intervals[offset-2]) and \
                not intersect([gene_interval], plotted_intervals[offset+2]) and \
                not intersect([gene_interval], plotted_intervals[offset-3]) and \
                not intersect([gene_interval], plotted_intervals[offset+3]):
                break
        if plotted_intervals[offset]:
            plotted_intervals[offset] = union(plotted_intervals[offset], 
                                              [gene_interval])
        else:
            plotted_intervals[offset] = [gene_interval]

        if type(highlight) is list or type(highlight) is set:
            hl = name in highlight
        elif type(highlight) is dict or type(highlight) is defaultdict:
            hl = highlight[name]
        else:
            hl = None

        _plot_gene(name, txstart, txend, exons, 
                  offset, line_width, min_visible_exon_width, font_size, 
                  highlight=hl,
                  ax=axes[-1], clip_on=clip_on)

    if plotted_intervals:
        offset = max(plotted_intervals.keys())
    else:
        offset = 1

    if hard_limits:
        axes[-1].set_xlim(start, end)
    else:
        s, e = axes[-1].get_xlim()
        axes[-1].set_xlim(min(s-label_width/2, start), max(e, end))

    axes[-1].set_ylim(-2, offset+2)
    axes[-1].get_yaxis().set_visible(False)
    axes[-1].invert_yaxis()
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['left'].set_visible(False)

    # TODO: add optional title explaining gene name styles

    for ax in axes[:-1]:
        ax.set_xlim(axes[-1].get_xlim())

        if despine:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    if return_axes == 1:
        return axes[0]
    else:
        return axes[:-1]

