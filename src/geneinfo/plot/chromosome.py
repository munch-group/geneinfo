
import numpy as np
from collections import defaultdict
import pandas as pd
from pandas.api.types import is_object_dtype
from collections.abc import Callable, MutableSequence
from typing import Any, TypeVar, List, Tuple, Dict, Union, Iterable

from ..intervals import *
from ..utils import horizon
from ..coords import gene_coords

import math
from math import isclose, floor, log10
from operator import sub
import warnings
from itertools import chain
from statsmodels.nonparametric.smoothers_lowess import lowess
import re

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib import patches
from matplotlib.artist import Artist
import matplotlib.gridspec as gridspec
from matplotlib.transforms import (Bbox, TransformedBbox,
                                   blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector,
                                                   BboxConnectorPatch,
                                                   BboxPatch)
import seaborn as sns

from matplotlib.text import OffsetFrom
import textwrap

from ..utils import chrom_lengths, centromeres

from .genome import GenomeIdeogram, _get_chrom_info

class ChromIdeogram(GenomeIdeogram):
    """
    Child class of GenomeIdeogram for plotting single chromosome ideograms.
    """

    def __init__(self, chrom:str, assembly:str, axes_height_inches:int=2, axes_width_inches:int=12,
                 hspace:float=0.3, ylim:tuple=(0, 10), zooms:list=[], zoom_font_size:float=None,
                 zoom_height_ratio=1.0, zoom_effect_color:str='lightgray', zoom_effect_alpha:float=0.3,
                 wspace:float=0.1, font_size:float=4,
                ):
        """
        Initialize canvas for plotting an ideogram for a single chromosomes.

        Parameters
        ----------
        chrom : 
            Chromosome identifier (E.g. 'chr1', 'chrX').
        assembly : 
            Genome assembly as ucsc identifier (E.g. hg39, rheMac10).
        axes_height_inches : 
            Height of panel for each chromosome ideogram, by default 0.5
        axes_width_inches : 
            Width of panel for longest chromosome ideogram, by default 12
        hspace : 
            Space between additional axes, by default 0
        ylim : 
            Value range on y-axis for placing elements, by default (0, 10)
        font_size : 
            Font size as percentage of panel height (axes_height_inches), 
            by default 4
        zooms : 
            List of chromosome intervals (start, end) to show in seperate detail plots.
        zoom_height_ratio : 
            Height of zoom detail plots relative to height of main plot.
        zoom_font_size : 
            Height of zoom detail plots relative to height of main plot, by default same as font_size
        zoom_effect_color : 
            Face color of zoom effect patches, by default 'lightgray'
        zoom_effect_alpha: 
            Alpha of zoom effect patches, by default 0.3
        """
        # self.species = species
        self.assembly = assembly
        self.ideogram_base = None
        self.ideogram_height = None
        self.legend_handles = []
        self.zooms = zooms
        self.zoom_axes = []
        self.height_ratios = [1]        
        self.end_padding = 300000
        self.chr_names = [chrom]

        self.zoom_height_ratio = zoom_height_ratio
        self.zoom_effect_color = zoom_effect_color
        self.zoom_effect_alpha = zoom_effect_alpha

        # self.chr_sizes = [self.chrom_lengths[assembly][chrom] 
        #                   for chrom in self.chr_names]
        self.chr_names = [chrom]
        if self.assembly is not None and assembly in self._chrom_lengths:
            #self.chr_names = [f'chr{x}' for x in list(range(1, 23))+['X', 'Y']]
            self.chr_sizes = [self._chrom_lengths[assembly][c] for c in self.chr_names]
            self.centromeres = self._centromeres
            self.coord_system = {'hg38':'GRCh38', 
                                 'hg19':'GRCh37',
                                 'hg37':'GRCh37',
                                 }[assembly]
        else:
            self.chrom_lengths, self.centromeres = _get_chrom_info(self.assembly)
            self.chr_sizes = [self.chrom_lengths[chrom]]

        self.max_chrom_size = max(self.chr_sizes)
        self.aspect = axes_height_inches / axes_width_inches
        fig_height_inches = axes_height_inches
        point_size = 1/72
        self.ylim = ylim
        self.font_size = font_size/100 * axes_height_inches / point_size 

        if zooms:
            fig_height_inches += (1 + hspace) * axes_height_inches




        if zoom_font_size is None:
            self.zoom_font_size = self.font_size
        else:
            self.zoom_font_size = zoom_font_size/100 * axes_height_inches / point_size 

        fig_width_inches = axes_width_inches
        figsize = (fig_width_inches, fig_height_inches)

        with plt.rc_context(self.d):

            xlim = (-self.end_padding, self.chr_sizes[0]+self.end_padding)
            scaled_y_lim = xlim[0] * self.aspect, xlim[1] * self.aspect

            if zooms:
                self.fig =  plt.figure(figsize=figsize)
                axs = self.fig.subplot_mosaic([
                    ["main" for i in range(len(zooms))],
                    [f"zoom{i}" for i in range(len(zooms))]],
                     width_ratios=[-sub(*tup) for tup in zooms],
                     height_ratios=[1, self.zoom_height_ratio]
                )
                axs["main"].set(xlim=xlim)

                for i in range(len(zooms)):
                    self.zoom_effect(axs[f"zoom{i}"], axs["main"], alpha=self.zoom_effect_alpha, facecolor=self.zoom_effect_color)
                    axs[f"zoom{i}"].set(xlim=zooms[i])
                    axs[f"zoom{i}"].set_facecolor((0, 0, 0, 0))
                    # axs[f"zoom{i}"].set_xticks(np.arange(0, self.chr_sizes[0]+1, 10_000_000))
                    # axs[f"zoom{i}"].xaxis.tick_bottom()
                    xtl = axs[f"zoom{i}"].get_xticklabels()
                    xtl[-1] = ""
                    axs[f"zoom{i}"].get_xticklabels(xtl)

                    zoom_scaled_y_lim = self._scaled_y_lim(axs[f"zoom{i}"])
                    axs[f"zoom{i}"].set(ylim=zoom_scaled_y_lim)

                self.ax = axs["main"]
                self.zoom_axes = [axs[f"zoom{i}"] for i in range(len(zooms))]
                
                for i in range(len(self.zoom_axes)):
                    self.zoom_axes[i].sharey(self.zoom_axes[0])
                    self.zoom_axes[i].yaxis.set_ticks_position('none')
                    self.zoom_axes[i].set_yticklabels([])

                    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
                    # self.zoom_axes[i].xaxis.set_major_locator(MultipleLocator(1_000_000))
                    self.zoom_axes[i].xaxis.set_minor_locator(MultipleLocator(200_000))
                    x_ticks = self.zoom_axes[i].xaxis.get_major_ticks()
                    x_ticks[-1].label1.set_visible(False)

                plt.subplots_adjust(wspace=wspace)

            else:        
                self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
            
            if hspace is not None:
                plt.subplots_adjust(hspace=hspace)
    
            plt.minorticks_on()
        
            gs = matplotlib.gridspec.GridSpec(1, 1)
            gs.update(wspace=0, hspace=hspace) 

            self.ax.set_xlim(xlim)
            self.ax.set_ylim(scaled_y_lim)                

            self.ax.spines[['right', 'top', 'left']].set_visible(False)
            x = -self.end_padding * 10 / figsize[1]
            self.ax.set_yticklabels([])
            self.ax.set_xticks(np.arange(0, self.chr_sizes[0]+1, 10_000_000))
            self.ax.xaxis.tick_bottom()
            self.ax.spines['bottom'].set_visible(True)                    
            self.ax.yaxis.set_ticks_position('none')
            self.ax.yaxis.set_visible(False)
            self.ax.set_yticklabels([])

            self.ax_list = [self.ax]
        
            self.chr_axes = dict(zip(self.chr_names, self.ax_list))

    
    def connect_bbox(self, bbox1, bbox2,
                     loc1a, loc2a, loc1b, loc2b,
                     prop_lines, prop_patches=None):
        if prop_patches is None:
            prop_patches = {
                **prop_lines,
                "alpha": prop_lines.get("alpha", 1) * 0.2,
                "clip_on": False,
            }
    
        c1 = BboxConnector(
            bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
        c2 = BboxConnector(
            bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)
    
        bbox_patch1 = BboxPatch(bbox1, ec='none', fc='none')
        bbox_patch2 = BboxPatch(bbox2, **prop_patches)
    
        p = BboxConnectorPatch(bbox1, bbox2,
                               loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                               clip_on=False,
                               **prop_patches)
    
        return c1, c2, bbox_patch1, bbox_patch2, p
    
    
    def zoom_effect(self, ax1, ax2, **kwargs):
        """
        ax1 : the main Axes
        ax1 : the zoomed Axes
    
        Similar to zoom_effect01.  The xmin & xmax will be taken from the
        ax1.viewLim.
        """
    
        tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
        trans = blended_transform_factory(ax2.transData, tt)
    
        mybbox1 = ax1.bbox
        mybbox2 = TransformedBbox(ax1.viewLim, trans)
    
        defaults = {'edgecolor': 'black', 'facecolor': 'lightgray', 'alpha': 0.3, 'linewidth': 0.5}
        defaults.update(kwargs)

        # prop_patches = defaults
        # del kwargs['alpha']

        c1, c2, bbox_patch1, bbox_patch2, p = self.connect_bbox(
            mybbox1, mybbox2,
            loc1a=2, loc2a=3, loc1b=1, loc2b=4,
            prop_lines={}, 
            prop_patches=defaults)
    
        ax1.add_patch(bbox_patch1)
        ax2.add_patch(bbox_patch2)
        ax2.add_patch(c1)
        ax2.add_patch(c2)
        ax2.add_patch(p)
    
        return c1, c2, bbox_patch1, bbox_patch2, p


    def add_axes(self, nr_axes=1, height_ratio=1.0, hspace=None):

        new_axes = []
        for _ in range(nr_axes):
        
            gs = self.ax.get_gridspec()
            row = gs.nrows + 1 
            self.height_ratios.append(height_ratio)
            gs = gridspec.GridSpec(row, 1, height_ratios=self.height_ratios,
                                   hspace=hspace)
            for i, ax in enumerate(self.fig.axes):
                ax.set_position(gs[i].get_position(self.fig))
                ax.set_subplotspec(gs[i])
            new_ax = self.fig.add_subplot(gs[row-1], sharex=ax)
            self.fig.set_figheight(
                self.fig.get_figheight() * sum(self.height_ratios) \
                / sum(self.height_ratios[:-1]))
            new_ax.spines[['right', 'top']].set_visible(False)
            new_ax.xaxis.set_visible(False)
            new_axes.append(new_ax)

        new_axes[-1].xaxis.tick_bottom()
        new_axes[-1].xaxis.set_visible(True)

        if nr_axes == 1:
            return new_axes[0]
        else:
            return new_axes


