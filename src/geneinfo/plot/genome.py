
import numpy as np
from collections import defaultdict
import pandas as pd
from pandas.api.types import is_object_dtype
from collections.abc import Callable, MutableSequence
from typing import Any, TypeVar, List, Tuple, Dict, Union, Iterable
from itertools import cycle

from ..intervals import *
from ..utils import horizon
from ..coords import gene_coords, chromosome_lengths, centromere_coords

import math
from math import isclose, floor, log10
from operator import sub
import warnings
from itertools import chain
from statsmodels.nonparametric.smoothers_lowess import lowess
import requests, re

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
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

# from ..utils import chrom_lengths, centromeres

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Polygon:
    def __init__(self, points):
        self.points = points

    def get_points(self):
        return self.points

    def nudge_y(self, nudge):
        for point in self.points:
            point.y += nudge

def _get_chrom_info(assembly):
    lengths = {c: l for c, l in chromosome_lengths(assembly)}
    centromeres = {c: (s, e) for c, s, e in centromere_coords(assembly)}
    return lengths, centromeres


# def _get_chrom_info(species):

#     name = species.lower().replace(' ', '_')

#     server = "https://rest.ensembl.org"
#     # ext = "/info/assembly/homo_sapiens?bands=1?"
#     # ext = "/info/assembly/papio_anubis?bands=1?"
#     ext = f"/info/assembly/{name}?bands=1?"
    
#     r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    
#     if not r.ok:
#         r.raise_for_status()
    
#     data = r.json()
#     assembly_versions = data['coord_system_versions']
#     centromeres = {}
#     chrom_lengths = {}
    
#     assembly_coord_system = data['default_coord_system_version']
#     karyotype = data['karyotype']
#     assembly_accession = data['assembly_accession']
        
#     for d in data['top_level_region']:
#         if d['coord_system'] not in ['chromosome', 'primary_assembly']:
#             continue
#         if d['name'] not in karyotype:# .startswith('Un') or d['length'] < 1_000_000:
#             continue
#         if d['name'] == 'MT':# .startswith('Un') or d['length'] < 1_000_000:
#             continue
#         chrom_lengths['chr'+d['name']] = d['length']
    
        
#         if 'bands' in d:
#             for b in d['bands']:
#                 if b['stain'] == 'acen':
#                     centromeres['chr'+b['seq_region_name']] = (b['start'],  b['end'])


#     return assembly_coord_system, chrom_lengths, centromeres




class GenomeIdeogram:
    """
    Class to plot ideograms of chromosomes in a genome assembly.
    """

    d = {'axes.linewidth': 0.8, 'grid.linewidth': 0.64, 
         'lines.linewidth': 0.96, 'lines.markersize': 3.84, 
         'patch.linewidth': 0.64, 'xtick.major.width': 0.8,
         'ytick.major.width': 0.8, 'xtick.minor.width': 0.64, 
         'ytick.minor.width': 0.64, 'xtick.major.size': 3.84, 
         'ytick.major.size': 3.84, 'xtick.minor.size': 2.56, 
         'ytick.minor.size': 2.56, 'font.size': 7.68, 
         'axes.labelsize': 7.68, 'axes.titlesize': 7.68, 
         'xtick.labelsize': 7.04, 'ytick.labelsize': 7.04, 
         'legend.fontsize': 7.04, 'legend.title_fontsize': 7.68}

    _chrom_lengths = {'hg19': {'chr1': 249250621, 'chr2': 243199373, 
                              'chr3': 198022430, 'chr4': 191154276, 
                              'chr5': 180915260, 'chr6': 171115067, 
                              'chr7': 159138663, 'chr8': 146364022, 
                              'chr9': 141213431, 'chr10': 135534747, 
                              'chr11': 135006516, 'chr12': 133851895,
                              'chr13': 115169878, 'chr14': 107349540, 
                              'chr15': 102531392, 'chr16': 90354753, 
                              'chr17': 81195210, 'chr18': 78077248, 
                              'chr19': 59128983, 'chr20': 63025520, 
                              'chr21': 48129895, 'chr22': 51304566, 
                              'chrX': 155270560, 'chrY': 59373566},
                     'hg38': {'chr1': 248956422, 'chr2': 242193529, 
                              'chr3': 198295559, 'chr4': 190214555, 
                              'chr5': 181538259, 'chr6': 170805979, 
                              'chr7': 159345973, 'chr8': 145138636, 
                              'chr9': 138394717, 'chr10': 133797422, 
                              'chr11': 135086622, 'chr12': 133275309, 
                              'chr13': 114364328, 'chr14': 107043718, 
                              'chr15': 101991189, 'chr16': 90338345, 
                              'chr17': 83257441, 'chr18': 80373285, 
                              'chr19': 58617616, 'chr20': 64444167, 
                              'chr21': 46709983, 'chr22': 50818468, 
                              'chrX': 156040895, 'chrY': 57227415}}    

    # TODO: make the centromeres fit each assembly!
    _centromeres = {
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

    
    def __init__(self, assembly:str, axes_height_inches:float=0.5, axes_width_inches:float=12, 
                 hspace:float=0, ylim:tuple=(0, 10), font_size:float=10, 
                #  species:str='Homo sapiens', 
                 ):
        """
        Initialize canvas for plotting ideograms of chromosomes in a genome
        assembly.

        Parameters
        ----------
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
        # species : 
        #     Species latin name, by default panel 'Homo sapiens'.
        """
        
        # self.species = species
        self.assembly = assembly
        self.ideogram_base = None
        self.ideogram_height = None
        # self.min_stick_height = min_stick_height
        self.legend_handles = []
        self.zooms = []
        self.zoom_axes = []
        self.zoom_height_ratio = 1.0
        self.zoom_effect_color = 'lightgray'
        self.zoom_effect_alpha = 0.3
        self.end_padding = 300000

        # if self.assembly is not None and assembly in self._chrom_lengths:
        #     self.chrom_lengths = self._chrom_lengths[assembly]
        #     self.chr_names = [f'chr{x}' for x in list(range(1, 23))+['X', 'Y']]
        #     self.chr_sizes = [self.chrom_lengths[c] for c in self.chr_names]
        #     self.centromeres = self._centromeres
        # else:
        #     self.coord_system, self.chrom_lengths, self.centromeres = _get_chrom_info(self.species)
        #     def sort_key(name):
        #         _, num, alp = re.match(r'(chr)?(\d*)(.*)', name).groups()
        #         if not num:
        #             num = '999'
        #         return int(num), alp
                
        #     self.chr_names = sorted(self.chrom_lengths.keys(), key=sort_key)
        #     self.chr_sizes = [self.chrom_lengths[c] for c in self.chr_names]

        self.chrom_lengths, self.centromeres = _get_chrom_info(self.assembly)
        def sort_key(name):
            _, num, alp = re.match(r'(chr)?(\d*)(.*)', name).groups()
            if not num:
                num = '999'
            return int(num), alp
            
        self.chr_names = sorted(self.chrom_lengths.keys(), key=sort_key)
        self.chr_sizes = [self.chrom_lengths[c] for c in self.chr_names]

        # self.chr_names = [f'chr{x}' for x in list(range(1, 23))+['X', 'Y']]
        # self.chr_sizes = [self.chrom_lengths[assembly][chrom] 
        #                   for chrom in self.chr_names]
        self.max_chrom_size = max(self.chr_sizes)
        nr_rows = len(self.chr_names) - 1
        self.aspect = axes_height_inches / axes_width_inches
        axes_width = self.max_chrom_size + 2 * self.end_padding
        axes_height = self.aspect * axes_width
        fig_height_inches = axes_height_inches * (nr_rows-1) + hspace * (nr_rows-1)
        fig_width_inches = axes_width_inches
        figsize = (fig_width_inches, fig_height_inches)
        point_size = 1/72
        self.ylim = ylim
        self.font_size = font_size/100 * axes_height_inches / point_size

        self.zoom_font_size = self.font_size

        self.fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0)
        
        with plt.rc_context(self.d):
        
            nr_rows, nr_cols = len(self.chr_names)-2+1, 2
            gs = matplotlib.gridspec.GridSpec(
                nr_rows, 19, height_ratios=[0.4]+[1]*(nr_rows-1))
            gs.update(wspace=0, hspace=hspace) 

            dummy_ax = plt.subplot(gs[0, :])
            xlim = (-self.end_padding, self.max_chrom_size+self.end_padding)
            dummy_ax.set_xlim(xlim)
            scaled_y_lim = xlim[0] * self.aspect, xlim[1] * self.aspect
            # dummy_ax.set_ylim(scaled_y_lim)
            # dummy_ax.plot(1, 1)

            dummy_ax.spines['bottom'].set_visible(False)
            dummy_ax.spines['left'].set_visible(False)
            dummy_ax.spines['right'].set_visible(False)
            dummy_ax.spines['top'].set_visible(True)
            dummy_ax.xaxis.tick_top()
            dummy_ax.xaxis.set_label_position('top') 
            dummy_ax.yaxis.set_ticks_position('none')

            dummy_ax.set_yticklabels([])
        
            ax_list = [plt.subplot(gs[i, :]) for i in range(1, nr_rows-2)]

            ax_list.append(plt.subplot(gs[nr_rows-2, :12]))
            ax_list.append(plt.subplot(gs[nr_rows-1, :12]))
            ax_list.append(plt.subplot(gs[nr_rows-2, 7:]))
            ax_list.append(plt.subplot(gs[nr_rows-1, 7:]))

            self.ax_list = ax_list
            self.chr_axes = dict(zip(self.chr_names, self.ax_list))

            for i in range(len(self.ax_list)):
                chrom = self.chr_names[i]
                ax = ax_list[i]

                if i < len(self.ax_list) - 4:
                    xlim = (-self.end_padding, self.max_chrom_size+self.end_padding)
                    scaled_y_lim = xlim[0] * self.aspect, xlim[1] * self.aspect
                    ax.set_xlim(xlim)
                    ax.set_ylim(scaled_y_lim)
                elif i < len(self.ax_list) - 2:
                    xlim = (-self.end_padding, 12/19*self.max_chrom_size+self.end_padding)
                    scaled_y_lim = xlim[0] * self.aspect, xlim[1] * self.aspect * 19/12
                    ax.set_xlim(xlim)
                    ax.set_ylim(scaled_y_lim) 
                else:
                    xlim = (-self.end_padding, self.chrom_lengths['chrX']+self.end_padding)
                    scaled_y_lim = xlim[0] * self.aspect, xlim[1] * self.aspect * 19/12
                    # \
                    #     * (self.max_chrom_size+2*self.end_padding) \
                    #     / (self.chrom_lengths['hg38']['chrX']+2*self.end_padding)
                    ax.set_xlim(xlim)
                    ax.set_ylim(scaled_y_lim) 
                start, end = 0, self.chr_sizes[i]
                ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

                if i in [20, 21]:   
                    x = -2000000 * 10 / figsize[1]
                else:
                    x = -2000000 * 10 / figsize[1]

                ax.text(x, 
                        self.map_y(-sub(*self.ylim), ax)/2, 
                        chrom.replace('chr', ''), 
                        fontsize=7, 
                        verticalalignment='center', 
                        horizontalalignment='right', 
                        weight='bold')
                ax.set_yticklabels([])

                if i == len(self.ax_list)-1:
                    ax.xaxis.tick_bottom()
                    ax.spines['bottom'].set_visible(True)                    
                    ax.yaxis.set_ticks_position('none')
                    ax.yaxis.set_visible(False)
                    ax.set_yticklabels([])
                else:
                    ax.set_xticklabels([])
                    ax.xaxis.set_ticks_position('none')
                    ax.yaxis.set_ticks_position('none')


    def draw_chromosomes(self, base:float=4, height:float=2, 
                         facecolor:str='#EBEAEA', edgecolor:str='black', 
                         linewidth:float=0.7, **kwargs:dict) -> None:
        """
        Draws chromosome ideograms.

        Parameters
        ----------
        base : 
            Placement of ideogram lower edge on y-axis scale, by default 4
        height : 
            Height of ideogram on y-axis scale, by default 2
        facecolor : 
            Ideogram fill color, by default '#EBEAEA'
        edgecolor : 
            Ideogram edge color, by default 'black'
        linewidth : 
            Ideogram edge width, by default 0.7
        **kwargs :
            Additional keyword arguments for matplotlib.patches.Rectangle
        """
        self.ideogram_base = base
        self.ideogram_height = height
        
        with plt.rc_context(self.d):

            for i in range(len(self.ax_list)):
                chrom = self.chr_names[i]
                ax = self.ax_list[i]
                start, end = 0, self.chr_sizes[i]                

                ideogram_base = self.map_y(base, ax)
                ideogram_height = self.map_y(height, ax)
                # draw centromere
                ymin, ymax = ax.get_ylim()
                if chrom in self.centromeres:
                    cent_start, cent_end = self.centromeres[chrom]
                    xy = [[cent_start, ideogram_base], 
                        [cent_start, ideogram_base+ideogram_height], 
                        [cent_end, ideogram_base], 
                        [cent_end, ideogram_base+ideogram_height]]
                    g = ax.add_patch(patches.Polygon(xy, closed=True, zorder=2, 
                                                    fill=True, color='#777777'))
                else:
                    cent_start, cent_end = 0, 0 

                # draw chrom
                g = ax.add_patch(patches.Rectangle((start, ideogram_base), 
                                                   cent_start-start, 
                                                   ideogram_height, 
                                           facecolor=facecolor,
                                           edgecolor='none',
                                           zorder=0,
                                           linewidth=linewidth,
                                           **kwargs
                                          ))
                g = ax.add_patch(patches.Rectangle((start, ideogram_base), 
                                                   cent_start-start, 
                                                   ideogram_height, 
                                           facecolor='none',
                                           edgecolor=edgecolor,
                                           zorder=0.5,
                                           linewidth=linewidth,
                                           **kwargs
                                          )) 
                g = ax.add_patch(patches.Rectangle((cent_end, ideogram_base), 
                                                   end-cent_end, 
                                                   ideogram_height, 
                                           facecolor=facecolor,
                                           edgecolor='none',
                                           zorder=0,
                                           linewidth=linewidth,
                                           **kwargs
                                          ))
                g = ax.add_patch(patches.Rectangle((cent_end, ideogram_base), 
                                                   end-cent_end, 
                                                   ideogram_height, 
                                           facecolor='none',
                                           edgecolor=edgecolor,
                                           zorder=0.5,
                                           linewidth=linewidth,
                                           **kwargs
                                          ))                   


                for ax in self.zoom_axes:

                    ideogram_base = self.map_y(base, ax)
                    ideogram_height = self.map_y(height, ax)
                
                    # draw centromere
                    ymin, ymax = ax.get_ylim()
                    if chrom in self.centromeres:
                        cent_start, cent_end = self.centromeres[chrom]
                    else:
                        cent_start, cent_end = 0, 0 

                    xy = [[cent_start, ideogram_base], 
                        [cent_start, ideogram_base+ideogram_height], 
                        [cent_end, ideogram_base], 
                        [cent_end, ideogram_base+ideogram_height]]

                    g = ax.add_patch(patches.Polygon(xy, closed=True, zorder=-1, 
                                                     fill=True, color='#777777'))
                    
                    # draw chrom
                    g = ax.add_patch(patches.Rectangle((start, ideogram_base),
                                                       cent_start-start,
                                                       ideogram_height, 
                                               facecolor=facecolor,
                                               edgecolor='none',
                                               zorder=-1,
                                               linewidth=linewidth,
                                               **kwargs
                                              ))
                    g = ax.add_patch(patches.Rectangle((start, ideogram_base), 
                                                       cent_start-start, 
                                                       ideogram_height, 
                                               facecolor='none',
                                               edgecolor=edgecolor,
                                               zorder=0.5,
                                               linewidth=linewidth,
                                               **kwargs
                                              )) 
                    g = ax.add_patch(patches.Rectangle((cent_end, ideogram_base), 
                                                       end-cent_end, 
                                                       ideogram_height, 
                                               facecolor=facecolor,
                                               edgecolor='none',
                                               zorder=0,
                                               linewidth=linewidth,
                                               **kwargs
                                              ))
                    g = ax.add_patch(patches.Rectangle((cent_end, ideogram_base), 
                                                       end-cent_end, 
                                                       ideogram_height, 
                                               # fill=False,
                                               facecolor='none',
                                               edgecolor=edgecolor,
                                               zorder=0.5,
                                               linewidth=linewidth,
                                               **kwargs
                                              ))               
        
    
    def _is_polygons_intersecting(self, a:Polygon, b:Polygon) -> bool:
        """
        Tests if two polygons intersect.

        Parameters
        ----------
        a : 
            A polygon
        b : 
            Another polygon

        Returns
        -------
        :
            True if polygons intersect, False otherwise
        """
        for x in range(2):
            polygon = a if x == 0 else b
    
            for i1 in range(len(polygon.get_points())):
                i2 = (i1 + 1) % len(polygon.get_points())
                p1 = polygon.get_points()[i1]
                p2 = polygon.get_points()[i2]
    
                normal = Point(p2.y - p1.y, p1.x - p2.x)
    
                min_a = float('inf')
                max_a = float('-inf')
    
                for p in a.get_points():
                    projected = normal.x * p.x + normal.y * p.y
                    min_a = min(min_a, projected)
                    max_a = max(max_a, projected)
    
                min_b = float('inf')
                max_b = float('-inf')
    
                for p in b.get_points():
                    projected = normal.x * p.x + normal.y * p.y
                    min_b = min(min_b, projected)
                    max_b = max(max_b, projected)
    
                if max_a < min_b or max_b < min_a:
                    return False
    
        return True

    
    def _scaled_y_lim(self, ax:matplotlib.axes.Axes, zoom=False) -> tuple:
        """
        Returns y-axis limits in plotting coordinates.

        Parameters
        ----------
        ax : 
            Matplotlib axes object

        Returns
        -------
        :
            Y-axis limits in plotting coordinates as tuple
        """
        # xlim = ax.get_xlim() - self.end_padding

        # bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        # aspect = bbox.height / bbox.width
        
        # return 0, -sub(*xlim) * aspect 

        if zoom:
            x_span = -sub(*ax.get_xlim()) #* (-sub(*ax.get_xlim()) - 2*self.end_padding) / -sub(*ax.get_xlim())  #+ int(zoom) * 2 * self.end_padding
        else:
            x_span = -sub(*ax.get_xlim())

        bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        aspect = bbox.height / bbox.width
        
        return 0, x_span * aspect 

        # return xlim[0] * self.aspect, xlim[1] * self.aspect
    

    def map_y(self, y:float, ax:matplotlib.axes.Axes, bottom:float=0, 
              top:float=1) -> float:
        """
        Maps y-axis values from user-specified ylim to actual plotting coordinates.

        Parameters
        ----------
        y : 
            Y-axis value
        ax : 
            Matplotlib axes object

        Returns
        -------
        :
            y-axis value in plotting coordinates
        """
        miny, maxy = ax.get_ylim()

        return (y * (top - bottom) * (maxy - miny) 
                / (self.ylim[1] - self.ylim[0]) + bottom * (maxy - miny))
    # zero = -miny
    # if y >= 0:
    #     return y * (top - max(bottom, zero)) * (maxy - zero) \
    #         / (self.ylim[1] - self.ylim[0]) + max(bottom, zero) * (maxy - zero)
    # else:
    #     return y * (min(top, zero) - bottom) * (zero - miny) \
    #         / (self.ylim[1] - self.ylim[0]) + min(top, zero) * (zero - miny)

        
    def draw_text(self, coord:float, x_pos:float, y_pos:float, text:str, 
                  textcolor:str, fontsize:float, linecolor:str, 
                  ax:matplotlib.axes.Axes=None, y_line_bottom:float=0, 
                  highlight:dict=None, **kwargs:dict) -> None:
        """
        Draws text with a line pointing to a position on the y-axis.

        Parameters
        ----------
        coord : 
            Coordonate of annotation
        x_pos : 
            X-axis position of text
        y_pos : 
            Y-axis position of text
        text : 
            Text to display
        textcolor : 
            Text color, by default 'black'
        linecolor : 
            Line color, by default 'lightgray'
        ax : 
            Matplotlib axes, by default None
        y_line_bottom : 
            y coordinate for bottom of line, by default 0
        highlight : 
            Dictionary for styling text labels, by default None
        """

        y_unit = -sub(*self._scaled_y_lim(ax)) / -sub(*self.ylim)

        if 'color' in kwargs:
            textcolor = kwargs['color']
            linecolor = kwargs['color']
            del kwargs['color']

        kwargs.setdefault('zorder', 10)
        kwargs.setdefault('linewidth', 0.5)

        if highlight:
            text_props = highlight
        else:
            text_props = dict(color=textcolor,
                    # 'normal' | 'small-caps':
                    #variant = 'small-caps', 
                    # 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight':
                    fontweight='normal', 
                    # 'normal' | 'italic' | 'oblique'
                    fontstyle = 'normal', 
                    bbox=dict(boxstyle='square,pad=0', 
                              linewidth=0.2,
                              fc='none', 
                              alpha=1,
                              ec='none'),)

        t = ax.text(x_pos, y_pos, text, fontsize=fontsize,                     
                    rotation=45, zorder=10, 
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    **text_props)


        ax.plot((coord, coord, x_pos+y_unit/5),
                (y_line_bottom, y_pos, y_pos+y_unit/5),
                solid_capstyle='butt', 
                solid_joinstyle='miter',
                color=linecolor, 
                **kwargs)
    
    
    def get_polygon(self, text:str, x_pos:int, y_pos:float,
                    ax:matplotlib.axes.Axes, pad=0) -> matplotlib.patches.Polygon:
        """
        Computes polygon for rotated text label.

        Parameters
        ----------
        text : 
            Text to display
        x_pos : 
            X-axis position
        y_pos : 
            Y-axis position
        ax : 
            Matplotlib axes
        pad : 
            Text padding, by default 0

        Returns
        -------
        :
            Maplotlib polygon object
        """

        y_unit = -sub(*self._scaled_y_lim(ax)) / -sub(*self.ylim)
        # y_unit = -sub(*self.scaled_y_lim(ax)) / -sub(*ax.get_ylim())
        # y_pos = y_pos * y_unit

        y_pos = self.map_y(y_pos, ax)
        
        t = ax.text(x_pos, y_pos, text, fontsize=self.zoom_font_size,
                    horizontalalignment='left',
                    verticalalignment='bottom', 
                    rotation=0, zorder=3, 
                    bbox=dict(boxstyle='square', 
                              fc='none', ec='red', pad=pad, alpha=0.4))
        
        transf = ax.transData.inverted()
        bb = t.get_window_extent(renderer = self.fig.canvas.get_renderer())
        bbt = bb.transformed(transf)
        coords = bbt.get_points()
        
        Artist.remove(t)
        
        (x0, y0), (x1, y1) = coords
        pad = (y1 - y0)/5
        coords = np.array([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
            
        x, y = zip(*coords)
        df = pd.DataFrame(dict(x=x, y=y))
        df['y'] -= y_pos
        df['x'] -= x_pos
        dfx = df['x']*math.cos(math.radians(45))-df['y']*math.sin(math.radians(45))
        df['y'] = (df['x'] * math.sin(math.radians(45)) + df['y'] 
                   * math.cos(math.radians(45)))
        df['x'] = dfx
        df['y'] += y_pos
        df['x'] += x_pos

        _x = np.sort(df.x.to_numpy())
        df['x'] += _x[1] - _x[0]
        _y = np.sort(df.y.to_numpy())
        df['y'] -= (_y[1] - _y[0])/2

        coords = np.array(list(zip(df.x, df.y)))
        
        x_pos, y_pos = df['x'][0] - (df['x'][0] - df['x'][1]), df['y'][0]
    
        df['x'] = df.x + np.array([-pad, -pad, pad, pad])
        df['y'] = df.y + np.array([-pad, pad, pad, -pad])
        
        return x_pos, y_pos, Polygon([Point(x, y) for x, y in zip(df.x, df.y)])
        

    # def add_labels(self, data, labels='name', chrom='chrom', x='pos'):
    def add_labels(self, annot:MutableSequence, 
                   base:float=None, min_height:float=None, 
                   zoom_base:float=None, zoom_min_height:float=None, 
                   bold:MutableSequence=[], 
                   italic:MutableSequence=[], colored:MutableSequence=[], 
                   framed:MutableSequence=[], filled:MutableSequence=[], 
                   highlight_color='deeppink',
                   pad:float=0,  **kwargs:dict) -> None:
        """
        Add text labels to the chromosome ideograms. 

        Parameters
        ----------
        annot : 
            List of gene names or tuples of HGCN gene symbols and tuples with
            chromosome name, gene position, text label, and optionally text color,
            text size, and line color: `(<chrom>, <position>, <text>, [<textcolor>, 
            [ [textsize] [<linecolor>] ] ])`. Text size values are scaled to make 
            the largest one 1 and font size is then computed as self.font_size * 
            textsize. That way font sizes can only be smaller than the default.
        base : 
            Y coordinate for lower end of vertical line, by default None. 
            If None, the upper edge of ideogram is used.
        min_height : 
            Minimum length of vertical line in y coordinates, by default None. 
            If None, half the ideogram height is used.
        bold : 
            List of genes to highlight with bold text, by default []
        italic : 
            List of genes to highlight with italic text, by default []
        colored : 
            List of genes to highlight with color, by default []
        framed : 
            List of genes to highlight with framed label, by default []
        filled : 
            List of genes to highlight with filled label, by default []
        highlight_color : 
            Color used for highlighted genes, by default 'deeppink'
        pad : 
            Text padding, by default 0
        zoom_base : 
            Y coordinate for lower end of vertical line on zoom axes, by default None. 
            If None, the default value of font_size.
        zoom_min_height : 
            Minimum length of vertical line in y coordinate on zoom axes, by default None. 
            If None, the default is the value of min_height..
        """
        
        if base is None:
            base = self.ideogram_base + self.ideogram_height 

        if min_height is None:
            min_height = self.ideogram_height * 0.5

        if zoom_base is None:
            zoom_base = base

        if zoom_min_height is None:
            zoom_min_height = min_height

        if type(annot[0]) is str:
            _annot = []
            # annot is a list of gene names, not a list of tuples
            for gene_name, (chrom, start, end, strand) in gene_coord(
                annot, assembly=self.assembly).items():
                _annot.append((chrom, (start + end)/2, gene_name))
            annot = _annot

        _annot = []
        for a in annot:
            if len(a) == 3:
                a = a + ('black', 1.0, 'lightgray')
            elif len(a) == 4:
                a = a + (1.0, 'lightgray')
            elif len(a) == 5:
                a = a + ('lightgray',)
            elif len(a) == 6:
                pass
            else:
                raise ValueError('Invalid number of elements in annotation tuple')
            _annot.append(a)
        annot = _annot

        max_textsize = max([t[4] for t in annot])
        annot = [(a[0], a[1], a[2], a[3], a[4]/max_textsize, a[5]) for a in annot]

        y0 = base
        y1 = base + min_height

        zoom_y0 = zoom_base
        zoom_y1 = zoom_base + zoom_min_height


        hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(highlight_color))
        hsv[1] *= 0.15
        highlight_fill = matplotlib.colors.hsv_to_rgb(hsv)

        highlight = defaultdict(dict)
        for gene in bold:
            highlight[gene].update(dict(weight='bold'))
        for gene in italic:
            highlight[gene].update(dict(style='italic'))
        for gene in colored:
            highlight[gene].update(dict(color=highlight_color))
        for gene in framed:
            if 'bbox' not in highlight[gene]:
                highlight[gene]['bbox'] = {}            
            highlight[gene]['bbox'].update(dict(edgecolor='black', 
                                                pad=max(1.5, pad), 
                                                linewidth=0.5))
        for gene in filled:
            if 'bbox' not in highlight[gene]:
                highlight[gene]['bbox'] = {}
            highlight[gene]['bbox'].update(dict(facecolor=highlight_fill,
                                                pad=max(1.5, pad)))

        for gene in highlight:
            if 'bbox' in highlight[gene]:
                if 'edgecolor' not in highlight[gene]['bbox']:
                    highlight[gene]['bbox']['edgecolor'] = 'none'
                if 'facecolor' not in highlight[gene]['bbox']:
                    highlight[gene]['bbox']['facecolor'] = 'none'
        
        chrom_annot = defaultdict(list)
        for a in annot:
            chrom_annot[a[0]].append(a[1:])
            
        for chrom, annot in chrom_annot.items():
            if chrom not in self.chr_axes:
                continue
            ax = self.chr_axes[chrom]

            annot = sorted(annot, reverse=True)

            y_unit = -sub(*self._scaled_y_lim(ax)) / -sub(*self.ylim)
            nudge = 0.01 * y_unit
        
            polybuff = []
            for pos, name, textcolor, textsize, linecolor in annot:

                if type(highlight) is list or type(highlight) is set:
                    hl = name in highlight
                elif type(highlight) is dict or type(highlight) is defaultdict:
                    hl = highlight[name]
                else:
                    hl = None
                
                x, y, poly = self.get_polygon(name, pos, y1, ax, pad=pad)
                
                while any(self._is_polygons_intersecting(poly, p) 
                          for p in polybuff):
                    y += nudge
                    poly.nudge_y(nudge)

                self.draw_text(pos, x, y, name, textcolor=textcolor, 
                               fontsize=textsize*self.font_size, linecolor=linecolor,
                            #    ax=ax, y_line_bottom=y0*y_unit,                                
                               ax=ax, y_line_bottom=self.map_y(y0, ax),                                
                               highlight=hl, **kwargs)

                polybuff.append(poly)
                if len(polybuff) > 40:
                    del polybuff[0]        
            z = 100
            for i, t in enumerate(reversed(ax.texts)):
                t.set_zorder(z+i)

            for zoom, ax in zip(self.zooms, self.zoom_axes):

                # y_unit = -sub(*self._scaled_y_lim(ax, zoom=True)) / -sub(*self.ylim)
                # nudge = 0.01 * y_unit
                self.map_y(0.01, ax)
        
                zoom_polybuff = []
                for pos, name, *args in annot:

                    if type(highlight) is list or type(highlight) is set:
                        hl = name in highlight
                    elif type(highlight) is dict or type(highlight) is defaultdict:
                        hl = highlight[name]
                    else:
                        hl = None
                        
                    if pos >= zoom[0] and pos < zoom[1]:
                        x, y, poly = self.get_polygon(name, pos, zoom_y1, ax)
                        while any(self._is_polygons_intersecting(poly, p) 
                                  for p in zoom_polybuff):
                            y += nudge
                            poly.nudge_y(nudge)

                        self.draw_text(pos, x, y, name, textcolor=textcolor, 
                               fontsize=textsize*self.zoom_font_size, linecolor=linecolor,
                            #    ax=ax, y_line_bottom=y0*y_unit,                                
                               ax=ax, y_line_bottom=self.map_y(zoom_y0, ax),                                 
                               highlight=hl, **kwargs)
        
                        zoom_polybuff.append(poly)
                        if len(zoom_polybuff) > 20:
                            del zoom_polybuff[0]        
                z = 100
                for i, t in enumerate(reversed(ax.texts)):
                    t.set_zorder(z+i)
                    


    def add_segments(self, annot:MutableSequence, base:float=None, 
                     height:float=None,label:str=None, min_visible_width:int=200000, 
                     **kwargs:dict) -> None:
        """
        Add segments (rectangles) to the chromosome ideograms.

        Parameters
        ----------
        annot : 
            List of tuples with chromosome name, start and end positions of each
            segment: `(<chrom>, <start>, <end>, [<alpha>])`
        base : 
            Y coordinate for lower edge of rectangles, by default None. 
            If None, the lower edge of ideogram is used.
        height : 
            Height of rectangles in y coordinates, by default None. 
            If None, the ideogram height is used.
        label : 
            Label for plot legend, by default None
        min_visible_width : 
            Minimum with of rendered rectangles ensuring that very short segments
            remain visible, by default 200000
        **kwargs :
            Additional keyword arguments for matplotlib.patches. Rectangle
            (defaults to {facecolor='black', edgecolor='none'})
        """

        kwargs.setdefault('facecolor', 'black')
        kwargs.setdefault('edgecolor', 'none')
        
        if label is not None:
            self.legend_handles.append(
                patches.Patch(facecolor=kwargs['facecolor'], 
                              edgecolor=kwargs['edgecolor'],label=label)
            )
        
        if base is None:
            base = self.ideogram_base
        if height is None:
            height = self.ideogram_height            

        chrom_annot = defaultdict(list)
        for a in annot:
            chrom_annot[a[0]].append(a[1:])
            
        for chrom, annot in chrom_annot.items():
            ax = self.chr_axes[chrom]
            annot = sorted(annot, reverse=True)
            y_unit = -sub(*self._scaled_y_lim(ax)) / -sub(*self.ylim)
            for start, end, *extra in annot:
                _kwargs = kwargs.copy()
                if extra and 'alpha' not in kwargs:
                    _kwargs['alpha'] = extra[0]
                else:
                    _kwargs['alpha'] = 1
                # scaled_base = base * y_unit
                # scaled_height = height * y_unit                
                scaled_base = self.map_y(base, ax)
                scaled_height = self.map_y(height, ax)
                width = end - start
                if width < min_visible_width:
                    start -= min_visible_width/2
                    width += min_visible_width

                rect = patches.Rectangle((start, scaled_base), width, 
                                         scaled_height, linewidth=1, zorder=0,
                                         **_kwargs)
                ax.add_patch(rect)    
                
                for zoom_ax in self.zoom_axes:
                    zoom_y_unit = -sub(*self._scaled_y_lim(zoom_ax, zoom=True)) / -sub(*self.ylim)
                    # zoom_scaled_base = base * zoom_y_unit
                    # zoom_scaled_height = height * zoom_y_unit                      
                    zoom_scaled_base = self.map_y(base, zoom_ax)
                    zoom_scaled_height = self.map_y(height, zoom_ax)               
                    # print(start, zoom_scaled_base, width,
                    #                          zoom_scaled_height)
                    rect = patches.Rectangle((start, zoom_scaled_base), width,
                                             zoom_scaled_height, linewidth=1,
                                             zorder=0, **_kwargs)
                    zoom_ax.add_patch(rect)    


    def bezier_lines(self, ax, pairs, base, height, adjust_heights,
                     color, color_cycle, linewidth, zorder, **kwargs):
    
        def _get_color(n, lightness=0.4):
            color_cycle = cycle([matplotlib.colors.to_hex(c) for c in sns.husl_palette(n, l=lightness)])
            for color in color_cycle:
                yield color
        
        if color_cycle is None:
            husl_colors = _get_color(len(pairs), lightness=0.7)
        else:
            husl_colors = _get_color(color_cycle, lightness=0.7)
        
        bezier_path = np.linspace(0, 1, 100)

        for x1, x2 in pairs:

            y1, y2 = base, base
            if adjust_heights:
                h = self.map_y(height, ax) * np.log(abs(x1-x2)) / 10
            else:
                h = self.map_y(height, ax) * 2
    
            # Control point: midpoint lifted vertically
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2 + h
        
            # Compute Bézier curve points
            x = (1 - bezier_path) ** 2 * x1 + 2 * (1 - bezier_path) * bezier_path * xm + bezier_path ** 2 * x2
            y = (1 - bezier_path) ** 2 * y1 + 2 * (1 - bezier_path) * bezier_path * ym + bezier_path ** 2 * y2
        
            y += self.map_y(base, ax) 

            if color is None:
                _color = next(husl_colors)
            else:
                _color = color
            ax.plot(x, y, '-', linewidth=linewidth, color=_color, 
                    zorder=zorder, **kwargs)


    def add_connections(self, annot:MutableSequence, base:float=None, 
                    height:float=None, color:str=None, color_cycle=None, linewidth:float=1,
                    zorder:float=100, adjust_heights=True, **kwargs:dict) -> None:

        if base is None:
            base = self.ideogram_base + self.ideogram_height
        if height is None:
            height = self.ideogram_height * 2

        chrom_annot = defaultdict(list)
        for a in annot:
            chrom_annot[a[0]].append(a[1:])

        for chrom, annot in chrom_annot.items():
            if chrom not in self.chr_axes:
                continue
            ax = self.chr_axes[chrom]
            annot = sorted(annot, reverse=True)
        
            self.bezier_lines(ax, annot, base=base, height=height, 
                              adjust_heights=adjust_heights, color=color,
                              color_cycle=color_cycle,
                              linewidth=linewidth, zorder=zorder, **kwargs)
            
            for ax in self.zoom_axes:
                self.bezier_lines(ax, annot, base=base, height=height, 
                              adjust_heights=adjust_heights, color=color,
                              color_cycle=color_cycle,
                              linewidth=linewidth, zorder=zorder, **kwargs)
    

    def add_vlines(self, step:int=1000000, color:str='black', linewidth:float=0.1,
                   zorder:float=100, **kwargs:dict) -> None:
        """
        Adds vertical lines to the chromosome ideograms.

        Parameters
        ----------
        step : 
            Number of bases between lines, by default 1000000
        color : 
            Color of lines, by default 'black'
        linewidth : 
            Width of lines, by default 0.1
        zorder : 
            zorder of lines, by default 100
        """
        for i, ax in enumerate(self.ax_list):

            s, e = self.centromeres[self.chr_names[i]]

            ymin = self.map_y(self.ideogram_base, ax)        
            ymax = self.map_y(self.ideogram_base+self.ideogram_height, ax)
            ax.vlines(
                [x for x in range(0, self.chr_sizes[i], step) if x < s or x > e],
                ymin=ymin, ymax=ymax, linewidth=linewidth, color=color,
                zorder=zorder, **kwargs)
            for ax in self.zoom_axes:
                ymin = self.map_y(self.ideogram_base, ax)        
                ymax = self.map_y(self.ideogram_base+self.ideogram_height, ax)
                ax.vlines(
                    [x for x in range(0, self.chr_sizes[i], step) if x < s or x > e], 
                    ymin=ymin, ymax=ymax, linewidth=linewidth, color=color,
                    zorder=zorder, **kwargs)
                
    
    # TODO rename yaxis to data_scaling or something or use offset and scale 
    # or offset and height
    def map_method(self, method:Callable, data:pd.DataFrame=None, ch:str='chrom',
                   yaxis:tuple=(0.5, 3.5), **kwargs:dict) -> None:
        """
        Map a matplotib axes method like plot and scatter to each ideogram 
        panel (axis).
        
        Parameters
        ----------
        method : 
            Method of matplotib.axes.Axes to apply to each ideogram panel 
            (E.g. Axes.scatter).
        data : 
            Data frame with with data for x and y coordinates by chromsome
        ch : 
            Name of data frame column holding chromosome names, by default 
            'chrom'
        x : str
            Name of data frame column holding x coordinates, by default 'x'
        y : str
            Name of data frame column holding y coordinates, by default 'y'
        yaxis : 
            Y interval of ideogram panel axis ideogram to map data to, 
            by default (0.5, 3.5)
        **kwargs :
            Additional keyword arguments are passed to the plotting function 
            as keyword arguments.
        """
        def method_not_found(): # just in case we dont have the function
            print('No Function '+method+' Found!')

        grouped = data.groupby(ch, observed=True)
        for chrom, group in grouped:
            if chrom not in self.chr_axes:
                continue
            ax = self.chr_axes[chrom]
            bottom, top = yaxis            
            scaled_y_lim = ax.get_ylim()
            if 'ylim' in kwargs:
                dy = -sub(*kwargs['ylim'])
            else:
                dy = -sub(*self.ylim)
            df = group.reset_index() # create independent dataframe
            df['y'] -= df.y.min()
            df['y'] /= df.y.max()
            df['y'] = (df.y * ((top-bottom) * -sub(*scaled_y_lim)) 
                       / -sub(*self.ylim) + bottom 
                       /  -sub(*self.ylim) * -sub(*scaled_y_lim))
            x = df.x
            y = df.y
            if 'x' in kwargs: del kwargs['x']
            if 'y' in kwargs: del kwargs['y'] 

            if 'y2' in kwargs:
                if 'y2' not in df.columns:
                    df['y2'] = kwargs['y2']
                df['y2'] -= df.y2.min()
                df['y2'] /= df.y2.max()
                df['y2'] = (df.y2 * ((top-bottom) * -sub(*scaled_y_lim)) 
                        / -sub(*self.ylim) + bottom 
                        /  -sub(*self.ylim) * -sub(*scaled_y_lim))
                y2 = df.y2

            method_name = method.__name__
            method = getattr(ax, method_name, method_not_found) 
            g = method(x, y, **kwargs)
            
            for ax in self.zoom_axes:
                scaled_y_lim = ax.get_ylim()
                if 'ylim' in kwargs:
                    dy = -sub(*kwargs['ylim'])
                else:
                    dy = -sub(*self.ylim)
                df = group.reset_index() # create independent dataframe
                df['y'] -= df.y.min()
                df['y'] /= df.y.max()
                df['y'] = (df.y * ((top-bottom) * -sub(*scaled_y_lim)) 
                           / -sub(*self.ylim) + bottom 
                           /  -sub(*self.ylim) * -sub(*scaled_y_lim))
                x = df.x
                y = df.y
                if 'x' in kwargs: del kwargs['x']
                if 'y' in kwargs: del kwargs['y']   
                
                method = getattr(ax, method_name, method_not_found) 
                g = method(x, y, **kwargs)
                
                try:
                    ax.get_legend().remove()
                except:
                    pass
                ax.set_xlabel('')
                ax.set_ylabel('')  
            plt.xlabel('')
            plt.ylabel('')

    def map_fun(self, fun:Callable, data:pd.DataFrame=None, ch='chrom', 
                yaxis:tuple=(0.5, 3.5), **kwargs:dict) -> None:
        """
        Map a plotting function like seaborn.scatterplot to each ideogram panel (axis).
        
        Parameters
        ----------
        fun : 
            Function to apply to each ideogram panel (E.g. seaborn.scatterplot).
        data : 
            Data frame with with data for x and y coordinates by chromsome.
        ch : 
            Name of data frame column holding chromosome names, by default 'chrom'
        x : str
            Name of data frame column holding x coordinates, by default 'x'
        y : str
            Name of data frame column holding y coordinates, by default 'y'
        yaxis : 
            Y interval of ideogram panel axis ideogram to map data to, 
            by default (0.5, 3.5)
        **kwargs :
            Additional keyword arguments are passed to the plotting function as
            keyword arguments.            
        """            
        grouped = data.groupby(ch, observed=True)
        for chrom, group in grouped:
            if chrom not in self.chr_axes:
                continue
            ax = self.chr_axes[chrom]
            scaled_y_lim = ax.get_ylim()
            if 'ylim' in kwargs:
                dy = -sub(*kwargs['ylim'])
            else:
                dy = -sub(*self.ylim)
            df = group.reset_index() #.copy()
            if yaxis is None:
                df['y'] = df.y * -sub(*scaled_y_lim) / -sub(*self.ylim)
            else:
                bottom, top = yaxis
                df['y'] -= df.y.min()
                df['y'] /= df.y.max()
                df['y'] = (df.y * ((top-bottom) * -sub(*scaled_y_lim)) 
                           / -sub(*self.ylim) + bottom 
                           /  -sub(*self.ylim) * -sub(*scaled_y_lim))
            g = fun(df, ax=ax, **kwargs)
            
            for ax in self.zoom_axes:
                scaled_y_lim = ax.get_ylim()
                if 'ylim' in kwargs:
                    dy = -sub(*kwargs['ylim'])
                else:
                    dy = -sub(*self.ylim)
                df = group.reset_index() #.copy()
                if yaxis is None:
                    df['y'] = df.y * -sub(*scaled_y_lim) / -sub(*self.ylim)
                else:
                    bottom, top = yaxis
                    df['y'] -= df.y.min()
                    df['y'] /= df.y.max()
                    df['y'] = (df.y * ((top-bottom) * -sub(*scaled_y_lim)) 
                               / -sub(*self.ylim) + bottom 
                               / -sub(*self.ylim) * -sub(*scaled_y_lim))
                g = fun(df, ax=ax, **kwargs)
                try:
                    ax.get_legend().remove()
                except:
                    pass
                ax.set_xlabel('')
                ax.set_ylabel('')                               
            plt.xlabel('')
            plt.ylabel('')

    def add_legend(self, outside=False, **kwargs:dict) -> None:
        """
        Adds a legend to the chromosome ideograms.

        Parameters
        ----------
        outside : 
            Place the legend outside the plot, by default False
        **kwargs :
            Keyword arguments are passed to matplotlib's `legend`. 
            Defaults to {'loc': 'center left', 'bbox_to_anchor': (1.02, 0.5),
            'frameon': False} if `outside` is True, otherwise defaults to {'frameon': False}.
        """
        if outside:
            kwargs.setdefault('loc', 'center left')
            kwargs.setdefault('bbox_to_anchor', (1.02, 0.5))
        kwargs.setdefault('frameon', False)

        for i, ax in enumerate(self.ax_list):
            if i == 0:                
                handles, labels = ax.get_legend_handles_labels()
                if self.legend_handles:
                    handles = self.legend_handles + handles
                ax.legend(handles=handles, **kwargs)
            else:
                ax.legend_.remove()



    def add_horizon(self, data:pd.DataFrame=None, ch:str='chrom', y:str='y', x:str='x', 
                    cut:float=None, # float, takes precedence over quantile_span
                    quantile_span:float=None, beginzero:bool=True, 
                    base:float=None, height:float=None,
                    colors:List[str] = [
                        '#CCE2DF', '#59A9A8', '#374E9B', 'midnightblue',
                        '#F2DE9A', '#DA8630', '#972428', 'darkred',
                        '#D3D3D3'],
                    **kwargs:dict) -> None:
        """
        Adds a horizon plot to the chromosome ideograms.

        Parameters
        ----------
        data : 
            Pandas DataFrame with sorted x and y data grouped by chromosome
        ch : 
            Name of data frame column holding chromosome names, by default 'chrom'
        x : 
            Name of data frame column holding x coordinates, by default 'x'
        y : 
            Name of data frame column holding y coordinates, by default 'y'
        cut : 
            Lower and upper y values for folding the horizon plot. Default for each
            chromosome is a third of the y range on each side of zero.
        quantile_span : 
            Lower and upper quantiles of y values for each chromosome to include 
            in the horizon plot. `quantile_span=(0, 1)` produces the same result 
            as `cut=None`. Outlier values below and above this range is assigned
            separate darker colors. 
        beginzero : 
            Pad to make the x axis begin at zero, by default True
        base : 
            Y coordinate for lower edge of horizon plot, by default None.
            If None, the lower edge of ideogram is used.
        height : 
            Height of horizon plot in y coordinates, by default None. 
            If None, the ideogram height is used.
        colors : 
            _description_, by default [ '#CCE2DF', '#59A9A8', '#374E9B',
            'midnightblue', '#F2DE9A', '#DA8630', '#972428', 'darkred', '#D3D3D3']
        """
        if base is None:
            if self.ideogram_base is not None:
                base = self.ideogram_base
            else:
                base = 0
        if height is None:
            if self.ideogram_height is not None:
                height = self.ideogram_height
            else:
                height = 1

        grouped = data.groupby(ch, observed=True)
        for chrom, group in grouped:
            if chrom not in self.chr_axes:
                continue
            ax = self.chr_axes[chrom]
            
            df = group.reset_index() # make copy
            horizon(df, y=y, ax=ax,
                    cut=cut, 
                    quantile_span = quantile_span,
                    x=x,
                    beginzero=beginzero, 
                    offset=self.map_y(base, ax),
                    height=self.map_y(height, ax),
                    colors = colors,
                    **kwargs)
                             
            for ax in self.zoom_axes:
                horizon(df, y=y, ax=ax,
                    cut=cut, 
                    quantile_span = quantile_span,
                    x=x,
                    beginzero=beginzero, 
                    offset=self.map_y(base, ax),
                    height=self.map_y(height, ax),
                    colors = colors,
                    **kwargs)
    

    def annotate(self, xy:tuple=None, testxy:tuple=None, text:str=None, 
                 chrom:str=None, ax:matplotlib.axes.Axes=None, wrap:int=30, 
                 **kwargs:dict) -> None:
        """
        Draws an annotation with an arrow to a point from a text box.

        Parameters
        ----------
        xy : 
            Coordinate to point to as a tuple of chromosome position and 
            y plotting coordinate
        testxy : 
            Coordinate to text box as a tuple of chromosome position and 
            y plotting coordinate
        text : 
            Text for text box
        chrom : 
            Chromosome to annotate, by default the first or only chromosome 
            in the plot
        ax : 
            Axis with the point the arrow points to, by default the axis for 
            the chromosome ideogram.
        wrap : 
            Line wrap for text box, by default 30
        kwargs : 
            Additional keyword arguments passed to matplotlib's annotate
        """
        kwargs.setdefault('fontsize', 7)
        kwargs.setdefault('ha', 'left')
        kwargs.setdefault('va', 'bottom')
        kwargs.setdefault(
            'bbox', dict(facecolor='white', edgecolor='red', linewidth=1))
        kwargs.setdefault('arrowprops', dict(facecolor='red', # color of arrow
                                            shrink=0.01, # distance from point
                                            width=1.5, # width of arrow
                                            headwidth=5, # width of arrow head
                                            headlength=7, # length of arrow head
                                            connectionstyle='angle3,angleA=0,angleB=-90'))
        if chrom is None:
            chrom = self.chr_names[0]
        _ax = self.chr_axes[chrom]
        if ax is None:
            ax = _ax
            xy = (xy[0], self.map_y(xy[1], _ax))

        dpi = self.fig.dpi
        bbox = _ax.get_window_extent()
        width_inches = bbox.width / dpi  # in inches
        point_in_bases = _ax.get_xlim()[1]/(width_inches*72)

        ax.annotate(textwrap.fill(text, wrap), 
                    xy=xy, 
                    xycoords="data",
                    xytext=(testxy[0]/point_in_bases, 
                            self.map_y(testxy[1], _ax)/point_in_bases), 
                    textcoords=OffsetFrom(_ax.bbox, (0, 0), "points"),
                    **kwargs) 

