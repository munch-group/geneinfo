
import numpy as np
from collections import defaultdict
import pandas as pd
from pandas.api.types import is_object_dtype
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

from ..intervals import *

import math
from math import isclose, floor, log10
from operator import sub
import warnings
from itertools import chain
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.artist import Artist
import matplotlib.gridspec as gridspec
from matplotlib.transforms import (Bbox, TransformedBbox,
                                   blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector,
                                                   BboxConnectorPatch,
                                                   BboxPatch)
import seaborn as sns


def chrom_ideogram(annot:list, hspace:float=0.1, min_visible_width:int=200000, figsize:tuple=(10,10), assembly:str='hg38'):
    """
    Plots an ideogram of the human chromosomes with annotations.

    Parameters
    ----------
    annot : 
        List of tuples with annotations. Each tuple should contain the chromosome name, start and end position, color, label and optionally the width and height of the annotation.
    hspace : 
        Space between ideograms, by default 0.1
    min_visible_width : 
        Minum display width of very short annotations, by default 200000
    figsize : 
        Figure size, by default (10,10)
    assembly : 
        Human genome assembly, by default 'hg38'

    Examples
    --------

    ```python
    annot = [
        ('chr1', 20000000, 20100000, 'red', 'TP53'),
        ('chr5', 40000000, 70000000, 'red', None, 1, 0.5), 
        ('chr8', 90000000, 110000000)
    ]
    chrom_ideogram(annot, figsize=(15, 9), hspace=0.2)

    # black ticks every 10Mb on chrX
    annot = [('chrX', x[0], x[1], 'black', str(x[2]/1000000)) for x in zip(range(0, 150000000, 10000000), range(300000, 150000000, 10000000), range(0, 150000000, 10000000))]
    chrom_ideogram(annot, figsize=(15, 9), hspace=0.2)
    ```

    """

# annot = [('chr1', 20000000, 20100000, 'red', 'TP53'), ('chr7', 20000000, 30000000, 'orange', 'DYNLT3')] \
# + [('chr5', 40000000, 70000000, 'red', None, 1, 0.5), ('chr8', 90000000, 110000000)] \
#  + [('chrX', x[0], x[1], 'black', str(x[2]/1000000)) for x in zip(range(0, 150000000, 10000000), range(300000, 150000000, 10000000), range(0, 150000000, 10000000))]

# chrom_ideogram(annot, figsize=(15, 9), hspace=0.2) 

    d = {'axes.linewidth': 0.8, 'grid.linewidth': 0.64, 'lines.linewidth': 0.96, 
         'lines.markersize': 3.84, 'patch.linewidth': 0.64, 'xtick.major.width': 0.8,
         'ytick.major.width': 0.8, 'xtick.minor.width': 0.64, 'ytick.minor.width': 0.64,
         'xtick.major.size': 3.84, 'ytick.major.size': 3.84, 'xtick.minor.size': 2.56, 
         'ytick.minor.size': 2.56, 'font.size': 7.68, 'axes.labelsize': 7.68,
         'axes.titlesize': 7.68, 'xtick.labelsize': 7.04, 'ytick.labelsize': 7.04, 
         'legend.fontsize': 7.04, 'legend.title_fontsize': 7.68}
    
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

    chr_names = [f'chr{x}' for x in list(range(1, 23))+['X', 'Y']]
    chr_sizes = [chrom_lengths[assembly][chrom] for chrom in chr_names]
    figwidth = max(chr_sizes)
    
    with plt.rc_context(d):
    
        nr_rows, nr_cols = len(chr_names)-2, 2

        fig = plt.figure(figsize=figsize)

        gs = matplotlib.gridspec.GridSpec(nr_rows, 25)
        gs.update(wspace=0, hspace=hspace) # set the spacing between axes.             
        ax_list = [plt.subplot(gs[i, :]) for i in range(nr_rows-2)]
        ax_list.append(plt.subplot(gs[nr_rows-2, :9]))
        ax_list.append(plt.subplot(gs[nr_rows-1, :9]))
        ax_list.append(plt.subplot(gs[nr_rows-2, 9:]))
        ax_list.append(plt.subplot(gs[nr_rows-1, 9:]))

        chr_axes = dict(zip(chr_names, ax_list))

        for ax in ax_list[:-4]:
            ax.set_xlim((-200000, figwidth+100000))
        for ax in ax_list[-4:]:
            ax.set_xlim((-200000, ((25-9)/25)*figwidth+100000))

        for i in range(len(ax_list)):
            chrom = chr_names[i]
            ax = ax_list[i]
            start, end = 0, chr_sizes[i] 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_ylim((0, 3))
            # ax.set_ylim((0, 5))

            if i in [20, 21]:   
                x = -3500000 * 10 / figsize[1]
            else:
                x = -2000000 * 10 / figsize[1]
            ax.text(x, 1, chrom.replace('chr', ''), fontsize=7, horizontalalignment='right', weight='bold')

            # h = ax.set_ylabel(chrom)
            # h.set_rotation(0)
            ax.set_yticklabels([])

            if i == 0:
                ax.spines['top'].set_visible(True)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top') 
                ax.yaxis.set_ticks_position('none')
            elif i == len(ax_list)-1:
                ax.xaxis.tick_bottom()
                ax.spines['bottom'].set_visible(True)                    
                ax.yaxis.set_ticks_position('none')
            else:
                ax.set_xticklabels([])
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')


            # draw chrom
            g = ax.add_patch(patches.Rectangle((start, 1), end-start, 1, 
                                       # fill=False,
                                       facecolor='#EBEAEA',
                                       edgecolor='black',
                                       # edgecolor=None,
                                       zorder=1, linewidth=0.7
                                      ))

            # draw centromere
            cent_start, cent_end = centromeres[chrom]
            ax.add_patch(patches.Rectangle((cent_start, 0), cent_end-cent_start, 3, 
                                       fill=True, color='white',
                                       zorder=2))
            xy = [[cent_start, 1], [cent_start, 2], [cent_end, 1], [cent_end, 2]]
            g = ax.add_patch(patches.Polygon(xy, closed=True, zorder=3, fill=True,
                                     # color='#666666',
                                     color='#777777',
                                    ))


        def plot_segment(chrom, start, end, color='red', label=None, base=0, height=1):

            base += 1
            
            x, y, width = start, base, end-start

            if width < min_visible_width:
                x -= min_visible_width/2
                width += min_visible_width

            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='none', facecolor=color, zorder=3)
            chr_axes[chrom].add_patch(rect)    
            if label is not None:
                chr_axes[chrom].plot([x+width/2, x+width/2], [y+height, y+height+0.3], linewidth=0.5, color=color, zorder=3)
                t = chr_axes[chrom].text(x+width/2, y+height+0.3, label, fontsize=4, horizontalalignment='left',# weight='bold',
                         verticalalignment='bottom', rotation=45, zorder=3)

                transf = chr_axes[chrom].transData.inverted()
                bb = t.get_window_extent(renderer = fig.canvas.get_renderer())
                bb_datacoords = bb.transformed(transf)
                # print(bb_datacoords)

        for tup in annot:
            plot_segment(*tup) 

        # text_labels = defaultdict(list)
        # import textalloc as ta
        # for tup in annot:
        #     if len(tup) == 5:
        #         text_labels[tup[0]].append((tup[1], 1+1+0.3, tup[3], tup[4]))
        # for chrom in text_labels:
        #     x, y, _, texts = zip(*text_labels[chrom])
        #     ta.allocate(chr_axes[chrom],x,y,
        #                 texts,
        #                 x_scatter=x, y_scatter=y,
        #                 textsize=5,
        #                 # x_lines=[ (0, chrom_lengths[assembly][chrom]) ] * 3,
        #                 # y_lines=[(1, 1), (1.5, 1.5), (2, 2)],
        #                 # avoid_label_lines_overlap=True,
        #                 # avoid_crossing_label_lines=True,
        #                # direction='northeast',
        #                # linecolor=color
        #                )
                

# annot = [('chr1', 20000000, 20100000, 'red', 'TP53'), ('chr7', 20000000, 30000000, 'orange', 'DYNLT3')] \
# + [('chr5', 40000000, 70000000, 'red', None, 1, 0.5), ('chr8', 90000000, 110000000)] \
#  + [('chrX', x[0], x[1], 'black', str(x[2]/1000000)) for x in zip(range(0, 150000000, 10000000), range(300000, 150000000, 10000000), range(0, 150000000, 10000000))]

# chrom_ideogram(annot, figsize=(15, 9), hspace=0.2) 




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
            
class GenomeIdeogram(object):

    d = {'axes.linewidth': 0.8, 'grid.linewidth': 0.64, 'lines.linewidth': 0.96, 
         'lines.markersize': 3.84, 'patch.linewidth': 0.64, 'xtick.major.width': 0.8,
         'ytick.major.width': 0.8, 'xtick.minor.width': 0.64, 'ytick.minor.width': 0.64,
         'xtick.major.size': 3.84, 'ytick.major.size': 3.84, 'xtick.minor.size': 2.56, 
         'ytick.minor.size': 2.56, 'font.size': 7.68, 'axes.labelsize': 7.68,
         'axes.titlesize': 7.68, 'xtick.labelsize': 7.04, 'ytick.labelsize': 7.04, 
         'legend.fontsize': 7.04, 'legend.title_fontsize': 7.68}
    
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

    
    def __init__(self, axes_height_inches=1, axes_width_inches=12, hspace=0, ylim=(0, 1), 
                 rel_font_height=0.05, assembly:str='hg38', min_stick_height=0.3):

        self.ideogram_base = None
        self.ideogram_height = None
        self.min_stick_height = min_stick_height
        self.legend_handles = []
        self.zoom_axes = []
        
        self.end_padding = 300000
        self.chr_names = [f'chr{x}' for x in list(range(1, 23))+['X', 'Y']]
        self.chr_sizes = [self.chrom_lengths[assembly][chrom] for chrom in self.chr_names]
        self.max_chrom_size = max(self.chr_sizes)
        nr_rows = len(self.chr_names) - 1
        self.aspect = axes_height_inches / axes_width_inches
        axes_width = self.max_chrom_size + 2 * self.end_padding
        axes_height = self.aspect * axes_width
        fig_height_inches = axes_height_inches * (nr_rows-1) + hspace * (nr_rows-1)
        fig_width_inches = axes_width_inches #fig_height_inches / (nr_rows-1) / aspect 
        figsize = (fig_width_inches, fig_height_inches)
        point_size = 1/72
        self.ylim = ylim
        self.font_size = rel_font_height * axes_height_inches / point_size

        self.fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0)
        
        with plt.rc_context(self.d):
        
            nr_rows, nr_cols = len(self.chr_names)-2+1, 2
    
            # fig = plt.figure(figsize=figsize)

            
            # gs = matplotlib.gridspec.GridSpec(nr_rows, 25)
            gs = matplotlib.gridspec.GridSpec(nr_rows+1, 25, height_ratios=[1e-2]+[1]*nr_rows)
            gs.update(wspace=0, hspace=hspace) 

            dummy_ax = plt.subplot(gs[0, :])
            xlim = (-self.end_padding, self.max_chrom_size+self.end_padding)
            dummy_ax.set_xlim(xlim)
            # dummy_ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

            dummy_ax.spines['top'].set_visible(True)
            dummy_ax.xaxis.tick_top()
            dummy_ax.xaxis.set_label_position('top') 
            dummy_ax.yaxis.set_ticks_position('none')
            dummy_ax.set_yticklabels([])
        
            ax_list = [plt.subplot(gs[i, :]) for i in range(1, nr_rows-2)]
            ax_list.append(plt.subplot(gs[nr_rows-2, :9]))
            ax_list.append(plt.subplot(gs[nr_rows-1, :9]))
            ax_list.append(plt.subplot(gs[nr_rows-2, 9:]))
            ax_list.append(plt.subplot(gs[nr_rows-1, 9:]))

            self.ax_list = ax_list
            self.chr_axes = dict(zip(self.chr_names, self.ax_list))
    
            for ax in self.ax_list[:-4]:
                xlim = (-end_padding, self.max_chrom_size+end_padding)
                scaled_y_lim = xlim[0] * self.aspect, xlim[1] * self.aspect
                ax.set_xlim(xlim)
                ax.set_ylim(scaled_y_lim)
            for ax in ax_list[-4:]:
                xlim = (-end_padding, ((25-9)/25)*self.max_chrom_size+end_padding)
                scaled_y_lim = xlim[0] * self.aspect, xlim[1] * self.aspect
                ax.set_xlim(xlim)
                ax.set_ylim(scaled_y_lim)
    
            for i in range(len(self.ax_list)):
                chrom = self.chr_names[i]
                ax = ax_list[i]
                
                start, end = 0, self.chr_sizes[i]
                ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

                # ax.spines['top'].set_visible(False)
                # ax.spines['right'].set_visible(False)
                # ax.spines['bottom'].set_visible(False)
                # ax.spines['left'].set_visible(False)
                # ax.set_ylim((0, 3))
                # ax.set_ylim((0, 5))
    
                if i in [20, 21]:   
                    x = -3500000 * 10 / figsize[1]
                else:
                    x = -2000000 * 10 / figsize[1]

                ax.text(x, self.map_y(-sub(*self.ylim), ax)/2, chrom.replace('chr', ''), fontsize=7, verticalalignment='center', horizontalalignment='right', weight='bold')
    
                # h = ax.set_ylabel(chrom)
                # h.set_rotation(0)
                ax.set_yticklabels([])
    
                # if i == 0:
                #     ax.spines['top'].set_visible(True)
                #     ax.xaxis.tick_top()
                #     ax.xaxis.set_label_position('top') 
                #     ax.yaxis.set_ticks_position('none')
                # elif i == len(ax_list)-1:
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
                    # ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    def draw_chromosomes(self, base=0.05, height=0.25, facecolor='#EBEAEA', edgecolor='black', linewidth=0.7, **kwargs):

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
                cent_start, cent_end = self.centromeres[chrom]
                ymin, ymax = ax.get_ylim()
                # ax.add_patch(patches.Rectangle((cent_start, ymin), cent_end-cent_start, ymax, 
                #                            fill=True, color='white',
                #                            zorder=1))
                xy = [[cent_start, ideogram_base], [cent_start, ideogram_base+ideogram_height], [cent_end, ideogram_base], [cent_end, ideogram_base+ideogram_height]]
                g = ax.add_patch(patches.Polygon(xy, closed=True, zorder=2, fill=True,
                                         # color='#666666',
                                         color='#777777',
                                        ))
                
                # draw chrom
                g = ax.add_patch(patches.Rectangle((start, ideogram_base), cent_start-start, ideogram_height, 
                                           # fill=False,
                                           facecolor=facecolor,
                                           edgecolor='none',
                                           zorder=0,
                                           linewidth=linewidth,
                                           **kwargs
                                          ))
                g = ax.add_patch(patches.Rectangle((start, ideogram_base), cent_start-start, ideogram_height, 
                                           # fill=False,
                                           facecolor='none',
                                           edgecolor=edgecolor,
                                           zorder=0.5,
                                           linewidth=linewidth,
                                           **kwargs
                                          )) 
                g = ax.add_patch(patches.Rectangle((cent_end, ideogram_base), end-cent_end, ideogram_height, 
                                           # fill=False,
                                           facecolor=facecolor,
                                           edgecolor='none',
                                           zorder=0,
                                           linewidth=linewidth,
                                           **kwargs
                                          ))
                g = ax.add_patch(patches.Rectangle((cent_end, ideogram_base), end-cent_end, ideogram_height, 
                                           # fill=False,
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
                    cent_start, cent_end = self.centromeres[chrom]
                    ymin, ymax = ax.get_ylim()
                    # ax.add_patch(patches.Rectangle((cent_start, ymin), cent_end-cent_start, ymax, 
                    #                            fill=True, color='white',
                    #                            zorder=-2))
                    xy = [[cent_start, ideogram_base], [cent_start, ideogram_base+ideogram_height], [cent_end, ideogram_base], [cent_end, ideogram_base+ideogram_height]]
                    g = ax.add_patch(patches.Polygon(xy, closed=True, zorder=-1, fill=True,
                                             # color='#666666',
                                             color='#777777',
                                            ))
                    
                    # draw chrom
                    g = ax.add_patch(patches.Rectangle((start, ideogram_base), cent_start-start, ideogram_height, 
                                               # fill=False,
                                               facecolor=facecolor,
                                               edgecolor='none',
                                               zorder=0,
                                               linewidth=linewidth,
                                               **kwargs
                                              ))
                    g = ax.add_patch(patches.Rectangle((start, ideogram_base), cent_start-start, ideogram_height, 
                                               # fill=False,
                                               facecolor='none',
                                               edgecolor=edgecolor,
                                               zorder=0.5,
                                               linewidth=linewidth,
                                               **kwargs
                                              )) 
                    g = ax.add_patch(patches.Rectangle((cent_end, ideogram_base), end-cent_end, ideogram_height, 
                                               # fill=False,
                                               facecolor=facecolor,
                                               edgecolor='none',
                                               zorder=0,
                                               linewidth=linewidth,
                                               **kwargs
                                              ))
                    g = ax.add_patch(patches.Rectangle((cent_end, ideogram_base), end-cent_end, ideogram_height, 
                                               # fill=False,
                                               facecolor='none',
                                               edgecolor=edgecolor,
                                               zorder=0.5,
                                               linewidth=linewidth,
                                               **kwargs
                                              ))               
        

                



    
    def is_polygons_intersecting(self, a, b):
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

    
    def scaled_y_lim(self, ax):
        xlim = ax.get_xlim()

        bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        aspect = bbox.height / bbox.width
        
        return 0, -sub(*xlim) * aspect 
        # return xlim[0] * self.aspect, xlim[1] * self.aspect
    

    def map_y(self, y, ax, bottom=0, top=1):
        miny, maxy = ax.get_ylim()

        return y * (top - bottom) * (maxy - miny) / (self.ylim[1] - self.ylim[0]) + bottom * (maxy - miny)
        # zero = -miny
        # if y >= 0:
        #     return y * (top - max(bottom, zero)) * (maxy - zero) / (self.ylim[1] - self.ylim[0]) + max(bottom, zero) * (maxy - zero)
        # else:
        #     return y * (min(top, zero) - bottom) * (zero - miny) / (self.ylim[1] - self.ylim[0]) + min(top, zero) * (zero - miny)

        
    def draw_text(self, x_pos, y_pos, text, ax, color=None, y_line_bottom=0, highlight=None):
        y_unit = -sub(*self.scaled_y_lim(ax)) / -sub(*self.ylim)

        if bool(color is not None) == bool(highlight is not None):
            raise TypeError("Do not use color and highlight keyword arguments together")

        if highlight is True:
            text_props = dict(weight='bold', color='red')
            linecolor = 'lightgray'
        elif type(highlight) is dict:
            text_props = highlight
            linecolor = 'lightgray'
            # if 'color' in highlight:
            #     linecolor = highlight['color']
        else:
            text_props = dict(color='black',
                    fontweight='normal', # 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
                    # variant = 'small-caps', # [ 'normal' | 'small-caps' ]
                    fontstyle = 'normal', # [ 'normal' | 'italic' | 'oblique' ]
                    bbox=dict(boxstyle='square,pad=0', 
                              linewidth=0.2,
                              fc='none', 
                              alpha=1,
                              ec='none'),)
            linecolor = 'lightgray'
        
        t = ax.text(x_pos, y_pos, text, fontsize=self.font_size,                     
                    rotation=45, zorder=10, 
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    **text_props)
        ax.plot((x_pos, x_pos, x_pos+y_unit/10),
                (y_line_bottom, y_pos, y_pos+y_unit/10), 
                linewidth=0.5, 
                # color='darkgray', 
                color='lightgray',
                # alpha=0.3,
                zorder=-5)
    
    
    def get_polygon(self, text:str, x_pos:int, y_pos:float,ax:matplotlib.axes.Axes, pad=0) -> Polygon:

        y_unit = -sub(*self.scaled_y_lim(ax)) / -sub(*self.ylim)
        # y_unit = -sub(*self.scaled_y_lim(ax)) / -sub(*ax.get_ylim())

        y_pos = y_pos * y_unit
        
        t = ax.text(x_pos, y_pos, text, fontsize=self.font_size,
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
        df['y'] = df['x']*math.sin(math.radians(45))+df['y']*math.cos(math.radians(45))
        df['x'] = dfx
        df['y'] += y_pos
        df['x'] += x_pos
            
        coords = np.array(list(zip(df.x, df.y)))
        
        x_pos, y_pos = df['x'][0] - (df['x'][0] - df['x'][1]), df['y'][0]
    
        df['x'] = df.x + np.array([-pad, -pad, pad, pad])
        df['y'] = df.y + np.array([-pad, pad, pad, -pad])
        
        return x_pos, y_pos, Polygon([Point(x, y) for x, y in zip(df.x, df.y)])
        

    # def add_labels(self, data, labels='name', chrom='chrom', x='pos'):
    def add_labels(self, annot, y0=None, y1=None, bold=[], italic=[], colored=[], framed=[], filled=[], pad=0):

        if y0 is None:
            y0 = self.ideogram_base + self.ideogram_height
        if y1 is None:
            y1 = self.ideogram_base + self.ideogram_height + self.min_stick_height

    
        highlight = defaultdict(dict)
        for gene in bold:
            highlight[gene].update(dict(weight='bold'))
        for gene in italic:
            highlight[gene].update(dict(style='italic'))
        for gene in colored:
            highlight[gene].update(dict(color='red'))
        for gene in framed:
            if 'bbox' not in highlight[gene]:
                highlight[gene]['bbox'] = {}
            highlight[gene]['bbox'].update(dict(edgecolor='black', pad=pad, linewidth=0.5))
        for gene in filled:
            if 'bbox' not in highlight[gene]:
                highlight[gene]['bbox'] = {}
            highlight[gene]['bbox'].update(dict(facecolor='red', alpha=0.2, pad=pad))

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
            ax = self.chr_axes[chrom]

            annot = sorted(annot, reverse=True)

            y_unit = -sub(*self.scaled_y_lim(ax)) / -sub(*self.ylim)
            nudge = 0.01 * y_unit
        
            polybuff = []
            for pos, name, *args in annot:

                if type(highlight) is list or type(highlight) is set:
                    hl = name in highlight
                elif type(highlight) is dict or type(highlight) is defaultdict:
                    hl = highlight[name]
                else:
                    hl = None
                
                x, y, poly = self.get_polygon(name, pos, y1, ax, pad=pad)
                while any(self.is_polygons_intersecting(poly, p) for p in polybuff):
                    y += nudge
                    poly.nudge_y(nudge)

                self.draw_text(x, y, name, ax, *args, y_line_bottom=y0*y_unit, highlight=hl)

                polybuff.append(poly)
                if len(polybuff) > 40:
                    del polybuff[0]        
            z = 100
            for i, t in enumerate(reversed(ax.texts)):
                t.set_zorder(z+i)

            for zoom, ax in zip(self.zooms, self.zoom_axes):

                y_unit = -sub(*self.scaled_y_lim(ax)) / -sub(*self.ylim)
                nudge = 0.01 * y_unit
                
                zoom_polybuff = []
                for pos, name, *args in annot:

                    if type(highlight) is list or type(highlight) is set:
                        hl = name in highlight
                    elif type(highlight) is dict or type(highlight) is defaultdict:
                        hl = highlight[name]
                    else:
                        hl = None
                        
                    if pos >= zoom[0] and pos < zoom[1]:
                        x, y, poly = self.get_polygon(name, pos, y1, ax)
                        while any(self.is_polygons_intersecting(poly, p) for p in zoom_polybuff):
                            y += nudge
                            poly.nudge_y(nudge)
                        self.draw_text(x, y, name, ax, *args, y_line_bottom=y0*y_unit, highlight=hl)
        
                        zoom_polybuff.append(poly)
                        if len(zoom_polybuff) > 20:
                            del zoom_polybuff[0]        
                z = 100
                for i, t in enumerate(reversed(ax.texts)):
                    t.set_zorder(z+i)
                    


    def add_segments(self, annot, base=None, height=None, label=None,
                    min_visible_width:int=200000, **kwargs):

        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'black'
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'none'
        
        if label is not None:
            self.legend_handles.append(patches.Patch(facecolor=kwargs['facecolor'], edgecolor=kwargs['edgecolor'],label=label))
        
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
            y_unit = -sub(*self.scaled_y_lim(ax)) / -sub(*self.ylim)
            for start, end in annot:
                scaled_base = base * y_unit
                scaled_height = height * y_unit                
                width = end - start
                if width < min_visible_width:
                    start -= min_visible_width/2
                    width += min_visible_width

                rect = patches.Rectangle((start, scaled_base), width, scaled_height, linewidth=1, zorder=3, **kwargs)
                ax.add_patch(rect)    

                for zoom_ax in self.zoom_axes:
                    zoom_y_unit = -sub(*self.scaled_y_lim(zoom_ax)) / -sub(*self.ylim)
                    zoom_scaled_base = base * zoom_y_unit
                    zoom_scaled_height = height * zoom_y_unit                      

                    rect = patches.Rectangle((start, zoom_scaled_base), width, zoom_scaled_height, linewidth=1, zorder=3, **kwargs)
                    zoom_ax.add_patch(rect)    



    def add_vlines(self, step=1000000, color='black', linewidth=0.1, zorder=100, **kwargs):
        for i, ax in enumerate(self.ax_list):

            s, e = self.centromeres[self.chr_names[i]]

            ymin = self.map_y(self.ideogram_base, ax)        
            ymax = self.map_y(self.ideogram_base+self.ideogram_height, ax)
            ax.vlines([x for x in range(0, self.chr_sizes[i], step) if x < s or x > e], 
                      ymin=ymin, ymax=ymax, linewidth=linewidth, color=color, zorder=zorder, **kwargs)
            for ax in self.zoom_axes:
                ymin = self.map_y(self.ideogram_base, ax)        
                ymax = self.map_y(self.ideogram_base+self.ideogram_height, ax)
                ax.vlines([x for x in range(0, self.chr_sizes[i], step) if x < s or x > e], 
                          ymin=ymin, ymax=ymax, linewidth=linewidth, color=color, zorder=zorder, **kwargs)
                
    
    # TODO rename yaxis to data_scaling or something or use offset and scale or offset and height
    def map_method(self, fun, data=None, chrom_col='chrom', yaxis=(0, 1), **kwargs):

        def method_not_found(): # just in case we dont have the function
            print('No Function '+fun+' Found!')

        grouped = data.groupby(chrom_col, observed=True)
        for chrom, group in grouped:
            ax = self.chr_axes[chrom]
            bottom, top = yaxis            
            scaled_y_lim = ax.get_ylim()
            if 'ylim' in kwargs:
                dy = -sub(*ylim)
            else:
                dy = -sub(*self.ylim)
            df = group.reset_index() # create independent dataframe
            df['y'] -= df.y.min()
            df['y'] /= df.y.max()
            df['y'] = df.y * ((top-bottom) * -sub(*scaled_y_lim)) / -sub(*self.ylim) + bottom /  -sub(*self.ylim) * -sub(*scaled_y_lim)
            x = df.x
            y = df.y
            if 'x' in kwargs: del kwargs['x']
            if 'y' in kwargs: del kwargs['y']                
            fun_name = fun.__name__
            method = getattr(ax, fun_name, method_not_found) 
            g = method(x, y, **kwargs)
            
            for ax in self.zoom_axes:
                scaled_y_lim = ax.get_ylim()
                if 'ylim' in kwargs:
                    dy = -sub(*ylim)
                else:
                    dy = -sub(*self.ylim)
                df = group.reset_index() # create independent dataframe
                df['y'] -= df.y.min()
                df['y'] /= df.y.max()
                df['y'] = df.y * ((top-bottom) * -sub(*scaled_y_lim)) / -sub(*self.ylim) + bottom /  -sub(*self.ylim) * -sub(*scaled_y_lim)
                x = df.x
                y = df.y
                if 'x' in kwargs: del kwargs['x']
                if 'y' in kwargs: del kwargs['y']   
                
                method = getattr(ax, fun_name, method_not_found) 
                g = method(x, y, **kwargs)
                
                try:
                    ax.get_legend().remove()
                except:
                    pass
            plt.xlabel('')
            plt.ylabel('')

    
    def map_fun(self, fun, data=None, chrom_col='chrom', yaxis=None, **kwargs):
            
        grouped = data.groupby(chrom_col, observed=True)
        for chrom, group in grouped:
            ax = self.chr_axes[chrom]
            scaled_y_lim = ax.get_ylim()
            if 'ylim' in kwargs:
                dy = -sub(*ylim)
            else:
                dy = -sub(*self.ylim)
            df = group.copy()
            if yaxis is None:
                df['y'] = df.y * -sub(*scaled_y_lim) / -sub(*self.ylim)
            else:
                bottom, top = yaxis
                df['y'] -= df.y.min()
                df['y'] /= df.y.max()
                df['y'] = df.y * ((top-bottom) * -sub(*scaled_y_lim)) / -sub(*self.ylim) + bottom /  -sub(*self.ylim) * -sub(*scaled_y_lim)
            g = fun(df, ax=ax, **kwargs)
            for ax in self.zoom_axes:
                scaled_y_lim = ax.get_ylim()
                if 'ylim' in kwargs:
                    dy = -sub(*ylim)
                else:
                    dy = -sub(*self.ylim)
                df = group.copy()
                if yaxis is None:
                    df['y'] = df.y * -sub(*scaled_y_lim) / -sub(*self.ylim)
                else:
                    bottom, top = yaxis
                    df['y'] -= df.y.min()
                    df['y'] /= df.y.max()
                    df['y'] = df.y * ((top-bottom) * -sub(*scaled_y_lim)) / -sub(*self.ylim) + bottom /  -sub(*self.ylim) * -sub(*scaled_y_lim)
                g = fun(df, ax=ax, **kwargs)
                try:
                    ax.get_legend().remove()
                except:
                    pass                               
            plt.xlabel('')
            plt.ylabel('')

    def legend(self, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, **kwargs):
        for ax in self.ax_list:
            handles, labels = ax.get_legend_handles_labels()
            if self.legend_handles:
                handles = self.legend_handles + handles
            ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=frameon, **kwargs)


    def _horizon(self, row, i, cut):
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

    def _horizonplot(self, df, y=None, ax=None,
                    cut=None, # float, takes precedence over quantile_span
                    quantile_span = None,
                    x='start',
                    beginzero=True, 
                    offset=0,
                    height=None,
                    colors = ['#CCE2DF', '#59A9A8', '#374E9B', 'midnightblue',
                              '#F2DE9A', '#DA8630', '#972428', 'darkred',
                              '#D3D3D3'],
                    **kwargs):
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
        col_iterators = zip(*(self._horizon(row, y, cut) for row in row_iter))
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
                    # plt.setp(fig.texts, text="") # hack to make y facet labels align...
                    ax.fill_between(
                        df2[x], 
                        df2[col_name]*scale+offset, 
                        y2=offset,
                        color=color,
                        linewidth=0,
                        capstyle='butt',
                    **kwargs)
    

    def horizon(self, data=None, chrom_col='chrom', y=None, 
                    cut=None, # float, takes precedence over quantile_span
                    quantile_span = None,
                    x='start',
                    beginzero=True, 
                    base=0,
                    height=1,
                    colors = ['#CCE2DF', '#59A9A8', '#374E9B', 'midnightblue',
                              '#F2DE9A', '#DA8630', '#972428', 'darkred',
                              '#D3D3D3'],
                    **kwargs):

        grouped = data.groupby(chrom_col, observed=True)
        for chrom, group in grouped:
            ax = self.chr_axes[chrom]
            
            df = group.reset_index() # make copy
            # scaled_y_lim = ax.get_ylim()
            # if 'ylim' in kwargs:
            #     dy = -sub(*ylim)
            # else:
            #     dy = -sub(*self.ylim)
            # df = group.copy()
            # if yaxis is None:
            #     df['y'] = df.y * -sub(*scaled_y_lim) / -sub(*self.ylim)
            # else:
            #     bottom, top = yaxis
            #     df['y'] -= df.y.min()
            #     df['y'] /= df.y.max()
            #     df['y'] = df.y * ((top-bottom) * -sub(*scaled_y_lim)) / -sub(*self.ylim) + bottom /  -sub(*self.ylim) * -sub(*scaled_y_lim)
            # g = fun(df, ax=ax, **kwargs)
            self._horizonplot(df, y=y, ax=ax,
                    cut=cut, # float, takes precedence over quantile_span
                    quantile_span = quantile_span,
                    x=x,
                    beginzero=beginzero, 
                    offset=self.map_y(base, ax),
                    height=self.map_y(height, ax),
                    colors = colors,
                    **kwargs)
                             
            for ax in self.zoom_axes:
                # scaled_y_lim = ax.get_ylim()
                # if 'ylim' in kwargs:
                #     dy = -sub(*ylim)
                # else:
                #     dy = -sub(*self.ylim)
                # df = group.copy()
                # if yaxis is None:
                #     df['y'] = df.y * -sub(*scaled_y_lim) / -sub(*self.ylim)
                # else:
                #     bottom, top = yaxis
                #     df['y'] -= df.y.min()
                #     df['y'] /= df.y.max()
                #     df['y'] = df.y * ((top-bottom) * -sub(*scaled_y_lim)) / -sub(*self.ylim) + bottom /  -sub(*self.ylim) * -sub(*scaled_y_lim)
                self._horizonplot(df, y=y, ax=ax,
                    cut=cut, # float, takes precedence over quantile_span
                    quantile_span = quantile_span,
                    x=x,
                    beginzero=beginzero, 
                    offset=self.map_y(base, ax),
                    height=self.map_y(height, ax),
                    colors = colors,
                    **kwargs)
                # try:
                #     ax.get_legend().remove()
                # except:
                #     pass                               
            # plt.xlabel('')
            # plt.ylabel('')

    


class ChromIdeogram(GenomeIdeogram):

    def __init__(self, chrom, axes_height_inches=1, axes_width_inches=12, hspace=0.3, 
                 ylim=(0, 1), zooms=[], wspace=None,
                 rel_font_height=0.05, assembly:str='hg38', min_stick_height=0.5):

        self.ideogram_base = None
        self.ideogram_height = None
        self.legend_handles = []
        self.height_ratios = [1]
        self.zooms = zooms
        self.zoom_axes = []
        self.end_padding = 300000
        self.min_stick_height = min_stick_height
        
        self.chr_names = [chrom]
        self.chr_sizes = [self.chrom_lengths[assembly][chrom] for chrom in self.chr_names]
        self.max_chrom_size = max(self.chr_sizes)
        nr_rows = 1
        self.aspect = axes_height_inches / axes_width_inches
        axes_width = self.max_chrom_size
        axes_height = self.aspect * self.max_chrom_size + (2 * self.end_padding)
        fig_height_inches = axes_height_inches
        point_size = 1/72
        self.ylim = ylim
        self.font_size = rel_font_height * axes_height_inches / point_size
        if zooms:
             fig_height_inches += (1 + hspace) * axes_height_inches
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
                )
                axs["main"].set(xlim=xlim)

                for i in range(len(zooms)):
                    self.zoom_effect(axs[f"zoom{i}"], axs["main"])
                    axs[f"zoom{i}"].set(xlim=zooms[i])

                    zoom_scaled_y_lim = self.scaled_y_lim(axs[f"zoom{i}"])
                    axs[f"zoom{i}"].set(ylim=zoom_scaled_y_lim)

                self.ax = axs["main"]
                self.zoom_axes = [axs[f"zoom{i}"] for i in range(len(zooms))]
                
                for i in range(len(self.zoom_axes)):
                    self.zoom_axes[i].sharey(self.zoom_axes[0])

                plt.subplots_adjust(wspace=wspace)

            else:        
                self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
            
            if hspace is not None:
                plt.subplots_adjust(hspace=hspace)
    
            plt.minorticks_on()
        

            nr_rows, nr_cols = 1, 1

            
            # gs = matplotlib.gridspec.GridSpec(nr_rows, 25)
            gs = matplotlib.gridspec.GridSpec(1, 1)
            gs.update(wspace=0, hspace=hspace) 

            self.ax.set_xlim(xlim)
            self.ax.set_ylim(scaled_y_lim)                

            # start, end = 0, self.chr_sizes[i]
            # ax.set_xlim(start, end)
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
    
        prop_patches = {**kwargs, 'ec': 'black', 'fc': 'lightgray', 'alpha': 0.3, 'linewidth': 0.5}
    
        c1, c2, bbox_patch1, bbox_patch2, p = self.connect_bbox(
            mybbox1, mybbox2,
            loc1a=2, loc2a=3, loc1b=1, loc2b=4,
            prop_lines=kwargs, prop_patches=prop_patches)
    
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
            gs = gridspec.GridSpec(row, 1, height_ratios=self.height_ratios, hspace=hspace)
            for i, ax in enumerate(self.fig.axes):
                ax.set_position(gs[i].get_position(self.fig))
                ax.set_subplotspec(gs[i])
            new_ax = self.fig.add_subplot(gs[row-1], sharex=ax)
            self.fig.set_figheight(self.fig.get_figheight()*sum(self.height_ratios)/sum(self.height_ratios[:-1]))
            new_ax.spines[['right', 'top']].set_visible(False)
            new_ax.xaxis.set_visible(False)
            # new_ax.spines[['bottom']].set_visible(False)
            new_axes.append(new_ax)

        new_axes[-1].xaxis.tick_bottom()
        new_axes[-1].xaxis.set_visible(True)
        # new_axes[-1].spines[['bottom']].set_visible(True)

        if nr_axes == 1:
            return new_axes[0]
        else:
            return new_axes


