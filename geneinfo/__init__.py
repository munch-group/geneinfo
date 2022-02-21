
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

mg = None

def connect_mygene(mygene_connection):
    global mg
    mg = mygene_connection


def geneinfo(query):
    
    if type(query) is str:
        query = [query]
        
    for gene in query:

        top_hit = mg.query(gene, species='human', scopes='hgnc',
                           fields='symbol,alias,name,type_of_gene,summary,genomic_pos,genomic_pos_hg19')['hits'][0]

        tmpl = "**Symbol:** **_{symbol}_** "

        if 'type_of_gene' in top_hit:
            tmpl += "({type_of_gene})"

        if 'alias' in top_hit:
            if type(top_hit['alias']) is str:
                top_hit['aliases'] = top_hit['alias']
            else:
                top_hit['aliases'] = ', '.join(top_hit['alias'])
            tmpl += " &nbsp; &nbsp; &nbsp; &nbsp; **Aliases:** {aliases}"
        tmpl += '  \n'

        if 'name' in top_hit:
            tmpl += '*{name}*  \n'

        if 'summary' in top_hit:
            tmpl += "**Summary:** {summary}  \n"

        if 'genomic_pos' in top_hit and 'genomic_pos_hg19' in top_hit:
            if type(top_hit['genomic_pos']) is list:
                top_hit['hg38'] = ', '.join(['{chr}:{start}-{end}'.format(**d) for d in top_hit['genomic_pos']])
            else:
                top_hit['hg38'] = '{chr}:{start}-{end}'.format(**top_hit['genomic_pos'])
            if type(top_hit['genomic_pos_hg19']) is list:
                top_hit['hg19'] = ', '.join(['{chr}:{start}-{end}'.format(**d) for d in top_hit['genomic_pos_hg19']])
            else:
                top_hit['hg19'] = '{chr}:{start}-{end}'.format(**top_hit['genomic_pos_hg19'])            
            tmpl += "**Genomic position:** {hg38} (hg38), {hg19} (hg19)  \n"

        tmpl += "[Gene card](https://www.genecards.org/cgi-bin/carddisp.pl?gene={symbol})  \n".format(**top_hit)

        tmpl += "\n\n ----"

        display(Markdown(tmpl.format(**top_hit)))


def get_genes(chrom, start, end, hg19=False):
    query = 'q={}:{}-{}'.format(chrom, start, end)
    for gene in mg.query(query, species='human', fetch_all=True):
        if 'symbol' not in gene:
            continue
        if hg19:
            tophit = mg.query(gene['symbol'], species='human', scopes='hgnc',
                           fields='exons_hg19,type_of_gene')['hits'][0]
            if 'exons_hg19' in tophit:
                first_transcript = tophit['exons_hg19'][0]
                yield gene['symbol'], first_transcript['txstart'], first_transcript['txend'], first_transcript['strand'], first_transcript['position'], tophit['type_of_gene']
        else:
            tophit = mg.query(gene['symbol'], species='human', scopes='hgnc',
                           fields='exons,type_of_gene')['hits'][0]            
            if 'exons' in tophit:
                first_transcript = tophit['exons'][0]
                yield gene['symbol'], first_transcript['txstart'], first_transcript['txend'], first_transcript['strand'], first_transcript['position'], tophit['type_of_gene']


def plot_gene(name, txstart, txend, strand, exons, gene_type, offset, line_width, font_size, ax, highlight=False, clip_on=True):

    if gene_type == 'protein-coding':
        if strand == 1:
            color='black'
        else:
            color='grey'
    elif gene_type == 'ncRNA':
        if strand == 1:
            color='red'
        else:
            color='pink'
    else:
        color='green'

    line = ax.plot([txstart, txend], [offset, offset], color=color, linewidth=line_width/10)

    # arrowpoints = np.arange(txstart, txend, 5000)
    # y = np.full_like(arrowpoints, offset)
    # arrowline = ax.plot(arrowpoints, y, color=color)[0]    
    # [add_arrow(arrowline, position=x) for x in arrowpoints[:-1]]

    for start, end in exons:
        line = ax.plot([start, end], [offset, offset], linewidth=line_width, color=color)
        line[0].set_solid_capstyle('butt')
        
    if highlight:
        ax.text(txstart, offset+.5, name, horizontalalignment='right', verticalalignment='center', fontsize=font_size, weight='bold', color='red', clip_on=clip_on)#, transform=ax.transAxes)
    else:
        ax.text(txstart, offset+.5, name, horizontalalignment='right', verticalalignment='center', fontsize=font_size, clip_on=clip_on)#, transform=ax.transAxes)


CACHE = dict()

def geneplot(chrom, start, end, highlight=[], hg19=False, figsize=None, clip_on=True):
    "Specifying hg19 gives gene coordintes in hg19, but give chrom, start and end are still assumed to be 38"
    
    global CACHE

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if (chrom, start, end, hg19) in CACHE:
        genes = CACHE[(chrom, start, end, hg19)]
    else:
        genes = list(get_genes(chrom, start, end, hg19=hg19))
        CACHE[(chrom, start, end, hg19)] = genes

    label_width = (end - start) / 5
        
    plotted_intervals = defaultdict(list)
    for name, txstart, txend, strand, exons, gene_type in genes:
        gene_interval = [txstart-label_width, txend]
        max_gene_rows = 200
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
            plotted_intervals[offset] = union(plotted_intervals[offset], [gene_interval])
        else:
            plotted_intervals[offset] = [gene_interval]
    
        from math import log10
        line_width, font_size = min(12, int(12 / log10(len(genes))))-2, min(12, int(12 / log10(len(genes))))
        plot_gene(name, txstart, txend, strand, exons, gene_type, 
                  offset, line_width, font_size, 
                  highlight=name in highlight,
                  ax=ax2, clip_on=clip_on)

    offset = max(plotted_intervals.keys())

    ax2.set_ylim(-1, offset+3)
    ax2.get_yaxis().set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    points = ax2.plot(txstart, offset+.5, 'o', ms=25, alpha=0, zorder=10)

    return ax1


def geneinfo_region(chrom, start, end, hg19=False):
    
    for gene in mg.query(f'q={chrom}:{start}-{end}', species='human', fetch_all=True):
        geneinfo(gene['symbol'])


if __name__ == "__main__":

    import mygene

    mg = mygene.MyGeneInfo()
    connect_mygene(mg)

    chrom, start, end = 'chr3', 49500000, 50600000
    ax = geneplot(chrom, start, end, figsize=(10, 5))
    ax.plot(np.linspace(start, end, 1000), np.random.random(1000), 'o')
    plt.savefig('tmp.pdf')

    ax = geneplot(chrom, start, end, figsize=(10, 5))
    ax.plot(np.linspace(start, end, 1000), np.random.random(1000), 'o')
    plt.savefig('tmp2.pdf')
