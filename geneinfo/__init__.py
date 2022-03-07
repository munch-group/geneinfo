
from IPython.display import Markdown, display, Image, SVG, HTML
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import requests
import io

from .intervals import *

def mygene_query(query, species='human', scopes='hgnc', fields='symbol,alias,name,type_of_gene,summary,genomic_pos,genomic_pos_hg19'):
    api_url = f"https://mygene.info/v3/query?content-type=appliation/x-www-form-urlencoded;q={query};scopes={scopes};species={species};fields={fields}"
    response = requests.get(api_url)
    assert response.ok, response.status_code
    return response.json()

def geneinfo(query):
    
    if type(query) is str:
        query = [query]
        
    for gene in query:

        top_hit = mygene_query(gene, species='human', scopes='hgnc',
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


def ensembl_get_features(chrom, window_start, window_end, features=['gene', 'exon'], assembly=None, species='human'):
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    window_start, window_end = int(window_start), int(window_end)
    genes = {}    
    for start in range(window_start, window_end, 500000):
        end = min(start+500000, window_end)
        param_str = ';'.join([f"feature={f}" for f in features])
        if assembly:
            api_url = f"https://{assembly.lower()}.rest.ensembl.org/overlap/region/{species}/{chrom}:{start}-{end}?{param_str}"
        else:
            api_url = f"http://rest.ensembl.org/overlap/region/{species}/{chrom}:{start}-{end}?{param_str}"
        response = requests.get(api_url, headers={'content-type': 'application/json'})
        assert response.ok, response.status_code
        data = response.json()

        for gene in data:
            genes[gene['id']] = gene
            
    return genes
  
def ensembl_get_genes(chrom, window_start, window_end, assembly=None, species='human'):
    
    gene_info = ensembl_get_features(chrom, window_start, window_end, features=['gene'], assembly=assembly, species=species)
    exon_info = ensembl_get_features(chrom, window_start, window_end, features=['exon'], assembly=assembly, species=species)

    exons = defaultdict(list)
    for key, info in exon_info.items():
        exons[info['Parent']].append((info['start'], info['end']))

    for key in gene_info:
        gene_info[key]['exons'] = []
        if 'canonical_transcript' in gene_info[key]:
            transcript = gene_info[key]['canonical_transcript'].split('.')[0]
            if transcript in exons:
                gene_info[key]['exons'] = sorted(exons[transcript])

    return gene_info

def get_genes(chrom, window_start, window_end, hg19=False, species='human'):

    if hg19:
        assembly='GRCh37'
    else:
        assembly=None

    genes = []
    gene_info = ensembl_get_genes(chrom, window_start, window_end, assembly=assembly, species=species)
    for gene in gene_info.values():
        if 'external_name' in gene:
            name = gene['external_name']
        else:
            name = gene['id']
        genes.append((name, gene['start'], gene['end'], gene['strand'], gene['exons'], gene['biotype']))
    return genes

def get_genes_dataframe(chrom, start, end, hg19=False):
    try:
        import pandas as pd
    except ImportError:
        print("pandas must be installed to return data frame")
        return
    genes = get_genes(chrom, start, end, hg19=hg19)
    return pd.DataFrame().from_records([x[:4] for x in genes], columns=['name', 'start', 'end', 'strand'])

def plot_gene(name, txstart, txend, strand, exons, gene_type, offset, line_width, font_size, ax, highlight=False, clip_on=True):

    if gene_type == 'protein_coding':
        color='black'
    elif gene_type == 'ncrna':
        color='blue'
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
        ax.text(txstart, offset-.5, name, horizontalalignment='right', verticalalignment='center', fontsize=font_size, weight='bold', color='red', clip_on=clip_on)#, transform=ax.transAxes)
    else:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', verticalalignment='center', fontsize=font_size, clip_on=clip_on)#, transform=ax.transAxes)


CACHE = dict()

def geneplot(chrom, start, end, highlight=[], hg19=False, only_protein_coding=False, figsize=None, clip_on=True):
    
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
        if gene_type != 'protein_coding' and only_protein_coding:
            continue
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
    ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    points = ax2.plot(txstart, offset+.5, 'o', ms=25, alpha=0, zorder=10)

    ax1.set_xlim(ax2.get_xlim())

    return ax1


def geneinfo_region(chrom, start, end, hg19=False):
    
    for gene in mg.query(f'q={chrom}:{start}-{end}', species='human', fetch_all=True):
        geneinfo(gene['symbol'])


def map_interval(chrom, start, end, strand, map_from, map_to, species='human'):
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    start, end = int(start), int(end)    
    api_url = f"http://rest.ensembl.org/map/{species}/{map_from}/{chrom}:{start}..{end}:{strand}/{map_to}"
    params = {'content-type': 'application/json'}
    response = requests.get(api_url, data=params)
    assert response.ok
    #null = '' # json may include 'null' variables 
    return response.json()#eval(response.content.decode())


def _get_string_ids(my_genes):
    string_api_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"
    params = {
        "identifiers" : "\r".join(my_genes), # your protein list
        "species" : 9606, # species NCBI identifier 
        "limit" : 1, # only one (best) identifier per input protein
        "echo_query" : 1, # see your input identifiers in the output
        "caller_identity" : "geneinfo" # your app name
    }
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)
    string_identifiers = []
    for line in results.text.strip().split("\n"):
        print(line)
        l = line.split("\t")
        input_identifier, string_identifier = l[0], l[2]
        string_identifiers.append(string_identifier)
    return string_identifiers

def show_string_network(my_genes, nodes=10):
    if type(my_genes) is str:
        my_genes = list(my_genes)    
    string_identifiers = _get_string_ids(my_genes)
    string_api_url = "https://string-db.org/api"
    # string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "svg"
    method = "network"
    request_url = "/".join([string_api_url, output_format, method])
    params = {
        "identifiers" : "\r".join(string_identifiers), # your proteins
        "species" : 9606, # species NCBI identifier 
        "add_white_nodes": nodes, # add 15 white nodes to my protein 
        "network_flavor": "confidence", # show confidence links
        "caller_identity" : "geneinfo" # your app name
    }
    response = requests.post(request_url, data=params)
    file_name = "network.svg"
    with open(file_name, 'wb') as fh:
        fh.write(response.content)
    return SVG('network.svg') 

def string_network_table(my_genes, nodes=10):
    if type(my_genes) is str:
        my_genes = list(my_genes)
    string_api_url = "https://string-db.org/api"
    output_format = "tsv"
    method = "network"
    request_url = "/".join([string_api_url, output_format, method])
    params = {
        "identifiers" : "\r".join(my_genes), # your proteins
        "species" : 9606, # species NCBI identifier 
        "add_white_nodes": nodes, # add 15 white nodes to my protein 
        "network_flavor": "confidence", # show confidence links
        "caller_identity" : "geneinfo" # your app name
    }
    response = requests.post(request_url, data=params)
    return pd.read_table(io.StringIO(response.content.decode()))


if __name__ == "__main__":

    pass
    # import mygene

    # mg = mygene.MyGeneInfo()
    # connect_mygene(mg)

    # chrom, start, end = 'chr3', 49500000, 50600000
    # ax = geneplot(chrom, start, end, figsize=(10, 5))
    # ax.plot(np.linspace(start, end, 1000), np.random.random(1000), 'o')
    # plt.savefig('tmp.pdf')

    # ax = geneplot(chrom, start, end, figsize=(10, 5))
    # ax.plot(np.linspace(start, end, 1000), np.random.random(1000), 'o')
    # plt.savefig('tmp2.pdf')
