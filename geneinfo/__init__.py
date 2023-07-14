
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
import subprocess
import pandas as pd
from math import log10
import shutil

import matplotlib
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

# sys.path.append('.')

# if this runs on a cluster node we need to set proxies to access external resources:
if re.match(r's\d+n\d+', os.environ['HOSTNAME']):
    os.environ['http_proxy'] = 'http://proxy-default:3128'
    os.environ['https_proxy'] = 'http://proxy-default:3128'
    os.environ['ftp_proxy'] = 'http://proxy-default:3128'
    os.environ['ftps_proxy'] = 'http://proxy-default:3128'

CACHE = dict()

class NotFound(Exception):
    """query returned no result"""
    pass


def ensembl_id(name, species='homo_sapiens'):
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/symbol/{species}/{name}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
      r.raise_for_status()
    decoded = r.json()
    ensembl_ids = [x['id'] for x in decoded if x['type'] == 'gene']
    if not len(ensembl_ids) == 1:
        raise NotFound
    return ensembl_ids[0]

def ensembl2symbol(ensembl_id):
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/id/{ensembl_id}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
    decoded = r.json()
    symbols = [x['display_id'] for x in decoded if x['dbname'] == 'HGNC']
    if not len(symbols) == 1:
        raise NotFound
    return symbols[0]

def hgcn_symbol(name):
    if type(name) is list or type(name) is set:
        return [ensembl2symbol(ensembl_id(n)) for n in name]
    else:
        return ensembl2symbol(ensembl_id(name))

def ensembl2ncbi(ensembl_id):
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/id/{ensembl_id}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
      r.raise_for_status()
    decoded = r.json()
    ids = [x['primary_id'] for x in decoded if x['dbname'] == 'EntrezGene']
    if not len(ids) == 1:
        raise NotFound
    return int(ids[0])

def mygene_get_gene_info(query, species='human', scopes='hgnc', fields='symbol,alias,name,type_of_gene,summary,genomic_pos,genomic_pos_hg19'):
    api_url = f"https://mygene.info/v3/query?q={query}&scopes={scopes}&species={species}&fields={fields}"    
    response = requests.get(api_url)
    if not response.ok:
        response.raise_for_status()
    result = response.json()
    if 'hits' in result:
        for hit in result['hits']:
            if (type(query) is not int and hit['symbol'].upper() == query.upper()) or hit['_id'] == str(query):
                return hit
    print(f"Gene not found: {query}", file=sys.stderr)


def gene_info(query, species='human', scopes='hgnc'):
    
    if type(query) is not list:
        query = [query]
        
    for gene in query:

        for i in range(3):
            try:
                top_hit = mygene_get_gene_info(gene, species=species, scopes=scopes,
                                fields='symbol,alias,name,type_of_gene,summary,genomic_pos,genomic_pos_hg19')
            except KeyError:
                continue
            else:
                break


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


def _ensembl_get_features_region(chrom, window_start, window_end, features=['gene', 'exon'], assembly=None, species='homo_sapiens'):
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

        if not response.ok:
            response.raise_for_status()
        data = response.json()

        for gene in data:
            genes[gene['id']] = gene
            
    return genes


def ensembl_get_gene_info_by_symbol(symbols, assembly=None, species='homo_sapiens'):

    if type(symbols) is not list:
        symbols = [symbols]

    if assembly:
        server = f"https://{assembly.lower()}.rest.ensembl.org"
    else:
        server = "https://rest.ensembl.org"
    ext = f"/lookup/symbol/{species}"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    r = requests.post(server+ext, headers=headers, data=f'{{ "symbols": {json.dumps(symbols)} }}')
    if not r.ok:
        r.raise_for_status()
    return r.json()


def ensembl_get_genes_region(chrom, window_start, window_end, assembly=None, species='homo_sapiens'):
    
    gene_info = _ensembl_get_features_region(chrom, window_start, window_end, features=['gene'], assembly=assembly, species=species)
    exon_info = _ensembl_get_features_region(chrom, window_start, window_end, features=['exon'], assembly=assembly, species=species)

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


def get_genes_region(chrom, window_start, window_end, assembly='GRCh38', db='ncbiRefSeq'):

    
    api_url = f'https://api.genome.ucsc.edu/getData/track'
    params = {'track': db,
              'genome': assembly,
              'chrom': chrom,
              'start': window_start,
              'end': window_end
              }
    response = requests.get(api_url, data=params)
    if not response.ok:
        response.raise_for_status()

    genes = []
    for gene in response.json()[db]:
        exon_starts = [int(x) for x in gene['exonStarts'].split(',') if x]
        exon_ends = [int(x) for x in gene['exonEnds'].split(',') if x]
        exons = list(zip(exon_starts, exon_ends))
        genes.append((gene['name2'], gene['txStart'], gene['txEnd'], gene['strand'], exons))

    return genes


def gene_info_region(chrom, window_start, window_end, assembly='GRCh38', db='ncbiRefSeq'):

    for gene in get_genes_region(chrom, window_start, window_end, assembly, db):    
        gene_info(gene[0])


def get_genes_region_dataframe(chrom, start, end, assembly='GRCh38', db='ncbiRefSeq'):
    try:
        import pandas as pd
    except ImportError:
        print("pandas must be installed to return data frame")
        return
    genes = get_genes_region(chrom, start, end, assembly, db)
    return pd.DataFrame().from_records([x[:4] for x in genes], columns=['name', 'start', 'end', 'strand'])


def _plot_gene(name, txstart, txend, strand, exons, offset, line_width, min_visible_width, font_size, ax, highlight=False, clip_on=True):

    color='black'

    line = ax.plot([txstart, txend], [offset, offset], color=color, linewidth=line_width/5, alpha=0.5)
    line[0].set_solid_capstyle('butt')

    for start, end in exons:
        end = max(start+min_visible_width, end)
        line = ax.plot([start, end], [offset, offset], linewidth=line_width, color=color)
        line[0].set_solid_capstyle('butt')
        
    if highlight is True:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', verticalalignment='center', 
            fontsize=font_size, clip_on=clip_on,
            weight='bold', color='red')
    elif type(highlight) is dict:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', verticalalignment='center',
            fontsize=font_size, clip_on=clip_on, 
            **highlight)
    else:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', verticalalignment='center', 
            fontsize=font_size, color=color, clip_on=clip_on)


def gene_plot(chrom, start, end, assembly, highlight=[], db='ncbiRefSeq', 
                collapse_splice_var=True, hard_limits=False, exact_exons=False, 
                figsize=None, aspect=1, despine=False, clip_on=True, gene_density=60, font_size=None, return_axes=1):
    
    global CACHE

    fig, axes = plt.subplots(return_axes+1, 1, figsize=figsize, sharex='col', 
                                    sharey='row', gridspec_kw={'height_ratios': [1/return_axes]*return_axes + [aspect]})
    plt.subplots_adjust(wspace=0, hspace=0.15)

    if (chrom, start, end, assembly) in CACHE:
        genes = CACHE[(chrom, start, end, assembly)]
    else:
        genes = list(get_genes_region(chrom, start, end, assembly, db))
        CACHE[(chrom, start, end, assembly)] = genes

    if collapse_splice_var:
        d = {}
        for name, txstart, txend, strand, exons in genes:
            if name not in d:
                d[name] = [name, txstart, txend, strand, set(exons)]
            else:
                d[name][-1].update(exons)
        genes = d.values()


    line_width = max(6, int(50 / log10(end - start)))-2
    if font_size is None:
        font_size = max(6, int(50 / log10(end - start)))
    label_width = font_size * (end - start) / gene_density
    if exact_exons:
        min_visible_exon_width = 0
    else:
        min_visible_exon_width = (end - start) / 1000
        
    plotted_intervals = defaultdict(list)
    for name, txstart, txend, strand, exons in genes:

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
            plotted_intervals[offset] = union(plotted_intervals[offset], [gene_interval])
        else:
            plotted_intervals[offset] = [gene_interval]

        if type(highlight) is list or type(highlight) is set:
            hl = name in highlight
        elif type(highlight) is dict or type(highlight) is defaultdict:
            hl = highlight[name]
        else:
            hl = None

        _plot_gene(name, txstart, txend, strand, exons, 
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


##################################################################################
# Map between assembly coordinates
##################################################################################

def map_interval(chrom, start, end, strand, map_from, map_to, species='homo_sapiens'):
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    start, end = int(start), int(end)    
    api_url = f"http://rest.ensembl.org/map/{species}/{map_from}/{chrom}:{start}..{end}:{strand}/{map_to}"
    params = {'content-type': 'application/json'}
    response = requests.get(api_url, data=params)
    if not response.ok:
        response.raise_for_status()
    #null = '' # json may include 'null' variables 
    return response.json()#eval(response.content.decode())


##################################################################################
# STRING networks
##################################################################################

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
    if not results.ok:
        results.raise_for_status()    
    string_identifiers = []
    for line in results.text.strip().split("\n"):
        l = line.split("\t")
        input_identifier, string_identifier = l[0], l[2]
        string_identifiers.append(string_identifier)
    return string_identifiers


def show_string_network(my_genes, nodes=10):

    if not os.path.exists('geneinfo_cache'): os.makedirs('geneinfo_cache')

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
    if not response.ok:
        response.raise_for_status()    
    file_name = "geneinfo_cache/network.svg"
    with open(file_name, 'wb') as fh:
        fh.write(response.content)
    return SVG('geneinfo_cache/network.svg') 


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
    if not response.ok:
        response.raise_for_status()    
    return pd.read_table(io.StringIO(response.content.decode()))


##################################################################################
# Gene Ontology
##################################################################################

def email(email_address):
    Entrez.email = email_address

def _assert_entrez_email():
    if not Entrez.email:
        print("""Please provide your email for Entrez queries:

import geneinfo as gi
gi.email("youremail@address.com)
""", file=sys.stderr)
        return


def download_ncbi_associations(prt=sys.stdout):

    if not os.path.exists('geneinfo_cache'): os.makedirs('geneinfo_cache')

    if not os.path.exists('geneinfo_cache/gene2go'):
        process = subprocess.Popen(['wget', '-nv', '-O', 'geneinfo_cache/gene2go.gz', 'ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz'],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode(), file=prt)
        print(stderr.decode(), file=prt)
        assert not process.returncode

        process = subprocess.Popen(['gzip', '-f', '-d', 'geneinfo_cache/gene2go.gz'],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode(), file=prt)
        print(stderr.decode(), file=prt)    
        assert not process.returncode
    return 'geneinfo_cache/gene2go'


def download_and_move_go_basic_obo(prt):  

    if not os.path.exists('geneinfo_cache'): os.makedirs('geneinfo_cache')

    if not os.path.exists('geneinfo_cache/go-basic.obo'):
        obo_fname = download_go_basic_obo(prt=prt)
        shutil.move('go-basic.obo', 'geneinfo_cache/go-basic.obo')
    return 'geneinfo_cache/go-basic.obo'


def _fetch_ids_to_file(id_list, output_file_name):

    with open(output_file_name, 'w') as f:
        header = ['tax_id', 'Org_name', 'GeneID', 'CurrentID', 'Status', 'Symbol', 'Aliases', 
                'description', 'other_designations', 'map_location', 'chromosome', 
                'genomic_nucleotide_accession.version', 'start_position_on_the_genomic_accession', 
                'end_position_on_the_genomic_accession', 'orientation', 'exon_count', 'OMIM']
        print(*header, sep='\t', file=f)

        nr_genes_no_coordinates = 0

        batch_size = 2000
        for i in range(0, len(id_list), batch_size):
            to_fetch = id_list[i:i+batch_size]
            handle = Entrez.esummary(db="gene", id=",".join(to_fetch), retmax=batch_size)
            entry = Entrez.read(handle)
            docsums = entry['DocumentSummarySet']['DocumentSummary']
            for doc in docsums:

                # try:
                #     print(doc['Organism']['TaxID'], doc['Organism']['ScientificName'], doc.attributes['uid'], 
                #             doc['CurrentID'], 
                #             # doc['Status'],
                #             'live',
                #             doc['Name'], doc['OtherAliases'], doc['Description'], doc['OtherDesignations'],
                #             doc['MapLocation'], doc['Chromosome'], 
                #             doc['GenomicInfo'][0]['ChrAccVer'], doc['GenomicInfo'][0]['ChrStart'], doc['GenomicInfo'][0]['ChrStop'],
                #             'notspecified', doc['GenomicInfo'][0]['ExonCount'], '',
                #             sep='\t', file=f)
                # except Exception as e:
                #     print(doc['Name'], e)
                #     pass


                if doc['GenomicInfo']:
                    ver, start, stop, exon_count = doc['GenomicInfo'][0]['ChrAccVer'], doc['GenomicInfo'][0]['ChrStart'], \
                        doc['GenomicInfo'][0]['ChrStop'], doc['GenomicInfo'][0]['ExonCount']
                else:
                    ver, start, stop, exon_count = 'unknown', pd.NA, pd.NA, pd.NA
                    nr_genes_no_coordinates += 1

                print(doc['Organism']['TaxID'], doc['Organism']['ScientificName'], doc.attributes['uid'], 
                        doc['CurrentID'], 
                        'live',
                        doc['Name'], doc['OtherAliases'], doc['Description'], doc['OtherDesignations'],
                        doc['MapLocation'], doc['Chromosome'], 
                        ver, start, stop,
                        'notspecified', exon_count, '',
                        sep='\t', file=f)

    # print(f"NB: {nr_genes_no_coordinates} background genes are without genomic coordinates", file=sys.stderr)



def fetch_background_genes(taxid=9606):
    
    _assert_entrez_email()

    if not os.path.exists('geneinfo_cache'): os.makedirs('geneinfo_cache')

    output_file_name = f'geneinfo_cache/{taxid}_protein_genes.txt'        
    handle = Entrez.esearch(db="gene", term=f'{taxid}[Taxonomy ID] AND alive[property] AND genetype protein coding[Properties]', retmax="1000000")
    records = Entrez.read(handle)
    id_list = records["IdList"]

    _fetch_ids_to_file(id_list, output_file_name)

    # write mappings between symbol and ncbi id
    symbol2ncbi_file = f'geneinfo_cache/{taxid}_symbol2ncbi.h5'
    df = pd.read_table(output_file_name)
    df = df.loc[:, ['GeneID', 'Symbol']]
    df.set_index('Symbol').GeneID.to_hdf(symbol2ncbi_file, key='symbol2ncbi')
    df.set_index('GeneID').Symbol.to_hdf(symbol2ncbi_file, key='ncbi2symbol')


def _cached_symbol2ncbi(symbols, taxid=9606):

    symbol2ncbi_file = f'geneinfo_cache/{taxid}_symbol2ncbi.h5'
    symbol2ncbi = pd.read_hdf(symbol2ncbi_file, 'symbol2ncbi')
    try:    
        return symbol2ncbi.loc[symbols].tolist() 
    except KeyError:
        geneids = []
        for symbol in symbols:
            try:
                geneids.append(symbol2ncbi.loc[symbol])
            except KeyError:
                try:
                    ncbi_id = hgcn_symbol(symbol)
                    if ncbi_id not in symbol2ncbi.index:
                        print(ncbi_id, 'not in symbol2ncbi index')
                        raise NotFound
                        
                        
                    geneids.append(symbol2ncbi.loc[ncbi_id])
                    # geneids.append(ensembl2ncbi(ensembl_id(symbol)))                    
                except NotFound:
                    print(f'Could not map gene symbol "{symbol}" to ncbi id', file=sys.stderr)
        return geneids


def _cached_ncbi2symbol(geneids, taxid=9606):

    symbol2ncbi_file = f'geneinfo_cache/{taxid}_symbol2ncbi.h5'
    symbol2ncbi = pd.read_hdf(symbol2ncbi_file, 'ncbi2symbol')
    try:
        return symbol2ncbi.loc[geneids].tolist()
    except KeyError:
        symbols = []
        for geneid in geneids:
            try:
                symbols.append(ncbi2symbol.loc[geneid])
            except KeyError:
                print(f'Could not map ncbi id "{geneid}" to gene symbol', file=sys.stderr)
        return symbols


def _tidy_taxid(taxid):
    try:
        taxid = int(taxid)
    except ValueError:
        handle = Entrez.esearch(db="taxonomy", term=f'"{taxid}"[Scientific Name]')
        id_list = Entrez.read(handle)['IdList']
        if id_list:
            taxid = int(id_list[0])
        else:
            print(f'Could not find taxonomy id for "{taxid}"')
    return taxid  
    

def symbols_protein_coding(taxid=9606):
    fetch_background_genes(taxid=taxid)
    symbol2ncbi_file = f'geneinfo_cache/{taxid}_symbol2ncbi.h5'
    symbol2ncbi = pd.read_hdf(symbol2ncbi_file, 'ncbi2symbol')
    return symbol2ncbi.tolist()


def get_terms_for_go_regex(regex, taxid=9606, add_children=False):

    taxid = _tidy_taxid(taxid)
        
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_and_move_go_basic_obo(prt=null)

        gene2go = download_ncbi_associations(prt=null)

        objanno = Gene2GoReader("geneinfo_cache/gene2go", taxids=[taxid], prt=null)
        go2geneids = objanno.get_id2gos(namespace='*', go2geneids=True, prt=null)
        srchhelp = GoSearch("geneinfo_cache/go-basic.obo", go2items=go2geneids, log=null)

        results_all = re.compile(r'({})'.format(regex), flags=re.IGNORECASE)
        results_not = re.compile(r'({}).independent'.format(regex), flags=re.IGNORECASE)

        gos_all = srchhelp.get_matching_gos(results_all, prt=null)
        gos_no = srchhelp.get_matching_gos(results_not, gos=gos_all)
        gos = gos_all.difference(gos_no)
        if add_children:
            gos = srchhelp.add_children_gos(gos)

        return list(gos)


def get_genes_for_go_regex(regex, taxid=9606):

    _assert_entrez_email()

    taxid = _tidy_taxid(taxid)
 
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        gos_all_with_children = get_terms_for_go_regex(regex, taxid=taxid, add_children=True)

        objanno = Gene2GoReader("geneinfo_cache/gene2go", taxids=[taxid], prt=null)
        go2geneids = objanno.get_id2gos(namespace='*', go2geneids=True, prt=null)
        srchhelp = GoSearch("geneinfo_cache/go-basic.obo", go2items=go2geneids, log=null)
        geneids = srchhelp.get_items(gos_all_with_children)

        ncbi_tsv = f'geneinfo_cache/{taxid}_protein_genes.txt'
        if not os.path.exists(ncbi_tsv):
            fetch_background_genes(taxid)

        output_py = f'geneinfo_cache/{taxid}_protein_genes.py'
        ncbi_tsv_to_py(ncbi_tsv, output_py, prt=null)
        
        protein_genes = importlib.import_module(output_py.replace('.py', '').replace('/', '.'))
        GENEID2NT = protein_genes.GENEID2NT

    fetch_ids = geneids

    fetch_ids = list(map(str, fetch_ids))
    records = []
    found = []
    batch_size = 2000
    for i in range(0, len(fetch_ids), batch_size):
        to_fetch = fetch_ids[i:i+batch_size]
        handle = Entrez.esummary(db="gene", id=",".join(to_fetch), retmax=batch_size)
        entry = Entrez.read(handle)
        docsums = entry['DocumentSummarySet']['DocumentSummary']
        for doc in docsums:
            try:
                chrom_pos = (doc['Chromosome'], doc['GenomicInfo'][0]['ChrStart'], doc['GenomicInfo'][0]['ChrStop'])
            except:
                print(f"WARNING: missing chromosome coordinates for {doc['Name']} are listed as pandas.NA", file=sys.stderr)
                chrom_pos = (pd.NA, pd.NA, pd.NA)
            records.append((doc['Name'], doc['Description'], *chrom_pos))
            found.append(str(doc.attributes['uid']))
    missing = set(fetch_ids).difference(set(found))

    df = pd.DataFrame().from_records(records, columns=['symbol', 'name', 'chrom', 'start', 'end'])

    return df.sort_values(by='start').reset_index(drop=True)

    
def get_genes_for_go_terms(terms, taxid=9606):
    
    if type(terms) is not list:
        terms = [terms]

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_and_move_go_basic_obo(prt=null)
        gene2go = download_ncbi_associations(prt=null)
        objanno = Gene2GoReader("geneinfo_cache/gene2go", taxids=[taxid], prt=null)
        go2geneids = objanno.get_id2gos(namespace='*', go2geneids=True, prt=null)
        srchhelp = GoSearch("geneinfo_cache/go-basic.obo", go2items=go2geneids, log=null)

        geneids = srchhelp.get_items(terms)  

        ncbi_tsv = f'geneinfo_cache/{taxid}_protein_genes.txt' 
        if not os.path.exists(ncbi_tsv):
            fetch_background_genes(taxid)

        output_py = f'geneinfo_cache/{taxid}_protein_genes.py'
        ncbi_tsv_to_py(ncbi_tsv, output_py, prt=null)

    protein_genes = importlib.import_module(output_py.replace('.py', '').replace('/', '.'))
    GENEID2NT = protein_genes.GENEID2NT

    fetch_ids = geneids

    fetch_ids = list(map(str, fetch_ids))
    records = []
    found = []
    batch_size = 2000
    for i in range(0, len(fetch_ids), batch_size):
        to_fetch = fetch_ids[i:i+batch_size]
        handle = Entrez.esummary(db="gene", id=",".join(to_fetch), retmax=batch_size)
        entry = Entrez.read(handle)
        docsums = entry['DocumentSummarySet']['DocumentSummary']
        for doc in docsums:
            try:
                chrom_pos = (doc['Chromosome'], doc['GenomicInfo'][0]['ChrStart'], doc['GenomicInfo'][0]['ChrStop'])
            except:
                print(f"WARNING: missing chromosome coordinates for {doc['Name']} are listed as pandas.NA", file=sys.stderr)
                chrom_pos = (pd.NA, pd.NA, pd.NA)
            records.append((doc['Name'], doc['Description'], *chrom_pos))
            found.append(str(doc.attributes['uid']))
    missing = set(fetch_ids).difference(set(found))

    df = pd.DataFrame().from_records(records, columns=['symbol', 'name', 'chrom', 'start', 'end'])

    return df.sort_values(by='start').reset_index(drop=True)


def go_annotation_table(taxid=9606):

    _assert_entrez_email()

    try:
        taxid = int(taxid)
    except ValueError:
        handle = Entrez.esearch(db="taxonomy", term=f'"{taxid}"[Scientific Name]')
        taxid = int(Entrez.read(handle)['IdList'][0])
        
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_and_move_go_basic_obo(prt=null)

        gene2go = download_ncbi_associations(prt=null)
        
    df = pd.read_table(gene2go, sep='\t')
    df.rename(columns={'#tax_id': 'taxid'}, inplace=True)
    return df.loc[df['taxid'] == taxid]


def gene_annotation_table(taxid=9606):

    ncbi_tsv = f'geneinfo_cache/{taxid}_protein_genes.txt'
    if not os.path.exists(ncbi_tsv):
        fetch_background_genes(taxid)
    df = pd.read_table(ncbi_tsv)
    df.rename(columns={'tax_id': 'taxid'}, inplace=True)
    return df.loc[df['taxid'] == taxid]


def get_go_terms_for_genes(genes, taxid=9606, evidence=None):
    
    go_df = go_annotation_table(taxid)
    genes_df = gene_annotation_table(taxid)
    gene_ids = genes_df.loc[genes_df.Symbol.isin(genes)].GeneID

    df = go_df.loc[go_df.GeneID.isin(gene_ids)]
    if len(df.index) and evidence is not None:
        df = df.loc[df.Evidence.isin(evidence)]
        
    return list(sorted(df.GO_ID.unique().tolist()))

    
def show_go_dag_for_terms(terms, add_relationships=True):
    
    if type(terms) is pd.core.series.Series:
        terms = terms.tolist()

    if not terms:
        return

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_and_move_go_basic_obo(prt=null)
        
        file_gene2go = download_ncbi_associations(prt=null)

        if add_relationships:
            optional_attrs=['relationship', 'def']
        else:
            optional_attrs=['def']
        obodag = GODag("geneinfo_cache/go-basic.obo", optional_attrs=optional_attrs, prt=null)

        gosubdag = GoSubDag(terms, obodag, relationships=add_relationships) 
        GoSubDagPlot(gosubdag).plt_dag('geneinfo_cache/plot.png')

    return Image('geneinfo_cache/plot.png')    

# def show_go_dag_for_terms(terms, add_relationships=True):

#     with open(os.devnull, 'w') as null, redirect_stdout(null):
#         if add_relationships:
#             optional_attrs=['relationship', 'def']
#         else:
#             optional_attrs=['def']
#         obodag = GODag("go-basic.obo", optional_attrs=optional_attrs, prt=null)
#         plot_gos('plot.png', terms, obodag)
#     return Image('plot.png')  


# https://github.com/tanghaibao/goatools/blob/main/notebooks/goea_nbt3102_group_results.ipynb


def show_go_dag_for_gene(gene, taxid=9606, evidence=None, add_relationships=True):
    # evidence codes: http://geneontology.org/docs/guide-go-evidence-codes/
    go_terms = get_go_terms_for_genes([gene], taxid=taxid, evidence=evidence)
    if not go_terms:
        print('No GO terms to show', file=sys.stderr)
        return
    return show_go_dag_for_terms(go_terms, add_relationships=add_relationships)

    
def show_go_evidence_codes():   

    s = """
**Experimental evidence codes:** <br>
Inferred from Experiment (EXP) <br>
Inferred from Direct Assay (IDA) <br>
Inferred from Physical Interaction (IPI) <br>
Inferred from Mutant Phenotype (IMP) <br>
Inferred from Genetic Interaction (IGI) <br>
Inferred from Expression Pattern (IEP) <br>
Inferred from High Throughput Experiment (HTP) <br>
Inferred from High Throughput Direct Assay (HDA) <br>
Inferred from High Throughput Mutant Phenotype (HMP) <br>
Inferred from High Throughput Genetic Interaction (HGI) <br>
Inferred from High Throughput Expression Pattern (HEP) 

**Phylogenetically-inferred annotations:** <br>
Inferred from Biological aspect of Ancestor (IBA) <br>
Inferred from Biological aspect of Descendant (IBD) <br>
Inferred from Key Residues (IKR) <br>
Inferred from Rapid Divergence (IRD)

**Computational analysis evidence codes** <br>
Inferred from Sequence or structural Similarity (ISS) <br>
Inferred from Sequence Orthology (ISO) <br>
Inferred from Sequence Alignment (ISA) <br>
Inferred from Sequence Model (ISM) <br>
Inferred from Genomic Context (IGC) <br>
Inferred from Reviewed Computational Analysis (RCA)

**Author statement evidence codes:** <br>
Traceable Author Statement (TAS) <br>
Non-traceable Author Statement (NAS)

**Curator statement evidence codes:** <br>
Inferred by Curator (IC) <br>
No biological Data available (ND)

**Electronic annotation evidence code:** <br>
Inferred from Electronic Annotation (IEA)

"""    
    display(Markdown(s))  
                    
        
def  _write_go_hdf():

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        if not os.path.exists('geneinfo_cache/go-basic.h5'):

            # Get http://geneontology.org/ontology/go-basic.obo
            obo_fname = download_and_move_go_basic_obo(prt=null)

            # Download Associations, if necessary
            # Get ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
            file_gene2go = download_ncbi_associations(prt=null)

            # r = OBOReader(obo_file='geneinfo_cache/go-basic.obo', optional_attrs=['def'])
            rows = [(e.id, e.name, e.defn) for e in OBOReader(obo_file='geneinfo_cache/go-basic.obo', optional_attrs=['def'])]
            df = pd.DataFrame().from_records(rows, columns=['goterm', 'goname', 'description'])
            df.to_hdf('geneinfo_cache/go-basic.h5', key='df', format='table', data_columns=['goterm', 'goname'])


def go_term2name(term):

    _write_go_hdf()
    with pd.HDFStore('geneinfo_cache/go-basic.h5', 'r') as store:
        entry = store.select("df", "goterm == %r" % term).iloc[0]
    return entry.goterm


def go_name2term(name):

    _write_go_hdf()
    with pd.HDFStore('geneinfo_cache/go-basic.h5', 'r') as store:
        entry = store.select("df", "goname == %r" % name.lower()).iloc[0]
    return entry.goterm


def go_info(terms):
            
    if type(terms) is pd.core.series.Series:
        terms = terms.tolist()

    if type(terms) is not list:
        terms = [terms]

    _write_go_hdf()

    with pd.HDFStore('geneinfo_cache/go-basic.h5', 'r') as store:
        for term in terms:
            entry = store.select("df", "goterm == %r" % term.upper()).iloc[0]
            desc = re.search(r'"([^"]+)"', entry.description).group(1)
            s = f'**<span style="color:gray;">{term}:</span>** **{entry.goname}**  \n {desc}    \n\n ----'        
            display(Markdown(s))


class WrSubObo(object):
    """Read a large GO-DAG from an obo file. Write a subset GO-DAG into a small obo file."""

    def __init__(self, fin_obo=None, optional_attrs=None, load_obsolete=None):
        self.fin_obo = fin_obo
        self.godag = GODag(fin_obo, optional_attrs, load_obsolete) if fin_obo is not None else None
        self.relationships = optional_attrs is not None and 'relationship' in optional_attrs

    def wrobo(self, fout_obo, goid_sources):
        """Write a subset obo file containing GO ID sources and their parents."""
        goids_all = self._get_goids_all(goid_sources)
        with open(fout_obo, 'w') as prt:
            self._prt_info(prt, goid_sources, goids_all)
            self.prt_goterms(self.fin_obo, goids_all, prt)

    @staticmethod
    def prt_goterms(fin_obo, goids, prt, b_prt=True):
        """Print the specified GO terms for GO IDs in arg."""
        b_trm = False
        with open(fin_obo) as ifstrm:
            for line in ifstrm:
                if not b_trm:
                    if line[:6] == "[Term]":
                        b_trm = True
                        b_prt = False
                    elif line[:6] == "[Typedef]":
                        b_prt = True
                else:
                    if line[:6] == 'id: GO':
                        b_trm = False
                        b_prt = line[4:14] in goids
                        if b_prt:
                            prt.write("[Term]\n")
                if b_prt:
                    prt.write(line)

    @staticmethod
    def get_goids(fin_obo, name):
        """Get GO IDs whose name matches given name."""
        goids = set()
        # pylint: disable=unsubscriptable-object
        goterm = None
        with open(fin_obo) as ifstrm:
            for line in ifstrm:
                if goterm is not None:
                    semi = line.find(':')
                    if semi != -1:
                        goterm[line[:semi]] = line[semi+2:].rstrip()
                    else:
                        if name in goterm['name']:
                            goids.add(goterm['id'])
                        goterm = None
                elif line[:6] == "[Term]":
                    goterm = {}
        return goids

    def _get_goids_all(self, go_sources):
        """Given GO ID sources and optionally the relationship attribute, return all GO IDs."""
        go2obj_user = {}
        objrel = CurNHigher(self.relationships, self.godag)
        objrel.get_id2obj_cur_n_high(go2obj_user, go_sources)
        goids = set(go2obj_user)
        for goterm in go2obj_user.values():
            if goterm.alt_ids:
                goids.update(goterm.alt_ids)
        return goids

    def _prt_info(self, prt, goid_sources, goids_all):
        """Print information describing how this obo setset was created."""
        prt.write("! Contains {N} GO IDs. Created using {M} GO sources:\n".format(
            N=len(goids_all), M=len(goid_sources)))
        for goid in goid_sources:
            prt.write("!    {GO}\n".format(GO=str(self.godag.get(goid, ""))))
        prt.write("\n")


class My_GOEnrichemntRecord(GOEnrichmentRecord):

    def __str__(self):
        return f'<{self.GO}>'


def go_enrichment(gene_list, taxid=9606, background_chrom=None, background_genes=None, 
    terms=None, list_study_genes=False, alpha=0.05):

    if type(gene_list) is pd.core.series.Series:
        gene_list = gene_list.tolist()
    if type(terms) is pd.core.series.Series:
        terms = terms.tolist()

    _assert_entrez_email()

    gene_list = list(gene_list)
    
    taxid = _tidy_taxid(taxid)

    ncbi_tsv = f'geneinfo_cache/{taxid}_protein_genes.txt'
    if not os.path.exists(ncbi_tsv):
        fetch_background_genes(taxid)

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_and_move_go_basic_obo(prt=null)

        file_gene2go = download_ncbi_associations(prt=null)

        obodag = GODag("geneinfo_cache/go-basic.obo", optional_attrs=['relationship', 'def'], prt=null)

        # read NCBI's gene2go. Store annotations in a list of namedtuples
        objanno = Gene2GoReader(file_gene2go, taxids=[taxid])

        # get associations for each branch of the GO DAG (BP, MF, CC)
        ns2assoc = objanno.get_ns2assc()

        # limit go dag to a sub graph including only specified terms and their children
        if terms is not None:
            sub_obo_name = 'geneinfo_cache/' + str(hash(''.join(sorted(terms)).encode())) + '.obo'  
            wrsobo = WrSubObo(obo_fname, optional_attrs=['relationship', 'def'])
            wrsobo.wrobo(sub_obo_name, terms)    
            obodag = GODag(sub_obo_name, optional_attrs=['relationship', 'def'], prt=null)

        # load background gene set of all genes
        background_genes_file = f'geneinfo_cache/{taxid}_protein_genes.txt'
        if not os.path.exists(background_genes_file):
            fetch_background_genes(taxid)

        # # load any custum subset
        if background_genes:
            if not all(type(x) is int for x in background_genes):
                if all(x.isnumeric() for x in background_genes):
                    background_genes = list(map(str, background_genes))
                else:
                    background_genes = _cached_symbol2ncbi(background_genes, taxid=taxid)
            df = pd.read_csv(background_genes_file, sep='\t')
            no_suffix = os.path.splitext(background_genes_file)[0]
            background_genes_file = f'{no_suffix}_{hash("".join(map(str, sorted(background_genes))))}.txt'            
            df.loc[df.GeneID.isin(background_genes)].to_csv(background_genes_file, sep='\t', index=False)

        # limit background gene set
        if background_chrom is not None:
            df = pd.read_csv(background_genes_file, sep='\t')
            background_genes_file = f'{os.path.splitext(background_genes_file)[0]}_{background_chrom}.txt'            
            df.loc[df.chromosome == background_chrom].to_csv(background_genes_file, sep='\t', index=False)

        output_py = f'geneinfo_cache/{taxid}_background.py'
        ncbi_tsv_to_py(background_genes_file, output_py, prt=null)

        background_genes_name = output_py.replace('.py', '').replace('/', '.')
        background_genes = importlib.import_module(background_genes_name)
        importlib.reload(background_genes)
        GeneID2nt = background_genes.GENEID2NT

        if not all(type(x) is int for x in gene_list):
            gene_list = _cached_symbol2ncbi(gene_list, taxid=taxid)

        goeaobj = GOEnrichmentStudyNS(
                GeneID2nt, # List of mouse protein-coding genes
                ns2assoc, # geneid/GO associations
                obodag, # Ontologies
                propagate_counts = False,
                alpha = 0.05, # default significance cut-off
                methods=['fdr_bh'],
                pvalcalc='fisher_scipy_stats') 

        goea_results_all = goeaobj.run_study(gene_list)


        rows = []
        columns = ['namespace', 'term_id', 'e/p', 'pval_uncorr', 'p_fdr_bh', 
                'ratio', 'bg_ratio', 'obj']
        if list_study_genes:
            columns.append('study_genes')
        for ntd in goea_results_all:

            ntd.__class__ = My_GOEnrichemntRecord # Hack. Changes __class__ of all instances...

            row = [ntd.NS, ntd.GO, ntd.enrichment, ntd.p_uncorrected,
                        ntd.p_fdr_bh, 
                        ntd.ratio_in_study[0] / ntd.ratio_in_study[1],
                        ntd.ratio_in_pop[0] /  ntd.ratio_in_pop[1], ntd]

            if list_study_genes:
                row.append(_cached_ncbi2symbol(sorted(ntd.study_items)))
            rows.append(row)
        df = (pd.DataFrame()
        .from_records(rows, columns=columns)
        .sort_values(by=['p_fdr_bh', 'ratio'])
        .reset_index(drop=True)
        )
        return df.loc[df.p_fdr_bh < alpha]



def show_go_dag_enrichment_results(results):

    if type(results) is pd.core.series.Series:
        results = results.tolist()
    with open(os.devnull, 'w') as null, redirect_stdout(null):
        plot_results('geneinfo_cache/plot.png', results)
    return Image('geneinfo_cache/plot.png')    


def chrom_ideogram(annot, hspace=0.1, min_visible_width=200000, figsize=(10,10), assembly='hg38'):

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

            if i in [20, 21]:   
                x = -3500000 * 10 / figsize[1]
            else:
                x = -2000000 * 10 / figsize[1]
            ax.text(x, 1, chrom.replace('chr', ''), fontsize=8, horizontalalignment='right', weight='bold')

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
            g = ax.add_patch(Rectangle((start, 1), end-start, 1, 
                                       fill=False,
                                       color='black',
                                       edgecolor=None,
                                       zorder=1, linewidth=0.7
                                      ))

            # draw centromere
            cent_start, cent_end = centromeres[chrom]
            ax.add_patch(Rectangle((cent_start, 0), cent_end-cent_start, 3, 
                                       fill=True, color='white',
                                       zorder=2))
            xy = [[cent_start, 1], [cent_start, 2], [cent_end, 1], [cent_end, 2]]
            g = ax.add_patch(Polygon(xy, closed=True, zorder=3, fill=True,
                                     # color='#666666',
                                     color='#777777',
                                    ))


        def plot_segment(chrom, start, end, color='red', label=None, base=0, height=1):

            base += 1
            
            x, y, width = start, base, end-start

            if width < min_visible_width:
                x -= min_visible_width/2
                width += min_visible_width

            rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='none', facecolor=color, zorder=3)
            chr_axes[chrom].add_patch(rect)    
            if label is not None:
                chr_axes[chrom].plot([x+width/2, x+width/2], [y+height, y+height+0.5], linewidth=0.5, color=color, zorder=3)
                chr_axes[chrom].text(x+width/2, y+height+0.5, label, fontsize=4, horizontalalignment='left',# weight='bold',
                         verticalalignment='bottom', rotation=45, zorder=3)

        for tup in annot:
            plot_segment(*tup)                     
            

# annot = [('chr1', 20000000, 20100000, 'red', 'TP53'), ('chr7', 20000000, 30000000, 'orange', 'DYNLT3')] \
# + [('chr5', 40000000, 70000000, 'red', None, 1, 0.5), ('chr8', 90000000, 110000000)] \
#  + [('chrX', x[0], x[1], 'black', str(x[2]/1000000)) for x in zip(range(0, 150000000, 10000000), range(300000, 150000000, 10000000), range(0, 150000000, 10000000))]

# chrom_ideogram(annot, figsize=(15, 9), hspace=0.2) 
