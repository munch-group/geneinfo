
from IPython.display import Markdown, display, Image, SVG, HTML
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import sys
import os
import re
import json
import pandas as pd

from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.obo_parser import GODag, OBOReader
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot
from goatools.cli.ncbi_gene_results_to_python import ncbi_tsv_to_py
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.go_search import GoSearch

import requests
from Bio import Entrez

from .intervals import *

CACHE = dict()


def mygene_get_gene_info(query, species='human', scopes='hgnc', fields='symbol,alias,name,type_of_gene,summary,genomic_pos,genomic_pos_hg19'):
    api_url = f"https://mygene.info/v3/query?content-type=appliation/x-www-form-urlencoded;q={query};scopes={scopes};species={species};fields={fields}"
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

        top_hit = mygene_get_gene_info(gene, species=species, scopes=scopes,
                           fields='symbol,alias,name,type_of_gene,summary,genomic_pos,genomic_pos_hg19')

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


def _ensembl_get_features_region(chrom, window_start, window_end, features=['gene', 'exon'], assembly=None, species='human'):
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
        sys.exit()     
    return r.json()


def ensembl_get_genes_region(chrom, window_start, window_end, assembly=None, species='human'):
    
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

def get_genes_region(chrom, window_start, window_end, only_protein_coding=True, hg19=False, species='human'):

    if hg19:
        assembly='GRCh37'
    else:
        assembly=None

    genes = []
    gene_info = ensembl_get_genes_region(chrom, window_start, window_end, assembly=assembly, species=species)
    for gene in gene_info.values():


        if gene['biotype'] != 'protein_coding' and only_protein_coding:
            continue

        if 'external_name' in gene:
            name = gene['external_name']
        else:
            name = gene['id']
        genes.append((name, gene['start'], gene['end'], gene['strand'], gene['exons'], gene['biotype']))
    return genes


def gene_info_region(chrom, window_start, window_end, only_protein_coding=True, hg19=False, species='human'):
    # mg.query(f'q={chrom}:{start}-{end}', species='human', fetch_all=True):
    
    for gene in get_genes_region(chrom, window_start, window_end, 
                                    only_protein_coding=only_protein_coding, hg19=hg19, species=species):
        gene_info(gene[0])


def get_genes_region_dataframe(chrom, start, end, hg19=False):
    try:
        import pandas as pd
    except ImportError:
        print("pandas must be installed to return data frame")
        return
    genes = get_genes_region(chrom, start, end, hg19=hg19)
    return pd.DataFrame().from_records([x[:4] for x in genes], columns=['name', 'start', 'end', 'strand'])

def _plot_gene(name, txstart, txend, strand, exons, gene_type, offset, line_width, min_visible_width, font_size, ax, highlight=False, clip_on=True):

    if gene_type == 'protein_coding':
        color='black'
    elif gene_type == 'ncrna':
        color='blue'
    else:
        color='green'

    line = ax.plot([txstart, txend], [offset, offset], color=color, linewidth=line_width/5, alpha=0.5)
    line[0].set_solid_capstyle('butt')

    # arrowpoints = np.arange(txstart, txend, 5000)
    # y = np.full_like(arrowpoints, offset)
    # arrowline = ax.plot(arrowpoints, y, color=color)[0]    
    # [add_arrow(arrowline, position=x) for x in arrowpoints[:-1]]

    for start, end in exons:
        # line = ax.plot([start, end], [offset, offset], linewidth=line_width, color=color)
        # line[0].set_solid_capstyle('butt')

        end = max(start+min_visible_width, end)
        line = ax.plot([start, end], [offset, offset], linewidth=line_width, color=color)
        line[0].set_solid_capstyle('butt')
        
    if highlight:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', verticalalignment='center', fontsize=font_size, weight='bold', color='red', clip_on=clip_on)#, transform=ax.transAxes)
    else:
        ax.text(txstart, offset-.5, name, horizontalalignment='right', verticalalignment='center', fontsize=font_size, color=color, clip_on=clip_on)#, transform=ax.transAxes)


def gene_plot(chrom, start, end, highlight=[], hg19=False, only_protein_coding=False, hard_limits=False, exact_exons=False, figsize=None, clip_on=True):
    
    global CACHE

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0, hspace=0.05)

    if (chrom, start, end, hg19) in CACHE:
        genes = CACHE[(chrom, start, end, only_protein_coding, hg19)]
    else:
        genes = list(get_genes_region(chrom, start, end, only_protein_coding=only_protein_coding, hg19=hg19))
        CACHE[(chrom, start, end, only_protein_coding, hg19)] = genes

    # print(len(genes))
    if len(genes) == 200:
        print("Limit reached, make window smaller")
        return

    from math import log10
    # line_width, font_size = min(12, int(12 / log10(len(genes))))-2, max(8, int(12 / log10(len(genes))))
    # label_width = font_size * (end - start) / 100
    line_width = max(6, int(50 / log10(end - start)))-2
    font_size = max(6, int(50 / log10(end - start)))
    label_width = font_size * (end - start) / 60
    if exact_exons:
        min_visible_exon_width = 0
    else:
        min_visible_exon_width = (end - start) / 1000
    # print(font_size, line_width)
        
    plotted_intervals = defaultdict(list)
    for name, txstart, txend, strand, exons, gene_type in genes:

        gene_interval = [txstart-label_width, txend]
        max_gene_rows = 400
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
    
        _plot_gene(name, txstart, txend, strand, exons, gene_type, 
                  offset, line_width, min_visible_exon_width, font_size, 
                  highlight=name in highlight,
                  ax=ax2, clip_on=clip_on)

    if plotted_intervals:
        offset = max(plotted_intervals.keys())
    else:
        offset = 1

    if hard_limits:
        ax2.set_xlim(start, end)
    else:
        s, e = ax2.get_xlim()
        ax2.set_xlim(min(s-label_width/2, start), max(e, end))

    ax2.set_ylim(-2, offset+2)
    ax2.get_yaxis().set_visible(False)
    ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # points = ax2.plot(txstart, offset+.5, 'o', ms=25, alpha=0, zorder=10)

    ax1.set_xlim(ax2.get_xlim())

    return ax1


##################################################################################
# Map between assembly coordinates
##################################################################################

def map_interval(chrom, start, end, strand, map_from, map_to, species='human'):
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

def fetch_background_genes(taxid=9606):
    
    _assert_entrez_email()

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        output_base_name = f'{taxid}_protein_genes'
        
        handle = Entrez.esearch(db="gene", term=f'{taxid}[Taxonomy ID] AND alive[property] AND genetype protein coding[Properties]', retmax="1000000")
        records = Entrez.read(handle)

        with open(f'{output_base_name}.txt', 'w') as f:
            header = ['tax_id', 'Org_name', 'GeneID', 'CurrentID', 'Status', 'Symbol', 'Aliases', 
                  'description', 'other_designations', 'map_location', 'chromosome', 
                  'genomic_nucleotide_accession.version', 'start_position_on_the_genomic_accession', 
                  'end_position_on_the_genomic_accession', 'orientation', 'exon_count', 'OMIM']
            print(*header, sep='\t', file=f)

            batch_size = 2000
            for i in range(0, int(records['Count']), batch_size):
                to_fetch = records["IdList"][i:i+batch_size]
                handle = Entrez.esummary(db="gene", id=",".join(to_fetch), retmax=batch_size)
                entry = Entrez.read(handle)
                docsums = entry['DocumentSummarySet']['DocumentSummary']
                for doc in docsums:
                    try:
                        print(doc['Organism']['TaxID'], doc['Organism']['ScientificName'], doc.attributes['uid'], 
                              doc['CurrentID'], 
                              # doc['Status'],
                              'live',
                              doc['Name'], doc['OtherAliases'], doc['Description'], doc['OtherDesignations'],
                              doc['MapLocation'], doc['Chromosome'], 
                              doc['GenomicInfo'][0]['ChrAccVer'], doc['GenomicInfo'][0]['ChrStart'], doc['GenomicInfo'][0]['ChrStop'],
                              'notspecified', doc['GenomicInfo'][0]['ExonCount'], '',
                              sep='\t', file=f)
                    except Exception as e:
                        print(e)
                        pass
  
    
def get_terms_for_go_regex(regex, taxid=9606, add_children=True):

    try:
        taxid = int(taxid)
    except ValueError:
        handle = Entrez.esearch(db="taxonomy", term=f'"{taxid}"[Scientific Name]')
        taxid = int(Entrez.read(handle)['IdList'][0])
        
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_go_basic_obo(prt=null)

        gene2go = download_ncbi_associations(prt=null)

        objanno = Gene2GoReader("gene2go", taxids=[taxid], prt=null)
        # go2geneids_human = objanno.get_id2gos(namespace='BP', go2geneids=True)
        go2geneids = objanno.get_id2gos(namespace='*', go2geneids=True, prt=null)

        srchhelp = GoSearch("go-basic.obo", go2items=go2geneids, log=null)

        results_all = re.compile(r'({})'.format(regex), flags=re.IGNORECASE)
        results_not = re.compile(r'({}).independent'.format(regex), flags=re.IGNORECASE)

        gos_all = srchhelp.get_matching_gos(results_all, prt=null)
        gos_no = srchhelp.get_matching_gos(results_not, gos=gos_all)
        gos = gos_all.difference(gos_no)
        if add_children:
            gos = srchhelp.add_children_gos(gos)

        return gos

def get_genes_for_go_regex(regex, taxid=9606):

    _assert_entrez_email()

    try:
        taxid = int(taxid)
    except ValueError:
        handle = Entrez.esearch(db="taxonomy", term=f'"{taxid}"[Scientific Name]')
        taxid = int(Entrez.read(handle)['IdList'][0])
        
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        gos_all_with_children = get_terms_for_go_regex(regex, taxid=taxid)

        geneids = srchhelp.get_items(gos_all_with_children)

        ncbi_tsv = f'{taxid}_protein_genes.txt'
        if not os.path.exists(ncbi_tsv):
            fetch_background_genes(taxid)

        output_py = f'{taxid}_protein_genes.py'
        ncbi_tsv_to_py(ncbi_tsv, output_py, prt=null)

        protein_genes = importlib.import_module(output_py.replace('.py', ''))
        GENEID2NT = protein_genes.GENEID2NT

    #############################################################

        # fetch_ids = []
        # for geneid in geneids:
        #     nt = GENEID2NT.get(geneid, None)
        #     if nt is not None:
        #         fetch_ids.append(nt.GeneID)


    #     mg = mygene.MyGeneInfo()                
    #     d = mg.getgenes(fetch_ids, species='human', scopes='hgnc', fields=['symbol', 'name', 'genomic_pos.chr', 'genomic_pos.start', 'genomic_pos.end', 'genomic_pos.strand'])


    # records = []
    # for entry in d:

    #     if not 'genomic_pos' in entry:
    #         print("No coordinates for:", entry['symbol'], file=sys.stderr)
    #         continue
    #     pos = entry['genomic_pos']
    #     if type(pos) is list:
    #         pos = pos[0]
    #     records.append([entry['symbol'], entry['name'], 
    #                     pos['chr'], pos['start'],
    #                     pos['end'], pos['strand']])

    # df = pd.DataFrame().from_records(records, columns=['symbol', 'name', 'chrom', 'start', 'end', 'strand'])

    #############################################################

    # fetch_ids = []
    # for geneid in geneids:
    #     nt = GENEID2NT.get(geneid, None)
    #     if nt is not None:
    #         fetch_ids.append(nt.Symbol)

    # records = []
    # batch_size = 1000
    # found = []
    # for i in range(0, len(fetch_ids), batch_size):
    #     to_fetch = fetch_ids[i:i+batch_size]
    #     result = ensembl_get_gene_info_by_symbol(to_fetch)
    #     print(i, len(result))
    #     for name, info in result.items():
    #         print(info)
    #         assert 0
    #         records.append((info['display_name'], re.sub(r'\[.*\]', '', info['description']),
    #                         info['seq_region_name'], info['start'], info['end'], info['strand'])
    #                         )
    #         found.append(name)
    # missing = set(fetch_ids).difference(set(found))

    # # use ensembl id instead to get it with a GET request ENSG00000138185

    # # make a ensembl_get_gene_info function for that (maybe make return values the same as mygene_get_gene_info)

    # # look in mygene for the rest:

    # for symbol in missing:
    #     print(symbol)
    #     entry = mygene_get_gene_info(symbol)
    #     if not 'genomic_pos' in entry:
    #         print("No coordinates for:", entry['symbol'], file=sys.stderr)
    #         continue
    #     pos = entry['genomic_pos']
    #     if type(pos) is list:
    #         pos = pos[0]
    #     records.append([entry['symbol'], entry['name'], 
    #                     pos['chr'], pos['start'],
    #                     pos['end'], pos['strand']])

    # df = pd.DataFrame().from_records(records, columns=['symbol', 'name', 'chrom', 'start', 'end', 'strand'])

    #############################################################


    # TODO: Handle the case where geneid is not found in GENEID2NT (multiple instances in this file)
    # fetch_ids = []
    # for geneid in geneids:
    #     nt = GENEID2NT.get(geneid, None)
    #     if nt is not None:
    #         fetch_ids.append(nt.GeneID)
    #     else:
    #         print(geneid, 'not found in GENEID2NT', file=sys.stderr)


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

    #############################################################





    return df.sort_values(by='start').reset_index(drop=True)

    
def get_genes_for_go_terms(terms, taxid=9606):
    
    if type(terms) is not list:
        terms = [terms]

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_go_basic_obo(prt=null)
        gene2go = download_ncbi_associations(prt=null)
        objanno = Gene2GoReader("gene2go", taxids=[taxid], prt=null)
        go2geneids = objanno.get_id2gos(namespace='*', go2geneids=True, prt=null)
        srchhelp = GoSearch("go-basic.obo", go2items=go2geneids, log=null)

        geneids = srchhelp.get_items(terms)  

#######################################################



        ncbi_tsv = f'{taxid}_protein_genes.txt' 
        if not os.path.exists(ncbi_tsv):
            fetch_background_genes(taxid)

        output_py = f'{taxid}_protein_genes.py'
        ncbi_tsv_to_py(ncbi_tsv, output_py, prt=null)

    protein_genes = importlib.import_module(output_py.replace('.py', ''))
    GENEID2NT = protein_genes.GENEID2NT

    # TODO: Handle the case where geneid is not found in GENEID2NT (multiple instances in this file)
    # fetch_ids = []
    # for geneid in geneids:
    #     nt = GENEID2NT.get(geneid, None)
    #     if nt is not None:
    #         fetch_ids.append(nt.GeneID)
    #     else:
    #         print(geneid, 'not found in GENEID2NT', file=sys.stderr)

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





########################################################



    # mg = mygene.MyGeneInfo()                
    # d = mg.getgenes(geneids, species='human',  scopes='hgnc', fields=['symbol', 'name', 'genomic_pos.chr', 'genomic_pos.start', 'genomic_pos.end', 'genomic_pos.strand'])

    # records = []
    # for entry in d:
    #     if not 'genomic_pos' in entry:
    #         print("No coordinates for:", entry['symbol'])
    #         continue
    #     pos = entry['genomic_pos']
    #     if type(pos) is list:
    #         pos = pos[0]
    #     records.append([entry['symbol'], entry['name'], 
    #                     pos['chr'], pos['start'],
    #                     pos['end'], pos['strand']])
    # df = pd.DataFrame().from_records(records, columns=['symbol', 'name', 'chrom', 'start', 'end', 'strand'])

    return df.sort_values(by='start').reset_index(drop=True)


def go_annotation_table(taxid=9606):

    _assert_entrez_email()

    try:
        taxid = int(taxid)
    except ValueError:
        handle = Entrez.esearch(db="taxonomy", term=f'"{taxid}"[Scientific Name]')
        taxid = int(Entrez.read(handle)['IdList'][0])
        
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_go_basic_obo(prt=null)

        gene2go = download_ncbi_associations(prt=null)
        
    df = pd.read_table(gene2go, sep='\t')
    df.rename(columns={'#tax_id': 'taxid'}, inplace=True)
    return df.loc[df['taxid'] == taxid]


def gene_annotation_table(taxid=9606):

    ncbi_tsv = f'{taxid}_protein_genes.txt'
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

    
def show_go_dag_for_terms(terms):
    
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        # Get http://geneontology.org/ontology/go-basic.obo
        obo_fname = download_go_basic_obo(prt=null)

        # Download Associations, if necessary
        # Get ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
        file_gene2go = download_ncbi_associations(prt=null)

        #Load Ontologies
        obodag = GODag("go-basic.obo", optional_attrs=['relationship', 'def'], prt=null)

        gosubdag = GoSubDag(terms, obodag, relationships=True)
        GoSubDagPlot(gosubdag).plt_dag('plot.png')

    return Image('plot.png')    


def show_go_dag_for_gene(gene, taxid=9606, evidence=None):
    # evidence codes: http://geneontology.org/docs/guide-go-evidence-codes/
    go_terms = get_go_terms_for_genes([gene], taxid=taxid, evidence=evidence)
    if not go_terms:
        print('No GO terms to show', file=sys.stderr)
        return
    return show_go_dag_for_terms(go_terms)

    
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
                    
        
def go_info(terms):
            
    if type(terms) is not list:
        terms = [terms]

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        # Get http://geneontology.org/ontology/go-basic.obo
        obo_fname = download_go_basic_obo(prt=null)

        # Download Associations, if necessary
        # Get ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
        file_gene2go = download_ncbi_associations(prt=null)

        if not os.path.exists('go-basic.h5'):

            r = OBOReader(optional_attrs=['def'])
            rows = [(e.id, e.name, e.defn) for e in OBOReader(optional_attrs=['def'])]
            df = pd.DataFrame().from_records(rows, columns=['goterm', 'goname', 'description']).set_index('goterm')
            df.to_hdf('go-basic.h5', key='df', format='table')

    store = pd.HDFStore('go-basic.h5')

    for term in terms:
        entry = store.select("df", "index == %r" % term).iloc[0]
        desc = re.search(r'"([^"]+)"', entry.description).group(1)
        s = f'**<span style="color:gray;">{term}:</span>** **{entry.goname}**  \n {desc}    \n\n ----'        
        display(Markdown(s))


def go_enrichment(gene_list, taxid=9606, background_chrom=None, terms=None):

    _assert_entrez_email()

    gene_list = list(gene_list)
    
    try:
        taxid = int(taxid)
    except ValueError:
        handle = Entrez.esearch(db="taxonomy", term=f'"{taxid}"[Scientific Name]')
        taxid = int(Entrez.read(handle)['IdList'][0])
    
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        # Get http://geneontology.org/ontology/go-basic.obo
        obo_fname = download_go_basic_obo(prt=null)

        # Download Associations, if necessary
        # Get ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
        file_gene2go = download_ncbi_associations(prt=null)

        #Load Ontologies
        obodag = GODag("go-basic.obo", optional_attrs=['relationship', 'def'], prt=null)

        # limit go dag to a sub graph including only specified terms and their children
        if terms is not None:

            srchhelp = GoSearch("go-basic.obo", go2items=go2geneids, log=null)
            terms_with_children = srchhelp.add_children_gos(terms)

            obodag = GoSubDag(terms_with_children, obodag)

        # load associations
        # Read NCBI's gene2go. Store annotations in a list of namedtuples
        objanno = Gene2GoReader(file_gene2go, taxids=[taxid])

        # Get associations for each branch of the GO DAG (BP, MF, CC)
        ns2assoc = objanno.get_ns2assc()

        #Load Background gene set

        ncbi_tsv = f'{taxid}_protein_genes.txt'
        if not os.path.exists(ncbi_tsv):
            fetch_background_genes(taxid)

        # limit background gene set
        if background_chrom is not None:
            df = pd.read_csv(ncbi_tsv, sep='\t')
            ncbi_tsv = f'{taxid}_protein_genes_{background_chrom}.txt'            
            df.loc[lambda df: df.chromosome == background_chrom].to_csv(ncbi_tsv, sep='\t', index=False)

        output_py = f'{taxid}_background.py'
        ncbi_tsv_to_py(ncbi_tsv, output_py, prt=null)

        background_genes_name = output_py.replace('.py', '')
        background_genes = importlib.import_module(background_genes_name)
        importlib.reload(background_genes)
        GeneID2nt = background_genes.GENEID2NT

        goeaobj = GOEnrichmentStudyNS(
                GeneID2nt, # List of mouse protein-coding genes
                ns2assoc, # geneid/GO associations
                obodag, # Ontologies
                propagate_counts = False,
                alpha = 0.05, # default significance cut-off
                methods=['bonferroni', 'fdr_bh'],
                pvalcalc='fisher_scipy_stats') 

        goea_results_all = goeaobj.run_study(gene_list)

        rows = []
        columns = ['namespace', 'term_id', 'e/p', 'pval_uncorr', 'Benjamimi/Hochberg', 
                   'Bonferroni', 'study_ratio', 'population_ratio']
        for ntd in goea_results_all:
            rows.append([ntd.NS, ntd.GO, ntd.enrichment, ntd.p_uncorrected,
                         ntd.p_fdr_bh, ntd.p_bonferroni,
                         ntd.ratio_in_study[0] / ntd.ratio_in_study[1],
                         ntd.ratio_in_pop[0] /  ntd.ratio_in_pop[1]])
        df = (pd.DataFrame()
         .from_records(rows, columns=columns)
         .sort_values(by=['pval_uncorr', 'term_id'])
         .reset_index(drop=True)
        )
        return df
  


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
