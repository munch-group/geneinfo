
from dis import Positions
from IPython.display import Markdown, display
from collections import defaultdict
import sys
import os
import json
import pickle
import pandas as pd
from typing import Any, TypeVar, List, Tuple, Dict, Union
import requests
from pathlib import Path

from ..intervals import *

from ..utils import GeneList

from ..utils import shelve_it
cache_dir = Path(os.path.dirname(__file__)).parent / 'data'


class NotFound(Exception):
    """
    Exception raised when a gene or other entity is not found.

    Exception : 
        Does nothing. Just a return value placeholder.
    """
    pass


@shelve_it()
def _ensembl_id(name:str, species:str='homo_sapiens') -> str:
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


def ensembl_id(name:Union[str, list], species:str='homo_sapiens') -> str:
    """
    Get ENSEMBL ID for some gene identifier

    Parameters
    ----------
    name : 
        Gene identifier
    species :  optional
        Species, by default 'homo_sapiens'

    Returns
    -------
    :
        ENSEMBL ID

    Raises
    ------
    [](`~geneinfo.NotFound`)
        Raises exception if no ENSEMBL ID can be found.
    """

    if isinstance(name, GeneList):
        name = list(name)
        
    if type(name) is str:
        name_list = [name]
    else:
        name_list = name
    return [_ensembl_id(x) for x in name_list]


@shelve_it()
def ensembl2symbol(ensembl_id:str) -> str:
    """
    Converts ENSEMBL ID to gene HGCN gene symbol    

    Parameters
    ----------
    ensembl_id : 
        ENSEMBL ID

    Returns
    -------
    :
        HGCN gene symbol

    Raises
    ------
    [](`~geneinfo.NotFound`)
        Raises exception if no HGCN gene symbol can be found.
    """
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


def hgcn_symbol(name:str) -> str:
    """
    Get HGCN gene symbol for some gene identifier

    Parameters
    ----------
    name : 
        Gene identifier

    Returns
    -------
    :
        HGCN gene symbol

    Raises
    ------
    [](`~geneinfo.NotFound`)
        Raises exception if no HGCN gene symbol can be found.
    """ 
    if isinstance(name, GeneList):
        name = list(name)
    if type(name) is list or type(name) is set:
        return [ensembl2symbol(ensembl_id(n)) for n in name]
    else:
        return ensembl2symbol(ensembl_id(name))


@shelve_it()
def ensembl2ncbi(ensembl_id):
    """
    Converts ENSEMBL ID to gene NCBI ID

    Parameters
    ----------
    ensembl_id : 
        ENSEMBL ID

    Returns
    -------
    :
        NCBI ID

    Raises
    ------
    [](`~geneinfo.NotFound`)
        Raises exception if no NCBI ID can be found.
    """
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


@shelve_it()
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


@shelve_it()
def gene_coord(query: Union[str, List[str]], assembly:str, species='homo_sapiens', pos_list=False) -> dict:
    """
    Retrieves genome coordinates one or more genes.

    Parameters
    ----------
    query : 
        Gene symbol or list of gene symbols
    assembly :  
        Genome assembly.
    species :  
        Species, by default 'homo_sapiens'.
    pos_list :
        Wether to instead return a list of (chrom, position, name) tuples.

    Returns
    -------
    :
        Dictionary with gene names as keys and (chrom, start, end, strand) tuples as values, or a list of 
        (chrom, position, name) tuples.
    """

    coords = {}
    batch_size = 100
    for i in range(0, len(query), batch_size):
        data = ensembl_get_gene_info_by_symbol(query[i:i+batch_size], assembly=None, species='homo_sapiens')
        for name, props in data.items():
            chrom, start, end, strand = props['seq_region_name'], props['start'], props['end'], props['strand']
            if not chrom.lower().startswith('contig') and not chrom.lower().startswith('scaffold'):
                chrom = 'chr'+chrom
            if strand == -1:
                strand = '-'
            elif strand == 1:
                strand = '+'
            else:
                strand = None
    
            coords[name] = (chrom, start, end, strand)

    if pos_list:
        annot = []
        for (name, (chrom, start, end, strand)) in coords.items():
            annot.append((chrom, int((start+end)/2), name))
        return sorted(annot, key=lambda x: x[1:])

    return coords


@shelve_it()
def gene_info(query: Union[str, List[str]], species:str='human', scopes:str='hgnc') -> None:
    """
    Displays HTML formatted information about one or more genes.

    Parameters
    ----------
    query : 
        Gene symbol or list of gene symbols
    species :  optional
        Species, by default 'human'
    scopes :  optional
        Scopes for information search, by default 'hgnc'
    """

    if isinstance(query, GeneList):
        query = list(query)
            
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


@shelve_it()
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
        params = response.json()

        for gene in params:
            genes[gene['id']] = gene
            
    return genes


@shelve_it()
def ensembl_get_gene_info_by_symbol(symbols, assembly=None, species='homo_sapiens'):

    if assembly == 'hg38':
        assembly='GRCh38'
    if assembly == 'hg19':
        assembly='GRCh37'

    if isinstance(symbols, GeneList):
        symbols = list(symbols)

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


@shelve_it()
def get_genes_region(chrom:str, window_start:int, window_end:int, 
                     assembly:str='GRCh38', db:str='ncbiRefSeq') -> list:
    """
    Gets gene structure information for genes in a chromosomal region.

    Parameters
    ----------
    chrom : 
        Chromosome identifier
    window_start : 
        Start of region
    window_end : 
        End of region (end base not included)
    assembly : 
        Genome assembly, by default 'GRCh38'
    db : 
        Database, by default 'ncbiRefSeq'

    Returns
    -------
    :
        List of gene information. Each gene is a tuple with the following elements:
        - gene name
        - gene start
        - gene end
        - gene strand
        - list of exons (start, end)
    """
    api_url = f'https://api.genome.ucsc.edu/getData/track'
    params = {'track': db,
              'genome': assembly,
              'chrom': chrom,
              'start': window_start,
              'end': window_end
              }
    response = requests.get(api_url, params=params)
    if not response.ok:
        response.raise_for_status()

    genes = []
    for gene in response.json()[db]:
        exon_starts = [int(x) for x in gene['exonStarts'].split(',') if x]
        exon_ends = [int(x) for x in gene['exonEnds'].split(',') if x]
        exons = list(zip(exon_starts, exon_ends))
        genes.append((gene['name2'], gene['txStart'], gene['txEnd'], gene['strand'], exons))

    return genes


def get_genes_region_dataframe(chrom:str, window_start:int, window_end:int, 
                     assembly:str='GRCh38', db:str='ncbiRefSeq') -> pd.DataFrame:
    """
    Gets gene structure information for genes in a chromosomal region in the form
    of a pandas.DataFrame.

    Parameters
    ----------
    chrom : 
        Chromosome identifier
    window_start : 
        Start of region
    window_end : 
        End of region (end base not included)
    assembly :  optional
        Genome assembly, by default 'GRCh38'
    db :  optional
        Database, by default 'ncbiRefSeq'

    Returns
    -------
    :
        pandas.DataFrame with the following colunms:
        - name: gene name
        - start: gene start
        - end: gene end
        - strand: gene strand
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas must be installed to return data frame")
        return
    genes = get_genes_region(chrom, window_start, window_end, assembly, db)
    return pd.DataFrame().from_records([x[:4] for x in genes], columns=['name', 'start', 'end', 'strand'])


def gene_info_region(chrom:str, window_start:int, window_end:int, 
                     assembly:str='GRCh38', db:str='ncbiRefSeq') -> None:
    """
    Displays HTML formatted information about genes in a chromosomal region.

    Parameters
    ----------
    chrom : 
        Chromosome identifier
    window_start : 
        Start of region
    window_end : 
        End of region (end base not included)
    assembly : 
        Genome assembly, by default 'GRCh38'
    db : 
        Database, by default 'ncbiRefSeq'
    """
    for gene in get_genes_region(chrom, window_start, window_end, assembly, db):    
        gene_info(gene[0])


