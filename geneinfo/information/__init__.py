
from IPython.display import Markdown, display
from collections import defaultdict
import sys
import os
import json
import pandas as pd
from typing import Any, TypeVar, List, Tuple, Dict, Union
import requests
from pathlib import Path
from collections.abc import Sequence, MutableSequence

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


from collections import defaultdict
import re

def list_assemblies() -> None:
    """
    Lists available genome assemblies from UCSC genome browser.
    """
    api_url = f'https://api.genome.ucsc.edu/list/ucscGenomes'
    response = requests.get(api_url)
    if not response.ok:
        response.raise_for_status()
    records = []
    for name, data in response.json()['ucscGenomes'].items():
        records.append((name.ljust(8), data['organism'].ljust(20), data['scientificName']))
    df = pd.DataFrame().from_records(records, columns=['Assemblies', 'Species', 'Latin name'])
    # df.groupby(['species', 'latin']).apply(lambda _df: (_df.species, _df.latin, _df.assembly))
    df = df.groupby(['Species', 'Latin name']).agg(lambda sr: ', '.join(sr)).reset_index()

    df = df.style.set_properties(**{'text-align': 'left'})
     
    df = df.set_table_styles(
        [dict(selector='th', props=[('text-align', 'left')])])
     
    display(df)


def _chrom_sort_key(chrom):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', chrom)]

def chrom_sort_key(chrom):
    """
    Function for use as key in sorting chromosomes. Works for both
    Python lists and numpy arrays/pandas series.
    """
    if isinstance(chrom, (list, tuple)):
        return chrom_sort_key(chrom[0])
    elif isinstance(chrom, Sequence):
        return [_chrom_sort_key(x) for x in chrom]
    else:
        return _chrom_sort_key(chrom)


def chromosome_lengths(assembly:str) -> List[Tuple[str, int]]:
    """
    Retrieves chromosome lengths for a genome assembly from UCSC.

    Parameters
    ----------
    assembly : 
        Assembly identifier, e.g. 'hg38' or 'rheMac10'

    Returns
    -------
    :
        List of tuples with chromosome name and length.
    """
    api_url = f'https://api.genome.ucsc.edu/list/chromosomes?genome={assembly};track=gold'
    response = requests.get(api_url)
    if not response.ok:
        response.raise_for_status()
    records = []
    for chrom, length in response.json()['chromosomes'].items():
        if '_' in chrom or chrom == 'chrM':
            continue
        records.append((chrom, length))
    return sorted(records, key=chrom_sort_key)  


@shelve_it()
def centromere_coords(assembly:str) -> List[Tuple[str, int, int]]:
    """
    Retrieves centromere coordinates for a genome assembly from UCSC.

    Parameters
    ----------
    assembly : 
        Assembly identifier, e.g. 'hg38' or 'rheMac10'

    Returns
    -------
    :
        List of tuples with chromosome, start, and end of each centromere.
    """    
    api_url = f'https://api.genome.ucsc.edu/getData/track?genome={assembly};track=centromeres'
    response = requests.get(api_url)
    if not response.ok:
        return []
    data = defaultdict(list)
    for chrom, val in response.json()['centromeres'].items():
        for d in val:
            if '_' in d['chrom'] or d['chrom'] == 'chrM':
                continue
            data[d['chrom']].append((d['chromStart'], d['chromEnd']))
    records = []
    for ch, val in data.items():
        starts, ends = zip(*val)
        records.append((ch, min(starts), max(ends)))
    return sorted(records, key=chrom_sort_key)  

    # df = pd.DataFrame().from_records(records, columns=['chrom', 'start', 'end'])
    # df.sort_values('chrom', key=lambda sr: [chrom_sort_key(x) for x in sr]).reset_index(drop=True)


@shelve_it()
def gene_coords_region(chrom=None, start=None, end=None, assembly=None, as_dataframe=False):
    """
    Gets gene structure information for genes in a chromosomal region.

    Parameters
    ----------
    chrom : 
        Chromosome identifier
    start : 
        Start of region
    end : 
        End of region (end base not included)
    assembly : 
        Genome assembly as USCS genome identifier. E.g. hg38 or rheMac10
    as_dataframe : 
        Return dataframe instead of list of tuples, by default False

    Returns
    -------
    :
        List of gene information. Each gene is a tuple with the following elements:
        - gene name
        - gene start
        - gene end
        - list of list of exons (start, end) for a transcript
    """    
    assert assembly is not None
    api_url = f'https://api.genome.ucsc.edu/getData/track?genome={assembly};track=ncbiRefSeq;chrom={chrom}'
    if start is not None and end is not None:
        api_url += f';start={start};end={end}'
    
    response = requests.get(api_url)
    if not response.ok:
        response.raise_for_status()

    data = defaultdict(list)
    for d in response.json()['ncbiRefSeq']:
        if d['chrom'] != chrom:
            continue
        exon_starts = [int(x) for x in d['exonStarts'].split(',') if x]
        exon_ends = [int(x) for x in d['exonEnds'].split(',') if x]
        exons = list(zip(exon_starts, exon_ends))    
        data[(d['chrom'], d['name2'])].append((d['txStart'], d['txEnd'], exons))
    records = []
    for (ch, name), val in data.items():
        starts, ends, exons = zip(*val)
        # records.append((chrom, name, starts, ends, exons))
        records.append((name, ch, min(starts), max(ends), exons))
    # for (chrom, name), (starts, ends, exons) in data.items():
    #     records.append((name, chrom, min(starts), max(ends), exons))
    if as_dataframe:
        return pd.DataFrame().from_records(records, columns=['name', 'chrom', 'start', 'end', 'exons'])
    else:
        return records


@shelve_it()
def all_gene_coords(assembly:str) -> Dict[str, Tuple[str, int, int]]:
    """
    Get gene coordinates for all genes in a genome assembly.

    Parameters
    ----------
    assembly : 
        Assembly identifier, e.g. 'hg38' or 'rheMac10'

    Returns
    -------
    :
        Dictionary where keys are gene names and each value is a tuple with chromosome, start, and end.
    """
    all_coords = {}
    for chrom, length in chromosome_lengths(assembly):
        if '_' in chrom or chrom == 'chrM':
            continue
        for tup in gene_coords_region(chrom, assembly=assembly, as_dataframe=True).itertuples():
            all_coords[tup.name] = (tup.chrom, tup.start, tup.end)
    return all_coords

    
@shelve_it()
def gene_coords(names:Union[str,Sequence[str]], assembly:str) -> List[Tuple[str, int, int, str]]:
    """
    Get gene coordinates for a gene or list of genes.

    Parameters
    ----------
    names : 
        Gene name or list of gene names
    assembly : 
        Assembly identifier, e.g. 'hg38' or 'rheMac10'

    Returns
    -------
    :
        List of tuples with chromosome, start, end, and gene name.
    """
    if type(names) is str:
        names = [names]
    all_genes = all_gene_coords(assembly)
    coords = []
    for name in names:
        try:
            coords.append((*all_genes[name], name))
        except KeyError:
            print(f"{name} not found", file=sys.stderr)
    return coords 


@shelve_it()
def gene_labels_region(chrom:str, start:int, end:int, assembly:str) -> List[Tuple[str, int, str]]:
    """
    Gets gene labels for genes in a chromosomal region. For use with the add_labels
    method of GenomeIdeogram and ChromIdeogram.

    Parameters
    ----------
    chrom : 
        Chromosome identifier, e.g. 'chr1'
    start : 
        Start coordinate of region
    end : 
        Start coordinate of region
    assembly : 
        Assembly identifier, e.g. 'hg38' or 'rheMac10'

    Returns
    -------
    :
        List of tuples with chromosome, position, and gene name.
    """
    df = gene_coords(chrom, start, end, assembly)
    return list(zip(df.chrom, ((df.start+df.end)/2).astype(int), df.symbol))


@shelve_it()
def gene_labels(names:Sequence, assembly:str) -> List[Tuple[str, int, str]]:
    """
    Gets gene labels for a set of genes. For use with the add_labels
    method of GenomeIdeogram and ChromIdeogram.

    Parameters
    ----------
    names : 
        List of gene names.
    assembly : 
        Assembly identifier, e.g. 'hg38' or 'rheMac10'

    Returns
    -------
    :
        List of tuples with chromosome, position, and gene name.
    """
    if type(names) is str:
        names = [names]
    all_genes = all_gene_coords(assembly)
    coords = []
    for name in names:
        try:
            chrom, start, end = all_genes[name]
        except KeyError:
            print(f"{name} not found", file=sys.stderr)
        coords.append((chrom, (start + end)/2, name))
    return coords 


# @shelve_it()
# def get_genes_region(chrom:str, window_start:int, window_end:int, 
#                      assembly:str='GRCh38', db:str='ncbiRefSeq') -> list:
#     """
#     Gets gene structure information for genes in a chromosomal region.

#     Parameters
#     ----------
#     chrom : 
#         Chromosome identifier
#     window_start : 
#         Start of region
#     window_end : 
#         End of region (end base not included)
#     assembly : 
#         Genome assembly, by default 'GRCh38'
#     db : 
#         Database, by default 'ncbiRefSeq'

#     Returns
#     -------
#     :
#         List of gene information. Each gene is a tuple with the following elements:
#         - gene name
#         - gene start
#         - gene end
#         - gene strand
#         - list of exons (start, end)
#     """
#     api_url = f'https://api.genome.ucsc.edu/getData/track'
#     params = {'track': db,
#               'genome': assembly,
#               'chrom': chrom,
#               'start': window_start,
#               'end': window_end
#               }
#     response = requests.get(api_url, params=params)
#     if not response.ok:
#         response.raise_for_status()

#     genes = []
#     for gene in response.json()[db]:
#         exon_starts = [int(x) for x in gene['exonStarts'].split(',') if x]
#         exon_ends = [int(x) for x in gene['exonEnds'].split(',') if x]
#         exons = list(zip(exon_starts, exon_ends))
#         genes.append((gene['name2'], gene['txStart'], 
#                       gene['txEnd'], gene['strand'], exons))

#     return genes


# def get_genes_region_dataframe(chrom:str, window_start:int, window_end:int, 
#                      assembly:str=None, 
#                     #  assembly:str='GRCh38', 
#                      db:str='ncbiRefSeq') -> pd.DataFrame:
#     """
#     Gets gene structure information for genes in a chromosomal region in the form
#     of a pandas.DataFrame.

#     Parameters
#     ----------
#     chrom : 
#         Chromosome identifier
#     window_start : 
#         Start of region
#     window_end : 
#         End of region (end base not included)
#     assembly :  optional
#         Genome assembly, by default 'GRCh38'
#     db :  optional
#         Database, by default 'ncbiRefSeq'

#     Returns
#     -------
#     :
#         pandas.DataFrame with the following colunms:
#         - name: gene name
#         - start: gene start
#         - end: gene end
#         - strand: gene strand
#     """
#     try:
#         import pandas as pd
#     except ImportError:
#         print("pandas must be installed to return data frame")
#         return
#     genes = get_genes_region(chrom, window_start, window_end, assembly, db)
#     return pd.DataFrame().from_records([x[:4] for x in genes], 
#                                        columns=['name', 'start', 'end', 'strand'])


def gene_info_region(chrom:str, window_start:int, window_end:int, 
                    #  assembly:str='GRCh38', 
                     assembly:str=None) -> None:
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
        Genome assembly, e.g. 'hg38' or 'rheMac10'
    """
    for gene in gene_coords_region(chrom, window_start, window_end, assembly):    
        gene_info(gene[0])





@shelve_it()
def _ensembl_id(name:str, species:str) -> str:
    species = '_'.join(species.lower().split())
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


def ensembl_id(name:Union[str, list], species:str) -> str:
    """
    Get ENSEMBL ID for some gene identifier

    Parameters
    ----------
    name : 
        Gene identifier
    species : 
        Species latin name, e.g. "Homo sapiens"

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
        return _ensembl_id(name, species)
    else:
        return [_ensembl_id(x, species) for x in name]


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


def assemblies(species:str) -> str:
    species = '_'.join(species.lower().split())
    server = "https://rest.ensembl.org"
    ext = f"/info/assembly/{species}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
    decoded = r.json()
    genus, sp = species.split('_')

    markdown = (
        f"**{genus.capitalize()} {sp.lower()}:** "
         f"{decoded['default_coord_system_version']}. "
         f"Older: {','.join(decoded['coord_system_versions'])}"
    )
#    display(Markdown(markdown))
    return decoded['coord_system_versions']
    
    # symbols = [x['display_id'] for x in decoded if x['dbname'] == 'HGNC']
    # if not len(symbols) == 1:
    #     raise NotFound
    # return symbols[0]

# def assembly_info(assembly:str) -> str:
#     server = "https://rest.ensembl.org"
#     ext = f"/info/genomes/{assembly}?"
#     r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
#     if not r.ok:
#         r.raise_for_status()
#     decoded = r.json()
# #    display(Markdown(markdown))
#     return decoded


def hgcn_symbol(name:str|list) -> str:
    """
    Get HGCN gene symbol for some gene identifier

    Parameters
    ----------
    name : 
        Gene identifier or sequence of identifiers.

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
        return [ensembl2symbol(ensembl_id(n, species='Homo sapiens')) for n in name]
    else:
        return ensembl2symbol(ensembl_id(name, species='Homo sapiens'))


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
def mygene_get_gene_info(
    query, species='human', scopes='hgnc', 
    fields='symbol,alias,name,type_of_gene,summary,genomic_pos,genomic_pos_hg19'):

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


# @shelve_it()
# def gene_coord(query: Union[str, List[str]], species, assembly:str=None, 
#                pos_list=False) -> dict:
#     """
#     Retrieves genome coordinates one or more genes.

#     Parameters
#     ----------
#     query : 
#         Gene symbol or list of gene symbols
#     species :  
#         Species, E.g 'homo_sapiens'.
#     assembly :  
#         Genome assembly, by default most recent.
#     pos_list :
#         Wether to instead return a list of (chrom, position, name) tuples.

#     Returns
#     -------
#     :
#         Dictionary with gene names as keys and (chrom, start, end, strand) tuples 
#         as values, or a list of (chrom, position, name) tuples.
#     """

#     coords = {}
#     batch_size = 100
#     for i in range(0, len(query), batch_size):
#         data = ensembl_get_gene_info_by_symbol(
#             query[i:i+batch_size], assembly=assembly, species=species)
#         for name, props in data.items():
#             chrom, start, end, strand = (
#                 props['seq_region_name'], props['start'], props['end'], props['strand']
#             )
#             if not chrom.lower().startswith('contig') \
#                     and not chrom.lower().startswith('scaffold'):
#                 chrom = 'chr'+chrom
#             if strand == -1:
#                 strand = '-'
#             elif strand == 1:
#                 strand = '+'
#             else:
#                 strand = None
    
#             coords[name] = (chrom, start, end, strand)

#     if pos_list:
#         annot = []
#         for (name, (chrom, start, end, strand)) in coords.items():
#             annot.append((chrom, int((start+end)/2), name))
#         return sorted(annot, key=lambda x: x[1:])

#     return coords


def gene_info(query: Union[str, List[str]], 
              scopes:str='hgnc') -> None:
    """
    Displays HTML formatted information about one or more human genes.

    Parameters
    ----------
    query : 
        Gene symbol or list of gene symbols
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
                top_hit = mygene_get_gene_info(
                    gene, species='human', scopes=scopes,
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
                top_hit['hg38'] = ', '.join(['{chr}:{start}-{end}'.format(**d) 
                                             for d in top_hit['genomic_pos']])
            else:
                top_hit['hg38'] = '{chr}:{start}-{end}'.format(
                    **top_hit['genomic_pos'])
            if type(top_hit['genomic_pos_hg19']) is list:
                top_hit['hg19'] = ', '.join(['{chr}:{start}-{end}'.format(**d) 
                                             for d in top_hit['genomic_pos_hg19']])
            else:
                top_hit['hg19'] = '{chr}:{start}-{end}'.format(
                    **top_hit['genomic_pos_hg19'])            
            tmpl += "**Human genomic position:** {hg38} (hg38), {hg19} (hg19)  \n"

        tmpl += "[Gene card](https://www.genecards.org/"
        tmpl += "cgi-bin/carddisp.pl?gene={symbol})  \n".format(**top_hit)

        tmpl += "\n\n ----"

        display(Markdown(tmpl.format(**top_hit)))


@shelve_it()
def _ensembl_get_features_region(chrom, window_start, window_end, 
                                 features=['gene', 'exon'], assembly=None, 
                                 species=None):
                                #  species='homo_sapiens'):
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    window_start, window_end = int(window_start), int(window_end)
    genes = {}    
    for start in range(window_start, window_end, 500000):
        end = min(start+500000, window_end)
        param_str = ';'.join([f"feature={f}" for f in features])
        if assembly:
            url = f"https://{assembly.lower()}.rest.ensembl.org"
            api_url = f"{url}/overlap/region/{species}/{chrom}:{start}-{end}?{param_str}"
        else:
            url = "http://rest.ensembl.org"
            api_url = f"{url}/overlap/region/{species}/{chrom}:{start}-{end}?{param_str}"
        response = requests.get(api_url, headers={'content-type': 'application/json'})

        if not response.ok:
            response.raise_for_status()
        params = response.json()

        for gene in params:
            genes[gene['id']] = gene
            
    return genes


# @shelve_it()
# def ensembl_get_gene_info_by_symbol(symbols, assembly=None, species='homo_sapiens'):
def ensembl_get_gene_info_by_symbol(symbols, assembly=None, species=None):

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
    r = requests.post(server+ext, headers=headers, 
                      data=f'{{ "symbols": {json.dumps(symbols)} }}')
    if not r.ok:
        r.raise_for_status()
    return r.json()


def ensembl_get_genes_region(chrom, window_start, window_end, assembly=None, 
                             species=None):
                            #  species='homo_sapiens'):
    
    gene_info = _ensembl_get_features_region(
        chrom, window_start, window_end, features=['gene'], assembly=assembly, species=species)
    exon_info = _ensembl_get_features_region(
        chrom, window_start, window_end, features=['exon'], assembly=assembly, species=species)

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

