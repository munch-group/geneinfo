
from IPython.display import Markdown, display
from collections import defaultdict
import sys
import os
import json
import pandas as pd
from typing import Any, TypeVar, List, Tuple, Dict, Union
import requests
from requests.auth import HTTPBasicAuth
import textwrap
import re
import unicodedata
from tqdm.notebook import tqdm
from rapidfuzz import process, fuzz
from pathlib import Path
from collections.abc import Sequence, MutableSequence

from ..intervals import *

from ..genelist import GeneList
from ..coords import gene_coords_region
#from ..information import *

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


    
def fetch_tracks(assembly):
    resp = requests.get(f"https://api.genome.ucsc.edu/list/tracks?genome={assembly}")
    resp.raise_for_status()
    tracks = resp.json()[assembly]
    tups = [(name, data['longLabel']) for name, data in tracks.items()]
    return sorted(tups, key=lambda t: t[0].upper())

def list_ucsc_tracks(assembly=None, label_wrap=80):
    search_ucsc_tracks(assembly=assembly,label_wrap=label_wrap)

def search_ucsc_tracks(*queries, assembly=None, label_wrap=80):

    assert assembly is not None
    
    def normalize(name):
        name = name.lower()
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        name = re.sub(r"[^\w\s]", "", name)
        return name.strip()

    tracks = fetch_tracks(assembly=assembly)
    track_names, track_labels = zip(*tracks)

    if not queries:  
        entries = tracks #names, labels = track_names, track_labels
    else:        
        normalized_track_names = [normalize(n) for n in track_names]
        matches = defaultdict(float)
        for query in queries:
            for word in query.split():
                # match = process.extractOne(query_normalized, normalized_names, scorer=fuzz.WRatio)
                # match = process.extractOne(query_normalized, normalized_names, scorer=fuzz.WRatio, score_cutoff=90.0)
                search = process.extract(normalize(word), normalized_track_names, scorer=fuzz.WRatio, score_cutoff=80.0, limit=100)
                for name, score, index in search:
                    matches[(name, index)] += score
        
        sorted_hits = sorted([(v, k) for k, v in matches.items()], reverse=True)

        entries = []
        for score, (name, index) in sorted_hits:
            entries.append((track_names[index], track_labels[index]))

    if entries:
        ljust = max([len(e[0]) for e in entries]) + 2
        for name, label in entries:
            print(name.ljust(ljust) + '\n'.join(textwrap.wrap(label, width=label_wrap, subsequent_indent=' '*ljust)))

def get_ucsc_track(track_name, assembly, chrom=None, start=None, end=None):
    url = "https://api.genome.ucsc.edu/getData/track"
    params = {"genome": assembly, "track": track_name}
    if chrom is not None:
        assert chrom.startswith('chr')
        params['chrom'] = chrom
        if start is not None and end is not None:
            params['start'] = str(start)
            params['end'] = str(end)            
    response = requests.get(url, params=params)
    if response.ok:
    #    response.raise_for_status()
        try:
            track_data = response.json().get(track_name, [])
        except json.JSONDecodeError:
            print("Paramters does not represent a valid query", file=sys.stderr)
            return

        if type(track_data) is list:
            print("Track has heterogenous data records. Returning only attributes (columns) shared by all entries.", file=sys.stderr)
            
            shared_keys = list(set([k for d in track_data for k in d]))
            _track_data = defaultdict(list)
            for d in track_data:
                for k, v in d.items():
                    _track_data[k].append(v)
            track_data = _track_data
                
        try:
            return pd.DataFrame(track_data)
        except ValueError:
            raise        

def download_4dn(identifier, user_4dn, secret_4dn, dowload_dir=os.getcwd(), 
                 pgbar=False
                 ):
    
    def download_file(url, dowload_dir=dowload_dir):
        if not os.path.exists(dowload_dir):
            os.makedirs(dowload_dir)
        file_name = url.split('/')[-1]
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_name, 'wb') as f:
                if pgbar:
                    pbar = tqdm(total=int(r.headers['Content-Length']))
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)         
                        if pgbar:
                            pbar.update(len(chunk))

    url = f"https://data.4dnucleome.org/ga4gh/drs/v1/objects/{identifier}/access/https"
    response = requests.get(url, auth=HTTPBasicAuth(user_4dn, secret_4dn))
    if not response.ok:
        assert 0
    info = response.json()    
    download_file(info["url"])