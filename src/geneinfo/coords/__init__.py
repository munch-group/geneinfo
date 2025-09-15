from collections import defaultdict
import sys
import os
import pandas as pd
from typing import Any, TypeVar, List, Tuple, Dict, Union
import requests
from pathlib import Path
from collections.abc import Sequence, MutableSequence

from ..utils import shelve_it, chrom_sort_key

cache_dir = Path(os.path.dirname(__file__)).parent / 'data'


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
    if assembly == 'hs1':
        return [
            ('chr1', 248387328),
            ('chr2', 242696752),
            ('chr3', 201105948),
            ('chr4', 193574945),
            ('chr5', 182045439),
            ('chr6', 172126628),
            ('chr7', 160567428),
            ('chr8', 146259331),
            ('chr9', 150617247),
            ('chr10', 134758134),
            ('chr11', 135127769),
            ('chr12', 133324548),
            ('chr13', 113566686),
            ('chr14', 101161492),
            ('chr15', 99753195),
            ('chr16', 96330374),
            ('chr17', 84276897),
            ('chr18', 80542538),
            ('chr19', 61707364),
            ('chr20', 66210255),
            ('chr21', 45090682),
            ('chr22', 51324926),
            ('chrX', 154259566),
            ('chrY', 62460029),
        ]

    api_url = f'https://api.genome.ucsc.edu/list/chromosomes?genome={assembly};track=gold'
    response = requests.get(api_url)
    if not response.ok:
        response.raise_for_status()
    records = []
    for chrom, length in response.json()['chromosomes'].items():
        if '_' in chrom or chrom == 'chrMT' or chrom == 'chrM':
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

    if assembly == 'hs1':
        return [
            ('chr1', 121796048, 126300487),
            ('chr2', 92333543, 94673023),
            ('chr3', 91738002, 96415026),
            ('chr4', 49705154, 55199795),
            ('chr5', 47039134, 49596625),
            ('chr6', 58286706, 61058390),
            ('chr7', 60414372, 63714499),
            ('chr8', 44215832, 46325080),
            ('chr9', 44951775, 47582595),
            ('chr10', 39633793, 41664589),
            ('chr11', 51035789, 54450838),
            ('chr12', 34620838, 37202490),
            ('chr13', 15547593, 17498291),
            ('chr14', 10092112, 12708411),
            ('chr15', 16678794, 17694466),
            ('chr16', 35848286, 37829521),
            ('chr17', 23892419, 27486939),
            ('chr18', 15965699, 20933550),
            ('chr19', 25817676, 29768171),
            ('chr20', 26925852, 29099655),
            ('chr21', 10962853, 11306205),
            ('chr22', 12788180, 15711065),
            ('chrX', 57820107, 60927025),
            ('chrY', 10565750, 10883085),
        ]

    api_url = f'https://api.genome.ucsc.edu/getData/track?genome={assembly};track=centromeres'
    response = requests.get(api_url)
    if not response.ok:
        return []
    data = defaultdict(list)
    for chrom, val in response.json()['centromeres'].items():
        for d in val:
            if '_' in d['chrom'] or d['chrom'] == 'chrMT' or chrom == 'chrM':
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
    df = gene_coords_region(chrom, start, end, assembly, as_dataframe=True)
    return list(zip(df.chrom, ((df.start+df.end)/2).astype(int), df.name))


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




