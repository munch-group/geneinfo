
from IPython.display import Markdown, display, Image, SVG, HTML
from contextlib import redirect_stdout
import importlib
import importlib.util
import sys
import os
import re
import subprocess
import pandas as pd
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

import matplotlib.axes
from matplotlib.patches import Rectangle, Polygon
import matplotlib.pyplot as plt

# from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag, OBOReader
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot
from goatools.cli.ncbi_gene_results_to_python import ncbi_tsv_to_py
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
from goatools.go_enrichment import GOEnrichmentRecord
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.go_search import GoSearch
from goatools.godag.go_tasks import CurNHigher
from goatools.godag_plot import (plot_gos, plot_goid2goobj, plot_results, 
                                 plt_goea_results)

from Bio import Entrez

from ..intervals import *

from ..information import hgcn_symbol, ensembl_id, NotFound


def email(email_address:str) -> None:
    """
    Registers your email address for Entrez queries. Thay way, NCBI will contect 
    you before closeing your connection if you are making too many queries.

    Parameters
    ----------
    email_address : 
        your email address
    """
    Entrez.email = email_address

def _assert_entrez_email():
    if not Entrez.email:
        print("""Please provide your email for Entrez queries:

import geneinfo.information as gi
gi.email("youremail@address.com)
""", file=sys.stderr)
        return

cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')

def download_ncbi_associations(prt=sys.stdout):

    # if not os.path.exists('geneinfo_cache'): os.makedirs('geneinfo_cache')
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    if not os.path.exists(os.path.join(cache_dir, 'gene2go')):
        process = subprocess.Popen(['wget', '-nv', '-O', f'{cache_dir}/gene2go.gz',
                                    'https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz'],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode(), file=prt)
        print(stderr.decode(), file=prt)
        assert not process.returncode

        process = subprocess.Popen(['gzip', '-f', '-d', f'{cache_dir}/gene2go.gz'],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode(), file=prt)
        print(stderr.decode(), file=prt)    
        assert not process.returncode, process.returncode
    return f'{cache_dir}/gene2go'


def download_and_move_go_basic_obo(prt=sys.stdout):  

    if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    if not os.path.exists(f'{cache_dir}/go-basic.obo'):
        # obo_fname = download_go_basic_obo(prt=prt)
        # shutil.move('go-basic.obo', 'geneinfo_cache/go-basic.obo')
        process = subprocess.Popen(['wget', '-nv', '-O', f'{cache_dir}/go-basic.obo',
                                    'https://purl.obolibrary.org/obo/go/go-basic.obo'],
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode(), file=prt)
        print(stderr.decode(), file=prt)
        assert not process.returncode

    return f'{cache_dir}/go-basic.obo'


def all_protein_coding(taxid=9606):
    return sorted(set(pd.read_csv(f'{cache_dir}/{taxid}_protein_genes.txt', sep='\t').Symbol))


def download_data(prt=sys.stdout):  

    download_ncbi_associations(prt)
    download_and_move_go_basic_obo(prt)


def _fetch_ids_to_file(id_list, output_file_name):

    with open(output_file_name, 'w') as f:
        header = ['tax_id', 'Org_name', 'GeneID', 'CurrentID', 'Status', 'Symbol', 
                  'Aliases', 'description', 'other_designations', 'map_location', 
                  'chromosome', 'genomic_nucleotide_accession.version', 
                  'start_position_on_the_genomic_accession', 
                  'end_position_on_the_genomic_accession', 'orientation', 
                  'exon_count', 'OMIM']
        print(*header, sep='\t', file=f)

        nr_genes_no_coordinates = 0

        batch_size = 2000
        for i in range(0, len(id_list), batch_size):
            to_fetch = id_list[i:i+batch_size]
            handle = Entrez.esummary(db="gene", id=",".join(to_fetch), 
                                     retmax=batch_size)
            entry = Entrez.read(handle)
            docsums = entry['DocumentSummarySet']['DocumentSummary']
            for doc in docsums:

                if doc['GenomicInfo']:
                    ver, start, stop, exon_count = (
                        doc['GenomicInfo'][0]['ChrAccVer'], 
                        doc['GenomicInfo'][0]['ChrStart'],
                        doc['GenomicInfo'][0]['ChrStop'],
                        doc['GenomicInfo'][0]['ExonCount'])
                else:
                    ver, start, stop, exon_count = 'unknown', pd.NA, pd.NA, pd.NA
                    nr_genes_no_coordinates += 1

                print(doc['Organism']['TaxID'], doc['Organism']['ScientificName'], 
                      doc.attributes['uid'], 
                      doc['CurrentID'], 
                      'live',
                      doc['Name'], doc['OtherAliases'], doc['Description'], 
                      doc['OtherDesignations'], doc['MapLocation'], 
                      doc['Chromosome'], ver, start, stop,
                      'notspecified', exon_count, '',
                      sep='\t', file=f)


def fetch_background_genes(taxid=9606):
    
    _assert_entrez_email()

    if not os.path.exists(cache_dir): os.makedirs(cache_dir)

    output_file_name = f'{cache_dir}/{taxid}_protein_genes.txt'   
    query = '[Taxonomy ID] AND alive[property] AND genetype protein coding[Properties]'     
    handle = Entrez.esearch(
        db="gene", 
        term=f'{taxid}{query}', 
        retmax="1000000")
    records = Entrez.read(handle)
    id_list = records["IdList"]

    _fetch_ids_to_file(id_list, output_file_name)

    # write mappings between symbol and ncbi id
    symbol2ncbi_file = f'{cache_dir}/{taxid}_symbol2ncbi.h5'
    df = pd.read_table(output_file_name)
    df = df.loc[:, ['GeneID', 'Symbol']]
    df.set_index('Symbol').GeneID.to_hdf(symbol2ncbi_file, key='symbol2ncbi')
    df.set_index('GeneID').Symbol.to_hdf(symbol2ncbi_file, key='ncbi2symbol')


def _cached_symbol2ncbi(symbols, taxid=9606):

    symbol2ncbi_file = f'{cache_dir}/{taxid}_symbol2ncbi.h5'
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
                    print(f'Could not map gene symbol "{symbol}" to ncbi id', 
                          file=sys.stderr)
        return geneids


def _cached_ncbi2symbol(geneids, taxid=9606):

    symbol2ncbi_file = f'{cache_dir}/{taxid}_symbol2ncbi.h5'
    symbol2ncbi = pd.read_hdf(symbol2ncbi_file, 'ncbi2symbol')
    try:
        return symbol2ncbi.loc[geneids].tolist()
    except KeyError:
        symbols = []
        for geneid in geneids:
            try:
                symbols.append(ncbi2symbol.loc[geneid])
            except KeyError:
                print(f'Could not map ncbi id "{geneid}" to gene symbol', 
                      file=sys.stderr)
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
    

def symbols_protein_coding(taxid:int=9606) -> list:
    """
    List of protein coding gene symbols for a given taxonomy id.

    Parameters
    ----------
    taxid : 
        NCBI taxonomy ID, by default 9606 (which is human)

    Returns
    -------
    :
        List of gene symbols.
    """
    fetch_background_genes(taxid=taxid)
    symbol2ncbi_file = f'{cache_dir}/{taxid}_symbol2ncbi.h5'
    symbol2ncbi = pd.read_hdf(symbol2ncbi_file, 'ncbi2symbol')
    return symbol2ncbi.tolist()


def get_terms_for_go_regex(regex:str, taxid:int=9606, add_children:bool=False) -> list:
    """
    Get GO terms for terms matching a regular expression in their description string.

    Parameters
    ----------
    regex : 
        Regular expression to match GO term descriptions.
    taxid : 
        NCBI taxonomy ID, by default 9606 (which is human)
    add_children : 
        Add GO terms nested under GO terms found, by default False

    Returns
    -------
    :
        List of GO terms.
    """

    taxid = _tidy_taxid(taxid)
        
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_and_move_go_basic_obo(prt=null)

        gene2go = download_ncbi_associations(prt=null)

        objanno = Gene2GoReader(f"{cache_dir}/gene2go", taxids=[taxid], prt=null)
        go2geneids = objanno.get_id2gos(namespace='*', go2geneids=True, prt=null)
        srchhelp = GoSearch(f"{cache_dir}/go-basic.obo", 
                            go2items=go2geneids, log=null)

        results_all = re.compile(r'({})'.format(regex), flags=re.IGNORECASE)
        results_not = re.compile(r'({}).independent'.format(regex), flags=re.IGNORECASE)

        gos_all = srchhelp.get_matching_gos(results_all, prt=null)
        gos_no = srchhelp.get_matching_gos(results_not, gos=gos_all)
        gos = gos_all.difference(gos_no)
        if add_children:
            gos = srchhelp.add_children_gos(gos)

        return list(gos)


def get_genes_for_go_regex(regex:str, taxid:int=9606) -> pd.DataFrame:
    """
    Get gene information for GO terms matching a regular expression in 
    their description string.

    Parameters
    ----------
    regex : 
        Regular expression to match GO term descriptions.
    taxid : 
        NCBI taxonomy ID, by default 9606 (which is human)

    Returns
    -------
    :
        Columns: symbol, name, chrom, start, end.
    """
    _assert_entrez_email()

    taxid = _tidy_taxid(taxid)
 
    with open(os.devnull, 'w') as null, redirect_stdout(null):

        gos_all_with_children = get_terms_for_go_regex(
            regex, taxid=taxid, add_children=True)

        objanno = Gene2GoReader(f"{cache_dir}/gene2go", taxids=[taxid], prt=null)
        go2geneids = objanno.get_id2gos(namespace='*', go2geneids=True, prt=null)
        srchhelp = GoSearch(f"{cache_dir}/go-basic.obo", go2items=go2geneids, 
                            log=null)
        geneids = srchhelp.get_items(gos_all_with_children)

        ncbi_tsv = f'{cache_dir}/{taxid}_protein_genes.txt'
        if not os.path.exists(ncbi_tsv):
            fetch_background_genes(taxid)

        output_py = f'{cache_dir}/{taxid}_protein_genes.py'
        ncbi_tsv_to_py(ncbi_tsv, output_py, prt=null)
        
        spec = importlib.util.spec_from_file_location("protein_genes", output_py)
        protein_genes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protein_genes)
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
                chrom_pos = (doc['Chromosome'], doc['GenomicInfo'][0]['ChrStart'],
                              doc['GenomicInfo'][0]['ChrStop'])
            except:
                print(f"WARNING: no coordinates for {doc['Name']} (pandas.NA)", 
                      file=sys.stderr)
                chrom_pos = (pd.NA, pd.NA, pd.NA)
            records.append((doc['Name'], doc['Description'], *chrom_pos))
            found.append(str(doc.attributes['uid']))
    missing = set(fetch_ids).difference(set(found))

    df = pd.DataFrame().from_records(
        records, columns=['symbol', 'name', 'chrom', 'start', 'end'])

    return df.sort_values(by='start').reset_index(drop=True)

    
def get_genes_for_go_terms(terms, taxid=9606) -> pd.DataFrame:
    """
    Get gene information for genes with specified GO terms.

    Parameters
    ----------
    terms : 
        List of GO terms
    taxid : 
        NCBI taxonomy ID, by default 9606 (which is human)

    Returns
    -------
    :
        Columns: symbol, name, chrom, start, end.
    """

    if type(terms) is not list:
        terms = [terms]

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_and_move_go_basic_obo(prt=null)
        gene2go = download_ncbi_associations(prt=null)
        objanno = Gene2GoReader(f"{cache_dir}/gene2go", taxids=[taxid], prt=null)
        go2geneids = objanno.get_id2gos(namespace='*', go2geneids=True, prt=null)
        srchhelp = GoSearch(f"{cache_dir}/go-basic.obo", go2items=go2geneids, 
                            log=null)

        geneids = srchhelp.get_items(terms)  

        ncbi_tsv = f'{cache_dir}/{taxid}_protein_genes.txt' 
        if not os.path.exists(ncbi_tsv):
            fetch_background_genes(taxid)

        output_py = f'{cache_dir}/{taxid}_protein_genes.py'
        ncbi_tsv_to_py(ncbi_tsv, output_py, prt=null)

    protein_genes = importlib.import_module(
        output_py.replace('.py', '').replace('/', '.'))
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
                chrom_pos = (doc['Chromosome'], doc['GenomicInfo'][0]['ChrStart'], 
                             doc['GenomicInfo'][0]['ChrStop'])
            except:
                print(f"WARNING: no coordinates for {doc['Name']} (pandas.NA)", 
                      file=sys.stderr)
                chrom_pos = (pd.NA, pd.NA, pd.NA)
            records.append((doc['Name'], doc['Description'], *chrom_pos))
            found.append(str(doc.attributes['uid']))
    missing = set(fetch_ids).difference(set(found))

    df = pd.DataFrame().from_records(
        records, columns=['symbol', 'name', 'chrom', 'start', 'end'])

    return df.sort_values(by='start').reset_index(drop=True)


def go_annotation_table(taxid:int=9606) -> pd.DataFrame:
    """
    GO annotations for a given taxonomy id as a pandas.DataFrame.

    Parameters
    ----------
    taxid : 
        NCBI taxonomy ID, by default 9606, which is human

    Returns
    -------
    pd.DataFrame
        GO annotations for the specified taxonomy id.
    """
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


def gene_annotation_table(taxid:int=9606) -> pd.DataFrame:
    """
    Gene annotations for a given taxonomy id as a pandas.DataFrame.

    Parameters
    ----------
    taxid : 
        NCBI taxonomy ID, by default 9606, which is human

    Returns
    -------
    pd.DataFrame
        Gene annotations for the specified taxonomy id.
    """

    ncbi_tsv = f'{cache_dir}/{taxid}_protein_genes.txt'
    if not os.path.exists(ncbi_tsv):
        fetch_background_genes(taxid)
    df = pd.read_table(ncbi_tsv)
    df.rename(columns={'tax_id': 'taxid'}, inplace=True)
    return df.loc[df['taxid'] == taxid]


def get_go_terms_for_genes(genes:Union[str,list], taxid:int=9606, 
                           evidence:list=None) -> list:
    """
    Get the union of GO terms for a list of genes.

    Parameters
    ----------
    genes : 
        Gene name or list of gene names.
    taxid : 
        NCBI taxonomy ID, by default 9606, which is human.
    evidence : 
        Evidence codes, by default None

    Returns
    -------
    :
        Go terms for the specified genes.
    """
    go_df = go_annotation_table(taxid)
    genes_df = gene_annotation_table(taxid)
    gene_ids = genes_df.loc[genes_df.Symbol.isin(genes)].GeneID

    df = go_df.loc[go_df.GeneID.isin(gene_ids)]
    if len(df.index) and evidence is not None:
        df = df.loc[df.Evidence.isin(evidence)]
        
    return list(sorted(df.GO_ID.unique().tolist()))

    
def show_go_dag_for_terms(terms:Union[list, pd.Series], 
                          add_relationships:bool=True) -> None:
    """
    Display GO graph for a list of GO terms.

    Parameters
    ----------
    terms : 
        Go terms
    add_relationships : 
        Add edges representing relationships between GO terms, by default True
    """
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
        obodag = GODag(f"{cache_dir}/go-basic.obo", optional_attrs=optional_attrs, 
                       prt=null)

        gosubdag = GoSubDag(terms, obodag, relationships=add_relationships) 
        GoSubDagPlot(gosubdag).plt_dag(f'{cache_dir}/plot.png')

    return display(Image(f'{cache_dir}/plot.png'))

# def show_go_dag_for_terms(terms, add_relationships=True):

#     with open(os.devnull, 'w') as null, redirect_stdout(null):
#         if add_relationships:
#             optional_attrs=['relationship', 'def']
#         else:
#             optional_attrs=['def']
#         obodag = GODag("go-basic.obo", optional_attrs=optional_attrs, prt=null)
#         plot_gos('plot.png', terms, obodag)
#     return Image('plot.png')  


# https://github.com/tanghaibao/goatools/
# blob/main/notebooks/goea_nbt3102_group_results.ipynb


def show_go_dag_for_gene(gene:str, taxid:int=9606, evidence:list=None, 
                         add_relationships:bool=True) -> None:
    """
    Displays GO graph for a given gene.

    Parameters
    ----------
    gene : 
        Gene symbol
    taxid : 
        NCBI taxonomy ID, by default 9606, which is human
    evidence : 
        Limiting list of evidence categories to include, by default None. 
        See `show_go_evidence_codes()`.
    add_relationships : 
        Add edges representing relationships between GO terms, by default True
    """

    # evidence codes: http://geneontology.org/docs/guide-go-evidence-codes/
    go_terms = get_go_terms_for_genes([gene], taxid=taxid, evidence=evidence)
    if not go_terms:
        print('No GO terms to show', file=sys.stderr)
        return
    return show_go_dag_for_terms(go_terms, add_relationships=add_relationships)

    
def show_go_evidence_codes() -> None:   
    """
    Display list of GO evidence categories and their codes.
    """

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
                    
        
def _write_go_hdf():

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        if not os.path.exists(f'{cache_dir}/go-basic.h5'):

            # Get http://geneontology.org/ontology/go-basic.obo
            obo_fname = download_and_move_go_basic_obo(prt=null)

            # Download Associations, if necessary
            # Get ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
            file_gene2go = download_ncbi_associations(prt=null)

            rows = [(e.id, e.name, e.defn) 
                    for e in OBOReader(obo_file=f'{cache_dir}/go-basic.obo', 
                                       optional_attrs=['def'])]
            df = pd.DataFrame().from_records(
                rows, columns=['goterm', 'goname', 'description'])
            df.to_hdf(f'{cache_dir}/go-basic.h5', key='df', format='table', 
                      data_columns=['goterm', 'goname'])


def go_term2name(term:str) -> str:
    """
    Converts a GO term to its name.

    Parameters
    ----------
    term : 
        GO term

    Returns
    -------
    :
        GO term name.
    """
    _write_go_hdf()
    with pd.HDFStore(f'{cache_dir}/go-basic.h5', 'r') as store:
        entry = store.select("df", "goterm == %r" % term).iloc[0]

    return entry.goterm


def go_name2term(name:str) -> str:
    """
    Converts a GO term name to its term.

    Parameters
    ----------
    name : 
        GO term name

    Returns
    -------
    :
        GO term.
    """
    _write_go_hdf()
    with pd.HDFStore(f'{cache_dir}/go-basic.h5', 'r') as store:
        entry = store.select("df", "goname == %r" % name.lower()).iloc[0]
    return entry.goterm


def go_info(terms:Union[str,List[str]]) -> None:
    """
    Displays HML formatted information about the given GO terms.

    Parameters
    ----------
    terms : 
        A GO term or list of GO terms to display information for.
    """

    if type(terms) is pd.core.series.Series:
        terms = terms.tolist()

    if type(terms) is not list:
        terms = [terms]

    _write_go_hdf()

    df = pd.read_hdf(f'{cache_dir}/go-basic.h5')
    for term in terms:
        entry = df.loc[df.goterm == term].iloc[0]
        desc = re.search(r'"([^"]+)"', entry.description).group(1)
        s = f'**<span style="color:gray;">{term}:</span>** ' + \
            f'**{entry.goname}**  \n {desc}    \n\n ----'        
        display(Markdown(s))


class WrSubObo(object):
    """
    Read a large GO-DAG from an obo file. Write a subset GO-DAG into
    a small obo file.
    """

    def __init__(self, fin_obo=None, optional_attrs=None, load_obsolete=None):
        self.fin_obo = fin_obo
        if fin_obo is not None:
            self.godag = GODag(fin_obo, optional_attrs, load_obsolete) 
        else:
            self.godag = None            
        self.relationships = \
            optional_attrs is not None and 'relationship' in optional_attrs

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
        """
        Get GO IDs whose name matches given name.
        """
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
        """
        Given GO ID sources and optionally the relationship attribute, 
        return all GO IDs.
        """
        go2obj_user = {}
        objrel = CurNHigher(self.relationships, self.godag)
        objrel.get_id2obj_cur_n_high(go2obj_user, go_sources)
        goids = set(go2obj_user)
        for goterm in go2obj_user.values():
            if goterm.alt_ids:
                goids.update(goterm.alt_ids)
        return goids

    def _prt_info(self, prt, goid_sources, goids_all):
        """
        Print information describing how this obo setset was created.
        """
        prt.write("! Contains {N} GO IDs. Created using {M} GO sources:\n".format(
            N=len(goids_all), M=len(goid_sources)))
        for goid in goid_sources:
            prt.write("!    {GO}\n".format(GO=str(self.godag.get(goid, ""))))
        prt.write("\n")


class My_GOEnrichemntRecord(GOEnrichmentRecord):

    def __str__(self):
        return f'<{self.GO}>'


def go_enrichment(gene_list:list, taxid:int=9606, background_chrom:str=None, 
                  background_genes:list=None, terms:list=None, 
                  list_study_genes:list=False, alpha:float=0.05) -> pd.DataFrame:
    """
    Runs a GO enrichment analysis.

    Parameters
    ----------
    gene_list : 
        List of gene symbols or NCBI gene ids.
    taxid : 
        NCBI taxonomy ID, 9606 (human) or 1758 (mouse), by default 9606.
    background_chrom : 
        Name of chromosome, by default None. Limits analysis to this named chromosome
    background_genes : 
        List of genes for use as background in GO enrichment analysis, 
        by default None
    terms : 
        List of GO terms for use as background in GO enrichment analysis,
        by default None
    list_study_genes : 
        Whether to include lists of genes responsible for enrichment for each 
        identified GO term, by default False
    alpha : 
        False discovery significance cut-off, by default 0.05

    Returns
    -------
    :
        pd.DataFrame with columns: 
        - namespace: (BP, MF, CC)
        - term_id: GO term
        - e/p: enrichment or depletion
        - pval_uncorr: uncorrected p-value
        - p_fdr_bh: Benjamini-Hochberg corrected p-value
        - ratio: ratio of study genes in GO term
        - bg_ratio: ratio of background genes in GO term
        - obj: GOEnrichmentRecord object

    Examples
    --------
    ```python
    gene_list = ['TP53', 'BRCA1', 'BRCA2', 'EGFR', 'KRAS', 'PTEN', 'CDH1', 
                 'ATM', 'CHEK2', 'PALB2']
    results = go_enrichment(gene_list, taxid=9606, alpha=0.05)
    show_go_dag_enrichment_results(results.obj)
    ```

    """
    if taxid not in [9606, 17580]:
        raise ValueError('Only human and mouse (tax id 9606 and 17580) are supported')
    
    if issubclass(type(gene_list), list):
        gene_list = list(gene_list)
    
    if type(gene_list) is pd.core.series.Series:
        gene_list = gene_list.tolist()
    if type(terms) is pd.core.series.Series:
        terms = terms.tolist()

    _assert_entrez_email()

    gene_list = list(gene_list)
    
    taxid = _tidy_taxid(taxid)

    ncbi_tsv = f'{cache_dir}/{taxid}_protein_genes.txt'
    if not os.path.exists(ncbi_tsv):
        fetch_background_genes(taxid)

    with open(os.devnull, 'w') as null, redirect_stdout(null):

        obo_fname = download_and_move_go_basic_obo(prt=null)

        file_gene2go = download_ncbi_associations(prt=null)

        obodag = GODag(f"{cache_dir}/go-basic.obo", 
                       optional_attrs=['relationship', 'def'], prt=null)

        # read NCBI's gene2go. Store annotations in a list of namedtuples
        objanno = Gene2GoReader(file_gene2go, taxids=[taxid])

        # get associations for each branch of the GO DAG (BP, MF, CC)
        ns2assoc = objanno.get_ns2assc()

        # limit go dag to a sub graph including only specified terms and their children
        if terms is not None:
            sub_obo_name = f'{cache_dir}/' + \
                str(hash(''.join(sorted(terms)).encode())) + '.obo'  
            wrsobo = WrSubObo(obo_fname, optional_attrs=['relationship', 'def'])
            wrsobo.wrobo(sub_obo_name, terms)    
            obodag = GODag(sub_obo_name, 
                           optional_attrs=['relationship', 'def'], prt=null)

        # load background gene set of all genes
        background_genes_file = f'{cache_dir}/{taxid}_protein_genes.txt'
        if not os.path.exists(background_genes_file):
            fetch_background_genes(taxid)

        # # load any custum subset
        if background_genes:
            if not all(type(x) is int for x in background_genes):
                if all(x.isnumeric() for x in background_genes):
                    background_genes = list(map(str, background_genes))
                else:
                    background_genes = _cached_symbol2ncbi(
                        background_genes, taxid=taxid)
            df = pd.read_csv(background_genes_file, sep='\t')
            no_suffix = os.path.splitext(background_genes_file)[0]
            background_genes_file = \
                f'{no_suffix}_{hash("".join(map(str, sorted(background_genes))))}.txt'            
            df.loc[df.GeneID.isin(background_genes)].to_csv(background_genes_file, 
                                                            sep='\t', index=False)

        # limit background gene set
        if background_chrom is not None:
            df = pd.read_csv(background_genes_file, sep='\t')
            background_genes_file = \
                f'{os.path.splitext(background_genes_file)[0]}_{background_chrom}.txt'            
            df.loc[df.chromosome == background_chrom].to_csv(
                background_genes_file, sep='\t', index=False)

        output_py = f'{cache_dir}/{taxid}_background.py'
        ncbi_tsv_to_py(background_genes_file, output_py, prt=null)

        background_genes = importlib.import_module(f'geneinfo.cache.{taxid}_background')
        # background_genes_name = output_py.replace('.py', '').replace('/', '.')
        # background_genes = importlib.import_module(background_genes_name, 'geneinfo.cache')
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
            
            # Hack. Changes __class__ of all instances...
            ntd.__class__ = My_GOEnrichemntRecord

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



def show_go_dag_enrichment_results(
        results:Union[List[GOEnrichmentRecord],pd.Series]) -> None:
    """
    Displays a GO enrichment analysis results.

    Parameters
    ----------
    results : 
        List or Series of GO result objejcts from `obj` column in the 
        `pd.DataFrame` returned by `go_enrichment()`.

    Examples
    --------
    ```python
    gene_list = ['TP53', 'BRCA1', 'BRCA2', 'EGFR', 'KRAS', 'PTEN', 
                 'CDH1', 'ATM', 'CHEK2', 'PALB2']
    results = go_enrichment(gene_list, taxid=9606, alpha=0.05)
    show_go_dag_enrichment_results(results.obj)
    ```
    """
    if type(results) is pd.core.series.Series:
        results = results.tolist()
    with open(os.devnull, 'w') as null, redirect_stdout(null):
        plot_results(f'{cache_dir}/plot.png', results)
    return display(Image(f'{cache_dir}/plot.png'))
