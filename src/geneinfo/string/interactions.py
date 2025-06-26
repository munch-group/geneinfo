
from IPython.display import display, SVG
import io
import os
import pandas as pd
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

import requests

from ..intervals import *

from ..genelist import GeneList


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


def show_string_network(my_genes:Union[list,str], nodes:int=10) -> None:
    """
    Display STRING network for a list of genes.

    Parameters
    ----------
    my_genes : 
        List of gene symbols
    nodes : 
        Number of nodes to show, by default 10
    """

    if not os.path.exists('geneinfo_cache'): os.makedirs('geneinfo_cache')

    if isinstance(my_genes, GeneList):
        my_genes = list(my_genes)

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
    display(SVG('geneinfo_cache/network.svg'))


def string_network_table(my_genes:list, nodes:int=10) -> pd.DataFrame:
    """
    Retrieves STRING network for a list of genes and returns it as a pandas.DataFrame.

    Parameters
    ----------
    my_genes : 
        List of gene symbols
    nodes : 
        Number of nodes to show, by default 10

    Returns
    -------
    :
        STRING network information for specified genes.
    """
    if isinstance(my_genes, GeneList):
        my_genes = list(my_genes)
    
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