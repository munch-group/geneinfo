import setuptools, os, sys

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="geneinfo",
    version="3.4.42",
    author="Kasper Munch",
    author_email="kaspermunch@birc.au.dk",
    description="Functions for showing gene information in jupyter notebooks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/munch-group/geneinfo",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'geneinfo': [
        'data/*.csv',
        'data/*.pickle',
    ]},
    # package_data= {
        # # # all .dat files at any package depth
        # # '': ['**/*.dat'],
        # # into the data folder (being into a module) but w/o the init file
        # 'geneinfo.data': [ '**/*.csv', '**/*.picle', ]        
    # },    
    python_requires='>=3.6',
    install_requires=[
          'ipython',
          'jupyter',
          'matplotlib>=3.0',
          'numpy>=1.1',
          'requests',
          'biopython',
          'goatools>=1.2',
          'graphviz',
          'wget'
        ])
