import setuptools, os, sys

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="geneinfo",
    version="2.0.7",
    author="Kasper Munch",
    author_email="kaspermunch@birc.au.dk",
    description="Functions for showing gene information in jupyter notebooks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kaspermunch/geneinfo",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
          'jupyter',
          'matplotlib>=3.0',
          'numpy>=1.1',
          'requests',
          'biopython',
          'goatools'
        ])
