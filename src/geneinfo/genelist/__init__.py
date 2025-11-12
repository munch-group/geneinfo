import sys
from collections import UserList
# from scipy.stats import fisher_exact
from itertools import zip_longest
from math import sqrt
import pandas as pd
from IPython.display import Markdown, display

from ..coords import gene_coords
from ..utils import fisher_test

class GeneList(UserList):

    _highlight_color = '#1876D2'
    highlight_color = _highlight_color

    _markup = ['bold', 'color', 'underline', 'italic']
    markup = _markup

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @classmethod
    def set_highlight_color(cls, color):
        cls.highlight_color = color


    @classmethod
    def reset_highlight_color(cls):
        cls.highlight_color = cls._highlight_color


    def name(self, name=None):
        """
        Set/get a name for the gene list.
        """
        if name is None:
            if hasattr(self, '_name'):
                return self._name
            else:
                return 'unnamed'
        if not isinstance(name, str):
            raise ValueError('Name must be a string.')
        self._name = name
        return self


    def _distance_prune(self, other, distance, assembly):
        """
        Prune genes that are closer than distance to each other.
        """
        
        ovl = self & other
        coords = gene_coords(ovl, assembly=assembly)
        coords = sorted(coords, key=lambda x: (x[0], (x[1] + x[2]) // 2))
        pruned = []
        last_end = -distance
        for i in range(len(coords)):
            chrom, start, end, name = coords[i]
            if start - last_end < distance:
                pruned.append(name)
            else:
                last_end = end
        if not pruned:
            print('No genes removed', file=sys.stderr)
        else:
            print(f'Removed: {", ".join(sorted(pruned))}', file=sys.stderr)
                    
        return GeneList(sorted(set(self).difference(set(pruned))))


    def fisher(self, other, background, min_dist=None, return_counts=False):
        """
        Fisher's exact test for overlap to and another gene list.

        Parameters
        ----------
        other : 
            Other gene list.
        background : 
            Background gene list.
        min_dist : 
            Minimum distance between genes for distance pruning, by default None
        return_counts : 
            Return contingency table, by default False

        Returns
        -------
        :
            p-value from Fisher's exact test (and contingency table if return_counts is True).
        """
        if min_dist is not None:
            # not implemented yet
            raise NotImplementedError('Distance pruning not implemented yet in fisher method.')

        return fisher_test(self, other, background, min_dist=min_dist, return_counts=return_counts)

    # def fisher(self, other, background, min_dist=(None, None), return_counts=False):

    #     distance, assembly = min_dist
    #     a, b = self, other
    #     if min_dist[0] is not None:
    #         a = self._distance_prune(other, distance, assembly)

    #     M = len(background) 
    #     N = len(background & a) 
    #     n = len(background & b)
    #     x = len(background & a & b)
    #     table = [[  x,           n - x          ],
    #             [ N - x,        M - (n + N) + x]]
    #     if return_counts:
    #         return float(fisher_exact(table, alternative='greater').pvalue), table
    #     return float(fisher_exact(table, alternative='greater').pvalue)    
  

    # TODO: add alias mapping to GeneList
    def download_gene_aliases():
        """
        Download mapping from any alias to the cannonical hgcn name for use in set operations.
        """
        ...


    def _tabulate(self):
        """
        Turn list into square'ish matrix
        """
        n = len(self)
        col_width = max(map(len, self)) + 1
        ncols = min(max(100//col_width, 1+sqrt(n/col_width)), 150//col_width)
        # ncols = min(max(80//col_width, 1+sqrt(n/col_width)), 80//col_width)
        nrows = int(n/ncols) + 1
        rows = []
        for r in range(0, n, nrows):
            rows.append(self[r:r+nrows])
        return rows, col_width
        

    def __repr__(self):
        if not len(self):
            return '< Empty GeneList>'
        rows, col_width = self._tabulate()
        repr = []
        for row in list(zip_longest(*rows, fillvalue='')):
            line = []
            for gene in row:
                line.append(gene.ljust(col_width))
            repr.append(''.join(line))
        return('\n'.join(repr))


    def _repr_html_(self):
        if not len(self):
            return '< Empty GeneList >'
        try:
            rows, col_width = self._tabulate()
            style = 'background: transparent!important; line-height: 10px!important;text-align: left!important;'
            table = [f'<table data-quarto-disable-processing="true" {style}>']
            for row in list(zip_longest(*rows, fillvalue='')):
                table.append(f'<tr style="{style}">')
                for gene in row:
                    td_styles = []                
                    if hasattr(self, '_bold') and gene in self._bold:
                        td_styles.append('font-weight: bold;')
                    if hasattr(self, '_color') and gene in self._color:
                        td_styles.append(f'color:{self.highlight_color};')
                    if hasattr(self, '_underline') and gene in self._underline:
                        td_styles.append('text-decoration: underline;')
                    if hasattr(self, '_italic') and gene in self._italic:
                        td_styles.append('font-style: italic;')
                    td_style = style + ' '.join(td_styles)
                    table.append(f'<td style="{td_style}">{gene}</td>')
                table.append('</tr>')
            table.append('</table>')
            self._strip_styles()
            return '\n'.join(table)
        except Exception as e:
            self._strip_styles()
            raise e


    def expand_amplicon_abbrev(self):

        new_list = []
        for gene_name in self:
            abbrev = gene_name.rsplit('_', 1)[0]
            abbrev = abbrev.replace('-', '_')
            if abbrev in AMPL_ABBREV_MAP.keys():
                new_list.extend(AMPL_ABBREV_MAP[abbrev])
            else:
                new_list.append(gene_name)

        # new_list = []
        # for gene_name in old_list:
        #     if gene_name.startswith('amplicon') and '/' in gene_name:
        #         prefix, *variants = gene_name.split('/')
        #         first_amplicon = re.split(r'[_-]+', prefix, 2)[-1]
        #         new_list.append(first_amplicon)
        #         for var in variants:
        #             ampl_name = first_amplicon[:-1] + var
        #             new_list.append(ampl_name)
        #     else:
        #         new_list.append(gene_name)

        self.data = sorted(set(new_list))
    

    def __str__(self):
        return repr(self)


    def _strip_styles(self):
        for attr in ['_bold', '_color', '_italic', '_underline']:
            if hasattr(self, attr):
                delattr(self, attr)
        return self


    def __lshift__(self, other):
        for style in self.markup:
            if not hasattr(self, f'_{style}'):
                setattr(self, f'_{style}', list(other))            
                break
        # if not hasattr(self, '_bold'):
        #     setattr(self, '_bold', list(other))
        # elif not hasattr(self, '_color'):
        #     setattr(self, '_color', list(other))
        # elif not hasattr(self, '_underline'):
        #     setattr(self, '_underline', list(other))
        # elif not hasattr(self, '_italic'):
        #     setattr(self, '_italic', list(other))
        else:
            self._strip_styles()
            raise ValueError('Do not provide more than two three highlight lists')        
        return self


    def __or__(self, other):
        self._strip_styles()
        other._strip_styles()
        return GeneList(sorted(set(self.data + other.data)))


    def __and__(self, other):
        self._strip_styles()
        other._strip_styles()        
        return GeneList(sorted(set(self.data).intersection(set(other.data))))


    def __xor__(self, other):
        self._strip_styles()
        other._strip_styles()        
        inter = set(self.data).intersection(set(other.data))
        union = set(self.data + other.data)
        return GeneList(sorted(union.difference(inter)))


AMPL_ABBREV_MAP = {    
 'amplicon_chrX_CPXCR1': ['CPXCR1'],
 'amplicon_chrX_CSAG1/2/3': ['CSAG1', 'CSAG2', 'CSAG3'],
 'amplicon_chrX_CT45A1/2/3//6/7/8/9/10': ['CT45A1', 'CT45A2', 'CT45A3', 'CT45A6', 'CT45A7', 'CT45A8', 'CT45A9', 'CT45A10'],
 'amplicon_chrX_CT47A1/2/3/4/5/6/7/8/9/10/11/12/B1': ['CT47A1', 'CT47A2', 'CT47A3', 'CT47A4', 'CT47A5', 'CT47A6', 'CT47A7', 'CT47A8', 'CT47A9', 'CT47A10', 'CT47A11', 'CT47A12', 'CT47B1'],
 'amplicon_chrX_CT55': ['CT55'],
 'amplicon_chrX_CT83': ['CT83'],
 'amplicon_chrX_CTAG1A/1B/2': ['CTAG1A', 'CTAG1B', 'CTAG2'],
 'amplicon_chrX_CXorf49/B': ['CXorf49', 'CXorf49B'],
 'amplicon_chrX_CXorf51A/B': ['CXorf51A', 'CXorf51B'],
 'amplicon_chrX_DDX53': ['DDX53'],
 'amplicon_chrX_DMRTC1/B/FAM236A/B/C/D': ['DMRTC1', 'DMRTC1B', 'FAM236A', 'FAM236B', 'FAM236C', 'FAM236D'],
 'amplicon_chrX_EOLA1/2/HSFX3/4': ['EOLA1', 'EOLA2', 'HSFX3', 'HSFX4'],
 'amplicon_chrX_ETD1/B/ZNF75D': ['ETD1', 'ETD1B', 'ZNF75D'],
 'amplicon_chrX_F8/F8A1/2/3/H2AB1/2/3': ['F8', 'F8A1', 'F8A2', 'F8A3', 'H2AB1', 'H2AB2', 'H2AB3'],
 'amplicon_chrX_FAM156A/B': ['FAM156A', 'FAM156B'],
 'amplicon_chrX_FAM47A/B/C': ['FAM47A', 'FAM47B', 'FAM47C'],
 'amplicon_chrX_G6PD/IKBKG': ['G6PD', 'IKBKG'],
 'amplicon_chrX_GAGE10/1/2A/13/12B/12C/12D/12E/12F/12G/12H/12J': ['GAGE10', 'GAGE1', 'GAGE2A', 'GAGE13', 'GAGE12B', 'GAGE12C', 'GAGE12D', 'GAGE12E', 'GAGE12F', 'GAGE12G', 'GAGE12H', 'GAGE12J'],
 'amplicon_chrX_HSFX1/2': ['HSFX1', 'HSFX2'],
 'amplicon_chrX_IL3RA/P2RY8/SLC25A6': ['IL3RA', 'P2RY8', 'SLC25A6'],
 'amplicon_chrX_MAGEA4': ['MAGEA4'],
 'amplicon_chrX_MAGEA12/A2/A2B/A3/A6': ['MAGEA12', 'MAGEA2', 'MAGEA2B', 'MAGEA3', 'MAGEA6'],
 'amplicon_chrX_MAGEA9/9B': ['MAGEA9', 'MAGEA9B'],
 'amplicon_chrX_MAGEB6': ['MAGEB6'],
 'amplicon_chrX_MAGEC1': ['MAGEC1'],
 'amplicon_chrX_MBTPS2/YY2': ['MBTPS2', 'YY2'],
 'amplicon_chrX_NSDHL': ['NSDHL'],
 'amplicon_chrX_NUDT10/11': ['NUDT10', 'NUDT11'],
 'amplicon_chrX_NXF2/2B/5': ['NXF2', 'NXF2B', 'NXF5'],
 'amplicon_chrX_PABPC1L2A/B': ['PABPC1L2A', 'PABPC1L2B'],
 'amplicon_chrX_PAGE2/2B/5': ['PAGE2', 'PAGE2B', 'PAGE5'],
 'amplicon_chrX_RHOXF2/B': ['RHOXF2', 'RHOXF2B'],
 'amplicon_chrX_SPACA5/B': ['SPACA5', 'SPACA5B'],
 'amplicon_chrX_SPANXA1/A2/N1/N2/N3/N4/N5/B1/C/D': ['SPANXA1', 'SPANXA2', 'SPANXN1', 'SPANXN2', 'SPANXN3', 'SPANXN4', 'SPANXN5', 'SPANXB1', 'SPANXC', 'SPANXD'],
 'amplicon_chrX_SSX1/2/2B/344B/5/7': ['SSX1', 'SSX2', 'SSX2B', 'SS344B', 'SSX5', 'SSX7'],
 'amplicon_chrX_SUPT20HL1/2': ['SUPT20HL1', 'SUPT20HL2'],
 'amplicon_chrX_TCEAL2/3/4/5/6': ['TCEAL2', 'TCEAL3', 'TCEAL4', 'TCEAL5', 'TCEAL6'],
 'amplicon_chrX_TCP11X1/2': ['TCP11X1', 'TCP11X2'],
 'amplicon_chrX_TEX28': ['TEX28'],
 'amplicon_chrX_TMEM185A': ['TMEM185A'],
 'amplicon_chrX_VCX/2/3A/3B': ['VCX', 'VCX2', 'VCX3A', 'VCX3B'],
 'amplicon_chrX_XAGE1A/B': ['XAGE1A', 'XAGE1B'],
 'amplicon_chrX_XAGE3': ['XAGE3'],
 'amplicon_chrX_XAGE5': ['XAGE5'],
}


yaml_format = """
<list_label>:
  description: |
    <free text description of the gene list>
    <free text description of the gene list>
  genes: <gene_name>, <gene_name>, ...
<list_label>:
  description: |
    <free text description of the gene list>
    <free text description of the gene list>
  genes: <gene_name>, <gene_name>, ...
"""

class GeneListCollection(object):

    def __init__(self, yaml_file=None, url:str=None, google_sheet:str=None, tab='Sheet1'):

        if yaml_file is not None:
            import yaml
            with open(yaml_file) as f:
                try:
                    self.data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    print(f"Yaml must honor this this format:\n\n{yaml_format}\n\n", file=sys.stderr)
                    raise e
            for name in self.data.keys():
                if 'genes' not in self.data[name]:
                    raise ValueError(f'Gene list {name} must have a "genes" entry.')
                if 'description' not in self.data[name]:
                    self.data[name]['description'] = ''
        else:
            assert url or google_sheet, 'Either file/url or google_sheet id must be provided.'

            if url is None:
                url = f'https://docs.google.com/spreadsheets/d/{google_sheet}/gviz/tq?tqx=out:csv&sheet={tab}'

            self.desc = []
            for desc in pd.read_csv(url, header=None, low_memory=False).iloc[0]:
                if str(desc) == 'nan':
                    self.desc.append('')
                else:
                    self.desc.append(desc.replace('\n', ' '))
            self.df = pd.read_csv(url, header=1, low_memory=False)
            self.df = self.df.loc[:, [not x.startswith('Unnamed') for x in self.df.columns]]
            self.names = self.df.columns.tolist()
            self.data = {}
            for name, desc in zip(self.names, self.desc):
                self.data[name] = {}
                sr = self.df[name]
                self.data[name]['genes'] = self.df.loc[~sr.isnull(), name].to_list()
                self.data[name]['description'] = desc

    def all_genes(self):
        gene_names = []
        for list_name in self.data.keys():
            gene_names.extend(self.get(list_name))
        return GeneList(sorted(set(gene_names)))
    

    def get(self, name):
        # sr = self.df[name]
        # sr = self.df.loc[~sr.isnull(), name]
        # # lst = sorted(self.expand_amplicon_abbrev(sr.tolist()))
        # lst = sr.tolist()
        return GeneList(self.data[name]['genes']).name(name)


    def _repr_html_(self):
        records = []
        for name in self.data.keys():
            desc = self.data[name]['description']
            if pd.isnull(desc):
                desc = ''
            records.append((name, desc))
        df = pd.DataFrame.from_records(records, columns=['List label', 'Description'])
        # return df._repr_html_()

        # s = df.style.pipe(make_pretty)
        # s.set_table_styles(
        #     {c: [{'selector': '', 'props': [('text-align', 'left')]}] 
        #          for c in df.columns if is_object_dtype(df[c]) and c != 'strand'},
        #     overwrite=False
        # )
        # display(s)

        s = df.style.set_table_styles(
            {c: [{'selector': '', 'props': [('text-align', 'left')]}] 
                 for c in df.columns},
            overwrite=False
        ).hide(axis="index")
        
        return s._repr_html_()


        # out = ['| label | description |', '|:---|:---|']
        # for name in self.data.keys():
        #     desc = self.data[name]['description']
        #     if pd.isnull(desc):
        #         desc = ''
        #     # out.append(f"- **{(name+':**').ljust(130)} {desc}")
        #     out.append(f"| **{name}** | {desc} |")
            
        # display(Markdown('\n'.join(out)))


    def __repr__(self):
        return ""
  
  
    def __iter__(self):
         yield from self.data.keys()
