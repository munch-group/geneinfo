from pycirclize import Circos
from pycirclize.utils import ColorCycler, load_eukaryote_example_dataset
from collections import defaultdict
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations

from ..coords import chromosome_lengths

__all__ = ["circos_plot"]

def register_shifted_cmap(cmap_name, n_lines=24):
    name_shifted = f'shifted_{cmap_name}'
    n = n_lines // 2
    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, n))
    l = list(range(n))
    i = n//2
    colors = [colors[x] for y in zip(l, l[i:] + l[:i]) for x in y][:n_lines]
    _cmap = LinearSegmentedColormap.from_list(f'shifted_{cmap_name}', colors)
    try:
        matplotlib.colormaps.register(cmap=_cmap)
    except ValueError:
        # already registered, ignore
        pass
    return name_shifted

try:
    matplotlib.colormaps.register(
        cmap=LinearSegmentedColormap.from_list('black', ['#000000', '#000000'])
    )
except ValueError:
    # already registered, ignore
    pass
try:
    register_shifted_cmap('hsv')
except ValueError:
    # already registered, ignore
    pass

def circos_plot(connections, assembly, #reg, 
                ax=None,
                cmap='shifted_hsv', scalings={},
        ideogram_base = 97, ideogram_height = 3, figsize = (8, 8), link_kwargs={}):

    import matplotlib.pyplot as plt
    from geneinfo.coords import gene_coords

    _link_kwargs = dict(lw=0.5, alpha=1, zorder=0)
    _link_kwargs.update(link_kwargs)

    gene_coordinates = {}
    label_genes = set()
    for a, b in connections:
        if isinstance(a, str):
            a = gene_coords(a, assembly=assembly)[0]
        if isinstance(b, str):
            b = gene_coords(b, assembly=assembly)[0]
        print(a)
        label_genes.add(a[3])
        label_genes.add(b[3])
        gene_coordinates[a[3]] = a[:3]
        gene_coordinates[b[3]] = b[:3]

        stmts.append((a, b))
    # # label_genes = set([a.name for st in stmts for a in st.agent_list() if a])
    # gene_coordinates = {}
    # for chrom, start, end, name in gene_coords(label_genes, assembly=assembly):
    #     gene_coordinates[name] = (chrom, start, end)

    ColorCycler.set_cmap(cmap)
    sector_lengths = {chrom: length * scalings.get(chrom, 1) for chrom, length in chromosome_lengths[assembly].items()}
    
    circos = Circos(sectors=sector_lengths, space=3)
    chr_names = [s.name for s in circos.sectors]
    colors = ColorCycler.get_color_list(len(chr_names))
    chr_name2color = {name: color for name, color in zip(chr_names, colors)}

    gene_labels = defaultdict(list)
    for name, (chrom, start, end) in gene_coordinates.items():
        gene_labels[chrom].append([int((start+end)/2 * scalings.get(chrom, 1)), name])

    for sector in circos.sectors:
        sector.text(sector.name, r=105, size=8, color=chr_name2color[sector.name])
        outer_track = sector.add_track((ideogram_base, ideogram_base+ideogram_height))
        outer_track.axis(fc="#eeeeee")

        for pos, label in gene_labels.get(sector.name, []):
            outer_track.annotate(pos, label, label_size=7,
                                min_r=ideogram_base,
                                max_r=ideogram_base+5*ideogram_height,
                                )
    for st in stmts:
        agent_list = st.agent_list()
        if len(agent_list) < 2:
            print(f'Warning: statement with fewer than 2 agents, skipping: {st}')
            continue
        is_complex = type(st).__name__ == 'Complex'
        # if len(agent_list) > 2:
        #     print(f'More than than 2 agents in statement, skipping: {" ".join(a.name for a in agent_list if a)}')
        #     continue        
        for a, b in combinations(agent_list, 2):
            # for a, b in combinations(agent_list, 2):
            # a, b = agent_list
            if a and b and a.name in gene_coordinates and b.name in gene_coordinates:
                _from, _to = gene_coordinates[a.name], gene_coordinates[b.name]
                _from = (_from[0], _from[1]*scalings.get(_from[0], 1), _from[2]*scalings.get(_from[0], 1))
                _to = (_to[0], _to[1]*scalings.get(_to[0], 1), _to[2]*scalings.get(_to[0], 1))
                kwargs = _link_kwargs.copy()
                if 'color' not in kwargs:
                    kwargs['color'] = 'gray' if is_complex else chr_name2color[_from[0]                                                                               ]
                else:
                    if 'ls' in kwargs or 'linestyle' in kwargs:
                        print('Warning: linestyle specified in link_kwargs will be overridden for Complex statements to ensure they are dashed.')
                    kwargs['linestyle'] = (0, (5, 5)) if is_complex else 'solid'
                circos.link(_from, 
                            _to,
                            **kwargs)

    if ax is None:
        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(projection="polar")
    fig = circos.plotfig(ax=ax)
