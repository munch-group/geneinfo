import sys
from functools import partial
import itertools
import random
from collections import defaultdict
import numpy as np
from matplotlib.pyplot import subplots
from matplotlib.patches import Ellipse, Polygon, Patch
from matplotlib.colors import to_rgba
from matplotlib.cm import ScalarMappable

SHAPE_COORDS = {
    2: [(.375, .500), (.625, .500)],
    # 3: [(.333, .633), (.666, .633), (.500, .310)],
    3: [(.333, .633), (.666, .633), (.500, .310+0.07)],
    4: [(.350, .400), (.450, .500), (.544, .500), (.644, .400)],
    5: [(.428, .449), (.469, .543), (.558, .523), (.578, .432), (.489, .383)],
    6: [
        (.637, .921, .649, .274, .188, .667),
        (.981, .769, .335, .191, .393, .671),
        (.941, .397, .292, .475, .456, .747),
        (.662, .119, .316, .548, .662, .700),
        (.309, .081, .374, .718, .681, .488),
        (.016, .626, .726, .687, .522, .327)
    ]
}

SHAPE_DIMS = {
    2: [(.50, .50), (.50, .50)],
    3: [(.50, .50), (.50, .50), (.50, .50)],
    4: [(.72, .45), (.72, .45), (.72, .45), (.72, .45)],
    5: [(.87, .50), (.87, .50), (.87, .50), (.87, .50), (.87, .50)],
    6: [(None,)]*6
}

SHAPE_ANGLES = {
    2: [0, 0],
    3: [0, 0, 0],
    4: [140, 140, 40, 40],
    5: [155, 82, 10, 118, 46], 6: [None]*6
}

PETAL_LABEL_COORDS = {
    2: {
        "01": (.74, .50), "10": (.26, .50), "11": (.50, .50)
    },
    # 3: {
    #     "001": (.500, .270), "010": (.730, .650), "011": (.610, .460),
    #     "100": (.270, .650), "101": (.390, .460), "110": (.500, .650),
    #     "111": (.500, .508)
    # },
    3: {
        "001": (.500, .270 + .07), "010": (.730, .650), "011": (.610+0.02, .460 + .03),
        "100": (.270, .650), "101": (.390-0.02, .460 + .03), "110": (.500, .650+0.05),
        "111": (.500, .508 + .03)
    },    4: {
        "0001": (.85, .42), "0010": (.68, .72), "0011": (.77, .59),
        "0100": (.32, .72), "0101": (.71, .30), "0110": (.50, .66),
        "0111": (.65, .50), "1000": (.14, .42), "1001": (.50, .17),
        "1010": (.29, .30), "1011": (.39, .24), "1100": (.23, .59),
        "1101": (.61, .24), "1110": (.35, .50), "1111": (.50, .38)
    },
    5: {
        "00001": (.27, .11), "00010": (.72, .11), "00011": (.55, .13),
        "00100": (.91, .58), "00101": (.78, .64), "00110": (.84, .41),
        "00111": (.76, .55), "01000": (.51, .90), "01001": (.39, .15),
        "01010": (.42, .78), "01011": (.50, .15), "01100": (.67, .76),
        "01101": (.70, .71), "01110": (.51, .74), "01111": (.64, .67),
        "10000": (.10, .61), "10001": (.20, .31), "10010": (.76, .25),
        "10011": (.65, .23), "10100": (.18, .50), "10101": (.21, .37),
        "10110": (.81, .37), "10111": (.74, .40), "11000": (.27, .70),
        "11001": (.34, .25), "11010": (.33, .72), "11011": (.51, .22),
        "11100": (.25, .58), "11101": (.28, .39), "11110": (.36, .66),
        "11111": (.51, .47)
    },
    6: {
        "000001": (.212, .562), "000010": (.430, .249), "000011": (.356, .444),
        "000100": (.609, .255), "000101": (.323, .546), "000110": (.513, .316),
        "000111": (.523, .348), "001000": (.747, .458), "001001": (.325, .492),
        "001010": (.670, .481), "001011": (.359, .478), "001100": (.653, .444),
        "001101": (.344, .526), "001110": (.653, .466), "001111": (.363, .503),
        "010000": (.750, .616), "010001": (.682, .654), "010010": (.402, .310),
        "010011": (.392, .421), "010100": (.653, .691), "010101": (.651, .644),
        "010110": (.490, .340), "010111": (.468, .399), "011000": (.692, .545),
        "011001": (.666, .592), "011010": (.665, .496), "011011": (.374, .470),
        "011100": (.653, .537), "011101": (.652, .579), "011110": (.653, .488),
        "011111": (.389, .486), "100000": (.553, .806), "100001": (.313, .604),
        "100010": (.388, .694), "100011": (.375, .633), "100100": (.605, .359),
        "100101": (.334, .555), "100110": (.582, .397), "100111": (.542, .372),
        "101000": (.468, .708), "101001": (.355, .572), "101010": (.420, .679),
        "101011": (.375, .597), "101100": (.641, .436), "101101": (.348, .538),
        "101110": (.635, .453), "101111": (.370, .548), "110000": (.594, .689),
        "110001": (.579, .670), "110010": (.398, .670), "110011": (.395, .653),
        "110100": (.633, .682), "110101": (.616, .656), "110110": (.587, .427),
        "110111": (.526, .415), "111000": (.495, .677), "111001": (.505, .648),
        "111010": (.428, .663), "111011": (.430, .631), "111100": (.639, .524),
        "111101": (.591, .604), "111110": (.622, .477), "111111": (.501, .523)
    }
}

PSEUDOVENN_PETAL_COORDS = {
    6: {
        "100000": (.275, .875), "010000": (.725, .875), "001000": (.925, .500),
        "000100": (.725, .125), "000010": (.275, .125), "000001": (.075, .500),
        "110000": (.500, .850), "011000": (.800, .675), "001100": (.800, .325),
        "000110": (.500, .150), "000011": (.200, .325), "100001": (.200, .675),
        "110001": (.375, .700), "111000": (.625, .700), "011100": (.750, .500),
        "001110": (.625, .300), "000111": (.375, .300), "100011": (.250, .500),
        "111001": (.500, .650), "111100": (.635, .575), "011110": (.635, .415),
        "001111": (.500, .350), "100111": (.365, .415), "110011": (.365, .575),
        "111011": (.440, .600), "111101": (.560, .600), "111110": (.620, .500),
        "011111": (.560, .400), "101111": (.440, .400), "110111": (.380, .500),
        "111111": (.500, .500)
    }
}


def generate_colors(cmap="viridis", n_colors=6, alpha=.4):
    """Generate colors from matplotlib colormap; pass list to use exact colors"""
    if not isinstance(n_colors, int) or (n_colors < 2) or (n_colors > 6):
        raise ValueError("n_colors must be an integer between 2 and 6")
    if isinstance(cmap, list):
        colors = [to_rgba(color, alpha=alpha) for color in cmap]
    else:
        scalar_mappable = ScalarMappable(cmap=cmap)
        colors = scalar_mappable.to_rgba(range(n_colors), alpha=alpha).tolist()
    return colors[:n_colors]

def less_transparent_color(color, alpha_factor=2):
    """Bump up color's alpha"""
    new_alpha = (1 + to_rgba(color)[3]) / alpha_factor
    return to_rgba(color, alpha=new_alpha)

def draw_ellipse(ax, x, y, w, h, a, color):
    """Wrapper for drawing ellipse; called like `draw_ellipse(ax, *coords, *dims, angle, color)`"""
    ax.add_patch(
        Ellipse(
            xy=(x,y), width=w, height=h, angle=a,
            facecolor=color, edgecolor=less_transparent_color(color)
        )
    )

def draw_triangle(ax, x1, y1, x2, y2, x3, y3, _dim, _angle, color):
    """Wrapper for drawing triangle; called like `draw_triangle(ax, *coords, None, None, color)`"""
    ax.add_patch(
        Polygon(
            xy=[(x1, y1), (x2, y2), (x3, y3)], closed=True,
            facecolor=color, edgecolor=less_transparent_color(color)
        )
    )

def draw_text(ax, x, y, text, fontsize, color="black", **kwargs):
    """Wrapper for drawing text"""
    ax.text(
        x, y, text, fontsize=fontsize, color=color,
        horizontalalignment="center", verticalalignment="center", 
        zorder=10, **kwargs
    )

def generate_logics(n_sets):
    """Generate intersection identifiers in binary (0010 etc)"""
    for i in range(1, 2**n_sets):
        yield bin(i)[2:].zfill(n_sets)

def generate_petal_labels(datasets, fmt="{size}"):
    """Generate petal descriptions for venn diagram based on set sizes"""
    datasets = list(datasets)
    dataset_sizes = [len(dataset) for dataset in datasets]
    n_sets = len(datasets)
    dataset_union = set.union(*datasets)
    universe_size = len(dataset_union)
    petal_labels = {}
    for logic in generate_logics(n_sets):
        included_sets = [datasets[i] for i in range(n_sets) if logic[i] == "1"]
        excluded_sets = [datasets[i] for i in range(n_sets) if logic[i] == "0"]
        petal_set = (
            (dataset_union & set.intersection(*included_sets)) -
            set.union(set(), *excluded_sets)
        )
        if fmt is not None:
            petal_labels[logic] = fmt.format(
                logic=logic, size=len(petal_set),
                percentage=(100*len(petal_set)/universe_size)
            )
        else:
            petal_labels[logic] = len(petal_set)
    return petal_labels

def generate_bootstrap_pvalues(obs_labels, datasets, background):

    # overlaps are represented as binary strings, e.g. "110" for sets 1 and 2.

    def get_counts(petal_labels):
        counts = dict()
        for logic, size in petal_labels.items():
            size = int(size)
            # overlaps are counted only if they are not single sets. 
            if sum(map(int, logic)) == 1:
                continue
            # overlaps include nested overlaps ("111" counts are also
            # included in both the "110" counts). I.e. overlaps between all three stets
            # are also considered also overlaps between set one and two.
            for other_logic, other_size in petal_labels.items():
                other_size = int(other_size)            
                if logic == other_logic:
                    continue 

                if sum(map(int, logic)) == sum(i and j for i, j in zip(map(int, logic), map(int, other_logic))):
                    size += other_size
            counts[logic] = size

        return counts

    obs_counts = get_counts(obs_labels)

    datasets = list(datasets)
    dataset_sizes = [len(dataset) for dataset in datasets]

    # dataset_union = set.union(*datasets)
    # non_background = dataset_union.difference(set(background))
    # if non_background:
    #     print(f"Warning: {len(non_background)} genes not in background set are ignored", file=sys.stderr)

    # bootstrap
    boot_counts = defaultdict(list)
    nr_bootstraps = 10000
    for _ in range(nr_bootstraps):
        # new randomly sampled datasets of original sizes
        boot_datasets = []
        for s in dataset_sizes:
            boot_datasets.append(set(random.sample(background, s)))
        boot_labels = generate_petal_labels(boot_datasets)
        for logic, size in get_counts(boot_labels).items():
            boot_counts[logic].append(size)

    p_values = {}
    for logic, c in obs_counts.items():
        p_values[logic] = 0
        for bc in boot_counts[logic]:
            if bc >= c:
                p_values[logic] += 1/nr_bootstraps

    return p_values



# def generate_pvalues(datasets, maxp, background):
#     """Generate petal descriptions for venn diagram based on set sizes"""

#     from ..utils import fisher_test

#     datasets = list(datasets)
#     n_sets = len(datasets)
#     dataset_union = set.union(*datasets)
#     universe_size = len(dataset_union)
#     pvalues = {}
#     for logic in generate_logics(n_sets):
#         included_sets = [
#             datasets[i] for i in range(n_sets) if logic[i] == "1"
#         ]
#         if len(included_sets) == 2:
#             p = fisher_test(included_sets[0], included_sets[1], 
#                             background=background)
#             if p < maxp:
#                 pvalues[logic] = p

#     return pvalues

def init_axes(ax, figsize):
    """Create axes if do not exist, set axes parameters"""
    if ax is None:
        _, ax = subplots(nrows=1, ncols=1, figsize=figsize)
    ax.set(
        aspect="equal", frame_on=False,
        xlim=(-.05, 1.05), ylim=(-.05, 1.05),
        xticks=[], yticks=[]
    )
    return ax

def get_n_sets(petal_labels, dataset_labels):
    """Infer number of sets, check consistency"""
    n_sets = len(dataset_labels)
    for logic in petal_labels.keys():
        if len(logic) != n_sets:
            raise ValueError("Inconsistent petal and dataset labels")
        if not (set(logic) <= {"0", "1"}):
            raise KeyError("Key not understood: " + logic)
    return n_sets

def draw_venn(*, petal_labels, dataset_labels, pvalues, hint_hidden, colors, figsize, 
              fontsize, textcolor, shape_coords=None, legend_loc, ax):
    """Draw true Venn diagram, annotate petals and dataset labels"""
    n_sets = get_n_sets(petal_labels, dataset_labels)
    if 2 <= n_sets < 6:
        draw_shape = draw_ellipse
    elif n_sets == 6:
        draw_shape = draw_triangle
    else:
        raise ValueError("Number of sets must be between 2 and 6")
    ax = init_axes(ax, figsize)
    if shape_coords is None:
        shape_coords = SHAPE_COORDS[n_sets]
    shape_params = zip(
        shape_coords, SHAPE_DIMS[n_sets], SHAPE_ANGLES[n_sets], colors
        # SHAPE_COORDS[n_sets], SHAPE_DIMS[n_sets], SHAPE_ANGLES[n_sets], colors
    )
    for coords, dims, angle, color in shape_params:
        draw_shape(ax, *coords, *dims, angle, color)
    for logic, petal_label in petal_labels.items():
        # some petals could have been modified manually:
        if logic in PETAL_LABEL_COORDS[n_sets]:
            x, y = PETAL_LABEL_COORDS[n_sets][logic]


            # draw_text(ax, x, y, petal_label, fontsize=fontsize, color=textcolor)
            # if pvalues and logic in pvalues:
            #     p = max(pvalues[logic], 1/10000)
            #     stars = '\n' + '*' * (int(-np.log10(p/5)) - 1)                                       
            #     draw_text(ax, x, y, stars, fontsize=fontsize, 
            #                   color=textcolor, fontweight='bold'
            #                   ) 

                            
            if pvalues is not None:
                if logic in pvalues:
                    p = max(pvalues[logic], 1/10000)
                    stars = '*' * (int(-np.log10(p/5)) - 1)                                       

                    petal_label += '\n' + '*' * (int(-np.log10(p/5)) - 1)                                       
                    draw_text(ax, x, y, petal_label, fontsize=fontsize, 
                              color=textcolor, #fontweight='bold'
                              ) 
                else:                    
                    draw_text(ax, x, y, petal_label, fontsize=fontsize, 
                                 color=textcolor, 
                                #  color="#999999"
                              )                    
            else:
                draw_text(ax, x, y, petal_label, fontsize=fontsize, color=textcolor)

    if legend_loc is not None:
        ax.legend(dataset_labels, loc=legend_loc, prop={"size": fontsize})
    return ax

def update_hidden(hidden, logic, petal_labels):
    """Increment set's hidden count (sizes of intersections that are not displayed)"""
    for i, c in enumerate(logic):
        if c == "1":
            hidden[i] += int(petal_labels[logic])
    return hidden

def draw_hint_explanation(ax, dataset_labels, fontsize, textcolor):
    """Add explanation of 'n/d*' hints"""
    example_labels = list(dataset_labels)[0], list(dataset_labels)[3]
    hint_text = (
        "* elements of set in intersections that are not displayed,\n" +
        "such as shared only between {} and {}".format(*example_labels)
    )
    draw_text(ax, .5, -.1, hint_text, fontsize, textcolor)

def is_valid_dataset_dict(data):
    """Validate passed data (must be dictionary of sets)"""
    if not (hasattr(data, "keys") and hasattr(data, "values")):
        return False
    for dataset in data.values():
        if not isinstance(dataset, set):
            return False
    else:
        return True

def venn_dispatch(data, func, fmt="{size}", hint_hidden=False, cmap="Set2", 
                  alpha=.4, figsize=(8, 8), fontsize=13, textcolor='black', 
                  background=None, shape_coords=None, 
                  legend_loc="upper right", ax=None):
    """Check input, generate petal labels, draw venn diagram"""
    if not is_valid_dataset_dict(data):
        raise TypeError("Only dictionaries of sets are understood")
    n_sets = len(data)

    _data = data.copy()
    for k, v in data.items():
        _data[k] = _data[k].intersection(set(background))
        assert _data[k], f"Set {k} is empty after intersection with background"
        removed = len(data[k]) - len(_data[k])
        # print(f'Ignoring {removed} genes in "{k}" not part of background set', file=sys.stderr)
    data = _data

    petal_labels = generate_petal_labels(data.values(), fmt)

    pvalues = None
    if background:
        pvalues = generate_bootstrap_pvalues(petal_labels, data.values(), background)

        keys = list(pvalues.keys())
        for logic in keys:
            if pvalues[logic] > 0.05:
                del pvalues[logic]


    return func(
        petal_labels=petal_labels,
        dataset_labels=data.keys(), hint_hidden=hint_hidden,
        pvalues=pvalues,
        colors=generate_colors(n_colors=n_sets, cmap=cmap, alpha=alpha),
        figsize=figsize, fontsize=fontsize, textcolor=textcolor, 
        shape_coords=shape_coords, legend_loc=legend_loc, ax=ax,
    )

_venn = partial(venn_dispatch, func=draw_venn, hint_hidden=False)
    
def venn(*data, ncols=4, nrows=None, petals=3, palette='rainbow', 
          fontsize=9, textcolor=None, shape_coords=None, 
          figsize=None, alpha=0.4, background=None):

    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import ListedColormap

    if type(data[0]) is not tuple:
        _names = set()
        _data = []
        for x in data:
            assert x.name() not in _names, f"Duplicate name {x.name()} found in data"
            _names.add(x.name())
            _data.append((x.name(), set(x)))
        data = _data

    nrsets = len(data)

    petals = min(nrsets, petals)

    vals = np.linspace(0, 1, nrsets)
    viridis = matplotlib.colormaps[palette]
    combinations = list(itertools.combinations(range(nrsets), petals))

    ncols = min(ncols, len(combinations))

    if nrows is None:
        nrows = len(combinations)//ncols + int(len(combinations) % ncols > 0)

    if figsize is None:
        if nrsets == petals:
            figsize=(4, 3)
        else:
            figsize=(10, 2.5*nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout='constrained')

    if nrows == 1 and  ncols == 1:
        axes = np.array([axes])

    for i, ax in enumerate(axes.flat):
        if i >= len(combinations):
            ax.axis('off')
            continue
        combo = combinations[i]
        cmap = viridis([vals[i] for i in combo])
        cmap = ListedColormap(cmap)
        subset = dict([data[i] for i in combo])

        ax = _venn(subset, background=background, ax=ax, cmap=cmap, fontsize=fontsize, 
                   textcolor=textcolor, alpha=alpha, shape_coords=shape_coords)
        ax.get_legend().remove()
        ax.axis('off')

    handles = []
    cmap = viridis([vals[i] for i in range(len(data))])
    cmap = ListedColormap(cmap)
    for i, (label, _) in enumerate(data):
        cmap = viridis([vals[i]])
        cmap = ListedColormap(cmap)
        handles.extend([Patch(color=cmap(vals[i]), label=label, alpha=0.7)])

    if len(axes.flat) > len(combinations):
        l = fig.legend(loc='lower right', handles=handles, frameon=False, fontsize=fontsize, ncol=1)
    else:
        l = fig.legend(loc='outside center right', handles=handles, frameon=False, fontsize=fontsize, ncol=1)
    for text in l.get_texts():
        text.set_color(plt.rcParams['text.color'])