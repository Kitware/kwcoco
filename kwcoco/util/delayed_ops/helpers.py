import kwimage
import ubelt as ub
import numpy as np
import sys
from collections import defaultdict


try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


def _auto_dsize(transform, sub_dsize):
    sub_w, sub_h = sub_dsize
    sub_bounds = kwimage.Coords(
        np.array([[0,     0], [sub_w, 0],
                  [0, sub_h], [sub_w, sub_h]])
    )
    bounds = sub_bounds.warp(transform.matrix)
    max_xy = np.ceil(bounds.data.max(axis=0))
    max_x = int(max_xy[0])
    max_y = int(max_xy[1])
    dsize = (max_x, max_y)
    return dsize


def _largest_shape(shapes):
    """
    Finds maximum over all shapes

    Example:
        >>> shapes = [
        >>>     (10, 20), None, (None, 30), (40, 50, 60, None), (100,)
        >>> ]
        >>> largest = _largest_shape(shapes)
        >>> print('largest = {!r}'.format(largest))
        >>> assert largest == (100, 50, 60, None)
    """
    def _nonemax(a, b):
        if a is None or b is None:
            return a or b
        return max(a, b)
    import itertools as it
    largest = []
    for shape in shapes:
        if shape is not None:
            largest = [
                _nonemax(c1, c2)
                for c1, c2 in it.zip_longest(largest, shape, fillvalue=None)
            ]
    largest = tuple(largest)
    return largest


@profile
def _swap_warp_after_crop(root_region_bounds, tf_leaf_to_root):
    r"""
    Given a warp followed by a crop, compute the corresponding crop followed by
    a warp.

    Given a region in a "root" image and a trasnform between that "root" and
    some "leaf" image, compute the appropriate quantized region in the "leaf"
    image and the adjusted transformation between that root and leaf.

    Args:
        root_region_bounds (kwimage.Polygon):
            region representing the crop that happens after the warp

        tf_leaf_to_root (kwimage.Affine):
            the warp that happens before the input crop

    Returns:
        Tuple[Tuple[slice, slice], kwimage.Affine]:
            leaf_crop_slices - the crop that happens before the warp
            tf_newleaf_to_newroot - warp that happens after the crop.

    Example:
        >>> region_slices = (slice(33, 100), slice(22, 62))
        >>> region_shape = (100, 100, 1)
        >>> root_region_box = kwimage.Boxes.from_slice(region_slices, shape=region_shape)
        >>> root_region_bounds = root_region_box.to_polygons()[0]
        >>> tf_leaf_to_root = kwimage.Affine.affine(scale=7).matrix
        >>> slices, tf_new = _swap_warp_after_crop(root_region_bounds, tf_leaf_to_root)
        >>> print('tf_new =\n{!r}'.format(tf_new))
        >>> print('slices = {!r}'.format(slices))
    """
    # Transform the region bounds into the sub-image space
    tf_leaf_to_root = kwimage.Affine.coerce(tf_leaf_to_root)
    tf_root_to_leaf = tf_leaf_to_root.inv()
    tf_root_to_leaf = tf_root_to_leaf.__array__()
    leaf_region_bounds = root_region_bounds.warp(tf_root_to_leaf)
    leaf_region_box = leaf_region_bounds.bounding_box().to_ltrb()

    # Quantize to a region that is possible to sample from
    leaf_crop_box = leaf_region_box.quantize()

    # is this ok?
    leaf_crop_box = leaf_crop_box.clip(0, 0, None, None)

    # Because we sampled a large quantized region, we need to modify the
    # transform to nudge it a bit to the left, undoing the quantization,
    # which has a bit of extra padding on the left, before applying the
    # final transform.
    # subpixel_offset = leaf_region_box.data[0, 0:2]
    crop_offset = leaf_crop_box.data[0, 0:2]
    root_offset = root_region_bounds.exterior.data.min(axis=0)

    tf_root_to_newroot = kwimage.Affine.affine(offset=-root_offset).matrix
    tf_newleaf_to_leaf = kwimage.Affine.affine(offset=crop_offset).matrix

    # Resample the smaller region to align it with the root region
    # Note: The right most transform is applied first
    tf_newleaf_to_newroot = (
        tf_root_to_newroot @
        tf_leaf_to_root @
        tf_newleaf_to_leaf
    )

    lt_x, lt_y, rb_x, rb_y = leaf_crop_box.data[0, 0:4]
    leaf_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

    return leaf_crop_slices, tf_newleaf_to_newroot


@profile
def _swap_crop_after_warp(inner_region, outer_transform):
    r"""
    Given a crop followed by a warp (usually an overview), compute the
    corresponding warp followed by a crop followed by a small correction warp.

    Note that in general it is not possible to ensure the crop is the last
    operation, there may need to be a small warp after it.

    However, this is generally only useful when the warp being pushed early in
    the operation chain corresponds to an overview, and often - but not always
    - the final warp will simply be the identity.

    Args:
        inner_region (kwimage.Polygon):
            region representing the crop that happens before the warp

        outer_transform (kwimage.Affine):
            the warp that happens after the input crop

    Returns:
        Tuple[kwimage.Affine, Tuple[slice, slice], kwimage.Affine]:

            new_inner_warp - the new warp to happen before the crop

            outer_crop - the new crop after the main warp

            new_outer_warp - a small subpixel alignment warp to happen last

    Example:
        >>> from kwcoco.util.delayed_ops.helpers import *  # NOQA
        >>> region_slices = (slice(33, 100), slice(22, 62))
        >>> region_shape = (100, 100, 1)
        >>> inner_region = kwimage.Boxes.from_slice(region_slices)
        >>> inner_region = inner_region.to_polygons()[0]
        >>> outer_transform = kwimage.Affine.affine(scale=1/4)
        >>> new_inner_warp, outer_crop, new_outer_warp = _swap_crop_after_warp(inner_region, outer_transform)
        >>> print('new_inner_warp = {}'.format(ub.repr2(new_inner_warp, nl=1)))
        >>> print('outer_crop = {}'.format(ub.repr2(outer_crop, nl=1)))
        >>> print('new_outer_warp = {}'.format(ub.repr2(new_outer_warp, nl=1)))
    """
    # Find where the inner region maps to after the transform is applied
    outer_region = inner_region.warp(outer_transform)

    # Transform the region bounds into the sub-image space
    outer_box = outer_region.bounding_box().to_ltrb()

    # Quantize to a region that is possible to sample from
    outer_crop_box = outer_box.quantize()

    # is this ok?
    outer_crop_box = outer_crop_box.clip(0, 0, None, None)

    # Because the new crop might not be perfectly aligned, we might need to
    # nudge it a bit after we crop out its bounds.
    crop_offset = outer_crop_box.data[0, 0:2]
    outer_offset = outer_region.exterior.data.min(axis=0)

    # Compute the extra transform that will realign the quantized croped data
    # with the original warped inner crop.
    tf_crop_to_box = kwimage.Affine.affine(
        offset=crop_offset - outer_offset
    )

    lt_x, lt_y, rb_x, rb_y = outer_crop_box.data[0, 0:4]
    outer_crop = (slice(lt_y, rb_y), slice(lt_x, rb_x))
    new_outer_warp = tf_crop_to_box

    # The inner warp will be the same as the original outer warp.
    new_inner_warp = outer_transform

    return new_inner_warp, outer_crop, new_outer_warp


def dequantize(quant_data, quantization):
    """
    Helper for dequantization

    Args:
        quant_data (ndarray):
            data to dequantize

        quantization (Dict[str, Any]):
            quantization information dictionary to undo.
            Expected keys are:
            orig_type (str)
            orig_min (float)
            orig_max (float)
            quant_min (float)
            quant_max (float)
            nodata (None | int)

    Returns:
        ndarray : dequantized data

    Example:
        >>> quant_data = (np.random.rand(4, 4) * 256).astype(np.uint8)
        >>> quantization = {
        >>>     'orig_dtype': 'float32',
        >>>     'orig_min': 0,
        >>>     'orig_max': 1,
        >>>     'quant_min': 0,
        >>>     'quant_max': 255,
        >>>     'nodata': None,
        >>> }
        >>> dequantize(quant_data, quantization)

    Example:
        >>> quant_data = np.ones((4, 4), dtype=np.uint8)
        >>> quantization = {
        >>>     'orig_dtype': 'float32',
        >>>     'orig_min': 0,
        >>>     'orig_max': 1,
        >>>     'quant_min': 1,
        >>>     'quant_max': 1,
        >>>     'nodata': None,
        >>> }
        >>> dequantize(quant_data, quantization)
    """
    orig_dtype = quantization.get('orig_dtype', 'float32')
    orig_min = quantization.get('orig_min', 0)
    orig_max = quantization.get('orig_max', 1)
    quant_min = quantization.get('quant_min', 0)
    quant_max = quantization['quant_max']
    nodata = quantization.get('nodata', None)
    orig_extent = orig_max - orig_min
    quant_extent = quant_max - quant_min
    if quant_extent == 0:
        scale = 0
    else:
        scale = (orig_extent / quant_extent)
    dequant = quant_data.astype(orig_dtype)
    dequant = (dequant - quant_min) * scale + orig_min
    if nodata is not None:
        mask = quant_data == nodata
        dequant[mask] = np.nan
    return dequant


def quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.int16):
    """
    Note:
        Setting old_min / old_max indicates the possible extend of the input
        data (and it will be clipped to it). It does not mean that the input
        data has to have those min and max values, but it should be between
        them.

    Example:
        >>> from kwcoco.util.delayed_ops.helpers import *  # NOQA
        >>> # Test error when input is not nicely between 0 and 1
        >>> imdata = (np.random.randn(32, 32, 3) - 1.) * 2.5
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))
        >>> #
        >>> for i in range(1, 20):
        >>>     print('i = {!r}'.format(i))
        >>>     quant2, quantization2 = quantize_float01(imdata, old_min=-i, old_max=i)
        >>>     recon2 = dequantize(quant2, quantization2)
        >>>     error2 = np.abs((recon2 - imdata)).sum()
        >>>     print('error2 = {!r}'.format(error2))

    Example:
        >>> # Test dequantize with uint8
        >>> from kwcoco.util.util_delayed_poc import dequantize
        >>> imdata = np.random.randn(32, 32, 3)
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.uint8)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))

    Example:
        >>> # Test quantization with different signed / unsigned combos
        >>> print(quantize_float01(None, 0, 1, np.int16))
        >>> print(quantize_float01(None, 0, 1, np.int8))
        >>> print(quantize_float01(None, 0, 1, np.uint8))
        >>> print(quantize_float01(None, 0, 1, np.uint16))

    """
    # old_min = 0
    # old_max = 1
    quantize_iinfo = np.iinfo(quantize_dtype)
    quantize_max = quantize_iinfo.max
    if quantize_iinfo.kind == 'u':
        # Unsigned quantize
        quantize_nan = 0
        quantize_min = 1
    elif quantize_iinfo.kind == 'i':
        # Signed quantize
        quantize_min = 0
        quantize_nan = max(-9999, quantize_iinfo.min)

    quantization = {
        'orig_min': old_min,
        'orig_max': old_max,
        'quant_min': quantize_min,
        'quant_max': quantize_max,
        'nodata': quantize_nan,
    }

    old_extent = (old_max - old_min)
    new_extent = (quantize_max - quantize_min)
    quant_factor = new_extent / old_extent

    if imdata is not None:
        invalid_mask = np.isnan(imdata)
        new_imdata = (imdata.clip(old_min, old_max) - old_min) * quant_factor + quantize_min
        new_imdata = new_imdata.astype(quantize_dtype)
        new_imdata[invalid_mask] = quantize_nan
    else:
        new_imdata = None

    return new_imdata, quantization


### See: https://github.com/networkx/networkx/pull/5602


class _AsciiBaseGlyphs:
    empty = "+"
    newtree_last = "+-- "
    newtree_mid = "+-- "
    endof_forest = "    "
    within_forest = ":   "
    within_tree = "|   "


class AsciiDirectedGlyphs(_AsciiBaseGlyphs):
    last = "L-> "
    mid = "|-> "
    backedge = "<-"


class AsciiUndirectedGlyphs(_AsciiBaseGlyphs):
    last = "L-- "
    mid = "|-- "
    backedge = "-"


class _UtfBaseGlyphs:
    # Notes on available box and arrow characters
    # https://en.wikipedia.org/wiki/Box-drawing_character
    # https://stackoverflow.com/questions/2701192/triangle-arrow
    empty = "╙"
    newtree_last = "╙── "
    newtree_mid = "╟── "
    endof_forest = "    "
    within_forest = "╎   "
    within_tree = "│   "


class UtfDirectedGlyphs(_UtfBaseGlyphs):
    last = "└─╼ "
    mid = "├─╼ "
    backedge = "╾"


class UtfUndirectedGlyphs(_UtfBaseGlyphs):
    last = "└── "
    mid = "├── "
    backedge = "─"


def generate_network_text(
    graph, with_labels=True, sources=None, max_depth=None, ascii_only=False
):
    """Generate lines in the "network text" format

    This works via a depth-first traversal of the graph and writing a line for
    each unique node encountered. Non-tree edges are written to the right of
    each node, and connection to a non-tree edge is indicated with an ellipsis.
    This representation works best when the input graph is a forest, but any
    graph can be represented.

    This notation is original to networkx, although it is simple enough that it
    may be known in existing literature. See #5602 for details. The procedure
    is summarized as follows:

    1. Given a set of source nodes (which can be specified, or automatically
    discovered via finding the (strongly) connected components and choosing one
    node with minimum degree from each), we traverse the graph in depth first
    order.

    2. Each reachable node will be printed exactly once on it's own line.

    3. Edges are indicated in one of three ways:

        a. a parent "L-style" connection on the upper left. This corresponds to
        a traversal in the directed DFS tree.

        b. a backref "<-style" connection shown directly on the right. For
        directed graphs, these are drawn for any incoming edges to a node that
        is not a parent edge. For undirected graphs, these are drawn for only
        the non-parent edges that have already been represented (The edges that
        have not been represented will be handled in the recursive case).

        c. a child "L-style" connection on the lower right. Drawing of the
        children are handled recursively.

    4. The children of each node (wrt the directed DFS tree) are drawn
    underneath and to the right of it. In the case that a child node has already
    been drawn the connection is replaced with an ellipsis ("...") to indicate
    that there is one or more connections represented elsewhere.

    5. If a maximum depth is specified, an edge to nodes past this maximum
    depth will be represented by an ellipsis.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent

    with_labels : bool | str
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. If given as a
        string, then that attribte name will be used instead of "label".
        Defaults to True.

    sources : List
        Specifies which nodes to start traversal from. Note: nodes that are not
        reachable from one of these sources may not be shown. If unspecified,
        the minimal set of nodes needed to reach all others will be used.

    max_depth : int | None
        The maximum depth to traverse before stopping. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    Yields
    ------
    str : a line of generated text
    """
    is_directed = graph.is_directed()

    if is_directed:
        glyphs = AsciiDirectedGlyphs if ascii_only else UtfDirectedGlyphs
        succ = graph.succ
        pred = graph.pred
    else:
        glyphs = AsciiUndirectedGlyphs if ascii_only else UtfUndirectedGlyphs
        succ = graph.adj
        pred = graph.adj

    if isinstance(with_labels, str):
        label_attr = with_labels
    elif with_labels:
        label_attr = "label"
    else:
        label_attr = None

    if max_depth == 0:
        yield glyphs.empty + " ..."
    elif len(graph.nodes) == 0:
        yield glyphs.empty
    else:

        # If the nodes to traverse are unspecified, find the minimal set of
        # nodes that will reach the entire graph
        if sources is None:
            sources = _find_sources(graph)

        # Populate the stack with each:
        # 1. parent node in the DFS tree (or None for root nodes),
        # 2. the current node in the DFS tree
        # 2. a list of indentations indicating depth
        # 3. a flag indicating if the node is the final one to be written.
        # Reverse the stack so sources are popped in the correct order.
        last_idx = len(sources) - 1
        stack = [
            (None, node, [], (idx == last_idx)) for idx, node in enumerate(sources)
        ][::-1]

        num_skipped_children = defaultdict(lambda: 0)
        seen_nodes = set()
        while stack:
            parent, node, indents, this_islast = stack.pop()

            if node is not Ellipsis:
                skip = node in seen_nodes
                if skip:
                    # Mark that we skipped a parent's child
                    num_skipped_children[parent] += 1

                if this_islast:
                    # If we reached the last child of a parent, and we skipped
                    # any of that parents children, then we should emit an
                    # ellipsis at the end after this.
                    if num_skipped_children[parent] and parent is not None:

                        # Append the ellipsis to be emitted last
                        next_islast = True
                        try_frame = (node, Ellipsis, indents, next_islast)
                        stack.append(try_frame)

                        # Redo this frame, but not as a last object
                        next_islast = False
                        try_frame = (parent, node, indents, next_islast)
                        stack.append(try_frame)
                        continue

                if skip:
                    continue
                seen_nodes.add(node)

            if not indents:
                # Top level items (i.e. trees in the forest) get different
                # glyphs to indicate they are not actually connected
                if this_islast:
                    this_prefix = indents + [glyphs.newtree_last]
                    next_prefix = indents + [glyphs.endof_forest]
                else:
                    this_prefix = indents + [glyphs.newtree_mid]
                    next_prefix = indents + [glyphs.within_forest]

            else:
                # For individual tree edges distinguish between directed and
                # undirected cases
                if this_islast:
                    this_prefix = indents + [glyphs.last]
                    next_prefix = indents + [glyphs.endof_forest]
                else:
                    this_prefix = indents + [glyphs.mid]
                    next_prefix = indents + [glyphs.within_tree]

            if node is Ellipsis:
                label = " ..."
                suffix = ""
                children = []
            else:
                if label_attr is not None:
                    label = str(graph.nodes[node].get(label_attr, node))
                else:
                    label = str(node)

                # Determine:
                # (1) children to traverse into after showing this node.
                # (2) parents to immediately show to the right of this node.
                if is_directed:
                    # In the directed case we must show every successor node
                    # note: it may be skipped later, but we don't have that
                    # information here.
                    children = list(succ[node])
                    # In the directed case we must show every predecessor
                    # except for parent we directly traversed from.
                    handled_parents = {parent}
                else:
                    # Showing only the unseen children results in a more
                    # concise representation for the undirected case.
                    children = [
                        child for child in succ[node] if child not in seen_nodes
                    ]

                    # In the undirected case, parents are also children, so we
                    # only need to immediately show the ones we can no longer
                    # traverse
                    handled_parents = {*children, parent}

                if max_depth is not None and len(indents) == max_depth - 1:
                    # Use ellipsis to indicate we have reached maximum depth
                    if children:
                        children = [Ellipsis]
                    handled_parents = {parent}

                # The other parents are other predecessors of this node that
                # are not handled elsewhere.
                other_parents = [p for p in pred[node] if p not in handled_parents]
                if other_parents:
                    if label_attr is not None:
                        other_parents_labels = ", ".join(
                            [
                                str(graph.nodes[p].get(label_attr, p))
                                for p in other_parents
                            ]
                        )
                    else:
                        other_parents_labels = ", ".join(
                            [str(p) for p in other_parents]
                        )
                    suffix = " ".join(["", glyphs.backedge, other_parents_labels])
                else:
                    suffix = ""

            # Emit the line for this node, this will be called for each node
            # exactly once.
            yield "".join(this_prefix + [label, suffix])

            # Push children on the stack in reverse order so they are popped in
            # the original order.
            for idx, child in enumerate(children[::-1]):
                next_islast = idx == 0
                try_frame = (node, child, next_prefix, next_islast)
                stack.append(try_frame)


# @open_file(1, "w")
def write_network_text(
    graph,
    path=None,
    with_labels=True,
    sources=None,
    max_depth=None,
    ascii_only=False,
    end="\n",
):
    """Creates a nice text representation of a graph

    This works via a depth-first traversal of the graph and writing a line for
    each unique node encountered. Non-tree edges are written to the right of
    each node, and connection to a non-tree edge is indicated with an ellipsis.
    This representation works best when the input graph is a forest, but any
    graph can be represented.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent

    path : string or file or callable or None
       Filename or file handle for data output.
       if a function, then it will be called for each generated line.
       if None, this will default to "sys.stdout.write"

    with_labels : bool | str
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. If given as a
        string, then that attribte name will be used instead of "label".
        Defaults to True.

    sources : List
        Specifies which nodes to start traversal from. Note: nodes that are not
        reachable from one of these sources may not be shown. If unspecified,
        the minimal set of nodes needed to reach all others will be used.

    max_depth : int | None
        The maximum depth to traverse before stopping. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    end : string
        The line ending characater

    Example
    -------
    >>> import networkx as nx
    >>> graph = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    >>> write_network_text(graph)
    ╙── 0
        ├─╼ 1
        │   ├─╼ 3
        │   └─╼ 4
        └─╼ 2
            ├─╼ 5
            └─╼ 6

    >>> # A near tree with one non-tree edge
    >>> graph.add_edge(5, 1)
    >>> write_network_text(graph)
    ╙── 0
        ├─╼ 1 ╾ 5
        │   ├─╼ 3
        │   └─╼ 4
        └─╼ 2
            ├─╼ 5
            │   └─╼  ...
            └─╼ 6

    >>> graph = nx.cycle_graph(5)
    >>> write_network_text(graph)
    ╙── 0
        ├── 1
        │   └── 2
        │       └── 3
        │           └── 4 ─ 0
        └──  ...

    >>> graph = nx.generators.barbell_graph(4, 2)
    >>> write_network_text(graph)
    ╙── 4
        ├── 5
        │   └── 6
        │       ├── 7
        │       │   ├── 8 ─ 6
        │       │   │   └── 9 ─ 6, 7
        │       │   └──  ...
        │       └──  ...
        └── 3
            ├── 0
            │   ├── 1 ─ 3
            │   │   └── 2 ─ 0, 3
            │   └──  ...
            └──  ...

    >>> graph = nx.complete_graph(5, create_using=nx.Graph)
    >>> write_network_text(graph)
    ╙── 0
        ├── 1
        │   ├── 2 ─ 0
        │   │   ├── 3 ─ 0, 1
        │   │   │   └── 4 ─ 0, 1, 2
        │   │   └──  ...
        │   └──  ...
        └──  ...

    >>> graph = nx.complete_graph(3, create_using=nx.DiGraph)
    >>> write_network_text(graph)
    ╙── 0 ╾ 1, 2
        ├─╼ 1 ╾ 2
        │   ├─╼ 2 ╾ 0
        │   │   └─╼  ...
        │   └─╼  ...
        └─╼  ...
    """
    if path is None:
        # The path is unspecified, write to stdout
        _write = sys.stdout.write
    elif hasattr(path, "write"):
        # The path is already an open file
        _write = path.write
    elif callable(path):
        # The path is a custom callable
        _write = path
    else:
        raise TypeError(type(path))

    for line in generate_network_text(
        graph,
        with_labels=with_labels,
        sources=sources,
        max_depth=max_depth,
        ascii_only=ascii_only,
    ):
        _write(line + end)


def _find_sources(graph):
    """
    Determine a minimal set of nodes such that the entire graph is reachable
    """
    # For each connected part of the graph, choose at least
    # one node as a starting point, preferably without a parent
    import networkx as nx
    if graph.is_directed():
        # Choose one node from each SCC with minimum in_degree
        sccs = list(nx.strongly_connected_components(graph))
        # condensing the SCCs forms a dag, the nodes in this graph with
        # 0 in-degree correspond to the SCCs from which the minimum set
        # of nodes from which all other nodes can be reached.
        scc_graph = nx.condensation(graph, sccs)
        supernode_to_nodes = {sn: [] for sn in scc_graph.nodes()}
        # Note: the order of mapping differs between pypy and cpython
        # so we have to loop over graph nodes for consistency
        mapping = scc_graph.graph["mapping"]
        for n in graph.nodes:
            sn = mapping[n]
            supernode_to_nodes[sn].append(n)
        sources = []
        for sn in scc_graph.nodes():
            if scc_graph.in_degree[sn] == 0:
                scc = supernode_to_nodes[sn]
                node = min(scc, key=lambda n: graph.in_degree[n])
                sources.append(node)
    else:
        # For undirected graph, the entire graph will be reachable as
        # long as we consider one node from every connected component
        sources = [
            min(cc, key=lambda n: graph.degree[n])
            for cc in nx.connected_components(graph)
        ]
        sources = sorted(sources, key=lambda n: graph.degree[n])
    return sources


def graph_str(graph, with_labels=True, sources=None, write=None, ascii_only=False):
    """Creates a nice utf8 representation of a forest

    This function has been superseded by
    :func:`nx.readwrite.text.generate_network_text`, which should be used
    instead.

    Parameters
    ----------
    graph : nx.DiGraph | nx.Graph
        Graph to represent (must be a tree, forest, or the empty graph)

    with_labels : bool
        If True will use the "label" attribute of a node to display if it
        exists otherwise it will use the node value itself. Defaults to True.

    sources : List
        Mainly relevant for undirected forests, specifies which nodes to list
        first. If unspecified the root nodes of each tree will be used for
        directed forests; for undirected forests this defaults to the nodes
        with the smallest degree.

    write : callable
        Function to use to write to, if None new lines are appended to
        a list and returned. If set to the `print` function, lines will
        be written to stdout as they are generated. If specified,
        this function will return None. Defaults to None.

    ascii_only : Boolean
        If True only ASCII characters are used to construct the visualization

    Returns
    -------
    str | None :
        utf8 representation of the tree / forest

    Example
    -------
    >>> import networkx as nx
    >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
    >>> print(graph_str(graph))
    ╙── 0
        ├─╼ 1
        │   ├─╼ 3
        │   │   ├─╼ 7
        │   │   └─╼ 8
        │   └─╼ 4
        │       ├─╼ 9
        │       └─╼ 10
        └─╼ 2
            ├─╼ 5
            │   ├─╼ 11
            │   └─╼ 12
            └─╼ 6
                ├─╼ 13
                └─╼ 14


    >>> graph = nx.balanced_tree(r=1, h=2, create_using=nx.Graph)
    >>> print(graph_str(graph))
    ╙── 0
        └── 1
            └── 2

    >>> print(graph_str(graph, ascii_only=True))
    +-- 0
        L-- 1
            L-- 2
    """
    printbuf = []
    if write is None:
        _write = printbuf.append
    else:
        _write = write

    write_network_text(
        graph,
        _write,
        with_labels=with_labels,
        sources=sources,
        ascii_only=ascii_only,
        end="",
    )

    if write is None:
        # Only return a string if the custom write function was not specified
        return "\n".join(printbuf)
