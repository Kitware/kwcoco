"""
The :mod:`category_tree` module defines the :class:`CategoryTree` class, which
is used for maintaining flat or hierarchical category information. The kwcoco
version of this class only contains the datastructure and does not contain any
torch operations. See the ndsampler version for the extension with torch
operations.
"""
import itertools as it
import networkx as nx
import ubelt as ub
import numpy as np

__all__ = ['CategoryTree']

if hasattr(nx, 'OrderedDiGraph'):
    # For 3.6 compat
    # For old versions of networkx there was a real OrderedDiGraph, but now
    # everything is ordered
    OrderedDiGraph = nx.OrderedDiGraph
else:
    OrderedDiGraph = nx.DiGraph


RESPECT_INPUT_ORDER = 1


class CategoryTree(ub.NiceRepr):
    """
    Wrapper that maintains flat or hierarchical category information.

    Helps compute softmaxes and probabilities for tree-based categories
    where a directed edge (A, B) represents that A is a superclass of B.

    Note:

        There are three basic properties that this object maintains:

        .. code::

            node:
                Alphanumeric string names that should be generally descriptive.
                Using spaces and special characters in these names is
                discouraged, but can be done.  This is the COCO category "name"
                attribute.  For categories this may be denoted as (name, node,
                cname, catname).

            id:
                The integer id of a category should ideally remain consistent.
                These are often given by a dataset (e.g. a COCO dataset).  This
                is the COCO category "id" attribute. For categories this is
                often denoted as (id, cid).

            index:
                Contiguous zero-based indices that indexes the list of
                categories.  These should be used for the fastest access in
                backend computation tasks. Typically corresponds to the
                ordering of the channels in the final linear layer in an
                associated model.  For categories this is often denoted as
                (index, cidx, idx, or cx).

    Attributes:

        idx_to_node (List[str]): a list of class names. Implicitly maps from
            index to category name.

        id_to_node (Dict[int, str]): maps integer ids to category names

        node_to_id (Dict[str, int]): maps category names to ids

        node_to_idx (Dict[str, int]): maps category names to indexes

        graph (networkx.Graph): a Graph that stores any hierarchy information.
            For standard mutually exclusive classes, this graph is edgeless.
            Nodes in this graph can maintain category attributes / properties.

        idx_groups (List[List[int]]): groups of category indices that
            share the same parent category.

    Example:
        >>> from kwcoco.category_tree import *
        >>> graph = nx.from_dict_of_lists({
        >>>     'background': [],
        >>>     'foreground': ['animal'],
        >>>     'animal': ['mammal', 'fish', 'insect', 'reptile'],
        >>>     'mammal': ['dog', 'cat', 'human', 'zebra'],
        >>>     'zebra': ['grevys', 'plains'],
        >>>     'grevys': ['fred'],
        >>>     'dog': ['boxer', 'beagle', 'golden'],
        >>>     'cat': ['maine coon', 'persian', 'sphynx'],
        >>>     'reptile': ['bearded dragon', 't-rex'],
        >>> }, nx.DiGraph)
        >>> self = CategoryTree(graph)
        >>> print(self)
        <CategoryTree(nNodes=22, maxDepth=6, maxBreadth=4...)>

    Example:
        >>> # The coerce classmethod is the easiest way to create an instance
        >>> import kwcoco
        >>> kwcoco.CategoryTree.coerce(['a', 'b', 'c'])
        <CategoryTree...nNodes=3, nodes=...'a', 'b', 'c'...
        >>> kwcoco.CategoryTree.coerce(4)
        <CategoryTree...nNodes=4, nodes=...'class_1', 'class_2', 'class_3', ...
        >>> kwcoco.CategoryTree.coerce(4)
    """
    def __init__(self, graph=None, checks=True):
        """
        Args:
            graph (nx.DiGraph):
                either the graph representing a category hierarchy

            checks (bool, default=True):
                if false, bypass input checks

        """
        if graph is None:
            graph = OrderedDiGraph()
        elif checks:
            if len(graph) > 0:
                if not nx.is_directed_acyclic_graph(graph):
                    raise ValueError('The category graph must a DAG')
                if not nx.is_forest(graph):
                    raise ValueError('The category graph must be a forest')
            if not isinstance(graph, nx.Graph):
                raise TypeError('Input to CategoryTree must be a networkx graph not {}'.format(type(graph)))
        self.graph = graph  # :type: nx.Graph
        # Note: nodes are class names
        self.id_to_node = None
        self.node_to_id = None
        self.node_to_idx = None
        self.idx_to_node = None
        self.idx_groups = None

        self._build_index()

    def copy(self):
        new = self.__class__(self.graph.copy())
        return new

    @classmethod
    def from_mutex(cls, nodes, bg_hack=True):
        """
        Args:
            nodes (List[str]): or a list of class names (in which case they
                will all be assumed to be mutually exclusive)

        Example:
            >>> print(CategoryTree.from_mutex(['a', 'b', 'c']))
            <CategoryTree(nNodes=3, ...)>
        """
        nodes = list(nodes)
        graph = OrderedDiGraph()
        graph.add_nodes_from(nodes)
        start = 0

        if bg_hack:
            # TODO: deprecate the defaultness of this
            if 'background' in graph.nodes:
                # hack
                graph.nodes['background']['id'] = 0
                start = 1

        for i, node in enumerate(nodes, start=start):
            graph.nodes[node]['id'] = graph.nodes[node].get('id', i)

        return cls(graph)

    @classmethod
    def from_json(cls, state):
        """
        Args:
            state (Dict): see __getstate__ / __json__ for details
        """
        self = cls()
        self.__setstate__(state)
        return self

    @classmethod
    def from_coco(cls, categories):
        """
        Create a CategoryTree object from coco categories

        Args:
            List[Dict]: list of coco-style categories

        Example:
            >>> import kwcoco
            >>> classes1 = kwcoco.CategoryTree.coerce([{'name': 'cat1'}, {'name': 'cat2', 'id': 1}])
            >>> assert classes1.id_to_node == {2: 'cat1', 1: 'cat2'}
            >>> classes2 = kwcoco.CategoryTree.coerce([{'name': 'cat4'}, {'name': 'cat5'}])
            >>> assert classes2.id_to_node == {1: 'cat4', 2: 'cat5'}
        """
        graph = OrderedDiGraph()
        for cat in categories:
            graph.add_node(cat['name'], **cat)
            if cat.get('supercategory', None) is not None:
                graph.add_edge(cat['supercategory'], cat['name'])
        self = cls(graph)
        return self

    @classmethod
    def coerce(cls, data, **kw):
        """
        Attempt to coerce data as a CategoryTree object.

        This is primarily useful for when the software stack depends on
        categories being represent

        This will work if the input data is a specially formatted json dict, a
        list of mutually exclusive classes, or if it is already a CategoryTree.
        Otherwise an error will be thrown.

        Args:
            data (object): a known representation of a category tree.
            **kwargs: input type specific arguments

        Returns:
            CategoryTree: self

        Raises:
            TypeError - if the input format is unknown
            ValueError - if kwargs are not compatible with the input format

        Example:
            >>> import kwcoco
            >>> classes1 = kwcoco.CategoryTree.coerce(3)  # integer
            >>> classes2 = kwcoco.CategoryTree.coerce(classes1.__json__())  # graph dict
            >>> classes3 = kwcoco.CategoryTree.coerce(['class_1', 'class_2', 'class_3'])  # mutex list
            >>> classes4 = kwcoco.CategoryTree.coerce(classes1.graph)  # nx Graph
            >>> classes5 = kwcoco.CategoryTree.coerce(classes1)  # cls
            >>> classes_09 = kwcoco.CategoryTree.coerce([{'name': 'cat1'}])
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> import ndsampler
            >>> classes6 = ndsampler.CategoryTree.coerce(3)
            >>> classes7 = ndsampler.CategoryTree.coerce(classes1)
            >>> classes8 = kwcoco.CategoryTree.coerce(classes6)
        """
        if isinstance(data, int):
            # An integer specifies the number of classes.
            self = cls.from_mutex(
                ['class_{}'.format(i + 1) for i in range(data)], **kw)
        elif isinstance(data, dict):
            # A dictionary is assumed to be in a special json format
            self = cls.from_json(data, **kw)
        elif isinstance(data, list):
            if len(data) and isinstance(data[0], dict) and 'name' in data[0]:
                # The list is likely a coco categoyr list
                self = cls.from_coco(data, **kw)
            else:
                # A list is assumed to be a list of class names
                self = cls.from_mutex(data, **kw)
        elif isinstance(data, nx.DiGraph):
            # A nx.DiGraph should represent the category tree
            self = cls(data, **kw)
        elif isinstance(data, cls):
            # If data is already a CategoryTree, do nothing and just return it
            self = data
            if len(kw):
                raise ValueError(
                    'kwargs cannot with this cls={}, type(data)={}'.format(
                        cls, type(data)))
        elif issubclass(cls, type(data)):
            # If we are an object that inherits from kwcoco.CategoryTree (e.g.
            # ndsampler.CategoryTree), but we are given a raw
            # kwcoco.CategoryTree, we need to try and upgrade the data
            # structure.
            self = cls(data.graph)
        else:
            raise TypeError(
                'Unknown type cls={}, type(data)={}: data={!r}'.format(
                    cls, type(data), data))
        return self

    @classmethod
    def demo(cls, key='coco', **kwargs):
        """
        Args:
            key (str): specify which demo dataset to use.
                Can be 'coco' (which uses the default coco demo data).
                Can be 'btree' which creates a binary tree and accepts kwargs 'r' and 'h' for branching-factor and height.
                Can be 'btree2', which is the same as btree but returns strings

        CommandLine:
            xdoctest -m ~/code/kwcoco/kwcoco/category_tree.py CategoryTree.demo

        Example:
            >>> from kwcoco.category_tree import *
            >>> self = CategoryTree.demo()
            >>> print('self = {}'.format(self))
            self = <CategoryTree(nNodes=10, maxDepth=2, maxBreadth=4...)>
        """
        if key == 'coco':
            from kwcoco import coco_dataset
            dset = coco_dataset.CocoDataset.demo(**kwargs)
            dset.add_category('background', id=0)
            graph = dset.category_graph()
        elif key == 'btree':
            r = kwargs.pop('r', 3)
            h = kwargs.pop('h', 3)
            graph = nx.generators.balanced_tree(r=r, h=h, create_using=OrderedDiGraph())
            graph = nx.relabel_nodes(graph, {n: n + 1 for n in graph})
            if kwargs.pop('add_zero', True):
                graph.add_node(0)
            assert not kwargs
        elif key == 'btree2':
            r = kwargs.pop('r', 3)
            h = kwargs.pop('h', 3)
            graph = nx.generators.balanced_tree(r=r, h=h, create_using=OrderedDiGraph())
            graph = nx.relabel_nodes(graph, {n: str(n + 1) for n in graph})
            if kwargs.pop('add_zero', True):
                graph.add_node(str(0))
            assert not kwargs
        elif key == 'animals_v1':
            graph = nx.from_dict_of_lists({
                'background': [],
                'foreground': ['animal'],
                'animal': ['mammal', 'fish', 'insect', 'reptile'],
                'mammal': ['dog', 'cat', 'human', 'zebra'],
                'zebra': ['grevys', 'plains'],
                'grevys': ['fred'],
                'dog': ['boxer', 'beagle', 'golden'],
                'cat': ['maine coon', 'persian', 'sphynx'],
                'reptile': ['bearded dragon', 't-rex'],
            }, OrderedDiGraph)
        else:
            raise KeyError(key)
        self = cls(graph)
        return self

    def to_coco(self):
        """
        Converts to a coco-style data structure

        Yields:
            Dict[str, Any]: coco category dictionaries
        """
        for cid, node in self.id_to_node.items():
            # Skip if background already added
            cat = {
                'id': cid,
                'name': node,
                **self.graph.nodes[node]
            }
            parents = list(self.graph.predecessors(node))
            if len(parents) == 1:
                cat['supercategory'] = parents[0]
            else:
                if len(parents) > 1:
                    raise Exception('not a tree')
            yield cat

    @ub.memoize_property
    def id_to_idx(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CategoryTree.demo()
            >>> self.id_to_idx[1]
        """
        return _calldict({cid: self.node_to_idx[node]
                         for cid, node in self.id_to_node.items()})

    @ub.memoize_property
    def idx_to_id(self):
        """
        Example:
            >>> import kwcoco
            >>> self = kwcoco.CategoryTree.demo()
            >>> self.idx_to_id[0]
        """
        return [self.node_to_id[node]
                for node in self.idx_to_node]

    @ub.memoize_method
    def idx_to_ancestor_idxs(self, include_self=True):
        """
        Mapping from a class index to its ancestors

        Args:
            include_self (bool, default=True):
                if True includes each node as its own ancestor.
        """
        lut = {
            idx: set(ub.take(self.node_to_idx, nx.ancestors(self.graph, node)))
            for idx, node in enumerate(self.idx_to_node)
        }
        if include_self:
            for idx, idxs in lut.items():
                idxs.update({idx})
        return lut

    @ub.memoize_method
    def idx_to_descendants_idxs(self, include_self=False):
        """
        Mapping from a class index to its descendants (including itself)

        Args:
            include_self (bool, default=False):
                if True includes each node as its own descendant.
        """
        lut = {
            idx: set(ub.take(self.node_to_idx, nx.descendants(self.graph, node)))
            for idx, node in enumerate(self.idx_to_node)
        }
        if include_self:
            for idx, idxs in lut.items():
                idxs.update({idx})
        return lut

    @ub.memoize_method
    def idx_pairwise_distance(self):
        """
        Get a matrix encoding the distance from one class to another.

        Distances
            * from parents to children are positive (descendants),
            * from children to parents are negative (ancestors),
            * between unreachable nodes (wrt to forward and reverse graph) are nan.
        """
        pdist = np.full((len(self), len(self)), fill_value=-np.nan,
                        dtype=np.float32)
        for node1, dists in nx.all_pairs_shortest_path_length(self.graph):
            idx1 = self.node_to_idx[node1]
            for node2, dist in dists.items():
                idx2 = self.node_to_idx[node2]
                pdist[idx1, idx2] = dist
                pdist[idx2, idx1] = -dist
        return pdist

    def __len__(self):
        """
        Returns:
            int
        """
        return len(self.graph)

    def __iter__(self):
        return iter(self.idx_to_node)

    def __getitem__(self, index):
        """
        Returns:
            str
        """
        return self.idx_to_node[index]

    def __contains__(self, node):
        return node in self.idx_to_node

    def __json__(self):
        """
        Example:
            >>> import pickle
            >>> self = CategoryTree.demo()
            >>> print('self = {!r}'.format(self.__json__()))
        """
        return self.__getstate__()

    def __getstate__(self):
        """
        Serializes information in this class

        Example:
            >>> from kwcoco.category_tree import *
            >>> import pickle
            >>> self = CategoryTree.demo()
            >>> state = self.__getstate__()
            >>> serialization = pickle.dumps(self)
            >>> recon = pickle.loads(serialization)
            >>> assert recon.__json__() == self.__json__()
        """
        state = self.__dict__.copy()
        for key in list(state.keys()):
            if key.startswith('_cache'):
                state.pop(key)
        state['graph'] = to_directed_nested_tuples(self.graph)
        if True:
            # Remove redundant items
            state.pop('node_to_idx')
            state.pop('node_to_id')
            state.pop('idx_groups')
        return state

    def __setstate__(self, state):
        graph = from_directed_nested_tuples(state['graph'])
        need_reindex = False

        if True:
            # Reconstruct redundant items
            if 'node_to_idx' not in state:
                state['node_to_idx'] = {node: idx for idx, node in
                                        enumerate(state['idx_to_node'])}
            if 'node_to_id' not in state:
                state['node_to_id'] = {node: id for id, node in
                                       state['id_to_node'].items()}

            if 'idx_groups' not in state:
                node_groups = list(traverse_siblings(graph))
                node_to_idx = state['node_to_idx']
                try:
                    state['idx_groups'] = [sorted([node_to_idx[n] for n in group])
                                           for group in node_groups]
                except KeyError:
                    need_reindex = True
                    pass

        self.__dict__.update(state)
        self.graph = graph
        if need_reindex:
            self._build_index()

    def __nice__(self):
        max_depth = tree_depth(self.graph)
        if max_depth > 1:
            max_breadth = max(it.chain([0], map(len, self.idx_groups)))
            text = 'nNodes={}, maxDepth={}, maxBreadth={}, nodes={}'.format(
                self.num_classes, max_depth, max_breadth, self.idx_to_node,
            )
        else:
            text = 'nNodes={}, nodes={}'.format(
                self.num_classes, self.idx_to_node,
            )
        return text

    def is_mutex(self):
        """
        Returns True if all categories are mutually exclusive (i.e. flat)

        If true, then the classes may be represented as a simple list of class
        names without any loss of information, otherwise the underlying
        category graph is necessary to preserve all knowledge.

        TODO:
            - [ ] what happens when we have a dummy root?
        """
        return len(self.graph.edges) == 0

    @property
    def num_classes(self):
        return self.graph.number_of_nodes()

    @property
    def class_names(self):
        return self.idx_to_node

    @property
    def category_names(self):
        return self.idx_to_node

    @property
    def cats(self):
        """
        Returns a mapping from category names to category attributes.

        If this category tree was constructed from a coco-dataset, then this
        will contain the coco category attributes.

        Returns:
            Dict[str, Dict[str, object]]

        Example:
            >>> from kwcoco.category_tree import *
            >>> self = CategoryTree.demo()
            >>> print('self.cats = {!r}'.format(self.cats))
        """
        return dict(self.graph.nodes)

    def index(self, node):
        """
        Return the index that corresponds to the category name

        Args:
            node (str): the name of the category

        Returns:
            int
        """
        return self.node_to_idx[node]

    def take(self, indexes):
        """
        Create a subgraph based on the selected class indexes

        Ignore:
            >>> self = CategoryTree.from_mutex(['c', 'b', 'a', 'd'])
            >>> print(f'self.idx_to_node={self.idx_to_node}')
            >>> indexes = [1, 0, 2, 3]
            >>> print()
        """
        subnodes = list(ub.take(self.idx_to_node, indexes))
        return self.subgraph(subnodes)

    def subgraph(self, subnodes, closure=True):
        """
        Create a subgraph based on the selected class nodes (i.e. names)

        Example:
            >>> self = CategoryTree.from_coco([
            >>>     {'id': 130, 'name': 'n3', 'supercategory': 'n1'},
            >>>     {'id': 410, 'name': 'n1', 'supercategory': None},
            >>>     {'id': 640, 'name': 'n4', 'supercategory': 'n3'},
            >>>     {'id': 220, 'name': 'n2', 'supercategory': 'n1'},
            >>>     {'id': 560, 'name': 'n6', 'supercategory': 'n2'},
            >>>     {'id': 350, 'name': 'n5', 'supercategory': 'n2'},
            >>> ])
            >>> self.print_graph()
            >>> subnodes = ['n3', 'n6', 'n4', 'n1']
            >>> new1 = self.subgraph(subnodes, closure=1)
            >>> new1.print_graph()
            ...
            >>> print('new1.idx_to_id = {}'.format(ub.urepr(new1.idx_to_id, nl=0)))
            >>> print('new1.idx_to_node = {}'.format(ub.urepr(new1.idx_to_node, nl=0)))
            new1.idx_to_id = [130, 560, 640, 410]
            new1.idx_to_node = ['n3', 'n6', 'n4', 'n1']

            >>> indexes = [2, 1, 0, 5]
            >>> new2 = self.take(indexes)
            >>> new2.print_graph()
            ...
            >>> print('new2.idx_to_id = {}'.format(ub.urepr(new2.idx_to_id, nl=0)))
            >>> print('new2.idx_to_node = {}'.format(ub.urepr(new2.idx_to_node, nl=0)))
            new2.idx_to_id = [640, 410, 130, 350]
            new2.idx_to_node = ['n4', 'n1', 'n3', 'n5']

            >>> subnodes = ['n3', 'n6', 'n4', 'n1']
            >>> new3 = self.subgraph(subnodes, closure=0)
            >>> new3.print_graph()
        """
        assert RESPECT_INPUT_ORDER, 'subgraph requires new input order logic'
        new_graph  = self.graph.__class__()
        for node in subnodes:
            node_data = self.graph.nodes[node]
            new_graph.add_node(node, **node_data)
        if closure:
            nbunch = set(subnodes)
            graph_closure = nx.transitive_closure(self.graph)
            subgraph_closure = graph_closure.subgraph(subnodes)
            subgraph_reduction = nx.transitive_reduction(subgraph_closure)
            for u, v, edge_data in subgraph_reduction.edges(nbunch=subnodes, data=True):
                if u in nbunch and v in nbunch:
                    new_graph.add_edge(u, v, **edge_data)
        else:
            for u, v, edge_data in self.graph.edges(nbunch=subnodes, data=True):
                new_graph.add_edge(u, v, **edge_data)
        new = self.__class__(new_graph)
        return new

    def _build_index(self):
        """
        construct lookup tables
        """
        # Most of the categories should have been given integer ids
        max_id = max(it.chain([0], nx.get_node_attributes(self.graph, 'id').values()))

        if isinstance(self.graph, OrderedDiGraph):
            # Use the graph order if the graph is ordered
            # This is the only case that will trigger on Python 3.7+
            # with modern networkx
            node_attrs = list(self.graph.nodes.items())
        else:
            # Otherwise sort (this is only the case on older networkx versions)
            node_attrs = sorted(self.graph.nodes.items())

        # Fill in id-values for any node that doesn't have one
        node_to_id = {}
        for node, attrs in node_attrs:
            node_to_id[node] = attrs.get('id', max_id + 1)
            max_id = max(max_id, node_to_id[node])
        id_to_node = ub.invert_dict(node_to_id)

        if RESPECT_INPUT_ORDER:
            # list(self.graph.nodes.items())
            if isinstance(self.graph, OrderedDiGraph):
                idx_to_node = list(self.graph.nodes)
            else:
                idx_to_node = ub.argsort(node_to_id)
        else:
            # TODO: respect user-given ordering over sorting by ids
            # Compress ids into a flat index space (sorted by node ids)
            idx_to_node = ub.argsort(node_to_id)

        node_to_idx = {node: idx for idx, node in enumerate(idx_to_node)}

        # Find the sets of nodes that need to be softmax-ed together
        node_groups = list(traverse_siblings(self.graph))
        idx_groups = [sorted([node_to_idx[n] for n in group])
                      for group in node_groups]

        # Set instance attributes
        self.id_to_node = id_to_node
        self.node_to_id = node_to_id
        self.idx_to_node = idx_to_node
        self.node_to_idx = node_to_idx
        self.idx_groups = idx_groups

    def show(self):
        """

        Ignore:
            >>> import kwplot
            >>> kwplot.autompl()
            >>> from kwcoco import category_tree
            >>> self = category_tree.CategoryTree.demo()
            >>> self.show()

            python -c "import kwplot, kwcoco, graphid; kwplot.autompl(); graphid.util.show_nx(kwcoco.category_tree.CategoryTree.demo().graph); kwplot.show_if_requested()" --show
        """
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
        except ImportError:
            import warnings
            warnings.warn('pygraphviz is not available')
            pos = None
        nx.draw_networkx(self.graph, pos=pos)
        # import graphid
        # graphid.util.show_nx(self.graph)

    def forest_str(self):
        import networkx as nx
        text = nx.forest_str(self.graph)
        # print(text)
        return text

    def print_graph(self):
        import networkx as nx
        try:
            nx.write_network_text(self.graph)
        except AttributeError:
            from kwcoco.util import util_networkx
            util_networkx.write_network_text(self.graph)

    def normalize(self):
        """
        Applies a normalization scheme to the categories.

        Note: this may break other tasks that depend on exact category names.

        Returns:
            CategoryTree

        Example:
            >>> from kwcoco.category_tree import *  # NOQA
            >>> import kwcoco
            >>> orig = kwcoco.CategoryTree.demo('animals_v1')
            >>> self = kwcoco.CategoryTree(nx.relabel_nodes(orig.graph, str.upper))
            >>> norm = self.normalize()
        """
        # nx.adjacency_data(self.graph)
        def normalize_name(name):
            return name.lower().replace(' ', '')

        new_graph = self.graph.__class__()

        node_mapping = {}
        new_nodes = []
        for old_node, old_data in self.graph.nodes(data=True):
            new_node = normalize_name(old_node)
            new_data = old_data.copy()
            if 'id' not in old_data:
                new_data['id'] = self.node_to_id[old_node]
            if 'supercategory' in old_data:
                new_data['supercategory'] = normalize_name(old_data['supercategory'])
            if 'name' in old_data:
                new_data['name'] = normalize_name(old_data['name'])
            new_nodes.append((new_node, new_data))
            node_mapping[old_node] = new_node
            new_graph.add_node(new_node, **new_data)

        for old_u, old_v, old_data in self.graph.edges(data=True):
            new_u = node_mapping[old_u]
            new_v = node_mapping[old_v]
            new_data = old_data.copy()
            new_graph.add_edge(new_u, new_v, **new_data)

        new = self.__class__(new_graph)
        return new

        # json_data = nx.node_link_data(self.graph)
        # for path, data in ub.IndexableWalker(json_data):
        #     pass
        # nx.cytoscape_data(self.graph)
        # nx.node_link_graph(self.graph)
        # to_directed_nested_tuples(self.graph)
        # for
        # name.lower().replace(' ', '_')
        # self.idx_to_node


def source_nodes(graph):
    """ generates source nodes --- nodes without incoming edges """
    return (n for n in graph.nodes() if graph.in_degree(n) == 0)


def sink_nodes(graph):
    """ generates source nodes --- nodes without incoming edges """
    return (n for n in graph.nodes() if graph.out_degree(n) == 0)


def traverse_siblings(graph, sources=None):
    """ generates groups of nodes that have the same parent """
    if sources is None:
        sources = list(source_nodes(graph))
    yield sources
    for node in sources:
        children = list(graph.successors(node))
        if children:
            for _ in traverse_siblings(graph, children):
                yield _


def tree_depth(graph, root=None):
    """
    Maximum depth of the forest / tree

    Example:
        >>> from kwcoco.category_tree import *
        >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        >>> tree_depth(graph)
        4
        >>> tree_depth(nx.balanced_tree(r=2, h=0, create_using=nx.DiGraph))
        1
    """
    if len(graph) == 0:
        return 0
    if root is not None:
        assert root in graph.nodes
    assert nx.is_forest(graph)
    def _inner(root):
        if root is None:
            return max(it.chain([0], (_inner(n) for n in source_nodes(graph))))
        else:
            return max(it.chain([0], (_inner(n) for n in graph.successors(root)))) + 1
    depth = _inner(root)
    return depth


def to_directed_nested_tuples(graph, with_data=True):
    """
    Serialize a networkx graph.

    Encodes each node and its children in a tuple as:
        (node, children)
    """
    def _represent_node(node):
        if with_data:
            node_data = graph.nodes[node]
            return (node, node_data, _traverse_encode(node))
        else:
            return (node, _traverse_encode(node))

    def _traverse_encode(parent):
        children = sorted(graph.successors(parent))
        # graph.get_edge_data(node, child)
        return [_represent_node(node) for node in children]

    sources = sorted(source_nodes(graph))
    encoding = [_represent_node(node) for node in sources]
    return encoding


def from_directed_nested_tuples(encoding):
    """
    Unserialize a networkx graph.

    Example:
        >>> from kwcoco.category_tree import *
        >>> graph = nx.generators.gnr_graph(20, 0.3, seed=790).reverse()
        >>> graph.nodes[0]['color'] = 'black'
        >>> encoding = to_directed_nested_tuples(graph)
        >>> recon = from_directed_nested_tuples(encoding)
        >>> recon_encoding = to_directed_nested_tuples(recon)
        >>> assert recon_encoding == encoding
    """
    node_data_view = {}
    def _traverse_recon(tree):
        nodes = []
        edges = []
        for tup in tree:
            if len(tup) == 2:
                node, subtree = tup
            elif len(tup) == 3:
                node, node_data, subtree = tup
                node_data_view[node] = node_data
            else:
                raise AssertionError('invalid tup')
            children = [t[0] for t in subtree]
            nodes.append(node)
            edges.extend((node, child) for child in children)
            subnodes, subedges = _traverse_recon(subtree)
            nodes.extend(subnodes)
            edges.extend(subedges)
        return nodes, edges
    nodes, edges = _traverse_recon(encoding)
    graph = OrderedDiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    for k, v in node_data_view.items():
        graph.nodes[k].update(v)
    return graph


class _calldict(dict):
    """
    helper object to maintain backwards compatibility between new and old
    id_to_idx methods.

    Example:
        >>> self = _calldict({1: 2})
        >>> #assert self()[1] == 2
        >>> assert self[1] == 2
    """

    def __call__(self):
        import warnings
        warnings.warn('Calling id_to_idx as a method has been deprecated. '
                      'Use this dict as a property')
        return self


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m kwcoco.category_tree
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
