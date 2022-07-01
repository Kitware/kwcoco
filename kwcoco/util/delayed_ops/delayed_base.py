"""
Abstract nodes
"""

import numpy as np
import ubelt as ub


# from kwcoco.util.util_monkey import Reloadable  # NOQA
# @Reloadable.developing  # NOQA
class DelayedOperation2(ub.NiceRepr):

    def __init__(self):
        self.meta = {}

    def __nice__(self):
        return '{}'.format(self.shape)

    def nesting(self):
        def _child_nesting(child):
            if hasattr(child, 'nesting'):
                return child.nesting()
            elif isinstance(child, np.ndarray):
                return {
                    'type': 'ndarray',
                    'shape': self.subdata.shape,
                }
        item = {
            'type': self.__class__.__name__,
            'meta': self.meta,
        }
        child_nodes = list(self.children())
        if child_nodes:
            child_nestings = [_child_nesting(child) for child in child_nodes]
            item['children'] = child_nestings
        return item

    def as_graph(self):
        """
        """
        import networkx as nx
        import itertools as it
        counter = it.count(0)
        graph = nx.DiGraph()
        stack = [(None, self)]
        while stack:
            parent_id, item = stack.pop()
            # There might be copies of the same node in concat graphs so, we
            # cant assume the id will be unique. We can assert a forest
            # structure though.
            node_id = f'{next(counter):03d}_{id(item)}'
            graph.add_node(node_id)
            if parent_id is not None:
                graph.add_edge(parent_id, node_id)
            sub_meta = {k: v for k, v in item.meta.items() if v is not None}
            if 'transform' in sub_meta:
                sub_meta['transform'] = sub_meta['transform'].concise()
                sub_meta['transform'].pop('type')
            param_key = ub.repr2(sub_meta, sort=0, compact=1, nl=0)
            short_type = item.__class__.__name__.replace('Delayed', '').replace('2', '')
            node_data = graph.nodes[node_id]
            node_data['label'] = f'{short_type} {param_key}'
            node_data['short_type'] = short_type
            node_data['type'] = item.__class__.__name__
            node_data['meta'] = sub_meta
            for child in item.children():
                stack.append((node_id, child))
        return graph

    def write_network_text(self, with_labels=True):
        from kwcoco.util.delayed_ops.helpers import write_network_text
        graph = self.as_graph()
        # TODO: remove once this is merged into networkx itself
        write_network_text(graph, with_labels=with_labels)

    @property
    def shape(self):
        raise NotImplementedError

    def children(self):
        """
        Yields:
            Any:
        """
        raise NotImplementedError
        yield None

    def finalize(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError


class DelayedNaryOperation2(DelayedOperation2):
    """
    For operations that have multiple input arrays
    """
    def __init__(self, parts):
        super().__init__()
        self.parts = parts

    def children(self):
        """
        Yields:
            Any:
        """
        yield from iter(self.parts)


class DelayedUnaryOperation2(DelayedOperation2):
    """
    For operations that have a single input array
    """
    def __init__(self, subdata):
        super().__init__()
        self.subdata = subdata

    def children(self):
        """
        Yields:
            Any:
        """
        if self.subdata is not None:
            yield self.subdata
