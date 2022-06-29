"""
Abstract nodes
"""

import numpy as np
import ubelt as ub


# from kwcoco.util.util_monkey import Reloadable  # NOQA
# @Reloadable.developing
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
        graph = nx.DiGraph()
        stack = [self]
        while stack:
            item = stack.pop()
            node_id = id(item)
            graph.add_node(node_id)
            sub_meta = {k: v for k, v in item.meta.items() if v is not None}
            if 'transform' in sub_meta:
                sub_meta['transform'] = sub_meta['transform'].concise()
                sub_meta['transform'].pop('type')
            param_key = ub.repr2(sub_meta, sort=0, compact=1, nl=0)
            name = item.__class__.__name__.replace('Delayed', '')
            node_data = graph.nodes[node_id]
            node_data['label'] = f'{name} {param_key}'
            node_data['name'] = name
            node_data['meta'] = sub_meta
            for child in item.children():
                child_id = id(child)
                graph.add_edge(node_id, child_id)
                stack.append(child)
        return graph

    def write_network_text(self, with_labels=True):
        from cmd_queue.util import graph_str
        graph = self.as_graph()
        # import networkx as nx
        # nx.write_network_text(graph)
        print(graph_str(graph, with_labels=with_labels))

    @property
    def shape(self):
        raise NotImplementedError

    def children(self):
        raise NotImplementedError

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
        yield from iter(self.parts)


class DelayedUnaryOperation2(DelayedOperation2):
    """
    For operations that have a single input array
    """
    def __init__(self, subdata):
        super().__init__()
        self.subdata = subdata

    def children(self):
        if self.subdata is not None:
            yield self.subdata
