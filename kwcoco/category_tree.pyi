from typing import List
from typing import Dict
import networkx
import networkx as nx
import ubelt as ub
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any


class CategoryTree(ub.NiceRepr):
    idx_to_node: List[str]
    id_to_node: Dict[int, str]
    node_to_id: Dict[str, int]
    node_to_idx: Dict[str, int]
    graph: networkx.Graph
    idx_groups: List[List[int]]

    def __init__(self,
                 graph: nx.DiGraph | None = None,
                 checks: bool = True) -> None:
        ...

    def copy(self):
        ...

    @classmethod
    def from_mutex(cls, nodes: List[str], bg_hack: bool = ...):
        ...

    @classmethod
    def from_json(cls, state: Dict):
        ...

    @classmethod
    def from_coco(cls, categories):
        ...

    @classmethod
    def coerce(cls, data: object, **kw) -> CategoryTree:
        ...

    @classmethod
    def demo(cls, key: str = 'coco', **kwargs):
        ...

    def to_coco(self) -> Generator[Dict, None, None]:
        ...

    def id_to_idx(self):
        ...

    def idx_to_id(self):
        ...

    def idx_to_ancestor_idxs(self, include_self: bool = True):
        ...

    def idx_to_descendants_idxs(self, include_self: bool = False):
        ...

    def idx_pairwise_distance(self):
        ...

    def __len__(self):
        ...

    def __iter__(self):
        ...

    def __getitem__(self, index):
        ...

    def __contains__(self, node):
        ...

    def __json__(self):
        ...

    def __nice__(self):
        ...

    def is_mutex(self):
        ...

    @property
    def num_classes(self):
        ...

    @property
    def class_names(self):
        ...

    @property
    def category_names(self):
        ...

    @property
    def cats(self) -> Dict[str, Dict[str, object]]:
        ...

    def index(self, node):
        ...

    def show(self) -> None:
        ...

    def forest_str(self):
        ...

    def normalize(self) -> CategoryTree:
        ...


def source_nodes(graph):
    ...


def sink_nodes(graph):
    ...


def traverse_siblings(graph,
                      sources: Incomplete | None = ...
                      ) -> Generator[Any, None, None]:
    ...


def tree_depth(graph, root: Incomplete | None = ...):
    ...


def to_directed_nested_tuples(graph, with_data: bool = ...):
    ...


def from_directed_nested_tuples(encoding):
    ...


class _calldict(dict):

    def __call__(self):
        ...
