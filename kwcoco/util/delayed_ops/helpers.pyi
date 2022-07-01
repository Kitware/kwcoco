from numpy import ndarray
from typing import Dict
from typing import Any
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any

profile: Incomplete


def dequantize(quant_data: ndarray, quantization: Dict[str, Any]) -> ndarray:
    ...


def quantize_float01(imdata,
                     old_min: int = ...,
                     old_max: int = ...,
                     quantize_dtype=...):
    ...


class _AsciiBaseGlyphs:
    empty: str
    newtree_last: str
    newtree_mid: str
    endof_forest: str
    within_forest: str
    within_tree: str


class AsciiDirectedGlyphs(_AsciiBaseGlyphs):
    last: str
    mid: str
    backedge: str


class AsciiUndirectedGlyphs(_AsciiBaseGlyphs):
    last: str
    mid: str
    backedge: str


class _UtfBaseGlyphs:
    empty: str
    newtree_last: str
    newtree_mid: str
    endof_forest: str
    within_forest: str
    within_tree: str


class UtfDirectedGlyphs(_UtfBaseGlyphs):
    last: str
    mid: str
    backedge: str


class UtfUndirectedGlyphs(_UtfBaseGlyphs):
    last: str
    mid: str
    backedge: str


def generate_network_text(graph,
                          with_labels: bool = ...,
                          sources: Incomplete | None = ...,
                          max_depth: Incomplete | None = ...,
                          ascii_only: bool = ...) -> Generator[Any, None, Any]:
    ...


def write_network_text(graph,
                       path: Incomplete | None = ...,
                       with_labels: bool = ...,
                       sources: Incomplete | None = ...,
                       max_depth: Incomplete | None = ...,
                       ascii_only: bool = ...,
                       end: str = ...) -> None:
    ...


def graph_str(graph,
              with_labels: bool = ...,
              sources: Incomplete | None = ...,
              write: Incomplete | None = ...,
              ascii_only: bool = ...):
    ...
