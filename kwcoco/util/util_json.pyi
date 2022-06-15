from typing import Dict
from typing import List
from _typeshed import Incomplete
from collections.abc import Generator

IndexableWalker: Incomplete


def ensure_json_serializable(dict_,
                             normalize_containers: bool = False,
                             verbose: int = ...):
    ...


def find_json_unserializable(
        data: object,
        quickcheck: bool = False) -> Generator[List[Dict], None, None]:
    ...


def indexable_allclose(dct1, dct2, return_info: bool = ...):
    ...
