import copy
import numpy as np
import ubelt as ub
import json
from collections import OrderedDict
import decimal
import fractions
import pathlib
from typing import NamedTuple, Tuple, Any

# backwards compat
IndexableWalker = ub.IndexableWalker


def ensure_json_serializable(dict_, normalize_containers=False, verbose=0):
    """
    Attempt to convert common types (e.g. numpy) into something json complient

    Convert numpy and tuples into lists

    Args:
        normalize_containers (bool):
            if True, normalizes dict containers to be standard python
            structures. Defaults to False.

    Example:
        >>> data = ub.ddict(lambda: int)
        >>> data['foo'] = ub.ddict(lambda: int)
        >>> data['bar'] = np.array([1, 2, 3])
        >>> data['foo']['a'] = 1
        >>> data['foo']['b'] = (1, np.array([1, 2, 3]), {3: np.int32(3), 4: np.float16(1.0)})
        >>> dict_ = data
        >>> print(ub.urepr(data, nl=-1))
        >>> assert list(find_json_unserializable(data))
        >>> result = ensure_json_serializable(data, normalize_containers=True)
        >>> print(ub.urepr(result, nl=-1))
        >>> assert not list(find_json_unserializable(result))
        >>> assert type(result) is dict
    """
    dict_ = copy.deepcopy(dict_)

    def _norm_container(c):
        if isinstance(c, dict):
            # Cast to a normal dictionary
            if isinstance(c, OrderedDict):
                if type(c) is not OrderedDict:
                    c = OrderedDict(c)
            else:
                if type(c) is not dict:
                    c = dict(c)
        return c

    walker = ub.IndexableWalker(dict_)
    for prefix, value in walker:
        if isinstance(value, tuple):
            new_value = list(value)
            walker[prefix] = new_value
        elif isinstance(value, np.ndarray):
            new_value = value.tolist()
            walker[prefix] = new_value
        elif isinstance(value, (np.integer)):
            new_value = int(value)
            walker[prefix] = new_value
        elif isinstance(value, (np.floating)):
            new_value = float(value)
            walker[prefix] = new_value
        elif isinstance(value, (np.complexfloating)):
            new_value = complex(value)
            walker[prefix] = new_value
        elif isinstance(value, decimal.Decimal):
            new_value = float(value)
            walker[prefix] = new_value
        elif isinstance(value, fractions.Fraction):
            new_value = float(value)
            walker[prefix] = new_value
        elif isinstance(value, pathlib.Path):
            new_value = str(value)
            walker[prefix] = new_value
        elif hasattr(value, '__json__'):
            new_value = value.__json__()
            walker[prefix] = new_value
        elif normalize_containers:
            if isinstance(value, dict):
                new_value = _norm_container(value)
                walker[prefix] = new_value

    if normalize_containers:
        # normalize the outer layer
        dict_ = _norm_container(dict_)
    return dict_


def find_json_unserializable(data, quickcheck=False):
    """
    Recurse through json datastructure and find any component that
    causes a serialization error. Record the location of these errors
    in the datastructure as we recurse through the call tree.

    Args:
        data (object): data that should be json serializable
        quickcheck (bool): if True, check the entire datastructure assuming
            its ok before doing the python-based recursive logic.

    Returns:
        List[Dict]: list of "bad part" dictionaries containing items

            'value' - the value that caused the serialization error

            'loc' - which contains a list of key/indexes that can be used
            to lookup the location of the unserializable value.
            If the "loc" is a list, then it indicates a rare case where
            a key in a dictionary is causing the serialization error.

    Example:
        >>> from kwcoco.util.util_json import *  # NOQA
        >>> part = ub.ddict(lambda: int)
        >>> part['foo'] = ub.ddict(lambda: int)
        >>> part['bar'] = np.array([1, 2, 3])
        >>> part['foo']['a'] = 1
        >>> # Create a dictionary with two unserializable parts
        >>> data = [1, 2, {'nest1': [2, part]}, {frozenset({'badkey'}): 3, 2: 4}]
        >>> parts = list(find_json_unserializable(data))
        >>> print('parts = {}'.format(ub.urepr(parts, nl=1)))
        >>> # Check expected structure of bad parts
        >>> assert len(parts) == 2
        >>> part = parts[1]
        >>> assert list(part['loc']) == [2, 'nest1', 1, 'bar']
        >>> # We can use the "loc" to find the bad value
        >>> for part in parts:
        >>>     # "loc" is a list of directions containing which keys/indexes
        >>>     # to traverse at each descent into the data structure.
        >>>     directions = part['loc']
        >>>     curr = data
        >>>     special_flag = False
        >>>     for key in directions:
        >>>         if isinstance(key, list):
        >>>             # special case for bad keys
        >>>             special_flag = True
        >>>             break
        >>>         else:
        >>>             # normal case for bad values
        >>>             curr = curr[key]
        >>>     if special_flag:
        >>>         assert part['data'] in curr.keys()
        >>>         assert part['data'] is key[1]
        >>>     else:
        >>>         assert part['data'] is curr

    Example:
        >>> # xdoctest: +SKIP("TODO: circular ref detect algo is wrong, fix it")
        >>> from kwcoco.util.util_json import *  # NOQA
        >>> import pytest
        >>> # Test circular reference
        >>> data = [[], {'a': []}]
        >>> data[1]['a'].append(data)
        >>> with pytest.raises(ValueError, match="Circular reference detected at.*1, 'a', 1*"):
        ...     parts = list(find_json_unserializable(data))
        >>> # Should be ok here
        >>> shared_data = {'shared': 1}
        >>> data = [[shared_data], shared_data]
        >>> parts = list(find_json_unserializable(data))

    """
    needs_check = True

    if quickcheck:
        try:
            # Might be a more efficient way to do this check. We duplicate a lot of
            # work by doing the check for unserializable data this way.
            json.dumps(data)
        except Exception:
            # if 'Circular reference detected' in str(ex):
            #     has_circular_reference = True
            # If there is unserializable data, find out where it is.
            # is_serializable = False
            pass
        else:
            # is_serializable = True
            needs_check = False

    # FIXME: the algo is wrong, fails when
    CHECK_FOR_CIRCULAR_REFERENCES = 0

    if needs_check:
        # mode = 'new'
        # if mode == 'new':
        scalar_types = (int, float, str, type(None))
        container_types = (tuple, list, dict)
        serializable_types = scalar_types + container_types
        walker = ub.IndexableWalker(data)

        if CHECK_FOR_CIRCULAR_REFERENCES:
            seen_ids = set()

        for prefix, value in walker:

            if CHECK_FOR_CIRCULAR_REFERENCES:
                # FIXME: We need to know if this container id is in this paths
                # ancestors. It is allowed to be elsewhere in the data
                # structure (i.e. the pointer graph must be a DAG)
                if isinstance(value, container_types):
                    container_id = id(value)
                    if container_id in seen_ids:
                        circ_loc = {'loc': prefix, 'data': value}
                        raise ValueError(f'Circular reference detected at {circ_loc}')
                    seen_ids.add(container_id)

            *root, key = prefix
            if not isinstance(key, scalar_types):
                # Special case where a dict key is the error value
                # Purposely make loc non-hashable so its not confused with
                # an address. All we can know in this case is that they key
                # is at this level, there is no concept of where.
                yield {'loc': root + [['.keys', key]], 'data': key}
            elif not isinstance(value, serializable_types):
                yield {'loc': prefix, 'data': value}


def indexable_allclose(dct1, dct2, return_info=False):
    """
    Walks through two nested data structures and ensures that everything is
    roughly the same.

    NOTE:
        Use the version in ubelt instead

    Args:
        dct1: a nested indexable item
        dct2: a nested indexable item

    Example:
        >>> from kwcoco.util.util_json import indexable_allclose
        >>> dct1 = {
        >>>     'foo': [1.222222, 1.333],
        >>>     'bar': 1,
        >>>     'baz': [],
        >>> }
        >>> dct2 = {
        >>>     'foo': [1.22222, 1.333],
        >>>     'bar': 1,
        >>>     'baz': [],
        >>> }
        >>> assert indexable_allclose(dct1, dct2)
    """
    walker1 = ub.IndexableWalker(dct1)
    walker2 = ub.IndexableWalker(dct2)
    flat_items1 = [
        (path, value) for path, value in walker1
        if not isinstance(value, walker1.indexable_cls) or len(value) == 0]
    flat_items2 = [
        (path, value) for path, value in walker2
        if not isinstance(value, walker1.indexable_cls) or len(value) == 0]

    flat_items1 = sorted(flat_items1)
    flat_items2 = sorted(flat_items2)

    if len(flat_items1) != len(flat_items2):
        info = {
            'faillist': ['length mismatch']
        }
        final_flag = False
    else:
        passlist = []
        faillist = []

        for t1, t2 in zip(flat_items1, flat_items2):
            p1, v1 = t1
            p2, v2 = t2
            assert p1 == p2

            flag = (v1 == v2)
            if not flag:
                if isinstance(v1, float) and isinstance(v2, float) and np.isclose(v1, v2):
                    flag = True
            if flag:
                passlist.append(p1)
            else:
                faillist.append((p1, v1, v2))

        final_flag = len(faillist) == 0
        info = {
            'passlist': passlist,
            'faillist': faillist,
        }

    if return_info:
        return final_flag, info
    else:
        return final_flag


class Difference(NamedTuple):
    """
    A result class of indexable_diff that organizes what the difference between
    the indexables is.
    """
    path: Tuple
    value1: Any
    value2: Any


def indexable_diff(dct1, dct2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Walks through two nested data structures finds differences in the
    structures.

    Args:
        dct1: a nested indexable item
        dct2: a nested indexable item

    Returns:
        dict: information about the diff with
            "similarity": a score between 0 and 1
            "num_differences" being the number of paths not common plus the
                number of common paths with differing values.
            "unique1": being the paths that were unique to dct1
            "unique2": being the paths that were unique to dct2
            "faillist": a list 3-tuples of common path and differing values
            "approximations":
                is the number of approximately equal items (i.e. floats) there were

    Example:
        >>> from kwcoco.util.util_json import indexable_diff
        >>> dct1 = {
        >>>     'foo': [1.222222, 1.333],
        >>>     'bar': 1,
        >>>     'baz': [],
        >>>     'top': [1, 2, 3],
        >>>     'L0': {'L1': {'L2': {'K1': 'V1', 'K2': 'V2', 'D1': 1, 'D2': 2}}},
        >>> }
        >>> dct2 = {
        >>>     'foo': [1.22222, 1.333],
        >>>     'bar': 1,
        >>>     'baz': [],
        >>>     'buz': {1: 2},
        >>>     'top': [1, 1, 2],
        >>>     'L0': {'L1': {'L2': {'K1': 'V1', 'K2': 'V2', 'D1': 10, 'D2': 20}}},
        >>> }
        >>> info = indexable_diff(dct1, dct2)
        >>> print(f'info = {ub.urepr(info, nl=2)}')
    """
    walker1 = ub.IndexableWalker(dct1)
    walker2 = ub.IndexableWalker(dct2)
    flat_items1 = {
        tuple(path): value for path, value in walker1
        if not isinstance(value, walker1.indexable_cls) or len(value) == 0}
    flat_items2 = {
        tuple(path): value for path, value in walker2
        if not isinstance(value, walker1.indexable_cls) or len(value) == 0}

    common = flat_items1.keys() & flat_items2.keys()
    unique1 = flat_items1.keys() - flat_items2.keys()
    unique2 = flat_items2.keys() - flat_items1.keys()

    approximations = 0

    faillist = []
    passlist = []
    for key in common:
        v1 = flat_items1[key]
        v2 = flat_items2[key]
        flag = (v1 == v2)
        if not flag:
            if isinstance(v1, float) and isinstance(v2, float) and np.isclose(v1, v2, rtol=rtol, atol=atol, equal_nan=equal_nan):
                approximations += 1
                flag = True
        if flag:
            passlist.append(key)
        else:
            faillist.append(Difference(key, v1, v2))

    num_differences = len(unique1) + len(unique2) + len(faillist)
    num_similarities = len(passlist)

    similarity = num_similarities / (num_similarities + num_differences)
    info = {
        'similarity': similarity,
        'approximations': approximations,
        'num_differences': num_differences,
        'num_similarities': num_similarities,
        'unique1': unique1,
        'unique2': unique2,
        'faillist': faillist,
        'passlist': passlist,
    }
    return info


def coerce_indent(indent):
    """
    Example:
        .. code:: python
            print(repr(coerce_indent(None)))
            print(repr(coerce_indent('   ')))
            print(repr(coerce_indent(3)))
    """
    if indent is not None and isinstance(indent, str):
        assert indent.count(' ') == len(indent), (
            'must be all spaces, got {!r}'.format(indent))
        indent = len(indent)
    if indent is None:
        ...
        # indent = 0  # Can't do this. It introduces a bug
    return indent
