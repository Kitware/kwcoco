import copy
import numpy as np
import ubelt as ub
import json
from collections import OrderedDict
from collections import deque


def ensure_json_serializable(dict_, normalize_containers=False, verbose=0):
    """
    Attempt to convert common types (e.g. numpy) into something json complient

    Convert numpy and tuples into lists

    Args:
        normalize_containers (bool, default=False):
            if True, normalizes dict containers to be standard python
            structures.

    Example:
        >>> data = ub.ddict(lambda: int)
        >>> data['foo'] = ub.ddict(lambda: int)
        >>> data['bar'] = np.array([1, 2, 3])
        >>> data['foo']['a'] = 1
        >>> result = ensure_json_serializable(data, normalize_containers=True)
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

    # inplace convert any ndarrays to lists
    def _walk_json(data, prefix=[]):
        items = None
        if isinstance(data, list):
            items = enumerate(data)
        elif isinstance(data, tuple):
            items = enumerate(data)
        elif isinstance(data, dict):
            items = data.items()
        else:
            raise TypeError(type(data))

        root = prefix
        level = {}
        for key, value in items:
            level[key] = value

        # yield a dict so the user can choose to not walk down a path
        yield root, level

        for key, value in level.items():
            if isinstance(value, (dict, list, tuple)):
                path = prefix + [key]
                for _ in _walk_json(value, prefix=path):
                    yield _

    def _convert(dict_, root, key, new_value):
        d = dict_
        for k in root:
            d = d[k]
        d[key] = new_value

    def _flatmap(func, data):
        if isinstance(data, list):
            return [_flatmap(func, item) for item in data]
        else:
            return func(data)

    to_convert = []
    for root, level in ub.ProgIter(_walk_json(dict_), desc='walk json',
                                   verbose=verbose):
        for key, value in level.items():
            if isinstance(value, tuple):
                # Convert tuples on the fly so they become mutable
                new_value = list(value)
                _convert(dict_, root, key, new_value)
            elif isinstance(value, np.ndarray):
                new_value = value.tolist()
                if 0:
                    if len(value.shape) == 1:
                        if value.dtype.kind in {'i', 'u'}:
                            new_value = list(map(int, new_value))
                        elif value.dtype.kind in {'f'}:
                            new_value = list(map(float, new_value))
                        elif value.dtype.kind in {'c'}:
                            new_value = list(map(complex, new_value))
                        else:
                            pass
                    else:
                        if value.dtype.kind in {'i', 'u'}:
                            new_value = _flatmap(int, new_value)
                        elif value.dtype.kind in {'f'}:
                            new_value = _flatmap(float, new_value)
                        elif value.dtype.kind in {'c'}:
                            new_value = _flatmap(complex, new_value)
                        else:
                            pass
                            # raise TypeError(value.dtype)
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.int16, np.int32, np.int64,
                                    np.uint16, np.uint32, np.uint64)):
                new_value = int(value)
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.float32, np.float64)):
                new_value = float(value)
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.complex64, np.complex128)):
                new_value = complex(value)
                to_convert.append((root, key, new_value))
            elif hasattr(value, '__json__'):
                new_value = value.__json__()
                to_convert.append((root, key, new_value))
            elif normalize_containers:
                if isinstance(value, dict):
                    new_value = _norm_container(value)
                    to_convert.append((root, key, new_value))

    for root, key, new_value in to_convert:
        _convert(dict_, root, key, new_value)

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
        >>> print('parts = {}'.format(ub.repr2(parts, nl=1)))
        >>> # Check expected structure of bad parts
        >>> assert len(parts) == 2
        >>> part = parts[0]
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
    """
    needs_check = True
    is_serializable = None
    if quickcheck:
        try:
            # Might be a more efficient way to do this check. We duplicate a lot of
            # work by doing the check for unserializable data this way.
            json.dumps(data)
        except Exception:
            # If there is unserializable data, find out where it is.
            is_serializable = False
        else:
            is_serializable = True
            needs_check = False

    if needs_check:
        if isinstance(data, list):
            for idx, item in enumerate(data):
                subparts_item = find_json_unserializable(item, quickcheck=False)
                for sub in subparts_item:
                    sub['loc'].appendleft(idx)
                    yield sub
        elif isinstance(data, dict):
            for key, value in data.items():
                subparts_key = find_json_unserializable(key, quickcheck=False)
                for sub in subparts_key:
                    # Special case where a dict key is the error value
                    # Purposely make loc non-hashable so its not confused with
                    # an address. All we can know in this case is that they key
                    # is at this level, there is no concept of where.
                    sub['loc'].appendleft(['.keys', key])
                    yield sub

                subparts_val = find_json_unserializable(value, quickcheck=False)
                for sub in subparts_val:
                    sub['loc'].appendleft(key)
                    yield sub
        else:
            if is_serializable is None:
                try:
                    # Might be a more efficient way to do this check. We duplicate a lot of
                    # work by doing the check for unserializable data this way.
                    json.dumps(data)
                except Exception:
                    is_serializable = False
                else:
                    is_serializable = True

            if is_serializable is False:
                yield {'loc': deque(), 'data': data}
