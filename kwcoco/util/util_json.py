import copy
import numpy as np
import ubelt as ub  # NOQA
import json
from collections.abc import Generator
from collections import OrderedDict


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
        >>> data['foo']['b'] = (1, np.array([1, 2, 3]), {3: np.int32(3), 4: np.float16(1.0)})
        >>> dict_ = data
        >>> print(ub.repr2(data, nl=-1))
        >>> assert list(find_json_unserializable(data))
        >>> result = ensure_json_serializable(data, normalize_containers=True)
        >>> print(ub.repr2(result, nl=-1))
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

    walker = IndexableWalker(dict_)
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
        >>> print('parts = {}'.format(ub.repr2(parts, nl=1)))
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
    """
    needs_check = True
    if quickcheck:
        try:
            # Might be a more efficient way to do this check. We duplicate a lot of
            # work by doing the check for unserializable data this way.
            json.dumps(data)
        except Exception:
            # If there is unserializable data, find out where it is.
            # is_serializable = False
            pass
        else:
            # is_serializable = True
            needs_check = False

    if needs_check:
        # mode = 'new'
        # if mode == 'new':
        scalar_types = (int, float, str, type(None))
        container_types = (tuple, list, dict)
        serializable_types = scalar_types + container_types
        walker = IndexableWalker(data)
        for prefix, value in walker:
            *root, key = prefix
            if not isinstance(key, scalar_types):
                # Special case where a dict key is the error value
                # Purposely make loc non-hashable so its not confused with
                # an address. All we can know in this case is that they key
                # is at this level, there is no concept of where.
                yield {'loc': root + [['.keys', key]], 'data': key}
            elif not isinstance(value, serializable_types):
                yield {'loc': prefix, 'data': value}


class IndexableWalker(Generator):
    """
    Traverses through a nested tree-liked indexable structure.

    Generates a path and value to each node in the structure. The path is a
    list of indexes which if applied in order will reach the value.

    The ``__setitem__`` method can be used to modify a nested value based on the
    path returned by the generator.

    When generating values, you can use "send" to prevent traversal of a
    particular branch.

    Example:
        >>> # Create nested data
        >>> import numpy as np
        >>> data = ub.ddict(lambda: int)
        >>> data['foo'] = ub.ddict(lambda: int)
        >>> data['bar'] = np.array([1, 2, 3])
        >>> data['foo']['a'] = 1
        >>> data['foo']['b'] = np.array([1, 2, 3])
        >>> data['foo']['c'] = [1, 2, 3]
        >>> data['baz'] = 3
        >>> print('data = {}'.format(ub.repr2(data, nl=True)))
        >>> # We can walk through every node in the nested tree
        >>> walker = IndexableWalker(data)
        >>> for path, value in walker:
        >>>     print('walk path = {}'.format(ub.repr2(path, nl=0)))
        >>>     if path[-1] == 'c':
        >>>         # Use send to prevent traversing this branch
        >>>         got = walker.send(False)
        >>>         # We can modify the value based on the returned path
        >>>         walker[path] = 'changed the value of c'
        >>> print('data = {}'.format(ub.repr2(data, nl=True)))
        >>> assert data['foo']['c'] == 'changed the value of c'

    Example:
        >>> # Test sending false for every data item
        >>> import numpy as np
        >>> data = {1: 1}
        >>> walker = IndexableWalker(data)
        >>> for path, value in walker:
        >>>     print('walk path = {}'.format(ub.repr2(path, nl=0)))
        >>>     walker.send(False)
        >>> data = {}
        >>> walker = IndexableWalker(data)
        >>> for path, value in walker:
        >>>     walker.send(False)
    """

    def __init__(self, data, dict_cls=(dict,), list_cls=(list, tuple)):
        self.data = data
        self.dict_cls = dict_cls
        self.list_cls = list_cls
        self.indexable_cls = self.dict_cls + self.list_cls

        self._walk_gen = None

    def __iter__(self):
        """
        Iterates through the indexable ``self.data``

        Can send a False flag to prevent a branch from being traversed

        Yields:
            Tuple[List, Any] :
                path (List): list of index operations to arrive at the value
                value (object): the value at the path
        """
        return self

    def __next__(self):
        """ returns next item from this generator """
        if self._walk_gen is None:
            self._walk_gen = self._walk(self.data, prefix=[])
        return next(self._walk_gen)

    def send(self, arg):
        """
        send(arg) -> send 'arg' into generator,
        return next yielded value or raise StopIteration.
        """
        # Note: this will error if called before __next__
        self._walk_gen.send(arg)

    def throw(self, type=None, value=None, traceback=None):
        """
        throw(typ[,val[,tb]]) -> raise exception in generator,
        return next yielded value or raise StopIteration.
        """
        raise StopIteration

    def __setitem__(self, path, value):
        """
        Set nested value by path

        Args:
            path (List): list of indexes into the nested structure
            value (object): new value
        """
        d = self.data
        *prefix, key = path
        for k in prefix:
            d = d[k]
        d[key] = value

    def __getitem__(self, path):
        """
        Get nested value by path

        Args:
            path (List): list of indexes into the nested structure

        Returns:
            value
        """
        d = self.data
        *prefix, key = path
        for k in prefix:
            d = d[k]
        return d[key]

    def __delitem__(self, path):
        """
        Remove nested value by path

        Note:
            It can be dangerous to use this while iterating (because we may try
            to descend into a deleted location) or on leaf items that are
            list-like (because the indexes of all subsequent items will be
            modified).

        Args:
            path (List): list of indexes into the nested structure.
                The item at the last index will be removed.
        """
        d = self.data
        *prefix, key = path
        for k in prefix:
            d = d[k]
        del d[key]

    def _walk(self, data, prefix=[]):
        """
        Defines the underlying generator used by IndexableWalker

        Yields:
            Tuple[List, object] | None:
                path (List) - a "path" through the nested data structure
                value (object) - the value indexed by that "path".

            Can also yield None in the case that `send` is called on the
            generator.
        """
        stack = [(data, prefix)]
        while stack:
            _data, _prefix = stack.pop()
            # Create an items iterable of depending on the indexable data type
            if isinstance(_data, self.list_cls):
                items = enumerate(_data)
            elif isinstance(_data, self.dict_cls):
                items = _data.items()
            else:
                raise TypeError(type(_data))

            for key, value in items:
                # Yield the full path to this position and its value
                path = _prefix + [key]
                message = yield path, value
                # If the value at this path is also indexable, then continue
                # the traversal, unless the False message was explicitly sent
                # by the caller.
                if message is False:
                    # Because the `send` method will return the next value,
                    # we yield a dummy value so we don't clobber the next
                    # item in the traversal.
                    yield None
                else:
                    if isinstance(value, self.indexable_cls):
                        stack.append((value, path))


def indexable_allclose(dct1, dct2, return_info=False):
    """
    Walks through two nested data structures and ensures that everything is
    roughly the same.

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
    walker1 = IndexableWalker(dct1)
    walker2 = IndexableWalker(dct2)
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
