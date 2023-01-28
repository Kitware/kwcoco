import copy
import numpy as np
import ubelt as ub
import json
from collections import OrderedDict
import pathlib
from packaging.version import parse as Version
import os
import json as pjson
from types import ModuleType
# The ujson library is faster than Python's json, but the API has some
# limitations and requires a minimum version. Currently we only use it to read,
# we have to wait for https://github.com/ultrajson/ultrajson/pull/518 to land
# before we use it to write.
try:
    import ujson
except ImportError:
    ujson = None

KWCOCO_USE_UJSON = bool(os.environ.get('KWCOCO_USE_UJSON'))

if ujson is not None and Version(ujson.__version__) >= Version('5.2.0') and KWCOCO_USE_UJSON:
    json_r: ModuleType = ujson
    json_w: ModuleType = pjson
else:
    json_r: ModuleType = pjson
    json_w: ModuleType = pjson

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
        walker = ub.IndexableWalker(data)
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


def _special_kwcoco_pretty_dumps(data):
    """
    json dumps, but nicer using lark

    Ignore:
        import kwcoco
        dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
        data = dset.dataset
    """
    print(json_w.dumps(data, indent='    '))
    print('data = {}'.format(ub.urepr(data, nl=4)))
    # from lark import Lark, Transformer, v_args
    from lark import Lark

    reformat_grammar = r"""
        ?start: outer_tables

        ?value: object
              | array
              | string
              | SIGNED_NUMBER      -> number
              | "true"             -> true
              | "false"            -> false
              | "null"             -> null
              | "NaN"              -> nan
              | "Inf"              -> inf

        array  : "[" [value ("," value)*] "]"
        object : "{" [pair ("," pair)*] "}"
        pair   : string ":" value

        table_row : "{" [pair ("," pair)*] "}"
        outer_table  : "[" [table_row ("," table_row)*] "]"
        outer_pair   : string ":" outer_table
        outer_tables : "{" [outer_pair ("," outer_pair)*] "}"

        string : ESCAPED_STRING

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS

        %ignore WS
    """

    # https://github.com/lark-parser/lark/issues/12

    ### Create the JSON parser with Lark, using the LALR algorithm
    json_parser = Lark(reformat_grammar, parser='lalr',
                       # Using the basic lexer isn't required, and isn't usually recommended.
                       # But, it's good enough for JSON, and it's slightly faster.
                       # lexer='standard',
                       lexer='contextual',
                       propagate_positions=True,
                       maybe_placeholders=True,
                       # # Disabling propagate_positions and placeholders slightly improves speed
                       # propagate_positions=False,
                       # maybe_placeholders=False,
                       )

    raw_text = json.dumps(data)
    parsed = json_parser.parse(raw_text)
    # outer_table = parsed.children[2].children[1]

    table_splits = []
    table_indents = []
    table_indents.append('')
    assert parsed.data == 'outer_tables'
    table_splits.append(parsed.meta.start_pos + 1)
    for child in parsed.children:
        assert child.data == 'outer_pair'
        table = child.children[1]
        table_name = child.children[0]
        assert table.data == 'outer_table'

        table_splits.append(table.meta.start_pos + 1)
        table_indents.append('')

        if table_name.children[0] == '"images"':
            # Special case for image table
            for table_row in table.children:
                if table_row is not None:
                    assert table_row.data == 'table_row'
                    for img_field in table_row.children:
                        img_key = img_field.children[0]
                        if img_key.children[0] == '"auxiliary"':
                            img_val = img_field.children[1]
                            # table_splits.append(img_field.meta.start_pos)
                            table_splits.append(img_val.meta.start_pos + 1)
                            table_indents.append('')
                            for aux_token in img_val.children:
                                table_indents.append('    ')
                                table_splits.append(aux_token.meta.end_pos + 1)
                            # table_splits.append(pair_item.meta.end_pos + 1)
                    # table_splits.append(table_row.meta.start_pos)
                    table_splits.append(table_row.meta.end_pos + 1)
                    table_indents.append('')
        else:
            for table_row in table.children:
                if table_row is not None:
                    assert table_row.data == 'table_row'
                    # table_splits.append(table_row.meta.start_pos)
                    table_splits.append(table_row.meta.end_pos + 1)
                    table_indents.append('    ')

        table_splits.append(table.meta.end_pos + 1)
        table_indents.append('')
    table_splits.append(parsed.meta.end_pos + 1)
    table_indents.append('')
    table_indents.append('')

    parts = [
        raw_text[a:b]
        for a, b in ub.iter_window([0] + table_splits + [-1], 2)]

    x = list(ub.flatten(zip(['\n' + a for a in table_indents], parts)))
    print(''.join(x))
