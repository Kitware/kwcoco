"""
Special non-general json functions
"""
# import ubelt as ub
from packaging.version import parse as Version
import os
import json as pjson
from io import StringIO
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


def _json_dumps(data, indent=None):
    try:
        text = json_w.dumps(data, indent=indent, ensure_ascii=False)
    except Exception:
        if indent is not None:
            if isinstance(indent, str):
                assert indent.count(' ') == len(indent), 'must be all spaces, got {!r}'.format(indent)
                indent = len(indent)
        if indent is None:
            indent = 0
        fp = StringIO()
        json_w.dump(data, fp, indent=indent, ensure_ascii=False)
        fp.seek(0)
        text = fp.read()
    return text


def _json_lines_dumps(key, value, indent):
    value_lines = [_json_dumps(v) for v in value]
    if value_lines:
        value_body = (',\n' + indent).join(value_lines)
        value_repr = '[\n' + indent + value_body + '\n]'
    else:
        value_repr = '[]'
    item_repr = '{}: {}'.format(_json_dumps(key), value_repr)
    return item_repr


def _special_kwcoco_pretty_dumps_orig(data, indent=None):
    """
    The old way of doing "pretty" dumping, except it isn't that pretty.

    See Also:
        Tried to do a "principled" lark version, but this this way is faster
        ~/code/kwcoco/dev/devcheck/json_dumps_experiments.py

    Ignore:
        import kwcoco
        dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
        dset.clear_annotations()
        data = dset.dataset
        print(_special_kwcoco_pretty_dumps_orig(data, indent='    '))
    """
    SPEC_KEYS = [
        'info',
        'licenses',
        'categories',
        'keypoint_categories',  # support only partially implemented
        'videos',
        'images',
        'annotations',
    ]
    if indent is None:
        indent = ''
    if isinstance(indent, int):
        indent = ' ' * indent
    dict_lines = []
    main_keys = SPEC_KEYS
    other_keys = sorted(set(data.keys()) - set(main_keys))
    # TODO: optimize efficiency
    # TODO: general "flexible json" package that can read to/from
    # zipfiles, support ujson or pjson backends, has pretty newline
    # properties. This would abstrat much of the logic away from this
    # module and be generally useful when dealing with other larger
    # json files.
    for key in main_keys:
        if key not in data:
            continue
        # We know each main entry is a list, so make it such that
        # Each entry gets its own line
        value = data[key]
        if key == 'images':
            # Except image, where every auxiliary item also gets a line
            value_lines = []
            for img in value:
                asset_key = None
                if 'auxiliary' in img:
                    asset_key = 'auxiliary'
                elif 'assets' in img:
                    asset_key = 'assets'
                if asset_key is not None:
                    topimg = img.copy()
                    aux_items = topimg.pop(asset_key)
                    aux_items_repr = _json_lines_dumps(asset_key, aux_items, indent + indent)
                    topimg_repr = _json_dumps(topimg)
                    if len(topimg) == 0:
                        v2 = '{' + aux_items_repr + '}'
                    else:
                        v2 = topimg_repr[:-1] + ', ' + aux_items_repr + '}'
                else:
                    v2 = _json_dumps(img)
                value_lines.append(v2)
        else:
            value_lines = [_json_dumps(v) for v in value]
        if value_lines:
            value_body = (',\n' + indent).join(value_lines)
            value_repr = '[\n' + indent + value_body + '\n]'
        else:
            value_repr = '[]'
        item_repr = '{}: {}'.format(_json_dumps(key), value_repr)
        dict_lines.append(item_repr)

    for key in other_keys:
        # Dont assume anything about other data
        value = data.get(key, [])
        value_repr = _json_dumps(value)
        item_repr = '{}: {}'.format(_json_dumps(key), value_repr)
        dict_lines.append(item_repr)
    text = ''.join(['{\n', ',\n'.join(dict_lines), '\n}'])
    return text
