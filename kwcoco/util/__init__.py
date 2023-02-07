"""
mkinit ~/code/kwcoco/kwcoco/util/__init__.py -w
mkinit ~/code/kwcoco/kwcoco/util/__init__.py --lazy
"""
import sys

__protected__ = [
    'delayed_ops',
    'lazy_frame_backends',
    'util_monkey',
    'util_futures',
]

if sys.version_info[0:2] <= (3, 6):
    from kwcoco.util import delayed_ops
    from kwcoco.util import dict_like
    from kwcoco.util import jsonschema_elements
    from kwcoco.util import lazy_frame_backends
    from kwcoco.util import util_archive
    from kwcoco.util import util_futures
    from kwcoco.util import util_json
    from kwcoco.util import util_monkey
    from kwcoco.util import util_reroot
    from kwcoco.util import util_sklearn
    from kwcoco.util import util_truncate

    from kwcoco.util.dict_like import (DictLike,)
    from kwcoco.util.jsonschema_elements import (ALLOF, ANY, ANYOF, ARRAY, BOOLEAN,
                                                 ContainerElements, Element,
                                                 INTEGER, NOT, NULL, NUMBER,
                                                 OBJECT, ONEOF, QuantifierElements,
                                                 STRING, ScalarElements,
                                                 SchemaElements, elem,)
    from kwcoco.util.util_archive import (Archive, unarchive_file,)
    from kwcoco.util.util_json import (IndexableWalker, ensure_json_serializable,
                                       find_json_unserializable,
                                       indexable_allclose,)
    from kwcoco.util.util_reroot import (resolve_directory_symlinks,
                                         resolve_relative_to,
                                         special_reroot_single,)
    from kwcoco.util.util_sklearn import (StratifiedGroupKFold,)
    from kwcoco.util.util_truncate import (smart_truncate,)


def lazy_import(module_name, submodules, submod_attrs):
    import importlib
    import os
    name_to_submod = {
        func: mod for mod, funcs in submod_attrs.items()
        for func in funcs
    }

    def __getattr__(name):
        if name in submodules:
            attr = importlib.import_module(
                '{module_name}.{name}'.format(
                    module_name=module_name, name=name)
            )
        elif name in name_to_submod:
            submodname = name_to_submod[name]
            module = importlib.import_module(
                '{module_name}.{submodname}'.format(
                    module_name=module_name, submodname=submodname)
            )
            attr = getattr(module, name)
        else:
            raise AttributeError(
                'No {module_name} attribute {name}'.format(
                    module_name=module_name, name=name))
        globals()[name] = attr
        return attr

    if os.environ.get('EAGER_IMPORT', ''):
        for name in submodules:
            __getattr__(name)

        for attrs in submod_attrs.values():
            for attr in attrs:
                __getattr__(attr)
    return __getattr__


if sys.version_info[0:2] >= (3, 7):
    __getattr__ = lazy_import(
        __name__,
        submodules={
            'delayed_ops',
            'dict_like',
            'jsonschema_elements',
            'lazy_frame_backends',
            'util_archive',
            'util_futures',
            'util_json',
            'util_monkey',
            'util_reroot',
            'util_sklearn',
            'util_truncate',
        },
        submod_attrs={
            'dict_like': [
                'DictLike',
            ],
            'jsonschema_elements': [
                'ALLOF',
                'ANY',
                'ANYOF',
                'ARRAY',
                'BOOLEAN',
                'ContainerElements',
                'Element',
                'INTEGER',
                'NOT',
                'NULL',
                'NUMBER',
                'OBJECT',
                'ONEOF',
                'QuantifierElements',
                'STRING',
                'ScalarElements',
                'SchemaElements',
                'elem',
            ],
            'util_archive': [
                'Archive',
                'unarchive_file',
            ],
            'util_json': [
                'IndexableWalker',
                'ensure_json_serializable',
                'find_json_unserializable',
                'indexable_allclose',
            ],
            'util_reroot': [
                'resolve_directory_symlinks',
                'resolve_relative_to',
                'special_reroot_single',
            ],
            'util_sklearn': [
                'StratifiedGroupKFold',
            ],
            'util_truncate': [
                'smart_truncate',
            ],
        },
    )

    def __dir__():
        return __all__


__all__ = ['ALLOF', 'ANY', 'ANYOF', 'ARRAY', 'Archive', 'BOOLEAN',
           'ContainerElements', 'DictLike', 'Element', 'INTEGER',
           'IndexableWalker', 'NOT', 'NULL', 'NUMBER', 'OBJECT', 'ONEOF',
           'QuantifierElements', 'STRING', 'ScalarElements', 'SchemaElements',
           'StratifiedGroupKFold', 'delayed_ops', 'dict_like', 'elem',
           'ensure_json_serializable', 'find_json_unserializable',
           'indexable_allclose', 'jsonschema_elements', 'lazy_frame_backends',
           'resolve_directory_symlinks', 'resolve_relative_to',
           'smart_truncate', 'special_reroot_single', 'unarchive_file',
           'util_archive', 'util_futures', 'util_json',
           'util_monkey', 'util_reroot', 'util_sklearn', 'util_truncate']
