import copy

import pytest


def _rich_json_native_dataset():
    """A compact dataset that exercises most typed schema fields."""
    return {
        'categories': [
            {
                'id': 1,
                'name': 'object',
                'alias': ['thing'],
                'supercategory': None,
                'parents': ['root'],
            }
        ],
        'keypoint_categories': [
            {
                'id': 1,
                'name': 'nose',
                'supercategory': '',
                'reflection_id': None,
            }
        ],
        'videos': [
            {
                'id': 1,
                'name': 'video',
                'caption': 'caption',
                'coco_url': 'https://example.com/video.mp4',
                'width': 64,
                'height': 64,
                'resolution': 1.0,
            }
        ],
        'tracks': [{'id': 1, 'name': 'track'}],
        'images': [
            {
                'id': 1,
                'file_name': 'image.jpg',
                'name': 'image',
                'coco_url': 'https://example.com/image.jpg',
                'license': 1,
                'flickr_url': 'https://example.com/image',
                'date_captured': '2020-01-01',
                'width': 64,
                'height': 64,
                'video_id': 1,
                'timestamp': 0.0,
                'frame_index': 0,
                'channels': 'red|green|blue',
                'resolution': '1 meter',
                'auxiliary': [
                    {
                        'file_name': 'auxiliary.tif',
                        'channels': 'nir',
                        'id': 2,
                        'image_id': 1,
                        'width': 64,
                        'height': 64,
                    }
                ],
                'assets': [
                    {
                        'file_name': 'asset.tif',
                        'channels': 'depth',
                        'id': 3,
                        'image_id': 1,
                        'width': 64,
                        'height': 64,
                    }
                ],
            }
        ],
        'annotations': [
            {
                'id': 1,
                'image_id': 1,
                'bbox': [0.0, 0.0, 10.0, 10.0],
                'area': 100.0,
                'category_id': 1,
                'track_id': 1,
                'segmentation': [
                    [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
                ],
                'keypoints': [1.0, 2.0, 2.0],
                'prob': [0.9],
                'score': 0.5,
                'weight': 1.0,
                'iscrowd': 0,
                'ignore': False,
                'caption': 'annotation',
                'role': None,
            }
        ],
    }


def _set_path(data, path, value):
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_schema_backend_auto_resolution(monkeypatch):
    from kwcoco._dataset_components import mixin_schema

    monkeypatch.setattr(mixin_schema, '_fastjsonschema_available', lambda: False)
    assert mixin_schema._resolve_schema_backend('auto') == 'jsonschema'
    assert mixin_schema._resolve_schema_backend('jsonschema') == 'jsonschema'
    with pytest.raises(ModuleNotFoundError):
        mixin_schema._resolve_schema_backend('fastjsonschema')

    monkeypatch.setattr(mixin_schema, '_fastjsonschema_available', lambda: True)
    assert mixin_schema._resolve_schema_backend('auto') == 'fastjsonschema'
    assert mixin_schema._resolve_schema_backend('fastjsonschema') == 'fastjsonschema'

    with pytest.raises(KeyError):
        mixin_schema._resolve_schema_backend('unknown')

    from kwcoco.cli.coco_validate import CocoValidateCLI

    assert CocoValidateCLI()['schema_backend'] == 'auto'


def test_auto_validate_uses_reference_when_fast_is_unavailable(monkeypatch):
    import kwcoco
    from kwcoco._dataset_components import mixin_schema

    monkeypatch.setattr(mixin_schema, '_fastjsonschema_available', lambda: False)

    def _unexpected_fast_call(data):
        raise AssertionError('fast backend should not be called')

    monkeypatch.setattr(
        mixin_schema, '_validate_schema_fastjsonschema', _unexpected_fast_call
    )
    dset = kwcoco.CocoDataset({'images': [{'id': 1, 'file_name': 'image.jpg'}]})
    result = dset.validate(
        schema=True,
        schema_backend='auto',
        unique=False,
        missing=False,
        corrupted=False,
        channels=False,
        img_attrs=False,
        annot_attrs=False,
        verbose=0,
    )
    assert not result['errors']


def test_fastjsonschema_is_packaged_as_optional_dependency():
    from importlib import resources

    optional_text = (
        resources.files('kwcoco.rc.requirements') / 'optional.txt'
    ).read_text()
    assert 'fastjsonschema>=' in optional_text


def test_cached_jsonschema_preserves_reference_error_details():
    import jsonschema
    from kwcoco._dataset_components import mixin_schema
    from kwcoco.coco_schema import COCO_SCHEMA

    invalid_cases = [
        {'images': [{'id': 'bad-id', 'file_name': 'image.jpg'}]},
        {
            'annotations': [
                {'id': 1, 'image_id': 1, 'bbox': [0, 0, 'bad-width', 1]}
            ]
        },
        {'categories': [{'id': 1, 'name': '/'}]},
    ]
    cached_validator = mixin_schema._cached_jsonschema_validator()
    for data in invalid_cases:
        with pytest.raises(jsonschema.ValidationError) as legacy_ctx:
            COCO_SCHEMA.validate(data)
        with pytest.raises(jsonschema.ValidationError) as cached_ctx:
            cached_validator.validate(data)

        legacy = legacy_ctx.value
        cached = cached_ctx.value
        assert str(cached) == str(legacy)
        assert list(cached.absolute_path) == list(legacy.absolute_path)
        assert list(cached.absolute_schema_path) == list(legacy.absolute_schema_path)
        assert cached.validator == legacy.validator
        assert cached.validator_value == legacy.validator_value


def test_fastjsonschema_json_native_compatibility_matrix():
    """Fast and reference validators should agree on JSON-native inputs.

    This intentionally exercises both valid and invalid replacements over most
    typed fields instead of checking only a few known-good examples. The
    reference jsonschema implementation defines compatibility here.
    """
    fastjsonschema = pytest.importorskip('fastjsonschema')
    import jsonschema
    from jsonschema.validators import validator_for
    from kwcoco.coco_schema import COCO_SCHEMA

    validator_cls = validator_for(COCO_SCHEMA)
    validator_cls.check_schema(COCO_SCHEMA)
    reference_validator = validator_cls(COCO_SCHEMA)
    fast_validator = fastjsonschema.compile(COCO_SCHEMA)

    def reference_accepts(data):
        try:
            reference_validator.validate(data)
        except jsonschema.ValidationError:
            return False
        return True

    def fast_accepts(data):
        try:
            fast_validator(data)
        except fastjsonschema.JsonSchemaException:
            return False
        return True

    base = _rich_json_native_dataset()
    assert reference_accepts(base)
    assert fast_accepts(base)

    typed_paths = [
        ('categories', 0, 'id'),
        ('categories', 0, 'name'),
        ('categories', 0, 'alias'),
        ('categories', 0, 'supercategory'),
        ('categories', 0, 'parents'),
        ('keypoint_categories', 0, 'id'),
        ('keypoint_categories', 0, 'name'),
        ('keypoint_categories', 0, 'supercategory'),
        ('keypoint_categories', 0, 'reflection_id'),
        ('videos', 0, 'id'),
        ('videos', 0, 'name'),
        ('videos', 0, 'caption'),
        ('videos', 0, 'coco_url'),
        ('videos', 0, 'width'),
        ('videos', 0, 'height'),
        ('videos', 0, 'resolution'),
        ('tracks', 0, 'id'),
        ('tracks', 0, 'name'),
        ('images', 0, 'id'),
        ('images', 0, 'file_name'),
        ('images', 0, 'name'),
        ('images', 0, 'coco_url'),
        ('images', 0, 'license'),
        ('images', 0, 'flickr_url'),
        ('images', 0, 'date_captured'),
        ('images', 0, 'width'),
        ('images', 0, 'height'),
        ('images', 0, 'video_id'),
        ('images', 0, 'timestamp'),
        ('images', 0, 'frame_index'),
        ('images', 0, 'channels'),
        ('images', 0, 'resolution'),
        ('images', 0, 'auxiliary'),
        ('images', 0, 'assets'),
        ('annotations', 0, 'id'),
        ('annotations', 0, 'image_id'),
        ('annotations', 0, 'bbox'),
        ('annotations', 0, 'area'),
        ('annotations', 0, 'category_id'),
        ('annotations', 0, 'track_id'),
        ('annotations', 0, 'segmentation'),
        ('annotations', 0, 'keypoints'),
        ('annotations', 0, 'prob'),
        ('annotations', 0, 'score'),
        ('annotations', 0, 'weight'),
        ('annotations', 0, 'iscrowd'),
        ('annotations', 0, 'ignore'),
        ('annotations', 0, 'caption'),
        ('annotations', 0, 'role'),
    ]
    json_native_replacements = [
        None,
        True,
        False,
        0,
        1,
        1.0,
        1.5,
        '',
        'x',
        '/',
        'a/b',
        [],
        [1],
        ['x'],
        {},
        {'x': 1},
    ]

    disagreements = []
    for path in typed_paths:
        for value in json_native_replacements:
            candidate = copy.deepcopy(base)
            _set_path(candidate, path, value)
            reference_ok = reference_accepts(candidate)
            fast_ok = fast_accepts(candidate)
            if reference_ok != fast_ok:
                disagreements.append(
                    {
                        'path': path,
                        'value': value,
                        'jsonschema': reference_ok,
                        'fastjsonschema': fast_ok,
                    }
                )

    structural_variants = {
        'kwcoco_polygon_list': [
            {
                'exterior': [[0, 0], [10, 0], [10, 10]],
                'interiors': [],
            }
        ],
        'kwcoco_polygon_dict': {
            'exterior': [[0, 0], [10, 0], [10, 10]],
            'interiors': [],
        },
        'mscoco_flat_polygon': [0, 0, 10, 0, 10, 10],
        'rle_string': 'encoded-rle',
        'empty_segmentation': [],
    }
    for name, segmentation in structural_variants.items():
        candidate = copy.deepcopy(base)
        candidate['annotations'][0]['segmentation'] = segmentation
        reference_ok = reference_accepts(candidate)
        fast_ok = fast_accepts(candidate)
        if reference_ok != fast_ok:
            disagreements.append(
                {
                    'variant': name,
                    'jsonschema': reference_ok,
                    'fastjsonschema': fast_ok,
                }
            )

    keypoint_variants = {
        'kwcoco_keypoints': [
            {
                'xy': [1, 2],
                'visible': 2,
                'keypoint_category_id': 1,
                'keypoint_category': 'nose',
            }
        ],
        'empty_keypoints': [],
    }
    for name, keypoints in keypoint_variants.items():
        candidate = copy.deepcopy(base)
        candidate['annotations'][0]['keypoints'] = keypoints
        reference_ok = reference_accepts(candidate)
        fast_ok = fast_accepts(candidate)
        if reference_ok != fast_ok:
            disagreements.append(
                {
                    'variant': name,
                    'jsonschema': reference_ok,
                    'fastjsonschema': fast_ok,
                }
            )

    assert not disagreements, disagreements


@pytest.mark.parametrize(
    'demo_kwargs',
    [
        {
            'num_videos': 2,
            'num_frames': 3,
            'num_tracks': 3,
            'rng': 0,
        },
        {
            'num_videos': 1,
            'num_frames': 2,
            'num_tracks': 2,
            'multispectral': True,
            'rng': 1,
        },
        {
            'num_videos': 1,
            'num_frames': 2,
            'num_tracks': 2,
            'aux': True,
            'rng': 2,
        },
    ],
)
def test_fastjsonschema_kwcoco_demo_parity(demo_kwargs):
    """Exercise validators on KWCoco's own non-rendering demo generator."""
    pytest.importorskip('fastjsonschema')
    from kwcoco.demo.toydata_video import random_video_dset

    dset = random_video_dset(render=False, verbose=0, **demo_kwargs)
    common = {
        'schema': True,
        'unique': False,
        'missing': False,
        'corrupted': False,
        'channels': False,
        'img_attrs': False,
        'annot_attrs': False,
        'verbose': 0,
    }
    reference_result = dset.validate(schema_backend='jsonschema', **common)
    fast_result = dset.validate(schema_backend='fastjsonschema', **common)
    auto_result = dset.validate(schema_backend='auto', **common)

    assert fast_result['errors'] == reference_result['errors'] == []
    assert auto_result['errors'] == []


def test_auto_backend_preserves_jsonschema_numpy_number_acceptance():
    """Auto mode retries jsonschema when fastjsonschema is more restrictive."""
    pytest.importorskip('fastjsonschema')
    import numpy as np
    import kwcoco

    data = {
        'annotations': [
            {
                'id': 1,
                'image_id': 1,
                # jsonschema accepts numpy integer scalars as JSON numbers,
                # while fastjsonschema deliberately recognizes fewer Python
                # numeric classes.
                'score': np.int64(1),
            }
        ]
    }
    dset = kwcoco.CocoDataset(copy.deepcopy(data), autobuild=False)
    common = {
        'schema': True,
        'unique': False,
        'missing': False,
        'corrupted': False,
        'channels': False,
        'img_attrs': False,
        'annot_attrs': False,
        'verbose': 0,
    }

    reference_result = dset.validate(schema_backend='jsonschema', **common)
    fast_result = dset.validate(schema_backend='fastjsonschema', **common)
    auto_result = dset.validate(schema_backend='auto', **common)

    assert not reference_result['errors']
    assert fast_result['errors']
    assert not auto_result['errors']


def test_auto_backend_preserves_jsonschema_tuple_rejection():
    """fastjsonschema considers tuples arrays; auto retains old semantics."""
    pytest.importorskip('fastjsonschema')
    import kwcoco

    data = {
        'annotations': [
            {
                'id': 1,
                'image_id': 1,
                'bbox': (0, 0, 10, 10),
            }
        ]
    }
    dset = kwcoco.CocoDataset(copy.deepcopy(data), autobuild=False)
    common = {
        'schema': True,
        'unique': False,
        'missing': False,
        'corrupted': False,
        'channels': False,
        'img_attrs': False,
        'annot_attrs': False,
        'verbose': 0,
    }

    reference_result = dset.validate(schema_backend='jsonschema', **common)
    fast_result = dset.validate(schema_backend='fastjsonschema', **common)
    auto_result = dset.validate(schema_backend='auto', **common)

    assert reference_result['errors']
    assert not fast_result['errors']
    assert auto_result['errors']


def test_auto_backend_tuple_in_extension_field_remains_valid():
    """Tuple guard should defer to jsonschema, not globally forbid tuples."""
    pytest.importorskip('fastjsonschema')
    import kwcoco

    data = {
        'images': [
            {
                'id': 1,
                'file_name': 'image.jpg',
                # Unknown extension fields are permitted by the schema.
                'application_extension': ('a', 'b'),
            }
        ]
    }
    dset = kwcoco.CocoDataset(copy.deepcopy(data), autobuild=False)
    common = {
        'schema': True,
        'unique': False,
        'missing': False,
        'corrupted': False,
        'channels': False,
        'img_attrs': False,
        'annot_attrs': False,
        'verbose': 0,
    }

    reference_result = dset.validate(schema_backend='jsonschema', **common)
    auto_result = dset.validate(schema_backend='auto', **common)
    assert not reference_result['errors']
    assert not auto_result['errors']


def test_auto_backend_non_json_python_type_compatibility():
    """Auto should match the historical backend on Python-side edge types."""
    pytest.importorskip('fastjsonschema')
    from decimal import Decimal
    from fractions import Fraction

    import numpy as np
    import kwcoco

    cases = {
        'numpy_integer_number': {
            'annotations': [{'id': 1, 'image_id': 1, 'score': np.int64(1)}]
        },
        'numpy_float_number': {
            'annotations': [{'id': 1, 'image_id': 1, 'score': np.float32(1.5)}]
        },
        'decimal_number': {
            'annotations': [{'id': 1, 'image_id': 1, 'score': Decimal('1.5')}]
        },
        'fraction_number': {
            'annotations': [{'id': 1, 'image_id': 1, 'score': Fraction(1, 2)}]
        },
        'tuple_as_schema_array': {
            'annotations': [{'id': 1, 'image_id': 1, 'bbox': (0, 0, 1, 1)}]
        },
        'tuple_as_top_level_array': {
            'images': ({'id': 1, 'file_name': 'image.jpg'},)
        },
        'tuple_in_extension_field': {
            'images': [
                {
                    'id': 1,
                    'file_name': 'image.jpg',
                    'application_extension': ('a', 'b'),
                }
            ]
        },
        'numpy_array_as_schema_array': {
            'annotations': [
                {'id': 1, 'image_id': 1, 'bbox': np.array([0, 0, 1, 1])}
            ]
        },
    }
    common = {
        'schema': True,
        'unique': False,
        'missing': False,
        'corrupted': False,
        'channels': False,
        'img_attrs': False,
        'annot_attrs': False,
        'verbose': 0,
    }
    disagreements = []
    for name, data in cases.items():
        dset = kwcoco.CocoDataset(copy.deepcopy(data), autobuild=False)
        reference_result = dset.validate(schema_backend='jsonschema', **common)
        auto_result = dset.validate(schema_backend='auto', **common)
        reference_ok = not reference_result['errors']
        auto_ok = not auto_result['errors']
        if reference_ok != auto_ok:
            disagreements.append(
                {
                    'case': name,
                    'jsonschema': reference_ok,
                    'auto': auto_ok,
                }
            )
    assert not disagreements, disagreements


def test_auto_backend_rejects_when_both_backends_reject():
    pytest.importorskip('fastjsonschema')
    import kwcoco

    data = {'images': [{'id': 'bad-id', 'file_name': 'image.jpg'}]}
    dset = kwcoco.CocoDataset(data, autobuild=False)
    common = dict(
        schema=True,
        unique=False,
        missing=False,
        corrupted=False,
        channels=False,
        img_attrs=False,
        annot_attrs=False,
        verbose=0,
    )
    reference_result = dset.validate(schema_backend='jsonschema', **common)
    auto_result = dset.validate(schema_backend='auto', **common)
    assert auto_result['errors'] == reference_result['errors']
