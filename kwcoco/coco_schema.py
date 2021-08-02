"""
CommandLine:
    python -m kwcoco.coco_schema
    xdoctest -m kwcoco.coco_schema __doc__

Example:
    >>> import kwcoco
    >>> from kwcoco.coco_schema import COCO_SCHEMA
    >>> import jsonschema
    >>> dset = kwcoco.CocoDataset.demo('shapes1')
    >>> # print('dset.dataset = {}'.format(ub.repr2(dset.dataset, nl=2)))
    >>> COCO_SCHEMA.validate(dset.dataset)

    >>> try:
    >>>     jsonschema.validate(dset.dataset, schema=COCO_SCHEMA)
    >>> except jsonschema.exceptions.ValidationError as ex:
    >>>     vali_ex = ex
    >>>     print('ex = {!r}'.format(ex))
    >>>     raise
    >>> except jsonschema.exceptions.SchemaError as ex:
    >>>     print('ex = {!r}'.format(ex))
    >>>     schema_ex = ex
    >>>     print('schema_ex.instance = {}'.format(ub.repr2(schema_ex.instance, nl=-1)))
    >>>     raise

    >>> # Test the multispectral image defintino
    >>> import copy
    >>> dataset = dset.copy().dataset
    >>> img = dataset['images'][0]
    >>> img.pop('file_name')
    >>> import pytest
    >>> with pytest.raises(jsonschema.ValidationError):
    >>>     COCO_SCHEMA.validate(dataset)
    >>> import pytest
    >>> img['auxiliary'] = [{'file_name': 'foobar'}]
    >>> with pytest.raises(jsonschema.ValidationError):
    >>>     COCO_SCHEMA.validate(dataset)
    >>> img['name'] = 'aux-only images must have a name'
    >>> COCO_SCHEMA.validate(dataset)
"""

from kwcoco.util.jsonschema_elements import SchemaElements
from collections import OrderedDict
import ubelt as ub


def deprecated(*args):
    return ANY(description='deprecated')


def TUPLE(*args, **kw):
    if args and ub.allsame(args):
        return ARRAY(TYPE=ub.peek(args), numItems=len(args), **kw)
    else:
        return ARRAY(TYPE=ANY, numItems=len(args), **kw)

elem = SchemaElements()
ALLOF = elem.ALLOF
ANY = elem.ANY
ANYOF = elem.ANYOF
ARRAY = elem.ARRAY
BOOLEAN = elem.BOOLEAN
INTEGER = elem.INTEGER
NOT = elem.NOT
NULL = elem.NULL
NUMBER = elem.NUMBER
OBJECT = elem.OBJECT
ONEOF = elem.ONEOF
STRING = elem.STRING


UUID = STRING
PATH = STRING

KWCOCO_KEYPOINT = OBJECT(
    title='KWCOCO_KEYPOINT',
    PROPERTIES={
        'xy': TUPLE(NUMBER, NUMBER, description='<x1, y1> in pixels'),
        'visible': INTEGER(description='choice(0, 1, 2)'),
        'keypoint_category_id': INTEGER,
        'keypoint_category': STRING(description='only to be used as a hint')}
)

KWCOCO_POLYGON = OBJECT(
    PROPERTIES={
        'exterior': ARRAY(
            ARRAY(NUMBER, numItems=2),
            description='counter-clockwise xy exterior points'),
        'interiors': ARRAY(
            ARRAY(
                ARRAY(NUMBER, numItems=2),
                description='clockwise xy hole'),
        )
    },
    title='KWCOCO_POLYGON',
    description='a simply polygon format that supports holes',
)


ORIG_COCO_KEYPOINTS = ARRAY(
    INTEGER, description='old format (x1,y1,v1,...,xk,yk,vk)', title='MSCOCO_KEYPOINTS')
KWCOCO_KEYPOINTS = ARRAY(KWCOCO_KEYPOINT)
KEYPOINTS = ANYOF(ORIG_COCO_KEYPOINTS, KWCOCO_KEYPOINTS)


ORIG_COCO_POLYGON = ARRAY(
    TYPE=ARRAY(NUMBER),
    title='ORIG_COCO_POLYGON',
    description='[x1,y1,v1,...,xk,yk,vk]',
)

POLYGON = ANYOF(
    KWCOCO_POLYGON,
    ARRAY(KWCOCO_POLYGON),
    ORIG_COCO_POLYGON
)

RUN_LENGTH_ENCODING = STRING(description='format read by pycocotools')

BBOX = ARRAY(
    TYPE=NUMBER,
    title='bbox',
    numItems=4,
    description='[top-left x, top-left-y, width, height] in pixels'
)

### ------------------------


SEGMENTATION = ANYOF(POLYGON, RUN_LENGTH_ENCODING)


CATEGORY = OBJECT({
    'id': INTEGER(description='unique internal id'),
    'name': STRING(description='unique external name or identifier'),

    'alias': ARRAY(STRING, description='list of alter egos'),

    'supercategory': ANYOF(STRING(description='coarser category name'), NULL),
    'parents': ARRAY(STRING, description='used for multiple inheritence'),

    # Legacy
    'keypoints': deprecated(ARRAY(STRING)),
    'skeleton': deprecated(ARRAY(TUPLE(INTEGER, INTEGER))),
},
    required=['id', 'name'],
    title='CATEGORY')

KEYPOINT_CATEGORY = OBJECT({
    'name': STRING,
    'id': INTEGER,
    'supercategory': ANYOF(STRING, NULL),
    'reflection_id': ANYOF(INTEGER, NULL),
}, required=['id', 'name'], title='KEYPOINT_CATEGORY')

# Extension
VIDEO = OBJECT(
    PROPERTIES={
        'id': INTEGER,
        'name': STRING,
        'caption': STRING,
        },
    required=['id', 'name'],
    title='VIDEO'
)

CHANNELS = STRING(title='CHANNEL_SPEC', description='experimental. todo: refine')

IMAGE = OBJECT(OrderedDict((
    ('id', INTEGER),
    ('file_name', PATH(description=ub.paragraph(
        '''
        A relative or absolute path to the main image file. If this file_name
        is unspecified, then a name and auxiliary file paths must be specified.
        This should only be unspecified for multispectral observations that
        dont have a clear default file.
        ''')) | NULL),

    ('name', STRING(description=ub.paragraph(
        '''
        Unique name for the image.
        If unspecified the file_name should be used as the default value
        for the name property.
        ''')) | NULL),

    ('width', INTEGER),
    ('height', INTEGER),

    # Extension
    ('video_id', INTEGER),

    # FIXME: timestamp could be a float, integer, or string in an isoformat
    ('timestamp', NUMBER(description='todo describe format. flicks?')),

    ('frame_index', INTEGER),

    ('channels', CHANNELS | NULL),

    # TODO: optional world localization information
    # TODO: camera information?

    ('auxiliary', ARRAY(
        TYPE=OBJECT({
            'file_name': PATH,
            'channels': CHANNELS,
            'width': INTEGER,
            'height': INTEGER,
        }, title='aux', required=['file_name'])
    )),
)), title='IMAGE',
    # required=['id', 'file_name']
    anyOf=[
        {'required': ['id', 'file_name']},
        {'required': ['id', 'name', 'auxiliary']},
    ],
)

ANNOTATION = OBJECT(OrderedDict((
    ('id', INTEGER),
    ('image_id', INTEGER),

    ('bbox', BBOX),

    ('category_id', INTEGER),
    ('track_id', ANYOF(INTEGER, STRING, UUID)),

    ('segmentation', SEGMENTATION),
    ('keypoints', KEYPOINTS),

    ('prob', ARRAY(NUMBER, description=ub.paragraph(
        '''
        This needs to be in the same order as categories.
        probability order currently needs to be known a-priori,
        typically in *order* of the classes, but its hard to always
        keep that consistent.
        '''))),

    ('score', NUMBER(description='Typically assigned to predicted annotations')),
    ('weight', NUMBER(description='Typically given to truth annotations to indicate quality.')),

    ('iscrowd', ANYOF(INTEGER, BOOLEAN)),  # legacy
    ('caption', STRING),
)),
    required=['id', 'image_id'],
    title='ANNOTATION',
)


COCO_SCHEMA = OBJECT(
    title='KWCOCO_SCHEMA',
    required=[],
    PROPERTIES=ub.odict([
        ('info', ANY),
        ('licenses', ANY),

        ('categories', ARRAY(CATEGORY)),

        ('keypoint_categories', ARRAY(KEYPOINT_CATEGORY)),

        ('videos', ARRAY(VIDEO)),

        ('images', ARRAY(IMAGE)),

        ('annotations', ARRAY(ANNOTATION)),
    ])
)


if ub.argflag('--debug'):
    COCO_SCHEMA.validate()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/coco_schema.py
        python ~/code/kwcoco/kwcoco/coco_schema.py > ~/code/kwcoco/kwcoco/coco_schema.json
        jq .properties.images ~/code/kwcoco/kwcoco/coco_schema.json
    """
    # import json
    print(ub.repr2(COCO_SCHEMA, nl=-1, trailsep=False, sort=False).replace("'", '"'))
    # print(json.dumps(COCO_SCHEMA, indent='    '))
    # print('COCO_SCHEMA = {}'.format(ub.repr2(COCO_SCHEMA, nl=-1)))
