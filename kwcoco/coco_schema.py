"""
The place where the formal KWCOCO schema is defined.

CommandLine:
    python -m kwcoco.coco_schema
    xdoctest -m kwcoco.coco_schema __doc__

TODO:
    - [ ] Perhaps use `voluptuous <https://pypi.org/project/voluptuous/>`_ instead?

Example:
    >>> import kwcoco
    >>> from kwcoco.coco_schema import COCO_SCHEMA
    >>> import jsonschema
    >>> dset = kwcoco.CocoDataset.demo('shapes1')
    >>> # print('dset.dataset = {}'.format(ub.urepr(dset.dataset, nl=2)))
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
    >>>     print('schema_ex.instance = {}'.format(ub.urepr(schema_ex.instance, nl=-1)))
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
    >>> img['name'] = 'asset-only images must have a name'
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
    PROPERTIES={
        'xy': TUPLE(NUMBER, NUMBER, description='<x1, y1> in pixels'),
        'visible': INTEGER(description='choice(0, 1, 2)'),
        'keypoint_category_id': INTEGER,
        'keypoint_category': STRING(description='only to be used as a hint')
    },
    title='KWCOCO_KEYPOINT',
    descripton='A new-style point',
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
    description='A new-style polygon format that supports holes',
)


ORIG_COCO_KEYPOINTS = ARRAY(
    INTEGER,
    description='An old-style set of keypoints (x1,y1,v1,...,xk,yk,vk)',
    title='MSCOCO_KEYPOINTS'
)
KWCOCO_KEYPOINTS = ARRAY(KWCOCO_KEYPOINT)
KEYPOINTS = ANYOF(ORIG_COCO_KEYPOINTS, KWCOCO_KEYPOINTS)


MSCOCO_POLYGON = ARRAY(
    TYPE=NUMBER,
    description='an old-style polygon [x1,y1,v1,...,xk,yk,vk]',
    title='MSCOCO_POLYGON',
)
MSCOCO_MULTIPOLYGON = ARRAY(MSCOCO_POLYGON)

POLYGON = ANYOF(
    KWCOCO_POLYGON,
    ARRAY(KWCOCO_POLYGON),
    MSCOCO_POLYGON,
    MSCOCO_MULTIPOLYGON,
)

RUN_LENGTH_ENCODING = STRING(description='A run-length-encoding mask format read by pycocotools')

BBOX = ARRAY(
    TYPE=NUMBER,
    numItems=4,
    description='[top-left x, top-left-y, width, height] in image-space pixels',
    title='BBOX',
)

### ------------------------


SEGMENTATION = ANYOF(POLYGON, RUN_LENGTH_ENCODING)

# Names cannot contain certain special characters
NAME = STRING(pattern='[^/]+')


CATEGORY = OBJECT({
    'id': INTEGER(description='A unique internal category id'),
    'name': NAME(description='A unique external category name or identifier'),

    'alias': ARRAY(NAME, description='A list of alternate names that should be resolved to this category'),

    'supercategory': ANYOF(NAME(description='A coarser category name'), NULL),
    'parents': ARRAY(NAME, description='Used for multiple inheritance'),

    # Legacy
    'keypoints': deprecated(ARRAY(STRING)),
    'skeleton': deprecated(ARRAY(TUPLE(INTEGER, INTEGER))),
},
    required=['id', 'name'],
    description='High level information about an annotation category',
    title='CATEGORY')

KEYPOINT_CATEGORY = OBJECT(
    PROPERTIES={
        'name': NAME(description='The name of the keypoint category'),
        'id': INTEGER,
        'supercategory': ANYOF(NAME, NULL),
        # TODO: should have this name changed to reflect the fact it is horizontal.
        # TODO: should add a variant of this for vertical or other transforms.
        'reflection_id': ANYOF(INTEGER, NULL)(
            description='The keypoint category this should change to if the image is horizontally flipped'),
    },
    required=['id', 'name'],
    description='High level information about an annotation category',
    title='KEYPOINT_CATEGORY',
)

# Extension
VIDEO = OBJECT(
    PROPERTIES={
        'id': INTEGER(description='An internal video identifier'),
        'name': NAME(description='A unique name for this video'),
        'caption': STRING(description='A video level text caption'),
        'resolution': (NUMBER | STRING | NULL)(description='a unit representing the size of a pixel in video space'),
        },
    required=['id', 'name'],
    description='High level information about a group of temporally ordered images',
    title='VIDEO',
)

CHANNELS = STRING(
    pattern='[^/]*',  # a simple check, full pattern is a context free grammar
    description=(
        'A human readable channel name. '
        'Must be compatible with kwcoco.ChannelSpec'
    ),
    title='CHANNEL_SPEC',
)


ASSET = OBJECT(
    PROPERTIES={
        'file_name': PATH,
        'channels': CHANNELS,
        'id': INTEGER(description='The id of the asset (option for now, but will be required in the future when assets are moved to their own table)'),
        'image_id': INTEGER(description='The image id this asset is associated with (option for now, but will be required in the future)'),
        'width': INTEGER(description='The width in asset-space pixels'),
        'height': INTEGER(description='The height in asset-space pixels'),
    },
    required=['file_name'],
    description='Information about a single file belonging to an image',
    title='ASSET',
)

IMAGE = OBJECT(OrderedDict((
    ('id', INTEGER(description='a unique internal image identifier')),
    ('file_name', PATH(description=ub.paragraph(
        '''
        A relative or absolute path to the main image file. If this file_name
        is unspecified, then a name and auxiliary items or assets must be
        specified. Likewise this should be null if assets are used.
        ''')) | NULL),

    ('name', NAME(
        description=ub.paragraph(
            '''
            A unique name for the image.
            If unspecified the file_name should be used as the default value
            for the name property. Required if assets / auxiliary are
            specified.
            ''')) | NULL),

    ('width', INTEGER(description='The width of the image in image space pixels')),
    ('height', INTEGER(description='The height of the image in image space pixels')),

    # Extension
    ('video_id', INTEGER(description='The video this image belongs to')),

    ('timestamp', STRING(description='An ISO-8601 timestamp') | NUMBER(description='A UNIX timestamp')),

    ('frame_index', INTEGER(description='Used to temporally order the images in a video')),

    ('channels', CHANNELS | NULL),

    ('resolution', (NUMBER | STRING | NULL)(description='a unit representing the size of a pixel in image space')),

    ('auxiliary', ARRAY(TYPE=ASSET, description='This will be deprecated for assets in the future')),

    ('assets', ARRAY(TYPE=ASSET, description='A list of assets belonging to this image, used when image channels are split across multiple files')),

)),
    # required=['id', 'file_name']
    anyOf=[
        {'required': ['id', 'file_name']},
        {'required': ['id', 'name', 'auxiliary']},
        {'required': ['id', 'name', 'assets']},
    ],
    description=(
        'High level information about a image file or a collection of '
        'image files corresponding to a single point in (or small interval of) '
        'time'
    ),
    title='IMAGE',
)

TRACK = OBJECT(OrderedDict((
    ('id', INTEGER(description='A unique internal id for this track')),
    ('name', NAME(description='A unique external name or identifier')),
)))

ANNOTATION = OBJECT(OrderedDict((
    ('id', INTEGER(description='A unique internal id for this annotation')),
    ('image_id', INTEGER(description='The image id this annotation belongs to')),

    ('bbox', BBOX),

    ('category_id', INTEGER(description='The category id of this annotation')),
    ('track_id', ANYOF(INTEGER, STRING, UUID)(
        description='An identifier used to group annotations belonging to the same object over multiple frames in a video')),

    ('segmentation', SEGMENTATION(description='A polygon or mask specifying the pixels in this annotation in image-space')),
    ('keypoints', KEYPOINTS(description='A set of categorized points belonging to this annotation in image space')),

    ('prob', ARRAY(NUMBER, description=ub.paragraph(
        '''
        This needs to be in the same order as categories.
        The probability order currently needs to be known a-priori,
        typically in *order* of the classes, but its hard to always
        keep that consistent.
        This SPEC is subject to change in the future.
        '''))),

    ('score', NUMBER(description='Typically assigned to predicted annotations')),
    ('weight', NUMBER(description='Typically given to truth annotations to indicate quality.')),

    ('iscrowd', ANYOF(INTEGER, BOOLEAN)(description=(
        'A legacy mscoco field used to indicate if an annotation contains multiple objects'))),
    ('caption', STRING(description='An annotation-level text caption')),

    ('role', (STRING | NULL)(
        description=ub.paragraph(
            '''
            A optional application specific key used to differentiate between
            annotations used for different purposes: e.g. truth / prediction /
            confusion.
            '''))),
)),
    required=['id', 'image_id'],
    description='Metadata about some semantic attribute of an image.',
    title='ANNOTATION',
)


COCO_SCHEMA = OBJECT(
    PROPERTIES=ub.odict([
        ('info', ANY),
        ('licenses', ANY),

        ('categories', ARRAY(CATEGORY)),

        ('keypoint_categories', ARRAY(KEYPOINT_CATEGORY)),

        ('videos', ARRAY(VIDEO)),

        ('tracks', ARRAY(TRACK)),

        ('images', ARRAY(IMAGE)),

        ('annotations', ARRAY(ANNOTATION)),
    ]),
    required=[],
    description='The formal kwcoco schema',
    title='KWCOCO_SCHEMA',
)


if ub.argflag('--debug') or ub.argflag('--validate'):
    COCO_SCHEMA.validate()


if __name__ == '__main__':
    """
    CommandLine:
        KWCOCO_MODPATH=$(xdev modpath kwcoco)
        python $KWCOCO_MODPATH/coco_schema.py --validate
        python $KWCOCO_MODPATH/coco_schema.py > ~/code/kwcoco/kwcoco/coco_schema.json
        jq .properties.images $KWCOCO_MODPATH/coco_schema.json
        jq .properties.categories $KWCOCO_MODPATH/coco_schema.json
        jq . $KWCOCO_MODPATH/coco_schema.json
    """
    # import json
    print(ub.urepr(COCO_SCHEMA, nl=-1, trailsep=False, sort=False).replace("'", '"'))
    # print(json.dumps(COCO_SCHEMA, indent='    '))
    # print('COCO_SCHEMA = {}'.format(ub.urepr(COCO_SCHEMA, nl=-1)))
