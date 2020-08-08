"""
CommandLine:
    xdoctest -m kwcoco.coco_schema __doc__

Example:
    >>> import kwcoco
    >>> from kwcoco.coco_schema import COCO_SCHEMA
    >>> COCO_SCHEMA
    >>> dset = kwcoco.CocoDataset.demo('shapes1')
    >>> import jsonschema
    >>> print('dset.dataset = {}'.format(ub.repr2(dset.dataset, nl=2)))
    >>> jsonschema.validate(dset.dataset, schema=COCO_SCHEMA)
"""

from kwcoco.jsonschema_elements import SchemaElements
from collections import OrderedDict
import ubelt as ub


def deprecated(*args):
    return ANY

# class deprecated:
#     def __init__(self, item):
#         self.item = item

#     def __str__(self):
#         return 'DEPRECATED'

#     def __repr__(self):
#         return 'DEPRECATED'

#     def __call__(self, **kw):
#         return deprecated(self.item(**kw))


# class optional:
#     def __init__(self, item):
#         self.item = item

#     def __str__(self):
#         return 'OPTIONAL'

#     def __repr__(self):
#         return 'OPTIONAL'

#     def __call__(self, **kw):
#         return optional(self.item(**kw))


optional = ub.identity


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


# def optional(x):
#     return ANYOF(x, ANY)


UUID = STRING
PATH = STRING

KWCOCO_KEYPOINT = OBJECT(
    title='KWCOCO_KEYPOINT',
    PROPERTIES={
        'xy': TUPLE(NUMBER, NUMBER, description='<x1, y1>'),
        'visible': INTEGER(description='choice(0, 1, 2)'),
        'keypoint_category_id': INTEGER,
        'keypoint_category': optional(STRING)(description='only to be used as a hint')}
)

KWCOCO_POLYGON = OBJECT(
    PROPERTIES={
        'exterior': ARRAY(ARRAY(NUMBER, numItems=2), description='ccw xy exterior points'),
        'interiors': ARRAY(
            ARRAY(ARRAY(NUMBER, numItems=2), description='cw xy hole'),
        )
    },
    title='KWCOCO_POLYGON',
    description='a simply polygon format that supports holes',
)


KEYPOINTS_V1 = ARRAY(INTEGER, description='old format (x1,y1,v1,...,xk,yk,vk)')
KEYPOINTS_V2 = ARRAY(KWCOCO_KEYPOINT)
KEYPOINTS = ANYOF(KEYPOINTS_V1, KEYPOINTS_V2)


ORIG_COCO_POLYGON = ARRAY(
    TYPE=ARRAY(NUMBER, numItems=2),
    title='ORIG_COCO_POLYGON',
    description='[x1,y1,v1,...,xk,yk,vk]',
)

POLYGON = ANYOF(
    KWCOCO_POLYGON,
    ORIG_COCO_POLYGON
)

RUN_LENGTH_ENCODING = STRING(description='format read by pycocotools')

BBOX = ARRAY(
    TYPE=NUMBER,
    title='bbox',
    numItems=4,
    description='top-left x, top-left-y, width, height'
)

### ------------------------


SEGMENTATION = ANYOF(POLYGON, RUN_LENGTH_ENCODING)


CATEGORY = OBJECT({
    'id': INTEGER,
    'name': STRING,

    'alias': optional(ARRAY(STRING, description='list of alter egos')),

    'supercategory': optional(STRING(description='coarser category')),
    'parents': optional(ARRAY(STRING)),

    # Legacy
    'keypoints': deprecated(optional(ARRAY(STRING))),
    'skeleton': deprecated(optional(ARRAY(TUPLE(INTEGER, INTEGER)))),
},
    required=['id', 'name'],
    title='CATEGORY')

KEYPOINT_CATEGORY = OBJECT({
    'name': STRING,
    'id': INTEGER,
    'supercategory': STRING,
    'reflection_id': INTEGER,
}, required=['id', 'name'], title='KEYPOINT_CATEGORY')

# Extension
VIDEO = OBJECT({
    'id': INTEGER,
    'name': STRING,
    'caption': STRING,
}, required=['id', 'name'], title='VIDEO')

CHANNELS = STRING(title='CHANNEL_SPEC', description='experimental')

IMAGE = OBJECT(OrderedDict((
    ('id', INTEGER),
    ('file_name', PATH),

    ('width', optional(INTEGER)),
    ('height', optional(INTEGER)),

    # Extension
    ('video_id', optional(INTEGER)),
    ('timestamp', optional(NUMBER)(description='todo describe format. flicks?')),
    ('frame_index', optional(INTEGER)),

    ('channels', optional(CHANNELS)),

    # TODO: optional world localization information
    # TODO: camera information?

    # ('auxillary', ARRAY(
    #     TYPE=OBJECT({
    #         'file_name': PATH,
    #         'channels': CHANNELS,
    #         'width': optional(INTEGER),
    #         'height': optional(INTEGER),
    #     }, title='aux')
    # )),
)), required=['id', 'file_name'])

ANNOTATION = OBJECT(OrderedDict((
    ('id', INTEGER),
    ('image_id', INTEGER),

    ('bbox', BBOX),

    ('category_id', optional(INTEGER)),
    ('track_id', optional(ANYOF(INTEGER, STRING, UUID))),

    ('segmentation', optional(SEGMENTATION)),
    ('keypoints', optional(KEYPOINTS)),

    # this needs to be in the same order as categories
    ('prob', optional(ARRAY(NUMBER))),  # probability order currently needs to be known a-priori, typically in "order" of the classes, but its hard to always keep that consistent.

    ('score', optional(NUMBER)),
    ('weight', optional(NUMBER)),

    ('iscrowd', STRING),  # legacy
    ('caption', STRING),
)),
    required=['id', 'image_id']
)


COCO_SCHEMA = OBJECT(
    title='KWCOCO_SCHEMA',
    required=['images'],
    # propertyNames='.*',
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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/coco_schema.py > out.py
    """
    print('COCO_SCHEMA = {}'.format(ub.repr2(COCO_SCHEMA, nl=-1)))
    # print('COCO_SCHEMA = {}'.format(ub.repr2(COCO_SCHEMA, nl=2)))
    # print('COCO_SCHEMA = {}'.format(ub.repr2(COCO_SCHEMA, nl=2)))
