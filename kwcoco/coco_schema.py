from kwcoco.jsonschema_elements import SchemaElements
from collections import OrderedDict
import ubelt as ub


class deprecated:
    def __init__(self, *args):
        self.args = args


class optional:
    def __init__(self, *args):
        self.args = args


def TUPLE(*args):
    if args and ub.allsame(args):
        return ARRAY(TYPE=ub.peek(args), numItems=len(args))
    else:
        return ARRAY(TYPE=ANY, numItems=len(args))

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

KEYPOINT = OBJECT(
    title='KEYPOINT',
    PROPERTIES={
        'xy': TUPLE(INTEGER, INTEGER)(description='<x1, y1>'),
        'visible': INTEGER(description='choice(0, 1, 2)'),
        'keypoint_category_id': INTEGER,
        'KEYPOINT_CATEGORY': optional(STRING)(description='only to be used as a hint')}
)


OLD_POLYGON = ANYOF(ARRAY(NUMBER), ARRAY(ARRAY(NUMBER)))

KEYPOINTS_V1 = ARRAY(INTEGER, description='old format (x1,y1,v1,...,xk,yk,vk')
KEYPOINTS_V2 = ARRAY(KEYPOINT)
KEYPOINTS = ANYOF(KEYPOINTS_V1, KEYPOINTS_V2)

KWCOCO_POLYGON = OBJECT(
    PROPERTIES={
        'exterior': ARRAY(ARRAY(NUMBER, numItems=2), description='ccw xy exterior points'),
        'interiors': ARRAY(
            ARRAY(ARRAY(NUMBER, numItems=2), description='cw xy hole'),
            description='multiple cw xy holes'
        )
    },
    title='KWCOCO_POLYGON',
    description='a simply polygon format that supports holes',
)


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


SEGMENTATION = ANYOF(OLD_POLYGON, POLYGON, RUN_LENGTH_ENCODING)


CATEGORY = OBJECT(
    PROPERTIES={
        'id': INTEGER,
        'name': STRING,
        'supercategory': STRING,
    }
)

ANNOTATION = OBJECT()
#     PROPERTIES=OrderedDict([
#         ('id', INTEGER),
#         ('image_id', INTEGER),

#         ('bbox', BBOX),

#         ('category_id', INTEGER,
#         ('track_id', ANYOF(INTEGER, STRING)),

#         ('segmentation', optional(SEGMENTATION)),
#         ('keypoints', optional(ARRAY(KEYPOINT)),

#         # this needs to be in the same order as categories
#         ('prob', optional[ARRAY[NUMBER]]),  # probability order currently needs to be known a-priori, typically in "order" of the classes, but its hard to always keep that consistent.

#         ('score', optional[NUMBER]),
#         ('weight', optional[NUMBER]),

#         ('iscrowd', STRING),  # legacy
#         ('caption', STRING),

#     ])
# )

IMAGE = OBJECT(
    PROPERTIES={
        'id': INTEGER,
        'file_name': STRING,
    }
)

CATEGORY = OBJECT({
    'id': INTEGER,
    'name': STRING,

    # list of alter egos
    'alias': optional(ARRAY(STRING)),

    # coarser category
    'supercategory': optional(STRING),
    'parents': optional(ARRAY(STRING)),

    # Legacy
    'keypoints': deprecated(optional(ARRAY(STRING))),
    'skeleton': deprecated(optional(ARRAY(TUPLE(INTEGER, INTEGER)))),
})

KEYPOINT_CATEGORY = OBJECT({
    'name': STRING,
    'id': INTEGER,
    'supercategory': STRING,
    'reflection_id': INTEGER,
})

# Extension
VIDEO = {
    'id': INTEGER,
    'name': ANYOF(PATH, STRING),
    'caption': STRING,
}

CHANNELS = STRING(title='CHANNEL_SPEC', description='experimental')

IMAGE = OrderedDict((
    ('id', INTEGER),
    ('file_name', PATH),

    ('width', optional(INTEGER)),
    ('height', optional(INTEGER)),

    # Extension
    ('video_id', optional(INTEGER)),
    ('timestamp', optional(INTEGER)),  # optional in flicks
    ('frame_index', optional(INTEGER)),  # optional

    ('channels', optional(CHANNELS)),

    # TODO: optional world localization information
    # TODO: camera information?

    ('auxillary', (
        {
            'file_name': PATH,
            'channels': CHANNELS,
            'width': optional(INTEGER),
            'height': optional(INTEGER),
        }, ...,
    )),
))

ANNOTATION = OrderedDict((
    ('id', INTEGER),
    ('image_id', INTEGER),

    ('bbox', TUPLE(NUMBER, NUMBER, NUMBER, NUMBER)),  # tl-x, tl-y, w, h

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
))


COCO_SCHEMA = OBJECT(
    required=[
        'categories',
        'images',
        'annotations'
    ],
    PROPERTIES={
        'info': ANY,
        'licenses': ANY,

        'categories': ARRAY(CATEGORY),

        'keypoint_categories': ARRAY(ANY),

        'videos': ARRAY(ANY),

        'images': ARRAY(ANY),

        'annotations': ARRAY(ANNOTATION),
    }
)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/coco_schema.py
    """
    # print('COCO_SCHEMA = {}'.format(ub.repr2(COCO_SCHEMA, nl=-1)))

    print('COCO_SCHEMA = {}'.format(ub.repr2(COCO_SCHEMA, nl=2)))
