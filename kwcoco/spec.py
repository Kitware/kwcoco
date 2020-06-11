"""
TODO: need to investigate the correct way to use typing, what its limitations
are, and what we have gained by carefully designing a type spec.
"""

import pathlib
import uuid
import numbers
from typing import Optional, Any, Tuple, List, Set, Dict, AnyStr, Text, Union, NewType
from collections import OrderedDict

Int = NewType('Int', numbers.Integral)
Float = NewType('Float', numbers.Rational)
Str = NewType('Str', Text)
Path = NewType('Path', pathlib.Path)
UUID = NewType('UUID', uuid.UUID)

Int = numbers.Integral
Float = numbers.Rational
Str = Text
Path = pathlib.Path
UUID = uuid.UUID


category = {
    'id': Int,
    'name': Str,
    'supercategory': Optional[Str],

    # Legacy
    'keypoints': Optional[List[Str]],
    'skeleton': Optional[List[Tuple[Int, Int]]],
}

keypoint_category = {
    'name': Str,
    'id': Int,
    'supercategory': Str,
    'reflection_id': Int,
}

# Extension
video = {
    'id': Int,
    'name': Union[Path, Str],
    'caption': Text,
}

channel = NewType('ChannelSpec', {
    # experimental
    'spec': Str,
})

image = OrderedDict([
    ('id', Int),
    ('file_name', Path),

    ('width', Optional[Int]),
    ('height', Optional[Int]),

    # Extension
    ('video_id', Optional[Int]),
    ('timestamp', Optional[Int]),  # optional in flicks
    ('frame_index', Optional[Int]),  # optional

    ('channels', Optional[channel]),

    # TODO: optional world localization information
    # TODO: camera information?

    ('auxillary', [
        {
            'file_name': Path,
            'channels': channel,
            'width': Optional[Int],
            'height': Optional[Int],
        }, ...,
    ]),
])

Polygon = NewType('Polygon', {
    'exterior': List[Tuple[Float, Float]],  # ccw xy exterior points
    'interiors': List[List[Tuple[Float, Float]]],  # multiple cw xy holes
})

Keypoint = NewType('Keypoint', {
    'xy': Tuple[Int, Int],  # <x1, y1>,
    'visible': Int,  # choice(0, 1, 2),
    'keypoint_category_id': Int,
    'keypoint_category': Optional[Str],  # only to be used as a hint
})


RunLengthEncoding = Text  # original coco specification
OldPolygon = Union[List[Float], List[List[Float]]]
Segmentation = Union[OldPolygon, Polygon, RunLengthEncoding]

keypoints_v1 = List[Int]  # old format [x1,y1,v1,...,xk,yk,vk],
keypoints_v2 = List[Keypoint]
keypoints = Union[keypoints_v1, keypoints_v2]

annotation = OrderedDict([
    ('id', Int),
    ('image_id', Int),

    ('bbox', Tuple[Float, Float, Float, Float]),  # tl-x, tl-y, w, h

    ('category_id', Optional[Int]),
    ('track_id', Optional[Union[Int, Str, UUID]]),

    ('segmentation', Optional[Segmentation]),
    ('keypoints', Optional[List[Keypoint]]),

    # this needs to be in the same order as categories
    ('prob', Optional[List[Float]]),

    ('score', Optional[Float]),
    ('weight', Optional[Float]),

    ('iscrowd', Text),  # legacy
    ('caption', Text),
])


# The order is important for fast loading.
dataset = OrderedDict([
    ('info', Any),
    ('licenses', Any),

    ('categories', [category, ...]),

    ('keypoint_categories', [keypoint_category, ...]),

    ('videos', [video, ...]),

    ('images', [image, ...]),

    ('annotations', [annotation, ...]),
])

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/spec.py
    """
    import ubelt as ub
    print('dataset = {}'.format(ub.repr2(dataset, nl=True)))
