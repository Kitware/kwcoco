# Getting Started With KW-COCO


## FAQ

Q: What is `kwcoco`?
A: An extension of the MS-COCO data format for storing a "manifest" of
   categories, images, and annotations.

Q: Why yet another data format? 
A: MS-COCO did not have support for video and multimodal imagery. These are
   important problems in computer vision and it seems reasonable (although
   challenging) that there could be a data format that could be used as an
   interchange for almost all vision problems.

Q: Why extend MS-COCO and not create something else?
A: To draw on the existing adoption of the MS-COCO format.

Q: What's so great about MS-COCO?
A: It has an intuitive data structure that's simple to interface with.

Q: Why not pycocotools?
A: That module doesn't allow you to edit the dataset programmatically, and requires C backend. 
   This module allows dynamic modification addition and removal of images /
   categories / annotations / videos, in addition to other places where it goes
   beyond the functionality of the pycocotools module. We have a much more
   configurable / expressive way of computing and recording object detection
   metrics. If we are using an mscoco-compliant database (which can be verified
   / coerced from the `kwcoco conform` CLI tool), then we do call 
   pycocotools for functionality not directly implemented here.

Q: Would you ever extend kwcoco to go beyond computer vision?
A: Maybe, it would be something new though, and only use kwcoco as an
   inspiration. If extending past computer vision I would want to go back and
   rename / reorganize the spec.

## Design Goals

* Always be a strict superset of the original MS-COCO format 

* Extend the scope of MS-COCO to broader computer-vision domains.

* Have a fast pure-Python API to perform lower level tasks. (Allow optional C
  backends for features that need speed boosts)

* Have an easy-to-use command line interface to perform higher level tasks.

## Use cases

KWCoco has been designed to work with these tasks in these image modalities.


### Tasks

* Captioning

* Classification

* Segmentation

* Keypoint Detection / Pose Estimation

* Object Detection


### Modalities

* Single Image

* Video 

* Multispectral Imagery

* Images with auxiliary information (2.5d, flow, disparity, stereo) 

* Combinations of the above.



## Pseudo Spec

The following describes psuedo-code for the high level spec (some of which may
not be have full support in the Python API). A formal json-schema can be found
in the kwcoco module.

```
# All object categories are defined here.
category = {
    'id': int,
    'name': str,  # unique name of the category
    'supercategory': str,   # parent category name
}

# Videos are used to manage collections of sequences of images.
video = {
    "id": int,
    "name": str,  # a unique name for this video.
}

# Specifies how to find sensor data of a particular scene at a particular
# time. This is usually paths to rgb images, but auxiliary information
# can be used to specify multiple bands / etc...
image = {
    'id': int,

    'name': str,  # a unique name
    'file_name': str,  # relative path to the primary image data

    'auxiliary': [  # information about any auxiliary channels / bands
        {
            'file_name': str,  # relative path to associated file
            'channels': <spec>,  # a string encoding
        },
    ]

    'video_id': str  # if this image is a frame in a video sequence, this id is shared by all frames in that sequence.
    'timestamp': int  # timestamp (ideally in flicks), used to identify the timestamp of the frame. Only applicable video inputs.
    'frame_index': int  # ordinal frame index which can be used if timestamp is unknown.
}

# Ground truth is specified as annotations, each belongs to a spatial
# region in an image. This must reference a subregion of the image in pixel
# coordinates. Additional non-schma properties can be specified to track
# location in other coordinate systems. Annotations can be linked over time
# by specifying track-ids.
annotation = {
    'id': int,
    'image_id': int,
    'category_id': int,

    "track_id": <int | str | uuid>  # indicates association between annotations across frames

    'bbox': [tl_x, tl_y, w, h],  # xywh format)
    "score" : float,
    "prob" : List[float],
    "weight" : float,

    "caption": str,  # a text caption for this annotation
    "keypoints" : <Keypoints | List[int] > # an accepted keypoint format
    'segmentation': <RunLengthEncoding | Polygon | MaskPath | WKT >,  # an accepted segmentation format
}


# A dataset bundles a manifest of all aformentioned data into one file.
dataset = {
    # these are object level categories
    'categories': [category, ...],
    'images': [image, ...]
    'annotations': [annotation, ...]
    'licenses': [],
    'info': [],
}
```


## Technical Dept

Based on design decisions made in the original MS-COCO and KW-COCO, there are a
few weird things

* The "bbox" field gives no indication it should be xywh format.

* We can't use "vid" as a variable name for "video-id" because "vid" is also an
  abbreviation for "video". Hence, while category, image, and annotation all have
  a nice 1-letter prefix to their id in the standard variable names I use (i.e.
  cid, gid, aid). I have to use vidid to refer to "video-ids".

* I'm not in love with the way "keypoint_categories" are handled.

* Are "images" always "images"? Are "videos" always "videos"?

* Would we benefit from using JSON-LD?

* The "prob" field needs to be better defined 


## Code Examples. 

See the README and the doctests.
