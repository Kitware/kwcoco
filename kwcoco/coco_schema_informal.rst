An informal spec is as follows:

.. code:: python

    # All object categories are defined here.
    category = {
        'id': int,
        'name': str,  # unique name of the category
        'supercategory': str,   # parent category name
    }

    # Videos are used to manage collections or sequences of images.
    # Frames do not necesarilly have to be aligned or uniform time steps
    video = {
        'id': int,
        'name': str,  # a unique name for this video.

        'width': int  # the base width of this video (all associated images must have this width)
        'height': int  # the base height of this video (all associated images must have this height)

        'resolution': int | str,  # indicates the size of a pixel in video space

        # In the future this may be extended to allow pointing to video files
    }

    # Specifies how to find sensor data of a particular scene at a particular
    # time. This is usually paths to rgb images, but auxiliary information
    # can be used to specify multiple bands / etc...

    # NOTE: in the future we will transition from calling these auxiliary items
    # to calling these asset items. As such the key will change from
    # "auxiliary" to "asset". The API will be updated to maintain backwards
    # compatibility while this transition occurs.

    image = {
        'id': int,

        'name': str,  # an encouraged but optional unique name (ideally not larger than 256 characters)
        'file_name': str,  # relative path to the "base" image data (optional if auxiliary items are specified)

        'width': int,   # pixel width of "base" image
        'height': int,  # pixel height of "base" image

        'channels': <ChannelSpec>,   # a string encoding of the channels in the main image (optional if auxiliary items are specified)
        'resolution': int | str,  # indicates the size of a pixel in image space

        'auxiliary': [  # information about any auxiliary channels / bands
            {
                'file_name': str,     # relative path to associated file
                'channels': <ChannelSpec>,   # a string encoding
                'width':     <int>    # pixel width of image asset
                'height':    <int>    # pixel height of image asset
                'warp_aux_to_img': <TransformSpec>,  # tranform from "base" image space to auxiliary/asset space. (identity if unspecified)
                'quantization': <QuantizationSpec>,  # indicates that the underlying data was quantized
            }, ...
        ]

        'video_id': str  # if this image is a frame in a video sequence, this id is shared by all frames in that sequence.
        'timestamp': str | int  # a iso-8601 or unix timestamp.
        'frame_index': int  # ordinal frame index which can be used if timestamp is unknown.
        'warp_img_to_vid': <TransformSpec>  # a transform image space to video space (identity if unspecified), can be used for sensor alignment or video stabilization
    }

    TransformSpec:
        The spec can be anything coercable to a kwimage.Affine object.
        This can be an explicit affine transform matrix like:
            {'type': 'affine': 'matrix': <a-3x3 matrix>},

        But it can also be a concise dict containing one or more of these keys
            {
                'scale': <float|Tuple[float, float]>,
                'offset': <float|Tuple[float, float]>,
                'skew': <float>,
                'theta': <float>,  # radians counter-clock-wise
            }

    ChannelSpec:
        This is a string that describes the channel composition of an image.
        For the purposes of kwcoco, separate different channel names with a
        pipe ('|'). If the spec is not specified, methods may fall back on
        grayscale or rgb processing. There are special string. For instance
        'rgb' will expand into 'r|g|b'. In other applications you can "late
        fuse" inputs by separating them with a "," and "early fuse" by
        separating with a "|". Early fusion returns a solid array/tensor, late
        fusion returns separated arrays/tensors.

    QuantizationSpec:
        This is a dictionary of the form:
            {
                'orig_min': <float>, # min original intensity
                'orig_max': <float>, # min original intensity
                'quant_min': <int>, # min quantized intensity
                'quant_max': <int>, # max quantized intensity
                'nodata': <int|None>,  # integer value to interpret as nan
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

        'track_id': <int | str | uuid>  # indicates association between annotations across images

        'bbox': [tl_x, tl_y, w, h],  # xywh format)
        'score' : float,
        'prob' : List[float],
        'weight' : float,

        'caption': str,  # a text caption for this annotation
        'keypoints' : <Keypoints | List[int] > # an accepted keypoint format
        'segmentation': <RunLengthEncoding | Polygon | MaskPath | WKT >,  # an accepted segmentation format
    }

    # In development: tracks - an object to store properties associated with
    # all annotations with a track-id.
    tracks = {
        'id': int,
        'name': int,
    }

    # A dataset bundles a manifest of all aformentioned data into one structure.
    dataset = {
        'categories': [category, ...],
        'videos': [video, ...]
        'images': [image, ...]
        'annotations': [annotation, ...]
        'licenses': [],
        'info': [],
    }

    Polygon:
        A flattened list of xy coordinates.
        [x1, y1, x2, y2, ..., xn, yn]

        or a list of flattened list of xy coordinates if the CCs are disjoint
        [[x1, y1, x2, y2, ..., xn, yn], [x1, y1, ..., xm, ym],]

        Note: the original coco spec does not allow for holes in polygons.

        We also allow a non-standard dictionary encoding of polygons
            {'exterior': [(x1, y1)...],
             'interiors': [[(x1, y1), ...], ...]}

        TODO: Support WTK

    RunLengthEncoding:
        The RLE can be in a special bytes encoding or in a binary array
        encoding. We reuse the original C functions are in [PyCocoToolsMask]_
        in ``kwimage.structs.Mask`` to provide a convinient way to abstract
        this rather esoteric bytes encoding.

        For pure python implementations see kwimage:
            Converting from an image to RLE can be done via kwimage.run_length_encoding
            Converting from RLE back to an image can be done via:
                kwimage.decode_run_length

            For compatibility with the COCO specs ensure the binary flags
            for these functions are set to true.

    Keypoints:
        Annotation keypoints may also be specified in this non-standard (but
        ultimately more general) way:

        'annotations': [
            {
                'keypoints': [
                    {
                        'xy': <x1, y1>,
                        'visible': <0 or 1 or 2>,
                        'keypoint_category_id': <kp_cid>,
                        'keypoint_category': <kp_name, optional>,  # this can be specified instead of an id
                    }, ...
                ]
            }, ...
        ],
        'keypoint_categories': [{
            'name': <str>,
            'id': <int>,  # an id for this keypoint category
            'supercategory': <kp_name>  # name of coarser parent keypoint class (for hierarchical keypoints)
            'reflection_id': <kp_cid>  # specify only if the keypoint id would be swapped with another keypoint type
        },...
        ]

        In this scheme the "keypoints" property of each annotation (which used
        to be a list of floats) is now specified as a list of dictionaries that
        specify each keypoints location, id, and visibility explicitly. This
        allows for things like non-unique keypoints, partial keypoint
        annotations. This also removes the ordering requirement, which makes it
        simpler to keep track of each keypoints class type.

        We also have a new top-level dictionary to specify all the possible
        keypoint categories.

        TODO: Support WTK

    Auxiliary Channels / Image Assets:
        For multimodal or multispectral images it is possible to specify
        auxiliary channels in an image dictionary as follows:

        {
            'id': int,
            'file_name': str,    # path to the "base" image (may be None)
            'name': str,         # a unique name for the image (must be given if file_name is None)
            'channels': <ChannelSpec>,  # a spec code that indicates the layout of the "base" image channels.
            'auxiliary': [  # information about auxiliary channels
                {
                    'file_name': str,
                    'channels': <ChannelSpec>
                }, ... # can have many auxiliary channels with unique specs
            ]
        }

        Note that specifing a filename / channels for the base image is not
        necessary, and mainly useful for augmenting an existing single-image
        dataset with multimodal information. Typically if an image consists of
        more than one file, all file information should be stored in the
        "auxiliary" or "assets" list.

        NEW DOCS:
            In an MSI use case you should think of the "auxiliary" list as a
            list of single file assets that are composed to make the entire
            image. Your assets might include sensed bands, computed features,
            or quality information. For instance a list of auxiliary items may
            look like this:

            image = {
                "name": "my_msi_image",
                "width": 400,
                "height": 400,

                "video_id": 2,
                "timestamp": "2020-01-1",
                "frame_index": 5,
                "warp_img_to_vid": {"type": "affine", "scale", 1.4},

                "auxiliary": [
                   {"channels": "red|green|blue": "file_name": "rgb.tif", "warp_aux_to_img": {"scale": 1.0}, "height": 400, "width": 400, ...},
                   ...
                   {"channels": "cloudmask": "file_name": "cloudmask.tif", "warp_aux_to_img": {"scale": 4.0}, "height": 100, "width": 100, ...},
                   {"channels": "nir": "file_name": "nir.tif", "warp_aux_to_img": {"scale": 2.0}, "height": 200, "width": 200, ...},
                   {"channels": "swir": "file_name": "swir.tif", "warp_aux_to_img": {"scale": 2.0}, "height": 200, "width": 200, ...},
                   {"channels": "model1_predictions.0:6": "file_name": "model1_preds.tif", "warp_aux_to_img": {"scale": 8.0}, "height": 50, "width": 50, ...},
                   {"channels": "model2_predictions.0:3": "file_name": "model2_preds.tif", "warp_aux_to_img": {"scale": 8.0}, "height": 50, "width": 50, ...},
                ]
            }

            Note that there is no file_name or channels parameter in the image
            object itself. This pattern indicates that image is composed of
            multiple assets. One could indicate that an asset is primary by
            giving its information to the parent image, but for better STAC
            compatibility, all assets for MSI images should simply be listed
            as "auxiliary" items.

        NOTE: in the future assets may be moved to a top-level table to better
        support common relational database patterns.


    Video Sequences:
        For video sequences, we add the following video level index:

        'videos': [
            { 'id': <int>, 'name': <video_name:str> },
        ]

        Note that the videos might be given as encoded mp4/avi/etc.. files (in
        which case the name should correspond to a path) or as a series of
        frames in which case the images should be used to index the extracted
        frames and information in them.

        Then image dictionaries are augmented as follows:

        {
            'video_id': str  # optional, if this image is a frame in a video sequence, this id is shared by all frames in that sequence.
            'timestamp': str | int  # optional, an iso8601 or unix timestamp
            'frame_index': int  # optional, ordinal frame index which can be used if timestamp is unknown.
        }

        And annotations are augmented as follows:

        {
            'track_id': <int | str | uuid>  # optional, indicates association between annotations across frames
        }

    Tracks:

        Track level properties for groups of annotations can be stored in track
        dictionaries.  Sometimes it is useful to include summary geometry in
        track dictionaries. Unlike annotations - which store their geometry in
        **image space**, track geometry should be specified in **video space**.

        {
            'track_id': <int | str >  # internal integer id for the track
            'name': <str>  # external name id for the track.
        }

