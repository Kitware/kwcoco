"""
This example demonstrates how to use kwcoco to write a very simple torch
dataset. This assumes the dataset will be single-image RGB inputs.  This file
is intended to talk the reader through what we are doing and why.

This example aims for clairity over being concise. There are APIs exposed by
kwcoco (and its sister module ndsampler) that can perform the same tasks more
efficiently and with fewer lines of code.

If you run the doctest, it will produce a visualization that shows the images
with boxes drawn on it, running it multiple times will let you see the
augmentations. This can be done with the following command:


    xdoctest -m kwcoco.examples.simple_kwcoco_torch_dataset KWCocoSimpleTorchDataset --show


Or just copy the doctest into IPython and run it.
"""
try:
    import torch
    DatasetBase = torch.utils.data.Dataset
except Exception:
    torch = None
    DatasetBase = object

import kwcoco
import kwimage
import kwarray
import ubelt as ub


class KWCocoSimpleTorchDataset(DatasetBase):
    """
    A simple torch dataloader where each image is considered a single item.

    Args:
        coco_dset (kwcoco.CocoDataset | str):
            something coercable to a kwcoco dataset, this could either be a
            :class:`kwcoco.CocoDataset` object, a path to a kwcoco manifest on
            disk, or a special toydata code. See
            :func:`kwcoco.CocoDataset.coerce` for more details.


        input_dims (Tuple[int, int]): These are the (height, width) dimensions
            that the image will be resized to.

        antialias (bool, default=False): If true, we will antialias before
            downsampling.

        rng (RandomState | int | None): an existing random number generator or
            a random seed to produce deterministic augmentations.

    Example:
        >>> # xdoctest: +REQUIRES(module:torch)
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('shapes8')
        >>> input_dims = (384, 384)
        >>> self = torch_dset = KWCocoSimpleTorchDataset(coco_dset, input_dims=input_dims)
        >>> index = len(self) // 2
        >>> item = self[index]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.autompl()
        >>> canvas = item['inputs']['rgb'].numpy().transpose(1, 2, 0)
        >>> # Construct kwimage objects for batch item visualization
        >>> dets = kwimage.Detections(
        >>>     boxes=kwimage.Boxes(item['labels']['cxywh'], 'cxywh'),
        >>>     class_idxs=item['labels']['class_idxs'],
        >>>     classes=self.classes,
        >>> ).numpy()
        >>> # Overlay annotations on the image
        >>> canvas = dets.draw_on(canvas)
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()
    """

    def __init__(self, coco_dset, input_dims=None, antialias=False, rng=None):

        # Store a pointer to the coco dataset
        self.coco_dset = kwcoco.CocoDataset.coerce(coco_dset)

        if input_dims is None:
            raise ValueError(ub.paragraph(
                '''
                Must currently specify the height/width input dimensions to the
                network, so we can resample to that expected shape.
                '''))

        self.input_dims = input_dims
        self.antialias = antialias

        self.rng = kwarray.ensure_rng(rng)

        # Build a "grid" that maps an index to enough information to sample
        # data used to construct a batch item. In this case each sample
        # returned by __getitem__ will correspond to an entire image, so we
        # just store a list of image-ids. Note, if we are only interested in
        # some subset images, we could perform a filtering step here.
        self.gids = list(self.coco_dset.imgs.keys())

        # This is a kwcoco.CategoryTree object and it helps maintain the
        # mappings between contiguous class indexes (used by the network)
        # integer class ids (used by the kwcoco file) and string class names
        # (used by humans). Be sure that any torch network you build holds a
        # copy of this object (see self.classes.__json__), so the class
        # encoding is always coupled with the model.
        self.classes = self.coco_dset.object_categories()

        self.augment = True

    def __len__(self):
        """
        Return the number of items in the Dataset
        """
        return len(self.gids)

    def __getitem__(self, index):
        """
        Construct a batch item to be used in training
        """
        gid = self.gids[index]

        if 0:
            # Note: there is an experimental method for lazy image operations
            # we may default to recommending in the future.
            raw_img = self.coco_dset.delayed_load(gid).finalize()
        else:
            # For now, we will just do things in the most transparent way
            image_fpath = self.coco_dset.get_image_fpath(gid)
            # Note: might need to use itk imread (or better extend
            # kwimage.imread to also wrap itk formats)
            raw_img = kwimage.imread(image_fpath)

        # Lookup the annotations that correspond to this image
        aids = self.coco_dset.gid_to_aids[gid]

        # The the specific dataset, model, and loss function is what defines
        # what sort of label information is needed. Because that is not defined
        # for this tutorial, we will show how to manipulate all the annotation
        # information, but our final label will only consist of truth bounding
        # boxes and category ids.
        anns = [self.coco_dset.anns[aid] for aid in aids]

        # kwimage data structures makes handling spatial annotations on
        # images easier by bundling transformations of all annotations.
        raw_dets = kwimage.Detections.from_coco_annots(anns, dset=self.coco_dset)

        # Process the data and the annotations

        # Use if you want to ensure grayscale images are interpreted as rgb
        imdata = kwimage.atleast_3channels(raw_img)
        dets = raw_dets

        # Do whatever sort of augmentation you want here.  Remember, whenever
        # we transform the image, we also need to transform the annotations.
        if self.augment:
            # Build up an Affine augmentation

            aug_transform = kwimage.Affine.eye()
            if self.rng.rand() < 0.5:
                # horizontal flip with 0.5 probability
                h, w = imdata.shape[0:2]
                flip_transform = kwimage.Affine.affine(
                    scale=(-1, 1),
                    about=(w / 2, h / 2)
                )
                aug_transform = flip_transform @ aug_transform

            if self.rng.rand() < 0.8:
                # small translation / scale perterbation with 80% probability
                random_transform = kwimage.Affine.random(
                    # scale= not implemented as a distribution yet
                    # offset= not implemented as a distribution yet
                    shearx=0,
                    theta=0,
                    rng=self.rng
                )
                aug_transform = random_transform @ aug_transform

            # Augment the image and the dets
            imdata = kwimage.warp_affine(imdata, aug_transform)
            dets = dets.warp(aug_transform.matrix)

        # Use the convention where dims/shape are ordered as height,width and
        # size/dsize are width,height.
        input_dsize = self.input_dims[::-1]

        # Use imresize to finalize
        imdata, info = kwimage.imresize(imdata, dsize=input_dsize,
                                        antialias=self.antialias,
                                        return_info=True)

        resize_tf = kwimage.Affine.affine(offset=info['offset'],
                                          scale=info['scale'])
        dets = dets.warp(resize_tf.matrix)

        if 0:
            # The `dets.data` and `dets.meta` dictionaries contain annot info
            dets.data['boxes']
            dets.data['segmentations']
            dets.data['keypoints']
            dets.data['class_idxs']

        cxywh = torch.from_numpy(dets.data['boxes'].to_cxywh().data)
        class_idxs = torch.from_numpy(dets.data['class_idxs'])
        rgb_chw = torch.from_numpy(imdata.transpose(2, 0, 1)).float() / 255.  # Magic dataset-level normalization

        # It is best practices that a data loader returns a dictionary
        # so it is easy to add / remove data input and label information.
        item = {
            # Encode the inputs to the network for torch
            'inputs': {
                'rgb': rgb_chw,
            },

            # Encode the truth labels for torch
            'labels': {
                'cxywh': cxywh,
                'class_idxs': class_idxs,
            }
        }
        return item
