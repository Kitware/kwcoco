"""
Proof of concept script for converting PNG mask segmentations into kwcoco
format.

Currently, there is no mechanism to associate annotations with images, and that
would need to be created for this script to be useful in real-world cases.
"""
import scriptconfig as scfg
import ubelt as ub
import kwimage
import numpy as np
from os.path import join


def _make_intmask_demodata(rng=None):
    """
    Creates demo data to test the script
    """
    import kwarray

    rng = kwarray.ensure_rng(rng)  # seeded random number generator

    dpath = ub.ensure_app_cache_dir('kwcoco/tests/masks')
    shape = (128, 128)
    num_masks = 10

    def _random_multi_obj_mask(shape, rng):
        """
        Create random int mask objects that can contain multiple objects.
        Each object is a different positive integer

        Ignore:
            kwplot.imshow(kwimage.normalize(data))
            kwplot.imshow(data)
        """
        num_objects = rng.randint(0, 5)
        data = np.zeros(shape, dtype=np.uint8)
        for obj_idx in range(0, num_objects + 1):
            # Make a binay mask and add it as a new objct
            binmask = kwimage.Mask.random(shape=shape, rng=rng).data
            data[binmask > 0] = obj_idx
        return data
    fpaths = [join(dpath, 'mask_{:04d}.png'.format(mask_idx))
              for mask_idx in range(num_masks)]
    for fpath in fpaths:
        data = _random_multi_obj_mask(shape, rng=rng)
        kwimage.imwrite(fpath, data)
    return dpath


class MakeCocoFromMasksCLI(object):
    name = 'toydata'

    class CLIConfig(scfg.Config):
        """
        Create a COCO file from bitmasks
        """
        default = {
            'src': scfg.PathList(help='a file, globstr, or comma-separated list of files'),
            'dst': scfg.Value('masks.mscoco.json', help='output path'),
            'serialization': scfg.Value('vector', help='can be raster or vector'),
        }
        epilog = r"""
        Example Usage:
            # hack to make the demo data
            python -c "import make_coco_from_masks; print(make_coco_from_masks._make_intmask_demodata())"

            # Run the conversion tool in vector mode
            python make_coco_from_masks.py \
                --src=$HOME/.cache/kwcoco/tests/masks/*.png \
                --dst vector.kwcoco.json --serialization=vector

            # Run the conversion tool in raster mode
            python make_coco_from_masks.py \
                --src=$HOME/.cache/kwcoco/tests/masks/*.png \
                --dst raster.kwcoco.json --serialization=raster

            # Visualize the results of vector mode
            kwcoco show --src vector.kwcoco.json --gid 2

            # Visualize the results of raster mode
            kwcoco show --src raster.kwcoco.json --gid 2
        """

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        CommandLine:
            xdoctest -m make_coco_from_masks.py MakeCocoFromMasksCLI.main

        Example:
            >>> import ubelt as ub
            >>> cls = MakeCocoFromMasksCLI
            >>> cmdline = False
            >>> dpath = _make_intmask_demodata()
            >>> kw = {'src': join(dpath, '*.png'), 'dst': 'masks.mscoco.json'}
            >>> fpath = cls.main(cmdline, **kw)
            >>> # Check validity
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset(fpath)
            >>> dset._check_integrity()
            >>> # Ensure we can convert to kwimage and access segmentations
            >>> dset.annots().detections.data['segmentations'][0].data
        """
        import kwcoco
        config = cls.CLIConfig(kw, cmdline=cmdline)
        print('config = {}'.format(ub.repr2(dict(config), nl=2)))
        serialization_method = config['serialization']

        # Initialize an empty COCO dataset
        coco_dset = kwcoco.CocoDataset()

        # Path to each mask object
        mask_fpaths = config['src']
        for mask_fpath in mask_fpaths:
            # I assume each mask corresponds to a single image, but I dont know
            # what those images should be at the moment. TODO: if this is going
            # to be a real script, then we should find a nice way of specifying
            # the correspondences between masks and the images to which they
            # belong. For now, I'm going to use the mask itself as a dummy
            # value.

            img_fpath = mask_fpath
            image_id = coco_dset.add_image(file_name=img_fpath)

            # Parse the mask file, and add each object as a new annotation
            multi_mask = kwimage.imread(mask_fpath)
            # I recall there is a better opencv of splitting these sort of
            # masks up into binary masks, maybe it was a connected-component
            # function? I guess it depends if you want disconnected objects
            # represented as separte annotations or not.  I'm just going to do
            # the naive thing for now.
            obj_idxs = np.setdiff1d(np.unique(multi_mask), [0])
            for obj_idx in obj_idxs:
                bin_data = (multi_mask == obj_idx).astype(np.uint8)

                # Create a kwimage object which has nice `to_coco` methods
                bin_mask = kwimage.Mask(bin_data, format='c_mask')

                # We can either save in our coco file as a raster RLE style
                # mask, or we can use a vector polygon style mask. Either of
                # the resulting coco_sseg objects is a valid value for an
                # annotation's "segmentation" field.
                if serialization_method == 'raster':
                    sseg = bin_mask
                    coco_sseg = sseg.to_coco(style='new')
                elif serialization_method == 'vector':
                    bin_poly = bin_mask.to_multi_polygon()
                    sseg = bin_poly
                    coco_sseg = sseg.to_coco(style='new')

                # Now we add this segmentation to the coco dataset as a new
                # annotation. The annotation needs to know which image it
                # belongs to, and ideally it has a category and a bounding box
                # as well.

                # We can make up a dummy category (note that ensure_category
                # will not fail if there is a duplicate entry but add_category
                # will)
                category_id = coco_dset.ensure_category('class_{}'.format(obj_idx))

                # We can use the kwimage sseg object to get the bounding box
                # FIXME: apparently the MultiPolygon object doesnt implement
                # `to_boxes`, at the moment, so force an inefficient conversion
                # back to a mask as a hack and use its to_boxes method.
                # Technically, the bounding box should not be required, but its
                # a good idea to always include it.
                bbox = list(sseg.to_mask(dims=multi_mask.shape).to_boxes().to_coco())[0]

                METHOD1 = False
                if METHOD1:
                    # We could just add it diretly like this
                    # FIXME: it should be ok to add an annotation without a
                    # category, but it seems like a recent change in kwcoco has
                    # broken that. That will be fixed in the future.
                    annot_id = coco_dset.add_annotation(
                            image_id=image_id, category_id=category_id,
                            bbox=bbox, segmentation=coco_sseg)
                else:
                    # But lets do it as if we were adding a segmentation to an
                    # existing dataset. In this case we access the
                    # CocoDataset's annotation index structure.
                    #
                    # First add the basic annotation
                    annot_id = coco_dset.add_annotation(
                            image_id=image_id, category_id=category_id,
                            bbox=bbox)
                    # Then use the annotation id to look up its coco-dictionary
                    # representation and simply add the segmentation field
                    ann = coco_dset.anns[annot_id]
                    ann['segmentation'] = coco_sseg

        # You dont have to set the fpath attr, but I tend to like it
        coco_dset.fpath = config['dst']
        print('Writing to fpath = {}'.format(ub.repr2(coco_dset.fpath, nl=1)))
        coco_dset.dump(coco_dset.fpath, newlines=True)
        return coco_dset.fpath


_CLI = MakeCocoFromMasksCLI

if __name__ == '__main__':
    _CLI.main()
