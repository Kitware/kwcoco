"""
Small demo that shows how you can index and access multi-band images (e.g. SAR
/ MSI) via kwcoco, especially in the case where those bands have differing
resolutions on disk.


SeeAlso:

    More geospatial aware kwcoco logic can be found in
    https://gitlab.kitware.com/computer-vision/geowatch/-/blob/main/geowatch/utils/kwcoco_extensions.py

    Specifically, coco_populate_geo_heuristics will attempt to use geotiff
    metadata to populate the warp_aux_to_img transforms correctly.
"""
import kwcoco
import kwimage
import ubelt as ub
import numpy as np


def make_msi_demodata():
    """
    Write images for some number of random scenes on disk.
    """
    demo_dpath = ub.Path.appdir('kwcoco/sardemo').ensuredir()

    h = 512
    w = 256
    num_images = 10
    fpaths = []
    for idx in range(num_images):
        # Pretend we have bands at different resolutions
        # Pan is at base resolution, SAR is at 2x lower resolution,
        # and MSI is at 4x lower resolution
        pan_data = np.random.rand(h, w, 1)
        sar_data = np.random.rand(h // 2, w // 2, 2)
        msi_data = np.random.rand(h // 4, w // 4, 11)
        msi_fpath = demo_dpath / f'sar_{idx}.tif'
        sar_fpath = kwimage.imwrite(demo_dpath / f'sar_{idx}.tif', sar_data)
        pan_fpath = kwimage.imwrite(demo_dpath / f'pan_{idx}.tif', pan_data)
        msi_fpath = kwimage.imwrite(demo_dpath / f'msi_{idx}.tif', msi_data)
        fpaths.append((pan_fpath, sar_fpath, msi_fpath))
    return fpaths


def main():
    # Given file paths on disk, we organize them in a kwcoco dataset
    fpaths = make_msi_demodata()

    dset = kwcoco.CocoDataset()

    for fpath_tup in fpaths:
        pan_fpath, sar_fpath, msi_fpath = fpath_tup
        height1, width1 = kwimage.load_image_shape(pan_fpath, backend='gdal')[0:2]
        height2, width2 = kwimage.load_image_shape(sar_fpath, backend='gdal')[0:2]
        height3, width3 = kwimage.load_image_shape(msi_fpath, backend='gdal')[0:2]

        image_name = 'image_' + (ub.Path(pan_fpath).stem.split('_'))[1]

        # When adding images where bands are split over different files we keep
        # the top-level file_name as None, and then add assets for each
        # physical file on disk. The image still needs to have a "width/height"
        # that the assets can (virtually) align to.
        gid = dset.add_image(
            name=image_name,
            width=width1,
            height=height1
        )
        coco_image = dset.coco_image(gid)

        # When images are at different resolutions, we need to ensure we
        # specify an "warp_aux_to_img" to align them. In this case we simply
        # say that SAR scales up by 2x, and MSI scales up by 4x, which align
        # them with the PAN band.
        coco_image.add_asset(
            file_name=pan_fpath,
            width=width1,
            height=height1,
            channels='pan',
            warp_aux_to_img=kwimage.Affine.eye(),
        )
        coco_image.add_asset(
            file_name=sar_fpath,
            width=width2,
            height=height2,
            # Give a name to each channel in the image separated by a "|"
            channels='hh|vv',
            warp_aux_to_img=kwimage.Affine.coerce({'scale': 2})
        )
        coco_image.add_asset(
            file_name=msi_fpath,
            width=width3,
            height=height3,
            # Note: this code is a shorthand for 11 bands
            # shorthand codes are normalized as follows:
            # from delayed_image.channel_spec import ChannelSpec
            # ChannelSpec.coerce('msi.0:11').normalize()
            channels='msi.0:11',
            warp_aux_to_img=kwimage.Affine.coerce({'scale': 4})
        )

    # This is what the underlying kwcoco representation of the data looks like
    print(f'dset.dataset = {ub.urepr(dset.dataset, nl=-1)}')

    # Grab an arbitrary image from our dataset
    image_id = 3
    coco_image = dset.coco_image(image_id)

    # By default calling imdelay references all channels
    delayed = coco_image.imdelay()

    # Or specify channels (notice the expanded shorthand names)
    delayed = coco_image.imdelay('hh|vv|msi.2|msi.9|pan|red|green|blue')

    # In the delayed "image space", we can grab a crop
    cropped = delayed[100:124, 124:148]
    cropped = cropped.optimize()
    cropped.optimize().print_graph()

    # Finalize will always return an aligned "scaled" single ndarray
    cropped.finalize().shape

    # BUT, we can load "naive scale" corner-aligned crops by undoing the #
    # scale component of each transform.
    undone_parts, jagged_align = cropped.undo_warps(
        remove=['scale'], squash_nans=True,
        return_warps=True)

    # Can call finalize on each individual part to get the native unscaled
    # component of the ratser.
    datas = ub.udict({})
    for part in undone_parts:
        part.print_graph()
        data = part.finalize()
        datas[part.channels.spec] = data

    print(ub.repr2(datas.map_values(lambda a: a.shape), nl=1))

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/examples/example_sar_and_msi.py
    """
    main()
