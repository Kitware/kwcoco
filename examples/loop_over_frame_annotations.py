import kwcoco
import kwimage
from shapely.ops import unary_union

import geowatch
from geowatch.utils import util_resolution
dset = geowatch.coerce_kwcoco('geowatch', geodata=True, dates=True)

videos = dset.videos()
for video_id in videos:
    images = dset.images(video_id=video_id)

    for image_id in images:
        coco_img = dset.coco_image(image_id)

        _imgspace_resolution = coco_img.resolution(space='image')
        imgspace_resolution = util_resolution.ResolvedUnit(_imgspace_resolution['mag'], _imgspace_resolution['unit'])

        annots = coco_img.annots()
        annot_cat_ids = annots.lookup('category_id')
        annot_segmenations = annots.lookup('segmentation')
        annot_cat_names = dset.categories(annot_cat_ids).lookup('name')

        annot_polys = [kwimage.MultiPolygon.coerce(s).to_shapely() for s in annot_segmenations]

        do_not_ignore_poly = unary_union(annot_polys)

        # todo: Construct the buffered ignore polygons
        # subtract away the do_not_ignore_poly from them
        # add them as new annotations with an "ignore" label.
