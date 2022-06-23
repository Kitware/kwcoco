import kwimage
import numpy as np
from kwcoco.util.delayed_poc.delayed_leafs import DelayedWarp


def _devcheck_corner():
    """
    Debugging code
    """
    self = DelayedWarp.random(rng=0)
    print(self.nesting())
    region_slices = (slice(40, 90), slice(20, 62))
    region_box = kwimage.Boxes.from_slice(region_slices, shape=self.shape)
    region_bounds = region_box.to_polygons()[0]

    for leaf in self._optimize_paths():
        pass

    # tf_leaf_to_root = leaf['transform']
    tf_leaf_to_root = leaf.meta['transform']
    tf_root_to_leaf = np.linalg.inv(tf_leaf_to_root)

    leaf_region_bounds = region_bounds.warp(tf_root_to_leaf)
    leaf_region_box = leaf_region_bounds.bounding_box().to_ltrb()
    leaf_crop_box = leaf_region_box.quantize()
    lt_x, lt_y, rb_x, rb_y = leaf_crop_box.data[0, 0:4]

    root_crop_corners = leaf_crop_box.to_polygons()[0].warp(tf_leaf_to_root)

    # leaf_crop_slices = (slice(lt_y, rb_y), slice(lt_x, rb_x))

    crop_offset = leaf_crop_box.data[0, 0:2]
    corner_offset = leaf_region_box.data[0, 0:2]
    offset_xy = crop_offset - corner_offset

    tf_root_to_leaf

    # NOTE:

    # Cropping applies a translation in whatever space we do it in
    # We need to save the bounds of the crop.
    # But now we need to adjust the transform so it points to the
    # cropped-leaf-space not just the leaf-space, so we invert the implicit
    # crop

    tf_crop_to_leaf = kwimage.Affine.translate(offset=crop_offset)

    # tf_newroot_to_root = kwimage.Affine.affine(offset=region_box.data[0, 0:2])
    tf_root_to_newroot = kwimage.Affine.translate(offset=region_box.data[0, 0:2]).inv()

    tf_crop_to_leaf = kwimage.Affine.translate(offset=crop_offset)
    tf_crop_to_newroot = tf_root_to_newroot @ tf_leaf_to_root @ tf_crop_to_leaf
    tf_newroot_to_crop = tf_crop_to_newroot.inv()

    # tf_leaf_to_crop
    # tf_corner_offset = kwimage.Affine.translate(offset=offset_xy)

    subpixel_offset = kwimage.Affine.translate(offset=offset_xy).matrix
    tf_crop_to_leaf = subpixel_offset
    # tf_crop_to_root = tf_leaf_to_root @ tf_crop_to_leaf
    # tf_root_to_crop = np.linalg.inv(tf_crop_to_root)

    if 1:
        import kwplot
        kwplot.autoplt()

        lw, lh = leaf.sub_data.shape[0:2]
        leaf_box = kwimage.Boxes([[0, 0, lw, lh]], 'xywh')
        root_box = kwimage.Boxes([[0, 0, self.dsize[0], self.dsize[1]]], 'xywh')

        ax1 = kwplot.figure(fnum=1, pnum=(2, 2, 1), doclf=1).gca()
        ax2 = kwplot.figure(fnum=1, pnum=(2, 2, 2)).gca()
        ax3 = kwplot.figure(fnum=1, pnum=(2, 2, 3)).gca()
        ax4 = kwplot.figure(fnum=1, pnum=(2, 2, 4)).gca()
        root_box.draw(setlim=True, ax=ax1)
        leaf_box.draw(setlim=True, ax=ax2)

        region_bounds.draw(ax=ax1, color='green', alpha=.4)
        root_crop_corners.draw(ax=ax1, color='purple', alpha=.4)
        ax1.set_title('root')

        leaf_region_bounds.draw(ax=ax2, color='green', alpha=.4)
        leaf_crop_box.draw(ax=ax2, color='purple')
        ax2.set_title('leaf')

        new_w = region_box.to_xywh().data[0, 2]
        new_h = region_box.to_xywh().data[0, 3]
        ax3.set_xlim(0, new_w)
        ax3.set_ylim(0, new_h)

        crop_w = leaf_crop_box.to_xywh().data[0, 2]
        crop_h = leaf_crop_box.to_xywh().data[0, 3]
        ax4.set_xlim(0, crop_w)
        ax4.set_ylim(0, crop_h)

        pts3_ = kwimage.Points.random(3).scale((new_w, new_h))
        pts3 = kwimage.Points(xy=np.vstack([[[0, 0], [5, 5], [0, 49], [40, 45]], pts3_.xy]))
        pts4 = pts3.warp(tf_newroot_to_crop.matrix)
        pts3.draw(ax=ax3)
        pts4.draw(ax=ax4)

    # delayed_crop = band2.delayed_crop(region_slices)
    # final_crop = delayed_crop.finalize()
