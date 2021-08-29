import cv2
import six
import kwarray
import kwimage
import numpy as np
import skimage
import ubelt as ub


class CategoryPatterns(object):
    """
    Example:
        >>> from kwcoco.demo.toypatterns import *  # NOQA
        >>> self = CategoryPatterns.coerce()
        >>> chip = np.zeros((100, 100, 3))
        >>> offset = (20, 10)
        >>> dims = (160, 140)
        >>> info = self.random_category(chip, offset, dims)
        >>> print('info = {}'.format(ub.repr2(info, nl=1)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(info['data'], pnum=(1, 2, 1), fnum=1, title='chip-space')
        >>> kpts = kwimage.Points._from_coco(info['keypoints'])
        >>> kpts.translate(-np.array(offset)).draw(radius=3)
        >>> #####
        >>> mask = kwimage.Mask.coerce(info['segmentation'])
        >>> kwplot.imshow(mask.to_c_mask().data, pnum=(1, 2, 2), fnum=1, title='img-space')
        >>> kpts.draw(radius=3)
        >>> kwplot.show_if_requested()
    """

    _default_categories = [
        {'name': 'background', 'id': 0, 'keypoints': []},
        {'name': 'box', 'id': 1, 'supercategory': 'vector', 'keypoints': []},
        {'name': 'circle', 'id': 2, 'keypoints': [], 'supercategory': 'vector'},
        {'name': 'star', 'id': 3, 'supercategory': 'vector', 'keypoints': []},
        {'name': 'octagon', 'id': 4, 'supercategory': 'vector', 'keypoints': []},
        {'name': 'diamond', 'id': 5, 'supercategory': 'vector', 'keypoints': []},
        {'name': 'superstar', 'id': 6, 'supercategory': 'raster', 'keypoints': ['left_eye', 'right_eye']},
        {'name': 'eff', 'id': 7, 'supercategory': 'raster', 'keypoints': ['top_tip', 'mid_tip', 'bot_tip']},
        {'name': 'raster', 'id': 8, 'supercategory': 'raster', 'keypoints': []},
        {'name': 'vector', 'id': 9, 'supercategory': 'shape', 'keypoints': []},
        {'name': 'shape', 'id': 10, 'keypoints': []},
    ]

    _default_keypoint_categories = [
        {'name': 'left_eye', 'id': 1, 'reflection_id': 2},
        {'name': 'right_eye', 'id': 2, 'reflection_id': 1},
        {'name': 'top_tip', 'id': 3, 'reflection_id': None},
        {'name': 'mid_tip', 'id': 4, 'reflection_id': None},
        {'name': 'bot_tip', 'id': 5, 'reflection_id': None},
    ]

    _default_catnames = [
        # 'circle',
        'star',
        'eff',
        'superstar',
    ]

    @classmethod
    def coerce(CategoryPatterns, data=None, **kwargs):
        """
        Construct category patterns from either defaults or only with specific
        categories. Can accept either an existig category pattern object, a
        list of known catnames, or mscoco category dictionaries.

        Example:
            >>> data = ['superstar']
            >>> self = CategoryPatterns.coerce(data)
        """
        if isinstance(data, CategoryPatterns):
            return data
        else:
            if data is None:
                # use defaults
                catnames = CategoryPatterns._default_catnames
                cname_to_cat = {c['name']: c for c in CategoryPatterns._default_categories}
                arg = list(ub.take(cname_to_cat, catnames))
            elif ub.iterable(data) and len(data) > 0:
                # choose specific catgories
                if isinstance(data[0], six.string_types):
                    catnames = data
                    cname_to_cat = {c['name']: c for c in CategoryPatterns._default_categories}
                    arg = list(ub.take(cname_to_cat, catnames))
                elif isinstance(data[0], dict):
                    arg = data
                else:
                    raise Exception
            else:
                raise Exception
            return CategoryPatterns(categories=arg, **kwargs)

    def __init__(self, categories=None, fg_scale=0.5, fg_intensity=0.9, rng=None):
        """
        Args:
            categories (List[Dict]): List of coco category dictionaries
        """
        self.rng = kwarray.ensure_rng(rng)
        self.fg_scale = fg_scale
        self.fg_intensity = fg_intensity

        self._category_to_elemfunc = {
            'superstar': lambda x: Rasters.superstar(),
            'eff':       lambda x: Rasters.eff(),
            'box':     lambda x: (skimage.morphology.square(x), None),
            'star':    lambda x: (star(x), None),
            'circle':  lambda x: (skimage.morphology.disk(x), None),
            'octagon': lambda x: (skimage.morphology.octagon(x // 2, int(x / (2 * np.sqrt(2)))), None),
            'diamond': lambda x: (skimage.morphology.diamond(x), None),
        }
        # Make generation of shapes a bit faster?
        # Maybe there are too many input combinations for this?
        # If we only allow certain size generations it should be ok

        # for key in self._category_to_elemfunc.keys():
        #     self._category_to_elemfunc[key] = ub.memoize(self._category_to_elemfunc[key])

        # keep track of which keypoints belong to which categories
        self.categories = categories
        self.cname_to_kp = {c['name']: c.get('keypoints', [])
                            for c in self.categories}

        self.obj_catnames = sorted([c['name'] for c in self.categories])
        self.kp_catnames = sorted(ub.flatten(self.cname_to_kp.values()))

        kpname_to_cat = {c['name']: c for c in CategoryPatterns._default_keypoint_categories}
        self.keypoint_categories = list(ub.take(kpname_to_cat, self.kp_catnames))

        # flatten list of all keypoint categories
        # self.kp_catnames = list(
        #     ub.flatten([self.cname_to_kp.get(cname, [])
        #                 for cname in self.obj_catnames])
        # )
        self.cname_to_cid = {
            cat['name']: cat['id'] for cat in self.categories
        }
        self.cname_to_cx = {
            cat['name']: cx for cx, cat in enumerate(self.categories)
        }

    def __len__(self):
        return len(self.obj_catnames)

    def __getitem__(self, index):
        return self.categories[index]

    def __iter__(self):
        for cat in self.categories:
            yield cat

    def index(self, name):
        return self.cname_to_cx[name]

    def get(self, index, default=ub.NoParam):
        if default is ub.NoParam:
            return self.categories[index]
        else:
            try:
                return self.categories[index]
            except KeyError:
                return default

    def random_category(self, chip, xy_offset=None, dims=None,
                        newstyle=True, size=None):
        """
        Example:
            >>> from kwcoco.demo.toypatterns import *  # NOQA
            >>> self = CategoryPatterns.coerce(['superstar'])
            >>> chip = np.random.rand(64, 64)
            >>> info = self.random_category(chip)

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(self.random_category))

        """
        cname = self.rng.choice(self.obj_catnames)
        info = self.render_category(
                cname, chip, xy_offset=xy_offset, dims=dims, newstyle=newstyle,
                size=size)
        return info

    def render_category(self, cname, chip, xy_offset=None, dims=None,
                        newstyle=True, size=None):
        """
        Example:
            >>> from kwcoco.demo.toypatterns import *  # NOQA
            >>> self = CategoryPatterns.coerce(['superstar'])
            >>> chip = np.random.rand(64, 64)
            >>> info = self.render_category('superstar', chip, newstyle=True)
            >>> print('info = {}'.format(ub.repr2(info, nl=-1)))
            >>> info = self.render_category('superstar', chip, newstyle=False)
            >>> print('info = {}'.format(ub.repr2(info, nl=-1)))

        Example:
            >>> from kwcoco.demo.toypatterns import *  # NOQA
            >>> self = CategoryPatterns.coerce(['superstar'])
            >>> chip = None
            >>> dims = (64, 64)
            >>> info = self.render_category('superstar', chip, newstyle=True, dims=dims, size=dims)
            >>> print('info = {}'.format(ub.repr2(info, nl=-1)))

        Ignore:
            import xdev
            globals().update(xdev.get_func_kwargs(self.random_category))
        """
        data, mask, kpts = self._from_elem(cname, chip, size=size)
        info = self._package_info(cname, data, mask, kpts, xy_offset, dims,
                                  newstyle=newstyle)
        return info

    def _todo_refactor_geometric_info(self, cname, xy_offset, dims):
        """

        This function is used to populate kpts and sseg information in the
        autogenerated coco dataset before rendering. It is redundant with other
        functionality.

        TODO: rectify with _from_elem

        Example:
            >>> self = CategoryPatterns.coerce(['superstar'])
            >>> dims = (64, 64)
            >>> cname = 'superstar'
            >>> xy_offset = None
            >>> self._todo_refactor_geometric_info(cname, xy_offset, dims)
        """
        elem_func = self._category_to_elemfunc[cname]
        x = max(dims)
        # x = int(2 ** np.floor(np.log2(x)))
        elem, kpts_yx = elem_func(x)

        size = tuple(map(int, dims[::-1]))

        if kpts_yx is not None:
            kp_catnames = list(kpts_yx.keys())
            xy = np.array([yx[::-1] for yx in kpts_yx.values()])
            kpts = kwimage.Points(xy=xy, class_idxs=np.arange(len(xy)),
                                  classes=kp_catnames)
            sf = np.array(size) / np.array(elem.shape[0:2][::-1])
            kpts = kpts.scale(sf)
        else:
            kpts = None
            # center
            kpts = kwimage.Points(xy=np.array([]))
            # kpts = kwimage.Points(xy=np.array([[.5, .5]]))
            kpts = kpts.scale(size)

        template = cv2.resize(elem, size).astype(np.float32)
        mask = (template > 0.05).astype(np.uint8)
        sseg = kwimage.Mask(mask, 'c_mask').to_multi_polygon()

        if xy_offset is not None:
            sseg = sseg.translate(xy_offset, output_dims=dims)
            kpts = kpts.translate(xy_offset, output_dims=dims)

        info = {
            'segmentation': sseg,
            'kpts': kpts,
        }
        return info

    def _package_info(self, cname, data, mask, kpts, xy_offset, dims,
                      newstyle):
        """ packages data from _from_elem into coco-like annotation """
        import kwimage

        if newstyle:
            segmentation = kwimage.Mask(mask, 'c_mask').to_multi_polygon()
        else:
            segmentation = kwimage.Mask(mask, 'c_mask').to_array_rle()

        if xy_offset is not None:
            segmentation = segmentation.translate(xy_offset, output_dims=dims)
            kpts = kpts.translate(xy_offset, output_dims=dims)

        if not newstyle:
            try:
                segmentation = segmentation.to_mask().to_bytes_rle()
                segmentation.data['counts'] = segmentation.data['counts'].decode('utf8')
            except Exception:
                segmentation = segmentation.to_mask().to_array_rle()

        if newstyle:
            segmentation = segmentation.to_coco('new')
            keypoints = kpts.to_coco('new')
        else:
            # old style keypoints
            import kwarray
            keypoints = kwarray.ArrayAPI.tolist(kpts.to_coco('orig'))
            segmentation = segmentation.data

        info = {
            'name': cname,
            'data': data,
            'segmentation': segmentation,
            'keypoints': keypoints,
        }
        return info

    def _from_elem(self, cname, chip, size=None):
        """
        Example:
            >>> # hack to allow chip to be None
            >>> chip = None
            >>> size = (32, 32)
            >>> cname = 'superstar'
            >>> self = CategoryPatterns.coerce()
            >>> self._from_elem(cname, chip, size)
        """
        elem_func = self._category_to_elemfunc[cname]

        if chip is None:
            assert size is not None
            x = max(size)
        else:
            size = tuple(map(int, chip.shape[0:2][::-1]))
            x = max(chip.shape[0:2])

        # x = int(2 ** np.floor(np.log2(x)))
        elem, kpts_yx = elem_func(x)

        if kpts_yx is not None:
            kp_catnames = list(kpts_yx.keys())
            xy = np.array([yx[::-1] for yx in kpts_yx.values()])
            kpts = kwimage.Points(xy=xy, class_idxs=np.arange(len(xy)),
                                  classes=kp_catnames)
            sf = np.array(size) / np.array(elem.shape[0:2][::-1])
            kpts = kpts.scale(sf)
        else:
            kpts = None
            # center
            kpts = kwimage.Points(xy=np.array([]))
            # kpts = kwimage.Points(xy=np.array([[.5, .5]]))
            kpts = kpts.scale(size)

        template = cv2.resize(elem, size).astype(np.float32)
        fg_intensity = np.float32(self.fg_intensity)
        fg_scale = np.float32(self.fg_scale)

        if chip is not None:
            fgdata = kwarray.standard_normal(chip.shape, std=fg_scale,
                                             mean=fg_intensity, rng=self.rng,
                                             dtype=np.float32)
            fgdata = np.clip(fgdata , 0, 1, out=fgdata)
            fga = kwimage.ensure_alpha_channel(fgdata, alpha=template)
            data = kwimage.overlay_alpha_images(fga, chip, keepalpha=False)
        else:
            data = None
        mask = (template > 0.05).astype(np.uint8)
        return data, mask, kpts


def star(a, dtype=np.uint8):
    """Generates a star shaped structuring element.

    Much faster than skimage.morphology version
    """

    if a == 1:
        bfilter = np.zeros((3, 3), dtype)
        bfilter[:] = 1
        return bfilter

    m = 2 * a + 1
    n = a // 2
    selem_square = np.zeros((m + 2 * n, m + 2 * n), dtype=np.uint8)
    selem_square[n: m + n, n: m + n] = 1

    c = (m + 2 * n - 1) // 2
    if True:
        # We can do this much faster with opencv
        b = (m + 2 * n) - 1
        vertices = np.array([
            [0, c],
            [c, 0],
            [b, c],
            [c, b],
            [0, c],
        ])
        pts = vertices.astype(int)[:, None, :]
        mask = np.zeros_like(selem_square)
        mask = cv2.fillConvexPoly(mask, pts, color=1)
        selem_rotated = mask
    else:
        from skimage.morphology.convex_hull import convex_hull_image
        selem_rotated = np.zeros((m + 2 * n, m + 2 * n), dtype=np.float32)
        selem_rotated[0, c] = selem_rotated[-1, c] = 1
        selem_rotated[c, 0] = selem_rotated[c, -1] = 1
        selem_rotated = convex_hull_image(selem_rotated).astype(int)

    selem = np.add(selem_square, selem_rotated, out=selem_square)
    selem[selem > 0] = 1

    return selem.astype(dtype)


class Rasters:
    @staticmethod
    def superstar():
        """
        test data patch

        Ignore:
            >>> kwplot.autompl()
            >>> patch = Rasters.superstar()
            >>> data = np.clip(kwimage.imscale(patch, 2.2), 0, 1)
            >>> kwplot.imshow(data)

        """
        (_, i, O) = 0, 1.0, .5
        patch = np.array([
            [_, _, _, _, _, _, _, O, O, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, O, i, i, O, _, _, _, _, _, _],
            [_, _, _, _, _, _, O, i, i, O, _, _, _, _, _, _],
            [_, _, _, _, _, O, i, i, i, i, O, _, _, _, _, _],
            [O, O, O, O, O, O, i, i, i, i, O, O, O, O, O, O],
            [O, i, i, i, i, i, i, i, i, i, i, i, i, i, i, O],
            [_, O, i, i, i, i, O, i, i, O, i, i, i, i, O, _],
            [_, _, O, i, i, i, O, i, i, O, i, i, i, O, _, _],
            [_, _, _, O, i, i, O, i, i, O, i, i, O, _, _, _],
            [_, _, _, O, i, i, i, i, i, i, i, i, O, _, _, _],
            [_, _, O, i, i, i, i, i, i, i, i, i, i, O, _, _],
            [_, _, O, i, i, i, i, i, i, i, i, i, i, O, _, _],
            [_, O, i, i, i, i, i, O, O, i, i, i, i, i, O, _],
            [_, O, i, i, i, O, O, _, _, O, O, i, i, i, O, _],
            [O, i, i, O, O, _, _, _, _, _, _, O, O, i, i, O],
            [O, O, O, _, _, _, _, _, _, _, _, _, _, O, O, O]])

        keypoints_yx = {
            'left_eye': [7.5, 6.5],
            'right_eye': [7.5, 9.5],
        }
        return patch, keypoints_yx

    @staticmethod
    def eff():
        """
        test data patch

        Ignore:
            >>> kwplot.autompl()
            >>> eff = kwimage.draw_text_on_image(None, 'F', (0, 1), valign='top')
            >>> patch = Rasters.eff()
            >>> data = np.clip(kwimage.imscale(Rasters.eff(), 2.2), 0, 1)
            >>> kwplot.imshow(data)

        """
        (_, O) = 0, 1.0
        patch = np.array([
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, O, O, O, O, O, O, O, O, O, O, O, O, O, O, _, _],
            [_, _, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, _],
            [_, _, _, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, _],
            [_, _, _, O, O, O, O, O, O, O, O, O, O, O, O, O, O, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, O, O, O, O, O, O, O, _, _, _, _, _, _],
            [_, _, O, O, O, O, O, O, O, O, O, O, O, O, O, _, _, _, _, _],
            [_, _, O, O, O, O, O, O, O, O, O, O, O, O, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, O, O, O, O, O, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, O, O, O, _, _, _, _, _, _, _, _, _, _, _, _, _, _]])

        keypoints_yx = {
            'top_tip': (2.5, 18.5),
            'mid_tip': (12.5, 14.5),
            'bot_tip': (23.0, 4.5),
        }
        return patch, keypoints_yx

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/demo/toypatterns.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
