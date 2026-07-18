from __future__ import annotations
import ubelt as ub
import typing
from collections import defaultdict


if typing.TYPE_CHECKING:
    from numpy import ndarray
    from typing import Any
    from delayed_image.channel_spec import ChannelSpec


class MixinCocoDraw:
    """
    Matplotlib / display functionality
    """

    def draw_image(self, gid: int, channels: ChannelSpec | None = None) -> ndarray:
        """
        Use kwimage to draw all annotations on an image and return the pixels
        as a numpy array.

        Args:
            gid (int): image id to draw
            channels (delayed_image.channel_spec.ChannelSpec): the channel to draw on

        Returns:
            ndarray: canvas

        SeeAlso
            :func:`kwcoco.coco_dataset.MixinCocoDraw.draw_image`
            :func:`kwcoco.coco_dataset.MixinCocoDraw.show_image`

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes8')
            >>> canvas = self.draw_image(1)
            >>> # Now you can dump the annotated image to disk / whatever
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
        """
        import kwimage

        # Load the raw image pixels
        coco_img = self.coco_image(gid)  # type: ignore
        delayed = coco_img.imdelay(space='image', channels=channels)

        if delayed.channels is not None and delayed.channels.numel() > 3:
            import warnings

            warnings.warn(
                'Requested drawing more than 3 channels. Only picking first 3'
            )
            first_channels = coco_img.imdelay(space='image').channels.fuse()[0:3]
            delayed = delayed.take_channels(first_channels)

        canvas = delayed.finalize()
        canvas = _normalize_intensity_if_needed(canvas)
        # OpenCV's drawing primitives require an 8-bit image. In particular,
        # OpenCV 5 rejects float canvases when rendering annotation labels.
        canvas = kwimage.ensure_uint255(canvas)

        # Get annotation IDs from this image
        aids = self.index.gid_to_aids[gid]  # type: ignore
        # Grab relevant annotation dictionaries
        anns = [self.anns[aid] for aid in aids]  # type: ignore
        # Transform them into a kwimage.Detections datastructure

        dset = self
        try:
            classes = dset.object_categories()  # type: ignore
        except Exception:
            classes = list(dset.name_to_cat.keys())  # type: ignore
        try:
            kp_classes = dset.keypoint_categories()  # type: ignore
        except Exception:
            # hack
            anns = [ann.copy() for ann in anns]
            for ann in anns:
                ann.pop('keypoints', None)
            kp_classes = None
        cats = dset.dataset['categories']  # type: ignore
        # dets = kwimage.Detections.from_coco_annots(anns, dset=self)
        dets = kwimage.Detections.from_coco_annots(
            anns, classes=classes, cats=cats, kp_classes=kp_classes
        )

        # Segmentation drawing uses alpha compositing, which may promote an
        # otherwise uint8 canvas to float. OpenCV 5 requires uint8 input for
        # label text, so finish the segmentation pass first, restore the
        # display dtype, and then draw boxes, labels, and keypoints.
        if dets.data.get('segmentations', None) is not None:
            canvas = dets.draw_on(
                canvas, boxes=False, kpts=False, labels=False
            )
            canvas = kwimage.ensure_uint255(canvas)
            canvas = dets.draw_on(canvas, sseg=False)
        else:
            canvas = dets.draw_on(canvas)
        return canvas

    def show_image(
        self,
        gid: int | None = None,
        aids: list | None = None,
        aid: int | None = None,
        channels: Any | None = None,
        setlim: None | str = None,
        **kwargs: Any,
    ) -> Any:
        """
        Use matplotlib to show an image with annotations overlaid

        Args:
            gid (int | None):
                image id to show
            aids (list | None):
                annotation ids to highlight within the image
            aid (int | None):
                a specific annotation id to focus on. If gid is not give, look
                up gid based on this aid.
            setlim (None | str):
                if 'image' sets the limit to the image extent
            **kwargs:
                show_annots, show_aid, show_catname, show_kpname,
                show_segmentation, title, show_gid, show_filename,
                show_boxes,

        SeeAlso
            :func:`kwcoco.coco_dataset.MixinCocoDraw.draw_image`
            :func:`kwcoco.coco_dataset.MixinCocoDraw.show_image`

        Ignore:
            # Programmatically collect the kwargs for docs generation
            import xinspect
            import kwcoco
            kwargs = xinspect.get_kwargs(kwcoco.CocoDataset.show_image)
            print(ub.urepr(list(kwargs.keys()), nl=1, si=1))

        CommandLine:
            xdoctest -m kwcoco.coco_dataset MixinCocoDraw.show_image --show

        Example:
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes8-msi')
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl(verbose=3)
            >>> # xdoctest: -REQUIRES(--show)
            >>> dset.show_image(gid=1, channels='B8')
            >>> # xdoctest: +REQUIRES(--show)
            >>> kwplot.show_if_requested()
        """
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        import kwimage
        import kwplot
        import numpy as np

        figkw = {
            k: kwargs[k] for k in ['fnum', 'pnum', 'doclf', 'docla'] if k in kwargs
        }
        if figkw:
            kwplot.figure(**figkw)

        if gid is None:
            primary_ann = self.anns[aid]  # type: ignore
            gid = primary_ann['image_id']

        show_all = kwargs.get('show_all', True)
        show_annots = kwargs.get('show_annots', True)
        show_labels = kwargs.get('show_labels', True)

        highlight_aids = set()
        if aid is not None:
            highlight_aids.add(aid)
        if aids is not None:
            highlight_aids.update(aids)

        coco_img = self.coco_image(gid)  # type: ignore
        img = coco_img.img
        aids = self.index.gid_to_aids.get(img['id'], [])  # type: ignore

        # Collect annotation overlays
        colored_segments = defaultdict(list)
        keypoints = []
        rects = []
        texts = []

        sseg_masks = []
        sseg_polys = []

        if show_annots:
            for aid in aids:
                ann = self.anns[aid]  # type: ignore

                if 'keypoints' in ann:
                    cid = ann['category_id']
                    if ann['keypoints'] is not None and len(ann['keypoints']) > 0:
                        # TODO: rely on kwimage.Points to parse multiple format info?
                        kpts_data = ann['keypoints']
                        if isinstance(ub.peek(kpts_data), dict):
                            xys = np.array([p['xy'] for p in kpts_data])
                            isvisible = np.array(
                                [p.get('visible', True) for p in kpts_data]
                            )
                            kpnames = None
                            isvisible = np.array(
                                [p.get('visible', True) for p in kpts_data]
                            )
                        else:
                            try:
                                kpnames = self._lookup_kpnames(cid)  # type: ignore
                            except KeyError:
                                kpnames = None
                            kpts = np.array(ann['keypoints']).reshape(-1, 3)
                            isvisible = kpts.T[2] > 0
                            xys = kpts.T[0:2].T[isvisible]
                    else:
                        kpnames = None
                        xys = None
                else:
                    kpnames = None
                    xys = None

                # Note standard coco bbox is [x,y,width,height]
                if 'bbox' in ann:
                    x1, y1 = ann['bbox'][0:2]
                elif 'line' in ann:
                    x1, y1 = ann['line'][0:2]
                elif 'keypoints' in ann:
                    x1, y1 = xys.min(axis=0)  # type: ignore
                else:
                    raise Exception(f'no bbox, line, or keypoint position: ann={ann}')

                cid = ann.get('category_id', None)
                if cid is not None:
                    cat = self.cats[cid]  # type: ignore
                    catname = cat['name']
                else:
                    cat = None
                    catname = ann.get('category_name', 'None')
                textkw = {
                    'horizontalalignment': 'left',
                    'verticalalignment': 'top',
                    'backgroundcolor': (0, 0, 0, 0.3),
                    'color': 'white',
                    'fontproperties': mpl.font_manager.FontProperties(  # type: ignore
                        size=6, family='monospace'
                    ),
                }
                annot_text_parts = []
                if show_labels:
                    if kwargs.get('show_aid', show_all):
                        annot_text_parts.append('aid={}'.format(aid))
                    if kwargs.get('show_catname', show_all):
                        annot_text_parts.append(catname)
                    if annot_text_parts:
                        annot_text = ' '.join(annot_text_parts)
                        texts.append((x1, y1, annot_text, textkw))

                color = 'orange' if aid in highlight_aids else 'blue'
                if 'obox' in ann:
                    # Oriented bounding box
                    segs = np.array(ann['obox']).reshape(-1, 3)[:, 0:2]
                    for pt1, pt2 in ub.iter_window(segs, wrap=True):
                        colored_segments[color].append([pt1, pt2])
                elif 'bbox' in ann:
                    [x, y, w, h] = ann['bbox']
                    rect = mpl.patches.Rectangle(  # type: ignore
                        (x, y), w, h, facecolor='none', edgecolor=color
                    )
                    rects.append(rect)
                if 'line' in ann:
                    x1, y1, x2, y2 = ann['line']
                    pt1, pt2 = (x1, y1), (x2, y2)
                    colored_segments[color].append([pt1, pt2])
                if 'keypoints' in ann:
                    if xys is not None and len(xys):
                        keypoints.append(xys)
                        if show_labels and kwargs.get('show_kpname', True):
                            if kpnames is not None:
                                for (kp_x, kp_y), kpname in zip(xys, kpnames):
                                    texts.append((kp_x, kp_y, kpname, textkw))

                if ann.get('segmentation', None) is not None and kwargs.get(
                    'show_segmentation', True
                ):
                    sseg = ann['segmentation']
                    # Respect the 'color' attribute of categories
                    if cat is not None:
                        catcolor = cat.get('color', None)
                    else:
                        catcolor = None

                    if catcolor is not None:
                        catcolor = kwplot.Color(catcolor).as01()
                    # TODO: Unify masks and polygons into a kwimage
                    # segmentation class
                    sseg = kwimage.Segmentation.coerce(sseg).data
                    if isinstance(sseg, kwimage.Mask):
                        m = sseg.to_c_mask()
                        sseg_masks.append((m.data, catcolor))
                    else:
                        # TODO: interior
                        multipoly = sseg.to_multi_polygon()
                        for poly in multipoly.data:
                            poly_xys = poly.data['exterior'].data
                            polykw = {}
                            if catcolor is not None:
                                polykw['color'] = catcolor
                            poly = mpl.patches.Polygon(poly_xys, **polykw)  # type: ignore
                            try:
                                # hack
                                poly.area = sseg.to_shapely().area  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            sseg_polys.append(poly)

        # Show image
        avail_channels = coco_img.channels

        if channels is None and avail_channels is not None:
            avail_spec = avail_channels.fuse().normalize()
            num_chans = avail_spec.numel()

            if num_chans == 0:
                raise Exception('no channels!')
            elif num_chans <= 2:
                print('Auto choosing 1 channel')
                channels = avail_spec.parsed[0]
            elif num_chans > 3:
                print('Auto choosing 3 channel')
                sensible_defaults = [
                    'red|green|blue',
                    'r|g|b',
                ]
                chosen = None
                for sensible in sensible_defaults:
                    cand = avail_spec & sensible
                    if cand.numel() == 3:
                        chosen = cand
                        break
                if chosen is None:
                    chosen = avail_spec[0:3]
                channels = chosen

        print('loading image')
        delayed = coco_img.imdelay(space='image', channels=channels)
        np_img = delayed.finalize()
        print('loaded image')

        np_img = kwimage.atleast_3channels(np_img)

        np_img01 = None
        if np_img.max() > 255:
            np_img01 = _normalize_intensity_if_needed(np_img)

        fig = plt.gcf()
        ax = fig.gca()
        ax.cla()

        if sseg_masks:
            if np_img01 is None:
                np_img01 = kwimage.ensure_float01(np_img)
            layers = []
            layers.append(kwimage.ensure_alpha_channel(np_img01))
            distinct_colors = kwplot.Color.distinct(len(sseg_masks))

            for (mask, _catcolor), col in zip(sseg_masks, distinct_colors):
                if _catcolor is not None:
                    col = kwimage.ensure_float01(np.array(_catcolor)).tolist()

                col = np.array(col + [1])[None, None, :]
                alpha_mask = col * mask[:, :, None]
                alpha_mask[..., 3] = mask * 0.5
                layers.append(alpha_mask)

            masked_img = kwimage.overlay_alpha_layers(layers[::-1])
            ax.imshow(masked_img)
        else:
            if np_img01 is not None:
                ax.imshow(np_img01)
            else:
                ax.imshow(np_img)

        title = kwargs.get('title', None)
        if title is None:
            title_parts = []
            if kwargs.get('show_gid', True):
                title_parts.append('gid={}'.format(gid))
            if kwargs.get('show_filename', True):
                title_parts.append(str(img['file_name']))
            title = ' '.join(title_parts)
        if title:
            ax.set_title(title)

        if sseg_polys:
            # print('sseg_polys = {!r}'.format(sseg_polys))
            # hack: show smaller polygons first.
            if len(sseg_polys):
                areas = np.array([getattr(p, 'area', np.inf) for p in sseg_polys])
                sortx = np.argsort(areas)[::-1]
                sseg_polys = list(ub.take(sseg_polys, sortx))

            poly_col = mpl.collections.PatchCollection(  # type: ignore
                sseg_polys, match_original=True, alpha=0.4
            )
            ax.add_collection(poly_col)

        # Show all annotations inside it
        if kwargs.get('show_boxes', True):
            for x1, y1, catname, textkw in texts:
                ax.text(x1, y1, catname, **textkw)  # type: ignore

            for color, segments in colored_segments.items():
                line_col = mpl.collections.LineCollection(  # type: ignore
                    segments, linewidths=2, color=color
                )
                ax.add_collection(line_col)

            rect_col = mpl.collections.PatchCollection(rects, match_original=True)  # type: ignore
            ax.add_collection(rect_col)
            if keypoints:
                xs, ys = np.vstack(keypoints).T
                ax.plot(xs, ys, 'bo')

        if setlim:
            if not isinstance(setlim, str):
                setlim = 'image'

            if setlim == 'image':
                ax.set_xlim(0, img['width'])
                ax.set_ylim(img['height'], 0)
            else:
                raise NotImplementedError(setlim)

        return ax


def _normalize_intensity_if_needed(canvas: Any) -> Any:
    # We really need a function that can adapt to dynamic ranges
    # in a robust way that ensures at least something is visible.
    # Normalize intensity is pretty good, but has common edge cases.
    import kwimage

    max_value = canvas.max()
    min_value = canvas.min()
    if canvas.dtype.kind in ['u', 'i']:
        needs_normalization = min_value < 0 or max_value > 255
    else:
        needs_normalization = min_value < 0 or max_value > 1
    if needs_normalization:
        canvas = kwimage.normalize_intensity(canvas)
    return canvas
