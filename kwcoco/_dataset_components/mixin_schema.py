from __future__ import annotations

from typing import Any
import warnings

import ubelt as ub


class MixinCocoSchema:
    """
    Methods for validating and conforming dataset structure.
    """

    def conform(self, **config: Any):
        """
        Make the COCO file conform a stricter spec, infers attributes where
        possible.

        Corresponds to the ``kwcoco conform`` CLI tool.

        KWArgs:
            **config :

                pycocotools_info (default=True): returns info required by pycocotools

                ensure_imgsize (default=True): ensure image size is populated

                mmlab (default=False): if True tries to convert data
                to be compatible with open-mmlab tooling.

                legacy (default=False): if True tries to convert data
                structures to items compatible with the original
                pycocotools spec

                workers (int): number of parallel jobs for IO tasks

        Returns:
            Self: the inplace modified kwcoco dataset

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('shapes8')
            >>> dset.index.imgs[1].pop('width')
            >>> dset.conform(legacy=True)
            >>> assert 'width' in dset.index.imgs[1]
            >>> assert 'area' in dset.index.anns[1]
        """

        if config.get('ensure_imgsize', True):
            self._ensure_imgsize(workers=config.get('workers', 8))  # type: ignore

            for vidid, gids in self.index.vidid_to_gids.items():  # type: ignore
                if len(gids):
                    # Each image in a video should be the same size (unless we
                    # implement some transform from image to video space)
                    # populate each video with the consistent width/height
                    video = self.index.videos[vidid]  # type: ignore
                    vid_frames = self.images(gids)  # type: ignore
                    width_cand = vid_frames.lookup('width')
                    height_cand = vid_frames.lookup('height')
                    if ub.allsame(height_cand) and ub.allsame(width_cand):
                        video['width'] = width_cand[0]
                        video['height'] = height_cand[0]

        if config.get('pycocotools_info', True):
            for ann in ub.ProgIter(self.dataset['annotations'], desc='update anns'):  # type: ignore
                if 'iscrowd' not in ann:
                    ann['iscrowd'] = False

                if 'ignore' not in ann:
                    ann['ignore'] = ann.get('weight', 1.0) < 0.5

                if 'area' not in ann:
                    # Use segmentation if available
                    if 'segmentation' in ann:
                        try:
                            import kwimage

                            poly = kwimage.MultiPolygon.coerce(ann['segmentation'])
                            ann['area'] = float(poly.to_shapely().area)
                        except Exception:
                            warnings.warn(
                                ub.paragraph(
                                    """
                                Unable to coerce segmentation to a polygon.
                                This may be indicative of a bug in
                                `kwimage.MultiPolygon.coerce` or a misformatted
                                segmentation
                                """
                                )
                            )
                    else:
                        try:
                            x, y, w, h = ann['bbox']
                        except KeyError:
                            warnings.warn(
                                ub.paragraph(
                                    """
                                Unable to add "area" key because an annotation
                                is missing or has a malformed "bbox" entry
                                """
                                )
                            )
                        else:
                            ann['area'] = w * h

        if config.get('mmlab', False):
            # Add support for open-mmlab coco dataset ingestion
            if self.dataset.get('videos', None):  # type: ignore
                for images in self.videos().images:  # type: ignore
                    # mmdet wants the frame_id property
                    for frame_index, img in enumerate(images.objs):
                        img['frame_id'] = frame_index
                # mmdet wants the instance_id property
                for ann in self.dataset['annotations']:  # type: ignore
                    if 'track_id' in ann:
                        ann['instance_id'] = ann['track_id']  # type: ignore

        if config.get('legacy', False):
            # Original MS-COCO represents ``info`` as a single dictionary.
            # KWCOCO also permits a list of metadata / provenance records.
            # Keep that list intact under an extension key rather than
            # flattening records with potentially overlapping keys.
            info = self.dataset.get('info', None)  # type: ignore
            if isinstance(info, list):
                self.dataset['info'] = (  # type: ignore
                    {'kwcoco_info': info} if info else {}
                )
            elif info is None:
                self.dataset['info'] = {}  # type: ignore

            try:
                kpcats = self.keypoint_categories()  # type: ignore
            except Exception:
                kpcats = None
            for ann in ub.ProgIter(
                self.dataset['annotations'], desc='update orig coco anns'  # type: ignore
            ):
                # Use segmentation if available
                if 'segmentation' in ann:
                    # TODO: any original style coco dict is ok, we dont
                    # always need it to be a poly if it is RLE
                    import kwimage

                    poly = kwimage.MultiPolygon.coerce(ann['segmentation'])
                    # Hack, looks like kwimage does not wrap the original
                    # coco polygon with a list, but pycocotools needs that

                    if poly is None:
                        ann.pop('segmentation')
                    else:
                        # import xdev
                        # with xdev.embed_on_exception_context:
                        try:
                            ann['segmentation'] = poly.to_coco(style='orig')
                        except ValueError:
                            REMOVE_HOLES = 1
                            if REMOVE_HOLES:
                                print(
                                    'Warning: converting to legacy polygon is removing holes'
                                )
                                mpoly = poly.to_shapely()
                                from shapely.geometry import Polygon, MultiPolygon

                                fixed_mpoly = MultiPolygon(
                                    [Polygon(p.exterior) for p in mpoly.geoms]
                                )
                                ann['segmentation'] = kwimage.MultiPolygon.from_shapely(
                                    fixed_mpoly
                                ).to_coco(style='orig')
                            else:
                                raise
                            # from numbers import Number
                            # if len(segm) and isinstance(segm[0], Number):
                            #     segm = [segm]

                if 'keypoints' in ann:
                    import kwimage

                    # TODO: these have to be in some defined order for
                    # each category, currently it is arbitrary
                    pts = kwimage.Points.from_coco(ann['keypoints'], classes=kpcats)
                    ann['keypoints'] = pts.to_coco(style='orig')
        return self  # type: ignore

    def validate(self, **config: Any) -> dict:
        """
        Performs checks on this coco dataset.

        Corresponds to the ``kwcoco validate`` CLI tool.

        Args:
            **config :
                schema (default=True): if True, validate the json-schema

                unique (default=True): if True, validate unique secondary keys

                missing (default=True): if True, validate registered files exist

                corrupted (default=False): if True, validate data in registered files

                channels (default=True):
                if True, validate that channels in auxiliary/asset items
                are all unique.

                require_relative (default=False):
                if True, causes validation to fail if paths are
                non-portable, i.e.  all paths must be relative to the
                bundle directory. if>0, paths must be relative to bundle
                root.  if>1, paths must be inside bundle root.

                img_attrs (default='warn'):
                if truthy, check that image attributes contain width and
                height entries. If 'warn', then warn if they do not exist.
                If 'error', then fail.

                verbose (default=1): verbosity flag

                workers (int): number of workers for parallel checks. defaults to 0

                fastfail (default=False): if True raise errors immediately

        Returns:
            dict: result containing keys -
                status (bool): False if any errors occurred
                errors (List[str]): list of all error messages
                missing (List): List of any missing images
                corrupted (List): List of any corrupted images

        SeeAlso:
            :func:`_check_integrity` - performs internal checks

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> import pytest
            >>> with pytest.warns(UserWarning):
            >>>     result = self.validate()
            >>> assert not result['errors']
            >>> assert result['warnings']
        """
        dset = self

        result: dict[str, list[Any]] = {
            'errors': [],
            'warnings': [],
        }
        verbose = config.get('verbose', 1)
        fastfail = config.get('fastfail', 0)

        def _error(msg):
            if verbose:
                print(msg)
            if fastfail:
                raise Exception(msg)
            result['errors'].append(msg)

        def _warn(msg):
            if verbose:
                print(msg)
            warnings.warn(msg, UserWarning)
            result['warnings'].append(msg)

        if config.get('schema', True):
            import jsonschema
            from kwcoco.coco_schema import COCO_SCHEMA

            if verbose:
                print('Validate json-schema')
            try:
                COCO_SCHEMA.validate(dset.dataset)  # type: ignore
            except jsonschema.exceptions.ValidationError as ex:
                err = ex
                print(f'err.absolute_path={err.absolute_path}')
                msg = 'Failed to validate schema: {}'.format(str(err))
                _error(msg)

        def _check_unique(dset, table_key, col_key, required=True):
            if verbose:
                print('Check table {!r} has unique {!r}'.format(table_key, col_key))
            items = dset.dataset.get(table_key, [])
            seen = set()
            num_unset = 0
            for obj in items:
                value = obj.get(col_key, None)
                if value is None:
                    num_unset += 1
                else:
                    if value in seen:
                        msg = 'Duplicate {} {} = {!r}'.format(table_key, col_key, value)
                        _error(msg)
                    else:
                        seen.add(value)
            if num_unset > 0:
                msg = ub.paragraph(
                    """
                    Table {!r} is missing {} / {} values for column {!r}
                    """
                ).format(table_key, num_unset, len(items), col_key)
                if required:
                    _error(msg)
                else:
                    _warn(msg)

        def _check_attrs(dset, table_key, col_key, required=True):
            if verbose:
                print(
                    'Check table {!r} entries have attr {!r}'.format(table_key, col_key)
                )
            items = dset.dataset.get(table_key, [])
            num_unset = 0
            for obj in items:
                value = obj.get(col_key, None)
                if value is None:
                    num_unset += 1
            if num_unset > 0:
                msg = ub.paragraph(
                    """
                    Table {!r} is missing {} / {} values for column {!r}
                    """
                ).format(table_key, num_unset, len(items), col_key)
                if required:
                    _error(msg)
                else:
                    _warn(msg)

        def _check_subtable_attrs(
            dset, table_key, subtable_keys, col_key, required=True
        ):
            if verbose:
                print(
                    'Check subtable {!r} / {!r} entries have attr {!r}'.format(
                        table_key, subtable_keys, col_key
                    )
                )
            items = dset.dataset.get(table_key, [])
            num_unset = 0
            num_subobjs = 0
            for obj in items:
                # hack for asset/auxiliary
                _ = ub.dict_isect(obj, subtable_keys)
                sub_objs = ub.peek(_.values()) if _ else None
                if sub_objs:
                    for sub_obj in sub_objs:
                        num_subobjs += 1
                        value = sub_obj.get(col_key, None)
                        if value is None:
                            num_unset += 1
            if num_unset > 0:
                msg = ub.paragraph(
                    """
                    Subtable {!r}/{!r} is missing {} / {} values for column {!r}
                    """
                ).format(table_key, subtable_keys, num_unset, num_subobjs, col_key)
                if required:
                    _error(msg)
                else:
                    _warn(msg)

        if config.get('unique', True):
            _check_unique(dset, table_key='categories', col_key='name')
            _check_unique(dset, table_key='videos', col_key='name')
            _check_unique(dset, table_key='images', col_key='name', required=False)

            _check_unique(dset, table_key='images', col_key='id')
            _check_unique(dset, table_key='videos', col_key='id')
            _check_unique(dset, table_key='annotations', col_key='id')
            _check_unique(dset, table_key='categories', col_key='id')

        if config.get('img_attrs', False):
            required = config.get('img_attrs', False) == 'error'
            # Refine this?
            _check_attrs(dset, table_key='images', col_key='width', required=required)
            _check_attrs(dset, table_key='images', col_key='height', required=required)
            _check_subtable_attrs(
                dset,
                table_key='images',
                subtable_keys=['asset', 'auxiliary'],
                col_key='width',
                required=required,
            )
            _check_subtable_attrs(
                dset,
                table_key='images',
                subtable_keys=['asset', 'auxiliary'],
                col_key='height',
                required=required,
            )

        if config.get('annot_attrs', True):
            required = config.get('annot_attrs', False) == 'error'
            _check_attrs(dset, table_key='annotations', col_key='bbox', required=False)

        if config.get('channels', True):
            for img in self.dataset.get('images', []):  # type: ignore
                seen = set()
                assets = img.get('auxiliary', []) + img.get('assets', [])

                for obj in assets:
                    channels = obj.get('channels', None)
                    if channels is None:
                        gid = img['id']
                        _error(
                            'Asset in gid={} is missing channels, obj={}'.format(
                                gid, obj
                            )
                        )
                    else:
                        from delayed_image.channel_spec import FusedChannelSpec

                        channels = FusedChannelSpec.coerce(channels)
                        for chan in channels.as_list():
                            if chan in seen:
                                gid = img['id']
                                _error(
                                    'The chan={} is specified more than once in gid={}'.format(
                                        chan, gid
                                    )
                                )
                            seen.add(chan)
        if config.get('missing', True):
            missing = dset.missing_images(check_aux=True, verbose=verbose)  # type: ignore
            if missing:
                msg = ub.paragraph(
                    f"""
                    There are {len(missing)} missing images.
                    The first one is {missing[0][1]!r}.
                    """
                )
                _error(msg)
                result['missing'] = missing

        if config.get('corrupted', False):
            corrupted = dset.corrupted_images(  # type: ignore
                check_aux=True, verbose=verbose, workers=config.get('workers', 0)
            )
            if corrupted:
                msg = ub.paragraph(
                    f"""
                    There are {len(corrupted)} corrupted images.
                    The first one is {corrupted[0][1]!r}.
                    """
                )
                _error(msg)
                result['corrupted'] = corrupted

        if config.get('require_relative', False):
            rr = config.get('require_relative', False)
            for gid in dset.imgs.keys():  # type: ignore
                coco_img = dset.coco_image(gid)  # type: ignore
                for obj in coco_img.iter_asset_objs():
                    fname = ub.Path(obj['file_name'])
                    if rr > 0:
                        if fname.is_absolute():
                            _error('non relative fname = {!r}'.format(fname))
                    if rr > 1:
                        if '..' in fname.parts:
                            _error('non internal fname = {!r}'.format(fname))

        return result

