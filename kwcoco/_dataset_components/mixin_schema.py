from __future__ import annotations

import importlib.util
import logging
from typing import Any, TypeVar
import warnings

import ubelt as ub


logger = logging.getLogger(__name__)


MixinCocoSchemaT = TypeVar('MixinCocoSchemaT', bound='MixinCocoSchema')


@ub.memoize
def _cached_fast_coco_schema_validator():
    """Compile the optional fastjsonschema validator once per process."""
    import fastjsonschema
    from kwcoco.coco_schema import COCO_SCHEMA

    # Match the historical jsonschema behavior. The reference validation call
    # does not apply schema defaults and does not install a FormatChecker.
    return fastjsonschema.compile(
        COCO_SCHEMA,
        use_default=False,
        use_formats=False,
    )


@ub.memoize
def _cached_jsonschema_validator():
    """Build the reference validator once instead of rechecking the schema."""
    from jsonschema.validators import validator_for
    from kwcoco.coco_schema import COCO_SCHEMA

    validator_cls = validator_for(COCO_SCHEMA)
    validator_cls.check_schema(COCO_SCHEMA)
    return validator_cls(COCO_SCHEMA)


def _fastjsonschema_available() -> bool:
    """Check if the optional accelerated schema backend can be imported."""
    return importlib.util.find_spec('fastjsonschema') is not None


def _resolve_schema_backend(schema_backend: str) -> str:
    """Resolve ``auto`` while keeping explicit backend requests strict."""
    valid_backends = {'auto', 'jsonschema', 'fastjsonschema'}
    if schema_backend not in valid_backends:
        raise KeyError(
            'Unknown schema_backend={!r}. Expected one of: {}'.format(
                schema_backend, sorted(valid_backends)
            )
        )

    if schema_backend == 'auto':
        if _fastjsonschema_available():
            return 'fastjsonschema'
        return 'jsonschema'

    if schema_backend == 'fastjsonschema' and not _fastjsonschema_available():
        raise ModuleNotFoundError(
            "schema_backend='fastjsonschema' requires the optional "
            "'fastjsonschema' package. Install kwcoco with its optional "
            "dependencies or install fastjsonschema directly."
        )
    return schema_backend


def _validate_schema_jsonschema(data):
    """Return the jsonschema validation error, or ``None`` on success."""
    import jsonschema

    try:
        _cached_jsonschema_validator().validate(data)
    except jsonschema.exceptions.ValidationError as ex:
        return ex
    return None


def _validate_schema_fastjsonschema(data):
    """Return the fastjsonschema validation error, or ``None`` on success."""
    import fastjsonschema

    try:
        _cached_fast_coco_schema_validator()(data)
    except fastjsonschema.JsonSchemaException as ex:
        return ex
    return None


def _contains_tuple(data) -> bool:
    """Detect Python tuples that fastjsonschema treats as JSON arrays.

    The reference ``jsonschema`` backend intentionally recognizes Python
    lists, but not tuples, as JSON arrays. ``fastjsonschema`` recognizes both.
    Normal serialized KWCoco data is list/dict based, so this linear scan is a
    small compatibility tax on the accelerated path and only triggers the
    slower validator when tuple semantics could matter.
    """
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, tuple):
            return True
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


class MixinCocoSchema:
    """
    Methods for validating and conforming dataset structure.
    """

    # Supplied by CocoDataset, the concrete class that owns this mixin.
    dataset: dict[str, Any]

    def conform(self: MixinCocoSchemaT, **config: Any) -> MixinCocoSchemaT:
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

                verbose (int): verbosity / progress level. Defaults to 0.

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

        verbose = config.get('verbose', 0)

        if config.get('ensure_imgsize', True):
            self._ensure_imgsize(
                workers=config.get('workers', 8), verbose=verbose
            )  # type: ignore

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
            for ann in ub.ProgIter(
                self.dataset['annotations'], desc='update anns', verbose=verbose
            ):  # type: ignore
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
                                ),
                                stacklevel=2,
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
                                ),
                                stacklevel=2,
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
                self.dataset['annotations'],
                desc='update orig coco anns',
                verbose=verbose,  # type: ignore
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
                                warnings.warn(
                                    'Converting to a legacy polygon removes holes',
                                    UserWarning,
                                    stacklevel=2,
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

                schema_backend (default='auto'):
                    JSON-schema implementation to use. ``'auto'`` uses the
                    optional substantially faster ``fastjsonschema`` backend
                    when it is installed and otherwise falls back to the
                    required ``jsonschema`` backend. If the fast validator
                    rejects data in auto mode, validation is retried with
                    ``jsonschema`` to preserve compatibility with values the
                    reference backend accepts. Explicit ``'jsonschema'`` and
                    ``'fastjsonschema'`` modes are also available.

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

                img_attrs (default=False):
                if truthy, check that image attributes contain width and
                height entries. If 'warn', then warn if they do not exist.
                If 'error', then fail.

                annot_attrs (default=False):
                    if truthy, check that annotations contain bbox entries.
                    This is opt-in because bbox is not required by the kwcoco
                    schema. If 'warn', then warn if values are missing. If
                    'error', then fail.

                verbose (default=0): verbosity flag. Level 1 reports
                    validation problems. Level 2 also reports individual
                    validation checks and progress.

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
        verbose = config.get('verbose', 0)
        fastfail = config.get('fastfail', 0)

        def _error(msg):
            if verbose:
                print(msg)
            if fastfail:
                raise Exception(msg)
            result['errors'].append(msg)

        def _warn(msg):
            # Do not also print warnings. The warnings subsystem already gives
            # callers filtering / routing controls, and printing here produced
            # duplicate user-facing diagnostics.
            warnings.warn(msg, UserWarning, stacklevel=4)
            result['warnings'].append(msg)

        if config.get('schema', True):
            schema_backend = config.get('schema_backend', 'auto')
            resolved_schema_backend = _resolve_schema_backend(schema_backend)

            logger.debug(
                'Validate json-schema with %s (requested=%s)',
                resolved_schema_backend,
                schema_backend,
            )
            if verbose > 1:
                if schema_backend == resolved_schema_backend:
                    print(f'Validate json-schema with {resolved_schema_backend}')
                else:
                    print(
                        'Validate json-schema with {} '
                        '(auto resolved to {})'.format(
                            schema_backend, resolved_schema_backend
                        )
                    )

            if resolved_schema_backend == 'jsonschema':
                schema_error = _validate_schema_jsonschema(dset.dataset)
            else:
                has_tuple = schema_backend == 'auto' and _contains_tuple(dset.dataset)
                if has_tuple:
                    # fastjsonschema treats tuples as arrays whereas the
                    # historical jsonschema backend does not. Skip the fast
                    # validator when that semantic difference can matter.
                    schema_error = _validate_schema_jsonschema(dset.dataset)
                else:
                    schema_error = _validate_schema_fastjsonschema(dset.dataset)

                if schema_error is not None and schema_backend == 'auto' and not has_tuple:
                    # The two Python implementations intentionally differ in
                    # a few non-JSON-native type details. fastjsonschema is
                    # stricter for some numeric classes (e.g. numpy integer
                    # values used as JSON numbers). Normal JSON-native
                    # manifests avoid this case. On a fast rejection, pay the
                    # slower reference-validator cost to preserve historical
                    # jsonschema acceptance behavior.
                    reference_error = _validate_schema_jsonschema(dset.dataset)
                    if reference_error is None:
                        if schema_error is not None:
                            logger.debug(
                                'fastjsonschema rejected data accepted by '
                                'jsonschema; accepted via compatibility fallback'
                            )
                        if schema_error is not None and verbose > 1:
                            print(
                                'fastjsonschema rejected data accepted by '
                                'jsonschema; accepted via compatibility fallback'
                            )
                        schema_error = None
                    else:
                        # Preserve the reference backend's error behavior for
                        # genuinely invalid data in auto mode.
                        schema_error = reference_error

            if schema_error is not None:
                if hasattr(schema_error, 'absolute_path'):
                    logger.debug(
                        'schema validation error absolute_path=%r',
                        schema_error.absolute_path,
                    )
                if verbose > 1 and hasattr(schema_error, 'absolute_path'):
                    print(f'err.absolute_path={schema_error.absolute_path}')
                msg = 'Failed to validate schema: {}'.format(str(schema_error))
                _error(msg)

        def _check_unique(dset, table_key, col_key, required=True):
            logger.debug('Check table %r has unique %r', table_key, col_key)
            if verbose > 1:
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
            logger.debug('Check table %r entries have attr %r', table_key, col_key)
            if verbose > 1:
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
            logger.debug(
                'Check subtable %r / %r entries have attr %r',
                table_key,
                subtable_keys,
                col_key,
            )
            if verbose > 1:
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

        if config.get('annot_attrs', False):
            required = config.get('annot_attrs', False) == 'error'
            _check_attrs(dset, table_key='annotations', col_key='bbox', required=required)

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
            missing = dset.missing_images(  # type: ignore
                check_aux=True, verbose=max(verbose - 1, 0)
            )
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
                check_aux=True,
                verbose=max(verbose - 1, 0),
                workers=config.get('workers', 0),
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

