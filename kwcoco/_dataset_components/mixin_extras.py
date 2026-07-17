from __future__ import annotations

from typing import Any, Callable, Iterable, TYPE_CHECKING
from os.path import dirname, exists, join, relpath
import os
import ubelt as ub

from kwcoco._helpers import _image_corruption_check

if TYPE_CHECKING:
    from os import PathLike


class MixinCocoExtras:
    """
    Misc functions for coco
    """

    def _tree(self) -> None:
        """
        developer helper

        Ignore:
            import kwcoco
            self = kwcoco.CocoDataset.demo('photos')
            print('self = {!r}'.format(self))
            self._tree()

            self = kwcoco.CocoDataset.demo('shapes1')
            print(self.imgs[1])
            print('self = {!r}'.format(self))
            self._tree()

            self = kwcoco.CocoDataset.demo('vidshapes2')
            print('self = {!r}'.format(self))
            print(self.imgs[1])
            self._tree()
        """
        print('self.bundle_dpath = {!r}'.format(self.bundle_dpath))
        print('self.fpath = {!r}'.format(self.fpath))
        print('self.tag = {!r}'.format(self.tag))  # type: ignore
        if not self.bundle_dpath:
            raise Exception('We dont have a bundle')
        ub.cmd('tree {}'.format(self.bundle_dpath), verbose=2)

    def _dataset_id(self) -> str:
        """
        A human interpretable name that can be used to uniquely identify the
        dataset.

        Note:
            This function is currently subject to change.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> print(self._dataset_id())
            >>> self = kwcoco.CocoDataset.demo('vidshapes8')
            >>> print(self._dataset_id())
            >>> self = kwcoco.CocoDataset()
            >>> print(self._dataset_id())
        """
        hashid = self._cached_hashid()  # type: ignore
        if self.fpath is None:
            raise Exception('Dataset doesnt have a fpath')
        coco_fpath = ub.Path(self.fpath)
        dset_id = '_'.join([coco_fpath.parent.stem, coco_fpath.stem, hashid[0:8]])
        return dset_id

    def _ensure_imgsize(
        self, workers: int = 0, verbose: int = 1, fail: bool = False
    ) -> dict:
        """
        Populate the imgsize field if it does not exist.

        Args:
            workers (int): number of workers for parallel
                processing.

            verbose (int): verbosity level

            fail (bool): if True, raises an exception if
               anything size fails to load.

        Returns:
            Dict: summary: contains an overview of what was done / what failed.
                Key/Value pairs are:
                    message (str): overview of what happened
                    bad_images (List[dict]):
                        a list of "bad" image dictionaries where the size could
                        not be determined. Typically these are corrupted images
                        and should be removed.
                    num_attempts (int): number of size read jobs attempted
                    num_modified (int): number of images changed
                    num_failed (int): number of images changed

        Example:
            >>> # Normal case
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> summary = self._ensure_imgsize()
            >>> assert len(summary['bad_images']) == 0
            >>> assert self.imgs[1]['width'] == 512
            >>> assert self.imgs[2]['width'] == 328
            >>> assert self.imgs[3]['width'] == 256

            >>> # Fail cases
            >>> self = kwcoco.CocoDataset()
            >>> self.add_image('does-not-exist.jpg')
            >>> summary = self._ensure_imgsize()
            >>> assert len(summary['bad_images']) == 1
            >>> import pytest
            >>> with pytest.raises(Exception):
            >>>     self._ensure_imgsize(fail=True)
        """
        summary = {}
        bad_images = []
        if any(
            'width' not in img or 'height' not in img for img in self.dataset['images']  # type: ignore
        ):
            import kwimage

            if self.tag:  # type: ignore
                desc = 'populate imgsize for ' + self.tag  # type: ignore
            else:
                desc = 'populate imgsize for untagged coco dataset'

            pool = ub.JobPool('thread', max_workers=workers)
            bundle_dpath = self.bundle_dpath
            for img in ub.ProgIter(
                self.dataset['images'], verbose=verbose, desc='submit image size jobs'  # type: ignore
            ):
                auxiliary = img.get('auxiliary', img.get('assets', []))
                for obj in [img] + auxiliary:
                    fname = obj['file_name']
                    if fname is not None:
                        gpath = join(bundle_dpath, fname)
                        if 'width' not in obj or 'height' not in obj:
                            job = pool.submit(kwimage.load_image_shape, gpath)
                            setattr(job, 'obj', obj)

            summary['num_attempts'] = len(pool)
            num_modified = 0

            for job in pool.as_completed(desc=desc, progkw={'verbose': verbose}):
                try:
                    h, w = job.result()[0:2]
                except Exception:
                    if fail:
                        raise
                    bad_images.append(job.obj)  # type: ignore
                else:
                    job.obj['width'] = w  # type: ignore
                    job.obj['height'] = h  # type: ignore
                    num_modified += 1
            summary['num_modified'] = num_modified
            summary['num_failed'] = len(bad_images)
            summary['message'] = 'attempted to populated width/height'
        else:
            summary['message'] = 'all images had width/height keys, no work to be done'

        summary['bad_images'] = bad_images
        return summary

    def _ensure_image_data(self, gids: list | dict | int | None = None, verbose: int = 1) -> None:
        """
        Download data from image ``url`` / ``coco_url`` fields if specified.

        Args:
            gids (List[int | dict] | dict | int | None): subset of image ids
                or image dictionaries to download. If unspecified, checks all
                missing images in the dataset.
        """

        def _gen_missing_imgs() -> Iterable[dict]:
            for img in self.dataset['images']:  # type: ignore
                gpath = join(self.bundle_dpath, img['file_name'])
                if not exists(gpath):
                    yield img

        def _coerce_img_iter(gids: list | dict | int | None) -> Iterable[dict]:
            imgs: Iterable[dict]
            if gids is None:
                imgs = _gen_missing_imgs()
            elif isinstance(gids, dict):
                imgs = [gids]
            else:
                gid_or_imgs: list[int | dict] = list(  # type: ignore
                    gids if ub.iterable(gids) else [gids]  # type: ignore
                )
                if gid_or_imgs and isinstance(gid_or_imgs[0], dict):
                    imgs = gid_or_imgs  # type: ignore
                else:
                    imgs = list(ub.take(self.imgs, gid_or_imgs))  # type: ignore
            return imgs

        def _has_download_permission(_HAS_PERMISSION=[False]) -> bool:
            # TODO: remove argv-global behavior from this library helper. The
            # CLI should pass an explicit option sourced from argparse instead.
            if not _HAS_PERMISSION[0] or ub.argflag(('-y', '--yes')):
                try:
                    from rich.prompt import Confirm

                    ans = Confirm.ask('is it ok to download?')
                    if ans:
                        _HAS_PERMISSION[0] = True
                except ImportError:
                    ans = input('is it ok to download? (enter y for yes)')
                    if ans in ['yes', 'y']:
                        _HAS_PERMISSION[0] = True
            return _HAS_PERMISSION[0]

        imgs: Iterable[dict] = _coerce_img_iter(gids)

        for img in ub.ProgIter(imgs, desc='ensure image data', verbose=verbose):
            url: str | None = img.get('url')
            if url is None:
                url = img.get('coco_url')
            if url is not None:
                if _has_download_permission():
                    gpath = join(self.bundle_dpath, img['file_name'])
                    ub.ensuredir(dirname(gpath))
                    ub.grabdata(url, gpath)
                else:
                    raise Exception('no permission, abort')
            else:
                raise Exception('missing image, but no url')

    def missing_images(
        self, check_aux: bool = True, verbose: int = 0
    ) -> list[tuple[int, str, int]]:
        """
        Check for images that don't exist

        Args:
            check_aux (bool):
                if specified also checks auxiliary images

            verbose (int): verbosity level

        Returns:
            List[Tuple[int, str, int]]: bad indexes and paths and ids
        """
        bad_paths = []
        img_enum = enumerate(self.dataset['images'])  # type: ignore
        for index, img in ub.ProgIter(
            img_enum,
            total=len(self.dataset['images']),  # type: ignore
            desc='check for missing images',
            verbose=verbose,
        ):
            gid = img.get('id', None)
            fname = img.get('file_name', None)
            if fname is not None:
                gpath = join(self.bundle_dpath, fname)
                if not exists(gpath):
                    bad_paths.append((index, gpath, gid))

            if check_aux:
                for aux in img.get('auxiliary', img.get('assets', [])):
                    gpath = join(self.bundle_dpath, aux['file_name'])
                    if not exists(gpath):
                        bad_paths.append((index, gpath, gid))
        return bad_paths

    def corrupted_images(
        self, check_aux: bool = True, verbose: int = 0, workers: int = 0
    ) -> list[tuple[int, str, int]]:
        """
        Check for images that don't exist or can't be opened

        Args:
            check_aux (bool):
                if specified also checks auxiliary images
            verbose (int): verbosity level
            workers (int): number of background workers

        Returns:
            List[Tuple[int, str, int]]: bad indexes and paths and ids
        """
        bad_paths = []

        jobs = ub.JobPool(mode='process', max_workers=workers)

        img_enum = enumerate(self.dataset['images'])  # type: ignore
        for index, img in ub.ProgIter(
            img_enum,
            total=len(self.dataset['images']),  # type: ignore
            desc='submit corruption checks',
            verbose=verbose,
        ):
            gid = img.get('id', None)
            fname = img.get('file_name', None)
            if fname is not None:
                gpath = join(self.bundle_dpath, fname)
                job = jobs.submit(_image_corruption_check, gpath)
                job.input_info = (index, gpath, gid)  # type: ignore

            if check_aux:
                for aux in img.get('auxiliary', img.get('assets', [])):
                    gpath = join(self.bundle_dpath, aux['file_name'])
                    job = jobs.submit(_image_corruption_check, gpath)
                    job.input_info = (index, gpath, gid)  # type: ignore

        for job in jobs.as_completed(
            desc='check corrupted images', progkw={'verbose': verbose}
        ):
            info = job.result()
            if info['failed']:
                bad_paths.append(job.input_info)  # type: ignore

        return bad_paths

    def normalize_category_ids(
        self, start_id: int | None = None, order: list[str] | str | None = None
    ) -> Any:
        """
        Reassign category IDs to be consecutive integers starting at `start_id`.

        Optionally reorders category definitions to make the order canonical,
        sorted, or front-loaded based on a custom list of category names.

        Args:
            start_id (int | None):
                If specified, reassign each category consecutive ids starting
                with this id.

            order (List[str] | str | None): Optional control over category
                ordering. If:
                    - None: preserves original order.
                    - List[str]: category names to prioritize in that order.
                    - "sort": sorts categories alphabetically by name.

        Example:
            >>> import kwcoco
            >>> import ubelt as ub
            >>> self = kwcoco.CocoDataset.demo('vidshapes8', verbose=0)
            >>> print('Original categories:')
            >>> print(ub.urepr(self.dataset['categories'], nl=0))
            Original categories:
            [{'id': 1, 'name': 'star'}, {'id': 2, 'name': 'superstar'}, {'id': 3, 'name': 'eff'}]
            >>> # 1. Reindex starting at 0
            >>> self.normalize_category_ids(start_id=0)
            >>> print('After start_id=0:')
            >>> print(ub.urepr(self.dataset['categories'], nl=0))
            After start_id=0:
            [{'id': 0, 'name': 'star'}, {'id': 1, 'name': 'superstar'}, {'id': 2, 'name': 'eff'}]
            >>> # 2. Sort categories alphabetically
            >>> self.normalize_category_ids(order='sort', start_id=1)
            >>> print('After order="sort", start_id=1:')
            >>> print(ub.urepr(self.dataset['categories'], nl=0))
            After order="sort", start_id=1:
            [{'id': 1, 'name': 'eff'}, {'id': 2, 'name': 'star'}, {'id': 3, 'name': 'superstar'}]
            >>> # 3. Front-load a custom class order
            >>> self.normalize_category_ids(order=['superstar'], start_id=100)
            >>> print('After order=["superstar"], start_id=100:')
            >>> print(ub.urepr(self.dataset['categories'], nl=0))
            After order=["superstar"], start_id=100:
            [{'id': 100, 'name': 'superstar'}, {'id': 101, 'name': 'eff'}, {'id': 102, 'name': 'star'}]
        """
        # Get existing categories as a list of dicts
        orig_categories = self.dataset['categories']  # type: ignore

        # Determine category ordering
        if order is None:
            new_catname_order = None
        elif isinstance(order, str):
            if order == 'sort':
                new_catname_order = sorted(c['name'] for c in orig_categories)
            else:
                raise KeyError(f'Unknown ordering method: {order}')
        else:
            # Assume order: List[str]
            orig_order = ub.oset([cat['name'] for cat in orig_categories])
            suffix_order = list(orig_order - order)  # type: ignore
            new_catname_order = order + suffix_order

        if new_catname_order is not None:
            # If we are reordering the categoryes, build the new list
            new_categories = [
                self.index.name_to_cat[name] for name in new_catname_order  # type: ignore
            ]
        else:
            new_categories = orig_categories

        if start_id is not None:
            # If requested, replace category ids and build new ID map
            cat_id_map = {}
            for new_id, cat in enumerate(new_categories, start=start_id):
                old_id = cat['id']
                cat_id_map[old_id] = new_id
                cat['id'] = new_id

            # Update annotations
            for ann in self.dataset['annotations']:  # type: ignore
                old_cid = ann.get('category_id', None)
                if old_cid is not None:
                    ann['category_id'] = cat_id_map[old_cid]

        if new_catname_order is not None:
            # If we reordered the categories, replace the dataset category list
            self.dataset['categories'][:] = new_categories  # type: ignore

        # Invalidate and rebuild index
        self._invalidate_hashid()  # type: ignore
        self.rebuild_index()  # type: ignore

    def rename_categories(
        self,
        mapper: dict | Callable,
        rebuild: bool = True,
        merge_policy: str = 'ignore',
    ) -> None:
        """
        Rename categories with a potentially coarser categorization.

        Args:
            mapper (dict | Callable): maps old names to new names.
                If multiple names are mapped to the same category, those
                categories will be merged.

            merge_policy (str):
                How to handle multiple categories that map to the same name.
                Can be update or ignore.

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self.rename_categories({'astronomer': 'person',
            >>>                         'astronaut': 'person',
            >>>                         'mouth': 'person',
            >>>                         'helmet': 'hat'})
            >>> assert 'hat' in self.name_to_cat
            >>> assert 'helmet' not in self.name_to_cat
            >>> # Test merge case
            >>> self = kwcoco.CocoDataset.demo()
            >>> mapper = {
            >>>     'helmet': 'rocket',
            >>>     'astronomer': 'rocket',
            >>>     'human': 'rocket',
            >>>     'mouth': 'helmet',
            >>>     'star': 'gas'
            >>> }
            >>> self.rename_categories(mapper)
        """
        old_cats = self.dataset['categories']  # type: ignore

        # Ignore identity mappings
        mapper = {k: v for k, v in mapper.items() if k != v}  # type: ignore

        # Perform checks to determine what bookkeeping needs to be done
        orig_cnames = {cat['name'] for cat in old_cats}
        src_cnames = set(mapper.keys())
        dst_cnames = set(mapper.values())

        bad_cnames = src_cnames - orig_cnames
        if bad_cnames:
            raise ValueError(
                'The following categories to not exist: {}'.format(bad_cnames)
            )

        has_orig_merges = dst_cnames.intersection(orig_cnames)
        has_src_merges = dst_cnames.intersection(src_cnames)
        has_dup_dst = set(ub.find_duplicates(mapper.values()).keys())

        has_merges = has_orig_merges or has_src_merges or has_dup_dst

        if not has_merges:
            # In the simple case we are just changing the labels, so
            # nothing special needs to happen.
            for key, value in mapper.items():
                for cat in self.dataset['categories']:  # type: ignore
                    if cat['name'] == key:
                        cat['name'] = value
        else:
            # Remember the original categories
            orig_cats = {cat['name']: cat for cat in old_cats}

            # Remember all annotations of the original categories
            src_cids = [self.index.name_to_cat[c]['id'] for c in src_cnames]  # type: ignore
            src_cname_to_aids = {
                c: self.index.cid_to_aids[cid] for c, cid in zip(src_cnames, src_cids)  # type: ignore
            }

            # Track which srcs each dst is constructed from
            dst_to_srcs = ub.invert_dict(mapper, unique_vals=False)

            # Mark unreferenced cats for removal
            rm_cnames = src_cnames - dst_cnames

            # Mark unseen cats for addition
            add_cnames = dst_cnames - orig_cnames

            # Mark renamed cats for update
            update_cnames = dst_cnames - add_cnames

            # Populate new category information
            new_cats = {}
            for dst in sorted(dst_cnames):
                # Combine information between existing categories that are
                # being collapsed into a single category.
                srcs = dst_to_srcs[dst]
                new_cat = {}
                if len(srcs) <= 1:
                    # in the case of 1 source, then there is no merger
                    for src in srcs:
                        new_cat.update(orig_cats[src])
                elif merge_policy == 'update':
                    # When there are multiple sources then union all
                    # original source attributes.
                    # Note: the order of srcs is arbitrary so we sort
                    # to make the update order consistent
                    for src in sorted(srcs):
                        new_cat.update(orig_cats[src])
                elif merge_policy == 'ignore':
                    # When there are multiple sources then ignore all
                    # original source attributes.
                    pass
                else:
                    # There may be better merge policies that should be
                    # implemented
                    raise KeyError('Unknown merge_policy={}'.format(merge_policy))

                new_cat['name'] = dst
                new_cat.pop('id', None)
                new_cats[dst] = new_cat

            # Apply category deltas

            # Remove category names that were renamed
            self.remove_categories(rm_cnames, keep_annots=True)  # type: ignore

            # Update any existing categories that were merged
            for cname in sorted(update_cnames):
                if merge_policy == 'ignore':
                    new_cats[cname] = dict(orig_cats[cname])
                elif merge_policy == 'update':
                    new_cat = new_cats[cname]
                    orig_cat = orig_cats[cname]
                    # Only update name and non-existing information
                    # TODO: check for conflicts?
                    new_info = ub.dict_diff(new_cat, orig_cat)
                    orig_cat.update(new_info)
                else:
                    raise KeyError('Unknown merge_policy={}'.format(merge_policy))

            # Add new category names that were created
            for cname in sorted(add_cnames):
                self.add_category(**new_cats[cname])  # type: ignore

            # Apply the annotation deltas
            for src, aids in src_cname_to_aids.items():
                cat = self.name_to_cat[mapper[src]]  # type: ignore
                cid = cat['id']
                for aid in aids:
                    self.anns[aid]['category_id'] = cid  # type: ignore

            # force rebuild if any annotations we changed
            if src_cname_to_aids:
                rebuild = True

        if rebuild:
            self.rebuild_index()  # type: ignore
        else:
            self.index.clear()  # type: ignore
        self._invalidate_hashid()  # type: ignore

    def _ensure_json_serializable(self) -> None:
        # inplace convert any ndarrays to lists
        # TODO: use the kwutil version
        from kwcoco.util.util_json import ensure_json_serializable

        _ = ensure_json_serializable(self.dataset, verbose=1)  # type: ignore

    def _aspycoco(self) -> Any:
        """
        Converts to the official pycocotools.coco.COCO object

        TODO:
            - [ ] Maybe expose as a public API?
        """
        from pycocotools import coco

        pycoco = coco.COCO()
        pycoco.dataset = self.dataset  # type: ignore
        pycoco.createIndex()
        return pycoco

    def reroot(
        self,
        new_root: str | PathLike | None = None,
        old_prefix: str | None = None,
        new_prefix: str | None = None,
        absolute: bool = False,
        check: bool = True,
        safe: bool = True,
        verbose: int = 0,
    ) -> None:
        """
        Modify the prefix of the image/data paths onto a new image/data root.

        Args:
            new_root (str | PathLike | None):
                New image root. If unspecified the current
                ``self.bundle_dpath`` is used. If old_prefix and new_prefix are
                unspecified, they will attempt to be determined based on the
                current root (which assumes the file paths exist at that root)
                and this new root.  Defaults to None.

            old_prefix (str | None):
                If specified, removes this prefix from file names.
                This also prevents any inferences that might be made via
                "new_root". Defaults to None.

            new_prefix (str | None):
                If specified, adds this prefix to the file names.
                This also prevents any inferences that might be made via
                "new_root". Defaults to None.

            absolute (bool):
                if True, file names are stored as absolute paths, otherwise
                they are relative to the new image root. Defaults to False.

            check (bool):
                if True, checks that the images all exist.
                Defaults to True.

            safe (bool):
                if True, does not overwrite values until all checks pass.
                Defaults to True.

            verbose (int):
                verbosity level, default=0.

        CommandLine:
            xdoctest -m kwcoco.coco_dataset MixinCocoExtras.reroot

        TODO:
            - [ ] Incorporate maximum ordered subtree embedding?

        Example:
            >>> # xdoctest: +REQUIRES(module:rich)
            >>> import kwcoco
            >>> import ubelt as ub
            >>> import rich
            >>> def report(dset):
            >>>     gid = 1
            >>>     abs_fpath = ub.Path(dset.get_image_fpath(gid))
            >>>     rel_fpath = dset.index.imgs[gid]['file_name']
            >>>     color = 'green' if abs_fpath.exists() else 'red'
            >>>     print(ub.color_text(f'abs_fpath = {abs_fpath!r}', color))
            >>>     print(f'rel_fpath = {rel_fpath!r}')
            >>> dset = self = kwcoco.CocoDataset.demo()
            >>> # Change base relative directory
            >>> bundle_dpath = ub.expandpath('~')
            >>> rich.print('ORIG self.imgs = {}'.format(ub.urepr(self.imgs, nl=1)))
            >>> rich.print('ORIG dset.bundle_dpath = {!r}'.format(dset.bundle_dpath))
            >>> rich.print('NEW(1) bundle_dpath       = {!r}'.format(bundle_dpath))
            >>> # Test relative reroot
            >>> rich.print('[blue] --- 1. RELATIVE REROOT ---')
            >>> self.reroot(bundle_dpath, verbose=3)
            >>> report(self)
            >>> rich.print('NEW(1) self.imgs = {}'.format(ub.urepr(self.imgs, nl=1)))
            >>> if not ub.WIN32:
            >>>     assert self.imgs[1]['file_name'].startswith('.cache')
            >>> # Test absolute reroot
            >>> rich.print('[blue] --- 2. ABSOLUTE REROOT ---')
            >>> self.reroot(absolute=True, verbose=3)
            >>> rich.print('NEW(2) self.imgs = {}'.format(ub.urepr(self.imgs, nl=1)))
            >>> assert self.imgs[1]['file_name'].startswith(bundle_dpath)

            >>> # Switch back to relative paths
            >>> rich.print('[blue] --- 3. ABS->REL REROOT ---')
            >>> self.reroot()
            >>> rich.print('NEW(3) self.imgs = {}'.format(ub.urepr(self.imgs, nl=1)))
            >>> if not ub.WIN32:
            >>>     assert self.imgs[1]['file_name'].startswith('.cache')

        Example:
            >>> # demo with auxiliary data
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo('shapes8', aux=True)
            >>> bundle_dpath = ub.expandpath('~')
            >>> print(self.imgs[1]['file_name'])
            >>> print(self.imgs[1]['auxiliary'][0]['file_name'])
            >>> self.reroot(new_root=bundle_dpath)
            >>> print(self.imgs[1]['file_name'])
            >>> print(self.imgs[1]['auxiliary'][0]['file_name'])
            >>> if not ub.WIN32:
            >>>     assert self.imgs[1]['file_name'].startswith('.cache')
            >>>     assert self.imgs[1]['auxiliary'][0]['file_name'].startswith('.cache')

        Ignore:
            See ~/code/kwcoco/dev/devcheck/devcheck_reroot.py
        """
        cur_bundle_dpath = self.bundle_dpath
        if cur_bundle_dpath == '':
            cur_bundle_dpath = '.'
        new_bundle_dpath = self.bundle_dpath if new_root is None else new_root
        new_bundle_dpath = os.fspath(new_bundle_dpath)

        # Find the the prefix that to make new relative paths work.
        rel_prefix = os.path.relpath(cur_bundle_dpath, new_bundle_dpath)

        if absolute:
            new_bundle_dpath = os.fspath(ub.Path(new_bundle_dpath).absolute())

        if verbose > 1:
            print('kwcoco.reroot')
            print(' * new_bundle_dpath = {}'.format(ub.urepr(new_bundle_dpath, nl=1)))
            print(' * cur_bundle_dpath = {}'.format(ub.urepr(cur_bundle_dpath, nl=1)))
            print(f' * old_prefix={old_prefix}')
            print(f' * new_prefix={new_prefix}')
            print(f' * absolute={absolute}')
            print(f' * safe={safe}')
            print(f' * check={check}')
            if verbose > 2:
                # First assets
                if len(self.dataset['images']):  # type: ignore
                    from kwcoco.coco_image import CocoImage

                    first_image = CocoImage(self.dataset['images'][0], dset=self)  # type: ignore
                    first_image_filepaths = list(
                        first_image.iter_image_filepaths(with_bundle=False)
                    )
                    print(
                        ' * first_image_filepaths = {}'.format(
                            ub.urepr(first_image_filepaths, nl=1)
                        )
                    )
                else:
                    print('No images to reroot')

        def _normalize_prefix(path):
            # removes leading ./ on paths
            path = os.fspath(path)
            while path.startswith('./'):
                path = path[2:]
            return path

        def _reroot_path(file_name):
            """Reroot a single file"""

            # TODO: can this logic be cleaned up, its difficult to follow and
            # describe. The gist is we want reroot to modify the file names of
            # the images based on the new root, unless we are explicitly given
            # information about new and old prefixes. Can we do better than
            # this?
            _old_fname = file_name
            file_name = _normalize_prefix(file_name)
            cur_gpath = join(cur_bundle_dpath, file_name)

            if old_prefix is not None:
                if file_name.startswith(old_prefix):
                    file_name = relpath(file_name, old_prefix)
                elif cur_gpath.startswith(old_prefix):
                    file_name = relpath(cur_gpath, old_prefix)

            # If we don't specify new or old prefixes, but new_root was
            # specified then modify relative file names to be correct with
            # respect to this new root (assuming the previous root was also
            # valid)
            DO_OLD_CHECK = 0
            if DO_OLD_CHECK:
                if old_prefix is None and new_prefix is None:
                    # This is not a good check, fails if we want to
                    # do a relative reroot outside of the original dataset
                    if new_bundle_dpath is not None and cur_gpath.startswith(
                        new_bundle_dpath
                    ):
                        file_name = relpath(cur_gpath, new_bundle_dpath)

            if new_prefix is not None:
                file_name = join(new_prefix, file_name)

            if absolute:
                new_file_name = join(new_bundle_dpath, file_name)
            else:
                new_file_name = join(rel_prefix, file_name)
                if os.path.isabs(new_file_name):
                    # Handle absolute -> relative case
                    new_file_name = relpath(new_file_name, new_bundle_dpath)

            if check:
                new_gpath = join(new_bundle_dpath, new_file_name)
                if not exists(new_gpath):
                    print('ERROR')
                    print(' * file_name (old) = {!r}'.format(_old_fname))
                    print(' * file_name (new) = {!r}'.format(file_name))
                    print(' * cur_gpath = {!r}'.format(cur_gpath))
                    print(' * new_gpath = {!r}'.format(new_gpath))
                    print(' * new_file_name = {!r}'.format(new_file_name))
                    print(' * cur_bundle_dpath = {!r}'.format(cur_bundle_dpath))
                    print(' * new_bundle_dpath = {!r}'.format(new_bundle_dpath))
                    raise Exception('Image does not exist: {!r}'.format(new_gpath))
            return new_file_name

        # from kwcoco.util import util_reroot

        num_images = len(self.imgs)  # type: ignore
        enable_prog = verbose > 0

        if safe:
            # First compute all new values in memory but don't overwrite
            gid_to_new = {}
            prog = ub.ProgIter(
                self.imgs.items(),  # type: ignore
                total=num_images,
                enabled=enable_prog,
                desc='prepare reroot',
            )
            for gid, img in prog:
                gname = img.get('file_name', None)
                try:
                    new = {}
                    if gname is not None:
                        new['file_name'] = _reroot_path(gname)
                    if 'auxiliary' in img:
                        new['auxiliary'] = aux_fname = []
                        for aux in img.get('auxiliary', []):
                            aux_fname.append(_reroot_path(aux['file_name']))
                    if 'assets' in img:
                        new['assets'] = aux_fname = []
                        for aux in img.get('assets', []):
                            aux_fname.append(_reroot_path(aux['file_name']))
                    gid_to_new[gid] = new
                except Exception:
                    # img_repr = ub.urepr(img)
                    asset_fpaths = list(
                        self.coco_image(img['id']).iter_image_filepaths()  # type: ignore
                    )
                    raise Exception(
                        'Failed to reroot gid={} with fpaths={}'.format(
                            img['id'], asset_fpaths
                        )
                    )

            # Overwrite old values
            prog = ub.ProgIter(
                gid_to_new.items(),
                total=num_images,
                enabled=enable_prog,
                desc='finalize reroot',
            )
            for gid, new in prog:
                img = self.imgs[gid]  # type: ignore
                img['file_name'] = new.get('file_name', None)
                if 'auxiliary' in new:
                    for aux_fname, aux in zip(new['auxiliary'], img['auxiliary']):
                        aux['file_name'] = aux_fname
                if 'assets' in new:
                    for aux_fname, aux in zip(new['assets'], img['assets']):
                        aux['file_name'] = aux_fname
        else:
            prog = ub.ProgIter(
                self.imgs.values(),  # type: ignore
                total=num_images,
                enabled=enable_prog,
                desc='rerooting',
            )
            for img in prog:
                try:
                    gname = img.get('file_name', None)
                    if gname is not None:
                        img['file_name'] = _reroot_path(gname)
                    for aux in img.get('auxiliary', []):
                        aux['file_name'] = _reroot_path(aux['file_name'])
                    for aux in img.get('assets', []):
                        aux['file_name'] = _reroot_path(aux['file_name'])
                except Exception:
                    # img_repr = ub.urepr(img)
                    # raise Exception('Failed to reroot img={}'.format(ub.urepr(img)))
                    asset_fpaths = list(
                        self.coco_image(img['id']).iter_image_filepaths()  # type: ignore
                    )
                    raise Exception(
                        'Failed to reroot gid={} with fpaths={}'.format(
                            img['id'], asset_fpaths
                        )
                    )

        if self.index:  # type: ignore
            # Only need to recompute the self.index.file_name_to_img
            # We dont have to invalidate everything
            # FIXME: the index should have some method for doing this
            # (ideally lazilly)
            self.index.file_name_to_img = {  # type: ignore
                img['file_name']: img
                for img in self.index.imgs.values()  # type: ignore
                if img.get('file_name', None) is not None
            }

        self.bundle_dpath = new_bundle_dpath
        return self  # type: ignore

    @property
    def data_root(self) -> PathLike | None:
        """In the future we will deprecate data_root for bundle_dpath"""
        return self.bundle_dpath  # type: ignore

    @data_root.setter
    def data_root(self, value: Any) -> None:
        self.bundle_dpath = value if value is None else os.fspath(value)

    @property
    def img_root(self) -> PathLike | None:
        """In the future we will deprecate img_root for bundle_dpath"""
        return self.bundle_dpath  # type: ignore

    @img_root.setter
    def img_root(self, value: Any) -> None:
        self.bundle_dpath = value

    @property
    def data_fpath(self) -> PathLike | None:
        """data_fpath is an alias of fpath"""
        return self.fpath

    @data_fpath.setter
    def data_fpath(self, value: Any) -> None:
        self.fpath = value
