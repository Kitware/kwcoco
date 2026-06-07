from __future__ import annotations
from os.path import join
import ubelt as ub
import warnings
from typing import Any
from kwcoco._dataset_components.json_backend import json_r, json_w


class MixinCocoHashing:
    """
    Mixin for creating and maintaining hashids (i.e. content identifiers)
    """

    def _build_hashid(self, hash_pixels: bool = False, verbose: int = 0) -> str:
        """
        Construct a hash that uniquely identifies the state of this dataset.

        Args:
            hash_pixels (bool):
                If False the image data is not included in the hash, which can
                speed up computation, but is not 100% robust. Defaults to
                False.

            verbose (int): verbosity level

        Returns:
            str: the hashid

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            ...
            >>> # Shorten hashes for readability
            >>> import ubelt as ub
            >>> walker = ub.IndexableWalker(self.hashid_parts)
            >>> for path, val in walker:
            >>>     if isinstance(val, str):
            >>>         walker[path] = val[0:8]
            >>> # Note: this may change in different versions of kwcoco
            >>> print('self.hashid_parts = ' + ub.urepr(self.hashid_parts))
            >>> print('self.hashid = {!r}'.format(self.hashid[0:8]))
            self.hashid_parts = {
                'annotations': {
                    'json': 'c1d1b9c3',
                    'num': 11,
                },
                'images': {
                    'pixels': '88e37cc3',
                    'json': '9b8e8be3',
                    'num': 3,
                },
                'categories': {
                    'json': '82d22e00',
                    'num': 8,
                },
            }
            self.hashid = 'bf69bf15'

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            >>> self.hashid_parts
            >>> # Test that when we modify the dataset only the relevant
            >>> # hashid parts are recomputed.
            >>> orig = self.hashid_parts['categories']['json']
            >>> self.add_category('foobar')
            >>> assert 'categories' not in self.hashid_parts
            >>> self.hashid_parts
            >>> self.hashid_parts['images']['json'] = 'should not change'
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            >>> assert self.hashid_parts['categories']['json']
            >>> assert self.hashid_parts['categories']['json'] != orig
            >>> assert self.hashid_parts['images']['json'] == 'should not change'
        """
        # Construct nested container that we will populate with hashable
        # info corresponding to each type of data that we track.
        from collections import OrderedDict

        hashid_parts = self.hashid_parts
        if hashid_parts is None:
            hashid_parts = OrderedDict()

        # TODO: hashid for videos? Maybe?

        # Ensure hashid_parts has the proper root structure
        # TODO: rely on subet of SPEC_KEYS
        parts = ['annotations', 'images', 'categories']
        for part in parts:
            if not hashid_parts.get(part, None):
                hashid_parts[part] = OrderedDict()

        rebuild_parts = []
        reuse_parts = []

        gids = None
        if hash_pixels:
            if not hashid_parts['images'].get('pixels', None):
                gids = sorted(self.imgs.keys())  # type: ignore
                gpaths = [
                    join(self.bundle_dpath, gname)  # type: ignore
                    for gname in self.images(gids).lookup('file_name')  # type: ignore
                ]
                gpath_sha512s = [
                    ub.hash_file(gpath, hasher='sha512')
                    for gpath in ub.ProgIter(
                        gpaths, desc='hashing images', verbose=verbose
                    )
                ]
                hashid_parts['images']['pixels'] = ub.hash_data(gpath_sha512s)
                rebuild_parts.append('images.pixels')
            else:
                reuse_parts.append('images.pixels')

        # Hash individual components
        with ub.Timer(label='hash coco parts', verbose=verbose > 1):
            # Dumping annots to json takes the longest amount of time
            # However, its faster than hashing the data directly
            def _ditems(d):
                return (
                    list(d.items()) if isinstance(d, OrderedDict) else sorted(d.items())
                )
                # return sorted(d.items())

            if not hashid_parts['annotations'].get('json', None):
                aids = sorted(self.anns.keys())  # type: ignore
                _anns_ordered = (self.anns[aid] for aid in aids)  # type: ignore
                anns_ordered = [_ditems(ann) for ann in _anns_ordered]
                try:
                    anns_text = json_w.dumps(anns_ordered)
                except TypeError:
                    if __debug__:
                        for ann in anns_ordered:
                            try:
                                json_w.dumps(ann)
                            except TypeError:
                                print('FAILED TO ENCODE ann = {!r}'.format(ann))
                                break
                    raise

                hashid_parts['annotations']['json'] = ub.hash_data(
                    anns_text, hasher='sha512'
                )
                hashid_parts['annotations']['num'] = len(aids)
                rebuild_parts.append('annotations.json')
            else:
                reuse_parts.append('annotations.json')

            if not hashid_parts['images'].get('json', None):
                if gids is None:
                    gids = sorted(self.imgs.keys())  # type: ignore
                imgs_text = json_w.dumps([_ditems(self.imgs[gid]) for gid in gids])  # type: ignore
                hashid_parts['images']['json'] = ub.hash_data(
                    imgs_text, hasher='sha512'
                )
                hashid_parts['images']['num'] = len(gids)
                rebuild_parts.append('images.json')
            else:
                reuse_parts.append('images.json')

            if not hashid_parts['categories'].get('json', None):
                cids = sorted(self.cats.keys())  # type: ignore
                cats_text = json_w.dumps([_ditems(self.cats[cid]) for cid in cids])  # type: ignore
                hashid_parts['categories']['json'] = ub.hash_data(
                    cats_text, hasher='sha512'
                )
                hashid_parts['categories']['num'] = len(cids)
                rebuild_parts.append('categories.json')
            else:
                reuse_parts.append('categories.json')

            # TODO:
            # if not hashid_parts['tracks'].get('json', None):
            #     cids = sorted(self.index.tracks.keys())
            #     cats_text = json_w.dumps(
            #         [_ditems(self.index.tracks[cid]) for cid in cids])
            #     hashid_parts['tracks']['json'] = ub.hash_data(
            #         cats_text, hasher='sha512')
            #     hashid_parts['tracks']['num'] = len(cids)
            #     rebuild_parts.append('tracks.json')
            # else:
            #     reuse_parts.append('tracks.json')

        if verbose > 1:
            if reuse_parts:
                print('Reused hashid_parts: {}'.format(reuse_parts))
                print('Rebuilt hashid_parts: {}'.format(rebuild_parts))

        hashid = ub.hash_data(hashid_parts)
        self.hashid = hashid
        self.hashid_parts = hashid_parts
        return hashid

    def _invalidate_hashid(self, parts: Any = None) -> Any:
        """
        Called whenever the coco dataset is modified. It is possible to specify
        which parts were modified so unmodified parts can be reused next time
        the hash is constructed.

        TODO:
            - [ ] Rename to _notify_modification --- or something like that
        """
        self._state['was_modified'] = True  # type: ignore
        self.hashid = None
        if parts is not None and self.hashid_parts is not None:
            for part in parts:
                self.hashid_parts.pop(part, None)
        else:
            self.hashid_parts = None

    def _cached_hashid(self) -> Any:
        """
        Under Construction.

        The idea is to cache the hashid when we are sure that the dataset was
        loaded from a file and has not been modified.  We can record the
        modification time of the file (because we know it hasn't changed in
        memory), and use that as a key to the cache. If the modification time
        on the file is different than the one recorded in the cache, we know
        the cache could be invalid, so we recompute the hashid.

        TODO:
            - [ ] This is reasonably stable, elevate this to a public API function.
        """
        cache_miss = True
        enable_cache = self._state['was_loaded'] and not self._state['was_modified']  # type: ignore
        # print(f'[kwcoco-dct-cache] self._state={self._state}')
        if enable_cache:
            coco_fpath = ub.Path(self.fpath)  # type: ignore
            enable_cache = coco_fpath.exists()

        # print(f'[kwcoco-dct-cache] enable_cache={enable_cache}')
        if enable_cache:
            cache_dpath = coco_fpath.parent / '_cache'
            cache_fname = coco_fpath.name + '.hashid.cache'
            hashid_sidecar_fpath = cache_dpath / cache_fname
            # print(f'[kwcoco-dct-cache] hashid_sidecar_fpath={hashid_sidecar_fpath}')
            # Generate current lookup key
            fpath_stat = coco_fpath.stat()
            status_key = {
                'st_size': fpath_stat.st_size,
                'st_mtime': fpath_stat.st_mtime,
            }
            if hashid_sidecar_fpath.exists():
                cached_data = json_r.loads(hashid_sidecar_fpath.read_text())
                if cached_data['status_key'] == status_key:
                    self.hashid = cached_data['hashid']
                    self.hashid_parts = cached_data['hashid_parts']
                    cache_miss = False

        # print(f'[kwcoco-dct-cache] cache_miss={cache_miss}')
        if cache_miss:
            self._build_hashid()
            if enable_cache:
                hashid_cache_data = {
                    'hashid': self.hashid,
                    'hashid_parts': self.hashid_parts,
                    'status_key': status_key,
                }
                try:
                    hashid_sidecar_fpath.parent.ensuredir()
                    hashid_sidecar_fpath.write_text(json_w.dumps(hashid_cache_data))
                except PermissionError as ex:
                    warnings.warn(f'Cannot write a cached hashid: {repr(ex)}')
        return self.hashid

    @classmethod
    def _cached_hashid_for(cls, fpath: Any) -> Any:
        """
        Lookup the cached hashid for a kwcoco json file if it exists.
        """
        coco_fpath = ub.Path(fpath)
        enable_cache = coco_fpath.exists()

        if enable_cache:
            cache_dpath = coco_fpath.parent / '_cache'
            cache_fname = coco_fpath.name + '.hashid.cache'
            hashid_sidecar_fpath = cache_dpath / cache_fname
            # Generate current lookup key
            fpath_stat = coco_fpath.stat()
            status_key = {
                'st_size': fpath_stat.st_size,
                'st_mtime': fpath_stat.st_mtime,
            }
            if hashid_sidecar_fpath.exists():
                cached_data = json_r.loads(hashid_sidecar_fpath.read_text())
                if cached_data['status_key'] == status_key:
                    cached_data['hashid']
                    cached_data['hashid_parts']
                    cache_miss = False

        if cache_miss:
            # TODO: fixme?
            # self._build_hashid()
            # if enable_cache:
            #     hashid_cache_data = {
            #         'hashid': self.hashid,
            #         'hashid_parts': self.hashid_parts,
            #         'status_key': status_key,
            #     }
            #     hashid_sidecar_fpath.parent.ensuredir()
            #     hashid_sidecar_fpath.write_text(json_w.dumps(hashid_cache_data))
            raise Exception('cache miss')
