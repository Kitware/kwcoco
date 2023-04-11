"""
These items were split out of coco_dataset.py which is becoming too big

These are helper data structures used to do things like auto-increment ids,
recycle ids, do renaming, extend sortedcontainers etc...
"""
import sortedcontainers


class _NextId(object):
    """
    Helper class to tracks unused ids for new items
    """

    def __init__(self, parent):
        self.parent = parent
        self.unused = {
            'categories': None,
            'images': None,
            'annotations': None,
            'videos': None,
        }

    def _update_unused(self, key):
        """ Scans for what the next safe id can be for ``key`` """
        item_list = self.parent.dataset[key]
        max_id = max(item['id'] for item in item_list) if item_list else 0
        next_id = max(max_id + 1, len(item_list))
        self.unused[key] = next_id

    def get(self, key):
        """ Get the next safe item id for ``key`` """
        if self.unused[key] is None:
            self._update_unused(key)
        new_id = self.unused[key]
        self.unused[key] += 1
        return new_id


class _ID_Remapper(object):
    """
    Helper to recycle ids for unions.

    For each dataset we create a mapping between each old id and a new id.  If
    possible and reuse=True we allow the new id to match the old id.  After
    each dataset is finished we mark all those ids as used and subsequent
    new-ids cannot be chosen from that pool.

    Args:
        reuse (bool): if True we are allowed to reuse ids
            as long as they haven't been used before.

    Example:
        >>> video_trackids = [[1, 1, 3, 3, 200, 4], [204, 1, 2, 3, 3, 4, 5, 9]]
        >>> self = _ID_Remapper(reuse=True)
        >>> for tids in video_trackids:
        >>>     new_tids = [self.remap(old_tid) for old_tid in tids]
        >>>     self.block_seen()
        >>>     print('new_tids = {!r}'.format(new_tids))
        new_tids = [1, 1, 3, 3, 200, 4]
        new_tids = [204, 205, 2, 206, 206, 207, 5, 9]
        >>> #
        >>> self = _ID_Remapper(reuse=False)
        >>> for tids in video_trackids:
        >>>     new_tids = [self.remap(old_tid) for old_tid in tids]
        >>>     self.block_seen()
        >>>     print('new_tids = {!r}'.format(new_tids))
        new_tids = [0, 0, 1, 1, 2, 3]
        new_tids = [4, 5, 6, 7, 7, 8, 9, 10]
    """
    def __init__(self, reuse=False):
        self.blocklist = set()
        self.mapping = dict()
        self.reuse = reuse
        self._nextid = 0

    def remap(self, old_id):
        """
        Convert a old-id into a new-id. If self.reuse is True then we will
        return the same id if it hasn't been blocked yet.
        """
        if old_id in self.mapping:
            new_id = self.mapping[old_id]
        else:
            if not self.reuse or old_id in self.blocklist:
                # We cannot reuse the old-id
                new_id = self.next_id()
            else:
                # We can reuse the old-id
                new_id = old_id
                if isinstance(old_id, int) and old_id >= self._nextid:
                    self._nextid = old_id + 1
            self.mapping[old_id] = new_id
        return new_id

    def block_seen(self):
        """
        Mark all seen ids as unable to be used.
        Any ids sent to remap will now generate new ids.
        """
        self.blocklist.update(self.mapping.values())
        self.mapping = dict()

    def next_id(self):
        """ Generate a new id that hasnt been used yet """
        next_id = self._nextid
        self._nextid += 1
        return next_id


class UniqueNameRemapper(object):
    """
    helper to ensure names will be unique by appending suffixes

    Example:
        >>> from kwcoco.coco_dataset import *  # NOQA
        >>> self = UniqueNameRemapper()
        >>> assert self.remap('foo') == 'foo'
        >>> assert self.remap('foo') == 'foo_v001'
        >>> assert self.remap('foo') == 'foo_v002'
        >>> assert self.remap('foo_v001') == 'foo_v003'
    """
    def __init__(self):
        import re
        self._seen = set()
        self.suffix_pat = re.compile(r'(.*)_v(\d+)')

    def remap(self, name):
        suffix_pat = self.suffix_pat
        match = suffix_pat.match(name)
        if match:
            prefix, _num = match.groups()
            num = int(_num)
        else:
            prefix = name
            num = 0
        while name in self._seen:
            num += 1
            name = '{}_v{:03d}'.format(prefix, num)
        self._seen.add(name)
        return name


# Defined as a global for pickle
def _lut_image_frame_index(imgs, gid):
    return imgs[gid]['frame_index']


# backwards compat for pickles
_lut_frame_index = _lut_image_frame_index


def _lut_annot_frame_index(imgs, anns, aid):
    return imgs[anns[aid]['image_id']]['frame_index']


class SortedSet(sortedcontainers.SortedSet):
    def __repr__(self):
        """Return string representation of sorted set.

        ``ss.__repr__()`` <==> ``repr(ss)``

        :return: string representation
        """
        type_name = type(self).__name__
        return '{0}({1!r})'.format(type_name, list(self))


# Do not use.
# Just exist for backwards compatability with older pickeled data.
SortedSetQuiet = SortedSet


def _delitems(items, remove_idxs, thresh=750):
    """
    Args:
        items (List): list which will be modified
        remove_idxs (List[int]): integers to remove (MUST BE UNIQUE)
    """
    if len(remove_idxs) > thresh:
        # Its typically faster to just make a new list when there are
        # lots and lots of items to remove.
        keep_idxs = sorted(set(range(len(items))) - set(remove_idxs))
        newlist = [items[idx] for idx in keep_idxs]
        items[:] = newlist
    else:
        # However, when there are a few hundred items to remove, del is faster.
        for idx in sorted(remove_idxs, reverse=True):
            del items[idx]


def _load_and_postprocess(data, loader, postprocess, **loadkw):
    # Helper for CocoDataset.load_multiple
    dset = loader(data, **loadkw)
    if postprocess is not None:
        dset = postprocess(dset)
    return dset


def _image_corruption_check(fpath, only_shape=False):
    import kwimage
    from os.path import exists
    info = {'fpath': fpath}
    if not exists(fpath):
        info['failed'] = True
        info['error'] = 'does not exist'
    else:
        try:
            if only_shape:
                kwimage.load_image_shape(fpath)
            else:
                kwimage.imread(fpath)
            info['failed'] = False
        except Exception as ex:
            err = str(ex)
            info['failed'] = True
            info['error'] = err
    return info
