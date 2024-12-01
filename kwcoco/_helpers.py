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

        # TODO: use a single source of truth for what the top-level tables with
        # ids are.
        self.unused = {
            'categories': None,
            'images': None,
            'annotations': None,
            'videos': None,
            'tracks': None,
        }

    def _update_unused(self, key):
        """ Scans for what the next safe id can be for ``key`` """
        try:
            item_list = self.parent.dataset[key]
            max_id = max(item['id'] for item in item_list) if item_list else 0
            next_id = max(max_id + 1, len(item_list))
        except KeyError:
            # The table doesn't exist, so we can use anything
            next_id = 1
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
        >>> assert 'foo' in self
    """
    def __init__(self):
        import re
        self._seen = set()
        self.suffix_pat = re.compile(r'(.*)_v(\d+)')

    def __contains__(self, name):
        return name in self._seen

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


class _CategoryID_Remapper:
    """
    Helper for a category union that re-uses ids whenever possible.

    Given an old category dictionary, calling :func:`remap` will return a new
    category dictionary with updated properties if necessary.

    Example:
        >>> from kwcoco._helpers import _CategoryID_Remapper
        >>> self = _CategoryID_Remapper()
        >>> self.remap({'name': 'cat5', 'id': 5})
        >>> self.remap({'name': 'cat6', 'id': 9})
        >>> self.remap({'name': 'cat9', 'id': 5})
        >>> self.remap({'name': 'cat5', 'id': 9, 'special_property': 5})
        >>> assert self._id_to_cat == {
        >>>     5: {'name': 'cat5', 'id': 5, 'special_property': 5},
        >>>     9: {'name': 'cat6', 'id': 9},
        >>>     10: {'name': 'cat9', 'id': 10}}
    """
    def __init__(self):
        self._name_to_cat = {}
        self._id_to_cat = {}
        self._categories = []
        self._nextid = 1

    def remap(self, old_cat):
        import ubelt as ub
        catname = old_cat['name']
        new_cat = self._name_to_cat.get(catname, None)
        if new_cat is None:
            old_id = old_cat['id']
            if old_id in self._id_to_cat:
                # Need to update the ID
                new_id = self._nextid
                self._nextid += 1
            else:
                new_id = old_id
                if new_id >= self._nextid:
                    self._nextid = new_id + 1
            new_cat = {**old_cat}
            new_cat['id'] = new_id
            self._id_to_cat[new_id] = new_cat
            self._name_to_cat[catname] = new_cat
            self._categories.append(new_cat)
        else:
            # add in any special properties that dont disagree with
            # what already has been seen
            new_cat.update(ub.udict(old_cat) - new_cat.keys())
        return new_cat


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
# Just exist for backwards compatibility with older pickeled data.
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


def _image_corruption_check(fpath, only_shape=False, imread_kwargs=None):
    """
    Helper that checks if an image is readable or not
    """
    import kwimage
    from os.path import exists

    imread_kwargs = imread_kwargs or {}
    info = {'fpath': fpath}
    if not exists(fpath):
        info['failed'] = True
        info['error'] = 'does not exist'
    else:
        try:
            if only_shape:
                kwimage.load_image_shape(fpath)
            else:
                kwimage.imread(fpath, **imread_kwargs)
            info['failed'] = False
        except Exception as ex:
            err = str(ex)
            info['failed'] = True
            info['error'] = err
    return info


def _query_image_ids(coco_dset, select_images=None, select_videos=None):
    """
    Filters to a specific set of images given query parameters based on
    json-query (jq).

    Args:
        select_images(str | None):
            A json query (via the jq spec) that specifies which images
            belong in the subset. Note, this is a passed as the body of
            the following jq query format string to filter valid ids
            '.images[] | select({select_images}) | .id'.

            Examples for this argument are as follows:
            '.id < 3' will select all image ids less than 3.
            '.file_name | test(".*png")' will select only images with
            file names that end with png.
            '.file_name | test(".*png") | not' will select only images
            with file names that do not end with png.
            '.myattr == "foo"' will select only image dictionaries
            where the value of myattr is "foo".
            '.id < 3 and (.file_name | test(".*png"))' will select only
            images with id less than 3 that are also pngs.
            '.myattr | in({"val1": 1, "val4": 1})' will take images
            where myattr is either val1 or val4.
            An alternative syntax is:
            '[.myattr] | inside(["val1", "val4"])'

            Requires the "jq" python library is installed.

        select_videos(str | None):
            A json query (via the jq spec) that specifies which videos
            belong in the subset. Note, this is a passed as the body of
            the following jq query format string to filter valid ids
            '.videos[] | select({select_images}) | .id'.

            Examples for this argument are as follows:
            '.file_name | startswith("foo")' will select only videos
            where the name starts with foo. or
            '.file_name | contains("foo")' will select videos where
            any part of the filename contains foo.

            Only applicable for dataset that contain videos.

            Requires the "jq" python library is installed.

    SeeAlso:
        Based on ~/code/geowatch/geowatch/utils/kwcoco_extensions.py::filter_image_ids

    Example:
        >>> # xdoctest: +REQUIRES(module:jq)
        >>> from kwcoco._helpers import _query_image_ids
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> _query_image_ids(coco_dset, select_images='.id < 3')
        >>> _query_image_ids(coco_dset, select_images='.file_name | test(".*.png")')
        >>> _query_image_ids(coco_dset, select_images='.file_name | test(".*.png") | not')
        >>> _query_image_ids(coco_dset, select_images='.id < 3 and (.file_name | test(".*.png"))')
        >>> _query_image_ids(coco_dset, select_images='.id < 3 or (.file_name | test(".*.png"))')

    Ignore:
        # JQ Dev examples
        import jq
        dataset = [
            {'id': 1, 'name': 'foo'},
            {'id': 2, 'name': 'bar'},
            {'id': 3, 'name': 'baz'},
            {'id': 4, 'name': 'biz'},
        ]
        # The IN keyword doesnt seem to do what I want very well
        # jq.compile('.[] | select(.id | IN([1, 3]))').input(dataset).all()

        # This sorta works
        jq.compile('.[] | select(.id as $id | [1, 3] | index($id) != null)').input(dataset).all()

        # THERE WE GO, this is more reasonable
        jq.compile('.[] | select([.id] | inside([1, 3]))').input(dataset).all()
        jq.compile('.[] | select([.id] | inside([2, 4]))').input(dataset).all()
        jq.compile('.[] | select([.name] | inside(["foo", "baz"]))').input(dataset).all()

        jq.compile('.[] | select(.id < 3)').input(dataset).all()
        jq.compile('.[] | select(.name | test("b.*"))').input(dataset).all()

    """
    import ubelt as ub
    # Start with all images
    valid_gids = set(coco_dset.images())

    if select_images is not None:
        try:
            import jq
        except Exception:
            print('The jq library is required to run a generic image query')
            raise

        try:
            query_text = ".images[] | select({}) | .id".format(select_images)
            query = jq.compile(query_text)
            image_selected_gids = set(query.input(coco_dset.dataset).all())
            valid_gids &= image_selected_gids
        except Exception as ex:
            print('JQ Query Failed: {}, ex={}'.format(query_text, ex))
            raise

    if select_videos is not None:

        if isinstance(select_videos, list):
            # Interpret as video_ids
            ...
        else:
            try:
                import jq
            except Exception:
                print('The jq library is required to run a generic image query')
                raise

            try:
                query_text = ".videos[] | select({}) | .id".format(select_videos)
                query = jq.compile(query_text)
                selected_vidids = query.input(coco_dset.dataset).all()
                vid_selected_gids = set(ub.flatten(coco_dset.index.vidid_to_gids[vidid]
                                                   for vidid in selected_vidids))
                valid_gids &= vid_selected_gids
            except Exception:
                print('JQ Query Failed: {}'.format(query_text))
                raise

    valid_gids = sorted(valid_gids)
    return valid_gids
