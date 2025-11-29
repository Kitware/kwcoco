import ubelt as ub


def associate_images(dset1, dset2, key_fallback=None, valid_image_ids=None):
    """
    Builds an association between image-ids in two datasets.

    One use for this is if ``dset1`` is a truth dataset and ``dset2`` is a
    prediction dataset, and you need the to know which images are in common so
    they can be scored.

    Args:
        dset1 (kwcoco.CocoDataset): a kwcoco dataset.

        dset2 (kwcoco.CocoDataset): another kwcoco dataset

        key_fallback (str):
            The fallback key to use if the image "name" is not specified.
            This can either be "file_name" or "id" or None.

        valid_image_ids (set | None): if given, filter out matches where
            the truth image ids are not in this set. We may remove this option
            in the future.

    TODO:
        - [x] port to kwcoco proper
        - [x] use in kwcoco evaluate_detections as a robust image/video association method

    Example:
        >>> import kwcoco
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> dset1 = kwcoco.CocoDataset.demo('shapes2')
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>> }
        >>> dset2 = perterb_coco(dset1, **kwargs)
        >>> matches = associate_images(dset1, dset2, key_fallback='file_name')
        >>> assert len(matches['image']['match_gids1'])
        >>> assert len(matches['image']['match_gids2'])
        >>> assert not len(matches['video'])

    Example:
        >>> import kwcoco
        >>> from kwcoco.demo.perterb import perterb_coco
        >>> dset1 = kwcoco.CocoDataset.demo('vidshapes2')
        >>> kwargs = {
        >>>     'box_noise': 0.5,
        >>>     'n_fp': (0, 10),
        >>>     'n_fn': (0, 10),
        >>> }
        >>> dset2 = perterb_coco(dset1, **kwargs)
        >>> matches = associate_images(dset1, dset2, key_fallback='file_name')
        >>> assert not len(matches['image']['match_gids1'])
        >>> assert not len(matches['image']['match_gids2'])
        >>> assert len(matches['video'])
    """
    common_vidnames = (set(dset1.index.name_to_video) &
                       set(dset2.index.name_to_video))

    def image_keys(dset, gids):
        # Generate image "keys" that should be compatible between datasets
        for gid in gids:
            img = dset.imgs[gid]
            if img.get('name', None) is not None:
                yield img['name']
            else:
                if key_fallback is None:
                    raise Exception('images require names to associate')
                elif key_fallback == 'id':
                    yield img['id']
                elif key_fallback == 'file_name':
                    yield img['file_name']
                else:
                    raise KeyError(key_fallback)

    all_gids1 = list(dset1.imgs.keys())
    all_gids2 = list(dset2.imgs.keys())
    all_keys1 = list(image_keys(dset1, all_gids1))
    all_keys2 = list(image_keys(dset2, all_gids2))
    key_to_gid1 = ub.dzip(all_keys1, all_gids1)
    key_to_gid2 = ub.dzip(all_keys2, all_gids2)
    gid_to_key1 = ub.invert_dict(key_to_gid1)
    gid_to_key2 = ub.invert_dict(key_to_gid2)

    video_matches = []

    all_match_gids1 = set()
    all_match_gids2 = set()

    for vidname in common_vidnames:
        video1 = dset1.index.name_to_video[vidname]
        video2 = dset2.index.name_to_video[vidname]
        vidid1 = video1['id']
        vidid2 = video2['id']
        gids1 = dset1.index.vidid_to_gids[vidid1]
        gids2 = dset2.index.vidid_to_gids[vidid2]
        keys1 = ub.oset(ub.take(gid_to_key1, gids1))
        keys2 = ub.oset(ub.take(gid_to_key2, gids2))
        match_keys = ub.oset(keys1) & ub.oset(keys2)
        match_gids1 = list(ub.take(key_to_gid1, match_keys))
        match_gids2 = list(ub.take(key_to_gid2, match_keys))
        all_match_gids1.update(match_gids1)
        all_match_gids2.update(match_gids2)
        video_matches.append({
            'vidname': vidname,
            'match_gids1': match_gids1,
            'match_gids2': match_gids2,
        })

    # Associate loose images not belonging to any video
    unmatched_gid_to_key1 = ub.dict_diff(gid_to_key1, all_match_gids1)
    unmatched_gid_to_key2 = ub.dict_diff(gid_to_key2, all_match_gids2)

    remain_keys = (set(unmatched_gid_to_key1.values()) &
                   set(unmatched_gid_to_key2.values()))
    remain_gids1 = [key_to_gid1[key] for key in remain_keys]
    remain_gids2 = [key_to_gid2[key] for key in remain_keys]

    image_matches = {
        'match_gids1': remain_gids1,
        'match_gids2': remain_gids2,
    }

    matches = {
        'image': image_matches,
        'video': video_matches,
    }

    if valid_image_ids is not None:
        # Filter invalid images
        for item in video_matches + [image_matches]:
            gids1 = item['match_gids1']
            gids2 = item['match_gids1']
            new_gids1 = []
            new_gids2 = []
            for gid1, gid2 in zip(gids1, gids2):
                if gid1 in valid_image_ids:
                    new_gids1.append(gid1)
                    new_gids2.append(gid2)
            item['match_gids1'] = new_gids1
            item['match_gids2'] = new_gids2
    return matches
