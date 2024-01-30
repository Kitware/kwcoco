
def test_frames_are_in_order():
    import kwcoco
    import ubelt as ub
    import random

    def is_sorted(x):
        return x == sorted(x)

    total_frames = 30
    total_videos = 4
    max_frames_per_video = 20

    # Seed rng for reproducibility
    rng = random.Random(926960862)

    # Initialize empty dataset
    dset = kwcoco.CocoDataset()

    # Add some number of videos
    vidid_pool = [
        dset.add_video('vid_{:03d}'.format(vididx))
        for vididx in range(total_videos)
    ]
    vidid_to_frame_pool = {vidid: ub.oset(range(max_frames_per_video))
                           for vidid in vidid_pool}

    # Add some number of frames to the videos in a random order
    for imgidx in range(total_frames):
        vidid = rng.choice(vidid_pool)
        frame_pool = vidid_to_frame_pool[vidid]
        assert frame_pool, 'ran out of frames'
        frame_index = rng.choice(frame_pool)
        frame_pool.remove(frame_index)

        name = 'img_{:03d}'.format(imgidx)
        dset.add_image(video_id=vidid, frame_index=frame_index, name=name)

    # Test that our image ids are always ordered by frame ids
    vidid_to_gids = dset.index.vidid_to_gids
    gids_were_in_order = []
    for vidid, gids in vidid_to_gids.items():
        gids_were_in_order.append(is_sorted(gids))
        frame_idxs = [dset.imgs[gid]['frame_index'] for gid in gids]

        # Note: this check is always valid
        assert is_sorted(frame_idxs), (
            'images in vidid_to_gids must be sorted by frame_index')

    # Note: this check has a chance of failing for other params / seeds
    assert not all(gids_were_in_order), (
        'the probability we randomly have ordered image ids is low, '
        'and 0 when we seed the rng'
    )

    try:
        import sqlalchemy  # NOQA
    except Exception:
        pass
    else:
        # Test that the sql view works too
        sql_dset = dset.view_sql(memory=True)

        vidid_to_gids = dict(sql_dset.index.vidid_to_gids)
        gids_were_in_order = []
        for vidid, gids in vidid_to_gids.items():
            gids_were_in_order.append(is_sorted(gids))
            frame_idxs = [dset.imgs[gid]['frame_index'] for gid in gids]

            # Note: this check is always valid
            assert is_sorted(frame_idxs), (
                'images in vidid_to_gids must be sorted by frame_index')

        # Note: this check has a chance of failing for other params / seeds
        assert not all(gids_were_in_order), (
            'the probability we randomly have ordered image ids is low, '
            'and 0 when we seed the rng'
        )


def test_lookup_annots_from_video():
    # TODO: check this in SQL as well
    import kwcoco
    import ubelt as ub
    dset = kwcoco.CocoDataset.demo('vidshapes8')

    # Run lookup from video id
    video_id = list(dset.videos())[0]
    video_annots = dset.annots(video_id=video_id)

    # Check all annots are indeed from that video
    assert set(video_annots.images.lookup('video_id')) == {video_id}, (
        'annots not from the right video'
    )

    # Check against lookup with brute force and assert no annots are missing.
    annots = dset.annots()
    vid_to_naids = ub.dict_hist(annots.images.lookup('video_id'))
    assert len(video_annots) == vid_to_naids[video_id], (
        'lookup did not return all annots',
    )
