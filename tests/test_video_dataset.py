
def test_frames_are_in_order():
    import kwcoco
    import ubelt as ub
    import random

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
    for vidid, gids in dset.index.vidid_to_gids.items():
        # Note: this check might be invalid if the params or seed is changed
        assert sorted(gids) != gids, (
            'the probability we randomly have ordered image ids is low, '
            'and 0 when we seed the rng'
        )
        frame_idxs = [dset.imgs[gid]['frame_index'] for gid in gids]

        # Note: this check is always valid
        assert sorted(frame_idxs) == frame_idxs, (
            'images in vidid_to_gids must be sorted by frame_index')

    try:
        import sqlalchemy  # NOQA
    except Exception:
        pass
    else:
        # Test that the sql view works too
        sql_dset = dset.view_sql(memory=True)

        sql_vidid_to_gids = dict(sql_dset.index.vidid_to_gids)
        for vidid, gids in sql_vidid_to_gids.items():
            # Note: this check might be invalid if the params or seed is changed
            assert sorted(gids) != gids, (
                'the probability we randomly have ordered image ids is low, '
                'and 0 when we seed the rng'
            )
            frame_idxs = [dset.imgs[gid]['frame_index'] for gid in gids]

            # Note: this check is always valid
            assert sorted(frame_idxs) == frame_idxs, (
                'images in vidid_to_gids must be sorted by frame_index')
