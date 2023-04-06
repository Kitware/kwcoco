

def test_track_order():
    """
    Test that annotations indexed by track-id are returned in temporal order.
    """

    import pytest
    pytest.skip()

    import kwcoco
    dset = kwcoco.CocoDataset.demo(
        'vidshapes', image_size=(8, 8), num_videos=1, num_frames=10)
    vid_gids = list(dset.videos().images[0])
    import kwarray
    rng = kwarray.ensure_rng(10279128)
    # Add images in randomized orders and assert they are always returned in a
    # sorted order.
    N = 100
    for _ in range(N):
        new_trackid = max(dset.index.trackid_to_aids.keys()) + 1
        vid_gids = kwarray.shuffle(vid_gids, rng=rng)
        for gid in vid_gids:
            dset.add_annotation(gid, bbox=[0, 0, 1, 1], track_id=new_trackid)
        track_aids = list(dset.index.trackid_to_aids[new_trackid])
        frame_order = dset.annots(track_aids).images.lookup('frame_index')
        assert sorted(frame_order) == frame_order
