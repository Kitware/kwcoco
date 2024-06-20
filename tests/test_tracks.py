

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


def test_track_structures():
    import kwcoco
    self = kwcoco.CocoDataset.demo('vidshapes1', use_cache=False, verbose=100)
    assert len(self.dataset['tracks']) == self.n_tracks


def test_add_tracks():
    import kwcoco
    self = kwcoco.CocoDataset()

    video_id = self.add_video(name='video1', id=9001)

    image_id1 = self.add_image(name='image1', video_id=video_id, frame_index=1)
    image_id2 = self.add_image(name='image2', video_id=video_id, frame_index=2)
    image_id3 = self.add_image(name='image3', video_id=video_id, frame_index=3)
    image_id4 = self.add_image(name='image4', video_id=video_id, frame_index=4)
    image_id5 = self.add_image(name='image5', video_id=video_id, frame_index=5)

    track_id = self.add_track(name='track1')

    self.add_annotation(image_id=image_id1, track_id=track_id)
    self.add_annotation(image_id=image_id2, track_id=track_id)
    self.add_annotation(image_id=image_id4, track_id=track_id)
