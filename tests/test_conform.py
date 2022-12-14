def test_conform_mmlab():
    import kwcoco
    dset = kwcoco.CocoDataset.demo('vidshapes8')

    # We dont include frame_id by default
    assert 'frame_id' not in dset.images().objs[0]

    dset.conform(mmlab=True)

    assert 'frame_id' in dset.images().objs[0]

    for images in dset.videos().images:
        # Check that all images in the video have a consistent frame id / index
        frame_ids = images.lookup('frame_id')
        frame_idxs = images.lookup('frame_index')

        assert sorted(frame_ids) == frame_ids
        assert sorted(frame_idxs) == frame_idxs
