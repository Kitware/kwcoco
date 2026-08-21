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

    tracked_annots = [ann for ann in dset.anns.values() if 'track_id' in ann]
    assert tracked_annots
    assert all(ann['instance_id'] == ann['track_id'] for ann in tracked_annots)


def test_conform_preserves_info_representation_by_default():
    import kwcoco

    info = [
        {'type': 'process', 'properties': {'name': 'demo'}},
        {'type': 'measure', 'properties': {'score': 1.0}},
    ]
    dset = kwcoco.CocoDataset()
    dset.dataset['info'] = info

    dset.conform(ensure_imgsize=False, pycocotools_info=False)

    assert dset.dataset['info'] is info


def test_conform_legacy_wraps_kwcoco_info_list():
    import kwcoco

    info = [
        {'type': 'process', 'properties': {'name': 'demo'}},
        {'type': 'measure', 'properties': {'score': 1.0}},
    ]
    dset = kwcoco.CocoDataset()
    dset.dataset['info'] = info

    dset.conform(
        legacy=True,
        ensure_imgsize=False,
        pycocotools_info=False,
    )

    assert dset.dataset['info'] == {'kwcoco_info': info}


def test_conform_legacy_preserves_mscoco_info_dict():
    import kwcoco

    info = {
        'description': 'demo dataset',
        'year': 2026,
        'custom_field': 'accepted by pycocotools',
    }
    dset = kwcoco.CocoDataset()
    dset.dataset['info'] = info

    dset.conform(
        legacy=True,
        ensure_imgsize=False,
        pycocotools_info=False,
    )

    assert dset.dataset['info'] is info


def test_conform_legacy_uses_empty_mscoco_info_dict():
    import kwcoco

    dset = kwcoco.CocoDataset()
    assert dset.dataset['info'] == []

    dset.conform(
        legacy=True,
        ensure_imgsize=False,
        pycocotools_info=False,
    )

    assert dset.dataset['info'] == {}
