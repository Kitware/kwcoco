import copy
import warnings

import pytest

import kwcoco
from kwcoco import exceptions


def _assert_index_matches_rebuild(dset):
    """Check that dynamic index maintenance agrees with raw-data rebuild."""
    assert dset._check_index()


def test_add_images_nonvideo_batch_preserves_fast_path(monkeypatch):
    dset = kwcoco.CocoDataset()

    import kwcoco._dataset_components.mixin_index as mixin_index

    group_items_calls = []
    original_group_items = mixin_index.ub.group_items

    def record_group_items(*args, **kwargs):
        group_items_calls.append((args, kwargs))
        return original_group_items(*args, **kwargs)

    monkeypatch.setattr(mixin_index.ub, 'group_items', record_group_items)
    dset.add_images([
        {'id': 1, 'file_name': 'image-1.png'},
        {'id': 2, 'file_name': 'image-2.png'},
    ])

    assert group_items_calls == []
    _assert_index_matches_rebuild(dset)


def test_add_images_populates_initial_video_lookup():
    dset = kwcoco.CocoDataset()
    dset.add_images([
        {
            'id': 1,
            'file_name': 'frame-1.png',
            'video_id': 7,
            'frame_index': 1,
        },
        {
            'id': 2,
            'file_name': 'frame-0.png',
            'video_id': 7,
            'frame_index': 0,
        },
    ])

    assert list(dset.index.vidid_to_gids[7]) == [2, 1]
    _assert_index_matches_rebuild(dset)


def test_clear_images_clears_all_image_and_annotation_lookups():
    dset = kwcoco.CocoDataset()
    video_id = dset.add_video(name='video')
    category_id = dset.add_category(name='category')
    track_id = dset.add_track(name='track')
    image_id = dset.add_image(
        file_name='frame.png',
        name='named-frame',
        video_id=video_id,
        frame_index=0,
    )
    annotation_id = dset.add_annotation(
        image_id=image_id,
        category_id=category_id,
        track_id=track_id,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loose_image_id = dset.add_image(
            file_name='loose-frame.png',
            video_id=999,
            frame_index=0,
        )
        dset.add_annotation(
            image_id=loose_image_id,
            category_id=None,
            track_id=999,
        )

    assert dset.index.name_to_img['named-frame']['id'] == image_id
    assert annotation_id in dset.index.trackid_to_aids[track_id]

    dset.clear_images()

    assert dset.dataset['images'] == []
    assert dset.dataset['annotations'] == []
    assert dset.index.imgs == {}
    assert dset.index.anns == {}
    assert dset.index.name_to_img == {}
    assert dset.index.file_name_to_img == {}
    assert all(not aids for aids in dset.index.cid_to_aids.values())
    assert all(not gids for gids in dset.index.vidid_to_gids.values())
    assert all(not aids for aids in dset.index.trackid_to_aids.values())
    assert None not in dset.index.cid_to_aids
    assert 999 not in dset.index.vidid_to_gids
    assert 999 not in dset.index.trackid_to_aids
    _assert_index_matches_rebuild(dset)


def test_remove_tracks_keep_annots_reindexes_as_trackless():
    dset = kwcoco.CocoDataset()
    category_id = dset.add_category(name='category')
    track_id = dset.add_track(name='track')
    image_id = dset.add_image(file_name='image.png')
    annotation_id = dset.add_annotation(
        image_id=image_id,
        category_id=category_id,
        track_id=track_id,
    )

    dset.remove_tracks([track_id], keep_annots=True)

    assert track_id not in dset.index.tracks
    assert track_id not in dset.index.trackid_to_aids
    assert 'track_id' not in dset.anns[annotation_id]
    assert annotation_id in dset.index.trackid_to_aids[None]
    _assert_index_matches_rebuild(dset)


def test_remove_unreferenced_track_after_rebuild():
    raw = {
        'categories': [],
        'images': [],
        'annotations': [],
        'videos': [],
        'tracks': [{'id': 1, 'name': 'empty-track'}],
    }
    dset = kwcoco.CocoDataset(raw)

    # Registered tracks have explicit empty buckets, so strict mutation code
    # can distinguish an unreferenced track from a corrupted index.
    assert 1 in dset.index.trackid_to_aids
    assert not dset.index.trackid_to_aids[1]
    dset.remove_tracks([1], keep_annots=True)

    assert dset.dataset['tracks'] == []
    assert dset.index.tracks == {}
    _assert_index_matches_rebuild(dset)


def test_failed_add_image_is_atomic():
    dset = kwcoco.CocoDataset()
    before = copy.deepcopy(dset.dataset)

    with pytest.raises(exceptions.InvalidAddError):
        dset.add_image(
            file_name='invalid-frame.png',
            video_id=1,
            # Video images require a frame_index.
        )

    assert dset.dataset == before
    assert dset.index.imgs == {}
    assert dset.index.gid_to_aids == {}
    assert dset.index.file_name_to_img == {}
    assert dset.add_image(file_name='valid-frame.png') == 1
    _assert_index_matches_rebuild(dset)


def test_failed_add_annotation_is_atomic():
    dset = kwcoco.CocoDataset()
    category_id = dset.add_category(name='category')
    before = copy.deepcopy(dset.dataset)

    with pytest.raises(KeyError):
        dset.add_annotation(image_id=999, category_id=category_id)

    assert dset.dataset == before
    assert dset.index.anns == {}
    assert all(not aids for aids in dset.index.cid_to_aids.values())
    assert all(not aids for aids in dset.index.trackid_to_aids.values())
    image_id = dset.add_image(file_name='valid-image.png')
    assert dset.add_annotation(image_id=image_id, category_id=category_id) == 1
    _assert_index_matches_rebuild(dset)


def test_remove_mixed_referenced_and_unreferenced_tracks():
    raw = {
        'categories': [{'id': 1, 'name': 'category'}],
        'images': [{'id': 1, 'file_name': 'image.png'}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'track_id': 1},
        ],
        'videos': [],
        'tracks': [
            {'id': 1, 'name': 'referenced-track'},
            {'id': 2, 'name': 'empty-track'},
        ],
    }
    dset = kwcoco.CocoDataset(raw)

    assert 1 in dset.index.trackid_to_aids
    assert 2 in dset.index.trackid_to_aids
    assert not dset.index.trackid_to_aids[2]

    dset.remove_tracks([1, 2], keep_annots=True)

    assert dset.dataset['tracks'] == []
    assert 1 in dset.index.trackid_to_aids[None]
    assert 'track_id' not in dset.anns[1]
    _assert_index_matches_rebuild(dset)


def test_remove_track_does_not_hide_missing_index_bucket():
    raw = {
        'categories': [{'id': 1, 'name': 'category'}],
        'images': [{'id': 1, 'file_name': 'image.png'}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'track_id': 1},
        ],
        'videos': [],
        'tracks': [{'id': 1, 'name': 'track'}],
    }
    dset = kwcoco.CocoDataset(raw)

    # Simulate an externally corrupted live index. Removal must fail before
    # mutating the authoritative dataset rather than silently orphaning the
    # annotation.
    del dset.index.trackid_to_aids[1]
    with pytest.raises(KeyError):
        dset.remove_tracks([1], keep_annots=True)

    assert dset.dataset['tracks'] == [{'id': 1, 'name': 'track'}]
    assert dset.anns[1]['track_id'] == 1


def test_duplicate_ids_warn_when_rebuilding_index():
    raw = {
        'categories': [
            {'id': 1, 'name': 'category-a'},
            {'id': 1, 'name': 'category-b'},
        ],
        'keypoint_categories': [
            {'id': 2, 'name': 'keypoint-a'},
            {'id': 2, 'name': 'keypoint-b'},
        ],
        'tracks': [
            {'id': 3, 'name': 'track-a'},
            {'id': 3, 'name': 'track-b'},
        ],
        'images': [],
        'annotations': [],
        'videos': [],
    }

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        dset = kwcoco.CocoDataset(raw)

    messages = [str(record.message) for record in records]
    assert any('Categories have the same id' in message for message in messages)
    assert any('Keypoint' in message and 'same id' in message for message in messages)
    assert any('Tracks have the same id' in message for message in messages)

    # Rebuilding retains the historical last-item-wins behavior.
    assert dset.cats[1]['name'] == 'category-b'
    assert dset.index.kpcats[2]['name'] == 'keypoint-b'
    assert dset.index.tracks[3]['name'] == 'track-b'


def test_remove_tracks_keep_annots_without_index():
    raw = {
        'categories': [{'id': 1, 'name': 'category'}],
        'images': [{'id': 1, 'file_name': 'image.png'}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'track_id': 1},
            {'id': 2, 'image_id': 1, 'category_id': 1, 'track_id': 2},
        ],
        'videos': [],
        'tracks': [
            {'id': 1, 'name': 'remove-me'},
            {'id': 2, 'name': 'keep-me'},
        ],
    }
    dset = kwcoco.CocoDataset(raw)
    dset.index.clear()

    remove_info = dset.remove_tracks([1], keep_annots=True)

    assert remove_info == {'annotations': None, 'tracks': 1}
    assert dset.dataset['tracks'] == [{'id': 2, 'name': 'keep-me'}]
    assert 'track_id' not in dset.dataset['annotations'][0]
    assert dset.dataset['annotations'][1]['track_id'] == 2

    dset.rebuild_index()
    _assert_index_matches_rebuild(dset)
    assert 1 in dset.index.trackid_to_aids[None]
    assert dset.index.trackid_to_aids[2] == {2}


def test_remove_videos_without_index_uses_raw_images():
    raw = {
        'categories': [{'id': 1, 'name': 'category'}],
        'videos': [
            {'id': 1, 'name': 'remove-me'},
            {'id': 2, 'name': 'keep-me'},
        ],
        'images': [
            {'id': 1, 'file_name': 'remove.png', 'video_id': 1, 'frame_index': 0},
            {'id': 2, 'file_name': 'keep.png', 'video_id': 2, 'frame_index': 0},
            {'id': 3, 'file_name': 'loose.png'},
        ],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1},
            {'id': 2, 'image_id': 2, 'category_id': 1},
            {'id': 3, 'image_id': 3, 'category_id': 1},
        ],
        'tracks': [],
    }
    dset = kwcoco.CocoDataset(raw)
    dset.index.clear()

    remove_info = dset.remove_videos([1])

    assert remove_info == {'annotations': 1, 'images': 1, 'videos': 1}
    assert [video['id'] for video in dset.dataset['videos']] == [2]
    assert [img['id'] for img in dset.dataset['images']] == [2, 3]
    assert [ann['id'] for ann in dset.dataset['annotations']] == [2, 3]

    dset.rebuild_index()
    _assert_index_matches_rebuild(dset)


def test_remove_categories_keep_annots_without_index():
    raw = {
        'categories': [
            {'id': 1, 'name': 'remove-me'},
            {'id': 2, 'name': 'keep-me'},
        ],
        'images': [{'id': 1, 'file_name': 'image.png'}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1},
            {'id': 2, 'image_id': 1, 'category_id': 2},
        ],
        'videos': [],
        'tracks': [],
    }
    dset = kwcoco.CocoDataset(raw)
    dset.index.clear()

    remove_info = dset.remove_categories([1], keep_annots=True)

    assert remove_info == {'annotations': None, 'categories': 1}
    assert dset.dataset['categories'] == [{'id': 2, 'name': 'keep-me'}]
    assert 'category_id' not in dset.dataset['annotations'][0]
    assert dset.dataset['annotations'][1]['category_id'] == 2

    dset.rebuild_index()
    _assert_index_matches_rebuild(dset)
    assert 1 in dset.index.cid_to_aids[None]
    assert dset.index.cid_to_aids[2] == {2}
