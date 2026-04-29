import pytest
import kwcoco
from kwcoco.metrics.helpers import associate_images


def test_truth_reuse_policy():
    """
    Create multiple predicted bounding boxes that overlap the same truth
    and test that they are both assigned to the same box when
    truth_reuse_policy is least-used.
    """
    from kwcoco.metrics.assignment import _assign_confusion_vectors
    import numpy as np
    import pandas as pd
    import ubelt as ub
    import kwimage

    true_boxes = kwimage.Boxes(
        np.array(
            [
                [50, 50, 23, 31],
                [51, 51, 23, 31],
            ]
        ),
        'cxywh',
    )

    pred_boxes = kwimage.Boxes(
        np.array(
            [
                [50, 50, 23, 31],
                [50, 50, 23, 31],
                [50, 50, 23, 31],
                [50, 50, 23, 31],
                [500, 500, 23, 31],
            ]
        ),
        'cxywh',
    )

    true_dets = kwimage.Detections(
        boxes=true_boxes,
        weights=np.array([1] * len(true_boxes)),
        class_idxs=np.array([0] * len(true_boxes)),
    )

    pred_dets = kwimage.Detections(
        boxes=pred_boxes,
        scores=np.array([0.5] * len(pred_boxes)),
        class_idxs=np.array([0] * len(pred_boxes)),
    )

    true_dets.boxes.ious(pred_dets.boxes)

    bg_weight = 1.0
    compat = 'all'
    iou_thresh = 0.1
    bias = 0.0

    y1 = _assign_confusion_vectors(
        true_dets,
        pred_dets,
        bias=bias,
        bg_weight=bg_weight,
        iou_thresh=iou_thresh,
        compat=compat,
        truth_reuse_policy='never',
    )
    y1 = pd.DataFrame(y1)
    print(y1)
    # Test the policy
    truth_usage1 = ub.dict_hist(y1['txs'])
    truth_usage1.pop(-1, None)
    assert max(truth_usage1.values()) <= 1, (
        'Truth only allowed to match a maximum of one time'
    )

    y2 = _assign_confusion_vectors(
        true_dets,
        pred_dets,
        bias=bias,
        bg_weight=bg_weight,
        iou_thresh=iou_thresh,
        compat=compat,
        truth_reuse_policy='least_used',
    )
    y2 = pd.DataFrame(y2)
    print(y2)

    truth_usage2 = ub.dict_hist(y2['txs'])
    truth_usage2.pop(-1, None)

    if truth_usage2[0] != 2 or truth_usage2[1] != 2:
        raise AssertionError(
            'Truth only allowed to match multiple times, but unused objects '
            'should be matched before increasing another the usage of a '
            'different true box'
        )


@pytest.mark.parametrize(
    'flatten_video_structure, expect_video_key',
    [
        (False, True),
        (True, False),
    ],
)
def test_basic_loose_image_matching(flatten_video_structure, expect_video_key):
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()

    image_id1_a = dset1.add_image(name='a')
    _ = dset1.add_image(name='b')
    image_id2_a = dset2.add_image(name='a')
    _ = dset2.add_image(name='c')

    matches = associate_images(
        dset1, dset2, flatten_video_structure=flatten_video_structure
    )

    assert ('video' in matches) is expect_video_key
    assert set(matches['image']['match_gids1']) == {image_id1_a}
    assert set(matches['image']['match_gids2']) == {image_id2_a}
    if not flatten_video_structure:
        assert matches['video'] == []


@pytest.mark.parametrize(
    'flatten_video_structure',
    [False, True],
)
def test_video_and_loose_matching_flatten_vs_group(flatten_video_structure):
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()

    video_id1 = dset1.add_video(name='V')
    video_id2 = dset2.add_video(name='V')

    # Video frames: one shared key, one non-shared in each dataset
    image_id1_va = dset1.add_image(name='va', video_id=video_id1, frame_index=1)
    _ = dset1.add_image(name='vb', video_id=video_id1, frame_index=2)
    image_id2_va = dset2.add_image(name='va', video_id=video_id2, frame_index=1)
    _ = dset2.add_image(name='vx', video_id=video_id2, frame_index=2)

    # Loose images: one shared key
    image_id1_la = dset1.add_image(name='la')
    image_id2_la = dset2.add_image(name='la')

    matches = associate_images(
        dset1, dset2, flatten_video_structure=flatten_video_structure
    )

    if not flatten_video_structure:
        assert set(matches.keys()) == {'image', 'video'}
        by_vidname = {m['vidname']: m for m in matches['video']}
        assert set(by_vidname['V']['match_gids1']) == {image_id1_va}
        assert set(by_vidname['V']['match_gids2']) == {image_id2_va}
        assert set(matches['image']['match_gids1']) == {image_id1_la}
        assert set(matches['image']['match_gids2']) == {image_id2_la}
    else:
        assert 'video' not in matches
        assert set(matches['image']['match_gids1']) == {image_id1_va, image_id1_la}
        assert set(matches['image']['match_gids2']) == {image_id2_va, image_id2_la}


@pytest.mark.parametrize(
    'flatten_video_structure',
    [False, True],
)
def test_valid_image_ids_filters_matches(flatten_video_structure):
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()

    video_id1 = dset1.add_video(name='V')
    video_id2 = dset2.add_video(name='V')

    image_id1_keep = dset1.add_image(name='keep', video_id=video_id1, frame_index=1)
    _ = dset1.add_image(name='drop', video_id=video_id1, frame_index=2)

    image_id2_keep = dset2.add_image(name='keep', video_id=video_id2, frame_index=1)
    _ = dset2.add_image(name='drop', video_id=video_id2, frame_index=2)

    image_id1_loose_keep = dset1.add_image(name='lkeep')
    _ = dset1.add_image(name='ldrop')
    image_id2_loose_keep = dset2.add_image(name='lkeep')
    _ = dset2.add_image(name='ldrop')

    valid_image_ids = {image_id1_keep, image_id1_loose_keep}
    matches = associate_images(
        dset1,
        dset2,
        flatten_video_structure=flatten_video_structure,
        valid_image_ids=valid_image_ids,
    )

    if not flatten_video_structure:
        by_vidname = {m['vidname']: m for m in matches['video']}
        assert set(by_vidname['V']['match_gids1']) == {image_id1_keep}
        assert set(by_vidname['V']['match_gids2']) == {image_id2_keep}
        assert set(matches['image']['match_gids1']) == {image_id1_loose_keep}
        assert set(matches['image']['match_gids2']) == {image_id2_loose_keep}
    else:
        assert 'video' not in matches
        assert set(matches['image']['match_gids1']) == {
            image_id1_keep,
            image_id1_loose_keep,
        }
        assert set(matches['image']['match_gids2']) == {
            image_id2_keep,
            image_id2_loose_keep,
        }


@pytest.mark.parametrize(
    'key_fallback, expect_exc',
    [
        (None, Exception),
        ('bogus', KeyError),
    ],
)
def test_key_fallback_errors_when_names_missing(key_fallback, expect_exc):
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()

    # No names -> forces fallback usage
    dset1.add_image(file_name='a.png')
    dset2.add_image(file_name='a.png')

    with pytest.raises(expect_exc):
        associate_images(dset1, dset2, key_fallback=key_fallback)


@pytest.mark.parametrize(
    'key_fallback, add_kwargs1, add_kwargs2',
    [
        ('file_name', dict(file_name='a.png'), dict(file_name='a.png')),
        ('id', dict(id=10, file_name='x.png'), dict(id=10, file_name='y.png')),
    ],
)
def test_key_fallback_success_when_names_missing(
    key_fallback, add_kwargs1, add_kwargs2
):
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()

    image_id1 = dset1.add_image(**add_kwargs1)
    image_id2 = dset2.add_image(**add_kwargs2)

    matches = associate_images(dset1, dset2, key_fallback=key_fallback)

    assert set(matches['image']['match_gids1']) == {image_id1}
    assert set(matches['image']['match_gids2']) == {image_id2}
    if key_fallback == 'id':
        assert image_id1 == 10
        assert image_id2 == 10


def test_no_common_videos_grouped_falls_back_on_loose_images():
    """
    I'm not sure if this behavior is desirable or not.
    """
    dset1 = kwcoco.CocoDataset()
    dset2 = kwcoco.CocoDataset()

    video_id1 = dset1.add_video(name='V1')
    video_id2 = dset2.add_video(name='V2')

    image_id1 = dset1.add_image(name='a', video_id=video_id1, frame_index=1)
    dset2.add_image(name='a', video_id=video_id2, frame_index=1)

    matches = associate_images(dset1, dset2, flatten_video_structure=False)

    assert matches['video'] == []
    # The only common "a" images are in non-common videos, so they should not
    # be matched as loose images.
    assert matches['image']['match_gids1'] == [image_id1]
