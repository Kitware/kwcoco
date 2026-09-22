import ubelt as ub


def test_visualize_videos_cli_smoke(tmp_path):
    import numpy as np
    import kwcoco
    import kwimage
    from kwcoco.cli import coco_visualize_videos

    dset = kwcoco.CocoDataset()
    vidid = dset.add_video(name='video1', width=64, height=64)
    catid = dset.add_category('thing', color='lime')
    for frame in range(2):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[..., 1] = 255
        fpath = ub.Path(tmp_path) / f'frame_{frame}.png'
        kwimage.imwrite(fpath, image)
        gid = dset.add_image(
            file_name=str(fpath),
            width=64,
            height=64,
            video_id=vidid,
            frame_index=frame,
        )
        if frame == 0:
            dset.add_annotation(
                image_id=gid,
                category_id=catid,
                bbox=[10, 10, 20, 20],
            )

    dset.fpath = ub.Path(tmp_path) / 'data.kwcoco.json'
    dset.dump(dset.fpath)
    viz_dpath = ub.Path(tmp_path) / 'viz'

    coco_visualize_videos.__cli__.main(
        cmdline=False,
        src=dset.fpath,
        viz_dpath=viz_dpath,
        workers=0,
        max_dim=None,
        min_dim=None,
        draw_labels=False,
        draw_anns=False,
        draw_header=False,
        draw_chancode=False,
    )

    assert len(list(viz_dpath.glob('**/*.jpg'))) > 0


def test_visualize_respects_annotation_colors_and_bboxless_annots(tmp_path):
    """Confusion colors survive and bbox-less metadata does not crash."""
    import numpy as np
    import kwcoco
    import kwimage
    from kwcoco.cli import coco_visualize_videos

    dset = kwcoco.CocoDataset()
    catid = dset.add_category('poop')
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    fpath = ub.Path(tmp_path) / 'frame.png'
    kwimage.imwrite(fpath, image)
    gid = dset.add_image(file_name=str(fpath), width=128, height=64)

    # Same palette emitted by kwcoco.cli.coco_eval.build_confusion_datasets.
    expected = [
        '#f42836',  # predicted false positive
        '#0068c7',  # predicted true positive
        '#800080',  # truth false negative
        '#3eae2b',  # truth true positive
    ]
    xs = [5, 35, 65, 95]
    for x, color in zip(xs, expected):
        dset.add_annotation(
            image_id=gid,
            category_id=catid,
            bbox=[x, 20, 20, 20],
            color=color,
        )

    # Bbox-less annotations are valid KWCoco. In particular, some datasets
    # use point annotations only as metadata. They should not prevent normal
    # detection/confusion visualization.
    dset.add_annotation(
        image_id=gid,
        category_id=catid,
        keypoints=[16, 16, 2],
        color='yellow',
    )

    dset.fpath = ub.Path(tmp_path) / 'confusion.kwcoco.json'
    dset.dump(dset.fpath)
    viz_dpath = ub.Path(tmp_path) / 'viz'

    coco_visualize_videos.__cli__.main(
        cmdline=False,
        src=dset.fpath,
        viz_dpath=viz_dpath,
        workers=0,
        space='image',
        max_dim=None,
        min_dim=None,
        draw_imgs=False,
        draw_anns=True,
        draw_boxes=True,
        draw_segmentations=False,
        draw_labels=False,
        ann_thickness=6,
        draw_header=False,
        draw_chancode=False,
    )

    rendered_fpaths = list(viz_dpath.glob('**/*view_ann.jpg'))
    assert len(rendered_fpaths) == 1
    canvas = kwimage.imread(rendered_fpaths[0])
    palette = np.array([kwimage.Color.coerce(c).as255() for c in expected])

    # JPEG / antialiasing perturb exact RGB values. The center of each thick
    # top edge should nevertheless be closest to its requested color.
    for index, x in enumerate(xs):
        patch = canvas[19:23, x + 6:x + 15].astype(float)
        observed = np.median(patch.reshape(-1, 3), axis=0)
        dists = ((palette - observed[None, :]) ** 2).sum(axis=1)
        assert int(dists.argmin()) == index
