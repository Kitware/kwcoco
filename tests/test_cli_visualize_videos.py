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
