from __future__ import annotations


def test_get_img_ids_matches_pycocotools_intersection_semantics():
    from kwcoco.compat_dataset import COCO

    dset = COCO()
    gid1 = dset.add_image(file_name='img1.png')
    gid2 = dset.add_image(file_name='img2.png')
    gid3 = dset.add_image(file_name='img3.png')
    cid1 = dset.add_category('cat1')
    cid2 = dset.add_category('cat2')
    dset.add_annotation(image_id=gid1, category_id=cid1, bbox=[0, 0, 1, 1])
    dset.add_annotation(image_id=gid2, category_id=cid1, bbox=[0, 0, 1, 1])
    dset.add_annotation(image_id=gid2, category_id=cid2, bbox=[0, 0, 1, 1])
    dset.add_annotation(image_id=gid3, category_id=cid2, bbox=[0, 0, 1, 1])

    assert dset.getImgIds(catIds=[cid1]) == [gid1, gid2]
    assert dset.getImgIds(catIds=[cid2]) == [gid2, gid3]
    assert dset.getImgIds(catIds=[cid1, cid2]) == [gid2]
    assert dset.getImgIds(imgIds=[gid1, gid2], catIds=[cid1, cid2]) == [gid2]


def test_ensure_image_data_accepts_image_ids_and_coco_url(monkeypatch, tmp_path):
    import builtins
    from pathlib import Path

    import ubelt as ub

    from kwcoco.compat_dataset import COCO

    dset = COCO()
    gid: int = dset.add_image(
        file_name='image.png',
        url=None,
        coco_url='https://example.com/image.png',
    )
    dset.bundle_dpath = tmp_path

    calls: list[tuple[str, str]] = []

    def fake_grabdata(url: str, fpath: str) -> str:
        calls.append((url, fpath))
        return fpath

    monkeypatch.setattr(builtins, 'input', lambda prompt: 'y')
    monkeypatch.setattr(ub, 'grabdata', fake_grabdata)

    try:
        import rich.prompt
    except ImportError:
        pass
    else:
        monkeypatch.setattr(rich.prompt.Confirm, 'ask', lambda prompt: True)

    dset._ensure_image_data(gids=[gid], verbose=0)
    assert len(calls) == 1
    url, fpath = calls[0]
    assert url == 'https://example.com/image.png'
    assert Path(fpath) == tmp_path / 'image.png'
