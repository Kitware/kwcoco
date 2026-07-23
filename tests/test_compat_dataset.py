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


def test_coco_dataset_download(monkeypatch, tmp_path):
    from pathlib import Path

    import ubelt as ub

    import kwcoco

    bundle_dpath = tmp_path / 'bundle'
    download_dpath = tmp_path / 'downloads'
    dset = kwcoco.CocoDataset(bundle_dpath=bundle_dpath)
    gid1 = dset.add_image(
        file_name='nested/image1.png',
        coco_url='https://example.com/image1.png',
    )
    gid2 = dset.add_image(
        file_name='image2.png',
        coco_url='https://example.com/image2.png',
    )

    existing_fpath = download_dpath / 'nested/image1.png'
    existing_fpath.parent.mkdir(parents=True)
    existing_fpath.write_bytes(b'existing')

    calls = []

    def fake_download(coco_url, fpath, verbose=1):
        fpath = Path(fpath)
        calls.append((coco_url, fpath, verbose))
        fpath.write_bytes(b'downloaded')
        return fpath

    monkeypatch.setattr(ub, 'download', fake_download)

    original_bundle_dpath = dset.bundle_dpath
    local_paths = dset.download(
        dpath=download_dpath,
        image_ids=[gid1, gid2],
        verbose=0,
    )

    assert list(map(Path, local_paths)) == [
        download_dpath / 'nested/image1.png',
        download_dpath / 'image2.png',
    ]
    assert calls == [
        (
            'https://example.com/image2.png',
            download_dpath / 'image2.png',
            0,
        )
    ]
    assert existing_fpath.read_bytes() == b'existing'
    assert (download_dpath / 'image2.png').read_bytes() == b'downloaded'
    assert dset.bundle_dpath == original_bundle_dpath

    dset.download(
        dpath=download_dpath,
        image_ids=gid1,
        overwrite=True,
        verbose=0,
    )
    assert calls[-1][0:2] == (
        'https://example.com/image1.png',
        download_dpath / 'nested/image1.png',
    )


def test_coco_dataset_download_requires_coco_url(tmp_path):
    import pytest

    import kwcoco

    dset = kwcoco.CocoDataset(bundle_dpath=tmp_path)
    gid = dset.add_image(file_name='image.png')
    with pytest.raises(ValueError, match='missing coco_url'):
        dset.download(image_ids=[gid], verbose=0)


def test_compat_download_uses_target_without_reroot(monkeypatch, tmp_path):
    from pathlib import Path

    import ubelt as ub

    from kwcoco.compat_dataset import COCO

    dset = COCO(bundle_dpath=tmp_path / 'bundle')
    gid = dset.add_image(
        file_name='image.png',
        coco_url='https://example.com/image.png',
    )
    original_bundle_dpath = dset.bundle_dpath

    calls = []

    def fake_download(coco_url, fpath, verbose=1):
        fpath = Path(fpath)
        calls.append((coco_url, fpath))
        fpath.write_bytes(b'downloaded')
        return fpath

    monkeypatch.setattr(ub, 'download', fake_download)

    target = tmp_path / 'compat-download'
    result = dset.download(tarDir=target, imgIds=[gid])
    assert result is None
    assert calls == [
        ('https://example.com/image.png', target / 'image.png')
    ]
    assert dset.bundle_dpath == original_bundle_dpath
