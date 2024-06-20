import ubelt as ub
from os.path import join
from kwcoco.cli import coco_subset
import kwcoco
import kwimage


def _create_demo_dataset():
    # Create a demo coco dataset (explicitly)
    dset = kwcoco.CocoDataset()

    classes = kwcoco.CategoryTree.demo('btree2', r=2, h=2)
    cats = list(classes.to_coco())
    for cat in cats:
        dset.add_category(**cat)

    image_keys = list(kwimage.grab_test_image_fpath.keys())[0:4]

    for key in image_keys:
        fpath = kwimage.grab_test_image_fpath(key)
        h, w = kwimage.load_image_shape(fpath)[0:2]
        gid = dset.add_image(file_name=fpath, name=key, height=h, width=w)
        dets = kwimage.Detections.random(num=1, classes=classes).scale((w, h))

        for ann in dets.to_coco(dset=dset, image_id=gid):
            ann['user'] = 'me'
            dset.add_annotation(**ann)
    return dset


def test_subset_api_method():
    """
    Test that the custom "user" property is not dropped
    """
    dset = _create_demo_dataset()
    subset_gids = list(dset.imgs.keys())[::2]
    subdset = dset.subset(subset_gids)
    print('subdset.dataset = {}'.format(ub.urepr(subdset.dataset, nl=2)))
    print('dset.dataset = {}'.format(ub.urepr(dset.dataset, nl=2)))
    assert len(subdset.index.imgs) == 2
    assert ub.peek(subdset.index.anns.values())['user'] == 'me'


def test_subset_cli_with_gids():
    dset = _create_demo_dataset()
    subset_gids = list(dset.imgs.keys())[::2]

    print('dset.fpath = {}'.format(ub.urepr(dset.fpath, nl=1)))
    dpath = ub.Path.appdir('kwcoco/test/subset').ensuredir()
    dset.fpath = join(dpath, 'input.kwcoco.json')
    print('dset.fpath = {}'.format(ub.urepr(dset.fpath, nl=1)))
    dset.dump(dset.fpath)
    print('dset.fpath = {}'.format(ub.urepr(dset.fpath, nl=1)))

    dst_fpath2 = join(dpath, 'output.kwcoco.json')
    config1 = {
        'src': dset.fpath,
        'dst': dst_fpath2,
        'gids': subset_gids,
    }
    print('config1 = {}'.format(ub.urepr(config1, nl=1)))
    coco_subset.CocoSubsetCLI.main(cmdline=False, **config1)
    dst_dset2 = kwcoco.CocoDataset(dst_fpath2)
    print('dst_dset2.dataset = {}'.format(ub.urepr(dst_dset2.dataset, nl=2)))
    assert len(dst_dset2.index.imgs) == 2
    assert ub.peek(dst_dset2.index.anns.values())['user'] == 'me'


def test_subset_cli_with_jq():
    try:
        import jq  # NOQA
    except Exception:
        import pytest
        pytest.skip('test requires jq')
    dset = _create_demo_dataset()

    print('dset.fpath = {}'.format(ub.urepr(dset.fpath, nl=1)))
    dpath = ub.Path.appdir('kwcoco/test/subset').ensuredir()
    dset.fpath = join(dpath, 'input.kwcoco.json')
    print('dset.fpath = {}'.format(ub.urepr(dset.fpath, nl=1)))
    dset.dump(dset.fpath)
    print('dset.fpath = {}'.format(ub.urepr(dset.fpath, nl=1)))

    dst_fpath1 = join(dpath, 'output.kwcoco.json')
    config2 = {
        'src': dset.fpath,
        'dst': dst_fpath1,
        'select_images': '.id % 2 == 0'
    }
    print('config2 = {}'.format(ub.urepr(config2, nl=1)))
    coco_subset.CocoSubsetCLI.main(cmdline=False, **config2)
    dst_dset1 = kwcoco.CocoDataset(dst_fpath1)

    assert len(dst_dset1.index.imgs) == 2
    assert ub.peek(dst_dset1.index.anns.values())['user'] == 'me'
    print('dst_dset1.dataset = {}'.format(ub.urepr(dst_dset1.dataset, nl=2)))
