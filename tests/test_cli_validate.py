"""
Test validation cases in the CLI
"""


def test_validate_missing():
    from kwcoco.cli import coco_validate
    from kwcoco.cli import coco_subset
    import kwcoco
    import ubelt as ub
    dpath = ub.Path.appdir('kwcoco/validate').ensuredir()
    dset = kwcoco.CocoDataset.demo('vidshapes8-msi')

    tmp_coco_fpath = dpath / 'tmp.kwcoco.json'

    coco_subset.CocoSubsetCLI.main(cmdline=0, src=dset, dst=tmp_coco_fpath, copy_assets=True)

    dset2 = kwcoco.CocoDataset(tmp_coco_fpath)
    coco_img = dset2.images().coco_images[0]
    img_fpath = ub.Path(coco_img.primary_image_filepath())
    img_fpath.delete()

    import pytest
    with pytest.raises(Exception) as ex:
        coco_validate.CocoValidateCLI.main(cmdline=0, src=[dset2])
    assert 'The first one is' in ex.value.args[0]
