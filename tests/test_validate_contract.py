import pytest
import warnings


def _uncategorized_bboxless_dataset():
    import kwcoco

    dset = kwcoco.CocoDataset()
    gid = dset.add_image(
        id=1,
        name='image-1',
        file_name='image.jpg',
        width=32,
        height=32,
    )
    dset.add_annotation(id=1, image_id=gid, category_id=None)
    return dset


def test_validate_accepts_uncategorized_bboxless_annotations():
    dset = _uncategorized_bboxless_dataset()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result = dset.validate(missing=False)

    assert not record
    assert not result['errors']
    assert not result['warnings']


def test_bbox_check_is_opt_in(capsys):
    dset = _uncategorized_bboxless_dataset()

    with pytest.warns(UserWarning, match='bbox'):
        result = dset.validate(missing=False, annot_attrs='warn')
    assert not result['errors']
    assert len(result['warnings']) == 1
    assert capsys.readouterr().out == ''

    result = dset.validate(
        missing=False,
        annot_attrs='error',
        verbose=0,
    )
    assert len(result['errors']) == 1
    assert not result['warnings']


def test_validate_verbose_levels(capsys):
    dset = _uncategorized_bboxless_dataset()

    result = dset.validate(missing=False, verbose=1)
    assert not result['errors']
    assert capsys.readouterr().out == ''

    result = dset.validate(missing=False, verbose=2)
    assert not result['errors']
    stdout = capsys.readouterr().out
    assert 'Validate json-schema' in stdout
    assert 'Check table' in stdout
