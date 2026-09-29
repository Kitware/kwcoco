import pytest


def test_duplicate_single_band_validate():
    import kwcoco

    dset = kwcoco.CocoDataset()

    img = {
        'id': 1,
        'file_name': None,
        'name': 'my_image_name',
        'width': 200,
        'height': 200,
        'auxiliary': [
            {
                'channels': 'B0',
                'file_name': 'path/fake_B0.tif',
                'height': 200,
                'warp_aux_to_img': {
                    'type': 'affine',
                },
                'width': 200,
            },
        ],
    }

    gid = dset.add_image(**img)
    # First validate should pass
    dset.validate(missing=False, fastfail=True)

    # Add the auxiliary item as a duplicate, manually to avoid any API checks
    img = dset.index.imgs[gid]
    obj = img['auxiliary'][0]
    img['auxiliary'].append(obj)

    with pytest.raises(Exception):
        dset.validate(missing=False, fastfail=True)


def test_duplicate_multi_band_validate():
    import kwcoco

    dset = kwcoco.CocoDataset()

    aux1 = {
        'channels': 'red|green|blue',
        'file_name': 'path/fake_msi1.tif',
        'height': 200,
        'width': 200,
        'warp_aux_to_img': {},
    }
    img = {
        'id': 1,
        'file_name': None,
        'name': 'my_image_name',
        'width': 200,
        'height': 200,
        'auxiliary': [aux1],
    }
    gid = dset.add_image(**img)
    # First validate should pass
    dset.validate(missing=False, fastfail=True)

    # Add the auxiliary item as a duplicate, manually to avoid any API checks
    aux2 = {
        'channels': 'B0|B1|green|B3',
        'file_name': 'path/fake_msi2.tif',
        'height': 200,
        'width': 200,
        'warp_aux_to_img': {},
    }
    img = dset.index.imgs[gid]
    img['auxiliary'].append(aux2)

    with pytest.raises(Exception):
        dset.validate(missing=False, fastfail=True)


def test_fastjsonschema_backend_accept_reject_parity():
    """The explicit fast backend should agree on representative cases."""
    import copy

    import pytest

    pytest.importorskip('fastjsonschema')
    import kwcoco

    dset = kwcoco.CocoDataset()
    gid = dset.add_image(
        id=1, file_name='image.jpg', width=32, height=32, channels='rgb'
    )
    dset.add_category(id=1, name='object')
    dset.add_annotation(
        id=1,
        image_id=gid,
        category_id=1,
        bbox=[0, 0, 10, 10],
        segmentation=[{
            'exterior': [[0, 0], [10, 0], [10, 10], [0, 10]],
            'interiors': [],
        }],
    )

    def schema_errors(data, backend):
        candidate = kwcoco.CocoDataset(copy.deepcopy(data), autobuild=False)
        result = candidate.validate(
            schema=True,
            schema_backend=backend,
            unique=False,
            missing=False,
            corrupted=False,
            channels=False,
            img_attrs=False,
            annot_attrs=False,
            verbose=0,
        )
        return result['errors']

    valid_data = dset.dataset
    assert not schema_errors(valid_data, 'jsonschema')
    assert not schema_errors(valid_data, 'fastjsonschema')

    invalid_cases = []

    bad_image_id = copy.deepcopy(valid_data)
    bad_image_id['images'][0]['id'] = 'not-an-integer'
    invalid_cases.append(bad_image_id)

    bad_bbox = copy.deepcopy(valid_data)
    bad_bbox['annotations'][0]['bbox'] = [0, 0, 'bad-width', 10]
    invalid_cases.append(bad_bbox)

    bad_polygon = copy.deepcopy(valid_data)
    bad_polygon['annotations'][0]['segmentation'] = [['bad-coordinate']]
    invalid_cases.append(bad_polygon)

    for data in invalid_cases:
        assert schema_errors(data, 'jsonschema')
        assert schema_errors(data, 'fastjsonschema')
