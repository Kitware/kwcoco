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
