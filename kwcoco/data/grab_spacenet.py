"""
References:
    https://medium.com/the-downlinq/the-spacenet-7-multi-temporal-urban-development-challenge-algorithmic-baseline-4515ec9bd9fe
    https://arxiv.org/pdf/2102.11958.pdf
    https://spacenet.ai/sn7-challenge/
"""
import ubelt as ub
import os
from kwcoco.util import util_archive


def grab_spacenet7(data_dpath):
    """
    References:
        https://spacenet.ai/sn7-challenge/

    Requires:
        awscli

    Ignore:
        mkdir -p $HOME/.cache/kwcoco/data/spacenet/archives
        cd $HOME/.cache/kwcoco/data/spacenet/archives

        # Requires an AWS account
        export AWS_PROFILE=joncrall_at_kitware

        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_csvs.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz .
    """
    import ubelt as ub
    import pathlib
    import kwcoco

    dpath = ub.ensuredir((data_dpath, 'spacenet'))

    coco_fpath = os.path.join(dpath, 'spacenet7.kwcoco.json')
    archive_dpath = pathlib.Path(ub.ensuredir((dpath, 'archives')))
    extract_dpath = pathlib.Path(ub.ensuredir((dpath, 'extracted')))

    stamp = ub.CacheStamp('convert_spacenet', dpath=dpath, depends=['v001'])
    if stamp.expired():
        items = [
            {
                'uri': 's3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz',
                'sha512': '5f810682825859951e55f6a3bf8e96eb6eb85864a90d75349',
            },
            {
                'uri': 's3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train_csvs.tar.gz',
                'sha512': 'e4314ac129dd76e7984556c243b7b5c0c238085110ed7f7f619cb0',
            },
            {
                'uri': 's3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz',
                'sha512': '0677a20f972cc463828bbff8d2fae08e17fdade3cf17ce213dc978',
            },
        ]

        has_extracted = all([
            d.exists() for d in [extract_dpath / 'csvs',
                                 extract_dpath / 'test_public',
                                 extract_dpath / 'train']])

        for item in items:
            fname = pathlib.Path(item['uri']).name
            item['fpath'] = archive_dpath / fname

        need_download_archive = not has_extracted
        if need_download_archive:
            aws_exe = ub.find_exe('aws')
            if not aws_exe:
                raise Exception('requires aws exe')

            for item in items:
                if not item['fpath'].exists():
                    command = '{aws_exe} s3 cp {uri} {archive_dpath}'.format(
                        aws_exe=aws_exe, uri=item['uri'], archive_dpath=archive_dpath)
                    info = ub.cmd(command, verbose=3)
                    assert info['ret'] == 0
                    got_hash = ub.hash_file(item['fpath'], hasher='sha512')
                    assert got_hash.startswith(item['sha512'])

        need_unarchive = not has_extracted
        if need_unarchive:
            for item in ub.ProgIter(items, desc='extract spacenet', verbose=3):
                archive_fpath = item['fpath']
                util_archive.unarchive_file(archive_fpath, extract_dpath, overwrite=0, verbose=2)

        coco_dset = convert_spacenet_to_kwcoco(extract_dpath, coco_fpath)
        stamp.renew()

    coco_dset = kwcoco.CocoDataset(coco_fpath)
    dsets = [coco_dset]
    return dsets


def convert_spacenet_to_kwcoco(extract_dpath, coco_fpath):
    """
    Converts the raw SpaceNet7 dataset to kwcoco

    Note:
        * The "train" directory contains 60 "videos" representing a region over time.

        * Each "video" directory contains :
            * images           - unmasked images
            * images_masked    - images with masks applied
            * labels           - geojson polys in wgs84?
            * labels_match     - geojson polys in wgs84 with track ids?
            * labels_match_pix - geojson polys in pixels with track ids?
            * UDM_masks - unusable data masks (binary data corresponding with an image, may not exist)

        File names appear like:
            "global_monthly_2018_01_mosaic_L15-1538E-1163N_6154_3539_13"

    Ignore:
        dpath = pathlib.Path("/home/joncrall/data/dvc-repos/smart_watch_dvc/extern/spacenet/")
        extract_dpath = dpath / 'extracted'
        coco_fpath = dpath / 'spacenet7.kwcoco.json'
    """
    import kwcoco
    import json
    import kwimage
    import parse
    import datetime
    print('Convert Spacenet7 to kwcoco')

    coco_dset = kwcoco.CocoDataset()
    coco_dset.fpath = coco_fpath

    building_cid = coco_dset.ensure_category('building')
    ignore_cid = coco_dset.ensure_category('ignore')

    s7_fname_fmt = parse.Parser('global_monthly_{year:d}_{month:d}_mosaic_{}')

    # Add images
    tile_dpaths = list(extract_dpath.glob('train/*'))
    for tile_dpath in ub.ProgIter(tile_dpaths, desc='add video'):
        tile_name = tile_dpath.name
        vidid = coco_dset.add_video(name=tile_name)

        image_gpaths = sorted(tile_dpath.glob('images/*'))
        # sorted(tile_dpath.glob('labels/*'))
        # sorted(tile_dpath.glob('images_masked/*'))
        # sorted(tile_dpath.glob('labels_match/*'))
        # udm_fpaths = sorted(tile_dpath.glob('UDM_masks/*'))

        for frame_index, gpath in enumerate(image_gpaths):
            gname = str(gpath.stem)
            nameinfo = s7_fname_fmt.parse(gname)
            timestamp = datetime.datetime(
                year=nameinfo['year'], month=nameinfo['month'], day=1)
            gid = coco_dset.add_image(
                file_name=str(gpath.relative_to(coco_dset.bundle_dpath)),
                name=gname,
                video_id=vidid,
                frame_index=frame_index,
                date_captured=timestamp.isoformat(),
                channels='r|g|b',
            )

    coco_dset._ensure_imgsize()

    # Add annotations

    def _from_geojson2(geometry):
        import numpy as np
        coords = geometry['coordinates']
        exterior = np.array(coords[0])[:, 0:2]
        interiors = [np.array(h)[:, 0:2] for h in coords[1:]]
        poly_data = dict(exterior=kwimage.Coords(exterior),
                         interiors=[kwimage.Coords(hole)
                                    for hole in interiors])
        self = kwimage.Polygon(data=poly_data)
        return self

    all_label_fpaths = sorted(extract_dpath.glob('train/*/labels_match_pix/*'))
    for label_fpath in ub.ProgIter(all_label_fpaths, desc='add annots'):
        # Remove trailing suffix
        name_parts = label_fpath.stem.split('_')
        assert name_parts[-1] == 'Buildings'
        name = '_'.join(name_parts[:-1])
        with open(label_fpath, 'r') as file:
            label_data = json.load(file)

        assert label_data['type'] == 'FeatureCollection'
        for feat in label_data['features']:
            prop = feat['properties']
            gid = coco_dset.index.name_to_img[name]['id']

            # from_geojson is slow!
            # poly = kwimage.Polygon.from_geojson(feat['geometry'])
            poly = _from_geojson2(feat['geometry'])

            # This is a bottleneck
            boxes = poly.bounding_box()
            boxes = boxes.quantize()
            xywh = boxes.to_xywh().data[0].tolist()

            ann = {
                'bbox': xywh,
                'image_id': gid,
                'category_id': building_cid,
                'track_id': prop['Id'],
                'area': prop['area'],
                'segmentation': poly.to_coco(style='new')
            }
            coco_dset.add_annotation(**ann)

    all_udm_fpaths = sorted(extract_dpath.glob('train/*/UDM_masks/*'))
    for udm_fpath in ub.ProgIter(all_udm_fpaths, desc='add ignore masks'):
        name_parts = udm_fpath.stem.split('_')
        assert name_parts[-1] == 'UDM'
        name = '_'.join(name_parts[:-1])

        gid = coco_dset.index.name_to_img[name]['id']
        c_mask = kwimage.imread(str(udm_fpath))
        c_mask[c_mask == 255] = 1
        mask = kwimage.Mask(c_mask, 'c_mask')
        poly = mask.to_multi_polygon()
        xywh = ub.peek(poly.bounding_box().quantize().to_coco())
        ann = {
            'bbox': xywh,
            'image_id': gid,
            'category_id': ignore_cid,
            'segmentation': poly.to_coco(style='new')
        }
        coco_dset.add_annotation(**ann)

    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    print('coco_dset = {!r}'.format(coco_dset))
    coco_dset.dump(str(coco_dset.fpath))

    # We will generally want an SQL cache when working with this dataset
    if ub.argflag('--sql-hack'):
        from kwcoco.coco_sql_dataset import CocoSqlDatabase
        CocoSqlDatabase.coerce(coco_dset)

    return coco_dset


def main():
    data_dpath = ub.ensure_app_cache_dir('kwcoco', 'data')
    grab_spacenet7(data_dpath)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/data/grab_spacenet.py
    """
    main()
