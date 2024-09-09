"""
Requires the data is downloaded and unpacked. Might cleanup more later.
"""


def main():
    import ubelt as ub
    repo_dpath = ub.Path('.')
    dpath = repo_dpath / 'data'

    if 0:
        zip_names = ['gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip']
        for name in zip_names:
            zip_fpath = dpath / name
            unzip_dpath = (dpath / zip_fpath.stem).ensuredir()
            if not unzip_dpath.ls():
                ub.cmd(f'unzip {zip_fpath} -d {unzip_dpath}', verbose=3)

    img_root = dpath / 'leftImg8bit_trainvaltest/leftImg8bit/'
    for split in ['train', 'test', 'val']:
        split_gt_dpath = dpath / 'gtFine_trainvaltest/gtFine' / split
        split_img_root = img_root / split
        dset = convert_cityscape_split(split_gt_dpath, split_img_root)

        dset.fpath = repo_dpath.absolute() / f'{split}.kwcoco.zip'
        dset.reroot(dset.fpath.parent, absolute=False)
        dset.dump()


def convert_cityscape_split(split_gt_dpath, split_img_root):
    import json
    import kwcoco
    import kwutil
    dset = kwcoco.CocoDataset()
    pman = kwutil.ProgressManager()
    with pman:
        for vid_dpath in pman.ProgIter(split_gt_dpath.ls(), verbose=3):
            vidname = vid_dpath.name
            video_id = dset.add_video(vidname)
            json_fpaths = list(vid_dpath.glob('*.json'))
            for json_fpath in pman.ProgIter(json_fpaths, desc='read json'):
                rel_json_fpath = json_fpath.relative_to(split_gt_dpath)
                rel_img_fpath = rel_json_fpath.parent / rel_json_fpath.stem.rsplit('_', 2)[0] + '_leftImg8bit.png'
                rel_img_fpath = split_img_root / rel_img_fpath
                assert rel_img_fpath.exists()

                data = json.loads(json_fpath.read_text())
                img, anns = convert_cityscapes_json(data)
                img['video_id'] = video_id
                img['file_name'] = rel_img_fpath.absolute()
                img['frame_index'] = int(rel_img_fpath.name.split('_')[1])
                image_id = dset.add_image(**img)
                for ann in anns:
                    ann['image_id'] = image_id
                    ann['category_id'] = dset.ensure_category(ann.pop('category_name'))
                    dset.add_annotation(**ann)
    return dset


def convert_cityscapes_json(data):
    import kwimage
    img = {}
    img['height'] = data['imgHeight']
    img['width'] = data['imgWidth']

    anns = []
    for item in data['objects']:
        item['label']
        import numpy as np
        data = kwimage.Polygon.coerce(np.array(item['polygon']))
        ann = {}
        ann['category_name'] = item['label']
        ann['bbox'] = data.box().to_coco()
        ann['segmentation'] = data.to_coco('new')
        anns.append(ann)
    return img, anns


if __name__ == '__main__':
    main()
