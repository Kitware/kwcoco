"""
Basic script to convert VIAME-CSV to kwcoco

References:
    https://viame.readthedocs.io/en/latest/section_links/detection_file_conversions.html
"""
import ubelt as ub
from os.path import dirname, join, isdir
import scriptconfig as scfg


class ConvertConfig(scfg.Config):
    default = {
        'src': scfg.PathList('in.viame.csv'),
        'dst': scfg.Value('out.kwcoco.json'),
        'new_root': None,
        'old_root': None,
        'images': scfg.Value(None, help='image list file or path to image directory if the CSV does not specify image names'),
    }


def coco_from_viame_csv(csv_fpaths, images=None):
    @ub.memoize
    def lazy_image_list():
        if images is None:
            raise Exception('must specify where the image root is')
        if isdir(images):
            image_dpath = images
            all_gpaths = []
            import os
            for root, ds, fs in os.walk(image_dpath):
                IMG_EXT = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
                gpaths = [join(root, f) for f in fs if f.split('.')[-1].lower() in IMG_EXT]
                if len(gpaths) > 1 and len(ds) != 0:
                    raise Exception('Images must be in a leaf directory')
                if len(all_gpaths) > 0:
                    raise Exception('Images cannot be nested ATM')
                all_gpaths += gpaths
            all_gpaths = sorted(all_gpaths)
        else:
            raise NotImplementedError

        return all_gpaths

    import kwcoco
    dset = kwcoco.CocoDataset()
    for csv_fpath in csv_fpaths:
        with open(csv_fpath, 'r') as file:
            text = file.read()
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line and not line.startswith('#')]
        for line in lines:
            parts = line.split(',')
            tid = int(parts[0])
            gname = parts[1]
            frame_index = int(parts[2])

            if gname == '':
                # I GUESS WE ARE SUPPOSED TO GUESS WHAT IMAGE IS WHICH
                gname = lazy_image_list()[frame_index]

            tl_x, tl_y, br_x, br_y = map(float, parts[3:7])
            w = br_x - tl_x
            h = br_y - tl_y
            bbox = [tl_x, tl_y, w, h]
            score = float(parts[7])
            target_len = float(parts[8])

            rest = parts[9:]
            catparts = []
            rest_iter = iter(rest)
            for p in rest_iter:
                if p.startswith('('):
                    catparts.append(p)

            final_parts = list(rest_iter)
            if final_parts:
                raise NotImplementedError

            catnames = rest[0::2]
            catscores = list(map(float, rest[1::2]))

            cat_to_score = ub.dzip(catnames, catscores)
            if cat_to_score:
                catname = ub.argmax(cat_to_score)
                cid = dset.ensure_category(name=catname)
            else:
                cid = None

            gid = dset.ensure_image(file_name=gname, frame_index=frame_index)
            kw = {}
            if target_len >= 0:
                kw['target_len'] = target_len
            if score >= 0:
                kw['score'] = score

            dset.add_annotation(
                image_id=gid, category_id=cid, track_id=tid, bbox=bbox, **kw
            )
    return dset


def main(cmdline=True, **kw):
    config = ConvertConfig(default=kw, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    # TODO: ability to map image ids to agree with another coco file
    csv_fpaths = config['src']
    new_root = config['new_root']
    old_root = config['old_root']
    images = config['images']
    dst_fpath = config['dst']

    dst_root = dirname(dst_fpath)
    dset = coco_from_viame_csv(csv_fpaths, images)
    dset.fpath = dst_fpath
    dset.img_root = dst_root
    try:
        dset.reroot(new_root=new_root, old_root=old_root, check=1)
    except Exception as ex:
        print('Reroot failed')
        print('ex = {!r}'.format(ex))

    print('dset.fpath = {!r}'.format(dset.fpath))
    dset.dump(dset.fpath, newlines=True)


if __name__ == '__main__':
    main()
