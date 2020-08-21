"""
Basic script to convert VIAME-CSV to kwcoco

References:
    https://viame.readthedocs.io/en/latest/section_links/detection_file_conversions.html
"""
import ubelt as ub
from os.path import dirname
import scriptconfig as scfg


class ConvertConfig(scfg.Config):
    default = {
        'src': scfg.PathList('in.viame.csv'),
        'dst': scfg.Value('out.kwcoco.json'),
        'new_root': None,
        'old_root': None,
    }


def coco_from_viame_csv(csv_fpaths):
    import kwcoco
    dset = kwcoco.CocoDataset()
    for csv_fpath in csv_fpaths:
        with open(csv_fpath, 'r') as file:
            text = file.read()
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line and not line.startswith('#')]
        for line in lines:
            parts = line.split(',')
            tid = parts[0]
            gname = parts[1]

            frame_index = parts[2]
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
            catname = ub.argmax(cat_to_score)

            gid = dset.ensure_image(file_name=gname, frame_index=frame_index)
            cid = dset.ensure_category(name=catname)

            dset.add_annotation(
                image_id=gid,
                category_id=cid,
                track_id=int(tid),
                bbox=bbox, score=score,
                target_length=target_len,
            )
    return dset


def main(cmdline=True, **kw):
    config = ConvertConfig(default=kw, cmdline=cmdline)
    # TODO: ability to map image ids to agree with another coco file
    csv_fpaths = config['src']
    dset = coco_from_viame_csv(csv_fpaths)
    dset.fpath = config['dst']
    dset.img_root = dirname(dset.fpath)

    dset.reroot(
        new_root=config['new_root'], old_root=config['old_root'],
        check=0)

    print('dset.fpath = {!r}'.format(dset.fpath))
    dset.dump(dset.fpath, newlines=True)


if __name__ == '__main__':
    main()
