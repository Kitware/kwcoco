"""
Basic script to convert kwcoco to VIAME-CSV

References:
    https://viame.readthedocs.io/en/latest/section_links/detection_file_conversions.html
"""

import scriptconfig as scfg


class ConvertConfig(scfg.Config):
    default = {
        'src': scfg.Value('in.mscoco.json'),
        'dst': scfg.Value('out.viame.csv'),
    }


def main(**kw):
    """
    CommandLine:
        python $HOME/code/bioharn/dev/kwcoco_to_viame_csv.py \
            --src /data/public/Aerial/US_ALASKA_MML_SEALION/2007/sealions_2007_v9.kwcoco.json \
            --dst /data/public/Aerial/US_ALASKA_MML_SEALION/2007/sealions_2007_v9.viame.csv
    """
    config = ConvertConfig(default=kw, cmdline=True)

    import kwcoco
    import kwimage
    import ubelt as ub
    coco_dset = kwcoco.CocoDataset(config['src'])

    csv_rows = []
    for gid, img in ub.ProgIter(coco_dset.imgs.items(), total=coco_dset.n_images):
        gname = img['file_name']
        aids = coco_dset.gid_to_aids[gid]

        frame_index = img.get('frame_index', 0)
        # vidid = img.get('video_id', None)

        for aid in aids:
            ann = coco_dset.anns[aid]
            cat = coco_dset.cats[ann['category_id']]
            catname = cat['name']

            # just use annotation id if no tracks
            tid = ann.get('track_id', aid)
            # tracked_aids = tid_to_aids.get(tid, [aid])
            # track_len = len(tracked_aids)

            tl_x, tl_y, br_x, br_y = kwimage.Boxes([ann['bbox']], 'xywh').toformat('tlbr').data[0].tolist()

            score = ann.get('score', 1)

            row = [
                 tid,             # 1 - Detection or Track Unique ID
                 gname,           # 2 - Video or Image String Identifier
                 frame_index,     # 3 - Unique Frame Integer Identifier
                 round(tl_x, 3),  # 4 - TL-x (top left of the image is the origin: 0,0
                 round(tl_y, 3),  # 5 - TL-y
                 round(br_x, 3),  # 6 - BR-x
                 round(br_y, 3),  # 7 - BR-y
                 score,           # 8 - Auxiliary Confidence (how likely is this actually an object)
                 -1,              # 9 - Target Length
                 catname,         # 10+ - category name
                 score,           # 11+ - category score
            ]

            # Optional fields
            for kp in ann.get('keypoints', []):
                if 'keypoint_category_id' in kp:
                    cname = coco_dset._resolve_to_kpcat(kp['keypoint_category_id'])['name']
                elif 'category_name' in kp:
                    cname = kp['category_name']
                elif 'category' in kp:
                    cname = kp['category']
                else:
                    raise Exception(str(kp))
                kp_x, kp_y = kp['xy']
                row.append('(kp) {} {} {}'.format(
                    cname, round(kp_x, 3), round(kp_y, 3)))

            note_fields = [
                'box_source',
                'changelog',
                'color',
            ]
            for note_key in note_fields:
                if note_key in ann:
                    row.append('(note) {}: {}'.format(note_key, repr(ann[note_key]).replace(',', '<comma>')))

            row = list(map(str, row))
            for item in row:
                if ',' in row:
                    print('BAD row = {!r}'.format(row))
                    raise Exception('comma is in a row field')

            row_str = ','.join(row)
            csv_rows.append(row_str)

    csv_text = '\n'.join(csv_rows)
    dst_fpath = config['dst']
    print('dst_fpath = {!r}'.format(dst_fpath))
    with open(dst_fpath, 'w') as file:
        file.write(csv_text)

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/kwcoco_to_viame_csv.py
    """
    main()
