"""
WIP:

Conversions to and from KPF format.


.. code::

    +------------+----------------------------+
    | Packet     | Definition                 |
    +------------+----------------------------+
    | id         | object identifier          |
    +------------+----------------------------+
    | ts         | timestamp                  |
    +------------+----------------------------+
    | tsr        | timestamp range            |
    +------------+----------------------------+
    | loc        | location                   |
    +------------+----------------------------+
    | g          | bounding box               |
    +------------+----------------------------+
    | poly       | polygon                    |
    +------------+----------------------------+
    | conf       | confidence or likelihood   |
    +------------+----------------------------+
    | cset       | set of labels/ likelihoods |
    +------------+----------------------------+
    | act        | activity                   |
    +------------+----------------------------+
    | eval       | evaluation result          |
    +------------+----------------------------+
    | a          | attribute                  |
    +------------+----------------------------+
    | tag        | a packet / domain pair     |
    +------------+----------------------------+
    | kv         | key / value pair           |
    +------------+-+--------------------------+


References:
    https://github.com/Kitware/DIVA/blob/master/doc/manuals/kpf.rst
"""


def coco_to_kpf(coco_dset):
    """
    Example:
        >>> from kwcoco.formats.kpf import *  # NOQA
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('shapes8')
        >>> coco_to_kpf(coco_dset)
    """
    import kwimage
    import ubelt as ub
    domain = 0
    meta_id = {'meta': 'id{} : coco annotations'.format(domain)}
    meta_g = {'meta': 'g{} : tl_x, tl_y, br_x, br_y'.format(domain)}
    print('{}'.format(ub.urepr(meta_g, nl=0)))
    print('{}'.format(ub.urepr(meta_id, nl=0)))

    for ann in coco_dset.dataset['annotations']:
        geom = {}
        if 'score' in ann:
            cat = coco_dset._resolve_to_cat(ann['category_id'])
            cname = cat['name']
            geom['cset{}'.format(domain)] = {
                cname: ann['score'],
            }
        else:
            cat = coco_dset._resolve_to_cat(ann['category_id'])
            cname = cat['name']
            geom['cset{}'.format(domain)] = {
                cname: 1.0
            }
        box = kwimage.Boxes([ann['bbox']], 'xywh')
        geom['id{}'.format(domain)] = ann['id']
        geom['g{}'.format(domain)] = box.to_ltrb().data[0].tolist()
        geom['ts{}'.format(domain)] = ann['image_id']

        packet = {'geom': geom}
        print('{}'.format(ub.urepr(packet, nl=0)))


class KPFStream:
    ...

    @classmethod
    def demo_packets(cls):
        ...
        import kwutil
        import ubelt as ub
        lines = ub.codeblock(
            '''
            { meta: "cmdline0: run_detector param1 param2..." }
            { meta: "conf0: yolo person detector" }
            { meta: "id0 domain: yolo person detector" }
            { meta: "cmdline1: run_linker param1 param2..." }
            { meta: "id1 domain: track linker hash 0x85913" }
            { meta: "cmdline2: score_tracks param1 param2..." }
            { meta: "overall track pd/fa count: 0.5 / 1" }
            { meta: "eval0 domain against id0" }
            { meta: "eval1 domain against id1" }
            { meta: "id2 domain false negatives from official_ground_truth.kpf" }
            { geom: { id0: 0, ts0: 101, g0: 515 419 525 430, conf0: 0.8, id1: 100, eval0: tp, eval1: tp } }
            { geom: { id0: 1, ts0: 101, g0: 413 303 423 313, conf0: 0.3, id1: 102, eval0: fa, eval1: fa } }
            { geom: { id0: 2, ts0: 102, g0: 517 421 527 432, conf0: 0.7, id1: 100, eval0: fa, eval1: tp } }
            { geom: { id0: 3, ts0: 102, g0: 416 304 421 315, conf0: 0.2, id1: 102, eval0: fa, eval1: tp } }
            { geom: { id2: 0, ts0: 101, g0: 600 550 605 610, eval0: fn, eval1: fn } }
            { geom: { id2: 1, ts0: 102, g0: 603 553 608 615, eval0: fn, eval1: fn } }
            ''').split('\n')
        packets = [kwutil.Yaml.coerce(line) for line in lines if line]
        return packets


# class kpf_to_coco(packets):
#     ...


def demo():
    dataset = {
        "categories": [
            {"id": 0, "name": "background"},
            {"name": "star", "id": 3, "supercategory": "vector"},
            {"name": "superstar", "id": 6, "supercategory": "raster"},
            {"name": "eff", "id": 7, "supercategory": "raster"}
            ],
        "images": [
            {"width": 600, "height": 600, "id": 1, "file_name": "images/img_00001.png"},
            {"width": 600, "height": 600, "id": 2, "file_name": "images/img_00002.png"},
            {"width": 600, "height": 600, "id": 3, "file_name": "images/img_00003.png"}
            ],
        "annotations": [
            {"bbox": [234, 283, 162, 63], "id": 1, "image_id": 1, "category_id": 6},
            {"bbox": [195, 349, 60, 39], "id": 2, "image_id": 2, "category_id": 7},
            {"bbox": [297, 307, 51, 109], "id": 3, "image_id": 2, "category_id": 7},
            {"bbox": [408, 456, 37, 71], "id": 4, "image_id": 2, "category_id": 3},
            {"bbox": [298, 224, 105, 39], "id": 6, "image_id": 2, "category_id": 7},
            {"bbox": [293, 61, 136, 54], "id": 21, "image_id": 3, "category_id": 3},
            {"bbox": [74, 141, 62, 122], "id": 22, "image_id": 3, "category_id": 7},
            {"bbox": [224, 384, 127, 137], "id": 23, "image_id": 3, "category_id": 6}
        ]
    }
    import kwcoco
    coco_dset = kwcoco.CocoDataset(dataset)
