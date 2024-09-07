"""
WIP:

Conversions to and from KPF format.
"""


def coco_to_kpf(coco_dset):
    """
    import kwcoco
    coco_dset = kwcoco.CocoDataset.demo('shapes8')
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
