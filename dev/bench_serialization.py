
def benchmark_serialization_formats():
    """
    msgpack writes are ~20x faster than json
    msgpack reads are ~2x faster than json
    """
    import timerit
    ti = timerit.Timerit(10, bestof=3, verbose=1, unit='us')

    import kwcoco
    import msgpack
    import json
    coco_dset = kwcoco.CocoDataset.demo('shapes8')

    for timer in ti.reset('msgpack-write'):
        with timer:
            with open('test.mscoco.msgpack', 'wb') as file:
                msgpack.dump(coco_dset.dataset, file)

    for timer in ti.reset('msgpack-read'):
        with timer:
            with open('test.mscoco.msgpack', 'rb') as file:
                recon_msg = msgpack.load(file)

    for timer in ti.reset('json-write'):
        with timer:
            with open('test.mscoco.json', 'w') as file:
                json.dump(coco_dset.dataset, file)

    for timer in ti.reset('json-read'):
        with timer:
            with open('test.mscoco.json', 'r') as file:
                recon_json = json.load(file)

    if 0:
        # Very slow
        import amazon.ion.simpleion as ion
        for timer in ti.reset('ion-write'):
            with timer:
                with open('test.mscoco.ion', 'wb') as file:
                    ion.dump(coco_dset.dataset, file)

        for timer in ti.reset('ion-read'):
            with timer:
                with open('test.mscoco.ion', 'rb') as file:
                    recon_ion = ion.load(file)
