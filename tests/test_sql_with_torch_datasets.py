import ubelt as ub
import kwcoco
import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is None:
    DatasetBase = object
else:
    DatasetBase = torch.utils.data.Dataset


class DemoTorchDataset(DatasetBase):
    """
    References:
        https://discuss.pytorch.org/t/dataloader-and-postgres-or-other-sql-an-option/25927/7
    """
    def __init__(self, coco_dset):
        self.coco_dset = coco_dset
        # kwcoco.CocoDataset.coerce(coco_dset)
        self.gids = list(self.coco_dset.imgs.keys())

    def __len__(self):
        return len(self.gids)

    def __getitem__(self, index):
        coco_dset = self.coco_dset

        image_id = self.gids[index]

        coco_image = coco_dset.coco_image(image_id)
        imdata = coco_image.imdelay().finalize()
        annot_ids = coco_dset.index.gid_to_aids[image_id]

        imdata = torch.from_numpy(imdata.astype(np.float32))
        anns = [
            coco_dset.index.anns[annot_id]
            for annot_id in annot_ids
        ]
        item = {
            'imdata': imdata,
            'img': coco_image.img,
            'anns': anns,
        }
        return item

    def make_loader(self, batch_size=1, num_workers=0, shuffle=False,
                    pin_memory=False):
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory, collate_fn=ub.identity,
            worker_init_fn=worker_init_fn)
        return loader


def worker_init_fn(worker_id):
    import torch.utils
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    print(f'[Worker {worker_id}] Initialize')
    self = worker_info.dataset
    if hasattr(self.coco_dset, 'connect'):
        # Reconnect to the backend if we are using SQL
        print(f'[Worker {worker_id}] Reconnecting to the backend SQL Dataset')
        self.coco_dset.connect(readonly=True)
        print(f'self.coco_dset={self.coco_dset}')


def test_torch_dataset_with_sql():
    """
    CommandLine:
        xdoctest -m ~/code/kwcoco/tests/test_sql_with_torch_datasets.py test_torch_dataset_with_sql
    """
    import pytest
    if torch is None:
        pytest.skip('requires torch')
    try:
        import psycopg2  # NOQA
        import sqlalchemy  # NOQA
    except ImportError:
        pytest.skip('psycopg2 and sqlalchemy')

    dct_dset = kwcoco.CocoDataset.coerce('special:vidshapes800')
    sql_dset = dct_dset.view_sql(backend='postgresql')
    # sql_dset = dct_dset.view_sql(backend='sqlite')

    torch_dataset = DemoTorchDataset(sql_dset)

    # OK, It looks like we have to ensure the session is closed before we do a
    # system fork to make the loader workers. We can improve kwcoco to help
    # make this automatic.

    # However, in a forking workflow we probably need to disconnect before we
    # fork otherwise each fork will have a defunct copy of the original session
    # in its memory, even if it reinitializes, and if the fork closes the
    # session then we run the risk of multiple actors attempting to clean up
    # the same resource.
    sql_dset.session.close()
    sql_dset.disconnect()

    loader = torch_dataset.make_loader(batch_size=2, num_workers=2, pin_memory=0)

    batch_iter = iter(loader)

    total = 0
    for batch in ub.ProgIter(batch_iter, total=len(loader), desc='data loading', verbose=3):
        for item in batch:
            imdata = item['imdata']
            total += imdata.sum()
    print(f'total={total}')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/tests/test_sql_with_torch_datasets.py
    """
    test_torch_dataset_with_sql()
