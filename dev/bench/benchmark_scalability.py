import kwcoco
import ubelt as ub
import timerit
import pint
import random
import kwarray


def _iter_images_worker(dset):
    gids = list(dset.images())
    gids = kwarray.shuffle(gids)
    for gid in gids:
        dset.index.imgs[gid]


USE_TORCH = 1
if USE_TORCH:
    import torch
    DatasetBase = torch.utils.data.Dataset
    class DemoTorchDataset(DatasetBase):
        def __init__(self, coco_dset):
            import kwcoco
            self.coco_dset = kwcoco.CocoDataset.coerce(coco_dset)
            self.gids = list(self.coco_dset.imgs.keys())

        def __len__(self):
            return len(self.gids)

        def __getitem__(self, index):
            gid = self.gids[index]
            img = self.coco_dset.index.imgs[gid]
            return img

        def make_loader(self, batch_size=1, num_workers=0, shuffle=False,
                        pin_memory=False):
            loader = torch.utils.data.DataLoader(
                self, batch_size=batch_size, num_workers=num_workers,
                shuffle=shuffle, pin_memory=pin_memory, collate_fn=ub.identity, worker_init_fn=worker_init_fn)
            return loader

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        self = worker_info.dataset
        if hasattr(self.coco_dset, 'connect'):
            # Reconnect to the backend if we are using SQL
            self.coco_dset.connect(readonly=True)


def main():
    """
    Demo / test scalability of the system (requires lots of disk space)

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/kwcoco/dev'))
        from benchmark_scalability import *  # NOQA
        from benchmark_scalability import _iter_images_worker
    """

    # Prepare data
    # num_videos_basis = [1, 5, 100, 1000, 10_000, 60_000, 100_000, 1_000_000]
    # num_videos_basis = [1, 5, 100, 1000, 10_000, 60_000, 100_000]
    # num_videos_basis = [1, 5, 100, 1000, 10_000, 60_000]
    # num_videos_basis = [1, 5, 100, 1000, 10_000]

    num_videos_basis = [1, 32, 64, 256]
    backend_to_paths = ub.ddict(list)
    for num_videos in num_videos_basis:
        print('num_videos = {!r}'.format(num_videos))
        dset = kwcoco.CocoDataset.demo('vidshapes', num_videos=num_videos, num_frames=5, render=False, num_tracks=0, verbose=3)
        backend_to_paths['json'].append(dset.fpath)
        backend_to_paths['sqlite'].append(dset.view_sql(backend='sqlite').fpath)
        # backend_to_paths['postgresql'].append(dset.view_sql(backend='postgresql').fpath)

    # if True:
    #     # Hack
    #     dpath = ub.Path.appdir('kwcoco/demo/benchmarks')
    #     dpath.ensuredir()
    #     base_dset = dset
    #     base_hashid = base_dset._cached_hashid()

    #     for i in ub.ProgIter(range(3), desc='hack bigger'):
    #         bigger = kwcoco.CocoDataset.union(dset, dset)
    #         dset = bigger
    #         dset.fpath = str(dpath / f'union_{base_hashid}_{i}.kwcoco.json')
    #         dset.dump(dset.fpath)
    #         json_fpaths.append(dset.fpath)

    #         sql_dset = dset.view_sql()
    #         sql_fpaths.append(sql_dset.fpath)

    ureg = pint.UnitRegistry()

    # Run benchmarks
    ti = timerit.Timerit(1, bestof=1, verbose=2)
    measures = []

    benchmark_grid = []
    for backend, paths in backend_to_paths.items():
        for fpath in paths:
            size_bytes = ub.Path(fpath).stat().st_size
            benchmark_grid.append({
                'backend': backend,
                'fpath': fpath,
                'size_bytes': size_bytes,
                'size': round((size_bytes * ureg.byte).to('gigabyte'), 2),
            })

    def log_task(task, ti, common):
        row = {
            'label': ti.label,
            'task': task,
            'mean_seconds': ti.mean(),
            'min_seconds': ti.min(),
            **common,
        }
        measures.append(row)
        return row

    for gridkw in ub.ProgIter(benchmark_grid, desc='benchmarks', verbose=3):

        fpath = gridkw['fpath']
        backend = gridkw['backend']
        size = gridkw['size']

        task = 'load'
        for timer in ti.reset(f'{backend}-{task}-{size}'):
            with timer:
                dset = kwcoco.CocoDataset.coerce(fpath)
        num_images = dset.n_images
        gids = list(dset.imgs.keys())
        common = {
            **gridkw,
            'num_images': num_images,
        }
        log_task(task, ti, common)

        if USE_TORCH:
            task = 'iter-torch'
            for timer in ti.reset(f'{backend}-{task}-{num_images}'):
                with timer:
                    torch_dset = DemoTorchDataset(dset)
                    loader = torch_dset.make_loader(num_workers=4)
                    for batch in loader:
                        pass

            log_task(task, ti, common)

        if 0:
            task = 'iter-shuffled-images-worker'
            for timer in ti.reset(f'{backend}-{task}-{num_images}'):
                with timer:
                    pool = ub.JobPool(mode='process', max_workers=4)
                    pool.submit(_iter_images_worker, dset)
                    pool.submit(_iter_images_worker, dset)
                    pool.submit(_iter_images_worker, dset)
                    pool.submit(_iter_images_worker, dset)
                    for job in pool.as_completed():
                        job.result()

            log_task(task, ti, common)

        if 0:
            task = 'iter-images'
            for timer in ti.reset(f'{backend}-{task}-{num_images}'):
                with timer:
                    for obj in dset.images().objs:
                        pass
            log_task(task, ti, common)

        if 0:
            task = 'random-image-by-index-serial'
            for timer in ti.reset(f'{backend}-{task}-{num_images}'):
                with timer:
                    for _ in range(100):
                        idx = random.randint(0, num_images - 1)
                        dset.dataset['images'][idx]
            log_task(task, ti, common)

        if 1:
            task = 'random-image-by-gid-serial'
            for timer in ti.reset(f'{backend}-{task}-{num_images}'):
                with timer:
                    for _ in range(100):
                        gid = random.choice(gids)
                        dset.index.imgs[gid]
            log_task(task, ti, common)

    import pandas as pd
    df = pd.DataFrame(measures)
    print(df.to_string())

    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()

    unique_tasks = df['task'].unique()

    pnum_ = kwplot.PlotNums(nSubplots=len(unique_tasks))
    fnum = 1
    fig = kwplot.figure(fnum=fnum)
    fig.clf()

    for task in unique_tasks:
        xlabel = 'num_images'
        ylabel = 'mean_seconds'
        plotkw = {
            'hue': 'backend',
        }
        task_df = df[df['task'] == task]
        ax = kwplot.figure(fnum=fnum, pnum=pnum_()).gca()
        ax = sns.lineplot(data=task_df, x=xlabel, y=ylabel, marker='o', ax=ax, **plotkw)
        ax.set_yscale('log')
        ax.set_title(task)

    if 0:
        ax = kwplot.figure(fnum=fnum, pnum=pnum_()).gca()
        size_df = df[df['task'] == 'load']
        size_df['size'] = [s.m for s in size_df['size']]
        xlabel = 'num_images'
        ylabel = 'size'
        ax = sns.lineplot(data=size_df, x=xlabel, y=ylabel, marker='o', ax=ax, **plotkw)
        ax.set_yscale('log')
        ax.set_title('file-size')
    plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/dev/bench/benchmark_scalability.py
    """
    main()
