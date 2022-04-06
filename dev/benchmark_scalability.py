
def main():
    """
    Demo scalability of the system (requires lots of disk space)
    """
    import ubelt as ub
    import kwcoco
    import timerit
    import pint
    import random

    # Prepare data
    num_videos_basis = [1, 5, 100, 1000, 10_000, 60_000, 100_000, 1_000_000]
    # num_videos_basis = [1, 5, 100, 1000, 10_000]
    # num_videos_basis = [1, 5, 100, 1000]
    json_fpaths = []
    sql_fpaths = []
    for num_videos in num_videos_basis:
        print('num_videos = {!r}'.format(num_videos))
        dset = kwcoco.CocoDataset.demo('vidshapes', num_videos=num_videos, num_frames=5, render=False, num_tracks=0, verbose=3)
        json_fpaths.append(dset.fpath)
        sql_dset = dset.view_sql()
        sql_fpaths.append(sql_dset.fpath)

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

    backend_to_paths = {
        'json': json_fpaths,
        'sql': sql_fpaths,
    }
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

        if 0:
            task = 'iter-images'
            for timer in ti.reset(f'{backend}-{task}-{num_images}'):
                with timer:
                    for obj in dset.images().objs:
                        pass
            log_task(task, ti, common)

        task = 'random-image-by-index'
        for timer in ti.reset(f'{backend}-{task}-{num_images}'):
            with timer:
                for _ in range(100):
                    idx = random.randint(0, num_images - 1)
                    dset.dataset['images'][idx]
        log_task(task, ti, common)

        task = 'random-image-by-gid'
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

    unique_tasks = df['task'].unique()

    pnum_ = kwplot.PlotNums(nSubplots=len(unique_tasks) + 1)
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

    ax = kwplot.figure(fnum=fnum, pnum=pnum_()).gca()
    size_df = df[df['task'] == 'load']
    size_df['size'] = [s.m for s in size_df['size']]
    xlabel = 'num_images'
    ylabel = 'size'
    ax = sns.lineplot(data=size_df, x=xlabel, y=ylabel, marker='o', ax=ax, **plotkw)
    ax.set_yscale('log')
    ax.set_title('file-size')
