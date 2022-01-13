import xdev


@xdev.profile
def benchmark_large_hyperspectral():
    import ubelt as ub
    # import kwimage
    import kwarray
    import kwcoco
    import numpy as np

    from osgeo import gdal
    gdal.UseExceptions()

    base_dpath = ub.Path(ub.ensure_app_cache_dir('kwcoco/demo/bench_large_hyperspectral'))

    MB = (2.0 ** 20)
    H = W = 1024

    basis = {
        # 'nbands': [3, 16, 32, 64, 128, 256],
        'nbands': [3, 16, 32, 64, 128],
    }

    rows = []

    for kw in ub.named_product(basis):

        # Write a "big" image to disk
        # H = W = 4016
        nbands = kw['nbands']
        imdata = np.random.randint(0, 2 ** 14, size=(H, W, nbands), dtype=np.int16)

        num_bytes = imdata.size * imdata.dtype.itemsize
        num_megabytes = num_bytes / MB
        print('num_megabytes = {!r}'.format(num_megabytes))

        fpath = base_dpath / 'big_img.tif'

        slider = kwarray.SlidingWindow(
            shape=(H, W), window=(64, 64), overlap=0.0,
            keepbound=True, allow_overshoot=True)

        # Using the RAW-COG format should be roughly equivalent to BSQ/BIL
        # kwimage.imwrite(fpath, imdata, compress='RAW', backend='gdal')

        # Compare to `spectral`
        # pip isntall spectral
        import spectral.io.envi
        import spectral  # NOQA
        spy_fpath = base_dpath / 'big_img_{}.hdr'.format(nbands)
        spectral.envi.save_image(str(spy_fpath), imdata, interleave='BSQ', ext='bsq', force=True)
        fpath = spy_fpath

        if 1:
            # Create an empty CocoDataset
            coco_dset = kwcoco.CocoDataset()
            # Set the target location for the kwcoco file so we can use relative paths
            coco_dset.fpath = str(base_dpath / 'data.kwcoco.json')

            img = {
                'name': 'big_img',
                'height': H,
                'width': W,
                'channels': 'hyper:{}'.format(nbands),  # code for channels
                'file_name': str(fpath.relative_to(coco_dset.bundle_dpath))
            }
            # Register the image the CocoDataset
            gid = coco_dset.add_image(**img)

            # Demonstrate how to load a single small slice
            delayed = coco_dset.delayed_load(gid)
            cropped = delayed.crop((slice(432, 432 + 64), slice(432, 432 + 64)))
            final_data = cropped.finalize()

            # Demonstrate how to loop over all slices in the image and time how long it
            # takes.
            # img_shape = (img['height'], img['width'])

            # Loop over the entire image and load in small parts
            # Time how long it takes to do this.
            kwcoco_prog = ub.ProgIter(slider, desc='read-sliding-via-kwcoco')
            for slices in kwcoco_prog:
                delayed = coco_dset.delayed_load(gid)
                cropped = delayed.crop(slices)
                final_data = cropped.finalize()

            rows.append({
                'method': 'kwcoco',
                'nbands': nbands,
                'hz': kwcoco_prog._iters_per_second,
            })

        if 1:
            fpath = spy_fpath.augment(ext='.bsq')
            from kwcoco.util import util_delayed_poc
            gdal_prog = ub.ProgIter(slider, desc='read-sliding-via-gdal')
            for slices in gdal_prog:
                frame = util_delayed_poc.LazyGDalFrameFile(str(fpath))
                final_data = frame[slices]
                frame = None

            rows.append({
                'method': 'gdal',
                'nbands': nbands,
                'hz': gdal_prog._iters_per_second,
            })

        if 1:
            fpath = spy_fpath.augment(ext='.bsq')
            import rasterio
            gdal_prog = ub.ProgIter(slider, desc='read-sliding-via-rasterio')
            for slices in gdal_prog:
                sl_y, sl_x = slices
                rio_ds = rasterio.open(str(fpath), 'r')
                indexes = list(range(1, nbands + 1))
                data = rio_ds.read(indexes=indexes, window=((sl_y.start, sl_y.stop), (sl_x.start, sl_x.stop)))
                final_data = data.transpose(1, 2, 0)

            rows.append({
                'method': 'rasterio',
                'nbands': nbands,
                'hz': gdal_prog._iters_per_second,
            })

        if 1:
            spy_fpath = base_dpath / 'big_img_{}.hdr'.format(nbands)
            spectral.envi.save_image(str(spy_fpath), imdata, interleave='BSQ', ext='bsq', force=True)

            spy_prog = ub.ProgIter(slider, desc='read-sliding-via-spectral')
            for slices in spy_prog:
                spy_img = spectral.envi.open(str(spy_fpath))
                sl_y, sl_x = slices
                final_data = spy_img.read_subregion((sl_y.start, sl_y.stop), (sl_x.start, sl_x.stop))

            rows.append({
                'method': 'spectral',
                'nbands': nbands,
                'hz': spy_prog._iters_per_second,
            })

    # Analysis
    import pandas as pd
    results = pd.DataFrame(rows)
    results['slowdown'] = results.iloc[0].hz / results.hz
    results['growth_factor'] = results.nbands / results.iloc[0].nbands
    results['overhead'] = results['slowdown'] / results['growth_factor']
    results['megabytes'] = results['nbands'] * np.prod(slider.window) * imdata.dtype.itemsize /  MB
    print(results)

    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()
    kwplot.figure(fnum=1, doclf=True)
    ax = sns.lineplot(data=results, x='nbands', hue='method', y='hz')
    ax.set_yscale('log')

    if 1:
        plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python -m kwcoco.examples.bench_large_hyperspectral --profile
    """
    benchmark_large_hyperspectral()
