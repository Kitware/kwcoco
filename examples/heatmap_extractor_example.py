"""Example: attach simple threshold heatmaps as kwcoco assets.

This script demonstrates how to:
    1. Load a kwcoco dataset (defaults to the ``vidshapes8`` demo).
    2. Build a trivial threshold-based heatmap for each image.
    3. Save the heatmaps to disk using :func:`kwimage.imwrite`.
    4. Register the saved files as new auxiliary assets with optional
       quantization metadata.

Run with ``python -m examples.heatmap_extractor_example --help`` for CLI
options. The CLI is implemented with :mod:`scriptconfig` to match the
project's other tools.
"""

from __future__ import annotations

import numpy as np
import scriptconfig as scfg
import ubelt as ub
import kwcoco
import kwimage


def simple_threshold_heatmap(image: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Return a float heatmap indicating values above ``threshold``.

    Parameters:
        image:
            Array in the range ``[0, 255]`` or ``[0, 1]``. If multi-channel,
            a grayscale conversion is performed first.
        threshold:
            Value in the ``[0, 1]`` range used to select high-valued pixels.
    """
    image = kwimage.ensure_float01(image)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] != 1:
        # Use a simple luminance-weighted grayscale conversion to avoid
        # optional dependencies.
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = (image[..., 0:3] * weights).sum(axis=2)
    else:
        gray = image[..., 0]
    heatmap = (gray > threshold).astype(np.float32)
    return heatmap


def quantize_heatmap(data: np.ndarray, *, old_min=0.0, old_max=1.0, dtype=np.uint8):
    """Quantize a float heatmap into an integer array with metadata."""
    quant_min = np.iinfo(dtype).min
    quant_max = np.iinfo(dtype).max
    scaled = (data - old_min) / (old_max - old_min)
    clipped = np.clip(scaled, 0, 1)
    quantized = np.round(clipped * quant_max).astype(dtype)
    quantization = {
        'orig_min': float(old_min),
        'orig_max': float(old_max),
        'quant_min': int(quant_min),
        'quant_max': int(quant_max),
        'nodata': None,
    }
    return quantized, quantization


def ensure_bundle_dpath(coco: kwcoco.CocoDataset) -> ub.Path:
    """Ensure the dataset has a bundle directory to write into."""
    if coco.bundle_dpath is not None:
        bundle_dpath = ub.Path(coco.bundle_dpath)
    elif coco.fpath is not None:
        bundle_dpath = ub.Path(coco.fpath).resolve().parent
    else:
        bundle_dpath = ub.Path.appdir('kwcoco/heatmap_example').ensuredir()
        coco.bundle_dpath = str(bundle_dpath)
    return bundle_dpath


def extract_heatmaps(coco: kwcoco.CocoDataset, *, channels: str = 'red|green|blue',
                     threshold: float = 0.5, heatmap_channel_name: str = 'heatmap',
                     quantize: bool = True, output_subdir: str = 'heatmaps'):
    """Compute and attach heatmaps to each image in ``coco``.

    The new assets are written inside ``<bundle_dpath>/<output_subdir>``.
    """
    bundle_dpath = ensure_bundle_dpath(coco)
    output_dir = (bundle_dpath / output_subdir).ensuredir()

    for gid in coco.imgs.keys():
        coco_img = coco.coco_image(gid)
        delayed = coco_img.imdelay(channels)
        image = delayed.finalize()

        heatmap = simple_threshold_heatmap(image, threshold=threshold)

        if quantize:
            write_data, quantization = quantize_heatmap(heatmap, old_min=0.0, old_max=1.0)
        else:
            write_data, quantization = heatmap, None

        name = coco_img.img.get('name', str(gid)).replace('/', '_')
        heatmap_fpath = output_dir / f'{name}_{heatmap_channel_name}.png'

        write_kwargs = {}
        if quantization is not None:
            write_kwargs['metadata'] = {'quantization': quantization}

        kwimage.imwrite(heatmap_fpath, write_data, space=None, **write_kwargs)

        try:
            relative_path = heatmap_fpath.relative_to(bundle_dpath)
        except ValueError:
            relative_path = heatmap_fpath

        coco_img.add_asset(
            file_name=str(relative_path),
            width=heatmap.shape[1],
            height=heatmap.shape[0],
            channels=heatmap_channel_name,
            warp_aux_to_img=kwimage.Affine.eye(),
            quantization=quantization,
        )

    return coco


class HeatmapExtractorConfig(scfg.DataConfig):
    """ScriptConfig interface for the heatmap extractor example."""

    __command__ = 'heatmap-extractor-example'

    coco = scfg.Value('special:vidshapes8', position=1,
                      help='Input kwcoco dataset path or special name (e.g. vidshapes8)')
    channels = scfg.Value('red|green|blue', help='Channel string used for extraction')
    threshold = scfg.Value(0.4, help='Threshold in [0, 1] for the toy heatmap')
    heatmap_channel = scfg.Value('heatmap', help='Channel code to assign to the new asset')
    quantize = scfg.Value(True, isflag=True, help='Quantize heatmap before writing')
    output_subdir = scfg.Value('heatmaps', help='Where to place generated heatmaps relative to the bundle')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        try:
            coco = kwcoco.CocoDataset.coerce(config.coco)
        except ModuleNotFoundError as ex:
            # Some demo datasets rely on optional dependencies like ``cv2``.
            # Fall back to a tiny synthetic dataset so the example remains
            # runnable in minimal environments.
            if ex.name != 'cv2':
                raise
            bundle_dpath = ub.Path.appdir('kwcoco/heatmap_example').ensuredir()
            print('Falling back to a synthetic dataset because cv2 is unavailable')
            image_dir = (bundle_dpath / 'images').ensuredir()
            coco = kwcoco.CocoDataset()
            coco.bundle_dpath = str(bundle_dpath)
            for idx in range(3):
                data = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                fpath = image_dir / f'synthetic_{idx}.png'
                kwimage.imwrite(fpath, data)
                coco.add_image(
                    file_name=str(fpath.relative_to(bundle_dpath)),
                    width=data.shape[1],
                    height=data.shape[0],
                    id=idx + 1,
                    name=f'synthetic_{idx}',
                    channels='red|green|blue',
                )
        extract_heatmaps(
            coco,
            channels=config.channels,
            threshold=config.threshold,
            heatmap_channel_name=config.heatmap_channel,
            quantize=config.quantize,
            output_subdir=config.output_subdir,
        )

        first_gid = next(iter(coco.imgs.keys()))
        first_coco_img = coco.coco_image(first_gid)
        assets_key = first_coco_img._assets_key()
        print('Assets on first image:')
        print(ub.urepr(first_coco_img.img.get(assets_key, []), nl=1, sort=True))


def main(argv=True):
    """Run the heatmap extraction demo via :class:`HeatmapExtractorConfig`."""
    return HeatmapExtractorConfig.main(argv=argv)


if __name__ == '__main__':
    main()
