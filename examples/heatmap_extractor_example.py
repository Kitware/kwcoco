r"""
Example: attach simple heatmaps as kwcoco assets.

This is an *illustrative* script that shows how to:
    1. Load a kwcoco dataset (defaults to the ``vidshapes8`` demo).
    2. Compute a toy heatmap per image via :func:`_predict_image_heatmap`.
    3. Write heatmaps to disk (float32 -> tif, uint8 -> png).
    4. Register the written files as kwcoco assets, optionally with quantization
       metadata so downstream pipelines can re-expand to float.

Assuing you are in the repo root, running the example:

.. code:: bash

    python ./examples/heatmap_extractor_example.py \
        --coco_fpath=special:vidshapes8 \
        --dst_coco_fpath=$HOME/.cache/kwcoco/heatmap_example/out.kwcoco.json \
        --asset_dpath=$HOME/.cache/kwcoco/heatmap_example/assets/heatmaps \
        --heatmap_channel=salient \
        --sigma=7 \
        --thresh=0.3 \
        --quantize=1

And check stats / visualize the results:

.. code:: bash

    kwcoco stats $HOME/.cache/kwcoco/heatmap_example/out.kwcoco.json --channels=True

    kwcoco show $HOME/.cache/kwcoco/heatmap_example/out.kwcoco.json --channels=salient

"""
from __future__ import annotations
import numpy as np
import scriptconfig as scfg
import ubelt as ub
import kwarray
import kwcoco
import kwimage


class HeatmapExtractorConfig(scfg.DataConfig):
    """
    CLI options for the illustrative heatmap extractor.

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/kwcoco/examples'))
        >>> from heatmap_extractor_example import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> dpath = ub.Path.appdir('kwcoco/examples/heatmap_extractor').ensuredir()
        >>> kwargs = {
        ...     'coco_fpath': dset.fpath,
        ...     'dst_coco_fpath': dpath / 'out.kwcoco.json',
        ...     'asset_dpath': dpath / 'assets/heatmaps',
        ...     'heatmap_channel': 'salient',
        ...     'sigma': 7.0,
        ...     'thresh': 0.3,
        ...     'quantize': True,
        ... }
        >>> HeatmapExtractorConfig.main(argv=False, **kwargs)
        >>> assert (dpath / 'out.kwcoco.json').exists()
    """
    coco_fpath = scfg.Value('special:vidshapes8', position=1,
                            help='Input kwcoco dataset path or special name')
    dst_coco_fpath = scfg.Value("heatmap.kwcoco.json", help="Output kwcoco file")
    asset_dpath = scfg.Value("assets/heatmaps",
                             help="Where to store written heatmaps (ideally next to dst_coco_fpath)")
    heatmap_channel = scfg.Value("salient", help="Name of the output heatmap channel")
    sigma = scfg.Value(7.0, help="Gaussian blur applied to whiteness response")
    thresh = scfg.Value(0.3, help="Threshold floor for minimum heatmap value")
    quantize = scfg.Value(True, isflag=True, help="Quantize (uint8/png) instead of float32/tif")

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose="auto")

        # --- inlined "run_heatmap_extractor" logic ---
        src_coco = kwcoco.CocoDataset.coerce(config.coco_fpath)

        pred_coco = src_coco.copy()
        pred_coco.reroot(absolute=True)
        pred_coco.fpath = str(config.dst_coco_fpath)

        dst_parent = ub.Path(config.dst_coco_fpath).parent
        pred_coco.bundle_dpath = str(dst_parent)

        asset_dpath = ub.Path(config.asset_dpath).ensuredir()

        for image_id in ub.ProgIter(list(pred_coco.imgs.keys()), desc="write heatmaps"):
            coco_img = pred_coco.coco_image(image_id)

            heatmap = _predict_image_heatmap(
                coco_img,
                sigma=float(config.sigma),
                thresh=float(config.thresh),
            )

            img_name = coco_img.img.get("name", f"image-{image_id}")
            stem = ub.Path(img_name).stem

            if bool(config.quantize):
                # uint8 -> png (+ quantization metadata)
                write_data, quantization = quantize_heatmap(
                    heatmap, old_min=0.0, old_max=1.0, dtype=np.uint8)
                heatmap_fname = f"{stem}_{config.heatmap_channel}.png"
                write_kwargs = {}
            else:
                # float32 -> tif (no quantization metadata)
                write_data, quantization = heatmap.astype(np.float32, copy=False), None
                heatmap_fname = f"{stem}_{config.heatmap_channel}.tif"
                write_kwargs = {}

            heatmap_fpath = asset_dpath / heatmap_fname
            kwimage.imwrite(heatmap_fpath, write_data, **write_kwargs)

            # Prefer relative paths if asset is inside the dst bundle
            try:
                rel_path = ub.Path(heatmap_fpath).relative_to(dst_parent)
            except Exception:
                rel_path = ub.Path(heatmap_fpath)

            coco_img.add_asset(
                file_name=str(rel_path),
                channels=config.heatmap_channel,
                width=heatmap.shape[1],
                height=heatmap.shape[0],
                quantization=quantization,
                warp_aux_to_img=kwimage.Affine.eye(),
            )

        pred_coco.dump(config.dst_coco_fpath, newlines=True)
        print(f"Wrote {config.dst_coco_fpath}")
        return pred_coco


def quantize_heatmap(data: np.ndarray, *, old_min=0.0, old_max=1.0, dtype=np.uint8):
    """
    Quantize a float heatmap into an integer array with kwcoco-friendly metadata.

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/kwcoco/examples'))
        >>> from heatmap_extractor_example import *  # NOQA
        >>> data = np.linspace(0, 1, 11, dtype=np.float32)[None, :]
        >>> q, meta = quantize_heatmap(data, old_min=0.0, old_max=1.0)
        >>> assert q.dtype == np.uint8
        >>> assert meta['orig_min'] == 0.0
        >>> assert meta['orig_max'] == 1.0
        >>> assert meta['quant_min'] == 0
        >>> assert meta['quant_max'] == 255
    """
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


def _predict_image_heatmap(coco_img, *, sigma: float, thresh: float) -> np.ndarray:
    """
    Compute a toy "white-blob" saliency heatmap for a single image.

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/kwcoco/examples'))
        >>> from heatmap_extractor_example import *  # NOQA
        >>> from heatmap_extractor_example import _predict_image_heatmap
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> coco_img = dset.coco_image(1)
        >>> smooth = _predict_image_heatmap(coco_img, sigma=7, thresh=0.0)
        >>> assert smooth.ndim == 2
        >>> assert smooth.shape[0] == coco_img.img['height']
        >>> assert smooth.shape[1] == coco_img.img['width']
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> rgb = coco_img.imdelay().finalize()
        >>> kwplot.imshow(rgb, pnum=(1, 2, 1), title='RGB')
        >>> kwplot.imshow(smooth, pnum=(1, 2, 2), title='White-blob heatmap')
        >>> kwplot.show_if_requested()
    """
    img = coco_img.imdelay().finalize()
    img = kwarray.atleast_nd(img, 3)
    rgb01 = kwimage.ensure_float01(img)

    hsv = kwimage.convert_colorspace(rgb01, src_space="rgb", dst_space="hsv")
    sat = hsv[..., 1]
    val = hsv[..., 2]

    whiteness = val * (1.0 - sat)
    smooth = kwimage.gaussian_blur(whiteness, sigma=float(sigma))

    if thresh is not None:
        smooth = smooth.astype(np.float32)
        smooth[smooth < float(thresh)] = 0.0

    return smooth.astype(np.float32, copy=False)


if __name__ == '__main__':
    HeatmapExtractorConfig.main(argv=True)
