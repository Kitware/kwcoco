import numpy as np


def test_normalize_intensity_preserves_display_ranges():
    from kwcoco._dataset_components.mixin_draw import (
        _normalize_intensity_if_needed,
    )

    uint8_canvas = np.array([0, 1, 127, 255], dtype=np.uint8)
    float_canvas = np.array([0.0, 0.25, 1.0], dtype=np.float32)

    assert _normalize_intensity_if_needed(uint8_canvas) is uint8_canvas
    assert _normalize_intensity_if_needed(float_canvas) is float_canvas


def test_draw_image_returns_opencv_compatible_canvas():
    import kwcoco

    dset = kwcoco.CocoDataset.demo('shapes8')
    canvas = dset.draw_image(1)

    assert canvas.dtype == np.uint8
    assert canvas.ndim == 3
