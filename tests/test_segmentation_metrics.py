from __future__ import annotations


def test_draw_chunked_confusion_accepts_default_config(monkeypatch):
    import numpy as np

    from kwcoco.metrics import segmentation_metrics as module

    class _DelayedImage:
        def finalize(self):
            return np.zeros((1, 2, 3), dtype=np.float32)

    class _Channels:
        spec = ''

    class _CocoImage:
        img = {'id': 1}
        channels = _Channels()

        def primary_asset(self):
            return {'channels': ''}

        def imdelay(self, *args, **kwargs):
            return _DelayedImage()

    def fake_draw_header_text(image, text='', **kwargs):
        return np.zeros((1, image['width'], 3), dtype=np.uint8)

    def fake_stack_images(images, axis=0, **kwargs):
        return np.concatenate(list(images), axis=axis)

    monkeypatch.setattr(module, 'build_image_header_text', lambda **kwargs: [])
    monkeypatch.setattr(module.kwimage, 'draw_header_text', fake_draw_header_text)
    monkeypatch.setattr(module.kwimage, 'stack_images', fake_stack_images)
    monkeypatch.setattr(module.kwimage, 'normalize_intensity', lambda data: data)
    monkeypatch.setattr(module.kwimage, 'fill_nans_with_checkers', lambda data: data)
    monkeypatch.setattr(
        module.kwimage,
        'ensure_uint255',
        lambda data: np.asarray(data, dtype=np.uint8),
    )

    chunk_info = [{'row': {}, 'shape': (1, 2)}]
    canvas, plot_info = module.draw_chunked_confusion(
        full_classes=None,
        true_coco_imgs=[_CocoImage()],
        chunk_info=chunk_info,
    )

    assert canvas.shape == (3, 2, 3)
    assert plot_info['vidname_part'] == '_loose_images'
