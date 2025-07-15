def test_single_image_sseg_with_weights():
    from kwcoco.metrics.segmentation_metrics import SingleImageSegmentationMetrics
    import kwcoco
    import numpy as np
    import ubelt as ub
    import kwimage
    dpath = ub.Path.appdir('kwcoco/tests/test_sseg_metrics').ensuredir()

    pred_bundle_dpath = (dpath / 'pred_bundle_1').ensuredir()

    # TODO: kwcoco demodata with easy dummy heatmap channels
    true_coco = kwcoco.CocoDataset.demo('vidshapes2', image_size=(512, 512))

    true_coco = true_coco.subset(image_ids=[1])
    base_pred_coco = true_coco.copy()

    base_pred_coco.reroot(absolute=True)
    base_pred_coco._update_fpath(pred_bundle_dpath / 'data.kwcoco.zip')
    base_pred_coco.reroot(absolute=True)

    true_coco_img = true_coco.images()[0:1].coco_images[0]

    # First write a perfect prediction
    if 1:
        pred_coco = base_pred_coco.copy()
        pred_coco_img = pred_coco.images()[0:1].coco_images[0]
        # Create a dummy heatmap based on annotations
        height, width = pred_coco_img.img['height'], pred_coco_img.img['width']
        heatmap = np.zeros((height, width, 1))
        pred_dets = pred_coco.annots(image_id=1).detections
        heatmap = pred_dets.data['segmentations'].fill(heatmap)

        asset_fpath = pred_bundle_dpath / 'perfect_heatmap.png'
        kwimage.imwrite(asset_fpath, heatmap)
        pred_coco_img.add_asset(asset_fpath, channels='salient',
                                width=pred_coco_img.img['width'],
                                height=pred_coco_img.img['height'])
        config = {}
        true_dets = true_coco_img.annots().detections
        video1 = true_coco_img.video
        true_classes = true_coco.object_categories()
        config['salient_channel'] = 'salient'

        self = SingleImageSegmentationMetrics(
            pred_coco_img, true_coco_img, true_classes, true_dets, config=config,
            video1=video1)
        # Should be nearly perfect at this point
        info = self.run()
        assert np.isclose(info['salient_measures']['ap'], 1)

    # Next remove an annotation to make a false negative
    if 1:
        pred_coco = base_pred_coco.copy()
        pred_coco.remove_annotation(pred_coco_img.annots().objs[0]['id'])
        pred_coco.rebuild_index()

        pred_coco_img = pred_coco.images()[0:1].coco_images[0]

        # Create a dummy heatmap based on annotations
        height, width = pred_coco_img.img['height'], pred_coco_img.img['width']
        heatmap = np.zeros((height, width, 1))
        pred_dets = pred_coco.annots(image_id=1).detections
        heatmap = pred_dets.data['segmentations'].fill(heatmap)

        asset_fpath = pred_bundle_dpath / 'imperfect_heatmap.png'
        kwimage.imwrite(asset_fpath, heatmap)
        pred_coco_img.add_asset(asset_fpath, channels='salient',
                                width=pred_coco_img.img['width'],
                                height=pred_coco_img.img['height'])
        config = {}
        true_dets = true_coco_img.annots().detections
        video1 = true_coco_img.video
        true_classes = true_coco.object_categories()
        config['salient_channel'] = 'salient'

        self = SingleImageSegmentationMetrics(
            pred_coco_img, true_coco_img, true_classes, true_dets, config=config,
            video1=video1)
        # Should be nearly perfect at this point
        info = self.run()
        # Score should be much worse
        assert info['salient_measures']['ap'] < 0.9

    # Finally, add a weight to the removed truth, which should bring
    # performance back up.
    if 1:
        for ann in true_coco_img.annots().objs:
            ann['weight'] = 1.0

        # The indexes don't make sense to me, but this is what makes the viz
        # reasonable.
        true_ann = true_coco_img.annots().objs[1]
        true_ann['weight'] = 0

        pred_coco = base_pred_coco.copy()
        pred_coco_img = pred_coco.images()[0:1].coco_images[0]
        pred_coco.remove_annotation(pred_coco_img.annots().objs[1]['id'])
        pred_coco.rebuild_index()

        # Create a dummy heatmap based on annotations
        height, width = pred_coco_img.img['height'], pred_coco_img.img['width']
        heatmap = np.zeros((height, width, 1))
        pred_dets = pred_coco.annots(image_id=1).detections
        heatmap = pred_dets.data['segmentations'].fill(heatmap)

        asset_fpath = pred_bundle_dpath / 'imperfect_heatmap.png'
        kwimage.imwrite(asset_fpath, heatmap)
        pred_coco_img.add_asset(asset_fpath, channels='salient',
                                width=pred_coco_img.img['width'],
                                height=pred_coco_img.img['height'])
        config = {}
        true_dets = true_coco_img.annots().detections
        video1 = true_coco_img.video
        true_classes = true_coco.object_categories()
        config['salient_channel'] = 'salient'

        self = SingleImageSegmentationMetrics(
            pred_coco_img, true_coco_img, true_classes, true_dets, config=config,
            video1=video1)
        # Should be nearly perfect at this point
        info = self.run()
        # Score should be much worse
        assert info['salient_measures']['ap'] > 0.9

    if 0:
        import kwplot
        kwplot.autompl()
        kwplot.imshow(info['true_saliency'].astype(np.float32), pnum=(1, 3, 1), fnum=1, title='true')
        kwplot.imshow(info['saliency_weights'], pnum=(1, 3, 2), fnum=1, title='weight')
        kwplot.imshow(info['salient_prob'], pnum=(1, 3, 3), fnum=1, title='pred')

        kwplot.imshow(pred_coco.draw_image(1), fnum=2)
