"""
How to explictly breakup a kwcoco file into smaller tiles.
"""
import kwcoco
import kwarray
import kwimage


def main():
    dset = kwcoco.CocoDataset.demo('shapes8')

    for image_id in dset.images():
        img_obj = dset.index.imgs[image_id]
        img_width = img_obj['width']
        img_height = img_obj['height']

        img_shape = (img_height, img_width)
        window_shape = (128, 128)

        slider = kwarray.SlidingWindow(shape=img_shape, window=window_shape,
                                       allow_overshoot=True, keepbound=True)

        image_dets: kwimage.Detections = dset.annots(gid=image_id).detections

        coco_img: kwcoco.CocoImage = dset.coco_image(image_id)
        delayed_img = coco_img.imdelay()

        # Pure coco variant
        for space_slice in slider:

            tl_y = space_slice[0].start
            tl_x = space_slice[1].start

            shift = (-tl_x, -tl_y)
            # Move the detections into the window
            shifted_dets = image_dets.translate(shift)

            # todo: Need to crop out things not in the window

            # Convert the detections back into coco format, suitable for
            # add_annotations
            shifted_coco_annots = list(shifted_dets.to_coco())

            delayed_crop = delayed_img[space_slice]

            # Raw image pixels
            window_image = delayed_crop.finalize()

        ### ndsampler variant
        import ndsampler
        sampler = ndsampler.CocoSampler(dset)

        for space_slice in slider:
            target = {
                'space_slice': space_slice,
                'gid': image_id,
            }
            sample = sampler.load_sample(target)

            # These detections are already cropped and shifted
            window_dets = sample['annots']['frame_dets'][0]
            window_image = sample['im']
            shifted_coco_annots = list(shifted_dets.to_coco())
            print(f'window_dets={window_dets}')
            print(f'window_image={window_image}')
            print(f'shifted_coco_annots={shifted_coco_annots}')
