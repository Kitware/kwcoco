"""
Demonstrates the core kwcoco spaces: video, image, and asset space.
"""


def setup_data():
    """
    This creates aligned video data similar to what might exist in a real world
    use case.
    """
    import ubelt as ub
    import kwimage

    demo_dpath = ub.Path.appdir('kwcoco/demo/demo_spaces').ensuredir()
    asset_dpath = (demo_dpath / 'assets').ensuredir()
    raw = kwimage.grab_test_image('amazon', dsize=(512, 512))
    raw = kwimage.ensure_float01(raw)

    import kwcoco
    dset = kwcoco.CocoDataset()
    dset.fpath = demo_dpath / 'data.kwcoco.json'

    video_id = dset.add_video(name='demo_video', width=512, height=512)

    observations = [
        {'warp_img_from_vid': kwimage.Affine.coerce(offset=(-32, -32), theta=0.02)},
        {'warp_img_from_vid': kwimage.Affine.coerce(offset=(-64, -64), theta=0.05)},
        {'warp_img_from_vid': kwimage.Affine.coerce(offset=(-128, -128), theta=0.08)},
    ]
    for frame_idx, obs in enumerate(observations):
        warp_img_from_vid = obs['warp_img_from_vid']

        ideal_img_frame = kwimage.warp_affine(raw, warp_img_from_vid, dsize='positive', border_value=float('nan'))
        img_h, img_w = ideal_img_frame.shape[0:2]

        # pan is shifted a little bit
        pan = kwimage.convert_colorspace(ideal_img_frame, 'rgb', 'gray')
        warp_pan_from_img = kwimage.Affine.coerce(offset=(-2, -3))

        image_name = f'frame_{frame_idx:03d}'
        frame_dpath = (asset_dpath / image_name).ensuredir()

        pan_fpath = frame_dpath / 'pan.tif'
        red_fpath = frame_dpath / 'red.tif'
        green_fpath = frame_dpath / 'green.tif'
        blue_fpath = frame_dpath / 'blue.tif'

        # Some assets are smaller than others
        warp_msi_from_img = kwimage.Affine.scale(0.5)
        msi = kwimage.warp_affine(ideal_img_frame, warp_msi_from_img, dsize='positive', border_value=float('nan'))
        red = msi[..., 0]
        green = msi[..., 1]
        blue = msi[..., 2]

        kwimage.imwrite(pan_fpath, pan)
        kwimage.imwrite(red_fpath, red)
        kwimage.imwrite(green_fpath, green)
        kwimage.imwrite(blue_fpath, blue)

        gid = dset.add_image(name=image_name, frame_index=frame_idx,
                             width=img_w, height=img_h, video_id=video_id,
                             warp_img_to_vid=warp_img_from_vid.inv())

        coco_img = dset.coco_image(gid)

        coco_img.add_asset(pan_fpath, channels='pan', warp_aux_to_img=warp_pan_from_img.inv())
        coco_img.add_asset(red_fpath, channels='red', warp_aux_to_img=warp_msi_from_img.inv())
        coco_img.add_asset(green_fpath, channels='green', warp_aux_to_img=warp_msi_from_img.inv())
        coco_img.add_asset(blue_fpath, channels='blue', warp_aux_to_img=warp_msi_from_img.inv())
    return dset


def main():
    import kwimage
    dset = setup_data()

    def build_space_frames(space):
        frame_stack = []
        for coco_img in dset.images().coco_images:
            frame_index = coco_img.img['frame_index']
            asset_stack = []
            for chan in coco_img.channels.fuse().to_list():
                asset_data = coco_img.imdelay(chan, space=space).finalize(nodata_method='float')
                asset_data = kwimage.fill_nans_with_checkers(asset_data)
                asset_data = kwimage.draw_header_text(asset_data, f'T={frame_index} : {chan}')
                asset_stack.append(asset_data)
            asset_row = kwimage.stack_images(asset_stack, axis=1, pad=10, bg_value='kitware_green')
            frame_stack.append(asset_row)
        canvas = kwimage.stack_images(frame_stack, axis=0, pad=10, bg_value='kitware_green')
        return canvas

    asset_canvas = build_space_frames(space='asset')
    image_canvas = build_space_frames(space='image')
    video_canvas = build_space_frames(space='video')

    import ubelt as ub
    asset_blurb = ub.codeblock(
        '''
        This is a demonstration of "asset space".
        Each frame is a row, and each channel is a column.
        This is how images are saved on disk.
        Any alignment is delayed until the last possible moment.
        ''')
    image_blurb = ub.codeblock(
        '''
        This is a demonstration of "image space".
        Assets are aligned within an image, but are not necesarilly aligned across frames.
        The resolution is typically that of the largest asset in the image, but can be arbitrary.
        ''')
    video_blurb = ub.codeblock(
        '''
        This is a demonstration of "video space".
        Assets are aligned within an image and across frames.
        Resolution is usually set to that of some reference image, but it can be arbitrary.
        ''')

    import kwplot
    kwplot.autompl()
    kwplot.imshow(asset_canvas, title=asset_blurb, fnum=1)
    kwplot.imshow(image_canvas, title=image_blurb, fnum=2)
    kwplot.imshow(video_canvas, title=video_blurb, fnum=3)
