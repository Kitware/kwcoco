def parse_quantity(expr):
    import re
    expr_pat = re.compile(
        r'^(?P<magnitude>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
        '(?P<spaces> *)'
        '(?P<unit>.*)$')
    match = expr_pat.match(expr.strip())
    return match.groupdict()


def parse_resolution(expr):
    if isinstance(expr, str):
        result = parse_quantity(expr)
        unit = result['unit']
        x = y = float(result['magnitude'])
    else:
        x = y = float(expr)
        unit = None

    parsed = {
        'mag': (x, y),
        'unit': unit,
    }
    return parsed


def space_resolution(coco_img, space='image', RESOLUTION_KEY='resolution'):
    import kwimage
    # Compute the offset transform from the requested space
    # Handle the cases where resolution is specified at the image or at the
    # video level.
    if space == 'video':
        vid_resolution_expr = coco_img.video.get(RESOLUTION_KEY, None)
        if vid_resolution_expr is None:
            # Do we have an image level resolution?
            img_resolution_expr = coco_img.img.get(RESOLUTION_KEY, None)
            assert img_resolution_expr is not None
            img_resolution_info = parse_resolution(img_resolution_expr)
            img_resolution_mat = kwimage.Affine.scale(img_resolution_info['mag'])
            vid_resolution = (coco_img.warp_vid_from_img @ img_resolution_mat.inv()).inv()
            vid_resolution_info = {
                'mag': vid_resolution.decompose()['scale'],
                'unit': img_resolution_info['unit']
            }
        else:
            vid_resolution_info = parse_resolution(vid_resolution_expr)
        space_resolution_info = vid_resolution_info
    elif space == 'image':
        img_resolution_expr = coco_img.img.get(RESOLUTION_KEY, None)
        if img_resolution_expr is None:
            # Do we have an image level resolution?
            vid_resolution_expr = coco_img.video.get(RESOLUTION_KEY, None)
            assert vid_resolution_expr is not None
            vid_resolution_info = parse_resolution(vid_resolution_expr)
            vid_resolution_mat = kwimage.Affine.scale(vid_resolution_info['mag'])
            img_resolution = (coco_img.warp_img_from_vid @ vid_resolution_mat.inv()).inv()
            img_resolution_info = {
                'mag': img_resolution.decompose()['scale'],
                'unit': vid_resolution_info['unit']
            }
        else:
            img_resolution_info = parse_resolution(img_resolution_expr)
        space_resolution_info = img_resolution_info
    elif space == 'asset':
        raise NotImplementedError(space)
    else:
        raise KeyError(space)
    return space_resolution_info


def scalefactor_for_resolution(coco_img, space, resolution, RESOLUTION_KEY='resolution'):
    """
    Given image or video space, compute the scale factor needed to achieve the
    target resolution.
    """
    space_resolution_info = space_resolution(coco_img, space=space, RESOLUTION_KEY=RESOLUTION_KEY)
    request_resolution_info = parse_resolution(resolution)
    assert space_resolution_info['unit'] == request_resolution_info['unit']
    x1, y1 = request_resolution_info['mag']
    x2, y2 = space_resolution_info['mag']
    scale_factor = (x2 / x1, y2 / y1)
    return scale_factor


def demo():
    import kwcoco
    import kwimage
    dset = kwcoco.CocoDataset.demo('vidshapes8-msi-multisensor')
    RESOLUTION_KEY = 'resolution'

    coco_img = dset.coco_image(1)
    coco_img.img['warp_img_to_vid'] = kwimage.Affine.scale(0.5).concise()
    coco_img.video[RESOLUTION_KEY] = '10 meters'
    coco_img.video['width']  = coco_img.img['width'] * 0.5
    coco_img.video['height'] = coco_img.img['height'] * 0.5

    resolution = '3 meters'
    vidspace_resolution = space_resolution(coco_img, space='video', RESOLUTION_KEY=RESOLUTION_KEY)
    imgspace_resolution = space_resolution(coco_img, space='image', RESOLUTION_KEY=RESOLUTION_KEY)
    print(f'vidspace_resolution={vidspace_resolution}')
    print(f'imgspace_resolution={imgspace_resolution}')

    space = 'video'
    scale_factor = scalefactor_for_resolution(coco_img, space, resolution, RESOLUTION_KEY=RESOLUTION_KEY)
    print(f'scale_factor={scale_factor}')
    delayed1 = coco_img.delay(space=space).scale(scale_factor)
    delayed1.write_network_text()

    space = 'image'
    scale_factor = scalefactor_for_resolution(coco_img, space, resolution, RESOLUTION_KEY=RESOLUTION_KEY)
    print(f'scale_factor={scale_factor}')
    delayed2 = coco_img.delay(space=space).scale(scale_factor)
    delayed2.write_network_text()

    assert delayed1.dsize == delayed2.dsize, (
        'requesting the same scale from different spaces should be the '
        'same shape as long as there is no translation')

    assert delayed1.transform.concise() != delayed2.transform.concise(), (
        'but the modifying transform will be different depending on the '
        'starting point (i.e. image or video)'
    )

    # ####

    # coco_img = dset.coco_image(2)
    # coco_img.video[RESOLUTION_KEY] = '10meters'

    # # If the coco dataset registers an image or video with a resolution with
    # # some units, then we should be able to request an absolute variant in
    # # terms of those units.
    # dset.coco_image(3).img[RESOLUTION_KEY] = '3arcunits'
