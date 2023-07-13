#!/usr/bin/env python
import ubelt as ub
import scriptconfig as scfg


class CocoSubsetCLI(object):
    name = 'subset'

    class CocoSubetConfig(scfg.DataConfig):
        """
        Take a subset of this dataset and write it to a new file
        """
        __command__ = 'subset'
        __default__ = {
            'src': scfg.Value(None, help='input dataset path', position=1),
            'dst': scfg.Value(None, help='output dataset path', position=2),
            'include_categories': scfg.Value(
                None, type=str, help=ub.paragraph(
                    '''
                    a comma separated list of categories, if specified only
                    images containing these categories will be included.
                    ''')),  # TODO: pattern matching?

            'gids': scfg.Value(
                None, help=ub.paragraph(
                    '''
                    A comma separated list of image ids.
                    If specified, only consider these image ids
                    ''')),

            'select_images': scfg.Value(
                None, type=str, help=ub.paragraph(
                    '''
                    A json query (via the jq spec) that specifies which images
                    belong in the subset. Note, this is a passed as the body of
                    the following jq query format string to filter valid ids
                    '.images[] | select({select_images}) | .id'.

                    Examples for this argument are as follows:
                    '.id < 3' will select all image ids less than 3.
                    '.file_name | test(".*png")' will select only images with
                    file names that end with png.
                    '.file_name | test(".*png") | not' will select only images
                    with file names that do not end with png.
                    '.myattr == "foo"' will select only image dictionaries
                    where the value of myattr is "foo".
                    '.id < 3 and (.file_name | test(".*png"))' will select only
                    images with id less than 3 that are also pngs.
                    .myattr | in({"val1": 1, "val4": 1}) will take images
                    where myattr is either val1 or val4.

                    Requries the "jq" python library is installed.
                    ''')),

            'channels': scfg.Value(
                None, help=ub.paragraph(
                    '''
                    if specified select only images that contain these channels
                    (specified as a kwcoco channel spec)
                    ''')),

            'select_videos': scfg.Value(
                None, help=ub.paragraph(
                    '''
                    A json query (via the jq spec) that specifies which videos
                    belong in the subset. Note, this is a passed as the body of
                    the following jq query format string to filter valid ids
                    '.videos[] | select({select_images}) | .id'.

                    Examples for this argument are as follows:
                    '.file_name | startswith("foo")' will select only videos
                    where the name starts with foo.

                    Only applicable for dataset that contain videos.

                    Requries the "jq" python library is installed.
                    ''')),

            'copy_assets': scfg.Value(False, help='if True copy the assests to the new bundle directory'),

            'compress': scfg.Value('auto', help='if True writes results with compression'),

            'absolute': scfg.Value('auto', help=ub.paragraph(
                '''
                if True will reroot all paths to be absolute before writing. If
                "auto", becomes True if the dest dataset is written outside of
                the source bundle directory and copy_assets is False.
                '''
            ))

            # TODO: Add more filter criteria
            #
            # image size
            # image timestamp
            # image file name matches
            # annotations with segmentations / keypoints?
            # iamges/annotations that contain a special attribute?
            # images with a maximum / minimum number of annotations?

            # 'rng': scfg.Value(None, help='random seed'),
        }
        __epilog__ = """
        Example Usage:
            kwcoco subset --src special:shapes8 --dst=foo.kwcoco.json

            # Take only the even image-ids
            kwcoco subset --src special:shapes8 --dst=foo-even.kwcoco.json --select_images '.id % 2 == 0'

            # Take only the videos where the name ends with 2
            kwcoco subset --src special:vidshapes8 --dst=vidsub.kwcoco.json --select_videos '.name | endswith("2")'
        """

    CLIConfig = CocoSubetConfig

    @classmethod
    def main(cls, cmdline=True, **kw):
        """
        Example:
            >>> from kwcoco.cli.coco_subset import *  # NOQA
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('kwcoco/tests/cli/union').ensuredir()
            >>> kw = {'src': 'special:shapes8',
            >>>       'dst': dpath / 'subset.json',
            >>>       'include_categories': 'superstar'}
            >>> cmdline = False
            >>> cls = CocoSubsetCLI
            >>> cls.main(cmdline, **kw)
        """
        import kwcoco

        config = cls.CLIConfig.cli(data=kw, cmdline=cmdline, strict=True)
        print('config = {}'.format(ub.urepr(config, nl=1)))

        if config['src'] is None:
            raise Exception('must specify subset src: {}'.format(config['src']))

        if config['dst'] is None:
            raise Exception('must specify subset dst: {}'.format(config['dst']))

        print('reading fpath = {!r}'.format(config['src']))
        dset = kwcoco.CocoDataset.coerce(config['src'])
        dst_fpath = ub.Path(config['dst'])

        if config['absolute'] == 'auto':
            if not config['copy_assets']:
                src_fpath = ub.Path(dset.fpath)

                src_bundle_dpath = src_fpath.absolute().parent
                dst_bundle_dpath = dst_fpath.absolute().parent

                # Destinations are different, we will need to force a reroot
                absolute = (src_bundle_dpath.resolve() !=
                            dst_bundle_dpath.resolve())
            else:
                absolute = False
        else:
            absolute = config['absolute']
        print(f'absolute={absolute}')

        new_dset = query_subset(dset, config)
        if absolute:
            new_dset.reroot(absolute=absolute)
        else:
            if config['copy_assets']:
                # a bit roundabout, but it seems to work
                new_dset.reroot(absolute=False)
        new_dset.fpath = dst_fpath
        print(f'new_dset.fpath={new_dset.fpath}')

        if config['copy_assets']:
            # Create a copy of the data, (currently only works for relative
            # kwcoco files)
            from os.path import join, dirname
            import shutil
            print('Copying assets')
            # new_dset.reroot(new_dset.bundle_dpath, old_prefix=dset.bundle_dpath)
            tocopy = []
            dstdirs = set()
            print(f'new_dset.bundle_dpath={new_dset.bundle_dpath}')
            for gid, new_img in new_dset.index.imgs.items():
                old_img = dset.index.imgs[gid]
                if new_img.get('file_name', None) is not None:
                    old_fpath = join(dset.bundle_dpath, old_img['file_name'])
                    new_fpath = join(new_dset.bundle_dpath, new_img['file_name'])
                    dstdirs.add(dirname(new_fpath))
                    tocopy.append((old_fpath, new_fpath))
                new_aux_list = new_img.get('auxiliary', [])
                old_aux_list = old_img.get('auxiliary', [])
                for old_aux, new_aux in zip(old_aux_list, new_aux_list):
                    old_fpath = join(dset.bundle_dpath, old_aux['file_name'])
                    new_fpath = join(new_dset.bundle_dpath, new_aux['file_name'])
                    dstdirs.add(dirname(new_fpath))
                    tocopy.append((old_fpath, new_fpath))

            # Ensure directories
            for dpath in dstdirs:
                ub.ensuredir(dpath)

            pool = ub.JobPool(max_workers=4)
            for src, dst in tocopy:
                pool.submit(shutil.copy2, src, dst)

            for future in pool.as_completed(desc='copy assets'):
                future.result()

        print('Writing new_dset.fpath = {!r}'.format(new_dset.fpath))
        dumpkw = {
            'newlines': True,
            'compress': config['compress'],
        }
        new_dset.dump(new_dset.fpath, **dumpkw)


def query_subset(dset, config):
    """

    Example:
        >>> # xdoctest: +REQUIRES(module:jq)
        >>> from kwcoco.cli.coco_subset import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo()
        >>> assert dset.n_images == 3
        >>> #
        >>> config = CocoSubsetCLI.CLIConfig(**{'select_images': '.id < 3'})
        >>> new_dset = query_subset(dset, config)
        >>> assert new_dset.n_images == 2
        >>> #
        >>> config = CocoSubsetCLI.CLIConfig(**{'select_images': '.file_name | test(".*.png")'})
        >>> new_dset = query_subset(dset, config)
        >>> assert all(n.endswith('.png') for n in new_dset.images().lookup('file_name'))
        >>> assert new_dset.n_images == 2
        >>> #
        >>> config = CocoSubsetCLI.CLIConfig(**{'select_images': '.file_name | test(".*.png") | not'})
        >>> new_dset = query_subset(dset, config)
        >>> assert not any(n.endswith('.png') for n in new_dset.images().lookup('file_name'))
        >>> assert new_dset.n_images == 1
        >>> #
        >>> config = CocoSubsetCLI.CLIConfig(**{'select_images': '.id < 3 and (.file_name | test(".*.png"))'})
        >>> new_dset = query_subset(dset, config)
        >>> assert new_dset.n_images == 1
        >>> #
        >>> config = CocoSubsetCLI.CLIConfig(**{'select_images': '.id < 3 or (.file_name | test(".*.png"))'})
        >>> new_dset = query_subset(dset, config)
        >>> assert new_dset.n_images == 3

    Example:
        >>> # xdoctest: +REQUIRES(module:jq)
        >>> from kwcoco.cli.coco_subset import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8')
        >>> assert dset.n_videos == 8
        >>> assert dset.n_images == 16
        >>> config = CocoSubsetCLI.CLIConfig(**{'select_videos': '.name == "toy_video_3"'})
        >>> new_dset = query_subset(dset, config)
        >>> assert new_dset.n_images == 2
        >>> assert new_dset.n_videos == 1
    """
    valid_gids = set(dset.imgs.keys())

    if config['gids'] is not None:
        if isinstance(config['gids'], str):
            valid_gids &= set(map(int, config['gids'].split(',')))
        elif ub.iterable(config['gids']):
            valid_gids &= set(map(int, config['gids']))
        else:
            raise KeyError(config['gids'])

    if config['include_categories'] is not None:
        catnames = config['include_categories'].split(',')
        chosen_cids = []
        for cname in catnames:
            cid = dset._resolve_to_cat(cname)['id']
            chosen_cids.append(cid)

        category_gids = set(ub.flatten(ub.take(
            dset.index.cid_to_gids, set(chosen_cids))))

        valid_gids &= category_gids

    if config['select_images'] is not None:
        try:
            import jq
        except Exception:
            print('The jq library is required to run a generic image query')
            raise

        try:
            query_text = ".images[] | select({select_images}) | .id".format(**config)
            query = jq.compile(query_text)
            found_gids = query.input(dset.dataset).all()
            found_gids = set(found_gids)
            valid_gids &= found_gids
        except Exception:
            print('JQ Query Failed: {}'.format(query_text))
            raise

    if config['select_videos'] is not None:
        if not dset.dataset.get('videos', []):
            raise ValueError('Dataset does not contain videos')

        try:
            import jq
        except Exception:
            print('The jq library is required to run a generic image query')
            raise

        try:
            query_text = ".videos[] | select({select_videos}) | .id".format(**config)
            query = jq.compile(query_text)
            found_vidids = query.input(dset.dataset).all()
            found_vidids = set(found_vidids)
            found_gids = set(ub.flatten(dset.index.vidid_to_gids[vidid]
                                        for vidid in found_vidids))
            valid_gids &= found_gids
        except Exception:
            print('JQ Query Failed: {}'.format(query_text))
            raise

    if config['channels'] is not None:
        import kwcoco
        requested_chans = kwcoco.ChannelSpec(config['channels'])
        valid_gids = [
            gid for gid in valid_gids
            if (requested_chans & dset.coco_image(gid).channels).numel() == requested_chans.numel()
        ]

    new_dset = dset.subset(valid_gids)
    return new_dset


_CLI = CocoSubsetCLI

if __name__ == '__main__':
    _CLI.main()
