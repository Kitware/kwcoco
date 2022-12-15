# from kwcoco.coco_dataset import *  # NOQA
import ubelt as ub
import numpy as np
from os.path import join, dirname, exists
import os


def main():
    # There might not be a way to easily handle the cases that I
    # want to here. Might need to discuss this.
    import kwcoco
    gname = 'images/foo.png'
    remote = '/remote/path'
    host = ub.Path.appdir('kwcoco/tests/reroot').ensuredir()
    fpath = join(host, gname)
    ub.ensuredir(dirname(fpath))
    # In this test the image exists on the host path
    import kwimage
    kwimage.imwrite(fpath, np.random.rand(8, 8))
    #
    cases = {}
    # * given absolute paths on current machine
    cases['abs_curr'] = kwcoco.CocoDataset.from_image_paths([join(host, gname)])
    # * given "remote" rooted relative paths on current machine
    cases['rel_remoterooted_curr'] = kwcoco.CocoDataset.from_image_paths([gname], bundle_dpath=remote)
    # * given "host" rooted relative paths on current machine
    cases['rel_hostrooted_curr'] = kwcoco.CocoDataset.from_image_paths([gname], bundle_dpath=host)
    # * given unrooted relative paths on current machine
    cases['rel_unrooted_curr'] = kwcoco.CocoDataset.from_image_paths([gname])
    # * given absolute paths on another machine
    cases['abs_remote'] = kwcoco.CocoDataset.from_image_paths([join(remote, gname)])
    def report(dset, name):
        gid = 1
        rel_fpath = dset.index.imgs[gid]['file_name']
        abs_fpath = dset.get_image_fpath(gid)
        color = 'green' if exists(abs_fpath) else 'red'
        print('   * strategy_name = {!r}'.format(name))
        print('       * rel_fpath = {!r}'.format(rel_fpath))
        print('       * ' + ub.color_text('abs_fpath = {!r}'.format(abs_fpath), color))
    for key, dset in cases.items():
        print('----')
        print('case key = {!r}'.format(key))
        print('ORIG = {!r}'.format(dset.index.imgs[1]['file_name']))
        print('dset.bundle_dpath = {!r}'.format(dset.bundle_dpath))
        print('missing_gids = {!r}'.format(dset.missing_images()))
        print('cwd = {!r}'.format(os.getcwd()))
        print('host = {!r}'.format(host))
        print('remote = {!r}'.format(remote))
        #
        dset_None_rel = dset.copy().reroot(absolute=False, check=0)
        report(dset_None_rel, 'dset_None_rel')
        #
        dset_None_abs = dset.copy().reroot(absolute=True, check=0)
        report(dset_None_abs, 'dset_None_abs')
        #
        dset_host_rel = dset.copy().reroot(host, absolute=False, check=0)
        report(dset_host_rel, 'dset_host_rel')
        #
        dset_host_abs = dset.copy().reroot(host, absolute=True, check=0)
        report(dset_host_abs, 'dset_host_abs')
        #
        dset_remote_rel = dset.copy().reroot(host, old_prefix=remote, absolute=False, check=0)
        report(dset_remote_rel, 'dset_remote_rel')
        #
        dset_remote_abs = dset.copy().reroot(host, old_prefix=remote, absolute=True, check=0)
        report(dset_remote_abs, 'dset_remote_abs')


def demo_reroot_bug1():
    """
    This demos a case that does not work correctly.

    TODO:
        Check case where:

            * Files (including auxiliary) are relative to a bundle path

                bundle_path1 = '/my/data/repos/my_bundle1'

            * Then we reroot to an absolute directory

            * Then we we change the bundle path to

                bundle_path2 = '/my/data/repos/my_bundle2'

            * Then we reroot to a relative directory. This should result in
                filepaths changing such that they all start with

                ../my_bundle1/
    """
    import kwcoco
    import ubelt as ub
    import kwimage
    dset = kwcoco.CocoDataset()
    dpath = ub.Path.appdir('kwcoco/tests/reroot').ensuredir()
    dset.fpath = dpath / 'data/repos/my_bundle1/data.kwcoco.json'
    dset.add_image(file_name='assets/img1.jpg')

    fpath1 = ub.Path(dset.coco_image(1).primary_image_filepath())
    fpath1.parent.ensuredir()
    # TODO: kwimage should error if the write fails
    kwimage.imwrite(fpath1, kwimage.grab_test_image())
    assert fpath1.exists()

    dset.reroot(absolute=True)
    dset.index.imgs[1]['file_name']

    # Now change the file path, which changes the bundle
    dset.fpath = dpath / 'data/repos/my_bundle2/data.kwcoco.json'
    assert len(dset.missing_images()) == 1

    dset.reroot(absolute=False)
    # Image should be relative, but it is not
    dset.index.imgs[1]['file_name']

    if 0:
        # Workaround
        import kwcoco
        import os
        import ubelt as ub
        import kwimage
        dset1 = kwcoco.CocoDataset()
        dpath = ub.Path.appdir('kwcoco/tests/reroot').ensuredir()

        # We want to modify a dataset in bundle1 and write it in bundle2
        # the new dataset should reference a relative path to bundle1
        bundle_dpath1 = (dpath / 'data/repos/my_bundle1').ensuredir()
        bundle_dpath2 = (dpath / 'data/repos/my_bundle2').ensuredir()

        dset1.fpath = bundle_dpath1 / 'data.kwcoco.json'
        dset1.add_image(file_name='assets/img1.jpg')

        fpath1 = ub.Path(dset1.coco_image(1).primary_image_filepath())
        fpath1.parent.ensuredir()
        kwimage.imwrite(fpath1, kwimage.grab_test_image())

        assert fpath1.exists()

        # Method1: reroot after changing the bundle
        dset2 = dset1.copy()
        dset2.fpath = bundle_dpath2 / 'data.kwcoco.json'
        assert not ub.Path(dset2.coco_image(1).primary_image_filepath()).exists(), (
            'naive changing of bundle dpath will cause image to be missreferenced'
        )
        new_prefix = os.path.relpath(bundle_dpath1, bundle_dpath2)
        dset2.reroot(new_prefix=new_prefix)
        assert not dset2.missing_images()

        # Method2: absolute reroot first
        dset3 = dset1.copy()
        dset3.reroot(absolute=True)
        dset3.fpath = bundle_dpath2 / 'data.kwcoco.json'
        new_prefix = os.path.relpath(bundle_dpath1, bundle_dpath2)
        dset3.reroot(old_prefix=dset1.bundle_dpath, new_prefix=new_prefix, absolute=False)
        dset3.imgs[1]
