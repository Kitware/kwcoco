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
    host = ub.ensure_app_cache_dir('kwcoco/tests/reroot')
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
