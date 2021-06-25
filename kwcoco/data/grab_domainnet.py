"""
References:
    http://ai.bu.edu/M3SDA/#dataset
"""


def grab_domain_net():
    """
    TODO:
        - [ ] Allow the user to specify the download directory, generalize this
        pattern across the data grab scripts.
    """
    import zipfile
    import ubelt as ub
    import kwcoco
    import kwimage
    import os

    infos = {
        'clipart_images': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
            'sha512': '3bcfb9ec1b4061e8d5b0b887d4ebd4a708732787fc563c6dfc2d'
        },
        'infograph_images': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
            'sha512': '47841f0d1b8606e4b02d508b250484d54f5cf04ef6c4875c6c5a39c',
        },
        'painting_images': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
            'sha512': '4c98b02563075948668a298c488660fda1d1a7ed85fd69caa7985fba',
        },
        'quickdraw_images': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
            'sha512': '127cce1dd57fc99992f8614de77d03ae8ed6af242973f7a013ec0a',
        },
        'real_images': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
            'sha512': '751713d4592d1278b50bf69787988cfae280cabdac80ee34ec4016e',
        },
        'sketch_images': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
            'sha512': '369b8b6d78ac61bfcc85f660d878ba5fc701524b82bbc2eb65ca9a',
        },

        'clipart_train': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt',
            'sha512': '400985ce2a0878df2d8e54f1996c4fc253e577c1f91136304e553b'
        },
        'clipart_test': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt',
            'sha512': '4d34a4a540d8b139499581875e2d18bbedd37c347233523d66beaf',
        },

        'infograph_train': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt',
            'sha512': 'd9ed4d86e2ea20b44315699964b6b7dc09442f0539300444d9329d'
        },
        'infograph_test': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt',
            'sha512': '700f2b5deaad3923b0b8b85aeb694b73f24b13255cf48db3e770eaf',
        },

        'quickdraw_train': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt',
            'sha512': '7410e8a1debd769e0412725a0fcc646d83055d297b5f44a92f50308'
        },
        'quickdraw_test': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt',
            'sha512': '07a97e84d723f58dea67683ae5ddf013ebc5d33b76deca9a5d34f72',
        },

        'real_train': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt',
            'sha512': 'a5a510fec614018546510d55c378bca008fcfd51062f6414583639e'
        },
        'real_test': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt',
            'sha512': '9586235e335340f065de720f99332bc906aa0ab147006a1a7ea4925',
        },

        'sketch_train': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt',
            'sha512': 'ee92102f2b98f11765e6a97afbab15c60f2bac6e30fa7e1bc5a7db'
        },
        'sketch_test': {
            'url': 'http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt',
            'sha512': '7384fc06670979911d7e1e9a34ba8a6cd20e9e77d6bc3536c739ff4',
        },
    }

    dpath = ub.ensure_app_cache_dir('kwcoco', 'domain_net')

    # Assign a coco filepath to each dataset split
    for key, info in infos.items():
        if key.endswith(('_train', '_test')):
            info['coco_fpath'] = os.path.join(dpath, key + '.kwcoco.json')

    stamp = ub.CacheStamp('domain_stamp', dpath=dpath, depends=['v001'])
    if stamp.expired():
        errors = []
        # TODO: Multi-file download manager with parallel jobs
        # TODO: Don't redownload if the data was already extracted
        for key, info in infos.items():
            try:
                zip_fpath = ub.grabdata(
                    info['url'], dpath=dpath,
                    hash_prefix=info.get('sha512', 'x' * 64))
                info['fpath'] = zip_fpath
            except Exception as ex:
                print('ex = {!r}'.format(ex))
                errors.append(repr(ex))

        if errors:
            raise Exception('download errors')

        # Extact images from archive files
        for key, info in infos.items():
            if key.endswith('_images'):
                print('extract {} images'.format(key))
                file = open(info['fpath'], 'rb')
                zfile = zipfile.ZipFile(file)
                zfile.extractall(path=dpath)

        # Construct the kwcoco manifests
        for key, info in infos.items():
            if key.endswith(('_train', '_test')):
                coco_dset = kwcoco.CocoDataset()
                coco_dset.fpath = info['coco_fpath']

                with open(info['fpath'], 'r') as file:
                    lines = file.read().split('\n')

                for line in ub.ProgIter(lines, desc='parse ' + key):
                    if line:
                        print('line = {!r}'.format(line))
                        path, num = line.split(' ')
                        gpath = os.path.join(dpath, path)
                        shape = kwimage.load_image_shape(gpath)
                        h, w = shape[0:2]
                        domain, catname, image_name = path.split('/')
                        gid = coco_dset.add_image(file_name=path, height=h,
                                                  width=w, name=image_name)
                        cid = int(num)
                        cid = coco_dset.ensure_category(name=catname, id=cid)
                        coco_dset.add_annotation(image_id=gid, category_id=cid,
                                                 bbox=[0, 0, w, h])
                        # Mark the domain in an non-standard field
                        coco_dset.index.imgs['domain'] = domain

                coco_dset.validate()
                coco_dset.dump(coco_dset.fpath, newlines=True)
        stamp.renew()

    # Read and return each domain-net dataset
    dsets = []
    for key, info in infos.items():
        if 'coco_fpath' in info:
            dset = kwcoco.CocoDataset(info['coco_fpath'])
            dsets.append(dset)
    return dsets


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/data/grab_domainnet.py
    """
    grab_domain_net()
