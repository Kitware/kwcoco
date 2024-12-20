"""
A wrapper around the basic kwcoco dataset with a pycocotools API.

We do not recommend using this API because it has some idiosyncrasies, where
names can be misleading and APIs are not always clear / efficient: e.g.

(1) catToImgs returns integer image ids but imgToAnns returns annotation
    dictionaries.

(2) showAnns takes a dictionary list as an argument instead of an integer list

The cool thing is that this extends the kwcoco API so you can drop this for
compatibility with the old API, but you still get access to all of the kwcoco
API including dynamic addition / removal of categories / annotations / images.
"""
from kwcoco.coco_dataset import CocoDataset
import itertools as it
import ubelt as ub
import numpy as np


class COCO(CocoDataset):
    """
    A wrapper around the basic kwcoco dataset with a pycocotools API.

    Example:
        >>> from kwcoco.compat_dataset import *  # NOQA
        >>> import kwcoco
        >>> basic = kwcoco.CocoDataset.demo('shapes8')
        >>> self = COCO(basic.dataset)
        >>> self.info()
        >>> print('self.imgToAnns = {!r}'.format(self.imgToAnns[1]))
        >>> print('self.catToImgs = {!r}'.format(self.catToImgs))
    """

    def __init__(self, annotation_file=None, **kw):
        if annotation_file is not None:
            if 'data' in kw:
                raise ValueError('cannot specify data and annotation file')
        if 'data' in kw and annotation_file is None:
            annotation_file = kw.pop('data')
        super().__init__(annotation_file, **kw)

    def createIndex(self):
        self._build_index()

    def info(self):
        """
        Print information about the annotation file.
        """
        for key, value in self.dataset.get('info', {}).items():
            print('{}: {}'.format(key, value))

    @property
    def imgToAnns(self):
        from scriptconfig.dict_like import DictLike
        class ProxyImgToAnns(DictLike):
            def __init__(self, parent):
                self.parent = parent
            def getitem(self, gid):
                aids = self.parent.index.gid_to_aids[gid]
                anns = list(ub.take(self.parent.index.anns, aids))
                return anns
            def keys(self):
                return self.parent.index.gid_to_aids.keys()
        imgToAnns = ProxyImgToAnns(parent=self)
        return imgToAnns

    @property
    def catToImgs(self):
        """
        unlike the name implies, this actually goes from category to image ids
        Name retained for backward compatibility
        """
        catToImgs = self.index.cid_to_gids
        return catToImgs

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter

        Args:

            imgIds  (List[int]): get anns for given imgs
            catIds  (List[int]): get anns for given cats
            areaRng (List[float]): get anns for given area range (e.g. [0 inf])
            iscrowd (bool | None): get anns for given crowd label (False or True)

        Returns:
            List[int]: integer array of ann ids

        Example:
            >>> from kwcoco.compat_dataset import *  # NOQA
            >>> import kwcoco
            >>> self = COCO(kwcoco.CocoDataset.demo('shapes8').dataset)
            >>> self.getAnnIds()
            >>> self.getAnnIds(imgIds=1)
            >>> self.getAnnIds(imgIds=[1])
            >>> self.getAnnIds(catIds=[3])
        """
        imgIds = imgIds if ub.iterable(imgIds) else [imgIds]
        catIds = catIds if ub.iterable(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId]
                         for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(it.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [
                ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [
                ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.

        Args:
            catNms (List[str]): get cats for given cat names
            supNms (List[str]): get cats for given supercategory names
            catIds (List[int]): get cats for given cat ids

        Returns:
            List[int]: integer array of cat ids

        Example:
            >>> from kwcoco.compat_dataset import *  # NOQA
            >>> import kwcoco
            >>> self = COCO(kwcoco.CocoDataset.demo('shapes8').dataset)
            >>> self.getCatIds()
            >>> self.getCatIds(catNms=['superstar'])
            >>> self.getCatIds(supNms=['raster'])
            >>> self.getCatIds(catIds=[3])
        """
        catNms = catNms if ub.iterable(catNms) else [catNms]
        supNms = supNms if ub.iterable(supNms) else [supNms]
        catIds = catIds if ub.iterable(catIds) else [catIds]

        cats = self.dataset['categories']
        if catNms or supNms or catIds:
            cats = self.dataset['categories']
            if catNms:
                cats = [cat for cat in cats if cat['name'] in catNms]
            if supNms:
                cats = [
                    cat for cat in cats if cat.get(
                        'supercategory',
                        None) in supNms]
            if catIds:
                cats = [cat for cat in cats if cat['id'] in catIds]

        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.

        Args:
            imgIds (List[int]) : get imgs for given ids
            catIds (List[int]) : get imgs with all given cats

        Returns:
            List[int]: integer array of img ids

        Example:
            >>> from kwcoco.compat_dataset import *  # NOQA
            >>> import kwcoco
            >>> self = COCO(kwcoco.CocoDataset.demo('shapes8').dataset)
            >>> self.getImgIds(imgIds=[1, 2])
            >>> self.getImgIds(catIds=[3, 6, 7])
            >>> self.getImgIds(catIds=[3, 6, 7], imgIds=[1, 2])
        '''
        imgIds = imgIds if ub.iterable(imgIds) else [imgIds]
        catIds = catIds if ub.iterable(catIds) else [catIds]

        if not imgIds:
            valid_gids = set(self.imgs.keys())
        else:
            valid_gids = set(imgIds)

        if catIds:
            hascat_gids = set()
            for aids in ub.take(self.index.cid_to_aids, catIds):
                hascat_gids |= set(self.annots(aids).lookup('image_id'))

            valid_gids &= hascat_gids
        return sorted(valid_gids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.

        Args:
            ids (List[int]) : integer ids specifying anns

        Returns:
            List[dict]: loaded ann objects
        """
        if isinstance(ids, int):
            return [self.anns[ids]]
        return [self.anns[id] for id in ids]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.

        Args:
            ids (List[int]) : integer ids specifying cats

        Returns:
            List[dict]: loaded cat objects
        """
        if isinstance(ids, int):
            return [self.cats[ids]]
        return [self.cats[id] for id in ids]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.

        Args:
            ids (List[int] | int) : integer ids specifying img

        Returns:
            List[dict]: loaded img objects
        """
        if isinstance(ids, int):
            return [self.imgs[ids]]
        return [self.imgs[id] for id in ids]

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.

        Args:
            anns (List[Dict]): annotations to display
        """
        aids = [ann['id'] for ann in anns]
        self.show_image(aids=aids, show_boxes=draw_bbox)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.

        Args:
            resFile (str | ndarray | List[Dict]):
                file name of result file or something else that resolves to a
                json list of annotation dictionaries corresponding to
                predictions.

        Returns:
            COCO: res result api object

        Example:
            >>> from kwcoco.compat_dataset import *  # NOQA
            >>> import kwcoco
            >>> from kwcoco.demo.perterb import perterb_coco
            >>> truth = kwcoco.CocoDataset.demo('shapes8').conform(legacy=True)
            >>> self = COCO(truth.dataset)
            >>> dpath = ub.Path.appdir('kwcoco/tests/compat').ensuredir()
            >>> pred = perterb_coco(truth)
            >>> # Get a type of input loadRes accepts
            >>> anns = pred.dataset['annotations']
            >>> # This function handles 4 different types of result annotations
            >>> capn_anns = [(ub.udict(ann) | {'caption': 'stuff'}) & {'caption', 'image_id'} for ann in anns]
            >>> bbox_anns = [ub.udict(ann) & {'bbox', 'image_id', 'category_id',} for ann in anns]
            >>> sseg_anns = [ub.udict(ann) & {'segmentation', 'image_id', 'category_id'} for ann in anns]
            >>> kpts_anns = [ub.udict(ann) & {'keypoints', 'image_id', 'category_id'} for ann in anns]
            >>> # Ensure we can get a result object for each
            >>> res = self.loadRes(capn_anns)
            >>> res = self.loadRes(bbox_anns)
            >>> res = self.loadRes(sseg_anns)
            >>> res = self.loadRes(kpts_anns)
        """
        import json
        import time
        import copy
        import kwimage
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if isinstance(resFile, str):
            anns = json.load(open(resFile))
        elif isinstance(resFile, np.ndarray):
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert isinstance(anns, list), 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'
        if len(anns):
            if 'caption' in anns[0]:
                imgIds = set([img['id'] for img in res.dataset['images']]) & set(
                    [ann['image_id'] for ann in anns])
                res.dataset['images'] = [
                    img for img in res.dataset['images'] if img['id'] in imgIds]
                for id, ann in enumerate(anns):
                    ann['id'] = id + 1
            elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
                res.dataset['categories'] = copy.deepcopy(
                    self.dataset['categories'])
                for id, ann in enumerate(anns):
                    bb = ann['bbox']
                    x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                    if 'segmentation' not in ann:
                        ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                    ann['area'] = bb[2] * bb[3]
                    ann['id'] = id + 1
                    ann['iscrowd'] = 0
            elif 'segmentation' in anns[0]:
                res.dataset['categories'] = copy.deepcopy(
                    self.dataset['categories'])
                for id, ann in enumerate(anns):
                    # Unlike the original, using kwimage will support multiple
                    # segmentation formats, even in legacy mode.
                    sseg = kwimage.Segmentation.coerce(ann['segmentation'])
                    # ann['area'] = maskUtils.area(ann['segmentation'])
                    if 'bbox' not in ann:
                        ann['bbox'] = sseg.to_multi_polygon().box().to_coco()
                        # ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                    ann['id'] = id + 1
                    ann['iscrowd'] = 0
            elif 'keypoints' in anns[0]:
                res.dataset['categories'] = copy.deepcopy(
                    self.dataset['categories'])
                for id, ann in enumerate(anns):
                    s = kwimage.Points.coerce(ann['keypoints']).to_coco()
                    if len(s):
                        x = s[0::3]
                        y = s[1::3]
                        x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                        ann['area'] = (x1 - x0) * (y1 - y0)
                        ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
                    else:
                        ann['area'] = 0
                        ann['bbox'] = [0, 0, 0, 0]
                        print('Warning: annotation missing keypoints')
                    ann['id'] = id + 1
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download(self, tarDir=None, imgIds=[]):
        '''
        Download COCO images from mscoco.org server.

        Args:
            tarDir (str | PathLike | None): COCO results directory name
            imgIds (list): images to be downloaded
        '''
        if tarDir is not None:
            self.reroot(tarDir)
        if not imgIds:
            imgIds = None
        self._ensure_image_data(gids=imgIds)

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}

        Args:
            data (numpy.ndarray)

        Returns:
            List[Dict]: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert isinstance(data, np.ndarray)
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in ub.ProgIter(range(N)):
            ann += [{
                'image_id': int(data[i, 0]),
                'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                'score': data[i, 5],
                'category_id': int(data[i, 6]),
            }]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.

        Returns:
            Dict: containing the size and counts to define the RLE.

        Note:
            * This requires the C-extensions for kwimage to be installed (i.e.
            ``pip install kwimage_ext``) due to the need to interface with the
            bytes RLE format.

        Example:
            >>> from kwcoco.compat_dataset import *  # NOQA
            >>> import kwcoco
            >>> self = COCO(kwcoco.CocoDataset.demo('shapes8').dataset)
            >>> try:
            >>>     rle = self.annToRLE(self.anns[1])
            >>> except NotImplementedError:
            >>>     import pytest
            >>>     pytest.skip('missing kwimage c-extensions')
            >>> else:
            >>>     assert len(rle['counts']) > 2
            >>> # xdoctest: +REQUIRES(module:pycocotools)
            >>> self.conform(legacy=True)
            >>> orig = self._aspycoco().annToRLE(self.anns[1])
        """
        import kwimage
        aid = ann['id']
        ann = self.anns[aid]
        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        data = ann['segmentation']
        dims = (h, w)
        sseg = kwimage.Segmentation.coerce(data, dims=dims)
        try:
            mask = sseg.to_mask(dims=dims)
            rle = mask.to_bytes_rle().data
        except NotImplementedError:
            raise NotImplementedError((
                'kwimage does not seem to have required '
                'c-extensions for bytes RLE'))
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to
        binary mask.

        Returns:
            ndarray: binary mask (numpy 2D array)

        Note:
            The mask is returned as a fortran (F-style) array with the same
            dimensions as the parent image.

        Ignore:
            >>> from kwcoco.compat_dataset import *  # NOQA
            >>> import kwcoco
            >>> self = COCO(kwcoco.CocoDataset.demo('shapes8').dataset)
            >>> mask = self.annToMask(self.anns[1])
            >>> # xdoctest: +REQUIRES(module:pycocotools)
            >>> self.conform(legacy=True)
            >>> orig = self._aspycoco().annToMask(self.anns[1])
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> diff = kwimage.normalize((compat_mask - orig_mask).astype(np.float32))
            >>> kwplot.imshow(diff)
            >>> kwplot.show_if_requested()
        """
        import kwimage
        aid = ann['id']
        ann = self.anns[aid]
        data = ann['segmentation']
        dims = None
        sseg = kwimage.Segmentation.coerce(data, dims=dims)
        try:
            mask = sseg.to_mask()
        except Exception:
            img = self.imgs[ann['image_id']]
            dims = (img['height'], img['width'])
            mask = sseg.to_mask(dims=dims)
        m = mask.to_fortran_mask().data
        return m
