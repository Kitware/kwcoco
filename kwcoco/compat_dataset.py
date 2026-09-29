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
from __future__ import annotations

from kwcoco.coco_dataset import CocoDataset
from kwcoco._helpers import _coerce_info_list
from typing import Any, Iterable, List, Optional, Protocol, Union
from os import PathLike
import itertools as it
import ubelt as ub
import numpy as np

class _ImgToAnnsProxy(Protocol):
    def __getitem__(self, gid: int, /) -> list[dict[str, Any]]: ...
    def __contains__(self, key: object, /) -> bool: ...
    def keys(self) -> Iterable[int]: ...


class _CatToImgsProxy(Protocol):
    def __getitem__(self, cid: int, /) -> Iterable[int]: ...
    def keys(self) -> Iterable[int]: ...


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

    def __init__(self, annotation_file: Any = None, **kw: Any) -> None:
        if annotation_file is not None:
            if 'data' in kw:
                raise ValueError('cannot specify data and annotation file')
        if 'data' in kw and annotation_file is None:
            annotation_file = kw.pop('data')
        super().__init__(annotation_file, **kw)

    def createIndex(self) -> None:
        self.rebuild_index()

    def info(self) -> None:
        """
        Print information about the annotation file.
        """
        for item in _coerce_info_list(self.dataset.get('info', {})):
            for key, value in item.items():
                print('{}: {}'.format(key, value))

    @property
    def imgToAnns(self) -> _ImgToAnnsProxy:
        from kwcoco.util.dict_proxy2 import DictInterface

        class ProxyImgToAnns(DictInterface):
            def __init__(self, parent: COCO) -> None:
                self.parent = parent

            def __getitem__(self, key: int) -> list[dict[str, Any]]:
                aids = self.parent.index.gid_to_aids[key]
                anns = list(ub.take(self.parent.index.anns, aids))
                return anns

            def keys(self) -> Iterable[int]:
                return self.parent.index.gid_to_aids.keys()

        imgToAnns = ProxyImgToAnns(parent=self)
        return imgToAnns

    @property
    def catToImgs(self) -> _CatToImgsProxy:
        """
        unlike the name implies, this actually goes from category to image ids
        Name retained for backward compatibility
        """
        catToImgs = self.index.cid_to_gids
        return catToImgs

    def getAnnIds(
        self,
        imgIds: int | Iterable[int] = [],
        catIds: int | Iterable[int] = [],
        areaRng: Iterable[float] = [],
        iscrowd: bool | None = None,
    ) -> list[int]:
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
        img_ids = [imgIds] if isinstance(imgIds, int) else list(imgIds)
        cat_ids = [catIds] if isinstance(catIds, int) else list(catIds)
        area_rng = list(areaRng)

        if len(img_ids) == len(cat_ids) == len(area_rng) == 0:
            anns = self.dataset['annotations']
        else:
            if img_ids:
                lists = [
                    self.imgToAnns[imgId]
                    for imgId in img_ids
                    if imgId in self.imgToAnns
                ]
                anns = list(it.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = (
                anns
                if not cat_ids
                else [ann for ann in anns if ann['category_id'] in cat_ids]
            )
            anns = (
                anns
                if not area_rng
                else [
                    ann
                    for ann in anns
                    if ann['area'] > area_rng[0] and ann['area'] < area_rng[1]
                ]
            )
        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(
        self,
        catNms: str | Iterable[str] = [],
        supNms: str | Iterable[str] = [],
        catIds: int | Iterable[int] = [],
    ) -> list[int]:
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
        cat_nms = [catNms] if isinstance(catNms, str) else list(catNms)
        sup_nms = [supNms] if isinstance(supNms, str) else list(supNms)
        cat_ids = [catIds] if isinstance(catIds, int) else list(catIds)

        cats = self.dataset['categories']
        if cat_nms or sup_nms or cat_ids:
            cats = self.dataset['categories']
            if cat_nms:
                cats = [cat for cat in cats if cat['name'] in cat_nms]
            if sup_nms:
                cats = [cat for cat in cats if cat.get('supercategory', None) in sup_nms]
            if cat_ids:
                cats = [cat for cat in cats if cat['id'] in cat_ids]

        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(
        self,
        imgIds: int | Iterable[int] = [],
        catIds: int | Iterable[int] = [],
    ) -> list[int]:
        """
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
        """
        img_ids = [imgIds] if isinstance(imgIds, int) else list(imgIds)
        cat_ids = [catIds] if isinstance(catIds, int) else list(catIds)

        if not img_ids:
            valid_gids = set(self.imgs.keys())
        else:
            valid_gids = set(img_ids)

        for catId in cat_ids:
            aids = self.index.cid_to_aids.get(catId, [])  # type: ignore
            hascat_gids = set(self.annots(list(aids)).lookup('image_id'))
            valid_gids &= hascat_gids
        return sorted(valid_gids)

    def loadAnns(
        self, ids: int | Iterable[int] = []
    ) -> list[dict[str, Any]]:
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

    def loadCats(
        self, ids: int | Iterable[int] = []
    ) -> list[dict[str, Any]]:
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

    def loadImgs(
        self, ids: int | Iterable[int] = []
    ) -> list[dict[str, Any]]:
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

    def showAnns(
        self, anns: Iterable[dict[str, Any]], draw_bbox: bool = False
    ) -> None:
        """
        Display the specified annotations.

        Args:
            anns (List[Dict]): annotations to display
        """
        aids = [ann['id'] for ann in anns]
        self.show_image(aids=aids, show_boxes=draw_bbox)

    def loadRes(
        self, resFile: str | np.ndarray | list[dict[str, Any]]
    ) -> COCO:
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
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), (
            'Results do not correspond to current coco set'
        )
        if len(anns):
            if 'caption' in anns[0]:
                imgIds = set([img['id'] for img in res.dataset['images']]) & set(
                    [ann['image_id'] for ann in anns]
                )
                res.dataset['images'] = [
                    img for img in res.dataset['images'] if img['id'] in imgIds
                ]
                for id, ann in enumerate(anns):
                    ann['id'] = id + 1  # type: ignore
            elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
                res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                for id, ann in enumerate(anns):
                    bb = ann['bbox']  # type: ignore
                    x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                    if 'segmentation' not in ann:  # type: ignore
                        ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]  # type: ignore
                    ann['area'] = bb[2] * bb[3]  # type: ignore
                    ann['id'] = id + 1  # type: ignore
                    ann['iscrowd'] = 0  # type: ignore
            elif 'segmentation' in anns[0]:
                res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                for id, ann in enumerate(anns):
                    # Unlike the original, using kwimage will support multiple
                    # segmentation formats, even in legacy mode.
                    sseg = kwimage.Segmentation.coerce(ann['segmentation'])  # type: ignore
                    # ann['area'] = maskUtils.area(ann['segmentation'])
                    if 'bbox' not in ann:  # type: ignore
                        ann['bbox'] = sseg.to_multi_polygon().box().to_coco()  # type: ignore
                        # ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                    ann['id'] = id + 1  # type: ignore
                    ann['iscrowd'] = 0  # type: ignore
            elif 'keypoints' in anns[0]:
                res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
                for id, ann in enumerate(anns):
                    s = kwimage.Points.coerce(ann['keypoints']).to_coco()  # type: ignore
                    if len(s):
                        x = s[0::3]
                        y = s[1::3]
                        x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                        ann['area'] = (x1 - x0) * (y1 - y0)  # type: ignore
                        ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]  # type: ignore
                    else:
                        ann['area'] = 0  # type: ignore
                        ann['bbox'] = [0, 0, 0, 0]  # type: ignore
                        print('Warning: annotation missing keypoints')
                    ann['id'] = id + 1  # type: ignore
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download(
        self,
        dpath: Optional[Union[str, PathLike]] = None,
        image_ids: Optional[Union[int, Iterable[int]]] = None,
        overwrite: bool = False,
        verbose: int = 1,
        *,
        tarDir: Optional[Union[str, PathLike]] = None,
        imgIds: Optional[Union[int, Iterable[int]]] = None,
    ) -> List[str]:
        """
        Download images from their standard ``coco_url`` fields.

        This accepts the native kwcoco argument names while retaining the
        legacy pycocotools keyword aliases ``tarDir`` and ``imgIds``.

        Args:
            dpath: native kwcoco target directory.
            image_ids: native kwcoco image identifiers.
            overwrite: if true, redownload existing files.
            verbose: verbosity level.
            tarDir: legacy pycocotools alias for ``dpath``.
            imgIds: legacy pycocotools alias for ``image_ids``. An empty
                legacy selection retains pycocotools' "all images" behavior.

        Returns:
            Local paths corresponding to the requested images.

        Note:
            The native downloader officially supports HTTPS URLs. Image
            ``file_name`` values are trusted destination paths and may escape
            the target directory when absolute or when they contain ``..``.
        """
        if tarDir is not None:
            if dpath is not None:
                raise TypeError('Specify only one of dpath or tarDir')
            dpath = tarDir
        if imgIds is not None:
            if image_ids is not None:
                raise TypeError('Specify only one of image_ids or imgIds')
            image_ids = imgIds
            if not image_ids:
                image_ids = None
        return super().download(
            dpath=dpath,
            image_ids=image_ids,
            overwrite=overwrite,
            verbose=verbose,
        )

    def loadNumpyAnnotations(
        self, data: np.ndarray
    ) -> list[dict[str, Any]]:
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
            ann += [
                {
                    'image_id': int(data[i, 0]),
                    'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                    'score': data[i, 5],
                    'category_id': int(data[i, 6]),
                }
            ]
        return ann

    def annToRLE(self, ann: dict[str, Any]) -> dict[str, Any]:
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
            raise NotImplementedError(
                ('kwimage does not seem to have required c-extensions for bytes RLE')
            )
        return rle

    def annToMask(self, ann: dict[str, Any]) -> np.ndarray:
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
