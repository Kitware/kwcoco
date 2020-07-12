"""
A wrapper around the basic kwcoco dataset with a pycocotools API.
"""
from kwcoco.coco_dataset import CocoDataset
import itertools as it
import ubelt as ub


class PyCocoAPI(CocoDataset):
    """
    A wrapper around the basic kwcoco dataset with a pycocotools API.

    Example:
        >>> from kwcoco.compat_dataset import *  # NOQA
        >>> import kwcoco
        >>> basic = kwcoco.CocoDataset.demo('shapes8')
        >>> self = PyCocoAPI(basic.dataset)
    """
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file)

    def createIndex(self):
        self._build_index()

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if ub.iterable(imgIds) else [imgIds]
        catIds = catIds if ub.iterable(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(it.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
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
                cats = [cat for cat in cats if cat['supercategory'] in supNms]
            if catIds:
                cats = [cat for cat in cats if cat['id'] in catIds]

        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
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
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        return list(ub.take(self.anns, ids))

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        return list(ub.take(self.cats, ids))

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        return list(map(self.load_image, ids))

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        raise NotImplementedError

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        raise NotImplementedError

    def download(self, tarDir=None, imgIds=[]):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        if tarDir is not None:
            self.reroot(tarDir)
        if not imgIds:
            imgIds = None
        self._ensure_image_data(gids=imgIds)

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        raise NotImplementedError

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        raise NotImplementedError

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        raise NotImplementedError
