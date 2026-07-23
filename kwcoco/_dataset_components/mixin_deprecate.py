from __future__ import annotations
import ubelt as ub
from typing import Any


class MixinCocoDeprecate:
    """
    These functions are marked for deprecation and will be removed
    """

    def keypoint_annotation_frequency(self) -> Any:
        """
        DEPRECATED

        Example:
            >>> import kwcoco
            >>> import ubelt as ub
            >>> self = kwcoco.CocoDataset.demo('shapes', rng=0)
            >>> hist = self.keypoint_annotation_frequency()
            >>> hist = ub.odict(sorted(hist.items()))
            >>> # FIXME: for whatever reason demodata generation is not deterministic when seeded
            >>> print(ub.urepr(hist))  # xdoc: +IGNORE_WANT
            {
                'bot_tip': 6,
                'left_eye': 14,
                'mid_tip': 6,
                'right_eye': 14,
                'top_tip': 6,
            }
        """
        ub.schedule_deprecation(
            'kwcoco',
            name='keypoint_annotation_frequency',
            type='method',
            deprecate='0.3.4',
            error='1.0.0',
            remove='1.1.0',
            migration=(
                'Implement this functionality explicitly. '
                'It is too niche for a the core API.'
                'Or propose a better way on '
                'https://gitlab.kitware.com/computer-vision/kwcoco/-/issues '
            ),
        )
        ann_kpcids = [
            kp['keypoint_category_id']
            for ann in self.dataset['annotations']  # type: ignore
            for kp in ann.get('keypoints', [])
        ]
        kpcid_to_name = {
            kpcat['id']: kpcat['name'] for kpcat in self.dataset['keypoint_categories']  # type: ignore
        }
        kpcid_to_num = ub.dict_hist(ann_kpcids, labels=list(kpcid_to_name.keys()))
        kpname_to_num = ub.map_keys(kpcid_to_name, kpcid_to_num)
        return kpname_to_num

    def category_annotation_type_frequency(self) -> Any:
        """
        DEPRECATED

        Reports the number of annotations of each type for each category

        Example:
            >>> import kwcoco
            >>> self = kwcoco.CocoDataset.demo()
            >>> hist = self.category_annotation_frequency()
            >>> print(ub.urepr(hist))
        """
        catname_to_nannot_types = {}
        ub.schedule_deprecation(
            'kwcoco',
            name='category_annotation_type_frequency',
            type='method',
            deprecate='0.3.4',
            error='1.0.0',
            remove='1.1.0',
            migration=(
                'Implement this functionality explicitly. '
                'It is too niche for a the core API.'
                'Or propose a better way on '
                'https://gitlab.kitware.com/computer-vision/kwcoco/-/issues '
            ),
        )

        def _annot_type(ann):
            """
            Returns what type of annotation ``ann`` is.
            """
            return tuple(sorted(set(ann) & {'bbox', 'line', 'keypoints'}))

        for cid, aids in self.index.cid_to_aids.items():  # type: ignore
            name = self.cats[cid]['name']  # type: ignore
            hist = ub.dict_hist(map(_annot_type, ub.take(self.anns, aids)))  # type: ignore
            catname_to_nannot_types[name] = ub.map_keys(
                lambda k: k[0] if len(k) == 1 else k, hist
            )
        return catname_to_nannot_types

    def imread(self, gid: Any) -> Any:
        """
        DEPRECATED: use load_image or delayed_image

        Loads a particular image
        """
        ub.schedule_deprecation(
            'kwcoco',
            name='imread',
            type='method',
            deprecate='0.3.4',
            error='1.0.0',
            remove='1.1.0',
            migration=('use `self.coco_image(gid).imdelay().finalize()`.'),
        )
        return self.load_image(gid)  # type: ignore
