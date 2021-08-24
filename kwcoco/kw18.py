# -*- coding: utf-8 -*-
"""
A helper for converting COCO to / from KW18 format.

KW18 File Format
https://docs.google.com/spreadsheets/d/1DFCwoTKnDv8qfy3raM7QXtir2Fjfj9j8-z8px5Bu0q8/edit#gid=10

The kw18.trk files are text files, space delimited; each row is one
frame of one track and all rows have the same number of columns. The fields are:

.. code ::

    01) track_ID         : identifies the track
    02) num_frames:     number of frames in the track
    03) frame_id        : frame number for this track sample
    04) loc_x        : X-coordinate of the track (image/ground coords)
    05) loc_y        : Y-coordinate of the track (image/ground coords)
    06) vel_x        : X-velocity of the object (image/ground coords)
    07) vel_y        : Y-velocity of the object (image/ground coords)
    08) obj_loc_x        : X-coordinate of the object (image coords)
    09) obj_loc_y        : Y-coordinate of the object (image coords)
    10) bbox_min_x    : minimum X-coordinate of bounding box (image coords)
    11) bbox_min_y    : minimum Y-coordinate of bounding box (image coords)
    12) bbox_max_x    : maximum X-coordinate of bounding box (image coords)
    13) bbox_max_y    : maximum Y-coordinate of bounding box (image coords)
    14) area        : area of object (pixels)
    15) world_loc_x    : X-coordinate of object in world
    16) world_loc_y    : Y-coordinate of object in world
    17) world_loc_z    : Z-coordiante of object in world
    18) timestamp        : timestamp of frame (frames)
    For the location and velocity of object centroids, use fields 4-7.
    Bounding box is specified using coordinates of the top-left and bottom
    right corners. Fields 15-17 may be ignored.

    The kw19.trk and kw20.trk files, when present, add the following field(s):
    19) object class: estimated class of the object, either 1 (person), 2
    (vehicle), or 3 (other).
    20) Activity ID -- refer to activities.txt for index and list of activities.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import kwarray
import numpy as np


class KW18(kwarray.DataFrameArray):
    """
    A DataFrame like object that stores KW18 column data

    Example:
        >>> import kwcoco
        >>> from kwcoco.kw18 import KW18
        >>> coco_dset = kwcoco.CocoDataset.demo('shapes')
        >>> kw18_dset = KW18.from_coco(coco_dset)
        >>> print(kw18_dset.pandas())
    """

    # Define the ordering of the kw18 columns
    DEFAULT_COLUMNS = [
        'track_id',                                      # 1
        'track_length',                                  # 2
        'frame_number',                                  # 3
        'tracking_plane_loc_x', 'tracking_plane_loc_y',  # 4-5
        'velocity_x', 'velocity_y',                      # 6-7
        'image_loc_x', 'image_loc_y',                    # 8-9
        'img_bbox_tl_x', 'img_bbox_tl_y',                # 10-13
        'img_bbox_br_x', 'img_bbox_br_y',
        'area',                                          # 14
        'world_loc_x', 'world_loc_y', 'world_loc_z',     # 15-17
        'timestamp',                                     # 18
        # kw18 can have more than 18 columns.
        'confidence',                                    # 19
        'object_type_id',                                # 20
        'activity_type_id',                              # 21
    ]

    def __init__(self, data):
        """
        Args:
            data : the kw18 data frame.
        """
        super().__init__(data)

    @classmethod
    def demo(KW18):
        import kwcoco
        coco_dset = kwcoco.CocoDataset.demo('shapes8')
        self = KW18.from_coco(coco_dset)
        return self

    @classmethod
    def from_coco(KW18, coco_dset):
        import kwimage
        raw = {col: None for col in KW18.DEFAULT_COLUMNS}
        anns = coco_dset.dataset['annotations']
        boxes = kwimage.Boxes(np.array([ann['bbox'] for ann in anns]), 'xywh')
        tlbr = boxes.to_tlbr()
        cxywh = tlbr.to_cxywh()
        tl_x, tl_y, br_x, br_y = tlbr.data.T

        cx = cxywh.data[:, 0]
        cy = cxywh.data[:, 1]

        # Create track ids if not given
        track_ids = np.array([ann.get('track_id', np.nan) for ann in anns])
        missing = np.isnan(track_ids)
        valid_track_ids = track_ids[~missing]
        if len(valid_track_ids) == 0:
            next_track_id = 1
        else:
            next_track_id = valid_track_ids.max() + 1
        num_need = np.sum(missing)
        new_track_ids = np.arange(next_track_id, next_track_id + num_need)
        track_ids[missing] = new_track_ids
        track_ids = track_ids.astype(int)

        scores = np.array([ann.get('score', -1) for ann in anns])
        image_ids = np.array([ann['image_id'] for ann in anns])
        cids = np.array([ann.get('category_id', -1) for ann in anns])

        num = len(anns)

        raw['track_id'] = track_ids
        raw['track_length'] = np.full(num, fill_value=-1)
        raw['frame_number'] = image_ids

        raw['tracking_plane_loc_x'] = cx
        raw['tracking_plane_loc_y'] = cy

        raw['velocity_x'] = np.full(num, fill_value=0)
        raw['velocity_y'] = np.full(num, fill_value=0)

        raw['image_loc_x'] = cx
        raw['image_loc_y'] = cy

        raw['img_bbox_tl_x'] = tl_x
        raw['img_bbox_tl_y'] = tl_y
        raw['img_bbox_br_x'] = br_x
        raw['img_bbox_br_y'] = br_y

        raw['area'] = boxes.area.ravel()

        raw['world_loc_x'] = np.full(num, fill_value=-1)
        raw['world_loc_y'] = np.full(num, fill_value=-1)
        raw['world_loc_z'] = np.full(num, fill_value=-1)

        raw['timestamp'] = np.full(num, fill_value=-1)

        raw['confidence'] = scores
        raw['object_type_id'] = cids

        raw = {k: v for k, v in raw.items() if v is not None}

        track_ids, groupxs = kwarray.group_indices(raw['track_id'])
        for groupx in groupxs:
            raw['track_length'][groupx] = len(groupx)

        self = KW18(raw)
        return self

    def to_coco(self, image_paths=None, video_name=None):
        """
        Translates a kw18 files to a CocoDataset.

        Note:
            kw18 does not contain complete information, and as such
            the returned coco dataset may need to be augmented.

        Args:
            image_paths (Dict[int, str], default=None):
                if specified, maps frame numbers to image file paths.

            video_name (str, default=None):
                if specified records the name of the video this kw18 belongs to

        TODO:
            - [X] allow kwargs to specify path to frames / videos

        Example:
            >>> from kwcoco.kw18 import KW18
            >>> from os.path import join
            >>> import ubelt as ub
            >>> import kwimage
            >>> # Prep test data - autogen a demo kw18 and write it to disk
            >>> dpath = ub.ensure_app_cache_dir('kwcoco/kw18')
            >>> kw18_fpath = join(dpath, 'test.kw18')
            >>> KW18.demo().dump(kw18_fpath)
            >>> #
            >>> # Load the kw18 file
            >>> self = KW18.load(kw18_fpath)
            >>> # Pretend that these image correspond to kw18 frame numbers
            >>> frame_names = [kwimage.grab_test_image_fpath(k) for k in kwimage.grab_test_image.keys()]
            >>> frame_ids = sorted(set(self['frame_number']))
            >>> image_paths = dict(zip(frame_ids, frame_names))
            >>> #
            >>> # Convert the kw18 to kwcoco and specify paths to images
            >>> coco_dset = self.to_coco(image_paths=image_paths, video_name='dummy.mp4')
            >>> #
            >>> # Now we can draw images
            >>> canvas = coco_dset.draw_image(1)
            >>> # xdoctest: +REQUIRES(--draw)
            >>> kwimage.imwrite('foo.jpg', canvas)
            >>> # Draw all iamges
            >>> for gid in coco_dset.imgs.keys():
            >>>     canvas = coco_dset.draw_image(gid)
            >>>     fpath = join(dpath, 'gid_{}.jpg'.format(gid))
            >>>     print('write fpath = {!r}'.format(fpath))
            >>>     kwimage.imwrite(fpath, canvas)
        """
        import kwcoco
        import ubelt as ub
        dset = kwcoco.CocoDataset()

        # kw18s don't have category names, so use ids as proxies
        unique_category_ids = sorted(set(self['object_type_id']))
        for cid in unique_category_ids:
            dset.ensure_category('class_{}'.format(cid), id=cid)

        unique_frame_idxs = ub.argunique(self['frame_number'])

        # kw18 files correspond to one video
        vidid = 1
        dset.add_video(id=vidid, name='unknown_kw18_video')

        # Index frames of the video
        for idx in unique_frame_idxs:
            frame_num = self['frame_number'][idx]
            timestamp = self['timestamp'][idx]
            if image_paths and frame_num in image_paths:
                file_name = image_paths[frame_num]
            else:
                file_name = '<unknown_image_{}>'.format(frame_num)
            dset.add_image(
                id=frame_num,
                file_name=file_name,
                video_id=vidid,
                frame_index=frame_num,
                timestamp=timestamp
            )

        for rx, row in self.iterrows():
            tl_x = row['img_bbox_tl_x']
            tl_y = row['img_bbox_tl_y']
            br_x = row['img_bbox_br_x']
            br_y = row['img_bbox_br_y']
            w = br_x - tl_x
            h = br_y - tl_y
            bbox = [tl_x, tl_y, w, h]

            world_loc = (row['world_loc_x'], row['world_loc_y'], row['world_loc_z'])
            velocity = (row['velocity_x'], row['velocity_y'])

            kw = {}
            if 'confidence' in row:
                kw['score'] = row['confidence']

            dset.add_annotation(
                id=rx,
                image_id=row['frame_number'],
                category_id=row['object_type_id'],
                track_id=row['track_id'],
                bbox=bbox,
                area=row['area'],
                velocity=velocity,
                world_loc=world_loc,
                **kw)
        return dset

    @classmethod
    def load(KW18, file):
        """
        Example:
            >>> import kwcoco
            >>> from kwcoco.kw18 import KW18
            >>> coco_dset = kwcoco.CocoDataset.demo('shapes')
            >>> kw18_dset = KW18.from_coco(coco_dset)
            >>> print(kw18_dset.pandas())
        """
        import pandas as pd
        try:
            EmptyDataError = pd.errors.EmptyDataError
        except Exception:
            EmptyDataError = pd.io.common.EmptyDataError

        try:
            df = pd.read_csv(
                file, sep=' +', comment='#', header=None, engine='python')
        except EmptyDataError:
            df = pd.DataFrame()
        renamer = dict(zip(df.columns, KW18.DEFAULT_COLUMNS))
        raw = df.rename(columns=renamer)
        raw = _ensure_kw18_column_order(raw)
        self = KW18(raw)
        return self

    @classmethod
    def loads(KW18, text):
        """
        Example:
            >>> self = KW18.demo()
            >>> text = self.dumps()
            >>> self2 = KW18.loads(text)
            >>> empty = KW18.loads('')
        """
        import io
        file = io.StringIO()
        file.write(text)
        file.seek(0)
        self = KW18.load(file)
        return self

    def dump(self, file):
        import six
        if isinstance(file, six.string_types):
            with open(file, 'w') as fp:
                self.dump(fp)
        else:
            df = self.pandas()
            # Write column header
            file.write('#' + ' '.join(df.columns) + '\n')
            df.to_csv(file, sep=' ', mode='a', index=False, header=False)

    def dumps(self):
        """
        Example:
            >>> self = KW18.demo()
            >>> text = self.dumps()
            >>> print(text)
        """
        import io
        file = io.StringIO()
        self.dump(file)
        file.seek(0)
        text = file.read()
        return text


def _ensure_kw18_column_order(df):
    """
    Ensure expected kw18 columns exist and are in the correct order.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(columns=KW18.DEFAULT_COLUMNS[0:18])
        >>> _ensure_kw18_column_order(df)
        >>> df = pd.DataFrame(columns=KW18.DEFAULT_COLUMNS[0:19])
        >>> _ensure_kw18_column_order(df)
        >>> df = pd.DataFrame(columns=KW18.DEFAULT_COLUMNS[0:18] + KW18.DEFAULT_COLUMNS[20:21])
        >>> assert np.all(_ensure_kw18_column_order(df).columns == df.columns)
    """
    columns = list(KW18.DEFAULT_COLUMNS)

    # Columns after the 18th are optional
    # (note: the post 18th column spec not well defined in general)
    optional_columns = KW18.DEFAULT_COLUMNS[18:]
    for col in optional_columns[::-1]:
        if col not in df.columns:
            columns.remove(col)

    if len(df) == 0:
        # Ensure empty data frames have columns
        df = df.reindex(columns=columns)

    missing_cols = [c for c in columns if c not in df.columns]
    unknown_cols = [c for c in df.columns if c not in columns]

    if missing_cols:
        raise ValueError('missing_cols = {!r}'.format(missing_cols))

    if unknown_cols:
        raise ValueError('unknown_cols = {!r}'.format(unknown_cols))

    df = df.reindex(columns=columns)
    return df
