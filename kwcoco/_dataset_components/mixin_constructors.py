from __future__ import annotations

from typing import Any, Callable, IO, Literal, overload, TYPE_CHECKING, cast
from functools import partial
from os.path import exists, join
import os
import warnings
import ubelt as ub

from kwcoco._helpers import _load_and_postprocess
from kwcoco._dataset_components.json_backend import json_r

if TYPE_CHECKING:
    from os import PathLike
    from kwcoco.coco_dataset import CocoDataset
    from kwcoco.coco_sql_dataset import CocoSqlDatabase


def demo_coco_data() -> dict:
    from kwcoco.coco_dataset import demo_coco_data as _demo_coco_data
    return _demo_coco_data()


class MixinCocoConstructors:
    """
    Classmethods for constructing CocoDataset objects
    """

    @overload
    @classmethod
    def coerce(
        cls,
        key: CocoDataset,
        sqlview: bool | str = False,
        verbose: int = 0,
        **kw: Any,
    ) -> CocoDataset: ...

    @overload
    @classmethod
    def coerce(
        cls,
        key: CocoSqlDatabase,
        sqlview: bool | str = False,
        verbose: int = 0,
        **kw: Any,
    ) -> CocoSqlDatabase: ...

    @overload
    @classmethod
    def coerce(
        cls,
        key: Any,
        sqlview: Literal[False] = False,
        verbose: int = 0,
        **kw: Any,
    ) -> CocoDataset: ...

    @overload
    @classmethod
    def coerce(
        cls,
        key: Any,
        sqlview: Literal[True] | str,
        verbose: int = 0,
        **kw: Any,
    ) -> CocoSqlDatabase: ...

    @overload
    @classmethod
    def coerce(
        cls,
        key: Any,
        sqlview: bool | str = False,
        verbose: int = 0,
        **kw: Any,
    ) -> CocoDataset | CocoSqlDatabase: ...

    @classmethod
    def coerce(
        cls, key: Any, sqlview: bool | str = False, verbose: int = 0, **kw: Any
    ) -> CocoDataset | CocoSqlDatabase:
        """
        Attempt to transform the input into the intended CocoDataset.

        Args:
            key : this can either be an instance of a CocoDataset, a
               string URI pointing to an on-disk dataset, or a special
               key for creating demodata.

            sqlview (bool | str):
                If truthy, will return the dataset as a cached sql view, which
                can be quicker to load and use in some instances. Can be given
                as a string, which sets the backend that is used: either sqlite
                or postgresql.  Defaults to False.

            verbose (int):
                passed to coerce or demo if special constructors are used.

            **kw: passed to whatever constructor is chosen (if any)

        Returns:
            kwcoco.CocoDataset | kwcoco.CocoSqlDatabase

        Example:
            >>> # test coerce for various input methods
            >>> import kwcoco
            >>> from kwcoco.coco_sql_dataset import assert_dsets_allclose
            >>> dct_dset = kwcoco.CocoDataset.coerce('special:shapes8')
            >>> copy1 = kwcoco.CocoDataset.coerce(dct_dset)
            >>> copy2 = kwcoco.CocoDataset.coerce(dct_dset.fpath)
            >>> assert assert_dsets_allclose(dct_dset, copy1)
            >>> assert assert_dsets_allclose(dct_dset, copy2)
            >>> # xdoctest: +REQUIRES(module:sqlalchemy)
            >>> sql_dset = dct_dset.view_sql()
            >>> copy3 = kwcoco.CocoDataset.coerce(sql_dset)
            >>> copy4 = kwcoco.CocoDataset.coerce(sql_dset.fpath)
            >>> assert assert_dsets_allclose(dct_dset, sql_dset)
            >>> assert assert_dsets_allclose(dct_dset, copy3)
            >>> assert assert_dsets_allclose(dct_dset, copy4)
        """
        import kwcoco

        if isinstance(key, cls):
            return cast('CocoDataset', key)
        if isinstance(key, os.PathLike):
            key = str(key)
        if isinstance(key, str):
            import uritools

            dset_fpath = ub.expandpath(key)
            # Parse the the "file" URI scheme
            # https://tools.ietf.org/html/rfc8089
            result = uritools.urisplit(dset_fpath)
            if result.scheme == 'sqlite' or result.path.endswith('.sqlite'):
                if verbose:
                    print(f'Reading {dset_fpath}')
                from kwcoco.coco_sql_dataset import CocoSqlDatabase

                self = CocoSqlDatabase(dset_fpath).connect()
            elif (
                result.path.endswith('.json')
                or '.json' in result.path
                or '.kwcoco' in result.path
            ):
                if verbose:
                    print(f'Reading {dset_fpath}')
                if sqlview:
                    from kwcoco.coco_sql_dataset import CocoSqlDatabase

                    kw['backend'] = sqlview
                    self = CocoSqlDatabase.coerce(dset_fpath, **kw)
                else:
                    self = kwcoco.CocoDataset(dset_fpath, **kw)
            elif result.scheme == 'special':
                self = cls.demo(key=key, verbose=verbose, **kw)
            else:
                # This case can be env-dependant in the unlikely case where you
                # have a file with the same name as a demo key. But hey, you
                # are using the coerce function, be more explicit if you want
                # predictable behavior.
                if exists(dset_fpath):
                    if verbose:
                        print(f'Reading {dset_fpath}')
                    self = kwcoco.CocoDataset(dset_fpath, **kw)
                else:
                    self = cls.demo(key=key, verbose=verbose, **kw)
        elif type(key).__name__ == 'CocoSqlDatabase':
            self = key
        elif type(key).__name__ == 'CocoDataset':
            self = key
        elif type(key).__name__ == 'CocoSampler':
            self = key.dset
        else:
            raise TypeError(type(key))
        return self

    @classmethod
    def demo(cls, key: str = 'photos', **kwargs: Any) -> CocoDataset:
        """
        Create a toy coco dataset for testing and demo purposes

        Args:
            key (str):
                Either 'photos' (default), 'shapes', or 'vidshapes'. There are
                also special suffixes that can control behavior.

                Basic options that define which flavor of demodata to generate
                are: `photos`, `shapes`, and `vidshapes`. A numeric suffix e.g.
                `vidshapes8` can be specified to indicate the size of the
                generated demo dataset.  There are other special suffixes that
                are available.  See the code in this function for explicit
                details on what is allowed.

                TODO: better documentation for these demo datasets.

                As a quick summary: the vidshapes key is the most robust and
                mature demodata set, and here are several useful variants of
                the vidshapes key.

                (1) vidshapes8 - the 8 suffix is the number of videos in this case.
                (2) vidshapes8-multispectral - generate 8 multispectral videos.
                (3) vidshapes8-msi - msi is an alias for multispectral.
                (4) vidshapes8-frames5 - generate 8 videos with 5 frames each.
                (5) vidshapes2-tracks5 - generate 2 videos with 5 tracks each.
                (6) vidshapes2-speed0.1-frames7 - generate 2 videos with 7
                frames where the objects move with with a speed of 0.1.

            **kwargs : if key is shapes, these arguments are passed to toydata
                generation. The Kwargs section of this docstring documents a
                subset of the available options. For full details, see
                :func:`demodata_toy_dset` and :func:`random_video_dset`.

                Notable options are:
                    bundle_dpath (PathLike): location to write the demo bundle
                    fpath (PathLike): location to write the demo kwcoco file

        Kwargs:
            image_size (Tuple[int, int]): width / height size of the images

            dpath (str | PathLike):
                path to the directory where any generated demo bundles will be
                written to.  Defaults to using kwcoco cache dir.

            aux (bool): if True generates dummy auxiliary channels

            rng (int | RandomState | None):
                random number generator or seed

            verbose (int): verbosity mode. Defaults to 3.

        Example:
            >>> import kwcoco
            >>> # Basic demodata keys
            >>> print(kwcoco.CocoDataset.demo('photos', verbose=1))
            >>> print(kwcoco.CocoDataset.demo('shapes', verbose=1))
            >>> print(kwcoco.CocoDataset.demo('vidshapes', verbose=1))
            >>> # Variants of demodata keys
            >>> print(kwcoco.CocoDataset.demo('shapes8', verbose=0))
            >>> print(kwcoco.CocoDataset.demo('shapes8-msi', verbose=0))
            >>> print(kwcoco.CocoDataset.demo('shapes8-frames1-speed0.2-msi', verbose=0))

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes5', num_frames=5,
            >>>                                verbose=0, rng=None)
            >>> dset = kwcoco.CocoDataset.demo('vidshapes5', num_frames=5,
            >>>                                num_tracks=4, verbose=0, rng=44)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> pnums = kwplot.PlotNums(nSubplots=len(dset.index.imgs))
            >>> fnum = 1
            >>> for gx, gid in enumerate(dset.index.imgs.keys()):
            >>>     canvas = dset.draw_image(gid=gid)
            >>>     kwplot.imshow(canvas, pnum=pnums[gx], fnum=fnum)
            >>>     #dset.show_image(gid=gid, pnum=pnums[gx])
            >>> kwplot.show_if_requested()

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes5-aux', num_frames=1,
            >>>                                verbose=0, rng=None)

        Example:
            >>> import kwcoco
            >>> dset = kwcoco.CocoDataset.demo('vidshapes1-multispectral', num_frames=5,
            >>>                                verbose=0, rng=None)
            >>> # This is the first use-case of image names
            >>> assert len(dset.index.file_name_to_img) == 0, (
            >>>     'the multispectral demo case has no "base" image')
            >>> assert len(dset.index.name_to_img) == len(dset.index.imgs) == 5
            >>> dset.remove_images([1])
            >>> assert len(dset.index.name_to_img) == len(dset.index.imgs) == 4
            >>> dset.remove_videos([1])
            >>> assert len(dset.index.name_to_img) == len(dset.index.imgs) == 0
        """
        import parse
        import re

        kwargs.pop('autobuild', None)

        if key.startswith('special:'):
            key = key.split(':')[1]

        if key.startswith('shapes'):
            from kwcoco.demo import toydata_image

            res = parse.parse('shapes{num_imgs:d}', key)
            if res:
                kwargs['n_imgs'] = int(res.named['num_imgs'])
            if 'rng' not in kwargs and 'n_imgs' in kwargs:
                kwargs['rng'] = kwargs['n_imgs']
            kwargs['verbose'] = kwargs.get('verbose', 0)
            self = toydata_image.demodata_toy_dset(**kwargs)
            self.tag = key
        elif key.startswith('vidshapes'):
            from kwcoco.demo import toydata_video

            verbose = kwargs.get('verbose', 0)
            res = parse.parse('vidshapes{num_videos:d}', key)
            if res is None:
                res = parse.parse('vidshapes{num_videos:d}-{suffix}', key)
            if res is None:
                res = parse.parse('vidshapes-{suffix}', key)
            """
            The rule is that the suffix will be split by the '-' character
            and any registered pattern or alias will impact the kwargs
            for random_video_dset
            """
            suff_parts = []
            if verbose > 3:
                print('res = {!r}'.format(res))
            if res:
                if 'num_videos' in res.named:
                    kwargs['num_videos'] = int(res.named['num_videos'])
                if 'suffix' in res.named:
                    suff_parts = [p for p in res.named['suffix'].split('-') if p]
            if verbose > 3:
                print('suff_parts = {!r}'.format(suff_parts))

            # The allowed suffix patterns and aliases are defined here
            vidkw: dict[str, Any] = {
                'render': True,
                'num_videos': 1,
                'num_frames': 2,
                'num_tracks': 2,
                'timestamps': False,
                'anchors': None,
                'image_size': (600, 600),
                'aux': None,
                'sensorchan': None,
                'multispectral': None,
                'multisensor': False,
                'max_speed': 0.01,
                'verbose': verbose,
            }
            vidkw_aliases = {
                'num_frames': {'frames'},
                'num_tracks': {'tracks'},
                'num_videos': {'videos'},
                'max_speed': {'speed'},
                'image_size': {'gsize'},
                'multispectral': {'msi'},
            }
            alias_to_key = {k: v for v, ks in vidkw_aliases.items() for k in ks}
            # These are the variables the vidshapes generator accepts
            for part in suff_parts:
                value: Any
                match = re.search(r'[\d]', part)
                if match is None:
                    value = True
                    key = part
                else:
                    key = part[: match.span()[0]]
                    value = part[match.span()[0] :]
                key = alias_to_key.get(key, key)
                if key == 'image_size':
                    value = int(value)
                if key == 'num_frames':
                    value = int(value)
                if key == 'num_tracks':
                    value = int(value)
                if key == 'num_videos':
                    value = int(value)
                if key == 'max_speed':
                    value = float(value)
                if key == 'multispectral':
                    value = bool(value)
                if key == 'multisensor':
                    value = bool(value)
                if key == 'render':
                    value = bool(value)

                # if key.startswith('rand'):
                #     pass
                if key in {'randgsize', 'randsize', 'image_sizerandom'}:
                    key = 'image_size'
                    value = 'random'

                if key == 'amazon':
                    # Hack to put in amazon background
                    key = 'render'
                    if isinstance(vidkw[key], dict):
                        value = vidkw[key]
                    else:
                        value = {}
                    value['background'] = 'amazon'  # type: ignore

                vidkw[key] = value

            vidkw.update(kwargs)
            if 'background' in vidkw:
                vidkw['render'] = {
                    'background': vidkw.pop('background'),
                }

            if isinstance(vidkw['image_size'], int):
                vidkw['image_size'] = (vidkw['image_size'], vidkw['image_size'])

            use_cache = vidkw.pop('use_cache', True)

            if 'rng' not in vidkw:
                # Make rng determined by other params by default
                vidkw['rng'] = int(ub.hash_data(sorted(vidkw.items()))[0:8], 16)

            depends_items = vidkw.copy()
            depends_items.pop('verbose', None)
            depends = ub.hash_data(sorted(depends_items.items()), base='abc')[0:14]

            if verbose > 0:
                print('vidkw = {}'.format(ub.urepr(vidkw, nl=1)))
            if verbose > 3:
                print('depends = {!r}'.format(depends))

            tag = key + '_' + depends
            dpath = vidkw.pop('dpath', None)
            fpath = vidkw.pop('fpath', None)
            bundle_dpath = vidkw.get('bundle_dpath', None)
            if fpath is not None:
                bundle_dpath = ub.Path(fpath).parent  # type: ignore
            elif dpath is not None:
                bundle_dpath = dpath

            if bundle_dpath is None:
                # Even if the cache is off, we still will need this because it
                # will write rendered data to disk. Perhaps we can make this
                # optional in the future.
                dpath = ub.Path.appdir('kwcoco', 'demo_vidshapes').ensuredir()
                bundle_dpath = vidkw['bundle_dpath'] = join(dpath, tag)

            cache_dpath = join(bundle_dpath, '_cache')  # type: ignore
            if fpath is None:
                fpath = join(bundle_dpath, 'data.kwcoco.json')  # type: ignore

            stamp = ub.CacheStamp(
                'vidshape_stamp_v{:03d}'.format(toydata_video.TOYDATA_VIDEO_VERSION),
                dpath=cache_dpath,
                depends=depends,
                enabled=use_cache,  # type: ignore
                product=[fpath],  # type: ignore
                verbose=verbose,
                meta=vidkw,
            )
            if verbose > 3:
                print('stamp = {!r}'.format(stamp))
            if stamp.expired():
                vidkw['dpath'] = bundle_dpath
                vidkw.pop('bundle_dpath', None)
                self = toydata_video.random_video_dset(**vidkw)
                if verbose > 3:
                    print('self.fpath = {!r}'.format(self.fpath))
                    print('self.bundle_dpath = {!r}'.format(self.bundle_dpath))

                self.fpath = fpath
                if fpath is not None:
                    ub.Path(fpath).parent.ensuredir()  # type: ignore
                    self.dump(fpath, newlines=True)
                    stamp.renew()
                    # Mark the state as if this file was loaded from disk This
                    # is ok, because we know the disk state exactly agrees with
                    # the current in-memory state (we just saved the dataset).
                    # TODO: might be a better idea to have a state flag that
                    # indicates if the in memory dataset is in-sync with the
                    # on-disk dataset.
                    self._state['was_loaded'] = True
            else:
                self = cls(data=fpath, bundle_dpath=bundle_dpath)  # type: ignore

        elif key == 'photos':
            dataset = demo_coco_data()
            self = cls(dataset, tag=key)  # type: ignore
        else:
            raise KeyError(key)
        return self  # type: ignore

    @classmethod
    def random(cls, rng: Any = None) -> Any:
        """
        Creates a random CocoDataset according to distribution parameters

        TODO:
            - [ ] parametarize
        """
        from kwcoco.demo.toydata_video import random_single_video_dset

        dset = random_single_video_dset(
            num_frames=5, num_tracks=3, tid_start=1, rng=rng
        )
        return dset

    @classmethod
    def empty(cls) -> CocoDataset:
        """
        Create an empty dataset
        """
        self = cls()
        return self  # type: ignore

    @classmethod
    def load(
        cls,
        file: PathLike | IO,
        bundle_dpath: Any = None,
        autobuild: Any = True,
    ) -> CocoDataset:
        """
        Constructor from a open file or file path.

        Args:
            file (PathLike | IO):
                Where to read the data. Can either be a path to a file or an
                open file pointer / stream.

        Returns:
            CocoDataset: the loaded dataset

        Example:
            >>> import kwcoco
            >>> # Create demo data to load
            >>> demo_dset = kwcoco.CocoDataset.demo('shapes1')
            >>> fpath = demo_dset.fpath
            >>> bundle_dpath = demo_dset.bundle_dpath
            >>> # Create a CocoDataset from the filepath
            >>> dset1 = kwcoco.CocoDataset.load(fpath)
            >>> # Create a CocoDataset from an open file
            >>> with open(fpath, 'r') as file:
            >>>     dset2 = kwcoco.CocoDataset.load(file, bundle_dpath=bundle_dpath)
        """
        try:
            fpath = os.fspath(file)  # type: ignore
        except TypeError:
            input_was_pathlike = False
        else:
            input_was_pathlike = True
        if input_was_pathlike:
            coco_dset = cls(fpath, bundle_dpath=bundle_dpath, autobuild=autobuild)  # type: ignore
        else:
            data = json_r.load(file)
            coco_dset = cls(data, bundle_dpath=bundle_dpath, autobuild=autobuild)  # type: ignore
        return coco_dset  # type: ignore

    @classmethod
    def from_data(
        cls, data: Any, bundle_dpath: Any = None, img_root: Any = None
    ) -> CocoDataset:
        """
        Constructor from a json dictionary

        Returns:
            CocoDataset:
        """
        coco_dset = cls(data, bundle_dpath=bundle_dpath, img_root=img_root)  # type: ignore
        return coco_dset  # type: ignore

    @classmethod
    def from_image_paths(
        cls,
        gpaths: list[str],
        bundle_dpath: Any = None,
        img_root: Any = None,
    ) -> CocoDataset:
        """
        Constructor from a list of images paths.

        This is a convenience method.

        Args:
            gpaths (List[str]): list of image paths

        Returns:
            CocoDataset:

        Example:
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.from_image_paths(['a.png', 'b.png'])
            >>> assert coco_dset.n_images == 2
        """
        coco_dset = cls(bundle_dpath=bundle_dpath, img_root=img_root)  # type: ignore
        for gpath in gpaths:
            coco_dset.add_image(gpath)  # type: ignore
        return coco_dset  # type: ignore

    @classmethod
    def from_class_image_paths(cls, root: str | PathLike) -> CocoDataset:
        """
        Ingest classification data in the common format where images of
        different categories are stored in folders with the category label.

        Args:
            root (str | PathLike):
                the path to a directory containing class-subdirectories

        Returns:
            CocoDataset:
        """
        import kwimage

        root_path = ub.Path(root)
        subdirs = [child for child in root_path.glob('*') if child.is_dir()]
        coco_dset = cls(bundle_dpath=root_path)  # type: ignore
        for subdir in subdirs:
            catname = subdir.name
            cat_id = coco_dset.ensure_category(catname)  # type: ignore
            for gpath in subdir.glob('*'):
                h, w = kwimage.load_image_shape(gpath)[0:2]
                image_id = coco_dset.add_image(gpath)  # type: ignore
                coco_dset.add_annotation(  # type: ignore
                    bbox=[0, 0, w, h], category_id=cat_id, image_id=image_id
                )
        return coco_dset  # type: ignore

    @classmethod
    def coerce_multiple(
        cls,
        datas: list,
        workers: int | str = 0,
        mode: str = 'process',
        verbose: int = 0,
        postprocess: Callable | None = None,
        ordered: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Coerce multiple CocoDataset objects in parallel.

        Args:
            datas (List): list of kwcoco coercables to load

            workers (int | str): number of worker threads / processes.
                Can also accept coerceable workers.

            mode (str):
                Parallelism type, can be: 'thread', 'process', or 'serial'.
                Defaults to 'process'.

            verbose (int): verbosity level

            postprocess (Callable | None):
                A function taking one arg (the loaded dataset) to run on the
                loaded kwcoco dataset in background workers. This can be more
                efficient when postprocessing is independent per kwcoco file.

            ordered (bool):
                if True yields datasets in the same order as given. Otherwise
                results are yielded as they become available. Defaults to True.

            **kwargs:
                arguments passed to the constructor

        Yields:
            CocoDataset

        SeeAlso:
            * load_multiple - like this function but is a strict file-path-only loader

        CommandLine:
            xdoctest -m kwcoco.coco_dataset MixinCocoConstructors.coerce_multiple

        Example:
            >>> import kwcoco
            >>> dset1 = kwcoco.CocoDataset.demo('shapes1')
            >>> dset2 = kwcoco.CocoDataset.demo('shapes2')
            >>> dset3 = kwcoco.CocoDataset.demo('vidshapes8')
            >>> dsets = [dset1, dset2, dset3]
            >>> input_fpaths = [d.fpath for d in dsets]
            >>> results = list(kwcoco.CocoDataset.coerce_multiple(input_fpaths, ordered=True))
            >>> result_fpaths = [r.fpath for r in results]
            >>> assert result_fpaths == input_fpaths
            >>> # Test unordered
            >>> results1 = list(kwcoco.CocoDataset.coerce_multiple(input_fpaths, ordered=False))
            >>> result_fpaths = [r.fpath for r in results]
            >>> assert set(result_fpaths) == set(input_fpaths)
            >>> #
            >>> # Coerce from existing datasets
            >>> results2 = list(kwcoco.CocoDataset.coerce_multiple(dsets, ordered=True, workers=0))
            >>> assert results2[0] is dsets[0]
        """
        import kwcoco
        from kwcoco.util.util_parallel import coerce_num_workers

        _loader = partial(kwcoco.CocoDataset.coerce, verbose=verbose)
        resolved_workers = coerce_num_workers(workers)
        if resolved_workers is None:
            resolved_workers = 0
        resolved_workers = min(resolved_workers, len(datas))
        # Reuse coerce_multiple logic but overload the loader function.
        yield from cls._load_multiple(
            _loader,
            datas,
            workers=resolved_workers,
            mode=mode,
            verbose=verbose,
            postprocess=postprocess,
            ordered=ordered,
            **kwargs,
        )

    @classmethod
    def load_multiple(
        cls,
        fpaths: list[str | PathLike],
        workers: int = 0,
        mode: str = 'process',
        verbose: int = 1,
        postprocess: Callable | None = None,
        ordered: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Load multiple CocoDataset objects in parallel.

        Args:
            fpaths (List[str | PathLike]):
                list of paths to multiple coco files to be loaded

            workers (int): number of worker threads / processes

            mode (str):
                Parallelism type, can be: 'thread', 'process', or 'serial'.
                Defaults to 'process'.

            verbose (int): verbosity level

            postprocess (Callable | None):
                A function taking one arg (the loaded dataset) to run on the
                loaded kwcoco dataset in background workers and returns the
                modified dataset. This can be more efficient when
                postprocessing is independent per kwcoco file.

            ordered (bool):
                if True yields datasets in the same order as given. Otherwise
                results are yielded as they become available. Defaults to True.

            **kwargs:
                arguments passed to the constructor

        Yields:
            CocoDataset

        SeeAlso:
            * coerce_multiple - like this function but accepts general
                coercible inputs.
        """
        import kwcoco

        _loader = kwcoco.CocoDataset
        # Reuse coerce_multiple logic but overload the loader function.
        yield from cls._load_multiple(
            _loader,
            fpaths,
            workers=workers,
            mode=mode,
            verbose=verbose,
            postprocess=postprocess,
            ordered=ordered,
            **kwargs,
        )

    @classmethod
    def _load_multiple(
        cls,
        _loader: Any,
        inputs: Any,
        workers: Any = 0,
        mode: Any = 'process',
        verbose: Any = 1,
        postprocess: Any = None,
        ordered: Any = True,
        **kwargs: Any,
    ) -> Any:
        """
        Shared logic for multiprocessing loaders.

        SeeAlso:
            * coerce_multiple
            * load_multiple
        """
        _submit_prog = ub.ProgIter(
            inputs, desc='submit load kwcoco jobs', enabled=workers > 0, verbose=verbose
        )
        executor = ub.Executor(mode=mode, max_workers=workers)
        with executor:
            jobs = []
            for job_idx, data in enumerate(_submit_prog):
                job = executor.submit(
                    _load_and_postprocess,
                    data=data,
                    loader=_loader,
                    postprocess=postprocess,
                    **kwargs,
                )
                job.job_idx = job_idx  # type: ignore
                jobs.append(job)

            if ordered:
                _jobiter = jobs
            else:
                from concurrent.futures import as_completed

                _jobiter = as_completed(jobs)  # type: ignore

            _collect_prog = ub.ProgIter(
                _jobiter, total=len(jobs), desc='loading kwcoco files', verbose=verbose
            )
            for job in _collect_prog:
                # Clear the reference to this job
                jobs[job.job_idx] = None  # type: ignore
                yield job.result()  # type: ignore

    @classmethod
    def from_coco_paths(
        CocoDataset: Any,
        fpaths: list[str],
        max_workers: int = 0,
        verbose: int = 1,
        mode: str = 'thread',
        union: str | bool = 'try',
    ) -> Any:
        """
        Constructor from multiple coco file paths.

        Loads multiple coco datasets and unions the result

        Note:
            if the union operation fails, the list of individually loaded files
            is returned instead.

        Args:
            fpaths (List[str]): list of paths to multiple coco files to be
                loaded and unioned.

            max_workers (int): number of worker threads / processes

            verbose (int): verbosity level

            mode (str): thread, process, or serial

            union (str | bool): If True, unions the result
                datasets after loading. If False, just returns the result list.
                If 'try', then try to perform the union, but return the result
                list if it fails. Default='try'

        Note:
            This may be deprecated. Use load_multiple or coerce_multiple and
            then manually perform the union.
        """
        results = CocoDataset.load_multiple(
            fpaths,
            workers=max_workers,
            verbose=verbose,
            mode=mode,
            ordered=False,
            autobuild=False,
        )

        results = list(results)

        if union:
            try:
                if verbose:
                    # TODO: it would be nice if we had a way to combine results
                    # on the fly, so we can work while the remaining io jobs
                    # are loading
                    print('combining results')
                coco_dset = CocoDataset.union(*results)
            except Exception as ex:
                if union == 'try':
                    warnings.warn('Failed to union coco results: {!r}'.format(ex))
                    return results
                else:
                    raise
            else:
                return coco_dset
        else:
            return results
