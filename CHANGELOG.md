# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Version 0.8.3 - Unreleased

### Added
* New `load` classmethod (analogous to `dump`) as a placeholder for a potential
  future refactor.
* Ported segmentation metrics from geowatch. Eventually metrics may be ported
  to a new package: kwmetrics in order to improve modularity and keep
  components well-scoped.

### Fixed
* CIFAR grabdata now respects dpath


## Version 0.8.2 - Released 2024-06-24

### Fixed
* Handle deprecated `scipy.integrate.trapz`


## Version 0.8.1 - Released 2024-06-24

### Fixed

* CIFAR grabdata now converts train and test variants of the dataset and
  includes channel information.


## Version 0.8.0 - Released 2024-06-20

### Removed
* Drop 3.6 and 3.7 support

### Changed
* Optimize CocoObject1D get calls with SQL backends
* CocoObject1D.lookup with default now "works"-ish with SQL. Still experimental.

### Fixed

* Added workaround for an issue where a postgresql cache of a coco dataset is unable to remember its original json filepath.
* Fixed issue with `CocoSqlDatabase.disconnect` where a session would be dereferenced, but not closed.
* Worked around issue in postgresql backend where track-ids were not decoded properly in their string form.


## Version 0.7.8 - Released 2024-03-20


## Version 0.7.7 - Released 2024-03-19

### Added
* Add CLI for `find_unregistered_images`


## Version 0.7.6 - Released 2024-01-29

### Added
* Add `video_id` argument to `CocoDataset.annots`
* Can now specify render dictionary with `main_ext` and `main_channels` to `random_video_dset` to control demo data file format and the channel codes that are used.

### Fixed:

* Prevent resource crash: `find_json_unserializable` now checks for circular references.

* Coco union now handles tracks.

### Changed

* The `disjoint_tracks` in `CocoDataset.union` now works on a "track-name"
  level. Old behavior where track-ids did not need a corresponding "track"
  entry is still preserved but may be removed in the future.


## Version 0.7.5 - Released 2023-12-06

### Changed
* Added `imgs_with_annots` to extended stats.
* Background param now works as an argument to `demo` for vidshapes

### Fixed
* Added missing `name_to_video` index in sql dataset


## Version 0.7.4 - Released 2023-11-12

### Added

* Added `take` and `subgraph` to `kwcoco.CategoryTree`.

### Changed

* `kwcoco.CategoryTree` no longer modifies node order to be sorted by id. If
  this causes something to break, old behavior can be temporarily restored by
  setting `RESPECT_INPUT_ORDER  = 0`, but this option will be removed in the
  future.

* `kwcoco.CategoryTree.from_coco` and other `CategoryTree` constructors now
  respects the order in which the categories were given if id is not present.


## Version 0.7.3 - Released 2023-11-08

### Added
* Add `datetime` attribute to `CocoImage`.
* Add `ensure_video` method to `CocoDataset`.
* Add `ensure_track` method to `CocoDataset`.
* Add `sort_values` method to `ObjectList1D` (and hence vectorized `Images` and `Annots` objects).
* Add `--inplace` to `kwcoco conform`.

### Fixed
* kwcoco.Coerce with sqlview now works for .kwcoco.zip files.


## Version 0.7.2 - Released 2023-10-04

### Fixed
* `clear_annotations` now also clears tracks.
* `Images.annots` now returns `AnnotGroups` where the underlying `Annots`
  objects use id lists instead of id sets.
* Issue with new matplotlib 3.8 and `show_image`.

### Added
* `kwcoco info` can now query images by name.
* Add `io_workers` to kwcoco stats


## Version 0.7.1 - Released 2023-08-16

### Fixed:
* Fixed issue where imdelay would sometimes not respect antialias and interpolation

## Version 0.7.0 - Released 2023-08-15

### Added

* Top level track support in the dictionary dataset with `add_track`,
  `remove_tracks`, and `index.name_to_track`.

* Single-object convenience classes for the major tables.

### Changed

* Default data structure for Python versions >= 3.7 is now a regular dictionary
  instead of OrderedDict. The behavior in 3.6 is unchanged.

* Working towards a better and consistent CocoImage API without breaking
  compatibility. Added `iter_assets` and added `as_dict` kwargs to
  `primary_asset`. The goal is that these high level functions will return
  objects instead of dictionaries.


## Version 0.6.4 - Released 2023-07-19

### Fixed

* `CocoImage.add_asset` can now accept an image_id property.

### Changed

* Modified `kwcoco info` script to be able to stream the first n items in each
  table in a json compliant way.


## Version 0.6.3 - Released 2023-07-13

### Added
* New CLI script: `coco_move_assets`.
* New CLI script: `coco_info`.

### Fixed
* Issue in `coco_subset` CLI when copy-assets is on and the input is absolute paths.
* reroot can now accept pathlib.Path arguments.
* Improved reroot logic. 
* fixed issue in coco union where output paths were not rerooted correctly in some cases.

### Changed
* `kwcoco validate` now reports the path to the first missing / corrupted image in the error.
* Started work on new top-level track table.
* Started laying groundwork for moving assets to their own table.
* Moved examples out of the module and into the main repo.


## Version 0.6.2 - Released 2023-05-02

### Changed
* sliced getitems into a CocoObject1d (e.g. Images, Annotations, etc...) now
  returns a new CocoObject1d representing the subset instead of a list of
  integer ids.
* reroot now ignores leading "./" when looking for `old_prefix`
* `CocoDataset.corrupted_images` now defaults `check_aux` to True
* `CocoDataset.corrupted_images` now can execute in parallel.
* `CocoDataset.corrupted_images` now checks if it can read the entire image again.
* Changed the folder "aux" to "auxiliary" in test data, due to windows issues.
* `CocoDataset.get_image_fpath` now returns a ubelt.Path object
* `CocoDataset.get_auxiliary_fpath` now returns a ubelt.Path object
* `CocoImage.iter_image_filepaths` now yields ubelt.Path objects
* `CocoImage.primary_image_filepath` now returns a ubelt.Path object

### Added
* Add id property to CocoObject1d

### Fixed
* Windows path fixes
* safer no longer tries to use temp files on windows


## Version 0.6.1 - Released 2023-04-10

### Added:
* Add `remember_parent` option to CocoDataset.union
* Add `CocoDataset.coerce_multiple` and `load_multiple` loader with parallel processing.
* Add parallel IO to `kwcoco union`
* Add util parallel and eval

### Fixed:
* Issue in detection metrics when classes was not specified. Now uses the true
  detections to infer them if possible.
* Detection metric assign workers was never used
* Subsets of metrics now work when TN is not available.

### Changed
* CLI backend changed to use scriptconfig ModalCLI


## Version 0.6.0 - Released 2023-04-04

### Changed
* CocoImage and CocoAsset use the new AliasDictProxy interface to smoothly
  transition between name conventions.
* The "assets" key is now an alias of "auxiliary" for CocoImage and we will switch to "assets" as the primary in the future.
* Toydata now generates RGB bands independently
* Change `__unstructured__` to `_unstructured` for sqlalchemy 2.x compat 

### Added
* demo data now has the ability to render on top of a background image.

### Fixed
* Fixed sqlalchemy 2.x compatability
* The track index now works properly with SQL variants and returns ordreing


## Version 0.5.8 - Released 2023-03-17

### Changed
* Toydata now has better polygon generation for small image sizes, but is still
  not perfect.
* Changed the internal name used in compressed zip files.

### Fixed
Bug in `CocoImage.add_asset` where json serialization was broken


## Version 0.5.7 - Released 2023-03-04

### Changed
* In `metrics.drawing`, make threshold colors more consistent
* Added `category_names` property to `Annots1d` object.
* Deprecate `CocoImage.add_auxiliary_item` in favor of `CocoImage.add_asset`

### Added
* Added `annots` property to CocoImage
* Added `add_annotation` to `CocoImage`
* Added `autofix` option `kwcoco reroot`

### Fixed 
* Better asset space support
* Querying `annots` by track-id now return them in a sorted order.


## Version 0.5.6 - Released 2023-02-07

### Added
* Can now ask for the resolution of an asset in CocoImage.


### Changed
* Renamed `CocoImage.delay` to `CocoImage.imdelay`. The old name will be available as an alias, but will eventually be deprecated.
* Import time is now much faster.
* default `check_aux` to True in `CocoDataset.missing_images`


## Version 0.5.4 - Released 2023-02-02 

### Added
* New `compress=True` argument to `dump` which will compress the kwcoco file before writing it.
* Add CLI argument `--num_write` to `kwcoco split` to write multiple splits.


### Changed
* The kwcoco CLI is now strict (i.e. will error on unsupported arguments) by default. Set the `KWCOCO_LOOSE_CLI` environment variable to use the old behavior. 
* Added argcomplete support
* The `.fpath` attribute is no longer cast by default and is now allowed to be
  a `pathlib.Path` instance. In the future we may always convert this attribute
  to a pathlib object.


## Version 0.5.4 - Released 2023-01-23

### Added
* Experimental support for reading / writing kwcoco files to / from zipfiles. 

### Changed
* Improved splitting in kwcoco split CLI. Better handling for video datasets. Category balance is now optional.

### Fixed
* Bug where specifying resolution to `delay` would not work if the channels did not exist.


## Version 0.5.3 - Released 2023-01-20

### Added
* Add `role` property for annotations in the schema.

### Fixed
* Added `instance_id` for mmlab tracking
* Fixed np.bool
* Fixed np.float


## Version 0.5.2 - Released 2022-12-15

### Added

* Initial support to conform to mmlab coco specs
* Added ability to specify resolution in `CocoImage.delay`
* Add resolution to the image / video schema.


### Fixed
* kwcoco validate now reports errors for all inputs


### Changed

* Modified signatures of `annots` and `images` to use `annot_id` and `image_id`
  instead of `aid` and `gid`.

* Removed mentions of "flicks" in timestamps. We are simply using ISO strings
  and unix times for now.

* Updated timestamp details in the SPEC.

* Updated descriptions and organization in the formal schema.

* Require that names do not contain the "/" character in the formal schema.


## Version 0.5.1 - Released 2022-10-24


### Added
* Initial support for PostgreSQL
* In CocoDataset.coerce the arg `sqlview` can now be given as a string `sqlite` or `postgresql`.


### Fixed
* Lookup on SQL database now handles json columns


## Version 0.5.0 - Released 2022-09-28

### Fixed
* Taking non-existing channels from a single-image delayed node returns a
  concatenated node with existing and nan channels.

### Changed
* Taking subchannels from a concatenated delayed node now groups discontinuous
  indices from the same underlying image.
* `channel_spec`, `sensorchan_spec`, `util.lazy_frame_backends`, and
  `util.delayed_ops` have all been ported to the new `delayed_image` package.

### Removed
* Removed old `util_delayed_ops`


## Version 0.4.2 - Released 2022-09-06

### Fixed
* Fixed bug in vidshapes toydata code
* Respect masked arrays when doing delayed warp


## Version 0.4.1 - Released 2022-08-24

### Changed 
* Handle change in kwimage demo data, doctest hashes changed
* The SQLView changed the "extra" key to "__unstructured__" and now implicitly
  removes it before being passed to the user.
* The multisensor data now contains a "sensor" key in the image dictionary.
* Can now pass `kwcoco.CocoDataset.coerce` `sqlview=True` to attempt to grab an SQL view of the dataset.

### Fixed
* bug in coco subset when absolute=False


## Version 0.4.0 - Released 2022-08-11

### Changed 
* Officially deprecate `kwcoco.util.util_delayed_poc`
* Remove the 2 suffix on the official `delayed_ops` node names.
* The delayed crop method can now simulated a padded slice.
* Changed ujson integration from opt-out to opt-in. Specify `KWCOCO_USE_UJSON` to use ujson.


## Version 0.3.5 - Released 2022-08-08

### Changed
* kwcoco will now use ujson to read json files by default if it is available.
* Added runtime deprecation noticed to deprecated functions
* Doctests now generate images for readthedocs.


## Version 0.3.3 - Released 2022-07-26

### Added
* Ability to "undo warps" from delayed ops.

### Changed
* Removed Jagged from `delayed_ops`
* NaNs are now optimized like other leafs.

### Fixed
* Bug where delayed nan did not have the right warp signature.
* Bug in boids when speed was 0.


## Version 0.3.2 - Released 2022-07-22

### Added
* `with_bundle` flag to `iter_image_filepaths`

### Fixed
* Bug in sensorchan spec
* Issue in reroot and toydata generation that used incorrect bundle-dpaths
* Fixed issue in key parsing where numbers greater than 10 failed.

### Changed
* Added `fpath` as an option to `CocoDataset.demo`, which is currently semi-stable.
* The `finalize` method for a DelayedOperation2 now calls prepare and optimize
  beforehand by default, and also always returns an ndarray.
* DelayedWarp2 will now default to nan fill values when warping floats


## Version 0.3.1 - Released 2022-07-18


## Version 0.3.0 - Released 2022-07-14

### Added
* Initial non-experimental version of SensorChanSpec with limited functionality
* Updated `kwcoco validate` cli tool with full options (and added checks for img attrs)
* New Delayed operations with a more scalable design 

### Fixed
* Fix compatibility issue with datasets that do not contain an annotations list
* Issue where the original coco polygons did not validate correctly in the schema.
* The new delayed operations now always return an arraylike

### Changed
* FusedChannelSpec now has an `__eq__` method that returns True if the parsed reprs are the same.


## Version 0.2.31 - Released 2022-05-06

### Changed
* kwcoco subset can now accept dst as the second positional argument

### Fixed
* Fix issue in kwcoco subset where paths were invalid if written outside of
  the source kwcoco bundle.

## Version 0.2.30 - Released 2022-04-19

### Added
* kwcoco validate now accepts `--require_relative` which causes non-portable
  kwcoco files with non-relative paths to fail.
* kwcoco validate now checks for duplicate bands in the auxiliary dictionary.


## Version 0.2.29 - Released 2022-04-08

### Changed
* Better verbosity for demodata
* Experimental support for ujson
* Remove deprecated usage of cfgstr with ubelt.Cacher
* kwcoco stats output is modified

### Fixed
* CocoImage Affine transform `matrix` attribute is now always populated.


## Version 0.2.28 - Released 2022-04-02

### Changed
* CocoDataset.dump now uses `safer` with `temp_file=True` by default.
* Initial steps to rename "auxiliary" to "assets" in a backwards compatible way.


## Version 0.2.27 - Released 2022-03-31

### Fixed
* Bug in valid region warp


## Version 0.2.26 - Released 2022-03-29

### Added

* Method to get the valid region for a CocoImage in image or video space


### Changed

* CocoSqlDatabase is now lazy exposed at the top level in Python 3.7+
* Better SQL support in CocoImage
* Add maxlen in `path_sanitize`
* Warp properties in CocoImage
* Stub out CocoAsset


## Version 0.2.25 - Released 2022-03-16

### Added
* Can now load do delayed loading of data in auxiliary space. (throws an error
  if multiple auxiliary streams are mixed)
* issubset issuperset for FusedChannelSpec
* union for FusedChannelSpec and ChannelSpec
* `find_asset_obj` for CocoImage
* Added quantization spec. CocoImage.delay will now return a dequantized image
  if quantization metadata is available.


## Version 0.2.24 - Released 2022-03-15

### Added
* verbose flag to perterb function
* Added new exceptions `DuplicateAddError` and `InvalidAddError` to better
  indicate why an add image/category/annotation failed.

### Changed
* Combine measures can now take an explicit set of threshold bins to accumulate to.


## Version 0.2.23 - Released 2022-03-07

### Fixed
* Bug in delayed with nodata

## Version 0.2.22 - Released 2022-03-05

### Added

* `path_sanatize` to `kwcoco.ChannelSpec` to make a path-safe suffix for a filename
* Add experimental `_dataset_id` function to `kwcoco.CocoDataset` for unique
  but human interpretable dataset identifiers.

### Changed

* Conform now adds width / height to auxiliary objects
* Enhanced the `CocoImage.primary_asset` function
* gdal reader now uses an auto nodata mode, somewhat experimental.



## Version 0.2.21 - Released 2022-02-15

### Added
* Caching mechanism for model hashids

### Changed
* Minor improvements to confusion measures.
* AP / ROC curves now report real/support using concise SI for large numbers


## Version 0.2.20 - Released 2022-01-18

### Changed

* Removed old Python2 constructs
* Modified multisensor to use concise channel codes
* Moved `LazyGDalFrameFile` to `kwcoco.util.lazy_frame_backends` to experiment with different image subregion loaders


## Version 0.2.19 - Released 2022-01-04

### Added

* names kwarg to CocoDataset.images / videos

* `DelayedImage.finalize` now accepts a `nodata` argument, which handles invalid
  data pixels correctly under transformations.

* Toydata can now generate "multi-sensor" demodata.


### Fixed
* CocoImage now returns None for `video` if it doesn't have one.
 
* BOIDS is now deterministic given a seed, which fixes toydata determinism

* Fixed toydata bug where data previously only drawn on first channel


### Changed

* Tweaked default toydata settings


## Version 0.2.18 - Released 2021-12-01


### Added
* Added `add_auxiliary_item` to `CocoImage`
* `CocoImage.delay` can now handle the case where `imdata` is given directly in the image dictionary.
* `Measures.combine` now has a `growth` and `force_bins` parameter that works somewhat better
  than the previous `precision` parameter.
* Add measure combiner helper classes
* Support for detaching `CocoImage` from the `dset` object (for multiprocessing)


### Changed
* Split part of `confusion_vectors.py` into new file `confusion_measures.py`


## Version 0.2.17 - Released 2021-11-10

### Added
* Added "channels" argument to kwcoco subset

### Fixed
* Bug in `delayed_load` when none of the requested channels exist


## Version 0.2.16 - Released 2021-11-05


### Changed
* Added more docs to demodata
* Moved eval CLI code to the cli module.


### Fixed
* Annotation outside the image bounds now render correctly in the toydata module.
* Fixed issue in FusedChannelSpec where a unnormalized getitem did not work correctly


## Version 0.2.15 - Released 2021-10-29

### Added
* Ported channel related features to CocoImage
* vidshapes-msi is now an alias for vidshapes-multispectral
* Can now specify general channels for video toydata

### Changed
* Split `kwcoco.demo.toydata` into `kwcoco.demo.toydata_image` and `kwcoco.demo.toydata_video`
* Moved code from `coco_dataset` to `_helpers`

### Fixed
* Bug in `show_image` with MSI
* Fixed bug in ObjectList1D.peek


## Version 0.2.14 - Released 2021-10-21


### Added
* Moved Archive code to its own util module
* Extended channel slice syntax. Can now use "." to separate root from the slice notation.

### Fixed
* Removed debugging print statements
* Fixed issue with pickling category tree
* Fixed bugs with numel and zero channels


## Version 0.2.13 - Released 2021-10-13

### Added
* Add `.images` to `Videos` 1D object.
* Initial `copy_assets` behavior for kwcoco subset.

### Changed
* Improved speed of repeated calls to FusedChannelSpec.coerce and normalize

### Fixed
* Fixed bug in delayed image where nans did not correctly change size when warped
* Fixed bug in delayed image where warps were not applied correctly to concatenated objects


## Version 0.2.12 - Released 2021-09-22


### Added
* Initial implementation of shorthand channels
* Parameterized `max_speed` of toydata objects
* Add `combine_kwcoco_measures` function
* Add new API methods to ChannelSpec objects


### Changed

* Reworking kwcoco channel spec backend implementation. Trying to maintain
  compatibility and insert warnings where things might change.
* Removed six from CocoDataset
* Added support for pathlib


### Fixed
* Fixed bug in union with relative paths


## Version 0.2.11 - Released 2021-08-25


### Added

* `ChannelSpec` can now be coerced from a `FusedChannelSpec`
* `FusedChannelSpec` now implements the `__set__` interface

## Version 0.2.10 - Released 2021-08-24


### Fixed
* Fixed bug when `track_id` is given to `add_annotation`
* Fixed but in `delayed_load` where requested channels were returned in the
  wrong order, or with incorrect data.
* Bug in `delayed_load` where nans did not resize properly

### Added
* Can now specify `-frames{num}` in demo names to control number of frames in each video


### Changed

* In detection metrics, annotations now get a default score of 1.0 if not provided.
* In detection metrics, fixed AUC to report 1.0 when detections are perfect


## Version 0.2.9 - Released 2021-08-12


### Added

* Added `vidid` argument to `CocoDataset.images` 1D object mixin.
* Added `trackid` argument to `CocoDataset.annots` 1D object mixin.

### Fixed

* `perterb_coco` now correctly perturbs segmentations

### Changed

* Changed argument name from `gsize` to `image_size` in toydata methods.
  Backwards compatibility is preserved via kwargs for now, but `gsize` will be
  deprecated in the future.


## Version 0.2.8 - Released 2021-08-02


### Added

* CocoSQLDataset.coerce
* `kwcoco subset` CLI now has a `--select_images` and `--select_videos` option that takes a json query.
* Add initial implementation of `DelayedChannelStack.take_channels`, returns nans if the channel is not available.

### Changed

* Made URIs for CocoSQLDataset slightly more robust
* kwcoco show now defaults `show_labels` to False


## Version 0.2.7 - Released 2021-06-28

### Added

* Added "domain net" to `kwcoco grab`.
* Added "spacenet7" to `kwcoco grab`.


### Fixed
* Fixed issue in `add_annotation` where if failed to add keypoints, and segmentations.


## Version 0.2.6 - Released 2021-06-22

### Added

* `CocoDataset.index.trackid_to_aids` is a new index that maps a track-id to
  all annotations assigned to that track.

* `kwcoco grab` CLI for downloading and converting standard datasets

* `CocoDataset.delayed_load` now has a limited ability to specify a list of
  channels to load, and if it is returned in image or video space.

### Changed
* `kwcoco.add_annotation` can now accept and automatically convert kwimage data
  structures for segmentation and keypoints.

### Fixed
* Channels in a base image can now be None. 
* `validate` now prints which field caused the issue

## Version 0.2.5 - Released 2021-06-09

### Changes
* Updated README
* Code cleanup
* More docs

### Fixes
* Fixed error with empty union
* Fixed error in `delayed_load`

## Version 0.2.4 - Released 2021-05-20

### Added

* Experimental `delayed_load` method for an image

### Fixed
* Fixed `cats_per_img` in kwcoco stats.
* Fixed `grab_camvid`.
* Fixed legacy `img_root` key not being respected.


## Version 0.2.3 - Released 2021-05-13

### Changed
* Made bezier dependency optional
* stats now returns video stats


## Version 0.2.2 - Released 2021-05-10


### Added

* Added `channel_spec.py` as a helper and documentation
* Auxiliary images now have `warp_aux_to_img` property
* Base images now have `warp_img_to_vid` property

### Changed

* Removed `base_to_aux` transform in favor of inverse `warp_aux_to_img`


### Fixes
* Fixed error in kwcoco conform documentation 
* The conform operation now warns if the segmentation cannot be converted to a polygon


## Version 0.2.1 - Released 2021-04-26


### Changed

* The `dset.index.vidid_to_gids` is now guaranteed to always return image ids ordered by their frame index.
* `dset.view_sql` now has a `memory` kwarg that will force the database to live in memory.
* The SQLite backend now agrees with the dict backend and does not map None to values for image names and file names.

### Removed
* `kwcoco.utils.util_slice` use `kwarray.util_slice` instead.


## Version 0.2.0 - Released 2021-04-22


### Fixed
* Fixed numpy warning by using `int` instead of `np.int`.
* Issue in rename categories with non-deterministic behavior
* Reference gdal from the `osgeo` package
* Fixed "name" attribute in sqlview.
* Implemented the `name_to_video` index
* annToMask and annToRLE in the `compat_dataset` now work like the original.

### Removed
* Several deprecated APIs
* `kwcoco.toydata` and `kwcoco.toypatterns`, use `kwcoco.demo.toydata` and `kwcoco.demo.toypatterns` instead.
* `remove_all_images` and `remove_all_annotations` (use clear variants instead)


### Changed
* `kwcoco coco_subset` can now take a list of image-ids explicitly.
* Separate the vectorized ORM-like objects out of `coco_dataset` and into their own `coco_objects` module.


## Version 0.1.13 - Released 2021-04-01

### Changed
* The image schema now has two modalities. The normal `id` + `file_name`  still
  exists, but now we can do `id` + `name` + `auxiliary` without a default
  `file_name`, which allows us to better handle multispectral images.

* The `name` is a new primary text key for images. 

* In the SQL view of the database, non-schema properties are now stored in
  "extra" instead of "foreign". (I confused "foreign keys" for "additionalProperties")

### Added
* Add `name` property to the `image`.
* Add `kwcoco.Subset` create a subset of a coco file based on filter criteria
* Add `kwcoco.Subset` create a subset of a coco file based on filter criteria
* Experimental interactive mode to kwcoco show
* Start to merge cli and class functionality.

### Fixed
* Fixed the default bundle root to be the cwd instead of
  `untitled_kwcoco_bundle`.

* `CocoDataset.union` will now correctly insert prefixes into file names such
  that they are relative to the common bundle directory.

* Bug in `rename_categories`

## Version 0.1.12 - Released 2021-03-04


## Version 0.1.11 - Released 2021-02-24

### Added
* Introduced the concept of a bundled dataset
* Added `kwcoco conform` script to standardize / make coco files compliant with `pycocotools`. 
* Added `--image_attrs=True` to `kwcoco stats`.
* Added `AbstractCocoDataset` base class for `CocoDataset` and `CocoSqlDataset`.
* Added `examples` subdirectory for answers to FAQ
* Added `kwcoco validate` script to check that json and assets exist.
* Added `conform` method to the main coco dataset

### Changed
* `CocoDataset.subset` will now only return the videos that are supported by
  the chosen images.

## Version 0.1.10 - Released 2021-02-05

### Added
* Add `ascii_only` keyword to clf-report to disable unicode glyphs
* Add `ASCII_ONLY` environment variable to disable unicode glyphs
* Initial implementation for `CocoSqlDatabase`

### Fixed
* Fix bug in `show_image` when segmentation is None


## Version 0.1.9 - Released 2021-01-08

### Added
* Add `remove_video` method to CocoDataset
* kwcoco show can now show auxiliary data


### Changed
* `CocoDataset.union` now will remap track-ids assuming all input datasets are
  disjoint by default.

* Fixed issues in metrics classification report


## Version 0.1.8 - Released 2020-12-02

### Added
* Standalone algorithm for BOID positions

### Changed
* Make kwplot, matplotlib, and seaborn optional

### Fixed
* Fixed bug in `draw_threshold_curves` when measures are empty.


## Version 0.1.7 - Released 2020-10-23

### Changed
* Better support for pycocotools 
* Better evaluation code
* Better auxiliary data
* Tweaks in CLI stats script

## Version 0.1.6 - Released 2020-10-23

### Added
* `kwcoco.kw18.KW18.to_coco` can now accept `image_paths` and `video_name`

### Changed
* Yet more change to `reroot`, this needs to get reviewed and fixed.
* Fix spelling of auxillary to auxiliary
* Better auxiliary channel support in toydata

### Fixed
* Added version flag to the CLI
* Fixes to `ensure_json_serializable` for edge cases and recursion depth


### Removed
* Removed `pred_raw` from ConfusionVectors


## Version 0.1.5 - Released 2020-08-26

### Added

* Ported the non-torch components of CategoryTree from ndsampler to kwcoco. 
* This helps kwcoco work better as a standalone package.
* Union now works with videos
* New formal schema with jsonschema validation support
* Add `autobuild` kwarg to `CocoDset.subset`


### Changed
* Removed optional dependency on ndsampler, functionality should now be available without ndsampler.


### Fixed
* the `_check_index` method now works correctly.
* `add_image` with `video_id` now works even if the video doesn't exist yet.
* `add_annotation` no longer fails if `category_id` is None.


## Version 0.1.4 - Released 2020-07-20

### Added
* Added `compat_dataset.COCO` which supports 99% of the original `pycocotools` API.
* Added `kwcoco modify_categories` CLI which supports renaming and removal of categories.
* New `random_video_dset` demo function to generate toydata in video sequences. Added new tooling for this. STATUS: INCOMPLETE
* Add new video-based APIs to the CocoDataset based on the spec. STATUS: incomplete

### Fixed
* Fix bug in find_representative_images.


## Version 0.1.3 - Released

### Added
* Basic KW18 read/write/from-COCO support

### Fixed
* Fixed json serialization error when using perterb_coco.

## Version 0.1.2 - Released 2020-06-11

### Added 
* ported metrics from netharn, might be moved in the future.
* Add `perterb_coco`
* Add `CocoEvaluator` and CLI `kwcoco eval`

## Version 0.1.1 - Released 2020-10-23

### Added
* stats CLI can now accept multiple datasets.
* ported `data.grab_voc` and `data.grab_camvid` from netharn.
* Add `kwcoco.coco_evaluator.CocoEvaluator`
* Add `safe` kwarg to `CocoDataset.reroot`

### Fixed
* Python2 error with json dump

## Version 0.1.0 - Released 2020-04-08

### Added
* Example usage section in CLI help
* Moved toydata from ndsampler to kwcoco

### Fixed
* Multipolygon segmentations are now displayed correctly.
* Tracebacks on errors are now displayed.

## [Version 0.0.1] - 

### Added
* Initial version
