# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Version 0.2.7 - Unreleased


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
