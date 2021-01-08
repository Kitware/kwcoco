# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Version 0.1.9 - Unreleased

### Added
* Add `remove_video` method to CocoDataset


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
