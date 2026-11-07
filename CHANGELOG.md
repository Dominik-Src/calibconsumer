# Changelog

## 0.2.0 – 2026-11-07

- Images of a camera's viewing directions (`calibconsumer.remap`): `directions_maps` for any image of directions,
  `sphere_maps` (an equirectangular panorama, the pole up or on the optical axis), `cube_maps` (six 90° pinhole faces),
  their directions and angles; every map from a coarse grid (`step`, the edges of the calibrated area solved exactly) –
  `pinhole_maps` gained `step`.
- `calibconsumer.visual`: the sensor's checkerboard on the sphere, the sphere's latitude-longitude grid, checkerboard or
  a texture (an earth map) on the sensor, several cameras' coverage on one sphere – a rig kept in its relative orientation.
- `calibconsumer.stereo.rectify`: the stereo rectification of a pair of cameras of any model from their relative pose.
- Examples `mapping_images.py` and `rectify_stereo_pair.py`.

## 0.1.1 – 2026-11-06

- The repository's address in the package metadata (`project.urls`), `CITATION.cff` (`repository-code`) and the
  installation notes; `pip install` straight from a release tag; INSTALL.md's clone example went into `src` instead of
  the clone.

## 0.1.0 – 2026-11-06

- First version: `mapping.json` format version 1 (uniform cubic tensor B-spline of the ray direction in
  azimuthal-equidistant coordinates, the foot point of non-central rays); `Camera.pixel_to_ray`,
  `Camera.world_to_pixel` with a bilinear start in the ray grid's cells and Gauss–Newton, the extrapolation switch, the OpenCV frame; the
  re-fit of ray samples (`calibconsumer.fit`, SciPy); `calibconsumer.remap.pinhole_maps` for `cv2.remap`; folds of the direction map flagged (status `FOLDED`,
  `inverse.foldedShare`); the classic models from their own files: `OpenCVCamera` (14 coefficients, an
  optional cv2 backend and comparison), `BrownCamera` (photogrammetric Brown parameters); `calibconsumer.load` of a result file or
  folder; `calibconsumer.fit.from_camera`. Uncertainties: the optional `uncertainty` block (the calibration's ray σ over the
  sensor, the re-fit error, the image noise), `pixel_to_ray(return_sigma=True)`, `world_to_pixel(return_covariance=True,
  world_covariance=…)`, `calibconsumer.Uncertainty`, `calibconsumer.ellipse`. The re-evaluation `calibconsumer evaluate` (command line, `calibconsumer.evaluate`):
  reprojection, angular, normalized and object-space errors, expected σ in the image and in the world, outliers by
  Mahalanobis distance, pose estimation for views without one, plots (`calibconsumer.plots`, extra `plot`); readers for
  a calibration's project and result and for one's own points, world, σ and poses (`calibconsumer.inputs`); one's own
  OpenCV parameters as JSON or OpenCV YAML/XML and Brown parameters as JSON; `pixel_angle` on every camera kind.
