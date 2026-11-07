# calibconsumer

Rays and pixels of a calibrated camera – for central cameras (pinhole, fisheye, any lens) and non-central ones
(cameras behind glass, catadioptric systems). A calibration result is shipped as one file, `mapping.json`; this
package reads it and maps

- a **pixel to its ray** (closed form), and
- a **3D point to its pixel** (a start from a grid of rays, then Gauss–Newton).

```python
import numpy as np
import calibconsumer

camera = calibconsumer.load("mapping.json")
origin, direction = camera.pixel_to_ray(np.array([[812.5, 400.0]]))      # camera frame, unit direction
uv, status = camera.world_to_pixel(np.array([[0.1, 0.2, -1.5]]), return_status=True)
uv = camera.world_to_pixel(points_in_world, pose=camera.views["image-0003"])  # world coordinates and a view's pose
opencv_camera = camera.with_frame("opencv")                                    # x right, y down, z forward
```

**Conventions.** Pixel (0, 0) is the centre of the top-left pixel, u to the right, v down. The camera frame of the
file is x right, y up, viewing along −z (`with_frame("opencv")` gives x right, y down, viewing along +z). Lengths are in
the units of the calibration's world points.

**Accuracy.** The file's `refit` block states how closely the mapping reproduces the calibrated model (in pixels, as
an angle, and in object space). That is not the calibration's accuracy, which the calibration report states.

**Folds.** Where a calibrated lens model folds over (two pixels see the same direction – it happens in small spots at
the edge of the calibrated area), `world_to_pixel` returns NaN and the status `FOLDED` unless `allow_folded=True`; the
file's `inverse.foldedShare` says how much of the area is affected, and loading such a file warns.

**The calibrated area.** Pixels outside the convex hull of the calibration's points, and 3D points that map there,
give NaN and the status `OUTSIDE_HULL` unless `allow_extrapolation=True` is passed: a model is not known outside the
area its data covered.

**Classic models.** Results of the Brown or OpenCV model keep their own parameter files, and the package evaluates them
directly from the parameters – `calibconsumer.load("results_OpenCV.json")` gives an `OpenCVCamera` (all 14 coefficients;
`to_opencv()` and `rvec_tvec(view)` hand the parameters to cv2, `backend="opencv"` lets cv2 compute and
`compare_with_opencv()` compares), `calibconsumer.load("results_BrownCamera.txt")` a `BrownCamera` (the photogrammetric
Brown parameters), both with the same methods as `Camera`. `calibconsumer.fit.from_camera(camera)` turns any of them into
the spline format of `mapping.json` if one format for every camera is wanted. `calibconsumer.load(folder)` takes a camera's
result folder: its `mapping.json`, else its classic file.

**Uncertainties.** A `mapping.json` with an `uncertainty` block (or a result folder with its `error-statistics.json`)
gives every ray and every predicted pixel a σ: `pixel_to_ray(uv, return_sigma=True)` appends each ray's σ (rad), and
`world_to_pixel(X, return_covariance=True, world_covariance=Σ)` appends each pixel's covariance. The covariance
combines the calibration's ray σ over the sensor, the mapping's re-fit error, the world points' own covariances and,
on request, the image noise (`docs/accuracy.md`).

**Re-evaluation.** `calibconsumer evaluate` measures a camera against measured image points and world points:
reprojection, angular and object-space errors per point, their expected σ in the image and in the world, outliers,
and plots of the residuals and of the error over the sensor. It takes a calibration's result folder or your own files:
OpenCV or Brown parameters, point tables, world points with σ, poses (estimated when missing). See `docs/evaluate.md`.

```
calibconsumer evaluate resultsCalibration/<run>/cameraLensInstance-1 --out evaluation
calibconsumer evaluate --camera camera.yml --points points.csv --world target.txt --out evaluation
```

**Installation** (`INSTALL.md`): from the source folder with `PYTHONPATH=src`, or `pip install .` (into a virtual
environment), or `pip install "calibconsumer @ git+https://github.com/Dominik-Src/calibconsumer.git@v0.1.1"`, or a wheel from `python3 -m build`. Requires Python ≥ 3.10 and NumPy. Optional extras: `[fit]` adds
SciPy for `calibconsumer.fit` (re-fit your own ray samples to the format), `[plot]` matplotlib for the plots, and
`[opencv]` OpenCV for its backend and its YAML/XML files. `calibconsumer.remap` builds `cv2.remap` maps – a virtual pinhole, the sphere, a cube map; `calibconsumer.visual` draws a camera's mapping without a photograph; `calibconsumer.stereo` rectifies a pair.

Documentation: `docs/format.md` (the file format, normative), `docs/conventions.md`, `docs/accuracy.md`,
`docs/api.md`, `docs/evaluate.md`. Examples: `examples/`. Licence: Apache-2.0 (`LICENSE`, `NOTICE`).
