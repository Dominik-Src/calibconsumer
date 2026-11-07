# API

- `calibconsumer.load(path, quiet=False)`: `mapping.json` → `Camera`, `results_OpenCV.json` → `OpenCVCamera`, `results_BrownCamera.txt` → `BrownCamera`, a result folder → its `mapping.json`, else its classic file (quiet: no warnings about folds or failed re-fit bounds)
- `OpenCVCamera(camera_matrix, distortion, resolution, views=None)`, `.from_file()`, `.to_opencv()`, `.rvec_tvec(name)`, `.distort(xy)`, `pixel_to_ray`/`world_to_pixel` with `backend="numpy" | "opencv"`, `.compare_with_opencv(uv, X)`, `.valid_radius`
- `BrownCamera(parameters, pixel_size_mm, resolution, views=None)`, `.from_file()`, `.to_sensor()`, `.to_pixels()`, `.correct()`, `.valid_radius`
- `Camera.pixel_to_ray(uv, allow_extrapolation=False, return_status=False)` → `origin (..., 3)`, `direction (..., 3)`
  [, `status`]: unit directions; origin = the ray's foot point (zero for central cameras).
- `Camera.world_to_pixel(X, pose=None, allow_extrapolation=False, return_status=False, max_iterations=50, allow_folded=False)` →
  `uv (..., 2)` [, `status`]; X in the camera frame, or world coordinates with `pose=(R camera→world, centre)`.
- Uncertainties (every camera kind; `docs/accuracy.md`): `camera.uncertainty` (a `calibconsumer.Uncertainty`, from the
  file's `uncertainty` block or from an `error-statistics.json` beside the loaded file; None = unknown; assign your own,
  e.g. `Uncertainty(ray=0.05, image_sigma=0.3)` in px).
  - `pixel_to_ray(uv, …, return_sigma=True)` appends the directions' σ (rad, per axis normal to the ray);
    `camera.ray_sigma(uv)` alone. NaN where unknown.
  - `world_to_pixel(X, …, return_covariance=True, world_covariance=None, image_noise=False)` appends the pixels'
    covariances (…, 2, 2; px²): the model σ (calibration ⊕ re-fit), the world points' covariances (…, 3, 3, in X's frame)
    through the local linearization, the image noise if asked. `camera.pixel_covariance(X, uv, pose=None,
    world_covariance=None, model=True, image_noise=False)` alone. Without a calibration ray σ for the whole camera it
    warns and holds only the known parts.
  - `calibconsumer.ellipse(cov)` → σ_major, σ_minor, angle (deg, +u towards +v).
- `Camera.contains(uv)`: inside the calibrated area; `Camera.folded(uv)`: in a fold of the direction map.
- `Camera.pixel_angle(uv)`: the angle between neighbouring pixels' rays (rad/px).
- `Camera.with_frame("opencv" | "calibconsumer")`; attributes `central`, `resolution`, `refit`, `views`, `hull`, `rectangle`.
- `calibconsumer.Status`: `OK`, `OUTSIDE_HULL`, `OUTSIDE_DOMAIN`, `NO_CONVERGENCE`, `BEHIND`, `AMBIGUOUS_NEAR`, `FOLDED`.
- `calibconsumer.fit` (SciPy): `fit_spline`, `from_ray_samples(uv, directions, origins=None, *, hull, resolution, intervals,
  rectangle=None, rotation=None)`, `from_camera(camera, intervals=None)` (any camera to the spline format), `errors`,
  `injective`, `folded_share`.
- `calibconsumer.remap` – maps for `cv2.remap` (−1 where unmapped; `step` > 1: every step-th pixel solved, the rest
  interpolated, the edges of the calibrated area exact): `pinhole_maps(camera, focal_px, size, rotation=None,
  distance=None, step=1)` (OpenCV frame), `directions_maps(camera, directions, distance=None, step=1)` (any image of
  viewing directions), `sphere_maps(camera, size=(2048, 1024), pole="up" | "axis", rotation=None, distance=None, step=4)`
  (an equirectangular panorama: pole "up" – longitude across, latitude up, the horizon through the optical axis; pole
  "axis" – the polar angle from the optical axis down the rows, so each row is one field angle), `cube_maps(camera,
  face_size=1024, rotation=None, distance=None, step=4)` → {front, right, back, left, up, down} (90° pinholes),
  `sphere_directions(size, pole)`, `sphere_angles(directions, pole)`, `cube_directions(face_size, face)`. `rotation`
  takes a direction of the sphere's frame into the camera's (a rig: the transposed relative rotation of a camera).
- `calibconsumer.visual` – the mapping at a glance, no photograph needed (RGB uint8): `sensor_on_sphere(camera,
  size=(1024, 512), squares=16, pole="up", rotation=None, distance=None)` (the sensor's checkerboard on the sphere),
  `sphere_on_sensor(camera, width=1024, grid_deg=10, pole="axis", rotation=None, texture=None, checkerboard=False)` (a
  latitude-longitude grid, the sphere's checkerboard or an equirectangular texture such as an earth map on the sensor),
  `coverage_on_sphere(cameras, rotations=None, size=(1024, 512))` → (image, count) (several cameras on one sphere, a rig
  kept in its relative orientation).
- `calibconsumer.stereo.rectify(camera1, camera2, rotation, translation, size=(1280, 960), focal_px=None,
  vertical=None, distance=None, step=4)` → the rectified frames, one focal length, the maps of both cameras (any model:
  OpenCV's stereoRectify for generic cameras; the relative pose as relativeOrientations.json gives it, x₁ = R x₂ + t);
  `stereo.rectifying_rotation(rotation, translation, vertical=None)`.
- `calibconsumer.fileformat`: `read`, `write`, `validate`, `FormatError`.
- Re-evaluation (`docs/evaluate.md`): `calibconsumer.evaluate.evaluate(camera, views, world, poses_given=None,
  world_covariance=None, estimate_poses="missing", image_sigma=None, outlier_threshold=4.0)` → `Evaluation` (per
  observation arrays, `.summary()`, `.to_json(path)`, `.to_csv(path)`); `calibconsumer.inputs`: `read_points`,
  `read_points_table`, `read_points_any`, `read_world`, `read_world_sigma`, `read_poses`, `pose_from_opencv`,
  `from_project`, `from_result`; `calibconsumer.pose.estimate(camera, uv, X)` → ((R, centre), info);
  `calibconsumer.plots` (matplotlib): `residuals(evaluation, path, title=None)`, `error_map(…)`.
- `OpenCVCamera.from_dict(data)`, `OpenCVCamera.from_opencv_storage(path)` (YAML/XML, cv2), `BrownCamera.from_dict(data)`;
  `camera.pixel_angle(uv)` on every camera kind.
- The command line: `calibconsumer evaluate …`, `calibconsumer info <camera>` (`python3 -m calibconsumer …`).
