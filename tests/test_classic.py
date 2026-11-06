# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""M13: the OpenCV model (against cv2 where installed); M14: the Brown model; M15: loading result files, the opt-in
re-fit of a classic camera to the spline format."""

import json

import numpy as np
import pytest

import calibconsumer
from calibconsumer import BrownCamera, OpenCVCamera, Status

K = [[1700.0, 0.0, 2010.5], [0.0, 1705.0, 1490.25], [0.0, 0.0, 1.0]]
COEFFICIENTS = {
    4: [-0.17, 0.02, -5e-4, 8e-4],
    5: [-0.17, 0.02, -5e-4, 8e-4, -1e-3],
    8: [0.3, -0.05, -5e-4, 8e-4, 1e-3, 0.45, -0.04, 0.002],
    12: [0.3, -0.05, -5e-4, 8e-4, 1e-3, 0.45, -0.04, 0.002, 1e-3, -2e-4, -8e-4, 1e-4],
    14: [0.3, -0.05, -5e-4, 8e-4, 1e-3, 0.45, -0.04, 0.002, 1e-3, -2e-4, -8e-4, 1e-4, 0.01, -0.02],
}


def camera_points(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.9, 0.9, (n, 2))
    z = rng.uniform(0.5, 20.0, n)
    return np.c_[xy * z[:, None], z]  # OpenCV frame: in front, within ≈ ±42°


@pytest.mark.parametrize("count", sorted(COEFFICIENTS))
def test_m13_opencv_model(count):
    cam = OpenCVCamera(K, COEFFICIENTS[count], (4000, 3000), frame="opencv")
    X = camera_points()
    uv, status = cam.world_to_pixel(X, return_status=True, allow_extrapolation=True)
    assert np.all((status == Status.OK) | (status == Status.OUTSIDE_HULL))
    o, d, st = cam.pixel_to_ray(uv, return_status=True, allow_extrapolation=True)  # round trip
    assert np.all(st != Status.NO_CONVERGENCE)
    assert np.allclose(cam.world_to_pixel(d * 5.0, allow_extrapolation=True), uv, atol=1e-8)
    cv2 = pytest.importorskip("cv2")
    reference, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), np.array(K), np.array(COEFFICIENTS[count]))
    assert np.max(np.abs(reference.reshape(-1, 2) - uv)) < 1e-8
    assert cam.compare_with_opencv(uv=uv[:200], X=X[:200])["projectPx"] < 1e-8
    _, d_cv = cam.pixel_to_ray(uv[:200], allow_extrapolation=True, backend="opencv")
    assert np.max(np.arccos(np.clip(np.sum(d_cv * d[:200], -1), -1, 1))) < 1e-6


def test_m13_opencv_file_poses_frames_and_limits(tmp_path):
    rng = np.random.default_rng(3)
    rvec, tvec = rng.normal(scale=0.3, size=3), np.array([0.1, -0.2, 6.0])
    path = tmp_path / "results_OpenCV.json"
    path.write_text(json.dumps({"cameraMatrix": K, "distortionCoefficients": COEFFICIENTS[5], "imageSize": [4000, 3000],
                                "model": "Model OpenCV", "rvecs": {"image-0003.csv": rvec.tolist()},
                                "tvecs": {"image-0003.csv": tvec.tolist()}}))
    cam = calibconsumer.load(path)
    assert isinstance(cam, OpenCVCamera) and set(cam.views) == {"image-0003"}
    world = rng.uniform(-1, 1, (300, 3))
    uv = cam.world_to_pixel(world, pose=cam.views["image-0003"], allow_extrapolation=True)
    r, t = cam.rvec_tvec("image-0003")
    assert np.allclose(r, rvec) and np.allclose(t, tvec)
    cv2 = pytest.importorskip("cv2")
    reference, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec, tvec, np.array(K), np.array(COEFFICIENTS[5]))
    assert np.max(np.abs(reference.reshape(-1, 2) - uv)) < 1e-8
    _, d = cam.pixel_to_ray(np.array([[3500.0, 1500.0]]))
    assert d[0, 0] > 0 and d[0, 2] < 0  # the package's frame: right of the centre is +x, it looks along −z
    assert cam.with_frame("opencv").pixel_to_ray(np.array([[3500.0, 1500.0]]))[1][0, 2] > 0
    uv, status = cam.world_to_pixel(np.array([[0.0, 0.0, 5.0]]), return_status=True)  # behind (package frame)
    assert np.isnan(uv).all() and status[0] == Status.BEHIND
    folding = OpenCVCamera(K, [-0.6, 0.05, 0, 0], (4000, 3000))  # strong barrel: folds at r ≈ 0.83
    assert 0.5 < folding.valid_radius < 1.5
    far = np.array([[1.4, 0.0, 1.0]]) @ np.diag([1.0, -1.0, -1.0])  # OpenCV (1.4, 0, 1) in the package's frame
    assert folding.world_to_pixel(far, return_status=True)[1][0] == Status.OUTSIDE_DOMAIN
    assert np.isfinite(folding.world_to_pixel(far, allow_extrapolation=True)).all()
    with pytest.raises(ValueError, match="distortion coefficients"):
        OpenCVCamera(K, [0.1, 0.2, 0.3], (4000, 3000))


BROWN = """                           Brown Bundle Adjustment Results: Camera Parameters

Results for Camera 1    synthetic

Sensor Size        Pixel Size (mm)
  H      4000           0.001550
  V      3000           0.001550

  Camera    Initial      Total          Final        Initial         Final
 Variable    Value     Adjustment       Value       Std. Error     Std. Error
    C    2.718001064    0.000000000    2.718001064    0.000000000    0.000000000 (mm)
   XP    0.010817358    0.000000000    0.010817358    0.000000000    0.000000000 (mm)
   YP   -0.004519497    0.000000000   -0.004519497    0.000000000    0.000000000 (mm)
   K1  4.5301430e-02  0.0000000e+00  4.5301430e-02  0.0000000e+00  0.0000000e+00
   K2 -5.1200561e-04  0.0000000e+00 -5.1200561e-04  0.0000000e+00  0.0000000e+00
   K3  2.4660803e-04  0.0000000e+00  2.4660803e-04  0.0000000e+00  0.0000000e+00
   P1 -1.6556074e-04  0.0000000e+00 -1.6556074e-04  0.0000000e+00  0.0000000e+00
   P2 -1.4629027e-04  0.0000000e+00 -1.4629027e-04  0.0000000e+00  0.0000000e+00
   B1 -1.2265619e-04  0.0000000e+00 -1.2265619e-04  0.0000000e+00  0.0000000e+00
   B2  1.3961135e-03  0.0000000e+00  1.3961135e-03  0.0000000e+00  0.0000000e+00

Station       Image         Xc          Yc          Zc       Alpha       Elev.        Roll
   1       image-0000    36.63428   183.35466   124.65873  -75.823443  -62.089313 -146.875853
"""


def brown_reference(p, xy):
    """Δ of the Brown correction written out for the test: measured (mm) → ideal."""
    x, y = xy[:, 0] - p["XP"], xy[:, 1] - p["YP"]
    r2 = x * x + y * y
    d = p["K1"] * r2 + p["K2"] * r2 ** 2 + p["K3"] * r2 ** 3
    return np.c_[x * (1 + d) + p["P1"] * (r2 + 2 * x * x) + 2 * p["P2"] * x * y + p["B1"] * x + p["B2"] * y,
                 y * (1 + d) + p["P2"] * (r2 + 2 * y * y) + 2 * p["P1"] * x * y]


def test_m14_brown_model(tmp_path):
    path = tmp_path / "results_BrownCamera.txt"
    path.write_text(BROWN)
    cam = calibconsumer.load(path)
    assert isinstance(cam, BrownCamera) and cam.resolution == (4000, 3000)
    assert cam.parameters["C"] == 2.718001064 and cam.parameters["B2"] == 1.3961135e-03
    assert set(cam.views) == {"image-0000"} and np.allclose(cam.views["image-0000"][1], [36.63428, 183.35466, 124.65873])
    rng = np.random.default_rng(14)
    uv = rng.uniform([0, 0], [3999, 2999], (3000, 2))
    o, d = cam.pixel_to_ray(uv)
    ideal = brown_reference(cam.parameters, cam.to_sensor(uv))
    expected = np.c_[ideal, np.full(len(uv), -cam.parameters["C"])]
    assert np.allclose(d, expected / np.linalg.norm(expected, axis=-1, keepdims=True), atol=1e-14)
    back = cam.world_to_pixel(d * 300.0)
    assert np.max(np.abs(back - uv)) < 1e-8
    R, c = cam.views["image-0000"]
    world = (d * 250.0) @ R.T + c
    assert np.max(np.abs(cam.world_to_pixel(world, pose=(R, c)) - uv)) < 1e-7
    uv_b, status = cam.world_to_pixel(-d[:5] * 300.0, return_status=True)
    assert np.all(status == Status.BEHIND) and np.isnan(uv_b).all()
    odd = BrownCamera(cam.parameters, cam.pixel_size, (3900, 2613))  # an odd height: the centre at H/2, not H//2
    assert np.allclose(odd.to_sensor(np.array([[1949.5, 1306.0]])), 0.0)


def test_m15_load_folders_and_the_opt_in_refit(tmp_path):
    pytest.importorskip("scipy")
    from calibconsumer import fit
    folder = tmp_path / "cameraLensInstance-1"
    folder.mkdir()
    (folder / "results_OpenCV.json").write_text(json.dumps({"cameraMatrix": K, "distortionCoefficients": COEFFICIENTS[4],
                                                            "imageSize": [4000, 3000]}))
    cam = calibconsumer.load(folder)
    assert isinstance(cam, OpenCVCamera)
    document = fit.from_camera(cam)  # the grid chosen by the bound (a strong barrel needs 48 × 36)
    assert document["refit"]["passed"] and document["refit"]["grid"]["pixelMax"] <= 0.02
    (folder / "mapping.json").write_text(json.dumps(document))
    spline = calibconsumer.load(folder)  # the folder's mapping.json wins
    assert isinstance(spline, calibconsumer.Camera)
    uv = np.random.default_rng(15).uniform([200, 200], [3800, 2800], (500, 2))
    _, a = spline.pixel_to_ray(uv)
    _, b = cam.pixel_to_ray(uv)
    angle = np.arccos(np.clip(np.sum(a * b, -1), -1, 1))
    assert np.max(angle / spline.pixel_angle(uv)) < 0.05
    with pytest.raises(calibconsumer.FormatError):
        calibconsumer.load(tmp_path)
