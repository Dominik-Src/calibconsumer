# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""M19–M22: images of the viewing directions – sphere and cube maps (M19), maps from a coarse grid (M20), the mapping
images without a photograph (M21), the stereo rectification of a pair (M22)."""
import math
import os

import numpy as np
import pytest

import calibconsumer
from calibconsumer import remap, stereo, visual

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return calibconsumer.load(os.path.join(HERE, "data", f"{name}.json"), quiet=True)


def test_m19_sphere_and_cube_maps_find_the_truths_pixels(case):
    camera, truth = case
    if not camera.central or truth.name == "tele800":
        pytest.skip("central cameras with a field wider than the panorama's pixels (the telephoto's ±0.8° falls between them)")
    size = (256, 128)
    for pole in ("up", "axis"):
        map_x, map_y = remap.sphere_maps(camera, size, pole=pole, step=1)
        seen = map_x >= 0
        assert seen.any()
        rows, cols = np.nonzero(seen)
        pick = np.linspace(0, len(rows) - 1, 40).astype(int)
        uv = np.stack([map_x[rows[pick], cols[pick]], map_y[rows[pick], cols[pick]]], -1).astype(float)
        _, d = truth.rays(uv)
        expected = remap.sphere_directions(size, pole)[rows[pick], cols[pick]]
        angle = np.degrees(np.arccos(np.clip(np.sum(d * expected, -1), -1, 1)))
        assert angle.max() < 0.05
    # the cube's front face centre looks along the optical axis: the truth's centre pixel (up to a pixel)
    faces = remap.cube_maps(camera, 33, step=1)
    centre = np.array([faces["front"][0][16, 16], faces["front"][1][16, 16]])
    assert np.linalg.norm(centre - truth.centre) < 1.0
    assert set(faces) == {"front", "right", "back", "left", "up", "down"}


def test_m19_sphere_angles_invert_the_directions():
    for pole in ("up", "axis"):
        d = remap.sphere_directions((64, 32), pole)
        a, b = remap.sphere_angles(d, pole)
        u = (a + np.pi) / (2 * np.pi) * 64 - 0.5
        v = ((np.pi / 2 - b) if pole == "up" else b) / np.pi * 32 - 0.5
        assert np.allclose(u, np.arange(64)[None, :], atol=1e-9) and np.allclose(v, np.arange(32)[:, None], atol=1e-9)


def relative_error(exact, coarse):
    """The coarse map's largest error in units of the output pixel's footprint on the sensor (the exact map's step to
    the neighbouring output pixel), over the pixels both map."""
    seen = (exact[0] >= 0) & (coarse[0] >= 0)
    dx = np.hypot(np.gradient(exact[0], axis=1), np.gradient(exact[1], axis=1))
    dy = np.hypot(np.gradient(exact[0], axis=0), np.gradient(exact[1], axis=0))
    footprint = np.maximum(dx, dy)
    good = seen & (footprint > 0) & (footprint < 1e4)
    error = np.hypot(exact[0] - coarse[0], exact[1] - coarse[1])
    return (float((error[good] / footprint[good]).max()) if good.any() else 0.0), seen


def test_m20_a_coarse_grid_gives_the_exact_maps_within_a_quarter_of_an_output_pixel(case):
    camera, _ = case
    distance = None if camera.central else 2000.0
    exact = remap.sphere_maps(camera, (200, 100), step=1, distance=distance)
    coarse = remap.sphere_maps(camera, (200, 100), step=6, distance=distance)
    assert np.array_equal(exact[0] >= 0, coarse[0] >= 0)
    error, _ = relative_error(exact, coarse)
    assert error < 0.25  # cells of 10.8° on the sphere: 0.11–0.2 output pixels on the asymmetric and the axial truth
    # the pinhole of before (OpenCV frame) unchanged by the coarse grid
    W, H = camera.resolution
    a = remap.pinhole_maps(camera, 0.3 * W, (160, 120), distance=distance)
    b = remap.pinhole_maps(camera, 0.3 * W, (160, 120), distance=distance, step=8)
    assert np.array_equal(a[0] >= 0, b[0] >= 0)
    error, _ = relative_error(a, b)
    assert error < 0.25


def test_m21_the_mapping_images():
    camera = load("fisheye200")
    on_sphere = visual.sensor_on_sphere(camera, (256, 128), squares=8)
    assert on_sphere.shape == (128, 256, 3) and on_sphere.dtype == np.uint8
    colours = {tuple(c) for c in on_sphere.reshape(-1, 3)}
    assert {visual.LIGHT, visual.DARK, visual.BACKGROUND} <= colours
    # straight ahead the sensor's centre square; straight behind nothing (200° field of view: the rim but not the back)
    assert tuple(on_sphere[64, 128]) in (visual.LIGHT, visual.DARK)
    assert tuple(on_sphere[64, 0]) == visual.BACKGROUND
    on_sensor = visual.sphere_on_sensor(camera, width=320, grid_deg=15, pole="axis")
    assert on_sensor.shape == (240, 320, 3)
    grid = np.all(on_sensor == visual.GRID, axis=-1)
    assert 0.01 < grid.mean() < 0.25
    checker = visual.sphere_on_sensor(camera, width=160, checkerboard=True, pole="up")
    assert {visual.LIGHT, visual.DARK} <= {tuple(c) for c in checker.reshape(-1, 3)}
    texture = np.zeros((90, 180, 3), np.uint8)
    texture[:, 90:] = (200, 10, 10)  # the eastern hemisphere red
    painted = visual.sphere_on_sensor(camera, width=160, texture=texture)
    assert tuple(painted[60, 150]) == (200, 10, 10) and tuple(painted[60, 10]) == (0, 0, 0)
    # two cameras of one rig turned by 60° about y: each covers its part, both the overlap, nothing behind
    turn = math.radians(60)
    rig = np.array([[math.cos(turn), 0, math.sin(turn)], [0, 1, 0], [-math.sin(turn), 0, math.cos(turn)]])
    image, count = visual.coverage_on_sphere([camera, camera], [None, rig], (128, 64))
    assert image.shape == (64, 128, 3) and count.max() == 2 and count.min() == 0
    assert count[32, 64] >= 1 and tuple(image[32, 0]) == visual.BACKGROUND


def test_m22_a_rectified_pair_shares_its_rows():
    camera = load("fisheye150")
    turn = 0.04
    R = np.array([[math.cos(turn), 0, math.sin(turn)], [0, 1, 0], [-math.sin(turn), 0, math.cos(turn)]])
    t = np.array([60.0, 1.5, -2.0])
    result = stereo.rectify(camera, camera, R, t, size=(320, 240), focal_px=200.0, step=4)
    assert result["vertical"] is False and result["baseline"] == pytest.approx(np.linalg.norm(t))
    rect1 = result["rotation1"]
    assert np.allclose(rect1.T @ rect1, np.eye(3)) and np.linalg.det(rect1) == pytest.approx(1.0)
    f, (cx, cy) = result["focal"], result["principalPoint"]
    rng = np.random.default_rng(4)
    for X in rng.uniform([-400, -300, -2500], [400, 300, -900], (20, 3)):  # in camera 1's frame, in front of it
        p1 = rect1.T @ X                       # the point in the rectified frame of camera 1 …
        p2 = rect1.T @ (X - t)                 # … and of camera 2 (its centre at t)
        uv1 = np.array([cx + f * p1[0] / -p1[2], cy - f * p1[1] / -p1[2]])
        uv2 = np.array([cx + f * p2[0] / -p2[2], cy - f * p2[1] / -p2[2]])
        assert uv1[1] == pytest.approx(uv2[1], abs=1e-9)  # one row
        assert uv2[0] < uv1[0]                             # camera 2 on the right: a positive disparity
        # the maps carry the rectified pixel to each camera's own pixel of the point
        for (map_x, map_y), uv, Xk in ((result["maps1"], uv1, X), (result["maps2"], uv2, R.T @ (X - t))):
            if not (0 <= uv[0] <= 319 and 0 <= uv[1] <= 239):
                continue
            i, j = int(round(uv[1])), int(round(uv[0]))
            seen = np.asarray(camera.world_to_pixel(Xk))
            assert np.hypot(map_x[i, j] - seen[0], map_y[i, j] - seen[1]) < 4.0  # within the rounding of (i, j)


def test_m21_the_classic_models_too():
    """An OpenCV camera (no distortion: a pinhole) – the maps against the pinhole's closed form, the mapping images."""
    camera = calibconsumer.OpenCVCamera.from_dict({"cameraMatrix": [[500, 0, 319.5], [0, 500, 239.5], [0, 0, 1]],
                                                  "distortionCoefficients": [0, 0, 0, 0], "imageSize": [640, 480]})
    map_x, map_y = remap.sphere_maps(camera, (360, 180), step=4)
    d = remap.sphere_directions((360, 180))
    seen = map_x >= 0
    expected_u = 319.5 + 500 * d[..., 0] / -d[..., 2]
    expected_v = 239.5 - 500 * d[..., 1] / -d[..., 2]
    assert seen.sum() > 100
    assert np.abs(map_x[seen] - expected_u[seen]).max() < 0.5 and np.abs(map_y[seen] - expected_v[seen]).max() < 0.5
    image = visual.sphere_on_sensor(camera, width=160, grid_deg=10)
    assert image.shape == (120, 160, 3) and np.all(image == visual.GRID, axis=-1).any()
    result = stereo.rectify(camera, camera, np.eye(3), [-50.0, 0, 0], size=(160, 120), focal_px=120.0)
    assert result["vertical"] is False and (result["maps2"][0] >= 0).mean() > 0.5
