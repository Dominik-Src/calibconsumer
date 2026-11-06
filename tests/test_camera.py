# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""M3–M8, M11: round trips, accuracy against the truth, the extrapolation switch, near range, shapes, frames."""

import numpy as np

import calibconsumer
from calibconsumer import Status


def inside_pixels(camera, n=4000, seed=0):
    W, H = camera.resolution
    uv = np.random.default_rng(seed).uniform([0, 0], [W - 1, H - 1], size=(n, 2))
    return uv[camera.contains(uv)]


def test_m3_pixel_ray_pixel(case):
    camera, truth = case
    uv = inside_pixels(camera)
    o, d = camera.pixel_to_ray(uv)
    for Z in (camera.min_unique_distance or 1.0, 10.0, 1e3, 1e5):
        for allow in (False, True):  # the switch never moves a point that has its pixel inside the calibrated area
            back, status = camera.world_to_pixel(o + max(Z, 1.0) * d, allow_extrapolation=allow, return_status=True)
            good = status != Status.AMBIGUOUS_NEAR
            assert np.all(status[good] == Status.OK)
            assert np.nanmax(np.linalg.norm(back - uv, axis=-1)) < 1e-6


def test_m4_world_pixel_ray(case):
    camera, truth = case
    rng = np.random.default_rng(4)
    uv = inside_pixels(camera, seed=4)
    o, d = camera.pixel_to_ray(uv)
    Z = rng.uniform(50, 1e4, len(uv))[:, None]
    X = o + Z * d + rng.normal(scale=1e-6, size=o.shape)
    pixel, status = camera.world_to_pixel(X, return_status=True)
    o2, d2 = camera.pixel_to_ray(pixel)
    v = X - o2
    distance = np.linalg.norm(v - d2 * np.sum(v * d2, -1, keepdims=True), axis=-1)
    assert np.all(status == Status.OK) and np.all(distance <= 1e-9 * Z[:, 0] + 2e-6)


def test_m5_accuracy_against_the_truth_within_the_stated_refit(case):
    camera, truth = case
    from calibconsumer.fit import errors
    uv = inside_pixels(camera, seed=5)
    o, d = truth.rays(uv)
    e = errors(camera, uv, d, None if truth.central else o)
    assert e["pixel"].max() <= 1.5 * camera.refit["grid"]["pixelMax"] + 1e-9


def test_m6_extrapolation_switch(case):
    camera, _ = case
    W, H = camera.resolution
    corner = np.array([[1.0, 1.0]])  # outside the elliptic hull, inside the rectangle
    assert not camera.contains(corner)[0]
    o, d, status = camera.pixel_to_ray(corner, return_status=True)
    assert np.isnan(d).all() and status[0] == Status.OUTSIDE_HULL
    o, d, status = camera.pixel_to_ray(corner, allow_extrapolation=True, return_status=True)
    assert np.isfinite(d).all() and status[0] == Status.OUTSIDE_HULL
    far = np.array([[-10.0 * W, 0.5 * H]])
    for allow in (False, True):
        o, d, status = camera.pixel_to_ray(far, allow_extrapolation=allow, return_status=True)
        assert np.isnan(d).all() and status[0] == Status.OUTSIDE_DOMAIN
    o, d = camera.pixel_to_ray(corner, allow_extrapolation=True)
    X = o + 2000.0 * d
    uv, status = camera.world_to_pixel(X, return_status=True)
    assert np.isnan(uv).all() and status[0] == Status.OUTSIDE_HULL
    uv, status = camera.world_to_pixel(X, allow_extrapolation=True, return_status=True)
    assert status[0] == Status.OUTSIDE_HULL and np.allclose(uv, corner, atol=1e-6)
    if camera.central:  # a point behind a central camera's field of view
        o, d = camera.pixel_to_ray(np.array([[(W - 1) / 2, (H - 1) / 2]]))
        uv, status = camera.world_to_pixel(-1000.0 * d, return_status=True)
        assert np.isnan(uv).all() and status[0] in (Status.BEHIND, Status.NO_CONVERGENCE, Status.OUTSIDE_HULL)


def test_m7_non_central_near_range():
    camera = calibconsumer.load(__import__("os").path.join(__import__("os").path.dirname(__file__), "data", "axial.json"))
    uv = inside_pixels(camera, seed=7)
    o, d = camera.pixel_to_ray(uv)
    back, status = camera.world_to_pixel(o + 40.0 * d, return_status=True)  # beyond minUniqueDistance (20)
    assert np.mean(status == Status.OK) >= 0.999
    back, status = camera.world_to_pixel(o + 10.0 * d, return_status=True)
    assert np.all((status == Status.AMBIGUOUS_NEAR) | (status == Status.NO_CONVERGENCE))


def test_m8_shapes_types_and_no_mutation(case):
    camera, _ = case
    uv = inside_pixels(camera)[:24].astype(np.float32).reshape(2, 3, 4, 2)
    keep = uv.copy()
    o, d = camera.pixel_to_ray(uv)
    assert o.shape == (2, 3, 4, 3) and d.shape == (2, 3, 4, 3) and d.dtype == np.float64
    assert np.array_equal(uv, keep)
    X = (o + 100.0 * d)
    back = camera.world_to_pixel(X)
    assert back.shape == (2, 3, 4, 2) and np.allclose(back, uv, atol=1e-4)
    empty_o, empty_d = camera.pixel_to_ray(np.zeros((0, 2)))
    assert empty_d.shape == (0, 3) and camera.world_to_pixel(np.zeros((0, 3))).shape == (0, 2)
    nan_uv, status = camera.world_to_pixel(np.array([[np.nan, 0.0, 1.0]]), return_status=True)
    assert np.isnan(nan_uv).all()
    o, d = camera.pixel_to_ray(np.array([[np.nan, 3.0]]))
    assert np.isnan(d).all()


def test_m11_opencv_frame(case):
    camera, _ = case
    cv = camera.with_frame("opencv")
    uv = inside_pixels(camera)[:200]
    o, d = camera.pixel_to_ray(uv)
    o2, d2 = cv.pixel_to_ray(uv)
    assert np.allclose(d2, d * [1, -1, -1]) and np.allclose(o2, o * [1, -1, -1])
    assert np.median(d2[:, 2]) > 0  # OpenCV looks along +z
    assert np.allclose(cv.world_to_pixel(o2 + 500 * d2), uv, atol=1e-6)


def test_world_points_with_a_pose(case):
    camera, _ = case
    rng = np.random.default_rng(9)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    c = rng.normal(size=3) * 100
    uv = inside_pixels(camera)[:100]
    o, d = camera.pixel_to_ray(uv)
    X_world = (o + 300 * d) @ R.T + c  # camera → world
    assert np.allclose(camera.world_to_pixel(X_world, pose=(R, c)), uv, atol=1e-6)


def test_m16_bilinear_start(case):
    """The world → pixel start: the bilinear position in the start grid's cell beats the nearest grid ray's pixel, and
    from a coarse 4 × 3 grid two Gauss–Newton steps still reach the pixel."""
    from calibconsumer import inverse
    camera, truth = case
    uv = inside_pixels(camera, seed=16)
    o, d = camera.pixel_to_ray(uv)
    X = o + np.random.default_rng(16).uniform(50, 1e4, len(uv))[:, None] * d
    saved = inverse.BILINEAR, camera.start_rays
    try:
        errors = {}
        for bilinear in (False, True):
            inverse.BILINEAR, camera._starts = bilinear, {}
            start, _ = inverse._start(camera, X, False)
            errors[bilinear] = np.median(np.linalg.norm(start - uv, axis=-1))
        assert errors[True] < 0.1 * errors[False]
        camera.start_rays, camera._starts = (4, 3), {}
        start, _ = inverse._start(camera, X, False)
        p = inverse._iterate(camera, X, start, 3)
        assert np.mean(np.linalg.norm(p - uv, axis=-1) < 1e-6) > 0.99
    finally:
        (inverse.BILINEAR, camera.start_rays), camera._starts = saved, {}
