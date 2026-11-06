# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""M17: uncertainties – the σ grid's interpolation, the error-statistics reader, the file block's round trip, the
rays' σ, and world-point covariances propagated to pixels (against Monte Carlo) for the spline and the OpenCV camera."""

import json

import numpy as np
import pytest

import calibconsumer
from calibconsumer import Uncertainty
from calibconsumer.uncertainty import Grid


def test_m17a_grid_interpolation():
    grid = Grid(10, 3, 2, [1.0, 2.0, None, 3.0, 4.0, 5.0])
    assert grid(np.array([[4.5, 4.5]]))[0] == pytest.approx(1.0)  # a cell centre
    assert grid(np.array([[9.5, 4.5]]))[0] == pytest.approx(1.5)  # between two centres
    assert grid(np.array([[19.5, 4.5]]))[0] == pytest.approx(2.0)  # the NaN cell left out of the blend
    assert grid(np.array([[-100.0, 100.0]]))[0] == pytest.approx(3.0)  # outside: the border cell
    assert np.isnan(grid(np.array([[np.nan, 1.0]]))[0])
    assert Grid.of(grid.to_document()).values[1, 2] == 5.0


def test_m17b_error_statistics_and_the_file_block(tmp_path, case):
    camera, _ = case
    stats = {"quality": {"version": 1, "linearization": {"unit": "normalized"},
                         "sigmaMap": {"stepPx": 64, "width": 2, "height": 1, "values": [0.1, None]}},
             "noiseModel": {"sigmaPx": 0.2}}
    (tmp_path / "error-statistics.json").write_text(json.dumps(stats))
    u = Uncertainty.from_error_statistics(tmp_path / "error-statistics.json")
    assert u.ray.values[0, 0] == pytest.approx(0.1 * np.sqrt(2)) and u.image_sigma == 0.2
    document = dict(camera.document, uncertainty=u.to_document())
    (tmp_path / "mapping.json").write_text(json.dumps(document))
    loaded = calibconsumer.load(tmp_path / "mapping.json", quiet=True)
    assert loaded.uncertainty.ray.values[0, 0] == pytest.approx(0.1 * np.sqrt(2), rel=1e-3)
    assert loaded.with_frame("opencv").uncertainty is not None


def test_m17c_ray_sigma(case):
    camera, _ = case
    camera.uncertainty = Uncertainty(ray=0.3, refit_rms=0.4)
    try:
        W, H = camera.resolution
        uv = np.array([[W / 2, H / 2], [0.4 * W, 0.6 * H], [-1e6, 0.0]])
        o, d, sigma = camera.pixel_to_ray(uv, return_sigma=True)
        assert sigma[:2] == pytest.approx(0.5 * camera.pixel_angle(uv[:2]), rel=1e-3)
        assert np.isnan(sigma[2])  # refused pixel
    finally:
        camera.uncertainty = None
    assert np.isnan(camera.ray_sigma(uv[:1])[0])  # no uncertainty: unknown


def monte_carlo(camera, X, S, pose=None, n=40000, seed=17):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(S)
    samples = X[None] + np.einsum("nij,knj->kni", L, rng.normal(size=(n, len(X), 3)))
    pixels = camera.world_to_pixel(samples, pose=pose, allow_extrapolation=True)
    return np.array([np.cov(pixels[:, i].T) for i in range(len(X))])


@pytest.mark.parametrize("frame", ["calibconsumer", "opencv"])
def test_m17d_world_covariance_to_pixels(case, frame):
    camera, _ = case
    camera = camera.with_frame(frame)
    W, H = camera.resolution
    uv = np.array([[0.35 * W, 0.6 * H], [0.7 * W, 0.3 * H]])
    o, d = camera.pixel_to_ray(uv)
    angle = camera.pixel_angle(uv)
    X = o + 40 * d
    rng = np.random.default_rng(5)
    A = rng.normal(size=(2, 3, 3))
    S = A @ np.swapaxes(A, 1, 2) * ((3 * 40 * angle) ** 2 / 9)[:, None, None]  # a few px of spread
    with pytest.warns(UserWarning, match="ray σ is unknown"):  # no uncertainty: only the world points' part
        p, cov = camera.world_to_pixel(X, return_covariance=True, world_covariance=S)
    assert p == pytest.approx(uv, abs=1e-6)
    assert cov == pytest.approx(camera.pixel_covariance(X, p, world_covariance=S, model=False))
    expected = monte_carlo(camera, X, S)
    assert np.abs(cov - expected).max() <= 0.03 * np.abs(expected).max()
    if frame == "calibconsumer":  # world coordinates through a pose (the file's frame)
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        R *= np.linalg.det(R)
        c = rng.normal(size=3)
        Xw, Sw = X @ R.T + c, R @ S @ R.T
        cov_w = camera.pixel_covariance(Xw, p, pose=(R, c), world_covariance=Sw, model=False)
        assert cov_w == pytest.approx(cov, rel=1e-6, abs=1e-12)


def test_m17e_classic_camera_and_image_noise():
    camera = calibconsumer.OpenCVCamera(np.array([[800, 0, 640], [0, 810, 480], [0, 0, 1.0]]),
                                        [-0.3, 0.1, 0.001, -0.002, -0.02], (1280, 960))
    camera.uncertainty = Uncertainty(ray=0.05, image_sigma=0.3)
    X = np.array([[0.4, 0.3, -5.0], [-1.0, 0.5, -4.0]])
    S = np.broadcast_to(np.diag([1e-4, 4e-4, 9e-4]), (2, 3, 3))
    p, cov = camera.world_to_pixel(X, return_covariance=True, world_covariance=S, image_noise=True)
    world = camera.pixel_covariance(X, p, world_covariance=S, model=False)
    assert cov == pytest.approx(world + (0.05 ** 2 + 0.3 ** 2) * np.eye(2), rel=1e-9)
    assert np.abs(world - monte_carlo(camera, X, S)).max() <= 0.03 * np.abs(world).max()
    major, minor, _ = calibconsumer.ellipse(cov).T
    assert np.all(major >= minor) and np.all(minor > 0.3)
