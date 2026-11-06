# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""M9: the file format; M10: the re-fit (needs SciPy)."""

import copy
import os

import numpy as np
import pytest

import calibconsumer
from calibconsumer import fileformat

DATA = os.path.join(os.path.dirname(__file__), "data")


def document(name="fisheye150"):
    return fileformat.read(os.path.join(DATA, f"{name}.json"))


def test_m9_write_read_bit_identical(tmp_path):
    for name in ("fisheye150", "axial"):
        original = document(name)
        path = tmp_path / "m.json"
        fileformat.write(path, original)
        assert fileformat.read(path) == original
        a, b = calibconsumer.load(os.path.join(DATA, f"{name}.json")), calibconsumer.load(path)
        uv = np.array([[700.3, 600.1], [800.0, 300.25]])
        assert np.array_equal(a.pixel_to_ray(uv)[1], b.pixel_to_ray(uv)[1])


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(formatVersion=2), "formatVersion"),
    (lambda d: d.update(format="other"), "format"),
    (lambda d: d["spline"].update(intervals=[0, 3]), "intervals"),
    (lambda d: d["direction"]["controls"].pop(), "numbers"),
    (lambda d: d["direction"].update(axisRotation=[[2, 0, 0], [0, 1, 0], [0, 0, 1]]), "rotation"),
    (lambda d: d.pop("domain"), "rectangle"),
    (lambda d: d.update(central=False), "origin"),
    (lambda d: d["camera"].pop("resolution"), "resolution"),
])
def test_m9_refusals(change, message):
    d = copy.deepcopy(document())
    change(d)
    with pytest.raises(calibconsumer.FormatError, match=message):
        calibconsumer.Camera(d)


def test_m9_non_injective_and_failed_refit():
    d = copy.deepcopy(document())
    d["inverse"]["injective"], d["inverse"]["foldedShare"] = False, 0.004
    with pytest.warns(UserWarning, match="folds over"):
        calibconsumer.Camera(d)
    d = copy.deepcopy(document())
    d["refit"]["passed"] = False
    with pytest.warns(UserWarning, match="re-fit bounds"):
        calibconsumer.Camera(d)


def test_m9_strict_json(tmp_path):
    d = copy.deepcopy(document())
    d["refit"]["grid"]["pixelMax"] = float("nan")
    with pytest.raises(ValueError):
        fileformat.write(tmp_path / "x.json", d)


def test_m10_exact_spline_comes_back_exactly():
    pytest.importorskip("scipy")
    from calibconsumer import direction as ae, fit
    from calibconsumer.spline import Spline
    rng = np.random.default_rng(10)
    rect = np.array([[-0.5, -0.5], [639.5, 479.5]])
    truth = Spline(rect, (8, 6), rng.normal(scale=0.05, size=(9, 11, 2)) + 0.0)
    u, v = np.meshgrid(np.linspace(-0.5, 639.5, 81), np.linspace(-0.5, 479.5, 61))
    uv = np.stack([u, v], -1).reshape(-1, 2)
    again = fit.fit_spline(uv, truth.evaluate(uv), rect, (8, 6), smoothing=0)
    assert np.abs(again.controls - truth.controls).max() <= 1e-10
    R = np.eye(3)
    d = ae.to_direction(truth.evaluate(uv), R)
    hull = np.array([[0, 0], [639, 0], [639, 479], [0, 479]], float)
    doc = fit.from_ray_samples(uv, d, hull=hull, resolution=(640, 480), intervals=(8, 6), rectangle=rect,
                               rotation=R, smoothing=0)
    assert np.abs(np.asarray(doc["direction"]["controls"]).reshape(7 + 2, 11, 2) - truth.controls).max() <= 1e-10


def test_m10_refit_of_the_truths_meets_the_bounds():
    pytest.importorskip("scipy")
    import truths
    from calibconsumer import fit
    for name, intervals in (("fisheye150", (8, 6)), ("tele800", (4, 3)), ("axial", (12, 9))):
        t = truths.truths()[name]
        uv = truths.samples(t, step=10)
        o, d = t.rays(uv)
        doc = fit.from_ray_samples(uv, d, None if t.central else o, hull=truths.hull_points(t),
                                   resolution=t.resolution, intervals=intervals)
        camera = calibconsumer.Camera(doc)
        check = truths.samples(t, step=7) + 3.0
        check = check[camera.contains(check)]
        co, cd = t.rays(check)
        e = fit.errors(camera, check, cd, None if t.central else co, distances=(1000.0,))
        assert e["pixel"].max() <= 0.02 and np.sqrt(np.mean(e["pixel"] ** 2)) <= 0.005, name
        assert doc["inverse"]["injective"] is True


def test_m12_a_folded_map_loads_and_flags_its_folds():
    """M12: a file whose direction map folds over in part of the hull loads with a warning; world → pixel gives
    FOLDED exactly for the points whose pixel lies in the fold (NaN unless allow_folded=True), OK elsewhere."""
    pytest.importorskip("scipy")
    from calibconsumer import Status, fit
    d = copy.deepcopy(document())
    n, m = d["spline"]["intervals"]
    c = np.asarray(d["direction"]["controls"]).reshape(m + 3, n + 3, 2)
    c[5:10, 9, 0] = c[5:10, 12, 0]  # a bump in a along u: ∂a/∂u < 0 between columns 9 and 11
    d["direction"]["controls"] = c.ravel().tolist()
    share = fit.folded_share(d)
    assert 0 < share < 0.2
    d["inverse"]["injective"], d["inverse"]["foldedShare"] = False, share
    with pytest.warns(UserWarning, match="folds over"):
        camera = calibconsumer.Camera(d)
    rng = np.random.default_rng(12)
    uv = rng.uniform([0, 0], [1599, 1199], size=(6000, 2))
    uv = uv[camera.contains(uv)]
    assert camera.folded(uv).any() and not camera.folded(uv).all()
    o, dirs = camera.pixel_to_ray(uv)
    X = o + 1000.0 * dirs
    pixel, status = camera.world_to_pixel(X, allow_folded=True, return_status=True)
    found = np.all(np.isfinite(pixel), axis=-1)
    assert np.all(status[found][camera.folded(pixel[found])] == Status.FOLDED)
    assert (status == Status.FOLDED).any() and (status == Status.OK).mean() > 0.8
    o2, d2 = camera.pixel_to_ray(pixel[found])
    v = X[found] - o2  # whatever pixel was found, its ray passes through the point
    assert np.max(np.linalg.norm(v - d2 * np.sum(v * d2, -1, keepdims=True), axis=-1)) < 1e-6
    strict, strict_status = camera.world_to_pixel(X, return_status=True)
    assert np.isnan(strict[strict_status == Status.FOLDED]).all()
    # a point that a pixel in the fold sees has more than one pixel: FOLDED – except at the fold's very border, where
    # the determinant is ≈ 0 and the folded and the unfolded solution merge (2 of 134 here)
    seen_in_fold = camera.folded(uv)
    assert np.mean(status[seen_in_fold] == Status.FOLDED) >= 0.95
