# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""The re-fit of ray samples to the common model (docs/accuracy.md): any calibrated camera, given as rays at pixels,
becomes a uniform cubic B-spline of the direction (AE coordinates about an axis) and, if non-central, of the ray's
foot point. Needs SciPy (the extra `fit`)."""

from __future__ import annotations

import numpy as np

from . import direction as ae
from .camera import Camera, Status
from .fileformat import FORMAT, FORMAT_VERSION, validate
from .hull import ConvexPolygon
from .spline import Spline

SMOOTHING = 1e-7  # weight of the controls' second differences relative to the data (fills controls without data)


def fit_spline(uv, values, rectangle, intervals, smoothing=SMOOTHING, weights=None) -> Spline:
    """The least-squares spline through values (N, C) at pixels uv (N, 2): sparse normal equations plus a small
    second-difference penalty on the controls along u and v (it determines controls no sample reaches and leaves a
    well-sampled spline practically unchanged; 0 turns it off)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).reshape(len(uv), -1)
    n, m = intervals
    shape = Spline(rectangle, intervals, np.zeros((m + 3, n + 3, 1)))
    rows, cols, w = shape.design_rows(uv)
    K = (n + 3) * (m + 3)
    A = sparse.csr_matrix((w, (rows, cols)), shape=(len(uv), K))
    if weights is not None:
        A = sparse.diags(np.asarray(weights, dtype=float)) @ A
        values = values * np.asarray(weights, dtype=float)[:, None]
    normal = (A.T @ A).tocsc()
    if smoothing:
        def second_difference(count, stride, other, other_stride):
            r, c, v = [], [], []
            row = 0
            for o in range(other):
                for k in range(count - 2):
                    for d, coef in enumerate((1.0, -2.0, 1.0)):
                        r.append(row)
                        c.append(o * other_stride + (k + d) * stride)
                        v.append(coef)
                    row += 1
            return sparse.csr_matrix((v, (r, c)), shape=(row, K))
        D = sparse.vstack([second_difference(n + 3, 1, m + 3, n + 3), second_difference(m + 3, n + 3, n + 3, 1)])
        scale = normal.diagonal().mean() if normal.nnz else 1.0
        normal = normal + smoothing * scale * (D.T @ D).tocsc()
    rhs = A.T @ values
    controls = np.column_stack([np.atleast_1d(spsolve(normal, rhs[:, k])) for k in range(values.shape[1])])
    return Spline(rectangle, intervals, controls.reshape(m + 3, n + 3, -1))


def rectangle_of(points, resolution, margin=0.05):
    """The bounding box of points dilated by `margin` of its size, clipped to the sensor ±0.5 px."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    lo, hi = p.min(axis=0), p.max(axis=0)
    pad = margin * (hi - lo)
    sensor_lo, sensor_hi = np.array([-0.5, -0.5]), np.asarray(resolution, dtype=float) - 0.5
    return np.array([np.maximum(lo - pad, sensor_lo), np.minimum(hi + pad, sensor_hi)])


def from_ray_samples(uv, directions, origins=None, *, hull, resolution, intervals, rectangle=None,
                     rotation=None, smoothing=SMOOTHING) -> dict:
    """A complete mapping document (without refit, producer and views) from ray samples: uv (N, 2) pixels, their unit
    directions (N, 3) and – non-central – any point of each ray (N, 3), in the camera frame x right, y up, viewing
    along −z. hull: the calibrated area's points or polygon (pixels). rotation: the AE axis (default: the mean
    direction of the samples inside the hull)."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    good = np.all(np.isfinite(uv), axis=-1) & np.all(np.isfinite(d), axis=-1)
    polygon = hull if isinstance(hull, ConvexPolygon) else ConvexPolygon.of_points(hull)
    if rectangle is None:
        rectangle = rectangle_of(polygon.vertices, resolution)
    rectangle = np.asarray(rectangle, dtype=float)
    if rotation is None:
        inside = good & polygon.contains(uv)
        rotation = ae.axis_rotation(d[inside].mean(axis=0))
    ab = ae.from_direction(d, rotation)
    document = {
        "format": FORMAT, "formatVersion": FORMAT_VERSION,
        "camera": {"resolution": [int(resolution[0]), int(resolution[1])]},
        "conventions": {"pixel": "(0, 0) = centre of the top-left pixel, u right, v down",
                        "cameraFrame": "x right, y up, viewing along -z", "lengthUnit": "world file units"},
        "central": origins is None,
        "domain": {"rectangle": rectangle.tolist(), "hull": polygon.vertices.tolist()},
        "spline": {"type": "uniform-cubic-bspline", "intervals": [int(intervals[0]), int(intervals[1])]},
        "direction": {"parametrization": "azimuthal-equidistant", "axisRotation": np.asarray(rotation).tolist()},
        "origin": None,
    }
    if origins is not None:
        o = np.asarray(origins, dtype=float).reshape(-1, 3)
        good &= np.all(np.isfinite(o), axis=-1)
        foot = o - d * np.sum(d * o, axis=-1, keepdims=True)
        joint = fit_spline(uv[good], np.c_[ab[good], foot[good]], rectangle, intervals, smoothing)
        document["direction"]["controls"] = joint.controls[:, :, :2].ravel().tolist()
        document["origin"] = {"parametrization": "foot-point", "controls": joint.controls[:, :, 2:].ravel().tolist()}
    else:
        spline = fit_spline(uv[good], ab[good], rectangle, intervals, smoothing)
        document["direction"]["controls"] = spline.controls.ravel().tolist()
    share = folded_share(document)
    document["inverse"] = {"startRays": [32, 24], "injective": share == 0.0, "foldedShare": share}
    validate(document)
    return document


def from_camera(camera, intervals=None, hull=None, step=None, smoothing=SMOOTHING, pixel_max=0.02, cap=64) -> dict:
    """A mapping document re-fitted from any camera with pixel_to_ray – e.g. an OpenCVCamera or BrownCamera, to use the
    one spline format for every camera (opt-in). The calibrated area is the camera's hull, or the sensor (classic
    cameras) without the part beyond the model's valid radius; the refit block measures the fit on pixels it did not
    use. Without `intervals` the smallest grid of 8, 12, 16, 24, 32, 48, 64 intervals across the width that meets
    `pixel_max` (else the cap's)."""
    W, H = camera.resolution
    if intervals is None:
        document = None
        for n in (8, 12, 16, 24, 32, 48, 64):
            if n > cap:
                break
            m = max(2, min(cap, int(round(n * H / W))))
            document = from_camera(camera, (n, m), hull, step, smoothing, pixel_max, cap)
            if document["refit"]["grid"]["pixelMax"] <= pixel_max:
                break
        return document
    step = step or max(2.0, max(W, H) / 400.0)
    u, v = np.meshgrid(np.arange(0.0, W, step), np.arange(0.0, H, step))
    uv = np.stack([u, v], -1).reshape(-1, 2)
    o, d, status = camera.pixel_to_ray(uv, allow_extrapolation=True, return_status=True)
    good = np.all(np.isfinite(d), axis=-1) & (status != Status.OUTSIDE_DOMAIN) & (status != Status.NO_CONVERGENCE)
    if hull is None:
        hull = camera.hull.vertices if hasattr(camera, "hull") else uv[good]
    document = from_ray_samples(uv[good], d[good], None if camera.central else o[good], hull=hull,
                                resolution=(W, H), intervals=intervals, smoothing=smoothing)
    fitted = Camera(document, quiet=True)
    check = uv[good] + 0.5 * step
    check = check[fitted.contains(check)]
    o_ref, d_ref = camera.pixel_to_ray(check, allow_extrapolation=True)
    e = errors(fitted, check, d_ref, None if camera.central else o_ref)
    document["refit"] = {"grid": {"samples": int(len(check)), "pixelMax": float(np.nanmax(e["pixel"])),
                                  "pixelRms": float(np.sqrt(np.nanmean(e["pixel"] ** 2))),
                                  "angleMaxRad": float(np.nanmax(e["angle"]))},
                         "bounds": {"gridPixelMax": pixel_max},
                         "passed": bool(np.nanmax(e["pixel"]) <= pixel_max)}
    if getattr(camera, "views", None):
        document["views"] = {name: {"rotation": np.asarray(R).tolist(), "centre": np.asarray(c).tolist()}
                             for name, (R, c) in camera.views.items()}
    document["source"] = {"model": type(camera).__name__}
    return document


def injective(document, samples=200) -> bool:
    """Whether the direction map is one-to-one over the hull: the Jacobian determinant of (u, v) → (a, b) keeps one
    sign on a dense grid (central cameras; a non-central file is judged by its direction alone)."""
    return folded_share(document, samples) == 0.0


def folded_share(document, samples=200) -> float:
    """The share of the hull's area (on a dense grid) where the direction map's Jacobian determinant has the sign
    opposite to the majority: where the map folds over and world → pixel is not unique."""
    camera = Camera(dict(document, inverse={"startRays": [2, 2]}), quiet=True)
    lo, hi = camera.hull.vertices.min(axis=0), camera.hull.vertices.max(axis=0)
    u, v = np.meshgrid(np.linspace(lo[0], hi[0], samples), np.linspace(lo[1], hi[1], samples))
    uv = np.stack([u, v], -1).reshape(-1, 2)
    uv = uv[camera.hull.contains(uv)]
    _, ab_u, ab_v = camera._direction.evaluate(uv, derivatives=True)
    det = ab_u[:, 0] * ab_v[:, 1] - ab_u[:, 1] * ab_v[:, 0]
    return float(np.mean(det * np.sign(np.median(det)) <= 0)) if len(det) else 0.0


def errors(camera: Camera, uv, directions, origins=None, distances=()):
    """The re-fit's errors at pixels uv against reference rays: per sample the angle (rad), the pixel-equivalent
    (angle / local pixel angle, the normalized angular error) and – for each distance Z – the object-space distance
    between the reference ray's point at Z and the fitted ray."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    d_ref = np.asarray(directions, dtype=float).reshape(-1, 3)
    d_ref = d_ref / np.linalg.norm(d_ref, axis=-1, keepdims=True)
    o_ref = np.zeros_like(d_ref) if origins is None else np.asarray(origins, dtype=float).reshape(-1, 3)
    o, d = camera._rays(uv)
    angle = np.arctan2(np.linalg.norm(np.cross(d, d_ref), axis=-1), np.sum(d * d_ref, axis=-1))
    out = {"angle": angle, "pixel": angle / camera.pixel_angle(uv), "objectSpace": {}}
    for Z in distances:
        p = o_ref + Z * d_ref - o
        out["objectSpace"][float(Z)] = np.linalg.norm(p - d * np.sum(p * d, axis=-1, keepdims=True), axis=-1)
    return out
