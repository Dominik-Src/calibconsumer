# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""World → pixel: the start pixel from a grid of precomputed rays, then Gauss–Newton on the pixel.

The residual is the part of the unit vector from the ray's origin to the point that is orthogonal to the ray – a
3-vector whose norm is the sine of the angle between ray and point (an angle, not a distance, and a vector, not its
norm: Gauss–Newton then converges quadratically)."""

from __future__ import annotations

import numpy as np

from .camera import Status

CHUNK = 4096                # points per block of the start search
STEP_TOLERANCE = 1e-6       # px: the iteration stops below this step
RESIDUAL_TOLERANCE = 1e-3   # of the local pixel angle: a solution's ray must pass this close
FOLD_MARGIN = 0.01          # of the hull's size: a second, folded solution this far outside the hull still counts


def _start(camera, X, allow_extrapolation):
    """Per point the start pixel and the cosine to the grid ray closest in angle (as seen from that ray's origin): the
    bilinear position in the grid cell around the point – the four corner rays' offsets from the point in its tangent
    plane, the cell coordinates (s, t) where their bilinear blend vanishes, the corner pixels blended with them –, or
    the closest ray's pixel where no neighbouring cell contains the point."""
    grid = camera._start_grid(allow_extrapolation)
    rows, cols = grid["rows"], grid["cols"]
    uv, o, d = grid["uv"], grid["o"], grid["d"]
    start, cos, index = _closest(X, uv[rows, cols], o[rows, cols], d[rows, cols], with_index=True)
    if BILINEAR:
        _bilinear(X, grid, rows[index], cols[index], start)
    return start, cos


def _bilinear(X, grid, row, col, start):
    """Replaces start (in place) by the bilinear position in one of the four cells around each nearest grid node."""
    uv, o, d, finite, step = grid["uv"], grid["o"], grid["d"], grid["finite"], grid["step"]
    h, g = finite.shape
    w_ref = X - o[row, col]
    w_ref /= np.linalg.norm(w_ref, axis=-1, keepdims=True)
    e1 = np.cross(w_ref, np.where(np.abs(w_ref[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]))
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(w_ref, e1)
    found = np.zeros(len(X), bool)
    for dr, dc in ((0, 0), (-1, 0), (0, -1), (-1, -1)):  # the cells with the node as one of their corners
        r0, c0 = row + dr, col + dc
        ok = ~found & (r0 >= 0) & (c0 >= 0) & (r0 + 1 < h) & (c0 + 1 < g)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        r, c = r0[idx], c0[idx]
        corners = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]  # p00 (s = 0, t = 0), p10, p01, p11
        valid = np.all([finite[a, b] for a, b in corners], axis=0)
        p = []
        for a, b in corners:  # each corner ray's offset from the point, seen from its own origin, in the tangent plane
            w = X[idx] - o[a, b]
            w /= np.linalg.norm(w, axis=-1, keepdims=True)
            off = d[a, b] - w
            p.append(np.stack([np.sum(off * e1[idx], -1), np.sum(off * e2[idx], -1)], -1))
        p00, p10, p01, p11 = p
        s = np.full(len(idx), 0.5)
        t = np.full(len(idx), 0.5)
        for _ in range(8):  # Newton on the bilinear blend P(s, t) = 0
            P = (1 - s)[:, None] * (1 - t)[:, None] * p00 + s[:, None] * (1 - t)[:, None] * p10 + \
                (1 - s)[:, None] * t[:, None] * p01 + s[:, None] * t[:, None] * p11
            Ps = (1 - t)[:, None] * (p10 - p00) + t[:, None] * (p11 - p01)
            Pt = (1 - s)[:, None] * (p01 - p00) + s[:, None] * (p11 - p10)
            det = Ps[:, 0] * Pt[:, 1] - Ps[:, 1] * Pt[:, 0]
            det = np.where(np.abs(det) > 1e-300, det, np.nan)
            s = s - (Pt[:, 1] * P[:, 0] - Pt[:, 0] * P[:, 1]) / det
            t = t - (-Ps[:, 1] * P[:, 0] + Ps[:, 0] * P[:, 1]) / det
        inside = valid & np.isfinite(s) & np.isfinite(t) & (s >= -1e-9) & (s <= 1 + 1e-9) & (t >= -1e-9) & (t <= 1 + 1e-9)
        hit = idx[inside]
        start[hit] = uv[r[inside], c[inside]] + np.stack([s[inside], t[inside]], -1) * step
        found[hit] = True


BILINEAR = True  # the bilinear start of _start (False: the closest grid ray's pixel, as before 2026-10-30)


def _start_from(camera, X, uv0):
    """The same from given start pixels (e.g. the pixels in a fold)."""
    o, d = camera._rays(uv0)
    return _closest(X, uv0, o, d)


def _closest(X, uv0, o, d, with_index=False):
    best_uv = np.empty((len(X), 2))
    best_cos = np.empty(len(X))
    best_index = np.empty(len(X), dtype=np.int64)
    for k in range(0, len(X), CHUNK):
        x = X[k:k + CHUNK]
        w = x[:, None, :] - o[None, :, :]
        w /= np.linalg.norm(w, axis=-1, keepdims=True)
        cos = np.einsum("nki,ki->nk", w, d)
        index = np.nanargmax(np.where(np.isfinite(cos), cos, -np.inf), axis=1)
        best_uv[k:k + CHUNK] = uv0[index]
        best_cos[k:k + CHUNK] = cos[np.arange(len(x)), index]
        best_index[k:k + CHUNK] = index
    return (best_uv, best_cos, best_index) if with_index else (best_uv, best_cos)


def _residual(camera, uv, X, jacobian):
    o, d, o_u, o_v, d_u, d_v = camera._rays(uv, derivatives=True)
    v = X - o
    rho = np.linalg.norm(v, axis=-1, keepdims=True)
    w = v / rho
    dw = np.sum(d * w, axis=-1, keepdims=True)
    r = w - d * dw
    if not jacobian:
        return r, dw[..., 0], rho[..., 0]
    columns = []
    for o_k, d_k in ((o_u, d_u), (o_v, d_v)):
        w_k = -(o_k - w * np.sum(w * o_k, axis=-1, keepdims=True)) / rho
        columns.append(w_k - d_k * dw - d * (np.sum(d_k * w, axis=-1, keepdims=True)
                                              + np.sum(d * w_k, axis=-1, keepdims=True)))
    return r, dw[..., 0], rho[..., 0], np.stack(columns, axis=-1)


def _iterate(camera, x, p, max_iterations):
    """Gauss–Newton from start pixels p (modified in place), steps limited to a quarter of the sensor and clipped to
    the domain."""
    lo, hi = camera._outer
    limit = 0.25 * float(np.max(camera.resolution))
    active = np.ones(len(x), bool)
    for _ in range(max_iterations):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        r, _, _, J = _residual(camera, p[idx], x[idx], jacobian=True)
        JtJ = np.einsum("nik,nil->nkl", J, J)
        Jtr = np.einsum("nik,ni->nk", J, r)
        JtJ += 1e-12 * np.trace(JtJ, axis1=1, axis2=2)[:, None, None] * np.eye(2) + 1e-300 * np.eye(2)
        step = -np.linalg.solve(JtJ, Jtr[..., None])[..., 0]
        step = np.where(np.isfinite(step), step, 0.0)
        length = np.linalg.norm(step, axis=-1, keepdims=True)
        step = np.where(length > limit, step * limit / np.maximum(length, 1e-300), step)
        p[idx] = np.clip(p[idx] + step, lo, hi)
        active[idx] = np.linalg.norm(step, axis=-1) >= STEP_TOLERANCE
    return p


def _judge(camera, x, p, cos):
    """The status of each converged pixel (before the domain test)."""
    r, along, rho = _residual(camera, p, x, jacobian=False)
    angle = np.linalg.norm(r, axis=-1)
    pixel = camera.pixel_angle(p)
    converged = np.isfinite(angle) & (angle <= RESIDUAL_TOLERANCE * np.maximum(pixel, 1e-15))
    result = np.where(converged, int(Status.OK), int(Status.NO_CONVERGENCE))
    result = np.where(converged & (along <= 0), int(Status.BEHIND), result)
    result = np.where(~converged & (cos <= 0), int(Status.BEHIND), result)
    return result, rho


def solve(camera, X, allow_extrapolation, max_iterations, allow_folded=False):
    shape = X.shape[:-1]
    X = X.reshape(-1, 3)
    uv = np.full((len(X), 2), np.nan)
    status = np.full(len(X), int(Status.NO_CONVERGENCE))
    finite = np.all(np.isfinite(X), axis=-1)
    if finite.any():
        x = X[finite]
        # the calibrated area's rays first: a solution there is the one to report; the extrapolated border's rays
        # (only with allow_extrapolation) start the points that found none – their continued polynomial can bend
        # back and offer false solutions far outside
        p, cos = _start(camera, x, False)
        p = _iterate(camera, x, p, max_iterations)
        result, rho = _judge(camera, x, p, cos)
        if allow_extrapolation:
            retry = np.nonzero((result != int(Status.OK)) | ~camera.hull.contains(p))[0]
            if len(retry):
                q, cos_q = _start(camera, x[retry], True)
                q = _iterate(camera, x[retry], q, max_iterations)
                result_q, rho_q = _judge(camera, x[retry], q, cos_q)
                better = (result_q == int(Status.OK)) & (result[retry] != int(Status.OK))
                p[retry[better]], result[retry[better]], rho[retry[better]] = q[better], result_q[better], rho_q[better]
        domain = camera._domain_status(p, allow_extrapolation)
        ok = result == int(Status.OK)
        result = np.where(ok & (domain != Status.OK), domain, result)
        fold = np.zeros(len(p), bool)
        found = np.all(np.isfinite(p), axis=-1)
        fold[found] = camera.folded(p[found])
        result = np.where((result == int(Status.OK)) & fold, int(Status.FOLDED), result)
        if len(camera._fold_starts):  # a second pixel in a fold that sees the point makes it ambiguous as well
            ok = np.nonzero(result == int(Status.OK))[0]
            if len(ok):
                q, cos_q = _start_from(camera, x[ok], camera._fold_starts)
                q = _iterate(camera, x[ok], q, max_iterations)
                result_q, _ = _judge(camera, x[ok], q, cos_q)
                # a fold can lie at the calibrated area's edge, its solution just outside the hull
                extent = float(np.max(camera.hull.vertices.max(axis=0) - camera.hull.vertices.min(axis=0)))
                second = (result_q == int(Status.OK)) & camera.folded(q) & \
                    camera.hull.contains(q, tolerance=FOLD_MARGIN * extent) & (np.linalg.norm(q - p[ok], axis=-1) > 0.5)
                result[ok[second]] = int(Status.FOLDED)
        if camera.min_unique_distance is not None and not camera.central:
            result = np.where((result == int(Status.OK)) & (rho < camera.min_unique_distance),
                              int(Status.AMBIGUOUS_NEAR), result)
        keep = (result == int(Status.OK)) | (result == int(Status.AMBIGUOUS_NEAR)) | \
               ((result == int(Status.OUTSIDE_HULL)) & bool(allow_extrapolation)) | \
               ((result == int(Status.FOLDED)) & bool(allow_folded))
        uv[finite] = np.where(keep[:, None], p, np.nan)
        status[finite] = result
    return uv.reshape(shape + (2,)), status.reshape(shape)
