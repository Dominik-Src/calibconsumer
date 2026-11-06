# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""A view's pose from its measured pixels and their world points, for any camera of the package (central,
non-central, classic): the rays of the pixels, a linear start (a sample consensus over minimal sets, the camera taken
as central), then Gauss–Newton on the rays' angular residuals with robust weights.

Poses are (R, centre): R the rotation camera → world in the file's camera frame (x right, y up, looking along −z),
X_camera = Rᵀ(X − centre) – as world_to_pixel(pose=…) and camera.views take them."""

from __future__ import annotations

import numpy as np

TRIALS = 200
INLIER_PIXELS = 20.0  # the consensus' threshold: the angle of this many pixels
ITERATIONS = 50


def _skew(v):
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1), np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _rotation(omega):
    theta = float(np.linalg.norm(omega))
    if theta < 1e-15:
        return np.eye(3)
    K = _skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def _plane(X):
    """(centroid, Q 3 × 3 right-handed rows, planar): the world's principal axes; planar when the third axis' extent is
    below 1e-6 of the first's."""
    m = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - m, full_matrices=False)
    Q = Vt.copy()
    Q[2] = np.cross(Q[0], Q[1])
    return m, Q, s[2] <= 1e-6 * max(s[0], 1e-300)


def _linear(d, X, plane):
    """World → camera (A, t) with A X + t along d, from the homogeneous system [d]×(A X + t) = 0 on the world centred
    and scaled to unit size, or None."""
    S = _skew(d)
    m, Q, planar = plane
    size = max(float(np.sqrt(np.mean(np.sum((X - m) ** 2, -1)))), 1e-300)
    if planar:
        a = (X - m) @ Q[:2].T / size
        design = np.concatenate([np.einsum("nij,nk->nijk", S, a).reshape(len(d), 3, 6), S], -1)
    else:
        a = (X - m) / size
        design = np.concatenate([np.einsum("nij,nk->nijk", S, a).reshape(len(d), 3, 9), S], -1)
    try:
        _, _, Vt = np.linalg.svd(design.reshape(-1, design.shape[-1]), full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    v = Vt[-1]
    if planar:  # G = λ (the rotation's first two columns in the plane's axes), t' = λ (A m + t)
        G, t = v[:6].reshape(3, 2), v[6:]
        U, s, Wt = np.linalg.svd(G, full_matrices=False)
        M2 = U @ Wt
        scale = s.mean()
    else:
        G, t = v[:9].reshape(3, 3), v[9:]
        if np.linalg.det(G) < 0:
            G, t = -G, -t
        U, s, Wt = np.linalg.svd(G)
        scale = s.mean()
    candidates = []
    for sign in (1.0, -1.0):  # the null vector's sign: the points in front decide
        if planar:
            M = np.column_stack([sign * M2, np.cross(M2[:, 0], M2[:, 1])])
        elif sign < 0:
            continue
        else:
            M = U @ Wt
        A = (M @ Q) if planar else M
        t_world = sign * t / scale * size - A @ m
        if np.all(np.isfinite(A)) and np.all(np.isfinite(t_world)):
            candidates.append((int(np.sum(np.sum(d * (X @ A.T + t_world), -1) > 0)), A, t_world))
    if not candidates:
        return None
    front, A, t = max(candidates, key=lambda c: c[0])
    return (A, t) if front >= 0.5 * len(d) else None


def _angles(A, t, o, d, X):
    w = X @ A.T + t - o
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    return np.arctan2(np.linalg.norm(np.cross(d, w), axis=-1), np.sum(d * w, -1))


def _residual(R, c, o, d, X):
    w = (X - c) @ R - o
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    return w - d * np.sum(d * w, -1, keepdims=True)


def estimate(camera, uv, X, seed: int = 0):
    """The pose (R camera → world, centre) of a view from its pixels uv (n, 2) and world points X (n, 3), and a dict
    with `inliers` (the consensus' and the final weights' view: angle below 20 px), `rmsAngle` (rad, of those) and
    `points`. None where it cannot be estimated (too few points, no consensus)."""
    uv, X = np.asarray(uv, dtype=float), np.asarray(X, dtype=float)
    o, d = camera.pixel_to_ray(uv, allow_extrapolation=True)
    F = getattr(camera, "_F", np.eye(3))
    o, d = o @ F, d @ F  # the file's frame, the poses' frame
    ok = np.all(np.isfinite(d), -1) & np.all(np.isfinite(o), -1) & np.all(np.isfinite(X), -1)
    o, d, X, uv = o[ok], d[ok], X[ok], uv[ok]
    plane = _plane(X) if len(X) >= 3 else None
    need = 5 if plane is not None and plane[2] else 6
    if len(X) < need:
        return None
    angle = np.nanmedian(camera.pixel_angle(uv))
    threshold = INLIER_PIXELS * float(angle)
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    candidates = [np.arange(len(X))] + [rng.choice(len(X), need, replace=False) for _ in range(TRIALS)]
    for sample in candidates:
        solution = _linear(d[sample], X[sample], plane)
        if solution is None:
            continue
        count = int(np.sum(_angles(*solution, np.zeros_like(o), d, X) < threshold))
        if count > best_count:
            best, best_count = solution, count
    if best is None:
        return None
    inliers = _angles(*best, np.zeros_like(o), d, X) < threshold
    if inliers.sum() >= need:
        best = _linear(d[inliers], X[inliers], plane) or best
    A, t = best
    R, c = A.T, -A.T @ t
    for _ in range(ITERATIONS):  # Gauss–Newton, Cauchy weights at 2.5 robust σ of the residuals
        r = _residual(R, c, o, d, X)
        size = np.linalg.norm(r, axis=-1)
        scale = max(2.5 * 1.4826 * float(np.median(size)), 1e-3 * threshold)
        weight = 1.0 / (1.0 + (size / scale) ** 2)
        J = np.empty((len(X), 3, 6))
        eps = 1e-7
        for k in range(3):
            step = np.zeros(3)
            step[k] = eps
            J[:, :, k] = (_residual(R @ _rotation(step), c, o, d, X) - r) / eps
            J[:, :, 3 + k] = (_residual(R, c + step * max(1.0, float(np.linalg.norm(c))), o, d, X) - r) / \
                (eps * max(1.0, float(np.linalg.norm(c))))
        Jw = J * np.sqrt(weight)[:, None, None]
        rw = r * np.sqrt(weight)[:, None]
        H = np.einsum("nik,nil->kl", Jw, Jw)
        g = np.einsum("nik,ni->k", Jw, rw)
        try:
            delta = -np.linalg.solve(H + 1e-12 * np.trace(H) * np.eye(6), g)
        except np.linalg.LinAlgError:
            break
        R, c = R @ _rotation(delta[:3]), c + delta[3:]
        if np.linalg.norm(delta[:3]) < 1e-12 and np.linalg.norm(delta[3:]) < 1e-12 * max(1.0, np.linalg.norm(c)):
            break
    w = (X - c) @ R - o
    angles = np.arctan2(np.linalg.norm(np.cross(d, w), axis=-1), np.sum(d * w, -1))
    inliers = angles < threshold
    return (R, c), {"points": int(len(X)), "inliers": int(inliers.sum()),
                    "rmsAngle": float(np.sqrt(np.mean(angles[inliers] ** 2))) if inliers.any() else None}
