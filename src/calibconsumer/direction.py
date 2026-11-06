# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Directions in azimuthal-equidistant (AE) coordinates about an axis (docs/format.md, "direction").

(a, b) ↦ d = R · (a·sin r / r, b·sin r / r, −cos r), r = |(a, b)|: r is the angle between d and the axis R·(0, 0, −1),
and (a, b) points the way d leans from it. The camera frame is x right, y up, viewing along −z.
"""

from __future__ import annotations

import numpy as np

_SMALL = 1e-4  # below this angle the series of sin r / r and its derivative are used


def _sinc_and_q(r: np.ndarray):
    """sin r / r and q = d/dr(sin r / r) / r = (r cos r − sin r) / r³, both regular at r = 0."""
    r2 = r * r
    small = r < _SMALL
    safe = np.where(small, 1.0, r)
    s = np.where(small, 1 - r2 / 6 + r2 * r2 / 120, np.sin(safe) / safe)
    q = np.where(small, -1.0 / 3 + r2 / 30, (safe * np.cos(safe) - np.sin(safe)) / safe ** 3)
    return s, q


def to_direction(ab: np.ndarray, rotation: np.ndarray, jacobian: bool = False):
    """Unit directions (..., 3) from AE coordinates (..., 2); with jacobian also d(direction)/d(a, b), (..., 3, 2)."""
    ab = np.asarray(ab, dtype=float)
    a, b = ab[..., 0], ab[..., 1]
    r = np.hypot(a, b)
    s, q = _sinc_and_q(r)
    local = np.stack([a * s, b * s, -np.cos(r)], axis=-1)
    d = local @ np.asarray(rotation, dtype=float).T
    if not jacobian:
        return d
    j_local = np.stack([np.stack([s + a * a * q, a * b * q], -1),
                        np.stack([a * b * q, s + b * b * q], -1),
                        np.stack([a * s, b * s], -1)], axis=-2)
    return d, np.einsum("ij,...jk->...ik", np.asarray(rotation, dtype=float), j_local)


def from_direction(d: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """AE coordinates (..., 2) of directions (..., 3, need not be unit) – atan2-based, exact near the axis."""
    local = np.asarray(d, dtype=float) @ np.asarray(rotation, dtype=float)
    x, y, z = local[..., 0], local[..., 1], local[..., 2]
    lateral = np.hypot(x, y)
    theta = np.arctan2(lateral, -z)
    scale = np.where(lateral > 0, theta / np.where(lateral > 0, lateral, 1.0), 0.0)
    return np.stack([x * scale, y * scale], axis=-1)


def axis_rotation(mean_direction: np.ndarray) -> np.ndarray:
    """A rotation R with R·(0, 0, −1) = the given direction (the smallest rotation from −z)."""
    target = np.asarray(mean_direction, dtype=float)
    target = target / np.linalg.norm(target)
    source = np.array([0.0, 0.0, -1.0])
    v = np.cross(source, target)
    c = float(source @ target)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)
