# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Stereo rectification of a pair of calibrated cameras of any model (OpenCV's stereoRectify for generic cameras): both
cameras turned to one orientation whose x axis lies along the baseline, so that a scene point is on the same row of both
rectified images (on the same column with vertical=True), each a virtual pinhole of one focal length. The relative pose
is the one relativeOrientations.json gives a pair (the package's frame): x₁ = R x₂ + t, t camera 2's centre in camera 1's
frame. Non-central cameras: their rays meet the pinhole's at the scene distance `distance` – exact at that distance
only."""

from __future__ import annotations

import numpy as np

from .remap import directions_maps


def rectifying_rotation(rotation, translation, vertical: bool | None = None):
    """(R_rect1, vertical): the rectified frame in camera 1's frame (columns x, y, z; looking along −z) – x along the
    baseline (y with vertical, chosen by the baseline's larger in-image component when None), turned towards camera 1's
    own x (y) so the picture keeps its way up, z against the mean of both cameras' viewing directions."""
    R = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float)
    b = t / np.linalg.norm(t)
    if vertical is None:
        vertical = bool(abs(b[1]) > abs(b[0]))
    forward = np.array([0.0, 0.0, -1.0]) + R @ np.array([0.0, 0.0, -1.0])
    forward /= np.linalg.norm(forward)
    if vertical:
        y = b if b[1] >= 0 else -b
        z = -(forward - (forward @ y) * y)
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
    else:
        x = b if b[0] >= 0 else -b
        z = -(forward - (forward @ x) * x)
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
    return np.stack([x, y, z], axis=1), vertical


def rectify(camera1, camera2, rotation, translation, size=(1280, 960), focal_px: float | None = None,
            vertical: bool | None = None, distance: float | None = None, step: int = 4):
    """The rectification of a pair: {"rotation1", "rotation2" (rectified frame → camera k), "focal", "size",
    "principalPoint", "vertical", "baseline", "maps1", "maps2" (map_x, map_y for cv2.remap of camera k's image)}. focal_px
    defaults to camera 1's at its image centre (1 / its pixel angle there)."""
    R = np.asarray(rotation, dtype=float)
    rect1, vertical = rectifying_rotation(R, translation, vertical)
    rect2 = R.T @ rect1
    width, height = size
    if focal_px is None:
        W, H = camera1.resolution
        focal_px = 1.0 / float(np.asarray(camera1.pixel_angle(np.array([(W - 1) / 2, (H - 1) / 2]))))
    cx, cy = (width - 1) / 2, (height - 1) / 2
    u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    d = np.stack([(u - cx) / focal_px, -(v - cy) / focal_px, -np.ones_like(u)], -1)
    return {"rotation1": rect1, "rotation2": rect2, "focal": focal_px, "size": (width, height),
            "principalPoint": (cx, cy), "vertical": vertical, "baseline": float(np.linalg.norm(translation)),
            "maps1": directions_maps(camera1, d @ rect1.T, distance, step),
            "maps2": directions_maps(camera2, d @ rect2.T, distance, step)}
