# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Optional helpers for OpenCV users (OpenCV itself is not a dependency)."""

from __future__ import annotations

import numpy as np

from .camera import Camera


def pinhole_maps(camera: Camera, focal_px: float, size, rotation=None, distance: float | None = None):
    """map_x, map_y (float32, shape (height, width)) for cv2.remap: the image of a virtual pinhole camera with focal
    length focal_px (pixels), principal point at the centre of `size` = (width, height), looking along the camera's
    optical axis (or turned by `rotation`, OpenCV frame). Non-central cameras need the scene distance `distance` (the
    pinhole's rays are intersected at that distance). Unmapped pixels are −1."""
    width, height = size
    x, y = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rays = np.stack([(x - (width - 1) / 2) / focal_px, (y - (height - 1) / 2) / focal_px, np.ones_like(x)], -1)
    if rotation is not None:
        rays = rays @ np.asarray(rotation, dtype=float).T
    if not camera.central:
        if distance is None:
            raise ValueError("a non-central camera needs the scene distance")
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True) * distance
    cv = camera.with_frame("opencv")
    uv = cv.world_to_pixel(rays)
    uv = np.where(np.isfinite(uv), uv, -1.0)
    return uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
