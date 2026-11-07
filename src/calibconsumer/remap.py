# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Remap tables for cv2.remap (OpenCV itself is not a dependency): any image of the camera's viewing directions – a
virtual pinhole, the sphere as an equirectangular panorama, the six faces of a cube map – as map_x, map_y in the
camera's pixels. Directions are in the camera's frame (the package's: x right, y up, looking along −z, unless the camera
was turned to another with with_frame()); unmapped pixels are −1 (cv2.remap's border)."""

from __future__ import annotations

import numpy as np

# the six faces of a cube map: forward, right and up of each face in the camera's frame (looking along −z)
CUBE_FACES = {
    "front": ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
    "right": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "back": ((0, 0, 1), (-1, 0, 0), (0, 1, 0)),
    "left": ((-1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "up": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "down": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
}


def _solve(camera, directions, distance, allow_extrapolation):
    """The pixels (…, 2) of directions (…, 3); nan where the camera sees none of them."""
    d = np.asarray(directions, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    if not getattr(camera, "central", True):
        if distance is None:
            raise ValueError("a non-central camera needs the scene distance")
        d = d * distance
    uv = np.asarray(camera.world_to_pixel(d, allow_extrapolation=allow_extrapolation), dtype=float)
    return np.where(np.isfinite(uv).all(axis=-1, keepdims=True), uv, np.nan)


def directions_maps(camera, directions, distance: float | None = None, step: int = 1,
                    allow_extrapolation: bool = False):
    """map_x, map_y (float32, the shape of `directions` without its last axis) of an image whose pixels look along
    `directions` (h, w, 3, the camera's frame): the camera's pixel each direction is seen at, −1 where none. With step > 1
    every step-th pixel (and the last row and column) is solved and the others interpolated bilinearly; a pixel whose cell
    the camera sees at some corners but not all, or whose interpolated pixel it does not contain, is solved exactly – the
    edges of the calibrated area, of a fold and of the field of view stay exact; a cell seen at no corner stays unmapped
    (a feature narrower than `step` pixels of the image can be missed there). A non-central camera's rays meet the directions at the
    scene distance `distance` from its origin."""
    d = np.asarray(directions, dtype=float)
    h, w = d.shape[:2]
    if step <= 1 or h < 3 or w < 3:
        uv = _solve(camera, d, distance, allow_extrapolation)
    else:
        ys = np.unique(np.r_[np.arange(0, h, step), h - 1])
        xs = np.unique(np.r_[np.arange(0, w, step), w - 1])
        coarse = _solve(camera, d[np.ix_(ys, xs)], distance, allow_extrapolation)
        iy = np.clip(np.searchsorted(ys, np.arange(h), side="right") - 1, 0, len(ys) - 2)
        ix = np.clip(np.searchsorted(xs, np.arange(w), side="right") - 1, 0, len(xs) - 2)
        ty = ((np.arange(h) - ys[iy]) / (ys[iy + 1] - ys[iy]))[:, None, None]
        tx = ((np.arange(w) - xs[ix]) / (xs[ix + 1] - xs[ix]))[None, :, None]
        top = (1 - tx) * coarse[iy][:, ix] + tx * coarse[iy][:, ix + 1]
        bottom = (1 - tx) * coarse[iy + 1][:, ix] + tx * coarse[iy + 1][:, ix + 1]
        uv = (1 - ty) * top + ty * bottom
        finite = np.isfinite(uv).all(axis=-1)
        # a cell whose corners the camera sees in part lies at the edge of what it sees: its pixels solved exactly; a
        # cell it sees at no corner stays unmapped (no direction solved in vain – unseen directions are the slow ones)
        seen = np.isfinite(coarse).all(axis=-1).astype(int)
        corners = seen[iy][:, ix] + seen[iy][:, ix + 1] + seen[iy + 1][:, ix] + seen[iy + 1][:, ix + 1]
        exact = ~finite & (corners > 0)
        if finite.any():
            inside = np.zeros_like(finite)
            inside[finite] = np.asarray(camera.contains(uv[finite]), dtype=bool)
            exact |= finite & ~inside
        if exact.any():
            uv[exact] = _solve(camera, d[exact], distance, allow_extrapolation)
    uv = np.where(np.isfinite(uv), uv, -1.0)
    return uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)


def pinhole_maps(camera, focal_px: float, size, rotation=None, distance: float | None = None, step: int = 1):
    """map_x, map_y (float32, shape (height, width)) for cv2.remap: the image of a virtual pinhole camera with focal
    length focal_px (pixels), principal point at the centre of `size` = (width, height), looking along the camera's
    optical axis (or turned by `rotation`, OpenCV frame). Non-central cameras need the scene distance `distance` (the
    pinhole's rays are intersected at that distance). Unmapped pixels are −1. step > 1: see directions_maps()."""
    width, height = size
    x, y = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rays = np.stack([(x - (width - 1) / 2) / focal_px, (y - (height - 1) / 2) / focal_px, np.ones_like(x)], -1)
    if rotation is not None:
        rays = rays @ np.asarray(rotation, dtype=float).T
    return directions_maps(camera.with_frame("opencv"), rays, distance, step)


def sphere_directions(size, pole: str = "up"):
    """The viewing directions (h, w, 3) of an equirectangular image of size = (width, height): pole "up" – longitude
    across (−180° … 180°, 0 straight ahead along −z), latitude up the rows (90° at the top, the camera's +y), the
    panorama's horizon through the optical axis; pole "axis" – the polar angle from the optical axis down the rows (0 at
    the top row, 180° at the bottom), the azimuth about the axis across (0 = +x) – each row a circle of one field angle."""
    width, height = size
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    if pole == "up":
        lon, lat = np.meshgrid(2 * np.pi * u - np.pi, np.pi / 2 - np.pi * v)
        return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), -np.cos(lat) * np.cos(lon)], -1)
    if pole == "axis":
        azimuth, theta = np.meshgrid(2 * np.pi * u - np.pi, np.pi * v)
        return np.stack([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), -np.cos(theta)], -1)
    raise ValueError(f"pole must be 'up' or 'axis', not {pole!r}")


def sphere_angles(directions, pole: str = "up"):
    """(longitude, latitude) of sphere_directions(pole="up") or (azimuth, polar angle) of pole="axis" – radians – for
    directions (…, 3); the inverse of sphere_directions()."""
    d = np.asarray(directions, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    if pole == "up":
        return np.arctan2(d[..., 0], -d[..., 2]), np.arcsin(np.clip(d[..., 1], -1, 1))
    if pole == "axis":
        return np.arctan2(d[..., 1], d[..., 0]), np.arccos(np.clip(-d[..., 2], -1, 1))
    raise ValueError(f"pole must be 'up' or 'axis', not {pole!r}")


def sphere_maps(camera, size=(2048, 1024), pole: str = "up", rotation=None, distance: float | None = None,
                step: int = 4):
    """map_x, map_y (float32, (height, width)) of the camera's image on the sphere as an equirectangular panorama
    (sphere_directions()); `rotation` takes a direction of the sphere's frame into the camera's (for a rig: the
    reference camera's frame, a camera's rotation the transposed relative rotation); −1 where the camera sees nothing."""
    d = sphere_directions(size, pole)
    if rotation is not None:
        d = d @ np.asarray(rotation, dtype=float).T
    return directions_maps(camera, d, distance, step)


def cube_directions(face_size: int, face: str):
    """The viewing directions (n, n, 3) of one face of a cube map (CUBE_FACES: a 90° pinhole), row 0 at the face's top."""
    forward, right, up = (np.asarray(v, dtype=float) for v in CUBE_FACES[face])
    t = (np.arange(face_size) + 0.5) / face_size * 2 - 1
    a, b = np.meshgrid(t, -t)
    return forward + a[..., None] * right + b[..., None] * up


def cube_maps(camera, face_size: int = 1024, rotation=None, distance: float | None = None, step: int = 4):
    """{face: (map_x, map_y)} of the six faces of a cube map (CUBE_FACES), each a 90° pinhole of face_size² pixels – a
    "pinhole-like normalized" picture of any field of view; `rotation` as in sphere_maps()."""
    out = {}
    for face in CUBE_FACES:
        d = cube_directions(face_size, face)
        if rotation is not None:
            d = d @ np.asarray(rotation, dtype=float).T
        out[face] = directions_maps(camera, d, distance, step)
    return out
