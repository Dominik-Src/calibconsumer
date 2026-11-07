# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""A camera's mapping at a glance: images that need no photograph (RGB, uint8, numpy only).

- sensor_on_sphere(): a checkerboard on the sensor, seen on the sphere (an equirectangular panorama) – where each part
  of the image looks.
- sphere_on_sensor(): the sphere seen by the camera – a latitude-longitude grid, a checkerboard of the sphere or any
  equirectangular texture (an earth map, say) drawn on the sensor.
- coverage_on_sphere(): several cameras on one sphere, each in its colour, overlaps blended – a rig kept in its
  relative orientation (each camera's `rotation`).

`rotation` takes a direction of the sphere's frame into the camera's frame (None: the camera's own); see
calibconsumer.remap for the poles. Pixels the camera does not see (outside its calibrated area) are the background."""

from __future__ import annotations

import numpy as np

from .remap import directions_maps, sphere_angles, sphere_directions

BACKGROUND = (64, 64, 64)
LIGHT, DARK = (235, 235, 235), (40, 90, 160)
GRID = (255, 176, 0)
COLOURS = ((230, 97, 1), (94, 60, 153), (27, 158, 119), (217, 95, 2), (117, 112, 179), (231, 41, 138))


def _image(shape, colour):
    return np.broadcast_to(np.asarray(colour, dtype=np.uint8), shape + (3,)).copy()


def sensor_on_sphere(camera, size=(1024, 512), squares: int = 16, pole: str = "up", rotation=None,
                     distance: float | None = None, step: int = 4):
    """The sensor's checkerboard (squares along its longer side) on the sphere: an equirectangular image (size = (width,
    height)) – each sphere pixel coloured by the square of the camera pixel that sees it, the background where none."""
    d = sphere_directions(size, pole)
    if rotation is not None:
        d = d @ np.asarray(rotation, dtype=float).T
    map_x, map_y = directions_maps(camera, d, distance, step)
    width, height = camera.resolution
    edge = max(width, height) / squares
    seen = map_x >= 0
    parity = (np.floor(map_x / edge) + np.floor(map_y / edge)) % 2 == 0
    out = _image(map_x.shape, BACKGROUND)
    out[seen & parity] = LIGHT
    out[seen & ~parity] = DARK
    return out


def sphere_on_sensor(camera, width: int = 1024, grid_deg: float = 10.0, pole: str = "axis", rotation=None,
                     texture=None, checkerboard: bool = False, line_px: float = 1.5):
    """The sphere drawn on the sensor (an image of the sensor's aspect, `width` pixels wide): a latitude-longitude grid
    every grid_deg degrees (lines line_px pixels wide; with pole "axis" the circles of equal field angle and the radial
    lines about the optical axis, with pole "up" a horizon and meridians), a checkerboard of the sphere (checkerboard
    True), or an equirectangular `texture` (h, w, 3 – an earth map, pole "up"); the background outside the calibrated
    area. Non-central cameras: their ray directions."""
    W, H = camera.resolution
    scale = width / W
    height = max(1, round(H * scale))
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    uv = np.stack([(x + 0.5) / scale - 0.5, (y + 0.5) / scale - 0.5], -1).astype(float)
    rays = camera.pixel_to_ray(uv)
    d = np.asarray(rays[1] if isinstance(rays, tuple) else rays, dtype=float)
    seen = np.isfinite(d).all(axis=-1)
    out = _image((height, width), BACKGROUND)
    if not seen.any():
        return out
    ds = d[seen]
    if rotation is not None:
        ds = ds @ np.asarray(rotation, dtype=float)  # the camera's frame into the sphere's: the transposed rotation
    if texture is not None:
        lon, lat = sphere_angles(ds, "up")
        tex = np.asarray(texture)
        th, tw = tex.shape[:2]
        col = np.clip(((lon + np.pi) / (2 * np.pi) * tw).astype(int), 0, tw - 1)
        row = np.clip(((np.pi / 2 - lat) / np.pi * th).astype(int), 0, th - 1)
        out[seen] = tex[row, col, :3]
        return out
    a, b = sphere_angles(ds, pole)
    step = np.radians(grid_deg)
    if checkerboard:
        parity = (np.floor(a / step) + np.floor(b / step)) % 2 == 0
        out[seen] = np.where(parity[:, None], np.array(LIGHT, np.uint8), np.array(DARK, np.uint8))
        return out
    out[seen] = LIGHT
    # a line where the angle is within line_px output pixels of a multiple of the grid's step; the second angle's lines
    # (longitude, azimuth) shrink with the cosine of latitude (pole up) or the sine of the polar angle (pole axis)
    angle = np.asarray(camera.pixel_angle(uv[seen]), dtype=float) / scale * line_px
    near = lambda value: np.abs((value + step / 2) % step - step / 2)  # noqa: E731
    across = np.cos(b) if pole == "up" else np.sin(b)
    line = (near(b) < angle) | (near(a) * np.abs(across) < angle)
    picked = out[seen]
    picked[line] = GRID
    out[seen] = picked
    return out


def coverage_on_sphere(cameras, rotations=None, size=(1024, 512), pole: str = "up", distance: float | None = None,
                       step: int = 4, colours=COLOURS):
    """Where the cameras see on one sphere (equirectangular, size = (width, height)): each camera's coverage in its
    colour, overlaps the mean of their colours and brighter, the background elsewhere. rotations: per camera the
    direction of the sphere's frame into its own (a rig's reference camera None, the others the transposed relative
    rotation) – the rig kept as it is; returns (image, count) with the number of cameras seeing each pixel."""
    rotations = rotations or [None] * len(cameras)
    d = sphere_directions(size, pole)
    total = np.zeros(d.shape[:2] + (3,), dtype=float)
    count = np.zeros(d.shape[:2], dtype=int)
    for k, (camera, rotation) in enumerate(zip(cameras, rotations)):
        dk = d if rotation is None else d @ np.asarray(rotation, dtype=float).T
        map_x, _ = directions_maps(camera, dk, distance, step)
        seen = map_x >= 0
        total[seen] += colours[k % len(colours)]
        count[seen] += 1
    out = _image(d.shape[:2], BACKGROUND)
    any_ = count > 0
    mean = total[any_] / count[any_, None]
    boost = np.where(count[any_] > 1, 1.25, 1.0)[:, None]
    out[any_] = np.clip(mean * boost, 0, 255).astype(np.uint8)
    return out, count
