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
calibconsumer.remap for the poles. Pixels the camera does not see (outside its calibrated area) are the background.

The images on the sphere come two ways (`method`): "outline" maps only the outlines of the shapes – the checkerboard's
squares clipped to the calibrated area, the area itself – forward (pixel → ray, closed form), densified until
neighbouring points lie at most 1.5 sphere pixels apart, and fills them by scanline (even-odd); "maps" solves every
sphere pixel back to the sensor (calibconsumer.remap, world → pixel). "auto" (the default) takes the outlines for a
camera with a polygonal calibrated area whose direction map does not fold, else the maps (the classic models, folded
cameras: an outline is no outline there)."""

from __future__ import annotations

import numpy as np

from .remap import directions_maps, sphere_angles, sphere_directions

BACKGROUND = (64, 64, 64)
LIGHT, DARK = (235, 235, 235), (40, 90, 160)
GRID = (255, 176, 0)
COLOURS = ((230, 97, 1), (94, 60, 153), (27, 158, 119), (217, 95, 2), (117, 112, 179), (231, 41, 138))


def _image(shape, colour):
    return np.broadcast_to(np.asarray(colour, dtype=np.uint8), shape + (3,)).copy()


MAX_GAP = 1.5     # sphere pixels between neighbouring outline points
DENSIFY = 14      # halvings at most


def _outlines_apply(camera, method):
    """Whether the outline way applies (method "auto"/"outline"/"maps")."""
    if method not in ("auto", "outline", "maps"):
        raise ValueError(f"method must be 'auto', 'outline' or 'maps', not {method!r}")
    smooth = hasattr(getattr(camera, "hull", None), "vertices") and hasattr(camera, "document") \
        and (camera.document.get("inverse") or {}).get("injective") is not False
    if method == "outline" and not smooth:
        raise ValueError("outlines need a camera with a polygonal calibrated area whose direction map does not fold")
    return method == "outline" or (method == "auto" and smooth)


def _clip(polygon, hull):
    """A convex polygon (n, 2) clipped to a convex hull (counter-clockwise vertices; Sutherland–Hodgman)."""
    out = [np.asarray(p, dtype=float) for p in polygon]
    for a, e in zip(hull.vertices, hull.edges):
        if not out:
            break
        side = [e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) for p in out]
        kept = []
        for k in range(len(out)):
            p, q, sp, sq = out[k], out[(k + 1) % len(out)], side[k], side[(k + 1) % len(out)]
            if sp >= 0:
                kept.append(p)
            if (sp >= 0) != (sq >= 0):
                kept.append(p + sp / (sp - sq) * (q - p))
        out = kept
    return np.asarray(out).reshape(-1, 2)


def _to_sphere(camera, uv, size, pole, rotation, distance):
    """Sphere pixel coordinates (x, y; pixel centres at integers) of sensor pixels uv (n, 2)."""
    from .remap import sphere_angles
    o, d = camera.pixel_to_ray(uv, allow_extrapolation=True)
    o, d = np.asarray(o, dtype=float), np.asarray(d, dtype=float)
    if not getattr(camera, "central", True):
        if distance is None:
            raise ValueError("a non-central camera needs the scene distance")
        od = np.sum(o * d, axis=-1)
        t = -od + np.sqrt(np.maximum(od ** 2 - np.sum(o * o, axis=-1) + distance ** 2, 0.0))
        d = o + t[:, None] * d
    if rotation is not None:
        d = d @ np.asarray(rotation, dtype=float)  # the camera's frame into the sphere's
    a, b = sphere_angles(d, pole)
    W, H = size
    return (a + np.pi) / (2 * np.pi) * W - 0.5, ((np.pi / 2 - b) if pole == "up" else b) / np.pi * H - 0.5


def _outline(camera, polygon, size, pole, rotation, distance):
    """A sensor polygon's outline on the sphere, densified (MAX_GAP), unwrapped across the panorama's seam and closed
    through a pole it surrounds: (x, y) or None."""
    W, H = size
    pts = np.asarray(polygon, dtype=float)
    for _ in range(DENSIFY):
        x, y = _to_sphere(camera, pts, size, pole, rotation, distance)
        ok = np.isfinite(x) & np.isfinite(y)
        pts, x, y = pts[ok], x[ok], y[ok]
        if len(pts) < 3:
            return None
        nxt = np.roll(np.arange(len(pts)), -1)
        dx = (x[nxt] - x + W / 2) % W - W / 2
        far = np.hypot(dx, y[nxt] - y) > MAX_GAP
        if not far.any():
            break
        pts = np.insert(pts, np.nonzero(far)[0] + 1, (pts[far] + pts[nxt[far]]) / 2, axis=0)
    x, y = _to_sphere(camera, pts, size, pole, rotation, distance)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    xu = x[0] + np.concatenate([[0.0], np.cumsum((np.diff(x) + W / 2) % W - W / 2)])
    closing = (x[0] - x[-1] + W / 2) % W - W / 2
    winding = xu[-1] + closing - xu[0]
    if abs(winding) > W / 2:  # around a pole: closed along the image's edge on the pole's side
        edge = -1.0 if y.min() < (H - 1) - y.max() else float(H)
        end = xu[-1] + closing
        xu = np.concatenate([xu, [end, end, xu[0]]])
        y = np.concatenate([y, [y[0], edge, edge]])
    return xu, y


def _fill(size, outlines):
    """The pixels (centres) inside the outlines by the even-odd rule, each outline also shifted by ± the width."""
    W, H = size
    cross = np.zeros((H, W + 1), dtype=np.int32)
    for px, py in outlines:
        for shift in (-W, 0.0, W):
            x0, y0 = px + shift, py
            x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
            first = np.clip(np.ceil(np.minimum(y0, y1)), 0, H).astype(int)
            last = np.clip(np.ceil(np.maximum(y0, y1)) - 1, -1, H - 1).astype(int)
            n = np.maximum(last - first + 1, 0)
            if n.sum() == 0:
                continue
            e = np.repeat(np.arange(len(x0)), n)
            rows = np.repeat(first, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            t = (rows - y0[e]) / (y1[e] - y0[e])
            cols = np.clip(np.ceil(x0[e] + t * (x1[e] - x0[e])), 0, W).astype(int)
            np.add.at(cross, (rows, cols), 1)
    return np.cumsum(cross, axis=1)[:, :W] % 2 == 1


def sensor_on_sphere(camera, size=(1024, 512), squares: int = 16, pole: str = "up", rotation=None,
                     distance: float | None = None, step: int = 4, method: str = "auto"):
    """The sensor's checkerboard (squares along its longer side) on the sphere: an equirectangular image (size = (width,
    height)) – each sphere pixel coloured by the square of the camera pixel that sees it, the background where none.
    `method`: see the module's notes (the outlines: the calibrated area dark, the light squares on top)."""
    width, height = camera.resolution
    edge = max(width, height) / squares
    if _outlines_apply(camera, method):
        out = _image((size[1], size[0]), BACKGROUND)
        area = _outline(camera, camera.hull.vertices, size, pole, rotation, distance)
        if area is None:
            return out
        out[_fill(size, [area])] = DARK
        light = []
        for i in range(int(np.ceil(width / edge))):
            for j in range(int(np.ceil(height / edge))):
                if (i + j) % 2:
                    continue
                # the square u in [i·edge, (i + 1)·edge), v likewise (the maps' floor(u / edge))
                square = np.array([[i * edge, j * edge], [(i + 1) * edge, j * edge],
                                   [(i + 1) * edge, (j + 1) * edge], [i * edge, (j + 1) * edge]])
                part = _clip(square, camera.hull)
                if len(part) >= 3:
                    outline = _outline(camera, part, size, pole, rotation, distance)
                    if outline is not None:
                        light.append(outline)
        out[_fill(size, light) & np.all(out == np.array(DARK, np.uint8), axis=-1)] = LIGHT
        return out
    d = sphere_directions(size, pole)
    if rotation is not None:
        d = d @ np.asarray(rotation, dtype=float).T
    map_x, map_y = directions_maps(camera, d, distance, step)
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
                       step: int = 4, colours=COLOURS, method: str = "auto"):
    """Where the cameras see on one sphere (equirectangular, size = (width, height)): each camera's coverage in its
    colour, overlaps the mean of their colours and brighter, the background elsewhere. rotations: per camera the
    direction of the sphere's frame into its own (a rig's reference camera None, the others the transposed relative
    rotation) – the rig kept as it is; returns (image, count) with the number of cameras seeing each pixel."""
    rotations = rotations or [None] * len(cameras)
    d = None
    total = np.zeros((size[1], size[0], 3), dtype=float)
    count = np.zeros((size[1], size[0]), dtype=int)
    for k, (camera, rotation) in enumerate(zip(cameras, rotations)):
        if _outlines_apply(camera, method):  # the calibrated area's outline, filled
            area = _outline(camera, camera.hull.vertices, size, pole, rotation, distance)
            seen = _fill(size, [area]) if area is not None else np.zeros_like(count, dtype=bool)
        else:
            d = sphere_directions(size, pole) if d is None else d
            dk = d if rotation is None else d @ np.asarray(rotation, dtype=float).T
            map_x, _ = directions_maps(camera, dk, distance, step)
            seen = map_x >= 0
        total[seen] += colours[k % len(colours)]
        count[seen] += 1
    out = _image((size[1], size[0]), BACKGROUND)
    any_ = count > 0
    mean = total[any_] / count[any_, None]
    boost = np.where(count[any_] > 1, 1.25, 1.0)[:, None]
    out[any_] = np.clip(mean * boost, 0, 255).astype(np.uint8)
    return out, count
