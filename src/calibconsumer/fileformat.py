# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Reading, validating and writing mapping.json (format "calibconsumer-ray-mapping", version 1; docs/format.md)."""

from __future__ import annotations

import json
import math

import numpy as np

FORMAT = "calibconsumer-ray-mapping"
FORMAT_VERSION = 1


class FormatError(ValueError):
    """A mapping file that this version cannot read."""


def _require(condition, message):
    if not condition:
        raise FormatError(message)


def _controls(block, key, intervals, components):
    n, m = intervals
    values = np.asarray(block.get(key), dtype=float)
    _require(values.size == (m + 3) * (n + 3) * components,
             f"{key}: {values.size} numbers, expected (m + 3)·(n + 3)·{components} = {(m + 3) * (n + 3) * components}")
    _require(np.all(np.isfinite(values)), f"{key}: not every control value is finite")
    return values.reshape(m + 3, n + 3, components)


def validate(document: dict) -> dict:
    """The document's model in arrays – raises FormatError on anything this reader does not accept."""
    _require(isinstance(document, dict), "not a JSON object")
    _require(document.get("format") == FORMAT, f"format is {document.get('format')!r}, expected {FORMAT!r}")
    version = document.get("formatVersion")
    _require(isinstance(version, int) and version == FORMAT_VERSION,
             f"formatVersion {version!r}: this reader understands version {FORMAT_VERSION}")
    camera = document.get("camera") or {}
    resolution = camera.get("resolution")
    _require(isinstance(resolution, list) and len(resolution) == 2 and all(int(r) > 0 for r in resolution),
             "camera.resolution must be [width, height]")
    spline = document.get("spline") or {}
    _require(spline.get("type") == "uniform-cubic-bspline", f"spline.type {spline.get('type')!r} is not supported")
    intervals = spline.get("intervals")
    _require(isinstance(intervals, list) and len(intervals) == 2 and all(isinstance(k, int) and k >= 1
                                                                          for k in intervals),
             "spline.intervals must be [n, m], both ≥ 1")
    domain = document.get("domain") or {}
    rectangle = np.asarray(domain.get("rectangle"), dtype=float)
    _require(rectangle.shape == (2, 2) and np.all(rectangle[1] > rectangle[0]),
             "domain.rectangle must be [[u0, v0], [u1, v1]] with u1 > u0, v1 > v0")
    hull = np.asarray(domain.get("hull"), dtype=float)
    _require(hull.ndim == 2 and hull.shape[1] == 2 and len(hull) >= 3, "domain.hull needs ≥ 3 [u, v] vertices")
    direction = document.get("direction") or {}
    _require(direction.get("parametrization") == "azimuthal-equidistant",
             f"direction.parametrization {direction.get('parametrization')!r} is not supported")
    rotation = np.asarray(direction.get("axisRotation"), dtype=float)
    _require(rotation.shape == (3, 3) and np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
             and np.linalg.det(rotation) > 0, "direction.axisRotation must be a 3 × 3 rotation")
    model = {"resolution": tuple(int(r) for r in resolution), "rectangle": rectangle, "hull": hull,
             "intervals": tuple(intervals), "rotation": rotation,
             "direction": _controls(direction, "controls", intervals, 2)}
    central = document.get("central")
    _require(isinstance(central, bool), "central must be true or false")
    origin = document.get("origin")
    if central:
        _require(origin is None, "a central camera has no origin block")
        model["origin"] = None
    else:
        _require(isinstance(origin, dict), "a non-central camera needs an origin block")
        kind = origin.get("parametrization")
        if kind == "foot-point":
            model["origin"] = ("foot-point", _controls(origin, "controls", intervals, 3))
        elif kind == "two-surface":
            distances = origin.get("distances")
            _require(isinstance(distances, list) and len(distances) == 2 and distances[1] > distances[0],
                     "origin.distances must be [Z1, Z2] with Z2 > Z1")
            model["origin"] = ("two-surface", _controls(origin, "controlsNear", intervals, 3),
                               _controls(origin, "controlsFar", intervals, 3), tuple(map(float, distances)))
        else:
            raise FormatError(f"origin.parametrization {kind!r} is not supported")
    inverse = document.get("inverse") or {}
    start = inverse.get("startRays", [32, 24])
    _require(isinstance(start, list) and len(start) == 2 and all(isinstance(k, int) and k >= 2 for k in start),
             "inverse.startRays must be [G, G'] with both ≥ 2")
    model["startRays"] = tuple(start)
    model["injective"] = inverse.get("injective")
    unique = inverse.get("minUniqueDistance")
    _require(unique is None or (isinstance(unique, (int, float)) and math.isfinite(unique) and unique >= 0),
             "inverse.minUniqueDistance must be null or a distance ≥ 0")
    model["minUniqueDistance"] = unique
    views = {}
    for name, pose in (document.get("views") or {}).items():
        r = np.asarray(pose.get("rotation"), dtype=float)
        c = np.asarray(pose.get("centre"), dtype=float)
        _require(r.shape == (3, 3) and c.shape == (3,), f"views.{name}: rotation 3 × 3 and centre [x, y, z]")
        views[name] = (r, c)
    model["views"] = views
    return model


def read(path) -> dict:
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise FormatError(f"{path}: not JSON ({e})") from None


def write(path, document: dict) -> None:
    """Writes the document as strict JSON; floats in repr precision, so that reading it back is bit-identical."""
    validate(document)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_plain(document), f, indent=1, allow_nan=False, ensure_ascii=False)
        f.write("\n")


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
