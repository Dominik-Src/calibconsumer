# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Readers for the inputs of a re-evaluation (docs/evaluate.md): measured image points, world points and their σ,
view poses – in the calibration project's files or in plain text and JSON of one's own.

Text files: one row per item, the columns separated by spaces, tabs, commas or semicolons; `#` starts a comment; a row
whose numbers do not parse (a header) is skipped. Point ids are compared as text ("7" and "7.0" differ)."""

from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .camera import OPENCV

SEPARATORS = re.compile(r"[\s,;]+")


def view_name(path_or_name: str) -> str:
    """A view's name: the file's base without the folder, without .csv/.txt and without a trailing dot
    (`images/image-0007.csv` → `image-0007`, as the calibration's views are named)."""
    name = os.path.basename(str(path_or_name))
    for suffix in (".csv", ".txt"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name.rstrip(".")


def _rows(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                yield [v for v in SEPARATORS.split(line) if v]


def _floats(values):
    try:
        return [float(v) for v in values]
    except ValueError:
        return None


def _id(text: str) -> str:
    """Integer ids written as 7 or 7.0 (or with padding) compare equal; other ids stay as written."""
    try:
        value = float(text)
    except ValueError:
        return text
    return str(int(value)) if value.is_integer() else text


def read_points(path) -> tuple[np.ndarray, np.ndarray]:
    """One view's points: rows `id u v` (pixels, (0, 0) the centre of the top-left pixel – the calibration's point files,
    docs/evaluate.md). Returns (ids (n,) str, uv (n, 2)); a repeated id keeps its first row."""
    ids, uv, seen = [], [], set()
    for row in _rows(path):
        if len(row) < 3:
            continue
        xy = _floats(row[1:3])
        if xy is None or row[0] in seen:
            continue
        seen.add(_id(row[0]))
        ids.append(_id(row[0]))
        uv.append(xy)
    return np.array(ids, dtype=str), np.array(uv, dtype=float).reshape(-1, 2)


def read_points_table(path) -> dict:
    """Points of several views in one file: rows `view id u v`. Returns {view: (ids, uv)}."""
    views = {}
    for row in _rows(path):
        if len(row) < 4:
            continue
        xy = _floats(row[2:4])
        if xy is None:
            continue
        ids, uv = views.setdefault(view_name(row[0]), ([], []))
        if _id(row[1]) not in ids:
            ids.append(_id(row[1]))
            uv.append(xy)
    return {k: (np.array(i, dtype=str), np.array(u, dtype=float).reshape(-1, 2)) for k, (i, u) in views.items()}


def read_points_any(path) -> dict:
    """A folder of per-view files (`<view>.csv` or `.txt`, rows `id u v`), one per-view file, or one table
    (`view id u v`): {view: (ids, uv)}."""
    path = os.fspath(path)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")) + glob.glob(os.path.join(path, "*.txt")))
        files = [f for f in files if not f.endswith("cov.csv")]  # the per-point covariances (docs/evaluate.md)
        return {view_name(f): read_points(f) for f in files}
    first = next((row for row in _rows(path) if len(row) >= 3 and _floats(row[-2:]) is not None), [])
    if len(first) >= 4:
        return read_points_table(path)  # a view in front of the id
    return {view_name(path): read_points(path)}


def read_world(path) -> dict:
    """World points: rows `id X Y Z [pattern name]` (docs/evaluate.md). Returns {id: (3,)}."""
    world = {}
    for row in _rows(path):
        if len(row) < 4:
            continue
        xyz = _floats(row[1:4])
        if xyz is not None and _id(row[0]) not in world:
            world[_id(row[0])] = np.array(xyz)
    return world


def read_world_sigma(path) -> dict:
    """World points' σ: rows `id σ`, `id σx σy σz` or `id xx xy xz yy yz zz` (a covariance; world units), or a
    calibration's refined-target.json (its points' covariances). Returns {id: (3, 3)}."""
    path = os.fspath(path)
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        for p in data.get("points", []):
            c = p.get("covariance")
            if c and len(c) == 6 and all(isinstance(v, (int, float)) for v in c):
                out[_id(str(p["id"]))] = _covariance6(c)
        return out
    out = {}
    for row in _rows(path):
        values = _floats(row[1:])
        if not values:
            continue
        if len(values) >= 6:
            out[_id(row[0])] = _covariance6(values[:6])
        elif len(values) >= 3:
            out[_id(row[0])] = np.diag(np.square(values[:3]))
        else:
            out[_id(row[0])] = np.eye(3) * values[0] ** 2
    return out


def _covariance6(c):
    xx, xy, xz, yy, yz, zz = (float(v) for v in c)
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _rodrigues(rvec):
    r = np.asarray(rvec, dtype=float).ravel()
    theta = float(np.linalg.norm(r))
    if theta < 1e-15:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def pose_from_opencv(rvec, tvec):
    """OpenCV's world → camera (rvec, tvec) as the package's pose (rotation camera → world in the package's camera
    frame, centre)."""
    R = _rodrigues(rvec)
    return R.T @ OPENCV, -R.T @ np.asarray(tvec, dtype=float).ravel()


def read_poses(path) -> dict:
    """View poses from JSON: {"views": {name: {"rotation": 3 × 3 camera → world, "centre": [x, y, z]}}} (mapping.json's
    form, the package's camera frame), the same without "views", {name: {"rvec", "tvec"}} or {"rvecs": {…},
    "tvecs": {…}} (OpenCV's world → camera). Returns {view: (R, centre)}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "rvecs" in data and "tvecs" in data:
        return {view_name(k): pose_from_opencv(v, data["tvecs"][k]) for k, v in data["rvecs"].items()
                if k in data["tvecs"]}
    data = data.get("views", data)
    poses = {}
    for name, pose in data.items():
        if not isinstance(pose, dict):
            continue
        if "rvec" in pose and "tvec" in pose:
            poses[view_name(name)] = pose_from_opencv(pose["rvec"], pose["tvec"])
        elif "rotation" in pose and "centre" in pose:
            R = np.asarray(pose["rotation"], dtype=float)
            R = _rodrigues(R) if R.shape == (3,) else R.reshape(3, 3)
            poses[view_name(name)] = (R, np.asarray(pose["centre"], dtype=float))
    return poses


@dataclass
class Observations:
    """What a re-evaluation measures against: per view its points (ids, uv), the world points, their covariances and
    the poses the inputs gave; `sources` says where each came from."""

    views: dict = field(default_factory=dict)
    world: dict = field(default_factory=dict)
    world_covariance: dict = field(default_factory=dict)
    poses: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)


def project_file(project) -> str:
    project = os.fspath(project)
    return os.path.join(project, "projectData.json") if os.path.isdir(project) else project


def from_project(project, camera_id=None) -> Observations:
    """A calibration project (its folder or projectData.json, docs/evaluate.md): one camera's views
    (`cameraLensInstance-<id>/images/<base>csv`) and the project's world points (`worldPoints.csv`). camera_id is
    needed when the project has more than one camera."""
    path = project_file(project)
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cameras = data.get("cameraData") or []
    if camera_id is None:
        if len(cameras) != 1:
            ids = ", ".join(str(c.get("cameraLensInstanceId")) for c in cameras)
            raise ValueError(f"{path}: {len(cameras)} cameras ({ids}) – name one")
        camera = cameras[0]
    else:
        matches = [c for c in cameras if str(c.get("cameraLensInstanceId")) == str(camera_id)]
        if not matches:
            raise ValueError(f"{path}: no camera {camera_id}")
        camera = matches[0]
    folder = os.path.join(base, f"cameraLensInstance-{camera['cameraLensInstanceId']}", "images")
    obs = Observations()
    for image in camera.get("imageData") or []:
        file = os.path.join(folder, f"{image['fileNameBase']}csv")
        if os.path.isfile(file):
            obs.views[view_name(image["fileNameBase"])] = read_points(file)
    obs.sources["points"] = folder
    world = os.path.join(base, data.get("worldPointsOriginal") or "worldPoints.csv")
    if os.path.isfile(world):
        obs.world = read_world(world)
        obs.sources["world"] = world
    return obs


def from_result(camera_dir, project=None) -> tuple[Observations, dict]:
    """A calibration's result folder `resultsCalibration/<run>/cameraLensInstance-<id>`: the project it calibrated
    (results.json projectFilePath, or project), the run's world (updated-worldPoints.csv – the world the result was
    calibrated with, optimized or not) and its covariances (refined-target.json, with -w). The poses come with the
    camera (camera.views). Returns (observations, the camera's results.json or {})."""
    camera_dir = os.path.abspath(os.fspath(camera_dir))
    run = os.path.dirname(camera_dir)
    info, run_info = {}, {}
    for target, file in ((info, os.path.join(camera_dir, "results.json")), (run_info, os.path.join(run, "results.json"))):
        if os.path.isfile(file):
            with open(file, encoding="utf-8") as f:
                target.update(json.load(f))
    if project is None and run_info.get("projectFilePath"):
        project = os.path.normpath(os.path.join(run, run_info["projectFilePath"]))
    camera_id = info.get("cameraId")
    if camera_id is None:
        match = re.search(r"cameraLensInstance-(\d+)", os.path.basename(camera_dir))
        camera_id = match.group(1) if match else None
    obs = from_project(project, camera_id) if project is not None else Observations()
    world = os.path.join(run, run_info.get("updatedWorldFile") or "updated-worldPoints.csv")
    if os.path.isfile(world):
        obs.world = read_world(world)
        obs.sources["world"] = world
    target = os.path.join(run, "refined-target.json")
    if os.path.isfile(target):
        obs.world_covariance = read_world_sigma(target)
        obs.sources["worldCovariance"] = target
    return obs, info
