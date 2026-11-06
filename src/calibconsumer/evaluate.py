# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Re-evaluation of a camera against measured image points and their world points (docs/evaluate.md): per observation
the reprojection error (px), the angular error (the measured pixel's ray against the direction to the world point),
the object-space error (the world point's distance from that ray, world units), the normalized angular error (the
angle in pixels of that ray's footprint), each one's expected σ from the uncertainties (the camera's, the world
points', the image noise) and the outliers. Nothing is optimized: the poses are the inputs' (a calibration's result, a
pose file) or – for views without one – estimated from the view's points with the camera held fixed."""

from __future__ import annotations

import csv
import json
import math
import warnings

import numpy as np

from . import pose as poses
from .uncertainty import ellipse

ARCSEC = 180 / math.pi * 3600
RAYLEIGH_MEDIAN = math.sqrt(2 * math.log(2))  # the median of |r| for a 2-D Gaussian of σ 1 per axis


class Evaluation:
    """Per observation (arrays of length n, in the order view by view): `view` (index into `view_names`), `ids`, `uv`
    (measured), `predicted`, `residual` (predicted − measured, px), `reprojection` (px), `angle` (rad), `normalized`
    (px), `object_space` (world units), `distance` (from the ray's origin to the world point), `sigma_image` (n, 3:
    σ_major, σ_minor px, angle deg), `sigma_object` (world units, per axis normal to the ray), `mahalanobis`,
    `outlier`, `valid` (the camera gave a prediction and a ray). `poses` {view: (R, centre)}, `pose_source` {view:
    "given" | "estimated"}, `image_sigma` (px) and `image_sigma_source`, `resolution`."""

    def summary(self) -> dict:
        def stats(values):
            v = np.asarray(values, dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                return None
            return {"rms": float(np.sqrt(np.mean(v * v))), "mean": float(v.mean()), "median": float(np.median(v)),
                    "p90": float(np.percentile(v, 90)), "max": float(v.max())}

        def block(mask):
            use = mask & self.valid & ~self.outlier
            return {"observations": int(mask.sum()), "valid": int((mask & self.valid).sum()),
                    "outliers": int((mask & self.valid & self.outlier).sum()),
                    "reprojectionPx": stats(self.reprojection[use]),
                    "angleArcsec": stats(self.angle[use] * ARCSEC),
                    "normalizedPx": stats(self.normalized[use]),
                    "objectSpace": stats(self.object_space[use]),
                    "expectedSigmaImagePx": stats(np.hypot(self.sigma_image[use, 0], self.sigma_image[use, 1])
                                                  / math.sqrt(2)),
                    "expectedSigmaObject": stats(self.sigma_object[use]),
                    "mahalanobis": stats(self.mahalanobis[use])}

        views = {}
        for k, name in enumerate(self.view_names):
            views[name] = dict(block(self.view == k), pose=self.pose_source.get(name),
                               poseInfo=self.pose_info.get(name))
        return {"all": block(np.ones(len(self.view), bool)), "views": views,
                "imageSigmaPx": self.image_sigma, "imageSigmaSource": self.image_sigma_source,
                "outlierThreshold": self.outlier_threshold, "outlierRule": self.outlier_rule,
                "units": {"reprojection": "px", "angle": "arcsec", "normalized": "px (the angle in pixels of the "
                          "measured ray's footprint)", "objectSpace": "world units", "sigma": "1 σ"},
                "notes": self.notes}

    def to_json(self, path, inputs=None):
        document = {"format": "calibconsumer-evaluation", "formatVersion": 1, "inputs": inputs or {},
                    "summary": self.summary(),
                    "poses": {name: {"rotation": R.tolist(), "centre": c.tolist(), "source": self.pose_source[name]}
                              for name, (R, c) in self.poses.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_plain(document), f, indent=1)

    def to_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["view", "id", "u", "v", "uPredicted", "vPredicted", "du", "dv", "reprojectionPx",
                        "angleArcsec", "normalizedPx", "objectSpace", "distance", "sigmaImageMajorPx",
                        "sigmaImageMinorPx", "sigmaImageAngleDeg", "sigmaObject", "mahalanobis", "outlier", "valid"])
            for i in range(len(self.view)):
                w.writerow([self.view_names[self.view[i]], self.ids[i], *_fmt(self.uv[i]), *_fmt(self.predicted[i]),
                            *_fmt(self.residual[i]), *_fmt([self.reprojection[i], self.angle[i] * ARCSEC,
                                                            self.normalized[i], self.object_space[i],
                                                            self.distance[i]]),
                            *_fmt(self.sigma_image[i]), *_fmt([self.sigma_object[i], self.mahalanobis[i]]),
                            int(self.outlier[i]), int(self.valid[i])])


def _fmt(values):
    return ["" if not np.isfinite(v) else f"{v:.6g}" for v in np.ravel(values)]


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    return value


def evaluate(camera, views: dict, world: dict, poses_given: dict | None = None, world_covariance: dict | None = None,
             estimate_poses: str = "missing", image_sigma: float | None = None, outlier_threshold: float = 4.0):
    """Evaluates camera on views {name: (ids, uv)} against world {id: X} (world frame). poses_given {name: (R camera →
    world, centre)}; estimate_poses "missing" (views without a given pose), "all" or "never" (such views are left
    out). world_covariance {id: 3 × 3}. image_sigma (px) overrides the camera's; without either it is estimated from the
    residuals (robustly). An observation is an outlier when its residual's Mahalanobis distance under the expected
    covariance exceeds outlier_threshold."""
    poses_given = poses_given or {}
    world_covariance = world_covariance or {}
    e = Evaluation()
    e.resolution = tuple(camera.resolution)
    e.notes = []
    e.view_names, e.poses, e.pose_source, e.pose_info = [], {}, {}, {}
    chunks = []
    for name in sorted(views):
        ids, uv = views[name]
        keep = np.array([i in world for i in ids], dtype=bool)
        ids, uv = np.asarray(ids)[keep], np.asarray(uv, dtype=float)[keep]
        if not len(ids):
            e.notes.append(f"view {name}: no point with a world point")
            continue
        X = np.array([world[i] for i in ids])
        pose = None if estimate_poses == "all" else poses_given.get(name)
        if pose is not None:
            source = "given"
        elif estimate_poses == "never":
            e.notes.append(f"view {name}: no pose given – left out")
            continue
        else:
            result = poses.estimate(camera, uv, X)
            if result is None:
                e.notes.append(f"view {name}: its pose could not be estimated – left out")
                continue
            pose, e.pose_info[name] = result
            source = "estimated"
        e.view_names.append(name)
        e.poses[name], e.pose_source[name] = (np.asarray(pose[0], float), np.asarray(pose[1], float)), source
        chunks.append((len(e.view_names) - 1, ids, uv, X))
    if not chunks:
        raise ValueError("nothing to evaluate: no view with world points and a pose")
    e.view = np.concatenate([np.full(len(c[1]), c[0]) for c in chunks])
    e.ids = np.concatenate([c[1] for c in chunks])
    e.uv = np.concatenate([c[2] for c in chunks])
    X = np.concatenate([c[3] for c in chunks])
    n = len(e.view)
    F = getattr(camera, "_F", np.eye(3))
    Xc = np.empty((n, 3))
    Sc = np.zeros((n, 3, 3))
    has_cov = np.array([i in world_covariance for i in e.ids], dtype=bool)
    for k, name in enumerate(e.view_names):
        m = e.view == k
        R, c = e.poses[name]
        Xc[m] = ((X[m] - c) @ R) @ F  # the camera's frame (the poses are in the file's)
        for j in np.nonzero(m & has_cov)[0]:
            Sc[j] = F @ R.T @ world_covariance[e.ids[j]] @ R @ F
    if has_cov.any() and not has_cov.all():
        e.notes.append(f"{int((~has_cov).sum())} observations without a world point covariance: taken as exact")
    # the measured pixels' rays: angle, object space, normalized
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, d = camera.pixel_to_ray(e.uv, allow_extrapolation=True)
        e.predicted = camera.world_to_pixel(Xc, allow_extrapolation=True)
    w = Xc - o
    e.distance = np.linalg.norm(w, axis=-1)
    along = np.sum(w * d, -1)
    e.object_space = np.linalg.norm(w - d * along[:, None], axis=-1)
    e.angle = np.arctan2(e.object_space, along)
    pixel_angle = _pixel_angle(camera, e.uv)
    e.normalized = e.angle / pixel_angle
    e.residual = e.predicted - e.uv
    e.reprojection = np.linalg.norm(e.residual, axis=-1)
    e.valid = np.isfinite(e.reprojection) & np.isfinite(e.angle)
    # expected covariances: the model's, the world points' and the image noise
    u = getattr(camera, "uncertainty", None)
    at = np.where(np.isfinite(e.predicted), e.predicted, e.uv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = camera.pixel_covariance(Xc, at, world_covariance=Sc if has_cov.any() else None, model=True)
        model_px = u.model_px(at, known_only=True) if u is not None else np.zeros(n)
    if u is None or u.ray is None:
        e.notes.append("the calibration's ray σ is unknown: the expected σ holds the world points' and the image "
                       "noise's parts only")
    if image_sigma is not None:
        e.image_sigma, e.image_sigma_source = float(image_sigma), "given"
    elif u is not None and u.image_sigma is not None:
        e.image_sigma, e.image_sigma_source = float(u.image_sigma), "camera"
    else:
        r = e.reprojection[e.valid]
        e.image_sigma = float(np.median(r) / RAYLEIGH_MEDIAN) if len(r) else float("nan")
        e.image_sigma_source = "estimated from the residuals (median |r| / 1.1774)"
    cov = cov + e.image_sigma ** 2 * np.eye(2)
    e.sigma_image = ellipse(cov)
    world_perp = np.zeros(n)
    if has_cov.any():
        P = np.eye(3) - d[:, :, None] * d[:, None, :]
        world_perp = np.sqrt(np.maximum(np.trace(P @ Sc @ P, axis1=1, axis2=2) / 2, 0))
    ray_px = np.sqrt(np.square(np.nan_to_num(model_px)) + e.image_sigma ** 2)
    e.sigma_object = np.hypot(e.distance * pixel_angle * ray_px, world_perp)
    good = np.isfinite(cov).all(axis=(1, 2)) & (np.linalg.det(np.nan_to_num(cov)) > 0)
    e.mahalanobis = np.full(n, np.nan)
    rv = np.nan_to_num(e.residual)
    e.mahalanobis[good] = np.sqrt(np.einsum("ni,nij,nj->n", rv[good], np.linalg.inv(cov[good]), rv[good]))
    e.outlier_threshold = float(outlier_threshold)
    e.outlier_rule = "Mahalanobis distance of the reprojection residual under the expected covariance"
    e.outlier = e.valid & ~(e.mahalanobis <= outlier_threshold)
    return e


def _pixel_angle(camera, uv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return camera.pixel_angle(uv)
