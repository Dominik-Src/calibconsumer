# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Uncertainties of rays and pixels (docs/format.md "uncertainty", docs/api.md "Uncertainties").

Three sources, each optional:
- the calibration's ray σ over the sensor (the calibration's covariance – its `quality.sigmaMap` – in pixel units: the σ
  of a ray's direction per axis of the plane normal to it, divided by the pixel footprint there);
- the mapping's own re-fit error over the sensor (how far the file's spline lies from the calibrated model, px);
- the image noise σ (px), the measurement noise of one detected point – not part of a ray's or a predicted pixel's
  uncertainty, added only when asked (the expected spread of a measured point around its prediction).

The ray σ and the re-fit error add in quadrature to the model σ in px; an angle is that times the pixel angle there.
A world point's own covariance propagates to its pixel through the local linearization of world → pixel."""

import json
import math
import os
import warnings

import numpy as np

STEP = 0.5  # px: the central differences of the rays


class Grid:
    """Values on cells of stepPx pixels (cell (x, y) holds the value at its centre pixel ((x + ½)·step − ½, …)),
    interpolated bilinearly between cell centres; NaN cells are left out of the blend, outside the grid the border
    cells hold."""

    def __init__(self, step_px, width, height, values):
        self.step = float(step_px)
        self.values = np.asarray([np.nan if v is None else float(v) for v in np.ravel(values)], dtype=float)
        self.values = self.values.reshape(int(height), int(width))

    @classmethod
    def of(cls, block):
        if not isinstance(block, dict) or not block.get("values"):
            return None
        grid = cls(block["stepPx"], block["width"], block["height"], block["values"])
        return grid if np.isfinite(grid.values).any() else None

    def to_document(self):
        return {"stepPx": self.step, "width": int(self.values.shape[1]), "height": int(self.values.shape[0]),
                "values": [None if not math.isfinite(v) else float(f"{v:.4g}") for v in self.values.ravel()]}

    def __call__(self, uv):
        uv = np.asarray(uv, dtype=float)
        finite = np.all(np.isfinite(uv), axis=-1)
        uv = np.where(np.isfinite(uv), uv, 0.0)
        h, w = self.values.shape
        x = np.clip((uv[..., 0] + 0.5) / self.step - 0.5, 0, w - 1)
        y = np.clip((uv[..., 1] + 0.5) / self.step - 0.5, 0, h - 1)
        x0, y0 = np.minimum(np.floor(x).astype(int), max(w - 2, 0)), np.minimum(np.floor(y).astype(int), max(h - 2, 0))
        fx, fy = x - x0, y - y0
        total = np.zeros(x.shape)
        weight = np.zeros(x.shape)
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            v = self.values[np.minimum(y0 + dy, h - 1), np.minimum(x0 + dx, w - 1)]
            ok = np.isfinite(v) & (wt > 0)
            total += np.where(ok, wt * np.where(ok, v, 0.0), 0.0)
            weight += np.where(ok, wt, 0.0)
        out = np.where(weight > 1e-12, total / np.maximum(weight, 1e-300), np.nan)
        return np.where(finite, out, np.nan)


class Uncertainty:
    """ray: Grid of the calibration's ray σ (px) or a number; refit: Grid of the re-fit error (px), refit_rms the
    re-fit's RMS where that grid has no value (px); image_sigma: the image noise σ (px). Any can be None."""

    def __init__(self, ray=None, refit=None, refit_rms=None, image_sigma=None, source=None):
        self.ray, self.refit, self.refit_rms, self.image_sigma, self.source = ray, refit, refit_rms, image_sigma, source

    @classmethod
    def from_document(cls, block):
        if not isinstance(block, dict):
            return None
        ray = block.get("ray")
        refit = block.get("refit") or {}
        return cls(ray=Grid.of(ray) if isinstance(ray, dict) else ray,
                   refit=Grid.of(refit.get("map")), refit_rms=refit.get("rmsPx"),
                   image_sigma=block.get("imageSigmaPx"), source=block.get("source"))

    def to_document(self):
        ray = self.ray.to_document() if isinstance(self.ray, Grid) else self.ray
        refit = {"rmsPx": self.refit_rms, "map": self.refit.to_document() if self.refit is not None else None}
        return {"unit": "px", "ray": ray, "refit": refit, "imageSigmaPx": self.image_sigma, "source": self.source}

    @classmethod
    def from_error_statistics(cls, path, refit=None, refit_rms=None):
        """The calibration's part from a result's error-statistics.json (its quality block's σ-map – the CLI writes it
        for central cameras – and noiseModel.sigmaPx), or None without either."""
        with open(path) as f:
            stats = json.load(f)
        quality = stats.get("quality") if isinstance(stats.get("quality"), dict) else {}
        grid = Grid.of(quality.get("sigmaMap"))
        if grid is not None:
            unit = (quality.get("linearization") or {}).get("unit")
            # "normalized": the angle ÷ the diagonal neighbour's angle φ; one px is φ/√2
            grid.values = grid.values * (math.sqrt(2.0) if unit == "normalized" else 1.0)
        noise = (stats.get("noiseModel") or {}).get("sigmaPx")
        noise = float(noise) if isinstance(noise, (int, float)) and noise > 0 else None
        if grid is None and noise is None:
            return None
        return cls(ray=grid, refit=refit, refit_rms=refit_rms, image_sigma=noise,
                   source="calibration covariance (error-statistics.json quality.sigmaMap)" if grid is not None else None)

    def model_px(self, uv, known_only=False):
        """The model σ (px, per axis) at pixels uv: the ray σ and the re-fit error in quadrature; NaN where the ray σ
        is unknown – or, with known_only and no ray σ at all, the re-fit error alone."""
        uv = np.asarray(uv, dtype=float)
        if self.ray is None:
            ray = np.full(uv.shape[:-1], 0.0 if known_only else np.nan)
        elif isinstance(self.ray, Grid):
            ray = self.ray(uv)
        else:
            ray = np.full(uv.shape[:-1], float(self.ray))
        refit = self.refit(uv) if self.refit is not None else np.full(uv.shape[:-1], np.nan)
        refit = np.where(np.isfinite(refit), refit, float(self.refit_rms or 0.0))
        return np.sqrt(ray * ray + refit * refit)


def attach(camera, directory):
    """Gives a camera loaded from a result folder the folder's calibration uncertainty, unless it has one."""
    path = os.path.join(directory, "error-statistics.json")
    if getattr(camera, "uncertainty", None) is None and os.path.isfile(path):
        try:
            camera.uncertainty = Uncertainty.from_error_statistics(path)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return camera


def ellipse(covariance):
    """(…, 2, 2) covariances → (…, 3): σ_major, σ_minor and the major axis' angle (deg, image +u towards +v)."""
    c = np.asarray(covariance, dtype=float)
    a, b, d = c[..., 0, 0], c[..., 0, 1], c[..., 1, 1]
    mean, half = 0.5 * (a + d), np.sqrt(0.25 * (a - d) ** 2 + b * b)
    angle = 0.5 * np.degrees(np.arctan2(2 * b, a - d))
    return np.stack([np.sqrt(np.maximum(mean + half, 0)), np.sqrt(np.maximum(mean - half, 0)), angle], -1)


class Uncertain:
    """The uncertainty methods every camera shares (Camera, OpenCVCamera, BrownCamera); they need pixel_to_ray."""

    uncertainty = None

    def _ray_derivatives(self, uv):
        """(o, d, o_u, d_u, o_v, d_v) by central differences of pixel_to_ray (the caller's frame), extrapolating."""
        uv = np.asarray(uv, dtype=float)
        e = np.array([STEP, 0.0]), np.array([0.0, STEP])
        o, d = self.pixel_to_ray(uv, allow_extrapolation=True)
        out = [o, d]
        for step in e:
            o1, d1 = self.pixel_to_ray(uv + step, allow_extrapolation=True)
            o0, d0 = self.pixel_to_ray(uv - step, allow_extrapolation=True)
            out += [(o1 - o0) / (2 * STEP), (d1 - d0) / (2 * STEP)]
        return out

    def pixel_angle(self, uv):
        """The angle between the rays of neighbouring pixels at uv (rad/px; the mean over u and v)."""
        _, _, _, d_u, _, d_v = self._ray_derivatives(np.asarray(uv, dtype=float))
        return 0.5 * (np.linalg.norm(d_u, axis=-1) + np.linalg.norm(d_v, axis=-1))

    def ray_sigma(self, uv):
        """The σ of the rays' directions at pixels uv (rad, per axis of the plane normal to the ray): the model σ
        (calibration ⊕ re-fit) times the angle of one pixel there. NaN without an uncertainty or where it is unknown."""
        uv = np.asarray(uv, dtype=float)
        if self.uncertainty is None:
            return np.full(uv.shape[:-1], np.nan)
        return self.uncertainty.model_px(uv) * self.pixel_angle(uv)

    def pixel_covariance(self, X, uv, pose=None, world_covariance=None, model=True, image_noise=False):
        """The covariance (…, 2, 2) of the pixels uv of points X (as world_to_pixel takes them, with pose): the model
        σ (calibration ⊕ re-fit, isotropic in px) if model, the world points' covariances world_covariance (…, 3, 3,
        in X's frame) through the local linearization of world → pixel, and the image noise if image_noise."""
        X, uv = np.asarray(X, dtype=float), np.asarray(uv, dtype=float)
        shape = uv.shape[:-1]
        X, uv = X.reshape(-1, 3), uv.reshape(-1, 2)
        cov = np.zeros((len(uv), 2, 2))
        eye = np.eye(2)
        u = self.uncertainty
        if model:
            if u is None or u.ray is None:
                warnings.warn("the calibration's ray σ is unknown for this camera (no uncertainty, or a non-central "
                              "camera): the pixel covariance holds only the parts that are known", stacklevel=3)
            s = u.model_px(uv, known_only=True) if u is not None else np.zeros(len(uv))
            cov += (s * s)[:, None, None] * eye
        if image_noise:
            s = u.image_sigma if u is not None and u.image_sigma is not None else np.nan
            cov += s * s * eye
        if world_covariance is not None:
            S = np.broadcast_to(np.asarray(world_covariance, dtype=float), (*shape, 3, 3)).reshape(-1, 3, 3)
            R = np.eye(3) if pose is None else np.asarray(pose[0], dtype=float)
            Xc = X if pose is None else (X - np.asarray(pose[1], dtype=float)) @ R
            S = np.einsum("ji,njk,kl->nil", R, S, R)  # into the camera frame: Rᵀ S R
            o, d, o_u, d_u, o_v, d_v = self._ray_derivatives(uv)
            if pose is not None:  # the rays into the file's frame, the posed points' frame (F is its own inverse)
                o, d, o_u, d_u, o_v, d_v = (v @ self._F for v in (o, d, o_u, d_u, o_v, d_v))
            w = Xc - o
            rho = np.linalg.norm(w, axis=-1, keepdims=True)
            # the residual r(uv, X) = (I − d dᵀ)(X − o)/ρ vanishes at the solution; dr/dX = (I − d dᵀ)/ρ there
            P = np.eye(3) - d[:, :, None] * d[:, None, :]
            columns = []
            for o_k, d_k in ((o_u, d_u), (o_v, d_v)):
                dw = -o_k
                columns.append((P @ dw[..., None])[..., 0] / rho
                               - (d_k * np.sum(d * w, -1, keepdims=True) + d * np.sum(d_k * w, -1, keepdims=True)) / rho)
            J = np.stack(columns, -1)  # (n, 3, 2): dr/duv
            ok = np.all(np.isfinite(J), axis=(1, 2)) & np.isfinite(rho[:, 0])
            M = np.full((len(uv), 2, 3), np.nan)
            M[ok] = -np.linalg.pinv(J[ok]) @ (P[ok] / rho[ok][:, :, None])  # duv/dX = −J⁺ dr/dX
            cov += M @ S @ np.swapaxes(M, -1, -2)
        return cov.reshape(*shape, 2, 2)

    def _with_sigma(self, result, uv, return_sigma):
        if not return_sigma:
            return result
        sigma = self.ray_sigma(uv)
        return (*result, np.where(np.isfinite(result[1][..., 0]), sigma, np.nan))

    def _with_covariance(self, result, X, pose, return_covariance, world_covariance, image_noise):
        if not return_covariance:
            return result
        result = result if isinstance(result, tuple) else (result,)
        cov = self.pixel_covariance(X, result[0], pose=pose, world_covariance=world_covariance, image_noise=image_noise)
        return (*result, np.where(np.isfinite(result[0][..., :1, None]), cov, np.nan))
