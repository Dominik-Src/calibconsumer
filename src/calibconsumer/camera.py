# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""The camera of a mapping file: pixel → ray in closed form, world → pixel by a ray-grid start and Gauss–Newton."""

from __future__ import annotations

import enum
import warnings

import numpy as np

from . import direction as ae
from . import fileformat
from .hull import ConvexPolygon
from .spline import Spline
from .uncertainty import Uncertain, Uncertainty


class Status(enum.IntEnum):
    """Per point of world_to_pixel / pixel_to_ray."""
    OK = 0
    OUTSIDE_HULL = 1      # outside the calibrated image area (a result only with allow_extrapolation=True)
    OUTSIDE_DOMAIN = 2    # beyond the spline's rectangle plus 10 % – never mapped
    NO_CONVERGENCE = 3    # world → pixel found no pixel whose ray passes through the point
    BEHIND = 4            # the point lies behind the camera (no ray reaches it)
    AMBIGUOUS_NEAR = 5    # non-central: closer than the file's minUniqueDistance – the pixel found may not be the only one
    FOLDED = 6            # the point is seen by more than one pixel: the direction map folds over there (the file's
                          # inverse.foldedShare) – a result (the pixel outside the fold where there is one) only with
                          # allow_folded=True


OPENCV = np.diag([1.0, -1.0, -1.0])  # the file's frame (x right, y up, −z) to OpenCV's (x right, y down, +z)
DOMAIN_MARGIN = 0.1


class Camera(Uncertain):
    """A calibrated camera as described by a mapping.json file. Arrays are vectorized over leading dimensions; the
    camera frame is the file's (x right, y up, viewing along −z) unless with_frame("opencv") was used."""

    def __init__(self, document: dict, frame: str = "calibconsumer", quiet: bool = False):
        model = fileformat.validate(document)
        folded = (document.get("inverse") or {}).get("foldedShare")
        if model["injective"] is False and not quiet:
            warnings.warn("the direction map folds over in part of the calibrated area"
                          + (f" ({100 * folded:.2g} % of it)" if isinstance(folded, (int, float)) else "")
                          + ": world → pixel gives status FOLDED there", stacklevel=2)
        refit = document.get("refit") or {}
        if refit.get("passed") is False and not quiet:
            warnings.warn("the mapping did not meet its re-fit bounds (refit.passed is false); see refit for the "
                          "measured errors", stacklevel=2)
        self.document = document
        self.resolution = model["resolution"]
        self.refit = refit
        self.central = model["origin"] is None
        self.rectangle = model["rectangle"]
        self.hull = ConvexPolygon(model["hull"])
        self.rotation = model["rotation"]
        self._direction = Spline(self.rectangle, model["intervals"], model["direction"])
        self._origin = model["origin"]
        if self._origin is not None and self._origin[0] == "foot-point":
            self._foot = Spline(self.rectangle, model["intervals"], self._origin[1])
        elif self._origin is not None:
            self._near = Spline(self.rectangle, model["intervals"], self._origin[1])
            self._far = Spline(self.rectangle, model["intervals"], self._origin[2])
        self.start_rays = model["startRays"]
        self.min_unique_distance = model["minUniqueDistance"]
        self.views = model["views"]
        self.uncertainty = Uncertainty.from_document(document.get("uncertainty"))
        if frame not in ("calibconsumer", "opencv"):
            raise ValueError("frame is 'calibconsumer' or 'opencv'")
        self.frame = frame
        self._F = OPENCV if frame == "opencv" else np.eye(3)
        size = self.rectangle[1] - self.rectangle[0]
        self._outer = np.array([self.rectangle[0] - DOMAIN_MARGIN * size, self.rectangle[1] + DOMAIN_MARGIN * size])
        self._starts = {}
        self._fold_sign = self._majority_sign()

    def _majority_sign(self, samples=100):
        """The sign of the direction map's Jacobian determinant over most of the calibrated area; a pixel whose
        determinant has the other sign lies in a fold (world → pixel there is not unique)."""
        lo, hi = self.hull.vertices.min(axis=0), self.hull.vertices.max(axis=0)
        u, v = np.meshgrid(np.linspace(lo[0], hi[0], samples), np.linspace(lo[1], hi[1], samples))
        uv = np.stack([u, v], -1).reshape(-1, 2)
        det = self.fold_determinant(uv[self.hull.contains(uv)])
        return float(np.sign(np.median(det))) if len(det) else 1.0

    FOLD_GRID = 400  # the grid that finds a fold's pixels (a fold can be a few pixels narrow)
    FOLD_STARTS = 4000

    @property
    def _fold_starts(self):
        """Pixels in the folds (the starts of world → pixel's second search), found on a fine grid once, and only
        for a file that says its direction map folds (inverse.injective false)."""
        if not hasattr(self, "_fold_starts_cache"):
            starts = np.zeros((0, 2))
            if (self.document.get("inverse") or {}).get("injective") is False:
                lo, hi = self.hull.vertices.min(axis=0), self.hull.vertices.max(axis=0)
                g = self.FOLD_GRID
                u, v = np.meshgrid(np.linspace(lo[0], hi[0], g), np.linspace(lo[1], hi[1], g))
                uv = np.stack([u, v], -1).reshape(-1, 2)
                uv = uv[self.hull.contains(uv)]
                starts = uv[self.folded(uv)]
                if len(starts) > self.FOLD_STARTS:
                    starts = starts[:: int(np.ceil(len(starts) / self.FOLD_STARTS))]
            self._fold_starts_cache = starts
        return self._fold_starts_cache

    def fold_determinant(self, uv):
        """det ∂(a, b)/∂(u, v) of the direction spline at pixels uv."""
        _, ab_u, ab_v = self._direction.evaluate(np.asarray(uv, dtype=float), derivatives=True)
        return ab_u[..., 0] * ab_v[..., 1] - ab_u[..., 1] * ab_v[..., 0]

    def folded(self, uv) -> np.ndarray:
        """Whether pixels lie in a fold of the direction map (the determinant's minority sign)."""
        return self.fold_determinant(uv) * self._fold_sign <= 0

    @classmethod
    def load(cls, path, quiet: bool = False) -> "Camera":
        return cls(fileformat.read(path), quiet=quiet)

    def with_frame(self, frame: str) -> "Camera":
        """The same camera with rays and points in another camera frame: "calibconsumer" (the file's: x right, y up, viewing
        along −z) or "opencv" (x right, y down, viewing along +z)."""
        return Camera(self.document, frame=frame, quiet=True)

    # --- pixel → ray --------------------------------------------------------------------------------------------

    def _rays(self, uv: np.ndarray, derivatives: bool = False):
        """(origin, direction) in the file's frame, with derivatives d/du, d/dv of both if asked."""
        if derivatives:
            ab, ab_u, ab_v = self._direction.evaluate(uv, derivatives=True)
            d, J = ae.to_direction(ab, self.rotation, jacobian=True)
            d_u, d_v = np.einsum("...ij,...j->...i", J, ab_u), np.einsum("...ij,...j->...i", J, ab_v)
        else:
            d = ae.to_direction(self._direction.evaluate(uv), self.rotation)
        if self._origin is None:
            o = np.zeros_like(d)
            return (o, d, np.zeros_like(d), np.zeros_like(d), d_u, d_v) if derivatives else (o, d)
        if self._origin[0] == "foot-point":
            if derivatives:
                o, o_u, o_v = self._foot.evaluate(uv, derivatives=True)
                return o, d, o_u, o_v, d_u, d_v
            return self._foot.evaluate(uv), d
        # two-surface: the line through the near and far points; its direction replaces the AE spline's
        near = self._near.evaluate(uv, derivatives=derivatives)
        far = self._far.evaluate(uv, derivatives=derivatives)
        p1, p2 = (near[0], far[0]) if derivatives else (near, far)
        delta = p2 - p1
        length = np.linalg.norm(delta, axis=-1, keepdims=True)
        d = delta / length
        if not derivatives:
            return p1, d
        out = [p1, d, near[1], near[2]]
        for k in (1, 2):
            dd = far[k] - near[k]
            out.append((dd - d * np.sum(d * dd, axis=-1, keepdims=True)) / length)
        return out[0], out[1], out[2], out[3], out[4], out[5]

    def _domain_status(self, uv: np.ndarray, allow_extrapolation: bool = False) -> np.ndarray:
        inside = self.hull.contains(uv)
        within = np.all((uv >= self._outer[0]) & (uv <= self._outer[1]), axis=-1)
        status = np.where(inside, Status.OK, np.where(within, Status.OUTSIDE_HULL, Status.OUTSIDE_DOMAIN))
        return np.where(np.all(np.isfinite(uv), axis=-1), status, Status.OUTSIDE_DOMAIN)

    def pixel_to_ray(self, uv, allow_extrapolation: bool = False, return_status: bool = False,
                     return_sigma: bool = False):
        """(origin, direction) of the rays of pixels uv (..., 2): origin (..., 3) – the ray's foot point, zero for a
        central camera –, unit direction (..., 3). Pixels outside the calibrated area give NaN unless
        allow_extrapolation=True; beyond the domain always NaN. return_sigma appends the directions' σ (rad, see
        ray_sigma)."""
        uv = np.asarray(uv, dtype=float)
        o, d = self._rays(uv)
        status = self._domain_status(uv, allow_extrapolation)
        refused = (status == Status.OUTSIDE_DOMAIN) | ((status == Status.OUTSIDE_HULL) & (not allow_extrapolation))
        o, d = np.where(refused[..., None], np.nan, o), np.where(refused[..., None], np.nan, d)
        o, d = o @ self._F.T, d @ self._F.T
        return self._with_sigma((o, d, status) if return_status else (o, d), uv, return_sigma)

    def contains(self, uv) -> np.ndarray:
        """Whether pixels lie in the calibrated image area (the convex hull of the calibration's points)."""
        return self.hull.contains(np.asarray(uv, dtype=float))

    def pixel_angle(self, uv) -> np.ndarray:
        """The angle between the rays of neighbouring pixels at uv (rad/px; the mean over u and v)."""
        _, _, _, _, d_u, d_v = self._rays(np.asarray(uv, dtype=float), derivatives=True)
        return 0.5 * (np.linalg.norm(d_u, axis=-1) + np.linalg.norm(d_v, axis=-1))

    # --- world → pixel ------------------------------------------------------------------------------------------

    def world_to_pixel(self, X, pose=None, allow_extrapolation: bool = False, return_status: bool = False,
                       max_iterations: int = 50, allow_folded: bool = False, return_covariance: bool = False,
                       world_covariance=None, image_noise: bool = False):
        """The pixel (..., 2) whose ray passes through the points X (..., 3). X is in the camera frame, or in world
        coordinates with pose = (rotation camera → world, centre) – e.g. camera.views["image-0003"]. Points that map
        outside the calibrated area give NaN unless allow_extrapolation=True (status OUTSIDE_HULL either way); points
        whose pixel lies in a fold of the direction map NaN unless allow_folded=True (status FOLDED either way).
        return_covariance appends each pixel's covariance (..., 2, 2) – see pixel_covariance."""
        from .inverse import solve
        X = given = np.asarray(X, dtype=float)
        if pose is not None:
            R, c = (np.asarray(p, dtype=float) for p in pose)
            X = (X - c) @ R  # Rᵀ(X − c), in the file's frame
        else:
            X = X @ self._F  # the caller's frame → the file's (F is its own inverse)
        uv, status = solve(self, X, allow_extrapolation, max_iterations, allow_folded)
        return self._with_covariance((uv, status) if return_status else uv, given, pose, return_covariance,
                                     world_covariance, image_noise)

    def _start_grid(self, allow_extrapolation: bool):
        key = bool(allow_extrapolation)
        if key not in self._starts:
            g, h = self.start_rays
            if allow_extrapolation:
                lo, hi = self._outer
            else:
                lo, hi = self.hull.vertices.min(axis=0), self.hull.vertices.max(axis=0)
            u = np.linspace(lo[0], hi[0], g)
            v = np.linspace(lo[1], hi[1], h)
            uv = np.stack(np.meshgrid(u, v), axis=-1)  # (h, g, 2): the regular grid, kept whole for the cells
            o, d = self._rays(uv)
            finite = np.all(np.isfinite(d), axis=-1) & np.all(np.isfinite(o), axis=-1)
            nearest = finite.copy()  # the nearest-ray search: the grid's rays inside the hull
            if not allow_extrapolation:
                inside = self.hull.contains(uv, tolerance=1e-6) & finite
                nearest = inside if inside.sum() >= 3 else finite
            rows, cols = np.nonzero(nearest)
            self._starts[key] = {"uv": uv, "o": o, "d": d, "finite": finite, "rows": rows, "cols": cols,
                                 "step": np.array([u[1] - u[0], v[1] - v[0]])}
        return self._starts[key]


def load(path, quiet: bool = False) -> Camera:
    """Camera.load(path); quiet=True suppresses the warnings about folds and failed re-fit bounds."""
    return Camera.load(path, quiet=quiet)
