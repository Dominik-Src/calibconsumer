# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""The uniform cubic tensor B-spline over a pixel rectangle (docs/format.md, "spline").

A spline has n × m uniform intervals on the rectangle [u0, u1] × [v0, v1] and (m + 3) × (n + 3) control values per
component, stored as an array of shape (m + 3, n + 3, C): v index first, then u index, then the component. Inside an
interval the value is sum_{a,b} B_a(s) B_b(t) c[j + b, i + a] with the uniform cubic basis

    B(t) = 1/6 · [(1 − t)³, 3t³ − 6t² + 4, −3t³ + 3t² + 3t + 1, t³],

i, s the interval index and fraction along u, j, t along v. Outside the rectangle the border interval's polynomial
continues (s or t outside [0, 1]); callers decide whether that extrapolation is allowed.
"""

from __future__ import annotations

import numpy as np


def basis(t: np.ndarray) -> np.ndarray:
    """The four uniform cubic B-spline basis values at t, shape t.shape + (4,)."""
    t = np.asarray(t, dtype=float)
    t2, t3 = t * t, t * t * t
    return np.stack([(1 - t) ** 3, 3 * t3 - 6 * t2 + 4, -3 * t3 + 3 * t2 + 3 * t + 1, t3], axis=-1) / 6.0


def basis_derivative(t: np.ndarray) -> np.ndarray:
    """d/dt of basis(t), shape t.shape + (4,)."""
    t = np.asarray(t, dtype=float)
    t2 = t * t
    return np.stack([-3 * (1 - t) ** 2, 9 * t2 - 12 * t, -9 * t2 + 6 * t + 3, 3 * t2], axis=-1) / 6.0


class Spline:
    """A uniform cubic tensor B-spline with C components over a pixel rectangle."""

    def __init__(self, rectangle, intervals, controls):
        (self.u0, self.v0), (self.u1, self.v1) = (tuple(map(float, p)) for p in rectangle)
        self.n, self.m = (int(k) for k in intervals)
        if self.n < 1 or self.m < 1 or not (self.u1 > self.u0 and self.v1 > self.v0):
            raise ValueError("a spline needs at least one interval per axis and a non-empty rectangle")
        controls = np.asarray(controls, dtype=float)
        if controls.ndim == 2:
            controls = controls[:, :, None]
        if controls.shape[:2] != (self.m + 3, self.n + 3):
            raise ValueError(f"controls of shape {controls.shape[:2]} for {self.n} × {self.m} intervals "
                             f"(expected {(self.m + 3, self.n + 3)})")
        self.controls = controls
        self.components = controls.shape[2]
        self.du = (self.u1 - self.u0) / self.n
        self.dv = (self.v1 - self.v0) / self.m

    def locate(self, uv: np.ndarray):
        """(i, s, j, t): the interval indexes (clamped to the rectangle's intervals) and the fractions (unclamped)."""
        uv = np.asarray(uv, dtype=float)
        su = (uv[..., 0] - self.u0) / self.du
        sv = (uv[..., 1] - self.v0) / self.dv
        finite = np.isfinite(su) & np.isfinite(sv)
        i = np.clip(np.floor(np.where(finite, su, 0.0)), 0, self.n - 1).astype(np.int64)
        j = np.clip(np.floor(np.where(finite, sv, 0.0)), 0, self.m - 1).astype(np.int64)
        return i, su - i, j, sv - j

    def _patch(self, i, j):
        """The 4 × 4 × C control patch of each point, shape (..., 4, 4, C) indexed [b (v), a (u), component]."""
        a = np.arange(4)
        return self.controls[(j[..., None, None] + a[:, None]), (i[..., None, None] + a[None, :])]

    def evaluate(self, uv: np.ndarray, derivatives: bool = False):
        """Values (..., C); with derivatives also d/du and d/dv (..., C) each. NaN inputs give NaN."""
        uv = np.asarray(uv, dtype=float)
        i, s, j, t = self.locate(uv)
        patch = self._patch(i, j)
        bu, bv = basis(s), basis(t)
        value = np.einsum("...b,...a,...bac->...c", bv, bu, patch)
        bad = ~np.all(np.isfinite(uv), axis=-1)
        if bad.any():
            value[bad] = np.nan
        if not derivatives:
            return value
        dbu, dbv = basis_derivative(s) / self.du, basis_derivative(t) / self.dv
        du = np.einsum("...b,...a,...bac->...c", bv, dbu, patch)
        dv = np.einsum("...b,...a,...bac->...c", dbv, bu, patch)
        if bad.any():
            du[bad] = np.nan
            dv[bad] = np.nan
        return value, du, dv

    def design_rows(self, uv: np.ndarray):
        """(rows, columns, weights) of the linear map from the flattened controls (index j·(n + 3) + i, per component)
        to the values at uv: 16 weights per point – the fit's design matrix."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        i, s, j, t = self.locate(uv)
        bu, bv = basis(s), basis(t)
        a = np.arange(4)
        cols = ((j[:, None, None] + a[:, None]) * (self.n + 3) + (i[:, None, None] + a[None, :])).reshape(len(uv), 16)
        weights = (bv[:, :, None] * bu[:, None, :]).reshape(len(uv), 16)
        rows = np.repeat(np.arange(len(uv)), 16).reshape(len(uv), 16)
        return rows.ravel(), cols.ravel(), weights.ravel()
