# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""The convex polygon of the calibrated image area (docs/format.md, "domain.hull")."""

from __future__ import annotations

import numpy as np


class ConvexPolygon:
    """A convex polygon in pixels; its vertices in either orientation, without a repeated closing vertex."""

    def __init__(self, vertices):
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
            raise ValueError("a hull needs at least three (u, v) vertices")
        area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
        if area == 0:
            raise ValueError("the hull has no area")
        self.vertices = v if area > 0 else v[::-1]
        edges = np.roll(self.vertices, -1, axis=0) - self.vertices
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        if np.any(cross < -1e-9 * np.abs(cross).max()):
            raise ValueError("the hull is not convex")
        self.edges = edges

    def contains(self, uv: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """True for points inside or on the polygon, or at most `tolerance` pixels outside it (the default absorbs the
        rounding of the cross products at its own vertices); False for NaN."""
        uv = np.asarray(uv, dtype=float)
        rel = uv[..., None, :] - self.vertices
        cross = self.edges[:, 0] * rel[..., 1] - self.edges[:, 1] * rel[..., 0]
        length = np.hypot(self.edges[:, 0], self.edges[:, 1])
        return np.all(cross >= -tolerance * length, axis=-1)

    @staticmethod
    def of_points(points: np.ndarray) -> "ConvexPolygon":
        """The convex hull of points (monotone chain)."""
        p = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
        if len(p) < 3:
            raise ValueError("a hull needs at least three distinct points")

        def half(points):
            out = []
            for q in points:
                while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                         - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                    out.pop()
                out.append(q)
            return out
        lower, upper = half(p), half(p[::-1])
        return ConvexPolygon(np.array(lower[:-1] + upper[:-1]))
