# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Analytic truth cameras for the tests: pixel → ray in closed form, written from their formulas (not from the
package). Frame x right, y up, viewing along −z; pixel (0, 0) = centre of the top-left pixel, v down."""

import math

import numpy as np


class Truth:
    def __init__(self, name, resolution, theta_of_r, origin_of_theta=None, centre=None, squeeze=1.0):
        self.name, self.resolution = name, resolution
        self.centre = np.array(centre if centre is not None else ((resolution[0] - 1) / 2, (resolution[1] - 1) / 2))
        self.theta_of_r, self.origin_of_theta, self.squeeze = theta_of_r, origin_of_theta, squeeze

    @property
    def central(self):
        return self.origin_of_theta is None

    def rays(self, uv):
        uv = np.asarray(uv, dtype=float)
        x = uv[..., 0] - self.centre[0]
        y = -(uv[..., 1] - self.centre[1]) * self.squeeze
        r = np.hypot(x, y)
        theta = self.theta_of_r(r)
        phi = np.arctan2(y, x)
        d = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)], -1)
        if self.central:
            return np.zeros_like(d), d
        return np.stack([np.zeros_like(theta), np.zeros_like(theta), self.origin_of_theta(theta)], -1), d


def truths():
    f150 = 1000 / math.radians(75)
    out = [
        Truth("fisheye150", (1600, 1200), lambda r: (r / f150) * (1 - 0.02 * (r / f150) ** 2)),
        Truth("fisheye200", (1600, 1200), lambda r: np.arctan(0.4 * r / 600.0) / 0.4 * 1.0),
        Truth("tele800", (5760, 3840), lambda r: np.arctan(r / 128000.0)),
        Truth("axial", (1600, 1200), lambda r: np.arctan(r / 700.0) * (1 + 0.05 * (r / 1000) ** 2),
              origin_of_theta=lambda t: -2.0 * t * t),
        Truth("asymmetric", (1280, 960), lambda r: np.arctan(r / 900.0 * (1 + 0.08 * (r / 800) ** 2)),
              centre=(610.0, 500.0), squeeze=1.03),
    ]
    return {t.name: t for t in out}


def hull_points(truth, fraction=0.92):
    """The calibrated area: an ellipse covering `fraction` of the sensor (a fisheye's round coverage)."""
    W, H = truth.resolution
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    return np.c_[(W - 1) / 2 + fraction * (W - 1) / 2 * np.cos(a), (H - 1) / 2 + fraction * (H - 1) / 2 * np.sin(a)]


def samples(truth, step=8):
    W, H = truth.resolution
    u, v = np.meshgrid(np.arange(0, W, step, dtype=float), np.arange(0, H, step, dtype=float))
    return np.stack([u, v], -1).reshape(-1, 2)
