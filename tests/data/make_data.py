# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Writes the test mapping files tests/data/<truth>.json: each analytic truth of truths.py re-fitted by calibconsumer.fit
(needs SciPy). Run from the repository root: python3 tests/data/make_data.py"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
import truths  # noqa: E402
from calibconsumer import fileformat, fit  # noqa: E402

INTERVALS = {"fisheye150": (16, 12), "fisheye200": (16, 12), "tele800": (4, 3), "axial": (16, 12),
             "asymmetric": (16, 12)}


def make(truth):
    uv = truths.samples(truth)
    o, d = truth.rays(uv)
    hull = truths.hull_points(truth)
    document = fit.from_ray_samples(uv, d, None if truth.central else o, hull=hull, resolution=truth.resolution,
                                    intervals=INTERVALS[truth.name])
    document["camera"]["name"] = f"synthetic {truth.name}"
    document["producer"] = {"mapper": "calibconsumer tests/data/make_data.py"}
    from calibconsumer import Camera
    camera = Camera(document)
    check = truths.samples(truth, step=5) + 2.5
    check = check[camera.contains(check)]
    co, cd = truth.rays(check)
    e = fit.errors(camera, check, cd, None if truth.central else co, distances=(1000.0,))
    document["refit"] = {"grid": {"samples": int(len(check)), "pixelMax": float(e["pixel"].max()),
                                  "pixelRms": float(np.sqrt(np.mean(e["pixel"] ** 2))),
                                  "angleMaxRad": float(e["angle"].max())},
                         "objectSpace": [{"distance": 1000.0, "max": float(e["objectSpace"][1000.0].max())}],
                         "passed": True}
    if not truth.central:
        document["inverse"]["minUniqueDistance"] = 20.0
    fileformat.write(os.path.join(HERE, f"{truth.name}.json"), document)
    print(f"{truth.name}: {INTERVALS[truth.name]}, pixel max {e['pixel'].max():.2e}, rms "
          f"{np.sqrt(np.mean(e['pixel'] ** 2)):.2e}")


if __name__ == "__main__":
    for t in truths.truths().values():
        make(t)
