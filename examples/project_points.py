"""World points of a calibration view to pixels (python3 examples/project_points.py tests/data/fisheye150.json)."""
import sys

import numpy as np

import calibconsumer

camera = calibconsumer.load(sys.argv[1])
points = np.array([[0.0, 0.0, -1000.0], [300.0, 200.0, -800.0], [0.0, 0.0, 500.0]])  # camera frame, the last behind
uv, status = camera.world_to_pixel(points, return_status=True)
for p, q, s in zip(points, uv, status):
    print(p, "->", q, calibconsumer.Status(s).name)
