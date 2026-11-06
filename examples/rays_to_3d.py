"""Triangulation: the point closest to the rays of one pixel in two cameras (two files and their poses)."""
import numpy as np


def closest_point(o1, d1, o2, d2):
    """The midpoint of the shortest segment between the lines o1 + s d1 and o2 + t d2 (world frame)."""
    w = o1 - o2
    a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    s = (b * e - c * d) / (a * c - b * b)
    t = (a * e - b * d) / (a * c - b * b)
    return 0.5 * ((o1 + s * d1) + (o2 + t * d2))


def ray_in_world(camera, uv, pose):
    R, centre = pose
    o, d = camera.pixel_to_ray(np.asarray(uv, dtype=float))
    return o @ R.T + centre, d @ R.T
