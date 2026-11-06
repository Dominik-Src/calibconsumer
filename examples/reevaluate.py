"""Re-evaluating a camera against measured points, with the expected σ of every point in the image and in the world.

    python3 examples/reevaluate.py [out folder]

Builds a small synthetic case so that it runs anywhere: an OpenCV camera with a known calibration uncertainty, a
planar target of 17 × 13 points with a 0.05 mm σ each, six views with 0.3 px image noise and a few gross outliers.
With real data, the same calls take a camera file and your own point files (see docs/evaluate.md), or the command line
does it all: calibconsumer evaluate --camera camera.json --points points.csv --world world.txt --out evaluation
"""
import os
import sys

import numpy as np

import calibconsumer
from calibconsumer import pose
from calibconsumer.evaluate import evaluate

out = sys.argv[1] if len(sys.argv) > 1 else "evaluation-example"

# a camera of one's own: OpenCV's parameters, and what is known of its uncertainty (px)
camera = calibconsumer.OpenCVCamera([[800, 0, 640], [0, 810, 480], [0, 0, 1]], [-0.3, 0.1, 0.001, -0.002, -0.02],
                                    (1280, 960))
camera.uncertainty = calibconsumer.Uncertainty(ray=0.05, image_sigma=0.3)

# the target and its points' σ (world units, here mm)
grid = np.stack(np.meshgrid(np.arange(-8, 9), np.arange(-6, 7)), -1).reshape(-1, 2) * 20.0
world = {str(i): np.array([x, y, 0.0]) for i, (x, y) in enumerate(grid)}
world_covariance = {i: np.eye(3) * 0.05 ** 2 for i in world}

# six views: the measured pixels with noise and some outliers (here simulated; normally read from files)
rng = np.random.default_rng(1)
views = {}
ids = np.array(list(world))
X = np.array([world[i] for i in ids])
for k in range(6):
    R = pose._rotation(np.r_[np.pi + rng.normal(scale=0.25, size=2), rng.normal(scale=0.5)])
    centre = np.r_[rng.normal(scale=40, size=2), 0.0] - R @ np.array([0, 0, -330.0])
    uv = camera.world_to_pixel(X, pose=(R, centre), allow_extrapolation=True)
    keep = camera.contains(uv)
    uv = uv + rng.normal(scale=0.3, size=uv.shape)
    bad = keep & (rng.random(len(uv)) < 0.03)
    uv[bad] += rng.uniform(-30, 30, (bad.sum(), 2))
    views[f"image-{k:04d}"] = (ids[keep], uv[keep])

# no poses given: each view's pose is estimated from its points with the camera held fixed
result = evaluate(camera, views, world, poses_given={}, world_covariance=world_covariance)
summary = result.summary()["all"]
print(f"{summary['valid']} points, {summary['outliers']} outliers; reprojection RMS "
      f"{summary['reprojectionPx']['rms']:.3f} px, object space RMS {summary['objectSpace']['rms']:.4f} mm")

# per point: the residual, its expected σ in the image (an ellipse) and in the world (normal to the ray)
print("view        id   residual [px]   σ image major/minor [px]   σ world [mm]   Mahalanobis   outlier")
for i in range(8):
    print(f"{result.view_names[result.view[i]]}  {result.ids[i]:>4}   {result.reprojection[i]:8.3f}        "
          f"{result.sigma_image[i, 0]:.3f} / {result.sigma_image[i, 1]:.3f}            {result.sigma_object[i]:.4f}"
          f"         {result.mahalanobis[i]:6.2f}      {bool(result.outlier[i])}")

# the same σ straight from the camera, for any point and pose
R, centre = result.poses["image-0000"]
uv, cov = camera.world_to_pixel(X[:3], pose=(R, centre), return_covariance=True,
                                world_covariance=world_covariance["0"], image_noise=True)
print("σ ellipses of three predicted pixels:", np.round(calibconsumer.ellipse(cov), 3).tolist())

os.makedirs(out, exist_ok=True)
result.to_json(os.path.join(out, "evaluation.json"))
result.to_csv(os.path.join(out, "observations.csv"))
try:
    from calibconsumer import plots
    plots.residuals(result, os.path.join(out, "residuals.png"), "synthetic example")
    plots.error_map(result, os.path.join(out, "errorMap.png"), "synthetic example")
except ImportError:
    print("plots skipped: pip install calibconsumer[plot]")
print("written to", os.path.abspath(out))
