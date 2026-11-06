# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""The command line (`calibconsumer …` or `python3 -m calibconsumer …`; docs/evaluate.md).

  calibconsumer evaluate <result folder> [--out DIR]
  calibconsumer evaluate --camera <camera file> --points <folder or file> --world <file> [--poses <file>] …
  calibconsumer info <camera file or folder>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _parser():
    parser = argparse.ArgumentParser(prog="calibconsumer", description="Rays and pixels of a calibrated camera.")
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate", help="re-evaluate a camera against measured image points and world points",
                       description="Re-evaluates a camera – a calibration's result folder or a camera file of one's "
                       "own (mapping.json, OpenCV JSON/YAML/XML, Brown JSON/TXT) – against measured image points and "
                       "their world points: reprojection, angular, normalized and object-space errors per point, "
                       "their expected σ, outliers; nothing is optimized (views without a pose get one estimated). "
                       "Writes evaluation.json, observations.csv and the plots residuals.png, errorMap.png.")
    e.add_argument("result", nargs="?", help="a calibration's camera result folder "
                   "(resultsCalibration/<run>/cameraLensInstance-<id>): camera, poses, world and the project's points")
    e.add_argument("--camera", help="a camera file or folder (instead of, or overriding, the result's camera)")
    e.add_argument("--project", help="the calibration project (folder or projectData.json) – its points and world; "
                   "found from the result by default")
    e.add_argument("--camera-id", help="the project's camera (cameraLensInstanceId) when it has several")
    e.add_argument("--points", help="measured points: a folder of per-view files (<view>.csv: id u v) or one file "
                   "(view id u v, or id u v for a single view)")
    e.add_argument("--world", help="world points: id X Y Z [pattern]")
    e.add_argument("--world-sigma", help="world points' σ: id σ | id σx σy σz | id xx xy xz yy yz zz, or a "
                   "refined-target.json")
    e.add_argument("--poses", help="view poses (JSON: {view: {rotation, centre}} camera → world, or OpenCV rvec/tvec)")
    e.add_argument("--estimate-poses", choices=("missing", "all", "never"), default="missing",
                   help="estimate the poses of views without one (default), of all views, or of none")
    e.add_argument("--image-sigma", type=float, help="the image noise σ (px); default: the camera's, else estimated "
                   "from the residuals")
    e.add_argument("--outlier-threshold", type=float, default=4.0,
                   help="outlier above this Mahalanobis distance of the residual (default 4)")
    e.add_argument("--out", default="evaluation", help="output folder (default ./evaluation)")
    e.add_argument("--no-plots", action="store_true", help="skip the plots (no matplotlib needed)")
    e.add_argument("--title", help="a title line for the plots")
    i = sub.add_parser("info", help="what a camera file holds")
    i.add_argument("camera")
    return parser


def evaluate_command(args) -> int:
    import calibconsumer
    from . import inputs
    from .evaluate import evaluate
    obs, info = inputs.Observations(), {}
    if args.result:
        obs, info = inputs.from_result(args.result, args.project)
    elif args.project:
        obs = inputs.from_project(args.project, args.camera_id)
    camera_path = args.camera or args.result
    if camera_path is None:
        raise SystemExit("evaluate: give a result folder or --camera")
    camera = calibconsumer.load(camera_path, quiet=True)
    poses = dict(getattr(camera, "views", {}) or {})
    sources = dict(obs.sources, camera=os.path.abspath(camera_path))
    if poses:
        sources["poses"] = "the camera's views"
    if args.points:
        obs.views = inputs.read_points_any(args.points)
        sources["points"] = os.path.abspath(args.points)
    if args.world:
        obs.world = inputs.read_world(args.world)
        sources["world"] = os.path.abspath(args.world)
    if args.world_sigma:
        obs.world_covariance = inputs.read_world_sigma(args.world_sigma)
        sources["worldCovariance"] = os.path.abspath(args.world_sigma)
    if args.poses:
        poses.update(inputs.read_poses(args.poses))
        sources["poses"] = os.path.abspath(args.poses)
    if not obs.views:
        raise SystemExit("evaluate: no measured points (--points, --project or a result with its project)")
    if not obs.world:
        raise SystemExit("evaluate: no world points (--world, --project or a result)")
    result = evaluate(camera, obs.views, obs.world, poses, obs.world_covariance, args.estimate_poses,
                      args.image_sigma, args.outlier_threshold)
    os.makedirs(args.out, exist_ok=True)
    inputs_block = dict(sources, cameraKind=type(camera).__name__, estimatePoses=args.estimate_poses,
                        calibratedModel=info.get("projectionType"))
    result.to_json(os.path.join(args.out, "evaluation.json"), inputs_block)
    result.to_csv(os.path.join(args.out, "observations.csv"))
    written = ["evaluation.json", "observations.csv"]
    if not args.no_plots:
        try:
            from . import plots
        except ImportError:
            print("plots skipped: matplotlib is not installed (pip install calibconsumer[plot])", file=sys.stderr)
        else:
            title = args.title or info.get("name")
            try:
                plots.residuals(result, os.path.join(args.out, "residuals.png"), title)
                plots.error_map(result, os.path.join(args.out, "errorMap.png"), title)
                written += ["residuals.png", "errorMap.png"]
            except ImportError:
                print("plots skipped: matplotlib is not installed (pip install calibconsumer[plot])", file=sys.stderr)
    s = result.summary()
    a = s["all"]

    def show(block, key, unit, scale=1.0):
        v = block[key]
        return "–" if v is None else f"RMS {v['rms'] * scale:.4g} {unit}, median {v['median'] * scale:.4g} {unit}"
    print(f"{a['valid']} points in {len(result.view_names)} views ({sum(v == 'estimated' for v in result.pose_source.values())}"
          f" poses estimated), {a['outliers']} outliers (Mahalanobis > {args.outlier_threshold:g})")
    print(f"  reprojection   {show(a, 'reprojectionPx', 'px')}")
    print(f"  angle          {show(a, 'angleArcsec', '″')}")
    print(f"  normalized     {show(a, 'normalizedPx', 'px')}")
    print(f"  object space   {show(a, 'objectSpace', 'world units')}")
    print(f"  image σ        {result.image_sigma:.4g} px ({result.image_sigma_source})")
    for note in result.notes:
        print(f"  note: {note}")
    print(f"written to {os.path.abspath(args.out)}: {', '.join(written)}")
    return 0


def info_command(args) -> int:
    import calibconsumer
    camera = calibconsumer.load(args.camera, quiet=True)
    out = {"kind": type(camera).__name__, "resolution": list(camera.resolution), "central": bool(camera.central),
           "views": len(getattr(camera, "views", {}) or {})}
    if getattr(camera, "refit", None):
        out["refit"] = {k: camera.refit.get(k) for k in ("passed", "calibrationPoints", "grid") if k in camera.refit}
    u = camera.uncertainty
    if u is not None:
        W, H = camera.resolution
        centre = np.array([[(W - 1) / 2, (H - 1) / 2]])
        out["uncertainty"] = {"rayPxAtCentre": float(u.model_px(centre)[0]), "imageSigmaPx": u.image_sigma,
                              "source": u.source}
    print(json.dumps(out, indent=1, default=float))
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        return evaluate_command(args)
    return info_command(args)


if __name__ == "__main__":
    sys.exit(main())
