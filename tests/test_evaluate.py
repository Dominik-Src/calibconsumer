# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""M18: the re-evaluation – the input readers, the pose estimation, the errors, σ and outliers on a synthetic camera
with known poses, noise and outliers, one's own OpenCV and Brown parameters, and the command line."""

import json
import os

import numpy as np
import pytest

import calibconsumer
from calibconsumer import cli, inputs, pose
from calibconsumer.evaluate import evaluate

K = np.array([[800.0, 0, 640], [0, 810, 480], [0, 0, 1]])
DIST = [-0.3, 0.1, 0.001, -0.002, -0.02]
RESOLUTION = (1280, 960)


def opencv_camera():
    return calibconsumer.OpenCVCamera(K, DIST, RESOLUTION)


def scene(camera, views=6, planar=True, noise=0.3, outliers=0.05, seed=1):
    """A planar (or 3-D) target seen from `views` poses: {view: (ids, uv)}, world, true poses, planted outliers."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(-8, 9), np.arange(-6, 7)), -1).reshape(-1, 2) * 20.0
    world = np.c_[g, np.zeros(len(g)) if planar else rng.uniform(-40, 40, len(g))]
    ids = np.array([str(i) for i in range(len(world))])
    out, truth, planted = {}, {}, {}
    for k in range(views):
        R = pose._rotation(np.r_[np.pi + rng.normal(scale=0.25, size=2), rng.normal(scale=0.5)])
        centre = np.r_[rng.normal(scale=60, size=2), 0.0] - R @ np.array([0, 0, -550.0])  # looking at the target
        uv = camera.world_to_pixel(world, pose=(R, centre), allow_extrapolation=True)
        keep = camera.contains(uv)
        uv = uv + rng.normal(scale=noise, size=uv.shape)
        bad = keep & (rng.random(len(uv)) < outliers)
        uv[bad] += rng.choice([-1, 1], size=(bad.sum(), 2)) * rng.uniform(15, 40, (bad.sum(), 2))
        name = f"image-{k:04d}"
        out[name] = (ids[keep], uv[keep])
        truth[name] = (R, centre)
        planted[name] = set(ids[bad])
    return out, {i: x for i, x in zip(ids, world)}, truth, planted


def angle_between(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


def test_m18a_readers(tmp_path):
    (tmp_path / "points.txt").write_text("# view id u v\nimage-0001.csv,7,10.5,20.25\nimage-0001.csv,8,11,21\n"
                                         "image-0002,7.0,1,2\n")
    table = inputs.read_points_any(tmp_path / "points.txt")
    assert sorted(table) == ["image-0001", "image-0002"] and list(table["image-0002"][0]) == ["7"]
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "image-0003.csv").write_text("id u v\n  5\t1.5\t2.5\n5 9 9\n")
    (folder / "image-0003.cov.csv").write_text("5 1 0 1\n")
    views = inputs.read_points_any(folder)
    assert list(views) == ["image-0003"] and views["image-0003"][1].tolist() == [[1.5, 2.5]]
    (tmp_path / "world.csv").write_text("0\t0\t272.9\t0\tpattern_a\n1;8.0;272.9;0\n")
    assert inputs.read_world(tmp_path / "world.csv")["1"].tolist() == [8.0, 272.9, 0.0]
    (tmp_path / "sigma.txt").write_text("0 0.1\n1 0.1 0.2 0.3\n2 1 0 0 2 0 3\n")
    sigma = inputs.read_world_sigma(tmp_path / "sigma.txt")
    assert sigma["0"][2, 2] == pytest.approx(0.01) and sigma["1"][1, 1] == pytest.approx(0.04)
    assert sigma["2"][2, 2] == 3
    camera = opencv_camera()
    R, c = pose._rotation(np.array([2.9, 0.3, 0.2])), np.array([10.0, -20.0, 500.0])
    camera.views = {"v": (R, c)}
    rvec, tvec = camera.rvec_tvec("v")
    (tmp_path / "poses.json").write_text(json.dumps({"v.csv": {"rvec": rvec.tolist(), "tvec": tvec.tolist()},
                                                     "w": {"rotation": R.tolist(), "centre": c.tolist()}}))
    poses = inputs.read_poses(tmp_path / "poses.json")
    for name in ("v", "w"):
        assert angle_between(poses[name][0], R) < 1e-9 and np.allclose(poses[name][1], c)


@pytest.mark.parametrize("planar", [True, False])
def test_m18b_pose_estimation(case, planar):
    camera, _ = case
    rng = np.random.default_rng(7)
    W, H = camera.resolution
    uv = rng.uniform([0.1 * W, 0.1 * H], [0.9 * W, 0.9 * H], (150, 2))
    uv = uv[camera.contains(uv)]
    o, d = camera.pixel_to_ray(uv)
    if planar:
        n = np.array([0.1, -0.2, 1.0]) / np.linalg.norm([0.1, -0.2, 1.0])
        s = ((np.array([0, 0, -500.0]) - o) @ n) / (d @ n)
    else:
        s = 500 * rng.uniform(0.7, 1.3, len(uv))
    Xc = o + s[:, None] * d
    R, c = pose._rotation(rng.normal(size=3)), rng.normal(size=3) * 100
    X = Xc @ R.T + c
    noisy = uv + rng.normal(scale=0.3, size=uv.shape)
    bad = rng.random(len(uv)) < 0.1
    noisy[bad] += rng.uniform(-200, 200, (bad.sum(), 2))
    (Re, ce), info = pose.estimate(camera, noisy, X)
    assert angle_between(Re, R) < 0.1 and np.linalg.norm(ce - c) < 1.0
    assert info["inliers"] >= (~bad).sum()


def test_m18c_errors_sigma_and_outliers():
    camera = opencv_camera()
    views, world, truth, planted = scene(camera)
    e = evaluate(camera, views, world, truth, estimate_poses="missing", outlier_threshold=4.0)
    assert set(e.pose_source.values()) == {"given"}
    flagged = {(e.view_names[v], i) for v, i, o in zip(e.view, e.ids, e.outlier) if o}
    true = {(name, i) for name, ids in planted.items() for i in ids}
    assert len(flagged & true) >= 0.95 * len(true) and len(flagged - true) <= 0.01 * len(e.view)
    s = e.summary()["all"]
    assert e.image_sigma == pytest.approx(0.3, rel=0.15)  # estimated from the residuals
    assert s["reprojectionPx"]["rms"] == pytest.approx(0.3 * np.sqrt(2), rel=0.15)
    inlier = e.valid & ~e.outlier
    assert np.allclose(e.object_space[inlier], np.sin(e.angle[inlier]) * e.distance[inlier], rtol=1e-6)
    assert np.allclose(e.normalized, e.angle / camera.pixel_angle(e.uv))
    # the expected σ: the image noise alone here, isotropic
    assert np.allclose(e.sigma_image[:, 0], e.image_sigma) and np.allclose(e.sigma_image[:, 1], e.image_sigma)
    assert np.median(e.mahalanobis[inlier]) == pytest.approx(np.sqrt(2 * np.log(2)), rel=0.15)
    # the same with every pose estimated: the poses near the truth, the errors no larger
    f = evaluate(camera, views, world, {}, estimate_poses="missing")
    assert set(f.pose_source.values()) == {"estimated"}
    for name, (R, c) in truth.items():
        assert angle_between(f.poses[name][0], R) < 0.05 and np.linalg.norm(f.poses[name][1] - c) < 2.0
    assert f.summary()["all"]["reprojectionPx"]["rms"] <= s["reprojectionPx"]["rms"] * 1.01
    # world point covariances widen the expected σ
    covariance = {i: np.diag([0.5, 0.5, 0.5]) ** 2 for i in world}
    g = evaluate(camera, views, world, truth, covariance, image_sigma=0.3)
    assert np.all(g.sigma_image[:, 0] > 0.3) and np.all(g.sigma_object > e.sigma_object * 0 + 0.4)
    assert g.image_sigma_source == "given"


def test_m18d_own_parameters():
    data = {"camera_matrix": K.tolist(), "distortion_coefficients": DIST, "image_width": 1280, "image_height": 960}
    camera = calibconsumer.OpenCVCamera.from_dict(data)
    assert np.allclose(camera.camera_matrix, K) and camera.resolution == RESOLUTION
    brown = calibconsumer.BrownCamera.from_dict({"model": "Brown", "resolution": [1280, 960],
                                                 "pixelSizeMM": [0.005, 0.005],
                                                 "parameters": {"C": 4.0, "XP": 0.01, "YP": -0.02, "K1": 1e-3,
                                                                "K2": 0.0}})
    assert brown.parameters["P1"] == 0.0
    views, world, truth, _ = scene(brown, views=3, outliers=0.0, noise=0.2)
    e = evaluate(brown, views, world, {})
    assert e.summary()["all"]["reprojectionPx"]["median"] < 0.4


def test_m18e_opencv_storage(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "camera.yml")
    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    storage.write("image_width", 1280)
    storage.write("image_height", 960)
    storage.write("camera_matrix", K)
    storage.write("distortion_coefficients", np.array(DIST).reshape(1, -1))
    storage.release()
    camera = calibconsumer.load(path)
    assert np.allclose(camera.camera_matrix, K) and np.allclose(camera.distortion, DIST)


def test_m18f_command_line(tmp_path, capsys):
    camera = opencv_camera()
    views, world, truth, planted = scene(camera, views=4)
    (tmp_path / "camera.json").write_text(json.dumps({"cameraMatrix": K.tolist(), "distortionCoefficients": DIST,
                                                      "imageSize": list(RESOLUTION)}))
    with open(tmp_path / "points.csv", "w") as f:
        for name, (ids, uv) in views.items():
            for i, (u, v) in zip(ids, uv):
                f.write(f"{name},{i},{u},{v}\n")
    with open(tmp_path / "world.txt", "w") as f:
        for i, x in world.items():
            f.write(f"{i}\t{x[0]}\t{x[1]}\t{x[2]}\n")
    out = tmp_path / "out"
    args = ["evaluate", "--camera", str(tmp_path / "camera.json"), "--points", str(tmp_path / "points.csv"),
            "--world", str(tmp_path / "world.txt"), "--out", str(out), "--no-plots"]
    assert cli.main(args) == 0
    document = json.loads((out / "evaluation.json").read_text())
    assert document["format"] == "calibconsumer-evaluation" and len(document["poses"]) == 4
    assert document["summary"]["all"]["outliers"] >= sum(len(p) for p in planted.values()) * 0.95
    rows = (out / "observations.csv").read_text().splitlines()
    assert rows[0].startswith("view,id,u,v,uPredicted") and len(rows) == 1 + sum(len(v[0]) for v in views.values())
    assert "4 poses estimated" in capsys.readouterr().out
    pytest.importorskip("matplotlib")
    assert cli.main(args[:-1] + ["--title", "synthetic"]) == 0
    assert os.path.getsize(out / "residuals.png") > 10000 and os.path.getsize(out / "errorMap.png") > 10000
