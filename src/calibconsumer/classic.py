# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""The classic lens models straight from their parameters: OpenCV's camera model (results_OpenCV.json) and the
photogrammetric Brown model (results_BrownCamera.txt) – the same API as the spline Camera
(pixel_to_ray, world_to_pixel, views, frames, statuses), vectorized NumPy throughout. The models are written from their
published definitions (OpenCV's documentation of calib3d; Brown 1971 / Fraser 1997 with the photogrammetric parameter
names C, XP, YP, K1–K3, P1, P2, B1, B2), not from any calibration library.

The calibrated area of a classic model is the sensor; beyond the radius where its radial distortion stops being
monotonic (`valid_radius`, the model folds over there) a point is OUTSIDE_DOMAIN unless allow_extrapolation=True.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .camera import OPENCV, Status
from .uncertainty import Uncertain

NEWTON_ITERATIONS = 50
NEWTON_TOLERANCE = 1e-14


def _rodrigues(rvec):
    v = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-15:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K


def _rotation_vector(R):
    R = np.asarray(R, dtype=float)
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if angle < 1e-15:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:  # near π: the axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        axis *= np.sign([1, R[0, 1] + R[1, 0] or 1, R[0, 2] + R[2, 0] or 1])
        return angle * axis / np.linalg.norm(axis)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle))
    return angle * axis


def _newton(f, target, start, iterations=NEWTON_ITERATIONS, tolerance=NEWTON_TOLERANCE):
    """Solve f(x) = target for 2-vectors, vectorized: Newton with a central-difference Jacobian. Returns (x, ok)."""
    x = np.array(start, dtype=float)
    active = np.all(np.isfinite(x), axis=-1) & np.all(np.isfinite(target), axis=-1)
    scale = np.maximum(1.0, np.abs(target).max(axis=-1, keepdims=True))
    for _ in range(iterations):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        xi = x[idx]
        r = f(xi) - target[idx]
        h = 1e-7 * np.maximum(1.0, np.abs(xi))
        J = np.empty((len(idx), 2, 2))
        for k in range(2):
            e = np.zeros_like(xi)
            e[:, k] = h[:, k]
            J[:, :, k] = (f(xi + e) - f(xi - e)) / (2 * h[:, k:k + 1])
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        good = np.abs(det) > 1e-300
        step = np.zeros_like(xi)
        step[good, 0] = (J[good, 1, 1] * r[good, 0] - J[good, 0, 1] * r[good, 1]) / det[good]
        step[good, 1] = (-J[good, 1, 0] * r[good, 0] + J[good, 0, 0] * r[good, 1]) / det[good]
        x[idx] = xi - step
        active[idx] = good & (np.max(np.abs(step), axis=-1) > tolerance * scale[idx, 0])
    residual = np.max(np.abs(f(x) - target), axis=-1)
    ok = np.isfinite(residual) & (residual <= 1e-9 * scale[:, 0])
    return x, ok


def _valid_radius(radial_map, reach):
    """The smallest radius below `reach` where r ↦ radial_map(r) stops increasing (the model folds there), or inf."""
    r = np.linspace(0.0, reach, 20001)[1:]
    rho = radial_map(r)
    rising = np.diff(rho) > 0
    if rising.all():
        return math.inf
    return float(r[np.argmin(rising)])


class _Classic(Uncertain):
    """What the classic models share: poses, frames, the sensor as the calibrated area, the statuses."""

    central = True
    refit = None

    def __init__(self, resolution, views=None, frame="calibconsumer"):
        self.resolution = tuple(int(v) for v in resolution)
        self.views = views or {}
        if frame not in ("calibconsumer", "opencv"):
            raise ValueError("frame is 'calibconsumer' or 'opencv'")
        self.frame = frame
        self._F = OPENCV if frame == "opencv" else np.eye(3)

    def with_frame(self, frame):
        other = object.__new__(type(self))
        other.__dict__.update(self.__dict__)
        other.frame, other._F = frame, (OPENCV if frame == "opencv" else np.eye(3))
        return other

    def contains(self, uv) -> np.ndarray:
        """Inside the sensor (a classic result names no other calibrated area)."""
        uv = np.asarray(uv, dtype=float)
        W, H = self.resolution
        return np.all(np.isfinite(uv), axis=-1) & (uv[..., 0] >= -0.5) & (uv[..., 0] <= W - 0.5) & \
            (uv[..., 1] >= -0.5) & (uv[..., 1] <= H - 0.5)

    def _camera_points(self, X, pose):
        X = np.asarray(X, dtype=float)
        if pose is not None:
            R, c = (np.asarray(p, dtype=float) for p in pose)
            return (X - c) @ R  # the file's frame (x right, y up, −z)
        return X @ self._F

    def _finish_rays(self, d, status, allow_extrapolation, return_status):
        refused = (status == Status.OUTSIDE_DOMAIN) & (not allow_extrapolation) | \
            (status == Status.NO_CONVERGENCE) | ((status == Status.OUTSIDE_HULL) & (not allow_extrapolation))
        d = np.where(refused[..., None], np.nan, d / np.linalg.norm(d, axis=-1, keepdims=True)) @ self._F.T
        o = np.where(np.isfinite(d), 0.0, np.nan)
        return (o, d, status) if return_status else (o, d)

    def _finish_pixels(self, uv, status, allow_extrapolation, return_status):
        outside = ~self.contains(uv) & (status == Status.OK)
        status = np.where(outside, Status.OUTSIDE_HULL, status)
        keep = (status == Status.OK) | (allow_extrapolation & ((status == Status.OUTSIDE_HULL) |
                                                                (status == Status.OUTSIDE_DOMAIN)))
        uv = np.where(keep[..., None], uv, np.nan)
        return (uv, status) if return_status else uv


class OpenCVCamera(_Classic):
    """OpenCV's pinhole camera with lens distortion: the camera matrix and up to 14 coefficients (k1, k2, p1, p2, k3,
    k4, k5, k6, s1, s2, s3, s4, τx, τy), pixel (0, 0) the centre of the top-left pixel. Rays and points are in the
    file's frame of the package (x right, y up, −z) unless with_frame("opencv")."""

    def __init__(self, camera_matrix, distortion, resolution, views=None, frame="calibconsumer"):
        super().__init__(resolution, views, frame)
        self.camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        d = np.asarray(distortion, dtype=float).ravel()
        if len(d) not in (0, 4, 5, 8, 12, 14):
            raise ValueError(f"{len(d)} distortion coefficients (OpenCV takes 4, 5, 8, 12 or 14)")
        self.distortion = d
        self._d = np.zeros(14)
        self._d[:len(d)] = d
        self._tilt = self._tilt_matrix(self._d[12], self._d[13])
        self._tilt_inverse = np.linalg.inv(self._tilt)
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        corner = np.hypot(max(self.camera_matrix[0, 2], resolution[0] - self.camera_matrix[0, 2]) / fx,
                          max(self.camera_matrix[1, 2], resolution[1] - self.camera_matrix[1, 2]) / fy)
        k = self._d
        self.valid_radius = _valid_radius(
            lambda r: r * (1 + k[0] * r ** 2 + k[1] * r ** 4 + k[4] * r ** 6) / (1 + k[5] * r ** 2 + k[6] * r ** 4 + k[7] * r ** 6),
            4.0 * corner)

    @staticmethod
    def _tilt_matrix(tx, ty):
        """OpenCV's tilted-sensor projection (calib3d, computeTiltProjectionMatrix)."""
        cx, sx, cy, sy = math.cos(tx), math.sin(tx), math.cos(ty), math.sin(ty)
        rot_x = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
        rot_y = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
        rot = rot_y @ rot_x
        proj = np.array([[rot[2, 2], 0, -rot[0, 2]], [0, rot[2, 2], -rot[1, 2]], [0, 0, 1]])
        return proj @ rot

    @classmethod
    def from_file(cls, path, frame="calibconsumer"):
        """results_OpenCV.json (the calibration's OpenCV result, docs/evaluate.md): cameraMatrix, distortionCoefficients, imageSize, rvecs/tvecs per view
        (OpenCV's world → camera); views become camera → world poses in the package's frame."""
        if str(path).lower().endswith((".yml", ".yaml", ".xml")):
            return cls.from_opencv_storage(path, frame)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, frame)

    @classmethod
    def from_dict(cls, data, frame="calibconsumer"):
        """{cameraMatrix | camera_matrix | K, distortionCoefficients | distortion_coefficients | dist, imageSize [W, H] |
        image_width + image_height, optional rvecs/tvecs per view}."""
        def first(*keys):
            return next((data[k] for k in keys if k in data), None)
        K = first("cameraMatrix", "camera_matrix", "K")
        dist = first("distortionCoefficients", "distortion_coefficients", "dist", "D")
        size = first("imageSize", "image_size", "resolution")
        if size is None and "image_width" in data and "image_height" in data:
            size = [data["image_width"], data["image_height"]]
        if K is None or dist is None or size is None:
            raise ValueError("an OpenCV camera needs cameraMatrix, distortionCoefficients and imageSize (or "
                             "camera_matrix, distortion_coefficients, image_width and image_height)")
        views = {}
        for name, rvec in (data.get("rvecs") or {}).items():
            tvec = (data.get("tvecs") or {}).get(name)
            if tvec is None:
                continue
            R = _rodrigues(rvec)
            centre = -R.T @ np.asarray(tvec, dtype=float)
            views[os.path.splitext(name)[0].rstrip(".")] = (R.T @ OPENCV, centre)  # camera (package frame) → world
        return cls(np.asarray(K, dtype=float).reshape(3, 3), np.asarray(dist, dtype=float).ravel(),
                   [int(v) for v in np.ravel(size)], views, frame)

    @classmethod
    def from_opencv_storage(cls, path, frame="calibconsumer"):
        """OpenCV's own YAML or XML (cv2.FileStorage, as its calibration samples write: camera_matrix,
        distortion_coefficients, image_width, image_height); needs OpenCV."""
        cv2 = _cv2()
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            def node(*names):
                for name in names:
                    n = storage.getNode(name)
                    if not n.empty():
                        return n
                return None
            K, dist = node("camera_matrix", "cameraMatrix", "K"), node("distortion_coefficients",
                                                                        "distortionCoefficients", "dist", "D")
            width, height = node("image_width"), node("image_height")
            if K is None or dist is None or width is None or height is None:
                raise ValueError(f"{path}: needs camera_matrix, distortion_coefficients, image_width and image_height")
            return cls(K.mat(), dist.mat().ravel(), (int(width.real()), int(height.real())), None, frame)
        finally:
            storage.release()

    def to_opencv(self):
        """(camera_matrix, distortion coefficients) as cv2 takes them."""
        return self.camera_matrix.copy(), self.distortion.copy()

    def rvec_tvec(self, name):
        """A view's pose as OpenCV's world → camera rvec, tvec (cv2.projectPoints)."""
        R_c2w, centre = self.views[name]
        R = (R_c2w @ OPENCV).T
        return _rotation_vector(R), -R @ centre

    def distort(self, xy):
        """Ideal normalized coordinates (OpenCV frame: x right, y down) → distorted normalized coordinates."""
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = self._d[:12]
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + r2 * (s1 + s2 * r2)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + r2 * (s3 + s4 * r2)
        if self._d[12] or self._d[13]:
            v = np.stack([xd, yd, np.ones_like(xd)], -1) @ self._tilt.T
            return v[..., :2] / v[..., 2:]
        return np.stack([xd, yd], -1)

    def _to_pixels(self, xy):
        K = self.camera_matrix
        return np.stack([K[0, 0] * xy[..., 0] + K[0, 1] * xy[..., 1] + K[0, 2], K[1, 1] * xy[..., 1] + K[1, 2]], -1)

    def _from_pixels(self, uv):
        K = self.camera_matrix
        y = (uv[..., 1] - K[1, 2]) / K[1, 1]
        return np.stack([(uv[..., 0] - K[0, 2] - K[0, 1] * y) / K[0, 0], y], -1)

    def pixel_to_ray(self, uv, allow_extrapolation=False, return_status=False, backend="numpy", return_sigma=False):
        """(origin = 0, unit direction) of pixels uv (..., 2); backend "opencv" undistorts with cv2."""
        uv = np.asarray(uv, dtype=float)
        shape = uv.shape[:-1]
        flat = uv.reshape(-1, 2)
        distorted = self._from_pixels(flat)
        if backend == "opencv":
            ideal, ok = _cv2_undistort(self, flat), np.ones(len(flat), bool)
        else:
            ideal, ok = _newton(self.distort, distorted, distorted)
        status = np.where(ok, Status.OK, Status.NO_CONVERGENCE)
        status = np.where(ok & (np.hypot(ideal[:, 0], ideal[:, 1]) > self.valid_radius), Status.OUTSIDE_DOMAIN, status)
        status = np.where((status == Status.OK) & ~self.contains(flat), Status.OUTSIDE_HULL, status)
        d_cv = np.stack([ideal[:, 0], ideal[:, 1], np.ones(len(flat))], -1)
        d = d_cv @ OPENCV  # OpenCV's frame → the package's
        return self._with_sigma(self._finish_rays(d.reshape(shape + (3,)), status.reshape(shape), allow_extrapolation,
                                                    return_status), uv, return_sigma)

    def world_to_pixel(self, X, pose=None, allow_extrapolation=False, return_status=False, backend="numpy",
                       return_covariance=False, world_covariance=None, image_noise=False):
        """Pixels (..., 2) of points X (..., 3) in the camera frame, or world points with pose = a view's
        (rotation camera → world, centre); closed form, or cv2.projectPoints with backend "opencv"."""
        Xc = self._camera_points(X, pose) @ OPENCV  # the package's frame → OpenCV's
        shape = Xc.shape[:-1]
        Xc = Xc.reshape(-1, 3)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = Xc[:, :2] / z[:, None]
        if backend == "opencv":
            uv = _cv2_project(self, Xc)
        else:
            uv = self._to_pixels(self.distort(xy))
        status = np.where(z > 0, Status.OK, Status.BEHIND)
        status = np.where((status == Status.OK) & (np.hypot(xy[:, 0], xy[:, 1]) > self.valid_radius),
                          Status.OUTSIDE_DOMAIN, status)
        status = np.where(np.all(np.isfinite(Xc), axis=-1), status, Status.NO_CONVERGENCE)
        uv = np.where((status == Status.BEHIND)[:, None] | ~np.isfinite(uv), np.nan, uv)
        result = self._finish_pixels(uv.reshape(shape + (2,)), status.reshape(shape), allow_extrapolation,
                                     return_status)
        return self._with_covariance(result, X, pose, return_covariance, world_covariance, image_noise)

    def compare_with_opencv(self, uv=None, X=None):
        """The largest differences to cv2 (needs OpenCV): pixels of points X (px) and normalized rays of pixels uv."""
        out = {}
        if X is not None:
            a = self.world_to_pixel(X, allow_extrapolation=True)
            b = self.world_to_pixel(X, allow_extrapolation=True, backend="opencv")
            out["projectPx"] = float(np.nanmax(np.linalg.norm(a - b, axis=-1)))
        if uv is not None:
            _, a = self.pixel_to_ray(uv, allow_extrapolation=True)
            _, b = self.pixel_to_ray(uv, allow_extrapolation=True, backend="opencv")
            out["undistortRad"] = float(np.nanmax(np.arccos(np.clip(np.sum(a * b, -1), -1, 1))))
        return out


def _cv2():
    try:
        import cv2
    except ImportError:  # pragma: no cover - depends on the installation
        raise ImportError("backend 'opencv' needs OpenCV: pip install opencv-python-headless") from None
    return cv2


def _cv2_project(camera, Xc):
    cv2 = _cv2()
    uv, _ = cv2.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.camera_matrix, camera.distortion)
    return uv.reshape(-1, 2)


def _cv2_undistort(camera, uv):
    cv2 = _cv2()
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-15)
    xy = cv2.undistortPointsIter(uv.reshape(-1, 1, 2), camera.camera_matrix, camera.distortion, None, None, criteria)
    return xy.reshape(-1, 2)


class BrownCamera(_Classic):
    """The photogrammetric Brown model: principal distance C, principal point XP, YP
    (mm), radial K1–K3, decentring P1, P2, affinity and shear B1, B2, on sensor coordinates in mm (x right, y up, the
    origin at the sensor's centre). A measured point x is corrected to the ideal x + Δx(x); its ray looks along
    (x_ideal, y_ideal, −C) in the package's frame."""

    NAMES = ("C", "XP", "YP", "K1", "K2", "K3", "P1", "P2", "B1", "B2")

    def __init__(self, parameters, pixel_size_mm, resolution, views=None, frame="calibconsumer"):
        super().__init__(resolution, views, frame)
        missing = [n for n in self.NAMES if n not in parameters]
        if missing:
            raise ValueError("Brown parameters missing: " + ", ".join(missing))
        self.parameters = {n: float(parameters[n]) for n in self.NAMES}
        self.pixel_size = np.asarray(pixel_size_mm, dtype=float).reshape(2)
        W, H = self.resolution
        corner = np.hypot(W / 2 * self.pixel_size[0], H / 2 * self.pixel_size[1])
        p = self.parameters
        self.valid_radius = _valid_radius(lambda r: r * (1 + p["K1"] * r ** 2 + p["K2"] * r ** 4 + p["K3"] * r ** 6),
                                          4.0 * corner)

    @classmethod
    def from_dict(cls, data, frame="calibconsumer"):
        """{"model": "Brown", "parameters": {C, XP, YP, K1, K2, K3, P1, P2, B1, B2} (mm), "pixelSizeMM": [sx, sy],
        "resolution": [W, H], optional "views": {name: {"rotation", "centre"}}} (missing K3, P1, P2, B1, B2 are 0)."""
        parameters = {n: 0.0 for n in ("K3", "P1", "P2", "B1", "B2")}
        parameters.update(data["parameters"])
        views = {name: (np.asarray(v["rotation"], dtype=float).reshape(3, 3), np.asarray(v["centre"], dtype=float))
                 for name, v in (data.get("views") or {}).items()}
        return cls(parameters, data["pixelSizeMM"], data["resolution"], views, frame)

    @classmethod
    def from_file(cls, path, frame="calibconsumer"):
        """results_BrownCamera.txt (the calibration's Brown report): the final values, the sensor's size in
        pixels and its pixel size, and per station the centre and the angles Alpha, Elev., Roll (degrees)."""
        parameters, pixel_size, resolution, views = {}, [None, None], [None, None], {}
        in_stations = False
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                values = line.split()
                if line.startswith("Station"):
                    in_stations = True
                    continue
                if in_stations:
                    if len(values) == 8:
                        _, name, X, Y, Z, a, e, r = values
                        R = cls._station_rotation(*(math.radians(float(v)) for v in (a, e, r)))
                        views[name] = (R.T, np.array([float(X), float(Y), float(Z)]))
                    continue
                if values and values[0] in cls.NAMES and values[0] not in parameters and len(values) >= 4:
                    parameters[values[0]] = float(values[3])
                elif len(values) == 3 and values[0] in ("H", "V"):
                    k = 0 if values[0] == "H" else 1
                    resolution[k], pixel_size[k] = int(values[1]), float(values[2])
        if None in resolution or None in pixel_size:
            raise ValueError(f"{path}: no sensor size (lines H and V)")
        return cls(parameters, pixel_size, resolution, views, frame)

    @staticmethod
    def _station_rotation(alpha, elev, roll):
        """World → camera from the stations' angles Alpha, Elev., Roll (photogrammetric convention)."""
        ca, sa, ce, se, cr, sr = (math.cos(alpha), math.sin(alpha), math.cos(elev), math.sin(elev), math.cos(roll),
                                  math.sin(roll))
        return np.array([[sa * se * sr + ca * cr, -ca * se * sr + sa * cr, ce * sr],
                         [sa * se * cr - ca * sr, -ca * se * cr - sa * sr, ce * cr],
                         [sa * ce, -ca * ce, -se]])

    def to_sensor(self, uv):
        """Pixels → sensor coordinates in mm (x right, y up), the origin at the sensor's centre W/2, H/2 (for an odd
        size half a pixel from the integer division the evaluation's camera.py used until 2026-10-30)."""
        W, H = self.resolution
        return np.stack([(uv[..., 0] + 0.5 - W / 2) * self.pixel_size[0],
                         (H / 2 - uv[..., 1] - 0.5) * self.pixel_size[1]], -1)

    def to_pixels(self, xy):
        W, H = self.resolution
        return np.stack([xy[..., 0] / self.pixel_size[0] - 0.5 + W / 2,
                         H / 2 - 0.5 - xy[..., 1] / self.pixel_size[1]], -1)

    def correct(self, xy):
        """Measured sensor coordinates → ideal ones: x − x_p plus the radial, decentring, affinity and shear terms."""
        p = self.parameters
        x, y = xy[..., 0] - p["XP"], xy[..., 1] - p["YP"]
        r2 = x * x + y * y
        radial = r2 * (p["K1"] + r2 * (p["K2"] + r2 * p["K3"]))
        return np.stack([x + x * radial + p["P1"] * (r2 + 2 * x * x) + 2 * p["P2"] * x * y + p["B1"] * x + p["B2"] * y,
                         y + y * radial + p["P2"] * (r2 + 2 * y * y) + 2 * p["P1"] * x * y], -1)

    def pixel_to_ray(self, uv, allow_extrapolation=False, return_status=False, return_sigma=False):
        """(origin = 0, unit direction) of pixels uv (..., 2) – closed form."""
        uv = np.asarray(uv, dtype=float)
        xy = self.to_sensor(uv)
        ideal = self.correct(xy)
        radius = np.hypot(xy[..., 0] - self.parameters["XP"], xy[..., 1] - self.parameters["YP"])
        status = np.where(np.all(np.isfinite(uv), axis=-1), Status.OK, Status.NO_CONVERGENCE)
        status = np.where((status == Status.OK) & (radius > self.valid_radius), Status.OUTSIDE_DOMAIN, status)
        status = np.where((status == Status.OK) & ~self.contains(uv), Status.OUTSIDE_HULL, status)
        d = np.concatenate([ideal, np.full(ideal.shape[:-1] + (1,), -self.parameters["C"])], -1)
        return self._with_sigma(self._finish_rays(d, status, allow_extrapolation, return_status), uv, return_sigma)

    def world_to_pixel(self, X, pose=None, allow_extrapolation=False, return_status=False,
                       return_covariance=False, world_covariance=None, image_noise=False):
        """Pixels (..., 2) of points X (..., 3) in the camera frame, or world points with pose = a view's
        (rotation camera → world, centre): the ideal image point, then the measured one by Newton (vectorized)."""
        Xc = self._camera_points(X, pose)
        shape = Xc.shape[:-1]
        Xc = Xc.reshape(-1, 3)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ideal = -self.parameters["C"] * Xc[:, :2] / z[:, None]
        start = ideal + [self.parameters["XP"], self.parameters["YP"]]
        measured, ok = _newton(self.correct, ideal, start)
        status = np.where(z < 0, Status.OK, Status.BEHIND)
        status = np.where((status == Status.OK) & ~ok, Status.NO_CONVERGENCE, status)
        radius = np.hypot(measured[:, 0] - self.parameters["XP"], measured[:, 1] - self.parameters["YP"])
        status = np.where((status == Status.OK) & (radius > self.valid_radius), Status.OUTSIDE_DOMAIN, status)
        uv = np.where((status == Status.OK)[:, None] | (status == Status.OUTSIDE_DOMAIN)[:, None],
                      self.to_pixels(measured), np.nan)
        result = self._finish_pixels(uv.reshape(shape + (2,)), status.reshape(shape), allow_extrapolation,
                                     return_status)
        return self._with_covariance(result, X, pose, return_covariance, world_covariance, image_noise)
