# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""Rays and pixels of a calibrated camera from a mapping.json file (format "calibconsumer-ray-mapping")."""

import json
import os

from .camera import Camera, Status
from .classic import BrownCamera, OpenCVCamera
from .fileformat import FORMAT, FORMAT_VERSION, FormatError
from .uncertainty import Uncertainty, attach as _attach_uncertainty, ellipse

__version__ = "0.2.1"
__all__ = ["Camera", "OpenCVCamera", "BrownCamera", "Status", "load", "FormatError", "FORMAT", "FORMAT_VERSION",
           "Uncertainty", "ellipse", "__version__"]

CLASSIC_FILES = ("results_OpenCV.json", "results_BrownCamera.txt")


def load(path, quiet: bool = False):
    """The camera of a result file or folder: mapping.json (format "calibconsumer-ray-mapping") → Camera; results_OpenCV.json
    or one's own OpenCV parameters (JSON, or OpenCV's YAML/XML with cv2) → OpenCVCamera; results_BrownCamera.txt or a
    Brown JSON ({"model": "Brown", …}, docs/evaluate.md) → BrownCamera. A camera's result folder: its mapping.json if there is one,
    else its classic parameter file. A camera without an uncertainty block of its own takes the calibration's from an
    error-statistics.json beside the file (camera.uncertainty; see calibconsumer.uncertainty)."""
    path = os.fspath(path)
    if os.path.isdir(path):
        for name in ("mapping.json",) + CLASSIC_FILES:
            if os.path.isfile(os.path.join(path, name)):
                return load(os.path.join(path, name), quiet)
        raise FormatError(f"{path}: no mapping.json, results_OpenCV.json or results_BrownCamera.txt")
    return _attach_uncertainty(_load(path, quiet), os.path.dirname(os.path.abspath(path)))


def _load(path, quiet):
    if path.endswith(".txt"):
        return BrownCamera.from_file(path)
    if path.lower().endswith((".yml", ".yaml", ".xml")):
        return OpenCVCamera.from_opencv_storage(path)
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise FormatError(f"{path}: not JSON ({e})") from None
    if isinstance(data, dict) and str(data.get("model", "")).lower() == "brown":
        return BrownCamera.from_dict(data)
    if isinstance(data, dict) and any(k in data for k in ("cameraMatrix", "camera_matrix", "K")):
        return OpenCVCamera.from_dict(data)
    return Camera(data, quiet=quiet)
