"""Two images of a rig rectified (needs OpenCV): the pair's relative pose from relativeOrientations.json (a calibration
of several cameras; the pair "i-j" by the cameras' indexes in it), both cameras' mapping.json (or classic files):
python3 examples/rectify_stereo_pair.py relativeOrientations.json 0 1 camera0/ camera1/ image0.jpg image1.jpg out.png

Rows of the two rectified images show the same scene points (columns for a vertical rig); out.png puts them side by
side. Generic and classic cameras alike; for two OpenCV cameras the result also holds cv2.stereoRectify's own numbers
(stereo/stereo_<i>_<j>.json)."""
import json
import sys

import cv2
import numpy as np

import calibconsumer
from calibconsumer import stereo

orientations = json.load(open(sys.argv[1]))
i, j = int(sys.argv[2]), int(sys.argv[3])
pair = next(p for p in orientations["pairs"] if p["cameras"] == [i, j] and p.get("rig"))
camera1, camera2 = calibconsumer.load(sys.argv[4]), calibconsumer.load(sys.argv[5])
result = stereo.rectify(camera1, camera2, pair["rotation"], pair["translation"], size=(1280, 960))
left = cv2.remap(cv2.imread(sys.argv[6]), *result["maps1"], cv2.INTER_LINEAR)
right = cv2.remap(cv2.imread(sys.argv[7]), *result["maps2"], cv2.INTER_LINEAR)
cv2.imwrite(sys.argv[8], np.vstack([left, right]) if result["vertical"] else np.hstack([left, right]))
print(f"baseline {result['baseline']:.2f} (world units), focal {result['focal']:.1f} px")
