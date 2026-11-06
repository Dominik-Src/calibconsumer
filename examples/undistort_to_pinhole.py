"""An image of a central camera resampled to a virtual pinhole (needs OpenCV):
python3 examples/undistort_to_pinhole.py mapping.json photo.jpg out.png [focal_px]"""
import sys

import cv2

import calibconsumer
from calibconsumer.remap import pinhole_maps

camera = calibconsumer.load(sys.argv[1])
image = cv2.imread(sys.argv[2])
focal = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5 * camera.resolution[0]
map_x, map_y = pinhole_maps(camera, focal, camera.resolution)
cv2.imwrite(sys.argv[3], cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR))
