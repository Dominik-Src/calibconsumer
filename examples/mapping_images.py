"""A camera's mapping at a glance, no photograph needed – two images (PPM, no dependency):
python3 examples/mapping_images.py mapping.json out-directory [equirectangular texture.ppm]

sensor_on_sphere.ppm: the sensor's checkerboard on the sphere (where each part of the image looks);
sphere_on_sensor.ppm: the sphere's latitude-longitude grid on the sensor – or, with a texture, that texture. A free
earth map: NASA's Blue Marble (public domain, NASA asks for a credit: "NASA Earth Observatory") or Natural Earth's
rasters (public domain), both equirectangular – download one yourself and convert it to PPM (the package ships none)."""
import os
import sys

import numpy as np

import calibconsumer
from calibconsumer import visual


def write_ppm(path, image):
    height, width = image.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6 {width} {height} 255\n".encode() + np.ascontiguousarray(image[..., :3], dtype=np.uint8).tobytes())


def read_ppm(path):
    data = open(path, "rb").read()
    fields = data.split(maxsplit=4)
    width, height = int(fields[1]), int(fields[2])
    return np.frombuffer(fields[4], dtype=np.uint8)[: width * height * 3].reshape(height, width, 3)


camera = calibconsumer.load(sys.argv[1])
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
distance = None if getattr(camera, "central", True) else 2000.0  # a non-central camera: its rays at 2 m
write_ppm(os.path.join(out, "sensor_on_sphere.ppm"), visual.sensor_on_sphere(camera, (1024, 512), distance=distance))
texture = read_ppm(sys.argv[3]) if len(sys.argv) > 3 else None
write_ppm(os.path.join(out, "sphere_on_sensor.ppm"),
          visual.sphere_on_sensor(camera, width=1024, texture=texture, pole="up" if texture is not None else "axis"))
