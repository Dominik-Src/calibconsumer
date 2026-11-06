# Conventions

| | calibconsumer (the file) | OpenCV (`camera.with_frame("opencv")`) |
|---|---|---|
| pixel (0, 0) | centre of the top-left pixel | centre of the top-left pixel |
| u, v | right, down | right, down |
| camera x, y, z | right, up, backwards (looks along −z) | right, down, forwards (looks along +z) |

The conversion is (x, y, z)_OpenCV = (x, −y, −z)_calibconsumer. A view's pose in `views` is camera-to-world in the file's
frame: X_camera = Rᵀ(X_world − centre); `world_to_pixel(X, pose=camera.views[name])` applies it.
