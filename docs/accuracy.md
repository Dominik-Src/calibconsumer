# Accuracy: what `refit` states

A calibration produces a camera model of some type – a classic lens model, a generic central model, a non-central
one. The calibration service re-fits every model to the one representation of `format.md` and measures how closely
the re-fit reproduces the calibrated model's rays:

- `refit.grid`: on a dense grid of pixels inside the calibrated area that the fit did not use;
- `refit.calibrationPoints`: at the calibration's own measured pixels;
- `pixelMax`, `pixelRms`: the angle between the calibrated and the re-fitted ray divided by the angle between the rays
  of neighbouring pixels – a pixel-equivalent error;
- `angleMaxRad`: the angle itself;
- `objectSpace`: the distance between the two rays at given distances from the camera, in world units;
- `bounds`, `passed`: the service's acceptance bounds and whether they were met.

These numbers describe the mapping, not the calibration: the calibration's own uncertainty is in its report. With
`passed` false the reader warns; the numbers tell how far the mapping is off.

## Uncertainties: what `uncertainty` states

The optional `uncertainty` block combines the calibration's uncertainty with the mapping's:

- `ray`: the calibration's σ of a ray's direction (from its covariance, the camera frame's free rotation removed), per
  axis of the plane normal to the ray, as a fraction of one pixel's angle there – over the sensor, so it grows where
  the calibration extrapolates; null where unknown (non-central cameras, for now);
- `refit`: the mapping's own error against the calibrated model over the sensor (px);
- `imageSigmaPx`: the image noise of one measured point.

`pixel_to_ray(uv, return_sigma=True)` gives each ray's σ as an angle; `world_to_pixel(X, return_covariance=True,
world_covariance=Σ)` gives each pixel's covariance from the model and from the world points' own covariances, plus the
image noise with `image_noise=True` (the spread to expect of a measured point around its prediction). These are first-
order (linearized) values; the calibration's correlations between neighbouring rays are not part of them.
