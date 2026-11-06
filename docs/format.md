# mapping.json – format "calibconsumer-ray-mapping", version 1 (normative)

A UTF-8 JSON object. Readers refuse another `formatVersion`. Numbers are IEEE doubles written in shortest round-trip
form; no NaN or infinity anywhere.

## Conventions

- **Pixels** (u, v): (0, 0) is the centre of the top-left pixel; u grows to the right, v downwards.
- **Camera frame**: x right, y up, the camera looks along −z. Rays are given in this frame.
- **Lengths**: the units of the calibration's world points (`conventions.lengthUnit`).

## Fields

| Field | Content |
|---|---|
| `format` | `"calibconsumer-ray-mapping"` |
| `formatVersion` | `1` |
| `producer` | free provenance: `cli` (the calibration program's version), `mapper` (the writer), `dateTime`, `result` |
| `camera` | `resolution` [W, H] (required), `name`, `pixelPitchMM` [sx, sy] (optional) |
| `conventions` | the three conventions above as text (informative) |
| `source` | the calibrated model's type (informative) |
| `central` | `true`: every ray passes through the origin, `origin` is `null`; `false`: `origin` is required |
| `domain.rectangle` | [[u0, v0], [u1, v1]]: the splines' rectangle |
| `domain.hull` | [[u, v], …] ≥ 3 vertices of a convex polygon (either orientation, no repeated closing vertex): the calibrated image area |
| `spline` | `type` `"uniform-cubic-bspline"`, `intervals` [n, m] (≥ 1 each) |
| `direction` | `parametrization` `"azimuthal-equidistant"`, `axisRotation` (3 × 3 rotation R, rows), `controls` |
| `origin` | `null`, or `{"parametrization": "foot-point", "controls"}`, or `{"parametrization": "two-surface", "distances": [Z1, Z2], "controlsNear", "controlsFar"}` |
| `refit` | the measured re-fit errors (informative; `passed` false makes readers warn) |
| `inverse` | `startRays` [G, G′] (default [32, 24]), `injective` (false: the direction map folds over somewhere in the hull – readers warn and flag world points whose pixel lies in a fold), `foldedShare` (the share of the hull's area where the Jacobian determinant of the direction spline's (a, b) has the minority sign), `minUniqueDistance` (non-central: closer points may lie on more than one ray; null if unknown) |
| `uncertainty` | optional (readers that do not know it ignore it): `unit` `"px"`; `ray` – the calibration's σ of a ray's direction per axis of the plane normal to it, divided by the angle of one pixel there: a grid `{"stepPx", "width", "height", "values" [height × width, row-major, null = unknown]}` (cell (x, y) holds the value at pixel ((x + ½)·stepPx − ½, (y + ½)·stepPx − ½); readers interpolate bilinearly between cell centres, leave null cells out and hold the border cells outside), a number (the same everywhere) or null (unknown – e.g. non-central cameras); `refit` `{"rmsPx", "map"}` – the mapping's own error against the calibrated model (px; `map` a grid as above, its null cells take `rmsPx`); `imageSigmaPx` – the image noise σ of one measured point (px) or null; `source` (informative). The model σ of a pixel is √(ray² + refit²) px |
| `views` | optional: per calibrated image `{"rotation": 3 × 3 camera-to-world, "centre": [x, y, z]}`; X_camera = Rᵀ(X_world − centre) |

## The spline

n × m uniform intervals on the rectangle; `controls` holds (m + 3)·(n + 3)·C numbers in the order of a C-contiguous
array of shape (m + 3, n + 3, C): v index slowest, then u index, then the component. For a pixel (u, v):
s = (u − u0)/(u1 − u0)·n, i = clamp(⌊s⌋, 0, n − 1), σ = s − i; likewise t, j, τ along v with m. The value of component c
is Σ_{a=0..3} Σ_{b=0..3} B_a(σ) B_b(τ) controls[j + b, i + a, c] with

B(x) = 1/6 · [(1 − x)³, 3x³ − 6x² + 4, −3x³ + 3x² + 3x + 1, x³].

Outside the rectangle σ or τ leaves [0, 1] and the border interval's polynomial continues.

## The ray of a pixel

- **Direction**: the direction spline (C = 2) gives (a, b); with r = |(a, b)|, d = R · (a·sin r / r, b·sin r / r, −cos r)
  (sin r / r = 1 at r = 0).
- **Origin**: central – (0, 0, 0). Foot point – the origin spline (C = 3) gives the point of the ray closest to the
  camera frame's origin. Two-surface – the splines `controlsNear`, `controlsFar` give the ray's points P1, P2 at the
  working distances Z1, Z2; the ray is the line through them, d = (P2 − P1)/|P2 − P1|, and the direction spline is not
  used for the ray.

## The calibrated area

A reader maps only pixels inside `domain.hull` (on its border included) unless the user explicitly allows
extrapolation, and never pixels beyond the rectangle enlarged by 10 % of its size on every side.
