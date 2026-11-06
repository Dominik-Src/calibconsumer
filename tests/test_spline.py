# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
"""M1: the spline basis and evaluation; M2: the AE coordinates."""

import numpy as np

from calibconsumer import direction as ae
from calibconsumer.spline import Spline, basis, basis_derivative


def test_m1_partition_of_unity_and_derivative():
    t = np.linspace(-0.5, 1.5, 101)
    assert np.allclose(basis(t).sum(-1), 1.0)
    h = 1e-6
    assert np.allclose(basis_derivative(t), (basis(t + h) - basis(t - h)) / (2 * h), atol=1e-8)


def test_m1_continuity_and_derivatives_across_knots():
    rng = np.random.default_rng(0)
    s = Spline([[0, 0], [40, 30]], (4, 3), rng.normal(size=(6, 7, 2)))
    for u in (10.0, 20.0, 30.0):  # knot lines along u
        uv = np.array([[u - 1e-9, 7.3], [u + 1e-9, 7.3]])
        value, du, dv = s.evaluate(uv, derivatives=True)
        assert np.allclose(value[0], value[1], atol=1e-7) and np.allclose(du[0], du[1], atol=1e-6)
        h = 1e-4  # second derivative continuous: the derivative's central difference agrees on both sides
        left = (s.evaluate([[u - 2 * h, 7.3]], True)[1] - s.evaluate([[u - 3 * h, 7.3]], True)[1]) / h
        right = (s.evaluate([[u + 3 * h, 7.3]], True)[1] - s.evaluate([[u + 2 * h, 7.3]], True)[1]) / h
        assert np.allclose(left, right, atol=1e-2)
    uv = rng.uniform([0, 0], [40, 30], size=(50, 2))
    value, du, dv = s.evaluate(uv, derivatives=True)
    h = 1e-5
    assert np.allclose(du, (s.evaluate(uv + [h, 0]) - s.evaluate(uv - [h, 0])) / (2 * h), atol=1e-6)
    assert np.allclose(dv, (s.evaluate(uv + [0, h]) - s.evaluate(uv - [0, h])) / (2 * h), atol=1e-6)


def test_m1_extrapolation_continues_the_border_polynomial():
    rng = np.random.default_rng(1)
    s = Spline([[0, 0], [10, 10]], (2, 2), rng.normal(size=(5, 5, 1)))
    inside = s.evaluate(np.array([[9.0, 5.0], [9.5, 5.0], [9.9, 5.0]]))[:, 0]
    outside = s.evaluate(np.array([[11.0, 5.0]]))[0, 0]
    coefficients = np.polyfit([9.0, 9.5, 9.9, 9.2], np.r_[inside, s.evaluate([[9.2, 5.0]])[0, 0]], 3)
    assert abs(np.polyval(coefficients, 11.0) - outside) < 1e-9
    assert np.isnan(s.evaluate([[np.nan, 1.0]])).all()


def test_m2_ae_round_trip_and_the_axis():
    rng = np.random.default_rng(2)
    R = ae.axis_rotation(np.array([0.3, -0.2, -1.0]))
    assert np.allclose(R @ [0, 0, -1], np.array([0.3, -0.2, -1.0]) / np.linalg.norm([0.3, -0.2, -1.0]))
    theta = np.r_[0.0, 1e-12, 1e-9, 1e-6, rng.uniform(0, np.radians(175), 500)]
    phi = rng.uniform(-np.pi, np.pi, len(theta))
    local = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)], -1)
    d = local @ R.T
    ab = ae.from_direction(d, R)
    assert np.allclose(np.hypot(ab[:, 0], ab[:, 1]), theta, atol=1e-13, rtol=1e-13)
    assert np.allclose(ae.to_direction(ab, R), d, atol=1e-13)
    _, J = ae.to_direction(ab, R, jacobian=True)
    h = 1e-7
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        numeric = (ae.to_direction(ab + e, R) - ae.to_direction(ab - e, R)) / (2 * h)
        assert np.allclose(J[..., k], numeric, atol=1e-7)
