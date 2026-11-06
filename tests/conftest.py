# Copyright 2026 Dominik Rueß
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

NAMES = ("fisheye150", "fisheye200", "tele800", "axial", "asymmetric")


@pytest.fixture(params=NAMES)
def case(request):
    import calibconsumer
    import truths
    return calibconsumer.load(os.path.join(HERE, "data", f"{request.param}.json")), truths.truths()[request.param]
