"""Tests for orbit_api (no Ursina). Run from asteroid_game/: python -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import orbit_api  # noqa: E402


class TestSunDirection(unittest.TestCase):
    def test_unit_length(self):
        v = orbit_api.sun_direction(0.5, 52.0, 15.0)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_diurnal_sign_flips_azimuth(self):
        t, lat, dec = 0.62, 50.0, 10.0
        with patch.dict(os.environ, {"GAME_CELESTIAL_DIURNAL_SIGN": "1"}, clear=False):
            v_pos = orbit_api.sun_direction(t, lat, dec).copy()
        with patch.dict(os.environ, {"GAME_CELESTIAL_DIURNAL_SIGN": "-1"}, clear=False):
            v_neg = orbit_api.sun_direction(t, lat, dec).copy()
        self.assertAlmostEqual(float(v_pos[1]), float(v_neg[1]), places=5)
        self.assertAlmostEqual(float(v_pos[0]), float(v_neg[0]), places=5)
        self.assertNotAlmostEqual(float(v_pos[2]), float(v_neg[2]), places=3)

    def test_sun_direction_json(self):
        s = orbit_api.sun_direction_json(
            json.dumps({"t": 0.35, "latitude_deg": 50.0, "sun_declination_deg": 10.0})
        )
        d = json.loads(s)
        self.assertTrue(d["ok"])
        n = (d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2) ** 0.5
        self.assertAlmostEqual(n, 1.0, places=5)


class TestDomeIntersect(unittest.TestCase):
    def test_ray_up_from_origin(self):
        p = orbit_api.get_dome_intersect(1.0, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(float(np.linalg.norm(p)), 1.0, places=5)


class TestOrbitJson(unittest.TestCase):
    def test_seed_reproducible(self):
        payload = {
            "seed": 42,
            "observations": [
                {
                    "ra": 180.0,
                    "dec": 45.0,
                    "time": "2024-06-01T12:00:00",
                    "location": {"lat_deg": 51.5, "lon_deg": 0.0, "height_m": 0},
                },
                {
                    "ra": 180.2,
                    "dec": 45.02,
                    "time": "2024-06-02T12:00:00",
                    "location": {"lat_deg": 51.5, "lon_deg": 0.0, "height_m": 0},
                },
                {
                    "ra": 180.4,
                    "dec": 45.04,
                    "time": "2024-06-03T12:00:00",
                    "location": {"lat_deg": 51.5, "lon_deg": 0.0, "height_m": 0},
                },
            ],
        }
        a = json.loads(orbit_api.compute_orbit_json(json.dumps(payload)))
        b = json.loads(orbit_api.compute_orbit_json(json.dumps(payload)))
        self.assertTrue(a.get("ok"))
        self.assertEqual(a["ecc"], b["ecc"])
        self.assertEqual(a["a_km"], b["a_km"])


class TestFrameVisual(unittest.TestCase):
    def test_tuple_length(self):
        t = orbit_api.frame_visual_for_c(0.35)
        self.assertEqual(len(t), 6)


class TestTimeHelpers(unittest.TestCase):
    def test_day2range_matches_game_formula(self):
        self.assertAlmostEqual(orbit_api.day2range(0.0), 0.5, places=5)
        self.assertAlmostEqual(orbit_api.day2range(12.0), 0.0, places=5)

    def test_dispatch_day2range(self):
        s = orbit_api.dispatch_compute_json('{"op":"day2range","hours":0.0}')
        d = json.loads(s)
        self.assertTrue(d["ok"])
        self.assertAlmostEqual(d["t"], 0.5, places=5)

    def test_dispatch_vec_mag(self):
        s = orbit_api.dispatch_compute_json('{"op":"vec_mag","v":[3,4]}')
        d = json.loads(s)
        self.assertTrue(d["ok"])
        self.assertAlmostEqual(d["mag"], 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
