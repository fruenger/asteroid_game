"""b1_kinematics: optical axis must be able to align with arbitrary sky directions (grid search)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

import b1_kinematics as b1k
from game_state import align_dot_threshold


class TestB1Kinematics(unittest.TestCase):
    @mock.patch.dict(os.environ, {"B1_RA_OFFSET_DEG": "0"}, clear=False)
    def test_optical_axis_can_align_scenario_object(self) -> None:
        """Same object direction as generate_game_scenario — must be reachable by RA/DEC (coarse grid)."""
        from scenario import generate_game_scenario

        sc = generate_game_scenario(verbose=False)
        v = np.asarray(sc.object_dir_cartesian, dtype=np.float64).reshape(3)
        v = v / float(np.linalg.norm(v))
        best = -1.0
        for ra in np.linspace(-180.0, 180.0, 721):
            for dec in np.linspace(-89.0, 89.0, 200):
                u = b1k.optical_axis_world_unit(ra, dec)
                best = max(best, float(np.dot(u, v)))
        self.assertGreater(
            best,
            align_dot_threshold(),
            msg="b1_kinematics must match Ursina rig enough to point at the procedural target",
        )

    @mock.patch.dict(os.environ, {"B1_RA_OFFSET_DEG": "0"}, clear=False)
    def test_dome_hint_returns_reasonable_floats(self) -> None:
        d = b1k.dome_ray_exit_distance_b1(0.0, 10.0, 0.0)
        self.assertIsInstance(d, float)
        self.assertGreater(d, 0.0)

    def test_ra_offset_changes_optical_axis(self) -> None:
        with mock.patch.dict(os.environ, {"B1_RA_OFFSET_DEG": "0"}, clear=False):
            u0 = b1k.optical_axis_world_unit(12.0, 5.0)
        with mock.patch.dict(os.environ, {"B1_RA_OFFSET_DEG": "180"}, clear=False):
            u180 = b1k.optical_axis_world_unit(12.0, 5.0)
        d = float(np.dot(u0, u180))
        self.assertLess(abs(d), 0.999, msg="180° RA offset should change axis direction")

    @mock.patch.dict(os.environ, {"B1_RA_OFFSET_DEG": "0"}, clear=False)
    def test_ra_offset_deg_reads_env(self) -> None:
        self.assertEqual(b1k.ra_offset_deg(), 0.0)

    @mock.patch.dict(os.environ, {"B1_RIG_LAT_TILT": "-30"}, clear=False)
    def test_base_tilt_reads_env(self) -> None:
        self.assertEqual(b1k.base_tilt_deg(), -30.0)

    @mock.patch.dict(os.environ, {"B1_RIG_LAT_TILT": "200"}, clear=False)
    def test_base_tilt_invalid_falls_back(self) -> None:
        self.assertEqual(b1k.base_tilt_deg(), 52.0 - 90.0)


if __name__ == "__main__":
    unittest.main()
