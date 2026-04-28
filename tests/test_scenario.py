"""Tests for scenario.generate_game_scenario (needs poliastro / full venv)."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np


class TestGameScenario(unittest.TestCase):
    def test_time_window_sane(self) -> None:
        from scenario import generate_game_scenario

        s = generate_game_scenario(verbose=False)
        self.assertEqual(s.object_dir_cartesian.shape, (3,))
        self.assertGreaterEqual(float(s.min_time), 0.0)
        self.assertLess(float(s.min_time), 1.0)
        self.assertGreater(float(s.max_time), float(s.min_time))
        self.assertLess(float(s.max_time), 1.0)
        self.assertIsNotNone(s.orbit)

    def test_max_target_altitude_cap(self) -> None:
        from scenario import generate_game_scenario

        with mock.patch.dict("os.environ", {"GAME_MAX_TARGET_ALTITUDE_DEG": "30"}, clear=False):
            s = generate_game_scenario(verbose=False)
        y = float(s.object_dir_cartesian[1])
        # Sampled AltAz is capped at 30°; snap-to-rig may nudge the unit vector by a few degrees.
        self.assertLessEqual(y, float(np.sin(np.radians(36.0))))


if __name__ == "__main__":
    unittest.main()
