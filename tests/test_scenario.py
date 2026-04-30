"""Tests for scenario.generate_game_scenario (needs poliastro / full venv)."""

from __future__ import annotations

import os
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

    def test_ephemeris_fields_present(self) -> None:
        from scenario import generate_game_scenario

        with mock.patch.dict(
            os.environ,
            {
                "GAME_SESSION_SEED": "42",
                "GAME_REFERENCE_DATE": "2024-06-15",
                # Seed/orbit combos can yield a short grid window; keep the requested seed (no retry).
                "GAME_MIN_VISIBILITY_HOURS": "0.01",
                "GAME_ALLOW_SYNTHETIC_WINDOW": "1",
            },
            clear=False,
        ):
            s = generate_game_scenario(verbose=False)
        self.assertEqual(s.reference_date.isoformat(), "2024-06-15")
        self.assertEqual(s.session_seed, 42)
        self.assertLess(s.visibility_t_open_mjd, s.visibility_t_close_mjd)
        od = np.asarray(s.object_dir_cartesian, dtype=np.float64).reshape(3)
        self.assertAlmostEqual(float(np.linalg.norm(od)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
