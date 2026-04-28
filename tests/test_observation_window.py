"""Tests for observation_window + orbit_catalog (venv / poliastro)."""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from observation_window import (
    compute_night_visibility_window,
    helio_to_topocentric_manual_dir,
    utc_time_to_game_time,
)
from orbit_catalog import N_CATALOG, catalog_index_for_seed, classical_orbit_from_seed


class TestObservationWindow(unittest.TestCase):
    def test_visibility_window_ordered(self) -> None:
        today = dt.date(2024, 6, 15)
        loc = EarthLocation(lon=13.0 * u.deg, lat=52.0 * u.deg, height=0.0 * u.m)
        orb, _ = classical_orbit_from_seed(4242, today)
        t_open, t_close = compute_night_visibility_window(
            orb, today, loc, h_min_deg=15.0, step_minutes=15.0
        )
        self.assertLess(float(t_open.mjd), float(t_close.mjd))
        g0 = utc_time_to_game_time(t_open, loc)
        g1 = utc_time_to_game_time(t_close, loc)
        self.assertGreater(g1, g0)

    def test_helio_manual_unit_length(self) -> None:
        today = dt.date(2024, 6, 15)
        loc = EarthLocation(lon=13.0 * u.deg, lat=52.0 * u.deg, height=0.0 * u.m)
        orb, _ = classical_orbit_from_seed(7, today)
        t_open, t_close = compute_night_visibility_window(orb, today, loc, 10.0, step_minutes=10.0)
        t_mid = t_open + 0.5 * (t_close - t_open)
        v = helio_to_topocentric_manual_dir(orb, t_mid, loc)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_catalog_index_bounded_and_deterministic(self) -> None:
        self.assertEqual(catalog_index_for_seed(12345), catalog_index_for_seed(12345))
        self.assertGreaterEqual(catalog_index_for_seed(999), 0)
        self.assertLess(catalog_index_for_seed(999), N_CATALOG)


class TestGameTimeMapping(unittest.TestCase):
    def test_utc_roundtrip_monotone(self) -> None:
        loc = EarthLocation(lon=13.0 * u.deg, lat=52.0 * u.deg, height=0.0 * u.m)
        t0 = Time("2024-06-15T18:00:00", scale="utc")
        t1 = t0 + 2.5 * u.hour
        g0 = utc_time_to_game_time(t0, loc)
        g1 = utc_time_to_game_time(t1, loc)
        self.assertLess(g0, g1)


if __name__ == "__main__":
    unittest.main()
