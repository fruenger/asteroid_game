"""
Procedural game scenario: mock asteroid direction, observation triple, poliastro orbit, time window.

Ursina-free — safe to import from tests or tooling without Panda3D.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import astropy.units as u
import astropy.time as astropy_time
from astropy.coordinates import AltAz, EarthLocation, ICRS

import b1_kinematics as b1k

from celestial_settings import apply_horizontal_yaw_y_up
from orbit_api import day2range, preliminary_orbit


def _snap_object_dir_to_telescope_rig(manual_unit: np.ndarray) -> np.ndarray:
    """Game target must lie on directions reachable by ``b1_kinematics`` / Ursina rig (coarse grid)."""
    v = np.asarray(manual_unit, dtype=np.float64).reshape(3)
    vn = float(np.linalg.norm(v))
    if vn < 1e-12:
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        v = v / vn
    best_dot = -2.0
    best_u = v.copy()
    for ra in np.arange(-180.0, 180.0 + 1e-6, 5.0):
        for dec in np.arange(-85.0, 85.0 + 1e-6, 5.0):
            u = np.asarray(b1k.optical_axis_world_unit(float(ra), float(dec)), dtype=np.float64).reshape(3)
            un = float(np.linalg.norm(u))
            if un < 1e-12:
                continue
            u = u / un
            d = float(np.dot(v, u))
            if d > best_dot:
                best_dot = d
                best_u = u
    return best_u


@dataclass(frozen=True)
class GameScenario:
    """Values needed by game.py and AsteroidGameState (orbit kept for catalog / step 6)."""

    object_dir_cartesian: np.ndarray
    min_time: float
    max_time: float
    orbit: Any
    obs_time: datetime.time
    obs_location: EarthLocation
    today: datetime.date


def _max_target_altitude_deg() -> float:
    raw = os.environ.get("GAME_MAX_TARGET_ALTITUDE_DEG", "75").strip()
    try:
        v = float(raw)
    except ValueError:
        return 75.0
    return float(np.clip(v, 5.0, 89.0))


def _min_target_altitude_deg() -> float:
    raw = os.environ.get("GAME_MIN_TARGET_ALTITUDE_DEG", "10").strip()
    try:
        v = float(raw)
    except ValueError:
        return 10.0
    return float(np.clip(v, 0.0, 88.0))


def generate_game_scenario(*, verbose: bool = True) -> GameScenario:
    today = datetime.date.today()
    max_alt_deg = _max_target_altitude_deg()
    min_alt_deg = _min_target_altitude_deg()
    if min_alt_deg >= max_alt_deg - 1.0:
        min_alt_deg = max(max_alt_deg - 15.0, 0.0)
    alt_deg = float(np.random.uniform(min_alt_deg, max_alt_deg))
    alt_rad = float(np.radians(alt_deg))
    obs_time = datetime.time(
        int(np.random.uniform(21, 4 + 24) % 24),
        int(np.random.uniform(0, 60)),
        second=0,
        tzinfo=datetime.timezone(datetime.timedelta(hours=2)),
    )
    obs_location = EarthLocation(lon=13.0, lat=52.0, height=0.0)
    coord = AltAz(
        alt=alt_rad * u.rad,
        az=np.random.uniform(0.0, 2.0 * np.pi) * u.rad,
        obstime=astropy_time.Time(
            datetime.datetime(
                today.year,
                today.month,
                today.day,
                obs_time.hour,
                obs_time.minute,
                obs_time.second,
            ),
        ),
        location=obs_location,
    )
    manual_dir = np.array(
        [
            float(np.cos(coord.alt) * np.cos(coord.az)),
            float(np.sin(coord.alt)),
            float(np.cos(coord.alt) * np.sin(coord.az)),
        ],
        dtype=np.float64,
    )
    manual_dir = apply_horizontal_yaw_y_up(manual_dir)
    object_dir_cartesian = _snap_object_dir_to_telescope_rig(manual_dir)
    coord_icrs = coord.transform_to(ICRS())
    observations = [
        {
            "ra": coord_icrs.ra.to_value(u.deg),
            "dec": coord_icrs.dec.to_value(u.deg),
            "time": "%i-%i-%iT%i:%i:00"
            % (today.year, today.month, today.day, obs_time.hour, obs_time.minute),
            "location": obs_location,
        },
        {
            "ra": coord_icrs.ra.to_value(u.deg) + 0.2,
            "dec": coord_icrs.dec.to_value(u.deg) + 0.02,
            "time": "%i-%i-%iT%i:%i:00"
            % (today.year, today.month, today.day + 1, obs_time.hour, obs_time.minute),
            "location": obs_location,
        },
        {
            "ra": coord_icrs.ra.to_value(u.deg) + 0.4,
            "dec": coord_icrs.dec.to_value(u.deg) + 0.04,
            "time": "%i-%i-%iT%i:%i:00"
            % (today.year, today.month, today.day + 2, obs_time.hour, obs_time.minute),
            "location": obs_location,
        },
    ]
    orbit = preliminary_orbit(observations)
    if verbose:
        print(orbit.a, orbit.r_a, orbit.r_p, orbit.ecc, orbit.inc, orbit.period)

    min_time = day2range(obs_time.hour - 0.5 + obs_time.minute / 60.0)
    max_time = day2range(obs_time.hour + 0.5 + obs_time.minute / 60.0)

    return GameScenario(
        object_dir_cartesian=object_dir_cartesian,
        min_time=min_time,
        max_time=max_time,
        orbit=orbit,
        obs_time=obs_time,
        obs_location=obs_location,
        today=today,
    )
