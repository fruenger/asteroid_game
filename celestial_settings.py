"""Shared sky / diurnal conventions (read by game_state, orbit_api, Ursina)."""

from __future__ import annotations

import math
import os

import numpy as np


def diurnal_sign() -> float:
    """
    Multiplier for the diurnal (24 h) rotation: hour angle in sun_direction, star field in shaders.

    Default -1 matches northern mid-latitudes (sun and stars move westward across the sky as
    game time increases). Set GAME_CELESTIAL_DIURNAL_SIGN to 1 (or +1, south, sh) to use the
    opposite sense.
    """
    raw = os.environ.get("GAME_CELESTIAL_DIURNAL_SIGN", "-1").strip().lower()
    if raw in ("1", "+1", "sh", "south", "southern"):
        return 1.0
    if raw in ("-1", "nh", "north", "northern"):
        return -1.0
    try:
        v = float(raw)
    except ValueError:
        return -1.0
    if v == 0.0:
        return -1.0
    return 1.0 if v > 0.0 else -1.0


def horizontal_celestial_offset_deg() -> float:
    """
    Extra yaw (degrees, right-hand about +Y, Y-up world) applied to astronomical directions
    (sun, scenario targets) so they match the observatory rig / horizon labels.

    Default **180** corrects the historic mismatch where culmination lay toward the same horizon
    as the telescope “home” boresight instead of the opposite (e.g. Sun toward south at noon on
    the northern mid-latitudes while the rig faces geographic north). Set **0** to restore the
    previous mapping.
    """
    raw = os.environ.get("GAME_CELESTIAL_HORIZ_OFFSET_DEG", "180").strip()
    try:
        return float(raw)
    except ValueError:
        return 180.0


def horizontal_celestial_offset_rad() -> float:
    return math.radians(horizontal_celestial_offset_deg())


def apply_horizontal_yaw_y_up(v: np.ndarray, deg: float | None = None) -> np.ndarray:
    """Rotate (x, z) about +Y; leaves y unchanged (matches GLES / Ursina world)."""
    if deg is None:
        deg = horizontal_celestial_offset_deg()
    if abs(deg) < 1e-9:
        return np.asarray(v, dtype=np.float64).reshape(3).copy()
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    a = np.asarray(v, dtype=np.float64).reshape(3)
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    return np.array([c * x + s * z, y, -s * x + c * z], dtype=np.float64)
