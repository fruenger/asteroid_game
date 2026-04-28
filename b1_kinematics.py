"""
Telescope optical axis and pivot position in world space — matches the Ursina rig in game_scene.py
(base pivot Z = 52°−90°, RA about Y, DEC about X; optical marker at (2.2,0,0) in dec frame).

Used by the B1 host to fill game_state.FrameHints without Panda3D.

``B1_RA_OFFSET_DEG`` adds to game ``ra_deg`` before ``_rot_y`` (default **0** = Ursina ``game_scene``).
``b1_game_bridge.session_init`` sets **180** if unset so embedded B1 matches GLES ``b1_host``.
"""

from __future__ import annotations

import os

import numpy as np

# Default 0 matches Ursina ``game_scene`` (``ra_pivot.rotation_y = ra_deg`` with no extra twist).
# ``b1_game_bridge.session_init`` sets ``B1_RA_OFFSET_DEG`` to 180 by default so embedded B1 matches GLES.
_RA_OFFSET_DEFAULT_DEG = 0.0


def ra_offset_deg() -> float:
    """Degrees added to stored RA for rig math; reads ``B1_RA_OFFSET_DEG`` (default 0; B1 host sets 180)."""
    raw = os.environ.get("B1_RA_OFFSET_DEG", "").strip()
    if not raw:
        return _RA_OFFSET_DEFAULT_DEG
    try:
        return float(raw)
    except ValueError:
        return _RA_OFFSET_DEFAULT_DEG


def effective_ra_deg(ra_deg: float) -> float:
    """RA used inside ``_rot_y`` — game state angle plus host mount offset."""
    return float(ra_deg) + ra_offset_deg()

# game_scene.py: base_pivot rotation_z = 52 - 90 (overridable via ``B1_RIG_LAT_TILT``, same as b1_host)
_BASE_ROT_Z_DEG = 52.0 - 90.0


def base_tilt_deg() -> float:
    """Base mount tilt in degrees; reads ``B1_RIG_LAT_TILT`` so Python matches ``B1ObservatoryRig``."""
    raw = os.environ.get("B1_RIG_LAT_TILT", "").strip()
    if not raw:
        return float(_BASE_ROT_Z_DEG)
    try:
        v = float(raw)
    except ValueError:
        return float(_BASE_ROT_Z_DEG)
    if not (-90.0 < v < 90.0):
        return float(_BASE_ROT_Z_DEG)
    return float(v)
# dec_pivot position in ra_pivot space
_DEC_ORIGIN_IN_RA = np.array([1.8, 1.15, 0.0], dtype=np.float64)
# telecope_optical_axis position in dec_pivot space
_OPTICAL_IN_DEC = np.array([2.2, 0.0, 0.0], dtype=np.float64)
# base_pivot translation (parent = telescope_base at origin)
_BASE_TRANSLATION = np.array([-1.0, 3.0, 0.0], dtype=np.float64)


def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    """Rotation about +Y for column vectors; matches ``mat4_rotate_y`` in ``b1_observatory.cpp`` (not the transpose)."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def optical_axis_world_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Unit vector of telescope optical axis (Ursina default entity up = +Y in local space)."""
    ra_e = effective_ra_deg(ra_deg)
    r = _rot_z(base_tilt_deg()) @ _rot_y(ra_e) @ _rot_x(float(dec_deg))
    u = r @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return u / n


def telescope_optical_world_position(ra_deg: float, dec_deg: float) -> np.ndarray:
    """World position of the optical-axis origin (ray start), same rig as game_scene."""
    ra_e = effective_ra_deg(ra_deg)
    r_base = _rot_z(base_tilt_deg())
    r_ra = _rot_y(ra_e)
    r_dec = _rot_x(float(dec_deg))
    offset_in_ra = _DEC_ORIGIN_IN_RA + r_dec @ _OPTICAL_IN_DEC
    offset_world = r_base @ (r_ra @ offset_in_ra)
    return _BASE_TRANSLATION + offset_world


def dome_target_azimuth_deg(ra_deg: float, dec_deg: float) -> float | None:
    """
    Dome ``dome_az_deg`` that matches the horizontal (XZ) azimuth of the optical axis (Y up).

    Uses the **view direction**, not a sphere intersection: for an optical pivot offset from the
    dome centre, ``atan2(intersect_z, intersect_x)`` and ``atan2(dir_z, dir_x)`` differ — the slit
    should track boresight azimuth (same as **[I]** / ``align_dome_to_telescope``).

    Returns ``None`` if the axis is purely vertical in XZ (zenith / nadir); any dome azimuth is fine.
    """
    d = optical_axis_world_unit(ra_deg, dec_deg)
    h = float(np.hypot(float(d[0]), float(d[2])))
    if h < 1e-9:
        return None
    return float(-np.degrees(np.arctan2(float(d[2]), float(d[0]))))


def dome_ray_exit_distance_b1(
    ra_deg: float,
    dec_deg: float,
    dome_az_deg: float,
) -> float:
    """
    Step-3 hint: compare current dome azimuth to the target from ``dome_target_azimuth_deg``.
    Returns a distance above ``DOME_RAY_CLEAR_DISTANCE`` when aligned, else a small value.
    """
    target_az = dome_target_azimuth_deg(ra_deg, dec_deg)
    if target_az is None:
        return 200.0
    delta = (float(dome_az_deg) - target_az + 180.0) % 360.0 - 180.0
    # Slit alignment must be tight; 25° was far too loose (advance while the beam still missed the gap).
    if abs(delta) < 5.0:
        return 200.0
    return 5.0
