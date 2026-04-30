"""
Telescope optical axis and pivot position in world space — matches the Ursina rig in game_scene.py
(base pivot Z = 52°−90°, RA about Y, DEC about X; optical marker at (2.2,0,0) in dec frame).

Used by the native GLES host to fill game_state.FrameHints without Panda3D.

``GAME_RA_OFFSET_DEG`` adds to game ``ra_deg`` before ``_rot_y`` (default **0** = Ursina ``game_scene``).
``touch_game_bridge.session_init`` sets **180** if unset so embedded Python matches GLES ``asteroid_game_touch``.
"""


from __future__ import annotations

import os

import numpy as np

# Default 0 matches Ursina ``game_scene`` (``ra_pivot.rotation_y = ra_deg`` with no extra twist).
# ``touch_game_bridge.session_init`` sets ``GAME_RA_OFFSET_DEG`` to 180 by default so embedded Python matches GLES.
_RA_OFFSET_DEFAULT_DEG = 0.0


def ra_offset_deg() -> float:
    """Degrees added to stored RA for rig math; reads ``GAME_RA_OFFSET_DEG`` (default 0; native host sets 180 via bridge)."""
    raw = os.environ.get("GAME_RA_OFFSET_DEG", "").strip()
    if not raw:
        return _RA_OFFSET_DEFAULT_DEG
    try:
        return float(raw)
    except ValueError:
        return _RA_OFFSET_DEFAULT_DEG


def effective_ra_deg(ra_deg: float) -> float:
    """RA used inside ``_rot_y`` — game state angle plus host mount offset."""
    return float(ra_deg) + ra_offset_deg()

# game_scene.py: base_pivot rotation_z = 52 - 90 (overridable via ``GAME_RIG_LAT_TILT``, same as native host)
_BASE_ROT_Z_DEG = 52.0 - 90.0


def base_tilt_deg() -> float:
    """Base mount tilt in degrees; reads ``GAME_RIG_LAT_TILT`` so Python matches the native observatory GLES rig."""
    raw = os.environ.get("GAME_RIG_LAT_TILT", "").strip()
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
    """Rotation about +Y for column vectors; matches ``mat4_rotate_y`` in the native observatory module (same column-vector convention)."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x_batch(deg: np.ndarray) -> np.ndarray:
    r = np.radians(np.asarray(deg, dtype=np.float64))
    c, s = np.cos(r), np.sin(r)
    out = np.zeros(r.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = c
    out[..., 1, 2] = -s
    out[..., 2, 1] = s
    out[..., 2, 2] = c
    return out


def _rot_y_batch(deg: np.ndarray) -> np.ndarray:
    r = np.radians(np.asarray(deg, dtype=np.float64))
    c, s = np.cos(r), np.sin(r)
    out = np.zeros(r.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = c
    out[..., 0, 2] = -s
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = s
    out[..., 2, 2] = c
    return out


def optical_axis_world_unit_batch(
    ra_deg: np.ndarray, dec_deg: np.ndarray
) -> np.ndarray:
    """Batched ``optical_axis_world_unit`` — same kinematics, one contiguous NumPy primitive."""
    ra_e = np.asarray(ra_deg, dtype=np.float64) + float(ra_offset_deg())
    dec_v = np.asarray(dec_deg, dtype=np.float64)
    if ra_e.shape != dec_v.shape:
        raise ValueError("ra_deg and dec_deg must match in shape.")
    rz = _rot_z(base_tilt_deg())
    ry = _rot_y_batch(ra_e)
    rx = _rot_x_batch(dec_v)
    rcombo = rz @ ry @ rx
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.einsum("...ij,j->...i", rcombo, ey)
    nn = np.linalg.norm(u, axis=-1, keepdims=True)
    nn = np.maximum(nn, 1e-12)
    return u / nn


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


def dome_ray_exit_distance_rig(
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
