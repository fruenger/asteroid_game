"""Orbit-camera init from environment (no Ursina; shared by ``game_settings`` and tests)."""

from __future__ import annotations

import math
import os

# Defaults match native ``asteroid_game_touch`` (orbit around (0, pivot_y, 0)).
GAME_CAM_DEFAULT_PIVOT_Y = 1.4
GAME_CAM_DEFAULT_MAX_DIST = 14.0
GAME_CAM_DEFAULT_INIT_DIST = 7.0
GAME_CAM_DEFAULT_INIT_EYE_Y = 2.0

# Matches native ``asteroid_game_touch`` kCamPitchMin / kCamPitchMax (radians).
_GAME_CAM_PITCH_CLAMP_RAD = (-1.38, 1.38)


def _clamp_pitch_deg(pitch_deg: float) -> float:
    lo = math.degrees(_GAME_CAM_PITCH_CLAMP_RAD[0])
    hi = math.degrees(_GAME_CAM_PITCH_CLAMP_RAD[1])
    return max(lo, min(hi, pitch_deg))


def orbit_pitch_deg_for_eye(eye_y: float, pivot_y: float, dist: float, dist_max: float) -> float:
    """Pitch for ``eye_y = pivot_y + d*sin(pitch)`` with ``d`` clamped to ``[1, dist_max]`` (matches native host)."""
    d = max(1.0, min(dist, dist_max))
    if d <= 1e-9:
        return 0.0
    s = (eye_y - pivot_y) / d
    s = max(-1.0, min(1.0, s))
    pitch_rad = math.asin(s)
    lo, hi = _GAME_CAM_PITCH_CLAMP_RAD
    pitch_rad = max(lo, min(hi, pitch_rad))
    return math.degrees(pitch_rad)


# Fallback when parsing fails — same orbit as nominal defaults (native ``kCamDefInitEyeY``, default dist/pivot/max).
GAME_CAM_DEFAULT_INIT_PITCH_DEG = orbit_pitch_deg_for_eye(
    GAME_CAM_DEFAULT_INIT_EYE_Y,
    GAME_CAM_DEFAULT_PIVOT_Y,
    GAME_CAM_DEFAULT_INIT_DIST,
    GAME_CAM_DEFAULT_MAX_DIST,
)


def _float_env(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def initial_camera_yaw_pitch_deg() -> tuple[float | None, float]:
    """
    Initial orbit-camera yaw / pitch in degrees (native ``asteroid_game_touch`` parity).

    ``GAME_CAM_INIT_LOOK_DEG`` — ``yaw`` or ``yaw,pitch`` (comma or semicolon). One number sets yaw only; pitch is derived
    from ``GAME_CAM_INIT_HEIGHT`` (default ``2``) at ``GAME_CAM_PIVOT_Y`` + ``GAME_CAM_INIT_DIST``.
    Otherwise ``GAME_CAM_INIT_YAW_DEG`` / ``GAME_CAM_INIT_PITCH_DEG``; omitted pitch uses the same height-based orbit default.

    If ``GAME_CAM_INIT_HEIGHT`` or ``GAME_CAM_INIT_LOOK_DEG`` (single number) implies no explicit pitch — or HEIGHT is set
    in the environment — pitch is recomputed from eye Y, pivot, and distance.

    Explicit pitch from LOOK (two angles) / ``GAME_CAM_INIT_PITCH_DEG`` wins unless ``GAME_CAM_INIT_HEIGHT`` is set.

    Explicit pitch values are clamped to the native pitch limits.
    """
    default_p = GAME_CAM_DEFAULT_INIT_PITCH_DEG

    pitch_explicit = False
    yv: float | None = None
    pv = default_p

    combined = os.environ.get("GAME_CAM_INIT_LOOK_DEG", "").strip()
    if combined:
        parts = [p.strip() for p in combined.replace(";", ",").split(",") if p.strip()]
        try:
            if len(parts) >= 2:
                yv, pv = float(parts[0]), float(parts[1])
                pitch_explicit = True
            elif len(parts) == 1:
                yv = float(parts[0])
                pitch_explicit = False
            else:
                yv, pv = None, default_p
                pitch_explicit = False
        except ValueError:
            yv, pv = None, default_p
            pitch_explicit = False
    else:
        y_raw = os.environ.get("GAME_CAM_INIT_YAW_DEG", "").strip()
        p_raw = os.environ.get("GAME_CAM_INIT_PITCH_DEG", "").strip()
        try:
            yv = float(y_raw) if y_raw else None
        except ValueError:
            yv = None
        pitch_explicit = False
        try:
            if p_raw:
                pv = float(p_raw)
                pitch_explicit = True
            else:
                pv = default_p
        except ValueError:
            pv = default_p
            pitch_explicit = False

    pivot_y = GAME_CAM_DEFAULT_PIVOT_Y
    pvo = _float_env("GAME_CAM_PIVOT_Y")
    if pvo is not None and -5.0 < pvo < 40.0:
        pivot_y = pvo

    cam_dist_max = GAME_CAM_DEFAULT_MAX_DIST
    mdm = _float_env("GAME_CAM_MAX_DIST")
    if mdm is not None and 2.0 <= mdm <= 80.0:
        cam_dist_max = mdm

    cam_dist = GAME_CAM_DEFAULT_INIT_DIST
    cid = _float_env("GAME_CAM_INIT_DIST")
    if cid is not None and 1.0 <= cid <= 80.0:
        cam_dist = cid
    cam_dist = max(1.0, min(cam_dist, cam_dist_max))

    eye_tgt = GAME_CAM_DEFAULT_INIT_EYE_Y
    height_from_env = False
    h_raw = os.environ.get("GAME_CAM_INIT_HEIGHT", "").strip()
    if h_raw:
        try:
            v = float(h_raw)
            if -2.0 < v < 40.0:
                eye_tgt = v
                height_from_env = True
        except ValueError:
            pass

    if (not pitch_explicit) or height_from_env:
        pv = orbit_pitch_deg_for_eye(eye_tgt, pivot_y, cam_dist, cam_dist_max)
    else:
        pv = _clamp_pitch_deg(pv)

    return yv, pv
