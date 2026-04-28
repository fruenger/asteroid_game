"""Orbit-camera init from environment (no Ursina; shared by ``game_settings`` and tests)."""

from __future__ import annotations

import os

# Default pitch (deg) when ``GAME_CAM_INIT_PITCH_DEG`` / second part of ``GAME_CAM_INIT_LOOK_DEG`` is omitted.
GAME_CAM_DEFAULT_INIT_PITCH_DEG = -30.0


def initial_camera_yaw_pitch_deg() -> tuple[float | None, float]:
    """
    Initial orbit-camera yaw / pitch in degrees.

    ``GAME_CAM_INIT_LOOK_DEG`` — ``yaw`` or ``yaw,pitch`` (comma or semicolon). One number = yaw only, pitch =
    ``GAME_CAM_DEFAULT_INIT_PITCH_DEG``.
    Otherwise ``GAME_CAM_INIT_YAW_DEG`` / ``GAME_CAM_INIT_PITCH_DEG``; omitted pitch uses the default.
    """
    default_p = GAME_CAM_DEFAULT_INIT_PITCH_DEG
    combined = os.environ.get("GAME_CAM_INIT_LOOK_DEG", "").strip()
    if combined:
        parts = [p.strip() for p in combined.replace(";", ",").split(",") if p.strip()]
        try:
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
            if len(parts) == 1:
                return float(parts[0]), default_p
        except ValueError:
            return None, default_p
    y_raw = os.environ.get("GAME_CAM_INIT_YAW_DEG", "").strip()
    p_raw = os.environ.get("GAME_CAM_INIT_PITCH_DEG", "").strip()
    try:
        yv = float(y_raw) if y_raw else None
    except ValueError:
        yv = None
    try:
        pv = float(p_raw) if p_raw else default_p
    except ValueError:
        pv = default_p
    return yv, pv
