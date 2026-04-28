"""Environment-driven graphics / window flags and debug logging."""

from __future__ import annotations

import os

from game_cam_env import GAME_CAM_DEFAULT_INIT_PITCH_DEG, initial_camera_yaw_pitch_deg
from ursina import Vec2, application, camera, window
from ursina.shaders import lit_with_shadows_shader
from ursina.shaders.unlit_with_fog_shader import unlit_with_fog_shader

# Shadow crash on some Mesa/Intel+Wayland HW paths: ASTEROID_GAME_SHADOWS=0 or MESA_LOADER_DRIVER_OVERRIDE=llvmpipe.
# Window: ASTEROID_GAME_BORDERLESS=1 removes title bar; default decorated. Fullscreen uses OS hint (see apply_os_fullscreen_hint).
USE_SHADOWS = os.environ.get("ASTEROID_GAME_SHADOWS", "1") != "0"
FULLSCREEN = os.environ.get("ASTEROID_GAME_FULLSCREEN", "1") != "0"
BORDERLESS = os.environ.get("ASTEROID_GAME_BORDERLESS", "0") == "1"
DEBUG = os.environ.get("ASTEROID_GAME_DEBUG", "").lower() in ("1", "true", "yes")

# lit_with_shadows_shader samples shadow maps; with DirectionalLight(shadows=False) those samplers can be invalid and crash Mesa in igLoop.
SCENE_SHADER = lit_with_shadows_shader if USE_SHADOWS else unlit_with_fog_shader


def shadow_map_resolution() -> Vec2:
    """Vec2 shadow map size; power-of-two recommended. Smaller maps often avoid Mesa/Intel crashes."""
    raw = os.environ.get("ASTEROID_GAME_SHADOW_MAP", "512").strip()
    if "," in raw:
        a, _, b = raw.partition(",")
        w, h = int(a.strip()), int(b.strip())
    else:
        w = h = int(raw)
    w = max(128, min(4096, w))
    h = max(128, min(4096, h))
    return Vec2(w, h)


def dlog(msg: str) -> None:
    if DEBUG:
        print(f"[asteroid_game] {msg}", flush=True)


def debug_print_versions() -> None:
    if not DEBUG:
        return
    import sys

    print(f"[asteroid_game] Python {sys.version.splitlines()[0]}", flush=True)
    print(
        f"[asteroid_game] ASTEROID_GAME_SHADOWS={int(USE_SHADOWS)} "
        f"ASTEROID_GAME_FULLSCREEN={int(FULLSCREEN)} "
        f"ASTEROID_GAME_BORDERLESS={int(BORDERLESS)} "
        f"scene_shader={'lit_with_shadows' if USE_SHADOWS else 'unlit_with_fog'}",
        flush=True,
    )
    try:
        import importlib.metadata as im

        for name in ("ursina", "numpy", "pillow", "scipy", "astropy", "poliastro"):
            try:
                print(f"[asteroid_game] {name} {im.version(name)}", flush=True)
            except im.PackageNotFoundError:
                pass
    except Exception as exc:
        print(f"[asteroid_game] importlib.metadata: {exc}", flush=True)
    try:
        from panda3d.core import getPackageVersionString

        print(f"[asteroid_game] Panda3D {getPackageVersionString()}", flush=True)
    except Exception as exc:
        print(f"[asteroid_game] Panda3D version: {exc}", flush=True)


def apply_os_fullscreen_hint() -> None:
    """Ursina resizes for 'fullscreen' but leaves WindowProperties.set_fullscreen commented out.
    Without the real fullscreen flag, borderless windows often stay windowed on Linux."""
    if application.window_type != "onscreen" or not FULLSCREEN:
        return
    try:
        from panda3d.core import WindowProperties

        wp = WindowProperties()
        wp.set_fullscreen(True)
        wp.set_undecorated(BORDERLESS)
        if window.main_monitor:
            m = window.main_monitor
            wp.set_origin(int(m.x), int(m.y))
            wp.set_size(int(m.width), int(m.height))
        application.base.win.request_properties(wp)
    except Exception:
        pass


def apply_initial_editor_camera_from_env() -> None:
    """Call right after ``EditorCamera()`` so the rig starts at the requested look (pitch always applied)."""
    yaw, pitch = initial_camera_yaw_pitch_deg()
    from ursina import scene

    ec = None
    for e in scene.entities:
        if getattr(e, "name", "") == "editor_camera":
            ec = e
            break
    if ec is None:
        return
    if yaw is not None:
        ec.rotation_y = yaw
    ec.rotation_x = pitch
