"""Ursina app bootstrap, main loop hooks (frame_update / handle_input), and wiring."""

from __future__ import annotations

import math
import os
import time as wall_clock
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from PIL import Image
from ursina import (
    EditorCamera,
    Entity,
    Text,
    Texture,
    Ursina,
    application,
    camera,
    clamp,
    color,
    curve,
    held_keys,
    invoke,
    mouse,
    print_on_screen,
    raycast,
    scene,
    time as ursina_time,
    window,
)

import game_globals as gg
from celestial_settings import diurnal_sign, horizontal_celestial_offset_rad
from game_catalog_common import orbital_elements_block
from game_catalog_step6 import run_step6
from game_scene import SceneEntities, build_scene
from game_settings import (
    BORDERLESS,
    FULLSCREEN,
    apply_initial_editor_camera_from_env,
    apply_os_fullscreen_hint,
    debug_print_versions,
    dlog,
)
from game_state import (
    AsteroidGameState,
    FrameHints,
    KeysHeld,
    KeysInput,
    align_dot_threshold,
    telescope_target_alignment_misalignment_deg,
)
from game_stage_events import make_stageup_event
from game_synthetic_imaging import build_synthetic_perfect_stack
from game_strings import TIME_WINDOW_REJECT_TOAST
from game_ui_help import build_help_ui
from game_ui_onscreen import OnScreenMessage, blink_opacity
from orbit_api import time_str
from scenario import GameScenario, generate_game_scenario

_ursina_app = None

# Two-finger pinch: Panda3D reports each contact as a pointer device (see GraphicsWindow.has_pointer).
_PINCH_STATE: dict[str, float | None] = {"prev_dist": None}
_MAX_POINTER_DEVICES = 32


def _touch_pinch_zoom_step() -> None:
    """Apply pinch-to-zoom to the Ursina EditorCamera using multi-pointer distances."""
    base = getattr(application, "base", None)
    if base is None or base.win is None:
        return
    win = base.win

    pts: list[tuple[float, float]] = []
    for i in range(_MAX_POINTER_DEVICES):
        if not win.has_pointer(i):
            continue
        p = win.get_pointer(i)
        if not p.in_window:
            continue
        pts.append((float(p.get_x()), float(p.get_y())))

    if len(pts) < 2:
        _PINCH_STATE["prev_dist"] = None
        return

    dist = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    prev = _PINCH_STATE["prev_dist"]
    _PINCH_STATE["prev_dist"] = dist
    if prev is None:
        return

    dd = dist - prev
    if abs(dd) < 0.5:
        return

    ec = camera.parent
    if ec is None or not hasattr(ec, "target_z"):
        return

    if mouse.hovered_entity and getattr(ec, "ignore_scroll_on_ui", True):
        try:
            if mouse.hovered_entity.has_ancestor(camera.ui):
                return
        except Exception:
            pass

    zoom_speed = float(getattr(ec, "zoom_speed", 1.25))
    diag = math.hypot(float(window.size.x), float(window.size.y)) or 1.0
    scale = (dd / diag) * 18.0

    if not camera.orthographic:
        ec.target_z += zoom_speed * (abs(float(ec.target_z)) * 0.1) * scale
    else:
        tf = float(ec.target_fov)
        ec.target_fov = tf - zoom_speed * (abs(tf) * 0.1) * scale
        ec.target_fov = float(clamp(ec.target_fov, 1, 200))


def _debug_align_enabled() -> bool:
    v = os.environ.get("GAME_DEBUG_ALIGN", os.environ.get("ASTEROID_DEBUG_ALIGN", "")).strip().lower()
    return v in ("1", "true", "yes")


@dataclass
class GameRuntime:
    scenario: GameScenario
    scene: SceneEntities
    stageup_event: Any
    help_window: Any
    steuerung_window: Any
    infotext: OnScreenMessage
    time_display: Any
    align_debug_text: Text | None
    image_panel: Entity
    imsize: int
    sun_dec: float


def bootstrap() -> None:
    global _ursina_app

    scenario = generate_game_scenario()
    object_dir_cartesian = scenario.object_dir_cartesian
    min_time = scenario.min_time
    max_time = scenario.max_time

    debug_print_versions()
    dlog("before Ursina()")
    _ursina_app = Ursina()
    dlog("after Ursina()")
    EditorCamera()
    apply_initial_editor_camera_from_env()
    try:
        if application.base and application.base.win:
            application.base.win.enable_pointer_events()
    except Exception:
        pass
    window.fps_counter.enabled = False
    window.entity_counter.enabled = False
    window.collider_counter.enabled = False
    window.borderless = BORDERLESS
    window.fullscreen = FULLSCREEN
    apply_os_fullscreen_hint()
    camera.fov = 90.0
    dlog("window + camera configured")

    help_window, steuerung_window = build_help_ui()
    dlog("help_window UI ready")

    scene_entities = build_scene()

    sun_dec = 0.5

    imsize = 250
    all_images_perfect, image_locations = build_synthetic_perfect_stack(imsize=imsize, nstars=100)

    gg.game_gs = AsteroidGameState(
        min_time=min_time,
        max_time=max_time,
        object_dir_cartesian=np.asarray(object_dir_cartesian, dtype=np.float64),
        sun_dec_deg=float(sun_dec),
        latitude_deg=52.0,
        imsize=imsize,
        perfect_stack=all_images_perfect,
        locations_stack=image_locations,
    )
    gg.game_gs.step6_orbital_text_block = orbital_elements_block(scenario.orbit)
    gg.game_gs.step6_orbit_a_au = float(scenario.orbit.a.to_value(u.AU))
    gg.game_gs.step6_orbit_p_yr = float(scenario.orbit.period.to_value(u.yr))
    gg.laser_spawned = False

    sd = gg.game_gs.sun_unit_vector()
    scene_entities.sky.set_shader_input("u_sun_dir", (float(sd[0]), float(sd[1]), float(sd[2])))

    infotext = OnScreenMessage(
        message="[default text]",
        time_between_letters=0.01,
        origin=(-0.5, 0.5),
        parent=camera.ui,
        position=[-0.8, -0.42, -0.002],
    )
    gg.prev_message = infotext.message

    time_display = Text(text="hello world", origin=(-0.5, 0.5), parent=camera.ui, position=[-0.8, 0.45, 0.015], scale=2)

    align_debug_text: Text | None = None
    if _debug_align_enabled():
        align_debug_text = Text(
            text="",
            origin=(-0.5, 0.5),
            parent=camera.ui,
            position=(-0.8, 0.36, 0.016),
            scale=1.15,
            color=color.yellow,
            enabled=False,
        )

    image_panel = Entity(
        model="quad",
        parent=camera.ui,
        enabled=False,
        collider="box",
        position=(0, 0, 0.04),
        scale=0.8,
        origin=(0, 0),
    )
    dlog("image_panel ready")

    stageup_event = make_stageup_event(scene_entities, object_dir_cartesian)

    gg.runtime = GameRuntime(
        scenario=scenario,
        scene=scene_entities,
        stageup_event=stageup_event,
        help_window=help_window,
        steuerung_window=steuerung_window,
        infotext=infotext,
        time_display=time_display,
        align_debug_text=align_debug_text,
        image_panel=image_panel,
        imsize=imsize,
        sun_dec=sun_dec,
    )

    if os.environ.get("ASTEROID_GAME_STARTUP_HELP", "1") != "0":
        assert gg.game_gs is not None
        gg.game_gs.handle_discrete_input("h", wall_clock.time())
        help_window.enabled = True
        help_window.fade_in(0.5)
        gg.game_paused = True

    gg.cheat_through = False
    frame_update()
    dlog("initial update() returned")


def frame_update() -> None:
    rt = gg.runtime
    if rt is None or gg.game_gs is None:
        return

    game_gs = gg.game_gs
    game_gs.cheat_through = gg.cheat_through

    if gg.prev_message != rt.infotext.message:
        rt.infotext.reset_timer()

    _touch_pinch_zoom_step()

    rt.infotext.write()
    rt.infotext.wordwrap_setter(100)
    rt.infotext.alpha_setter(blink_opacity(3))

    if gg.game_paused:
        return

    wall_t = wall_clock.time()
    st = int(game_gs.step)
    if st == 2:
        left_k = bool(held_keys.get("left arrow", False) or held_keys.get("a", False))
        right_k = bool(held_keys.get("right arrow", False) or held_keys.get("d", False))
        up_k = bool(held_keys.get("up arrow", False) or held_keys.get("w", False))
        down_k = bool(held_keys.get("down arrow", False) or held_keys.get("s", False))
    else:
        left_k = bool(held_keys.get("left arrow", False))
        right_k = bool(held_keys.get("right arrow", False))
        up_k = bool(held_keys.get("up arrow", False))
        down_k = bool(held_keys.get("down arrow", False))
    keys = KeysInput(
        held=KeysHeld(
            space=bool(held_keys.get("space", False)),
            left=left_k,
            right=right_k,
            up=up_k,
            down=down_k,
            r=bool(held_keys.get("r", False)),
            dome_ccw=bool(st in (2, 3) and held_keys.get("[", False)),
            dome_cw=bool(st in (2, 3) and held_keys.get("]", False)),
        )
    )

    hints = FrameHints(defer_telescope_target_alignment=True)
    if game_gs.step == 3 or gg.cheat_through:
        if gg.cheat_through:
            hints.dome_ray_exit_distance = 99999.0
        else:
            # Dome cap alone is not enough: the static domewall had no collider before, so the beam could
            # appear to pass through concrete while the raycast still reported a “clear” slit exit.
            origin = rt.scene.telecope_optical_axis.world_position
            direction = rt.scene.telecope_optical_axis.up
            r_dome = raycast(origin, direction, traverse_target=rt.scene.dome_pivot)
            r_wall = raycast(origin, direction, traverse_target=rt.scene.domewall)
            ds = []
            if r_dome.hit:
                ds.append(float(r_dome.distance))
            if r_wall.hit:
                ds.append(float(r_wall.distance))
            if ds:
                hints.dome_ray_exit_distance = min(ds)
            else:
                hints.dome_ray_exit_distance = 200.0

    tick_out = game_gs.tick(ursina_time.dt, wall_t, keys, hints)

    rt.scene.ra_pivot.rotation_y = game_gs.ra_deg
    rt.scene.dec_pivot.rotation_x = game_gs.dec_deg
    rt.scene.dome_pivot.rotation_y = game_gs.dome_az_deg

    u_align: np.ndarray | None = None
    if game_gs.step == 2:
        u = np.asarray(rt.scene.telecope_optical_axis.up, dtype=np.float64).reshape(3)
        un = float(np.linalg.norm(u))
        if un > 1e-12:
            u_align = u / un
            for ev in game_gs.apply_telescope_target_alignment(u_align):
                tick_out["events"].append(ev)

    dbg = rt.align_debug_text
    if dbg is not None:
        if game_gs.step == 2 and u_align is not None:
            ang, dot = telescope_target_alignment_misalignment_deg(u_align, game_gs.object_dir_cartesian)
            th = float(np.clip(align_dot_threshold(), -1.0, 1.0))
            max_ang = float(np.degrees(np.arccos(th)))
            ph = "ja" if game_gs.paused_help else "nein"
            gp = "ja" if gg.game_paused else "nein"
            dbg.text = (
                f"[Align] Winkel Laser-Ziel: {ang:.2f} deg (max {max_ang:.2f} deg)  dot={dot:.4f}  "
                f"paused_help={ph} game_paused={gp}"
            )
            dbg.enabled = True
        else:
            dbg.text = ""
            dbg.enabled = False

    if not game_gs.time_stopped:
        rt.scene.sky.set_shader_input("u_time", game_gs.time_now)
        rt.scene.sky.set_shader_input("u_diurnal_sign", diurnal_sign())
        rt.scene.sky.set_shader_input("u_horiz_yaw_rad", horizontal_celestial_offset_rad())

    sun_dir = tick_out.get("sun_direction", game_gs.sun_unit_vector())
    rt.scene.light.look_at(-sun_dir)
    rt.scene.sky.set_shader_input(
        "u_sun_dir", (float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2]))
    )

    rt.time_display.text = tick_out.get("time_display", time_str(game_gs.time_now))

    gg.prev_message = rt.infotext.message
    rt.infotext.message = game_gs.infotext_message()
    if game_gs.step <= 7:
        rt.infotext.color = color.green
    rt.infotext.wordwrap_setter(100)

    rt.image_panel.enabled = game_gs.image_panel_enabled

    if game_gs.stage4_can_expose() and game_gs.image_array.size:
        arr = game_gs.image_array
        mx = float(arr.max())
        if mx > 0.0:
            rt.image_panel.texture = Texture(
                Image.fromarray(((arr / mx) * 255.0).astype(np.uint8), mode="L").convert("RGBA")
            )

    if game_gs.step == 5 and game_gs.all_images:
        img = game_gs.all_images[game_gs.image_shown]
        mx = float(img.max())
        if mx > 0.0:
            rt.image_panel.texture = Texture(
                Image.fromarray(((img / mx) * 255.0).astype(np.uint8), mode="L").convert("RGBA")
            )

    for ev in tick_out.get("events", []):
        if ev == "laser_and_marker_ready" and not gg.laser_spawned:
            rt.stageup_event(2)
            gg.laser_spawned = True
        if ev == "entered_stage4":
            rt.stageup_event(4)


def handle_input(key: str) -> None:
    rt = gg.runtime
    if rt is None or gg.game_gs is None:
        return

    game_gs = gg.game_gs
    wall_t = wall_clock.time()

    if key == "space" and game_gs.step == 0:
        r = game_gs.handle_discrete_input("space", wall_t)
        if "time_window_reject" in r["events"]:
            print_on_screen(
                TIME_WINDOW_REJECT_TOAST,
                origin=(0, 0),
                color=color.red,
                duration=2,
            )
        return

    if key == "z" and game_gs.step == 1:
        game_gs.handle_discrete_input("z", wall_t)
        rt.scene.shutter_pivot.animate_rotation([0.0, 0.0, -67.5], duration=10.0, curve=curve.linear)
        rt.scene.flap_pivot.animate_rotation([0.0, 0.0, 80.0], duration=10.0, curve=curve.linear)
        return

    if key == "x":
        rt.scene.shutter_pivot.animate_rotation([0.0, 0.0, 0.0], duration=10.0, curve=curve.linear)
        rt.scene.flap_pivot.animate_rotation([0.0, 0.0, 0.0], duration=10.0, curve=curve.linear)

    if key == "h":
        game_gs.handle_discrete_input("h", wall_t)
        rt.help_window.enabled = True
        rt.help_window.fade_in(0.5)
        gg.game_paused = True

    if key == "f1":
        game_gs.handle_discrete_input("f1", wall_t)
        rt.steuerung_window.enabled = True
        rt.steuerung_window.fade_in(0.5)
        gg.game_paused = True

    if key == "i":
        u = np.asarray(rt.scene.telecope_optical_axis.up, dtype=np.float64).reshape(3)
        un = float(np.linalg.norm(u))
        if un > 1e-12:
            u = u / un
        h = float(np.hypot(float(u[0]), float(u[2])))
        if h > 1e-9:
            target_azimuth = float(np.degrees(np.arctan2(float(u[2]), float(u[0]))))
            game_gs.dome_az_deg = -target_azimuth
            rt.scene.dome_pivot.rotation_y = game_gs.dome_az_deg

    if key == "left mouse down" and game_gs.step == 5:
        world_point = mouse.world_point
        try:
            local_point = rt.image_panel.get_relative_point(scene, world_point)
            local_point = (0.5 + np.array([-local_point[1], local_point[0]])) * rt.imsize
            distance = float(
                np.min(
                    np.sqrt(
                        np.sum(
                            (np.expand_dims(local_point, axis=1) - game_gs.image_locations) ** 2,
                            axis=0,
                        )
                    )
                )
            )
            print("[DEBUG] Distance to object: %i.1fpx" % distance)

            if game_gs.try_step5_pick((float(local_point[0]), float(local_point[1]))):
                rt.scene.tr_mask.fade_in(1.0)
                invoke(rt.scene.tr_mask.fade_out, 1.0, delay=1.0)
                invoke(rt.image_panel.enabled_setter, False, delay=1.0)
                run_step6(orbit=rt.scenario.orbit, game_gs=game_gs, infotext=rt.infotext)
        except TypeError as te:
            print("error", te)


def run_forever() -> None:
    dlog("entering app.run() — if segfault follows, crash is in Panda3D main loop / first frames")
    assert _ursina_app is not None
    _ursina_app.run()
