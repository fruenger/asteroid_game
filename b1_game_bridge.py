"""
Headless Asteroid session for the B1 native host: scenario, synthetic stack, game_state.tick / handle_discrete_input.

No Ursina — importable after sys.path includes ``asteroid_game``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

import b1_kinematics as b1k
from game_state import (
    AsteroidGameState,
    FrameHints,
    KeysHeld,
    KeysInput,
    align_dot_threshold,
    telescope_target_alignment_misalignment_deg,
    time_str,
    wrap_text,
)
from game_strings import TIME_WINDOW_REJECT_TOAST
from game_catalog_common import load_catalog_entries, orbital_elements_block
from game_strings import B1_STEUERUNG_TEXT
from game_synthetic_imaging import build_synthetic_perfect_stack
from scenario import generate_game_scenario

_gs: AsteroidGameState | None = None
_scenario = None


def _debug_align_overlay() -> bool:
    v = os.environ.get(
        "B1_DEBUG_ALIGN",
        os.environ.get("GAME_DEBUG_ALIGN", os.environ.get("ASTEROID_DEBUG_ALIGN", "")),
    ).strip().lower()
    return v in ("1", "true", "yes")


def _debug_pick() -> bool:
    return os.environ.get("B1_DEBUG_PICK", "").strip().lower() in ("1", "true", "yes")


def _catalog_lines() -> list[str]:
    if _gs is None or int(_gs.step) != 6:
        return []
    g = _gs
    foc = int(g.step6_focus_idx) % 2
    orb = (g.step6_orbital_text_block or "").strip()
    if not orb and _scenario is not None:
        orb = orbital_elements_block(_scenario.orbit).strip()
    # Intro text lives in HELP_TEXTS[6] / help_story_plain (left column in b1_host); here: form + Orbit + keys.
    lines = [
        "Objektname" + ("  ←" if foc == 0 else ""),
        g.step6_object_name + ("_" if foc == 0 else ""),
        "",
        "Entdeckerteam" + ("  ←" if foc == 1 else ""),
        g.step6_discoverer + ("_" if foc == 1 else ""),
        "",
        "Bahnelemente",
        orb,
        "",
        "[Tab] Feld wechseln   [Backspace] Zeichen löschen",
    ]
    return lines


def _catalog_lines_red() -> list[bool]:
    return [("←" in line) for line in _catalog_lines()]


def _summary_table_lines() -> list[str]:
    """Step 7: compact lines; wide first line uses horizontal space. (Rendering is top-anchored in b1_host.)"""
    if _gs is None or int(_gs.step) != 7:
        return []
    path = os.environ.get("B1_USER_SAVES_PATH", "user_saves.dat")
    rows = load_catalog_entries(path)

    def _trunc(s: str, n: int) -> str:
        t = s.replace("\n", " ").strip()
        if len(t) <= n:
            return t
        return t[: n - 1] + "\u2026"

    lines: list[str] = [
        "Katalog: Objekt | Entdeckerteam | Große Halbachse (a) | Umlaufszeit (P)",
        "",
    ]
    if not rows:
        lines.append("(noch keine Asteroiden entdeckt)")
        return lines
    wn = max(2, len(str(len(rows))))
    for i, (_ts, on, dn, a_au, p_yr) in enumerate(rows, start=1):
        ob = _trunc(on or "-", 46)
        tb = _trunc(dn or "-", 46)
        lines.append(f"{i:>{wn}}. {ob}  |  {tb}")
        if a_au is not None and p_yr is not None:
            lines.append(
                f"{' ' * wn}   a = {a_au:.3f} Abstand Erde-Sonne          P = {p_yr:.3f} Jahre"
            )
        else:
            lines.append(f"{' ' * wn}   a = -                 P = -")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def session_init(verbose_scenario: bool = False) -> None:
    """Build scenario, imaging stack, and ``AsteroidGameState``. Call once per process."""
    global _gs, _scenario

    # Match native b1_observatory: getenv("") leaves C++ at default 180, but b1k would read 0 — force a value.
    if not os.environ.get("B1_RA_OFFSET_DEG", "").strip():
        os.environ["B1_RA_OFFSET_DEG"] = "180"

    import astropy.units as u

    sc = generate_game_scenario(verbose=verbose_scenario)
    _scenario = sc
    imsize = 250
    perfect, locs = build_synthetic_perfect_stack(imsize=imsize, nstars=100)

    sun_dec = 0.5
    _gs = AsteroidGameState(
        min_time=sc.min_time,
        max_time=sc.max_time,
        object_dir_cartesian=np.asarray(sc.object_dir_cartesian, dtype=np.float64),
        sun_dec_deg=float(sun_dec),
        latitude_deg=52.0,
        imsize=imsize,
        perfect_stack=perfect,
        locations_stack=locs,
    )

    cheat = os.environ.get("B1_CHEAT_THROUGH", "").lower() in ("1", "true", "yes")
    if _gs is not None:
        _gs.cheat_through = cheat
        _gs.step6_orbital_text_block = orbital_elements_block(sc.orbit)
        _gs.step6_orbit_a_au = float(sc.orbit.a.to_value(u.AU))
        _gs.step6_orbit_p_yr = float(sc.orbit.period.to_value(u.yr))


def _panel_gray_u8(gs: AsteroidGameState) -> bytes | None:
    """Grayscale bytes row-major, top row first (matches C++ upload)."""
    if not gs.image_panel_enabled or gs.step not in (4, 5, 6):
        return None
    im = int(gs.imsize)
    if im <= 0:
        return None
    if gs.step == 6 and len(gs.all_images) == 3:
        coadd = np.sum(gs.all_images, axis=0)
        mx = float(np.max(coadd))
        if mx <= 0.0:
            mx = 1.0
        u8 = ((coadd / mx) * 255.0).astype(np.uint8)
        return u8.tobytes()
    if gs.step == 5 and len(gs.all_images) == 3:
        arr = np.asarray(gs.all_images[gs.image_shown], dtype=np.float64)
        mx = float(np.max(arr))
        if mx <= 0.0:
            mx = 1.0
        u8 = ((arr / mx) * 255.0).astype(np.uint8)
        return u8.tobytes()
    if gs.step == 4 and getattr(gs, "image_array", None) is not None and gs.image_array.size:
        arr = gs.image_array
        mx = float(np.max(arr))
        if mx <= 0.0:
            mx = 1.0
        u8 = ((arr / mx) * 255.0).astype(np.uint8)
        return u8.tobytes()
    return None


def _hints() -> FrameHints:
    assert _gs is not None
    g = _gs
    h = FrameHints()
    if g.step == 2:
        h.optical_axis_world_unit = b1k.optical_axis_world_unit(g.ra_deg, g.dec_deg)
    if g.step == 3 or g.cheat_through:
        if g.cheat_through:
            h.dome_ray_exit_distance = 99999.0
        else:
            h.dome_ray_exit_distance = b1k.dome_ray_exit_distance_b1(g.ra_deg, g.dec_deg, g.dome_az_deg)
    return h


def tick(
    dt: float,
    wall_t: float,
    k_space: int,
    k_left: int,
    k_right: int,
    k_up: int,
    k_down: int,
    k_r: int,
    k_dome_ccw: int = 0,
    k_dome_cw: int = 0,
) -> dict[str, Any]:
    """One frame; key flags are 0/1 (SDL-friendly). Native host maps keys by step (see b1_host): step 2
    telescope = arrows or WASD, dome = ``[``/``]``; step 3 dome = arrows or brackets (not A/D)."""
    if _gs is None:
        return {"ok": False, "error": "not_initialized"}

    keys = KeysInput(
        held=KeysHeld(
            space=bool(k_space),
            left=bool(k_left),
            right=bool(k_right),
            up=bool(k_up),
            down=bool(k_down),
            r=bool(k_r),
            dome_ccw=bool(k_dome_ccw),
            dome_cw=bool(k_dome_cw),
        )
    )
    out = _gs.tick(float(dt), float(wall_t), keys, _hints())
    sun = out.get("sun_direction", _gs.sun_unit_vector())
    infotext = str(out.get("infotext", _gs.infotext_message()))
    time_disp = str(out.get("time_display", time_str(_gs.time_now)))
    su = np.asarray(sun, dtype=np.float64).reshape(3)
    if _gs.paused_help:
        if _gs.help_overlay_kind == "controls":
            help_body = wrap_text(B1_STEUERUNG_TEXT, 42)
        else:
            help_body = wrap_text(_gs.help_body_text(), 42)
    else:
        help_body = ""
    help_story_plain = _gs.help_body_text()
    im = int(_gs.imsize)
    panel_bytes = _panel_gray_u8(_gs)
    toast_msg = str(out.get("toast", ""))
    if int(_gs.step) == 7:
        cl7 = _summary_table_lines()
        cl_red7 = [False] * len(cl7)
    else:
        cl7 = _catalog_lines()
        cl_red7 = _catalog_lines_red()
    ret: dict[str, Any] = {
        "ok": True,
        "step": int(_gs.step),
        "time_now": float(_gs.time_now),
        "time_stopped": bool(_gs.time_stopped),
        "paused_help": bool(_gs.paused_help),
        "help_overlay_kind": str(_gs.help_overlay_kind),
        "infotext": infotext,
        "help_body": help_body,
        "help_story_plain": help_story_plain,
        "catalog_lines": cl7,
        "catalog_lines_red": cl_red7,
        "toast": toast_msg,
        "toast_red": bool(toast_msg and toast_msg == TIME_WINDOW_REJECT_TOAST),
        "time_display": time_disp,
        "ra_deg": float(_gs.ra_deg),
        "dec_deg": float(_gs.dec_deg),
        "dome_az_deg": float(_gs.dome_az_deg),
        "sun_x": float(su[0]),
        "sun_y": float(su[1]),
        "sun_z": float(su[2]),
        "events": list(out.get("events", [])),
        "panel_enabled": bool(_gs.image_panel_enabled),
        "panel_w": im,
        "panel_h": im,
    }
    if panel_bytes is not None and len(panel_bytes) == im * im:
        ret["panel_gray"] = panel_bytes

    s4 = out.get("step4_primary_button_label")
    if isinstance(s4, str):
        ret["step4_primary_button_label"] = s4

    if _debug_align_overlay() and _gs is not None and int(_gs.step) == 2:
        u = b1k.optical_axis_world_unit(_gs.ra_deg, _gs.dec_deg)
        od = np.asarray(_gs.object_dir_cartesian, dtype=np.float64).reshape(3)
        ang, dot = telescope_target_alignment_misalignment_deg(u, od)
        dot_c = float(np.clip(dot, -1.0, 1.0))
        ang_acute = float(np.degrees(np.arccos(abs(dot_c))))
        th = float(np.clip(align_dot_threshold(), -1.0, 1.0))
        max_ang = float(np.degrees(np.arccos(th)))
        ra_raw = os.environ.get("B1_RA_OFFSET_DEG", "")
        lat_raw = os.environ.get("B1_RIG_LAT_TILT", "")
        lines = [
            f"[Align b1k] {ang:.2f} deg (acute {ang_acute:.2f}) max {max_ang:.2f} dot={dot:.4f} "
            f"paused_help={int(_gs.paused_help)} laser_ready={int(_gs.laser_and_marker_ready)}",
            f"[Align b1k env] B1_RA_OFFSET_DEG={ra_raw!r} → effective {b1k.ra_offset_deg():g} | "
            f"B1_RIG_LAT_TILT={lat_raw!r} → effective {b1k.base_tilt_deg():g}",
            f"[Align b1k vec] u=({u[0]:.5f},{u[1]:.5f},{u[2]:.5f}) tgt=({od[0]:.5f},{od[1]:.5f},{od[2]:.5f})",
        ]
        ret["align_debug_line"] = "\n".join(lines)

    tg: dict[str, float] | None = None
    if _gs.laser_and_marker_ready and 2 <= int(_gs.step) < 6:
        od = np.asarray(_gs.object_dir_cartesian, dtype=np.float64).reshape(3)
        on = float(np.linalg.norm(od))
        if on > 1e-12:
            od = od / on
        tg = {
            "ox": float(od[0]),
            "oy": float(od[1]),
            "oz": float(od[2]),
        }
    ret["target_guides"] = tg
    return ret


def input_key(key: str, wall_t: float) -> dict[str, Any]:
    if _gs is None:
        return {"ok": False, "error": "not_initialized"}
    r = _gs.handle_discrete_input(key, float(wall_t))
    return {
        "ok": True,
        "handled": bool(r.get("handled", False)),
        "events": list(r.get("events", [])),
    }


def close_help(wall_t: float | None = None) -> None:
    if _gs is not None and _gs.paused_help:
        _gs.close_help_overlay(wall_t)


def set_dome_az_deg(az_deg: float) -> dict[str, Any]:
    """Write ``AsteroidGameState.dome_az_deg`` (used by B1 host after computing azimuth from the rig laser)."""
    if _gs is None:
        return {"ok": False, "error": "not_initialized"}
    _gs.dome_az_deg = float(az_deg)
    return {"ok": True, "dome_az_deg": float(_gs.dome_az_deg)}


def align_dome_to_telescope() -> dict[str, Any]:
    """Set ``dome_az_deg`` from kinematics (headless / no observatory); B1 host prefers ``set_dome_az_deg`` from GLES laser."""
    if _gs is None:
        return {"ok": False, "error": "not_initialized"}
    target = b1k.dome_target_azimuth_deg(_gs.ra_deg, _gs.dec_deg)
    if target is None:
        return {"ok": True, "dome_az_deg": float(_gs.dome_az_deg), "note": "axis_vertical_xz"}
    return set_dome_az_deg(target)


def step6_key(kind: str, text: str = "") -> dict[str, Any]:
    """Schritt 6: Katalogeingabe; Schritt 7: [Enter] beendet (native host)."""
    if _gs is None:
        return {"ok": False, "error": "not_initialized"}
    if int(_gs.step) == 6:
        r = _gs.step6_key_action(str(kind), str(text))
    elif int(_gs.step) == 7:
        r = _gs.step7_key_action(str(kind))
    else:
        r = {"handled": False, "process_exit": False}
    return {
        "ok": True,
        "handled": bool(r.get("handled", False)),
        "process_exit": bool(r.get("process_exit", False)),
    }


def pick_at_normalized(nx: float, ny: float) -> dict[str, Any]:
    """Step 5: nx, ny in [0,1], origin top-left; nx→column, ny→row (same order as ``image_locations``)."""
    if _gs is None:
        if _debug_pick():
            print("[b1_pick] pick_at_normalized: _gs is None", file=sys.stderr, flush=True)
        return {"ok": False, "error": "not_initialized"}
    g = _gs
    im = float(g.imsize)
    col = float(nx) * im
    row = float(ny) * im
    before = int(g.step)
    nimg = len(g.all_images)
    panel_on = bool(g.image_panel_enabled)
    dist_norm = None
    if _debug_pick() and before == 5 and nimg == 3 and hasattr(g, "image_locations"):
        lp = np.array((row, col), dtype=float)
        dist = float(
            np.min(
                np.sqrt(
                    np.sum((np.expand_dims(lp, axis=1) - g.image_locations) ** 2, axis=0),
                )
            )
        )
        dist_norm = dist / im
    hit = g.try_step5_pick((row, col))
    if _debug_pick():
        extra = f" dist_norm={dist_norm:.4f} (hit_if_<0.05)" if dist_norm is not None else ""
        print(
            f"[b1_pick] pick_at_normalized nx={nx:.4f} ny={ny:.4f} -> row={row:.1f} col={col:.1f} "
            f"step_before={before} step_after={int(g.step)} n_images={nimg} panel_on={panel_on} "
            f"hit={bool(hit)}{extra}",
            file=sys.stderr,
            flush=True,
        )
    return {"ok": True, "picked": bool(hit), "step": int(g.step)}


def status() -> dict[str, Any]:
    if _gs is None:
        return {"ok": False}
    g = _gs
    return {
        "ok": True,
        "step": int(g.step),
        "cheat_through": bool(g.cheat_through),
        "ra_offset_deg": float(b1k.ra_offset_deg()),
    }
