"""Headless Asteroid session for asteroid_game_touch: scenario, imaging stack, game_state.tick.

No Ursina — import when ``sys.path`` includes the package directory (``PYTHONPATH=.`` from ``asteroid_game``).
"""

from __future__ import annotations

import os
from typing import Any

import astropy.units as u
import numpy as np

import rig_kinematics as rk
from game_catalog_common import load_catalog_entries, orbital_elements_block
from game_state import (
    AsteroidGameState,
    FrameHints,
    KeysHeld,
    KeysInput,
    ephemeris_hints_verbose_enabled,
    telescope_target_alignment_misalignment_deg,
    time_str,
)
from game_i18n import ensure_locale_env, get_locale, set_locale, tr
from game_strings import (
    help_text_char_delay_sec,
    native_touch_steuerung_text,
    toast_time_window_reject_text,
)
from game_synthetic_imaging import build_synthetic_perfect_stack
from scenario import generate_game_scenario

_gs: AsteroidGameState | None = None


class _HelpTypewriterSession:
    """Per-session state for progressive help text — matches Ursina HelpWindow pacing."""

    def __init__(self) -> None:
        self._story_key: tuple[object, ...] | None = None
        self._story_t0 = 0.0
        self._paused_key: tuple[object, ...] | None = None
        self._paused_t0 = 0.0

    def reset(self) -> None:
        self._story_key = None
        self._story_t0 = 0.0
        self._paused_key = None
        self._paused_t0 = 0.0

    def story_plain(self, gs: AsteroidGameState, wall_t: float) -> str:
        full = gs.help_body_text()
        # Step 8 UI combines catalog with plain intro inside C++; keep full ``help.steps.8`` without TW.
        if not (1 <= int(gs.step) <= 7):
            self._story_key = None
            return full
        key = (get_locale(), int(gs.step), full)
        if key != self._story_key:
            self._story_key = key
            self._story_t0 = float(wall_t)
        d = max(float(help_text_char_delay_sec()), 1e-9)
        n = int((float(wall_t) - self._story_t0) / d)
        n = max(0, min(len(full), n))
        # No character wrap here: ``asteroid_game_touch`` wraps by panel pixel width (``wrap_w``).
        return full[:n]

    def paused_body(self, gs: AsteroidGameState, wall_t: float, full: str) -> str:
        if not gs.paused_help:
            return full
        key = (get_locale(), str(gs.help_overlay_kind), full)
        if key != self._paused_key:
            self._paused_key = key
            self._paused_t0 = float(wall_t)
        d = max(float(help_text_char_delay_sec()), 1e-9)
        n = int((float(wall_t) - self._paused_t0) / d)
        n = max(0, min(len(full), n))
        return full[:n]


_help_tw = _HelpTypewriterSession()


def _cheat_through_from_env() -> bool:
    return os.environ.get("GAME_CHEAT_THROUGH", "").strip().lower() in ("1", "true", "yes")


def _chrome_ui_dict() -> dict[str, str]:
    return {
        "quit": tr("native_chrome.quit_app"),
        "save": tr("native_chrome.save"),
        "observation_start": tr("native_chrome.observation_start"),
        "observatory_open": tr("native_chrome.observatory_open"),
        "restart_from_beginning": tr("native_chrome.restart_from_beginning"),
        "exposure_fallback": tr("native_chrome.exposure_fallback"),
        "osk_tab": tr("native_chrome.osk_tab"),
        "osk_shift": tr("native_chrome.osk_shift"),
        "osk_space": tr("native_chrome.osk_space"),
        "osk_backspace": tr("native_chrome.osk_backspace"),
        "overlay_step_prefix": tr("native_chrome.overlay_step_prefix"),
        "idle_warning": tr("native_chrome.idle_warning"),
    }


def set_locale_from_host(locale_code: str) -> dict[str, Any]:
    """Native ``asteroid_game_touch`` toolbar: persists locale via ``game_i18n.set_locale``."""
    lc = set_locale(locale_code)
    return {"ok": True, "locale": lc}


def session_init(verbose_scenario: bool = False, verbose_ephemeris_hints: bool = False) -> None:
    """Build scenario, synthetic stack, and ``AsteroidGameState``. Safe once per process."""
    global _gs

    _help_tw.reset()
    # GLES ``TouchObservatoryRig`` defaults RA offset +180 unless ``GAME_RA_OFFSET_DEG`` is set;
    # keep Python ``rig_kinematics`` aligned so step-3 telescope checks match the drawn laser.
    if not os.environ.get("GAME_RA_OFFSET_DEG", "").strip():
        os.environ["GAME_RA_OFFSET_DEG"] = "180"

    ensure_locale_env()
    sc = generate_game_scenario(verbose=verbose_scenario)
    imsize = 250
    perfect, locs = build_synthetic_perfect_stack(imsize=imsize, nstars=100)

    sun_dec = 0.5
    hint_explicit = bool(verbose_ephemeris_hints)

    _gs = AsteroidGameState(
        min_time=sc.min_time,
        max_time=sc.max_time,
        object_dir_cartesian=np.asarray(sc.object_dir_cartesian, dtype=np.float64),
        sun_dec_deg=float(sun_dec),
        latitude_deg=float(sc.latitude_deg),
        longitude_deg=float(sc.longitude_deg),
        reference_date=sc.reference_date,
        session_seed=int(sc.session_seed),
        visibility_t_open_mjd=float(sc.visibility_t_open_mjd),
        visibility_t_close_mjd=float(sc.visibility_t_close_mjd),
        catalog_display_date=sc.catalog_display_date,
        ephemeris_orbit=sc.orbit,
        verbose_ephemeris_hints=ephemeris_hints_verbose_enabled(hint_explicit),
        imsize=imsize,
        perfect_stack=perfect,
        locations_stack=locs,
    )
    assert _gs is not None
    _gs.step6_orbital_text_block = orbital_elements_block(sc.orbit)
    _gs.step6_orbit_a_au = float(sc.orbit.a.to_value(u.AU))
    _gs.step6_orbit_p_yr = float(sc.orbit.period.to_value(u.yr))
    _gs.cheat_through = _cheat_through_from_env()


def _hints() -> FrameHints:
    assert _gs is not None
    g = _gs
    h = FrameHints(defer_telescope_target_alignment=False)
    if g.step == 4 or g.cheat_through:
        if g.cheat_through:
            h.dome_ray_exit_distance = 99999.0
        else:
            h.dome_ray_exit_distance = rk.dome_ray_exit_distance_rig(g.ra_deg, g.dec_deg, g.dome_az_deg)
    if g.step == 6:
        h.step5_pick_pixel_xy = None
    return h


def _align_debug_line(gs: AsteroidGameState) -> str:
    raw = os.environ.get("GAME_DEBUG_ALIGN", os.environ.get("ASTEROID_DEBUG_ALIGN", "")).strip().lower()
    if raw not in ("1", "true", "yes") or gs.step != 3:
        return ""
    u = rk.optical_axis_world_unit(gs.ra_deg, gs.dec_deg)
    ang, dot = telescope_target_alignment_misalignment_deg(u, gs.object_dir_visual_cartesian)
    if np.isnan(dot):
        return ""
    return f"[Align] Winkel Laser–Ziel: {ang:.2f} deg  dot={dot:.4f}"


def _paused_help_body(gs: AsteroidGameState) -> str:
    if gs.help_overlay_kind == "controls":
        return native_touch_steuerung_text()
    return gs.help_body_text()


def _step7_tw_cursor(wall_t: float) -> str:
    """Blinking block cursor at end of the active catalog field (native overlay)."""
    return "▎" if int(float(wall_t) * 2.4) % 2 == 0 else "\u2008"


# Fixed character widths per column for native step 8 table (truncate + pad; monospace overlay).
_CATALOG_TS_W = 22
_CATALOG_OBJ_W = int(round(26 * 1.8))  # 47
_CATALOG_TEAM_W = int(round(24 * 1.8))  # 43
_CATALOG_ORB_W = 34


def _catalog_cell_fit(s: str, width: int) -> str:
    t = (s or "").replace("\n", " ").replace("\r", " ")
    if len(t) > width:
        return t[: max(0, width - 1)] + "…"
    return t.ljust(width)


def _catalog_row_fixed(
    ts: str, on: str, dn: str, a_au: float | None, p_yr: float | None
) -> str:
    if a_au is not None and p_yr is not None:
        orb = f"a={a_au:.4f} P={p_yr:.3f}"
    else:
        orb = ""
    return "  ".join(
        (
            _catalog_cell_fit(ts, _CATALOG_TS_W),
            _catalog_cell_fit(on, _CATALOG_OBJ_W),
            _catalog_cell_fit(dn, _CATALOG_TEAM_W),
            _catalog_cell_fit(orb, _CATALOG_ORB_W),
        )
    )


def _catalog_table_header_line() -> str:
    return "  ".join(
        (
            _catalog_cell_fit(tr("catalog_ui.catalog_col_time"), _CATALOG_TS_W),
            _catalog_cell_fit(tr("catalog_ui.catalog_col_object"), _CATALOG_OBJ_W),
            _catalog_cell_fit(tr("catalog_ui.catalog_col_team"), _CATALOG_TEAM_W),
            _catalog_cell_fit(tr("catalog_ui.catalog_col_orbit"), _CATALOG_ORB_W),
        )
    )


def _catalog_table_separator_line() -> str:
    n = _CATALOG_TS_W + _CATALOG_OBJ_W + _CATALOG_TEAM_W + _CATALOG_ORB_W + 6
    return "-" * n


def _step7_orbital_overlay_text(gs: AsteroidGameState) -> str:
    loe = tr("catalog_ui.orbital_elements")
    block = (gs.step6_orbital_text_block or "").strip()
    lines = "\n".join(ln.rstrip() for ln in block.split("\n") if ln.rstrip())
    return f"{loe}:\n\n{lines}" if lines else f"{loe}:\n\n"


def _catalog_lines_bundle(
    gs: AsteroidGameState, wall_t: float
) -> tuple[list[str], list[bool], list[str], list[bool], list[str]]:
    """
    Plain catalog lines (+ red flags), and for step 7 prefix/value rows for split-color native overlay.

    Prefix is red when that row is the active field (caret + translated label highlight).
    """
    if gs.step == 8:
        lines: list[str] = []
        red: list[bool] = []
        lines.append(_catalog_table_header_line())
        red.append(False)
        lines.append(_catalog_table_separator_line())
        red.append(False)
        for ts, on, dn, a_au, p_yr in load_catalog_entries():
            lines.append(_catalog_row_fixed(ts, on, dn, a_au, p_yr))
            red.append(False)
        return lines, red, [], [], []
    if gs.step == 7:
        lo = tr("catalog_ui.object_name")
        lt = tr("catalog_ui.discoverer_team")
        loe = tr("catalog_ui.orbital_elements")
        fi = int(gs.step6_focus_idx) % 2
        pref_o = "> " if fi == 0 else "  "
        pref_d = "> " if fi == 1 else "  "
        lines: list[str] = []
        reds: list[bool] = []
        c_o = _step7_tw_cursor(wall_t) if fi == 0 else ""
        c_d = _step7_tw_cursor(wall_t) if fi == 1 else ""
        lines.append(f"{pref_o}{lo}: {gs.step6_object_name}{c_o}")
        lines.append(f"{pref_d}{lt}: {gs.step6_discoverer}{c_d}")
        reds.extend((False, False))
        lines.append("")
        reds.append(False)
        lines.append(f"{loe}:")
        reds.append(False)
        block = (gs.step6_orbital_text_block or "").strip()

        ox = fi == 0
        od = fi == 1
        pv_p: list[str] = []
        pv_r: list[bool] = []
        pv_v: list[str] = []
        pv_p.append(("> " if ox else "  ") + f"{lo}: ")
        pv_r.append(ox)
        pv_v.append(gs.step6_object_name + (c_o if ox else ""))
        pv_p.append(("> " if od else "  ") + f"{lt}: ")
        pv_r.append(od)
        pv_v.append(gs.step6_discoverer + (c_d if od else ""))
        for raw in block.split("\n"):
            ln = raw.rstrip()
            if ln:
                lines.append(ln)
                reds.append(False)
        return lines, reds, pv_p, pv_r, pv_v
    return [], [], [], [], []


def _panel_gray_bytes(gs: AsteroidGameState) -> tuple[int, int, bytes]:
    if not gs.image_panel_enabled:
        return 0, 0, b""
    im = int(gs.imsize)
    if gs.step in (6, 7) and gs.all_images:
        arr = gs.all_images[int(gs.image_shown) % len(gs.all_images)]
    else:
        arr = gs.image_array
    if arr.size == 0 or arr.shape != (im, im):
        return im, im, bytes(im * im)
    mx = float(arr.max())
    if mx <= 0.0:
        mx = 1.0
    gray = ((arr / mx) * 255.0).astype(np.uint8).ravel().tobytes()
    return im, im, gray


def _target_guides(gs: AsteroidGameState) -> dict[str, Any]:
    v = np.asarray(gs.object_dir_visual_cartesian, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n > 1e-12:
        v = v / n
    else:
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    draw = gs.step >= 3 and gs.laser_and_marker_ready
    return {
        "ox": float(v[0]),
        "oy": float(v[1]),
        "oz": float(v[2]),
        "draw_laser": int(bool(draw)),
    }


def tick(
    dt: float,
    wall_t: float,
    k_space: int,
    k_left: int,
    k_right: int,
    k_up: int,
    k_down: int,
    k_r: int,
    k_lb: int,
    k_rb: int,
) -> dict[str, Any]:
    """One frame; trailing ints are 0/1 key flags (SDL-style), matching native ``(ddiiiiiiii)``."""
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
            dome_ccw=bool(k_lb),
            dome_cw=bool(k_rb),
        )
    )
    gs = _gs
    tick_out = gs.tick(float(dt), float(wall_t), keys, _hints())

    sun = np.asarray(tick_out.get("sun_direction", gs.sun_unit_vector()), dtype=np.float64).reshape(3)
    catalog_lines, catalog_lines_red, cat_pv_p, cat_pv_pr, cat_pv_v = _catalog_lines_bundle(gs, float(wall_t))
    pw, ph, pg = _panel_gray_bytes(gs)

    toast = str(tick_out.get("toast", "") or "")
    toast_red = toast.strip() == toast_time_window_reject_text().strip()

    story_plain = _help_tw.story_plain(gs, float(wall_t))
    paused_full = _paused_help_body(gs)
    help_body_out = _help_tw.paused_body(gs, float(wall_t), paused_full)

    out: dict[str, Any] = {
        "ok": True,
        "locale": str(get_locale()),
        "paused_help": bool(gs.paused_help),
        "help_overlay_kind": str(gs.help_overlay_kind),
        "sun_x": float(sun[0]),
        "sun_y": float(sun[1]),
        "sun_z": float(sun[2]),
        "time_now": float(gs.time_now),
        "ra_deg": float(gs.ra_deg),
        "dec_deg": float(gs.dec_deg),
        "dome_az_deg": float(gs.dome_az_deg),
        "panel_enabled": bool(gs.image_panel_enabled),
        "panel_w": int(pw),
        "panel_h": int(ph),
        "panel_gray": pg,
        "step": int(gs.step),
        "step4_primary_button_label": str(tick_out.get("step4_primary_button_label", gs.step4_primary_button_label())),
        "target_guides": _target_guides(gs),
        "time_display": str(tick_out.get("time_display") or time_str(gs.time_now)),
        "infotext": str(tick_out.get("infotext") or gs.infotext_message()),
        "help_body": help_body_out,
        "help_story_plain": story_plain,
        "catalog_lines": catalog_lines,
        "catalog_lines_red": catalog_lines_red,
        "catalog_pv_prefix": cat_pv_p,
        "catalog_pv_prefix_red": cat_pv_pr,
        "catalog_pv_value": cat_pv_v,
        "toast": toast,
        "toast_red": toast_red,
        "align_debug_line": _align_debug_line(gs),
        "chrome_ui": _chrome_ui_dict(),
    }

    if gs.step == 7:
        row, col, rad = gs.step7_panel_marker_pixels()
        out["panel_circle_row"] = float(row)
        out["panel_circle_col"] = float(col)
        out["panel_circle_radius_px"] = float(rad)
        out["step7_orbital_block"] = _step7_orbital_overlay_text(gs)
        out["step7_save_heading"] = tr("catalog_ui.save_object_heading")

    return out


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


def pick_at_normalized(nx: float, ny: float) -> dict[str, Any]:
    """Step 6: ``nx``, ``ny`` in ``[0, 1]``, origin top-left (column ↔ ``nx``, row ↔ ``ny``)."""
    if _gs is None:
        return {"ok": False, "error": "not_initialized", "picked": False, "step": -1}
    gs = _gs
    im = float(gs.imsize)
    row = float(ny) * im
    col = float(nx) * im
    try:
        hit = gs.try_step5_pick((row, col))
        return {"ok": True, "picked": bool(hit), "step": int(gs.step)}
    except Exception as e:
        return {"ok": False, "error": str(e), "picked": False, "step": int(gs.step)}


def align_dome_to_telescope() -> dict[str, Any]:
    if _gs is None:
        return {"ok": False}
    gs = _gs
    az = rk.dome_target_azimuth_deg(gs.ra_deg, gs.dec_deg)
    if az is None:
        return {"ok": True, "dome_az_deg": float(gs.dome_az_deg)}
    gs.dome_az_deg = float(az)
    return {"ok": True, "dome_az_deg": float(gs.dome_az_deg)}


def set_dome_az_deg(az_deg: float) -> dict[str, Any]:
    if _gs is None:
        return {"ok": False}
    _gs.dome_az_deg = float(az_deg)
    return {"ok": True, "dome_az_deg": float(_gs.dome_az_deg)}


def debug_jump_step8(wall_t: float) -> dict[str, Any]:
    """Native dev shortcut (5 × ] within ~1.25 s): show step 8 catalog overlay without playing through."""
    if _gs is None:
        return {"ok": False}
    _help_tw.reset()
    _gs.debug_jump_step8_for_native_ui_test(float(wall_t))
    return {"ok": True, "step": int(_gs.step)}


def step6_key(kind: str, text: str = "") -> dict[str, Any]:
    """Catalog typing (step 7) and finish (step 8); kinds match native SDL routing."""
    if _gs is None:
        return {"ok": False, "handled": False, "process_exit": False}
    gs = _gs
    if gs.step == 7:
        r = gs.step6_key_action(kind, text)
        return {
            "ok": True,
            "handled": bool(r.get("handled", False)),
            "process_exit": bool(r.get("process_exit", False)),
        }
    if gs.step == 8:
        r = gs.step7_key_action(kind)
        return {
            "ok": True,
            "handled": bool(r.get("handled", False)),
            "process_exit": bool(r.get("process_exit", False)),
        }
    return {"ok": True, "handled": False, "process_exit": False}


def status() -> dict[str, Any]:
    if _gs is None:
        return {"ok": False}
    g = _gs
    return {
        "ok": True,
        "step": int(g.step),
        "cheat_through": bool(g.cheat_through),
    }
