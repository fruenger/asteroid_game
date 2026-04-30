"""Help / Steuerung overlay windows (TextField + fade)."""

from __future__ import annotations

import time as time_stdlib

import numpy as np
from ursina import Button, TextField, Vec3, camera, color, invoke

import game_globals as gg
from game_state import wrap_text
from game_strings import help_text_char_delay_sec, steuerung_text


def _help_wrap_width_chars(steuerung: bool = False) -> int:
    """Rough fill width for Ursina panels (scaled ~0.8·aspect × 0.72·aspect respectively)."""
    try:
        ar = float(camera.aspect_ratio_getter())
    except Exception:
        ar = 16.0 / 9.0
    base = int(70 * ar + 24)
    if steuerung:
        base = int(base * 0.9)
    return max(88, min(260, base))


def fade_panel_bg_text(panel, duration=1.0, update_frq=20, fade_in=True):
    if fade_in:
        for alpha_value in np.linspace(0.0, 1.0, int(duration * update_frq)):
            invoke(panel.bg.alpha_setter, alpha_value, delay=alpha_value * duration)
            invoke(panel.text_entity.alpha_setter, alpha_value, delay=alpha_value * duration)
    else:
        for inverse_alpha_value in np.linspace(0.0, 1.0, int(duration * update_frq)):
            invoke(panel.bg.alpha_setter, 1.0 - inverse_alpha_value, delay=inverse_alpha_value * duration)
            invoke(panel.text_entity.alpha_setter, 1.0 - inverse_alpha_value, delay=inverse_alpha_value * duration)


class HelpWindow(TextField):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._tw_target = ""
        self._tw_t0 = 0.0

    def update(self):
        if self.enabled:
            epoch = getattr(gg, "i18n_epoch", 0)
            if epoch != getattr(self, "_i18n_last", -1):
                self._i18n_last = epoch
                self._tw_target = ""
            body = gg.game_gs.help_body_text() if gg.game_gs is not None else ""
            if body != self._tw_target:
                self._tw_target = body
                self._tw_t0 = time_stdlib.time()
            d = max(float(help_text_char_delay_sec()), 1e-9)
            n = int((time_stdlib.time() - self._tw_t0) / d)
            n = max(0, min(len(body), n))
            self.text = wrap_text(body[:n], _help_wrap_width_chars(steuerung=False))
            self.render()

    def fade_in(self, duration=1.0, update_frq=20):
        fade_panel_bg_text(self, duration, update_frq, fade_in=True)

    def fade_out(self, duration=1.0, update_frq=20):
        fade_panel_bg_text(self, duration, update_frq, fade_in=False)


class SteuerungsWindow(TextField):
    """Nur Tasten- und Kameraübersicht; Spielablauf bleibt in HelpWindow."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._tw_target = ""
        self._tw_t0 = 0.0

    def update(self):
        if self.enabled:
            epoch = getattr(gg, "i18n_epoch", 0)
            if epoch != getattr(self, "_i18n_last", -1):
                self._i18n_last = epoch
                self._tw_target = ""
            body = steuerung_text()
            if body != self._tw_target:
                self._tw_target = body
                self._tw_t0 = time_stdlib.time()
            d = max(float(help_text_char_delay_sec()), 1e-9)
            n = int((time_stdlib.time() - self._tw_t0) / d)
            n = max(0, min(len(body), n))
            self.text = wrap_text(body[:n], _help_wrap_width_chars(steuerung=True))
            self.render()

    def fade_in(self, duration=1.0, update_frq=20):
        fade_panel_bg_text(self, duration, update_frq, fade_in=True)

    def fade_out(self, duration=1.0, update_frq=20):
        fade_panel_bg_text(self, duration, update_frq, fade_in=False)


def sync_game_paused_after_help(help_window, steuerung_window) -> None:
    gg.game_paused = bool(help_window.enabled or steuerung_window.enabled)


def build_help_ui():
    help_window = HelpWindow(character_limit=99999, x=0, y=0, origin=(-0.5, -0.5), max_lines=64, line_height=2.0)
    help_window.position = (-0.4 * camera.aspect_ratio_getter(), 0.4, 0.03)
    help_window.text_entity.position -= Vec3(-0.05, 0.05, 0)
    help_window.bg.scale = (0.8 * camera.aspect_ratio_getter(), 0.8)
    help_window.active = False
    help_window.enabled = False
    help_window_close_button = Button("X", parent=help_window, color=color.red, text_color=color.white, scale=0.05)
    help_window_close_button.position = (0.8 * camera.aspect_ratio_getter(), 0)

    steuerung_window = SteuerungsWindow(
        character_limit=99999, x=0, y=0, origin=(-0.5, -0.5), max_lines=96, line_height=2.0
    )
    steuerung_window.position = (0.42 * camera.aspect_ratio_getter(), 0.4, 0.03)
    steuerung_window.text_entity.position -= Vec3(-0.05, 0.05, 0)
    steuerung_window.bg.scale = (0.72 * camera.aspect_ratio_getter(), 0.88)
    steuerung_window.active = False
    steuerung_window.enabled = False
    steuerung_window_close_button = Button("X", parent=steuerung_window, color=color.red, text_color=color.white, scale=0.05)
    steuerung_window_close_button.position = (0.66 * camera.aspect_ratio_getter(), 0)

    def close_help_window():
        help_window.fade_out(0.5)

        def _after():
            help_window.enabled = False
            if gg.game_gs is not None:
                gg.game_gs.close_help_overlay()
            sync_game_paused_after_help(help_window, steuerung_window)

        invoke(_after, delay=0.5)

    def close_steuerung_window():
        steuerung_window.fade_out(0.5)

        def _after():
            steuerung_window.enabled = False
            if gg.game_gs is not None:
                gg.game_gs.close_help_overlay()
            sync_game_paused_after_help(help_window, steuerung_window)

        invoke(_after, delay=0.5)

    help_window_close_button.on_click = close_help_window
    steuerung_window_close_button.on_click = close_steuerung_window

    return help_window, steuerung_window
