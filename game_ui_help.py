"""Help / Steuerung overlay windows (TextField + fade)."""

from __future__ import annotations

import numpy as np
from ursina import Button, TextField, Vec3, camera, color, invoke

import game_globals as gg
from game_state import wrap_text
from game_strings import STEUERUNG_TEXT


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
    def update(self):
        if self.enabled:
            body = gg.game_gs.help_body_text() if gg.game_gs is not None else ""
            self.text = wrap_text(body, 80)
            self.render()

    def fade_in(self, duration=1.0, update_frq=20):
        fade_panel_bg_text(self, duration, update_frq, fade_in=True)

    def fade_out(self, duration=1.0, update_frq=20):
        fade_panel_bg_text(self, duration, update_frq, fade_in=False)


class SteuerungsWindow(TextField):
    """Nur Tasten- und Kameraübersicht; Spielablauf bleibt in HelpWindow."""

    def update(self):
        if self.enabled:
            self.text = wrap_text(STEUERUNG_TEXT, 80)
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
