"""UI copy accessors — YAML under ``locales/`` (see ``game_i18n``).

Runtime language: ``ASTRO_LANG`` (`de`|`en`|`es`), user file ``~/.local/share/astro_mini_games/locale.yaml``.
"""

from __future__ import annotations

import os

from game_i18n import tr


def toast_time_window_reject_text() -> str:
    return tr("toast.time_window_reject")


def native_touch_steuerung_text() -> str:
    return tr("native_touch.steuerung")


def steuerung_text() -> str:
    return tr("controls.steuerung")


DEFAULT_INFOTEXT_CHAR_DELAY_SEC = 0.028


def infotext_char_delay_sec() -> float:
    """
    Delay between characters for bottom infotext typewriter (Ursina + ``asteroid_game_touch``).
    ``GAME_INFOTEXT_CHAR_DELAY_SEC`` overrides (seconds per character, must be > 0).
    """
    raw = os.environ.get("GAME_INFOTEXT_CHAR_DELAY_SEC", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0.0:
                return v
        except ValueError:
            pass
    return DEFAULT_INFOTEXT_CHAR_DELAY_SEC


DEFAULT_HELP_TEXT_CHAR_DELAY_SEC = 0.022


def help_text_char_delay_sec() -> float:
    """
    Typewriter pacing for HELP_TEXTS passed as ``help_story_plain`` / paused ``help_body`` to ``asteroid_game_touch``.
    Override with ``GAME_HELP_TEXT_CHAR_DELAY_SEC`` (> 0, seconds per character).
    """
    raw = os.environ.get("GAME_HELP_TEXT_CHAR_DELAY_SEC", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0.0:
                return v
        except ValueError:
            pass
    return DEFAULT_HELP_TEXT_CHAR_DELAY_SEC
