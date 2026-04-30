"""On-screen typed message and small UI helpers."""

from __future__ import annotations

import time as time_stdlib

import numpy as np
from ursina import Text

from game_strings import infotext_char_delay_sec


def blink_opacity(duration: float, min_alpha: float = 0.25) -> float:
    return min_alpha + (np.sin(np.pi * time_stdlib.time() / duration) ** 2) * (1.0 - min_alpha)


class OnScreenMessage(Text):
    """
    Bottom infotext: revealed character-by-character. The full string is stored in ``full_text``;
    Ursina ``Text.text`` is only the visible prefix (typewriter). Use ``.message`` as an alias for
    ``full_text`` (setting it restarts the animation when the string changes).
    """

    def __init__(self, full_text: str = "", *, time_between_letters: float | None = None, **kwargs: object) -> None:
        kw = dict(kwargs)
        if "text" not in kw:
            kw["text"] = ""
        super().__init__(**kw)
        self._char_delay_override = time_between_letters
        self._trigger = time_stdlib.time()
        self.full_text = full_text

    def _effective_delay_sec(self) -> float:
        if self._char_delay_override is not None:
            return max(float(self._char_delay_override), 1e-9)
        return max(float(infotext_char_delay_sec()), 1e-9)

    def write(self) -> None:
        d = self._effective_delay_sec()
        n = int((time_stdlib.time() - self._trigger) / d)
        n = int(np.clip(n, 0, len(self.full_text)))
        self.text = self.full_text[:n]

    def reset_timer(self) -> None:
        self._trigger = time_stdlib.time()

    @property
    def message(self) -> str:
        return self.full_text

    @message.setter
    def message(self, value: str) -> None:
        v = "" if value is None else str(value)
        if v != self.full_text:
            self.full_text = v
            self._trigger = time_stdlib.time()
